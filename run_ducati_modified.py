import os
import dgl
import time
import torch
import random
import bifeat
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import distributed as dist
import tqdm

from mylog import get_logger
mlog = get_logger()

import DUCATI
from model import SAGE
from load_graph import load_dc_realtime_process, load_graph_all_data
from common import set_random_seeds, get_seeds_list

def entry(args, graph, all_data, seeds_list, counts):
    mlog(f"Start training")
    fanouts = [int(x) for x in args.fanouts.split(",")]
    cached_indptr, cached_indices = DUCATI.CacheConstructor.form_adj_cache(args, graph, counts)
    sampler = DUCATI.NeighborSampler(cached_indptr, cached_indices, fanouts)
    gpu_flag, gpu_map, all_cache, _ = DUCATI.CacheConstructor.form_nfeat_cache(args, all_data, counts)

    # prepare a buffer
    mlog(f"Prepare Buffer")
    input_nodes, _, _ = sampler.sample(graph, seeds_list[0])
    estimate_max_batch = int(1.2*input_nodes.shape[0])
    nfeat_buf = torch.zeros((estimate_max_batch, args.fake_dim), dtype=torch.float).cuda()
    label_buf = torch.zeros((args.bs, ), dtype=torch.long).cuda()
    mlog(f"buffer size: {(estimate_max_batch*args.fake_dim*4+args.bs*8)/(1024**3):.3f} GB")

    nfeat_loader = DUCATI.NfeatLoader(all_data[0], all_cache[0], gpu_map, gpu_flag)
    label_loader = DUCATI.NfeatLoader(all_data[1], all_cache[1], gpu_map, gpu_flag)
    mlog(f"finish prepare loader")
    # prepare model
    model = SAGE(args.fake_dim, args.num_hidden, args.n_classes, len(fanouts), F.relu, args.dropout)
    model = model.cuda()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mlog(f"finish prepare model")

    def run_one_list(target_list):
        nonlocal gpu_flag, gpu_map, all_cache, all_data, sampler
        for seeds in tqdm.tqdm(target_list):
            # Adj-Sampling
            input_nodes, output_nodes, blocks = sampler.sample(graph, seeds)
            # Nfeat-Selecting
            cur_nfeat = nfeat_loader.load(input_nodes, nfeat_buf) # fetch nfeat
            cur_label = label_loader.load(input_nodes[:args.bs], label_buf) # fetch label
            # train
            batch_pred = model(blocks, cur_nfeat)
            loss = loss_fcn(batch_pred, cur_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # add the first run as warmup
    avgs = []
    mlog(f"start running")
    for i in range(args.runs+1):
        mlog(f"running epoch: {i}")
        torch.cuda.synchronize()
        tic = time.time()
        run_one_list(seeds_list)
        torch.cuda.synchronize()
        avg_duration = 1000*(time.time() - tic)/len(seeds_list)
        avgs.append(avg_duration)
    avgs = avgs[1:]
    mlog(f"ducati: {args.adj_budget:.3f}GB adj cache & {args.nfeat_budget:.3f}GB nfeat cache time: {np.mean(avgs):.2f} ± {np.std(avgs):.2f}ms\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset params
    parser.add_argument("--dataset", type=str, choices=['ogbn-papers100M', "ogbn-products", 'uk', 'uk-union', 'twitter'],
                        default='ogbn-products')
    parser.add_argument("--root",
                           type=str,
                           default="./preprocess/ogbn_products/")
    parser.add_argument("--pre-epochs", type=int, default=2) # PreSC params

    # running params
    parser.add_argument("--nfeat-budget", type=float, default=0) # in GB
    parser.add_argument("--adj-budget", type=float, default=0) # in GB
    parser.add_argument("--bs", type=int, default=1000)
    parser.add_argument("--fanouts", type=str, default='12,12,12')
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--fake-dim", type=int, default=100)
    parser.add_argument("--pre-batches", type=int, default=100)

    # gnn model params
    parser.add_argument('--num-hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.003)

    args = parser.parse_args()
    mlog(args)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=1,
                            rank=0)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("{} of {}".format(rank, world_size))

    omp_thread_num = os.cpu_count() // world_size
    torch.cuda.set_device(rank)
    torch.set_num_threads(omp_thread_num)
    print("Set device to {} and cpu threads num {}".format(
        rank, omp_thread_num))
    
    set_random_seeds(0)

    # DUCATI
    ### LOAD
    shm_manager = bifeat.shm.ShmManager(0, 1, args.root, args.dataset, pin_memory=True)
    graph_tensors, meta_data = shm_manager.load_dataset(with_feature=True, with_valid=False, with_test=False)
    indptr, indices = graph_tensors['indptr'], graph_tensors['indices']
    n_classes = meta_data["num_classes"]
    num_nodes = meta_data["num_nodes"]
    graph = dgl.graph(('csc', (indptr, indices, torch.tensor([]))), num_nodes=num_nodes)
    graph = graph.formats(['csc'])
    train_idx = torch.nonzero(graph_tensors["train_idx"]).reshape(-1)
    adj_counts = graph_tensors["labels"]
    nfeat_counts = graph_tensors["features"]
    # cleanup # maybe can remove
    graph.ndata.clear()
    graph.edata.clear()
    separate_tic = time.time()
    # we prepare fake input for all datasets
    fake_nfeat = dgl.contrib.UnifiedTensor(torch.rand((graph.num_nodes(), args.fake_dim), dtype=torch.float), device='cuda')
    fake_label = dgl.contrib.UnifiedTensor(torch.randint(n_classes, (graph.num_nodes(), ), dtype=torch.long), device='cuda')
    mlog(f'finish generating random features with dim={args.fake_dim}, time elapsed: {time.time()-separate_tic:.2f}s')
    ### LOAD
    
    all_data = [fake_nfeat, fake_label]
    counts = [adj_counts, nfeat_counts]
    
    print(indptr)
    args.n_classes = n_classes
    train_idx = train_idx.cuda()
    # graph.pin_memory_()
    mlog(graph)

    # get seeds candidate
    seeds_list = get_seeds_list(args, train_idx)
    # del train_idx
    
    entry(args, graph, all_data, seeds_list, counts)
