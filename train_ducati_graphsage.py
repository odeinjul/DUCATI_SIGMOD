import dgl
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import os

import DUCATI
from models import SAGE, compute_acc
from common import set_random_seeds
from shm import ShmManager

torch.manual_seed(25)


class SeedGenerator(object):

    def __init__(self,
                 data: torch.Tensor,
                 batch_size: int,
                 shuffle: bool = False,
                 drop_last: bool = False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle:
            indexes = torch.randperm(self.data.shape[0],
                                     device=self.data.device)
            self.data = self.data[indexes]

        self.step = 0
        if self.drop_last:
            self.last_step = int((self.data.shape[0]) / self.batch_size)
        else:
            self.last_step = int(
                (self.data.shape[0] + self.batch_size - 1) / self.batch_size)

        return self

    def __next__(self):
        if self.step >= self.last_step:
            raise StopIteration

        ret = self.data[self.step * self.batch_size:(self.step + 1) *
                        self.batch_size]
        self.step += 1

        return ret

    def __len__(self):
        return self.last_step

    def is_finished(self):
        return self.step >= self.last_step


def run(rank, world_size, data, args):
    device = torch.cuda.current_device()
    # Unpack data
    train_nid, metadata, graph, dgl_g = data
    dgl_g.pin_memory_()
    num_train_per_gpu = (train_nid.numel() + args.num_trainers -
                         1) // args.num_trainers
    local_train_nid = train_nid[rank * num_train_per_gpu:(rank + 1) *
                                num_train_per_gpu]

    # sampler
    fan_out = [int(x) for x in args.fan_out.split(",")]
    cached_indptr, cached_indices, adj_cache_num = DUCATI.CacheConstructor.form_adj_cache(
        args, graph)
    sampler = DUCATI.NeighborSampler(cached_indptr, cached_indices, fan_out)

    # feature
    gpu_flag, gpu_map, all_cache = DUCATI.CacheConstructor.form_nfeat_cache(
        args, graph)
    # prepare a buffer for feature
    input_nodes, _, _ = sampler.sample(
        dgl_g, local_train_nid[torch.randperm(
            local_train_nid.shape[0])][:args.batch_size])
    estimate_max_batch = int(input_nodes.shape[0] * 1.5)
    nfeat_buf = torch.zeros((estimate_max_batch, metadata["feature_dim"]),
                            dtype=torch.float).cuda()
    label_buf = torch.zeros((args.batch_size, ), dtype=torch.long).cuda()
    nfeat_loader = DUCATI.NfeatLoader(graph["features"], all_cache[0], gpu_map,
                                      gpu_flag)
    label_loader = DUCATI.NfeatLoader(graph["labels"], all_cache[1], gpu_map,
                                      gpu_flag)

    # Define model and optimizer
    model = SAGE(metadata["feature_dim"], args.num_hidden,
                 metadata["num_classes"], len(fan_out), F.relu, args.dropout)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # prepare dataloader
    dataloader = SeedGenerator(local_train_nid.cuda(),
                               args.batch_size,
                               shuffle=True)

    epoch_time_log = []
    sample_time_log = []
    load_time_log = []
    forward_time_log = []
    backward_time_log = []
    update_time_log = []
    num_layer_seeds_log = []
    num_layer_neighbors_log = []
    num_inputs_log = []
    for epoch in range(args.num_epochs):

        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        num_iters = 0
        num_layer_seeds = 0
        num_layer_neighbors = 0

        if args.breakdown:
            dist.barrier()
            torch.cuda.synchronize()
        epoch_tic = time.time()
        model.train()

        for it, seed_nids in enumerate(dataloader):
            num_iters += 1
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            sample_begin = time.time()
            input_nodes, seeds, blocks = sampler.sample(dgl_g, seed_nids)
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            sample_time += time.time() - sample_begin

            load_begin = time.time()
            batch_inputs = nfeat_loader.load(input_nodes, nfeat_buf)
            batch_labels = label_loader.load(seeds, label_buf)
            batch_labels = batch_labels.long()
            num_seeds += len(blocks[-1].dstdata[dgl.NID])
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            load_time += time.time() - load_begin

            num_inputs += torch.sum(~gpu_flag[input_nodes]).item()
            for l, block in enumerate(blocks):
                layer_seeds = block.dstdata[dgl.NID]
                uncached_seeds_num = torch.sum(
                    layer_seeds >= adj_cache_num).item()
                num_layer_seeds += uncached_seeds_num
                num_layer_neighbors += uncached_seeds_num * fan_out[l]

            forward_start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            forward_time += time.time() - forward_start

            backward_begin = time.time()
            optimizer.zero_grad()
            loss.backward()
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            backward_time += time.time() - backward_begin

            update_start = time.time()
            optimizer.step()
            if args.breakdown:
                dist.barrier()
                torch.cuda.synchronize()
            update_time += time.time() - update_start

            if (it + 1) % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = (torch.cuda.max_memory_allocated() /
                                 1000000 if torch.cuda.is_available() else 0)
                print("Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                      "Train Acc {:.4f} | GPU {:.1f} MB".format(
                          rank, epoch, it + 1, loss.item(), acc.item(),
                          gpu_mem_alloc))
                train_acc_tensor = torch.tensor([acc.item()]).cuda()
                dist.all_reduce(train_acc_tensor, dist.ReduceOp.SUM)
                train_acc_tensor /= world_size
                if rank == 0:
                    print("Avg train acc {:.4f}".format(
                        train_acc_tensor[0].item()))

        epoch_toc = time.time()

        for i in range(args.num_trainers):
            dist.barrier()
            if i == rank % args.num_trainers:
                timetable = ("=====================\n"
                             "Part {}, Epoch Time(s): {:.4f}\n"
                             "Sampling Time(s): {:.4f}\n"
                             "Loading Time(s): {:.4f}\n"
                             "Forward Time(s): {:.4f}\n"
                             "Backward Time(s): {:.4f}\n"
                             "Update Time(s): {:.4f}\n"
                             "#seeds: {}\n"
                             "#inputs: {}\n"
                             "#iterations: {}\n"
                             "#sampling_seeds: {}\n"
                             "#sampled_neighbors: {}\n"
                             "=====================".format(
                                 rank,
                                 epoch_toc - epoch_tic,
                                 sample_time,
                                 load_time,
                                 forward_time,
                                 backward_time,
                                 update_time,
                                 num_seeds,
                                 num_inputs,
                                 num_iters,
                                 num_layer_seeds,
                                 num_layer_neighbors,
                             ))
                print(timetable)

        sample_time_log.append(sample_time)
        load_time_log.append(load_time)
        forward_time_log.append(forward_time)
        backward_time_log.append(backward_time)
        update_time_log.append(update_time)
        epoch_time_log.append(epoch_toc - epoch_tic)
        num_layer_seeds_log.append(num_layer_seeds)
        num_layer_neighbors_log.append(num_layer_neighbors)
        num_inputs_log.append(num_inputs)

    avg_epoch_time = np.mean(epoch_time_log[2:])
    avg_sample_time = np.mean(sample_time_log[2:])
    avg_load_time = np.mean(load_time_log[2:])
    avg_forward_time = np.mean(forward_time_log[2:])
    avg_backward_time = np.mean(backward_time_log[2:])
    avg_update_time = np.mean(update_time_log[2:])

    for i in range(args.num_trainers):
        dist.barrier()
        if i == rank % args.num_trainers:
            timetable = ("=====================\n"
                         "Part {}, Avg Time:\n"
                         "Epoch Time(s): {:.4f}\n"
                         "Sampling Time(s): {:.4f}\n"
                         "Loading Time(s): {:.4f}\n"
                         "Forward Time(s): {:.4f}\n"
                         "Backward Time(s): {:.4f}\n"
                         "Update Time(s): {:.4f}\n"
                         "#inputs: {}\n"
                         "#sampling_seeds: {}\n"
                         "#sampled_neighbors: {}\n"
                         "=====================".format(
                             rank, avg_epoch_time, avg_sample_time,
                             avg_load_time, avg_forward_time,
                             avg_backward_time, avg_update_time,
                             np.mean(num_inputs_log[2:]),
                             np.mean(num_layer_seeds_log[2:]),
                             np.mean(num_layer_neighbors_log[2:])))
            print(timetable)
    all_reduce_tensor = torch.tensor([0], device="cuda", dtype=torch.float32)

    all_reduce_tensor[0] = avg_epoch_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_epoch_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_sample_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_sample_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_load_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_load_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_forward_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_forward_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_backward_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_backward_time = all_reduce_tensor[0].item() / world_size

    all_reduce_tensor[0] = avg_update_time
    dist.all_reduce(all_reduce_tensor, dist.ReduceOp.SUM)
    all_reduce_update_time = all_reduce_tensor[0].item() / world_size

    if rank == 0:
        timetable = ("=====================\n"
                     "All reduce time:\n"
                     "Throughput(seeds/sec): {:.4f}\n"
                     "Epoch Time(s): {:.4f}\n"
                     "Sampling Time(s): {:.4f}\n"
                     "Loading Time(s): {:.4f}\n"
                     "Forward Time(s): {:.4f}\n"
                     "Backward Time(s): {:.4f}\n"
                     "Update Time(s): {:.4f}\n"
                     "=====================".format(
                         train_nid.shape[0] / all_reduce_epoch_time,
                         all_reduce_epoch_time,
                         all_reduce_sample_time,
                         all_reduce_load_time,
                         all_reduce_forward_time,
                         all_reduce_backward_time,
                         all_reduce_update_time,
                     ))
        print(timetable)


def main(args):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == args.num_trainers
    omp_thread_num = os.cpu_count() // args.num_trainers
    torch.cuda.set_device(rank)
    torch.set_num_threads(omp_thread_num)
    print("Set device to {} and cpu threads num {}".format(
        rank, omp_thread_num))
    set_random_seeds(0)

    shm_manager = ShmManager(rank,
                             args.num_trainers,
                             args.root,
                             args.dataset,
                             pin_memory=False)
    if args.dataset == "friendster":
        with_feature = False
        feat_dtype = torch.float32
        feat_dim = 256
    elif args.dataset == "mag240M":
        with_feature = False
        feat_dtype = torch.float16
        feat_dim = 768
    elif args.dataset == "ogbn-papers100M":
        with_feature = False
        feat_dtype = torch.float32
        feat_dim = 128
    else:
        with_feature = False
        feat_dtype = torch.float32
        feat_dim = 100
    g, metadata = shm_manager.load_dataset(with_feature=with_feature,
                                           with_test=False,
                                           with_valid=False)
    metadata["feature_dim"] = feat_dim
    dgl_g = dgl.graph(('csc', (g["indptr"], g["indices"], torch.tensor([]))))
    if not with_feature:
        if shm_manager._is_chief:
            fake_feat = torch.randn(
                (metadata["num_nodes"], ),
                dtype=feat_dtype).reshape(-1,
                                          1).repeat(1, metadata["feature_dim"])
            g["features"] = shm_manager.create_shm_tensor(
                args.dataset + "_shm_features", feat_dtype, fake_feat.shape)
            g["features"].copy_(fake_feat)
            del fake_feat
        else:
            g["features"] = shm_manager.create_shm_tensor(
                args.dataset + "_shm_features", None, None)
    dist.barrier()
    if shm_manager._is_chief:
        train_nid = torch.nonzero(g.pop("train_idx")).flatten()
        train_nid = train_nid[torch.randperm(train_nid.shape[0])]
        shm_train_nid = shm_manager.create_shm_tensor(
            args.dataset + "_shm_shuffled_train_idx", train_nid.dtype,
            train_nid.shape)
        shm_train_nid.copy_(train_nid)
        del train_nid
    else:
        shm_train_nid = shm_manager.create_shm_tensor(
            args.dataset + "_shm_shuffled_train_idx", None, None)
    dist.barrier()
    print(shm_train_nid)

    g["labels"][torch.isnan(g["labels"])] = 0
    g["labels"] = g["labels"].long()
    print("start")
    data = shm_train_nid, metadata, g, dgl_g

    run(rank, world_size, data, args)


# usage: torchrun --nproc_per_node ${num_trainers} train_ducati_graphsage.py [args]
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        "Train nodeclassification GraphSAGE model")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="datasets: ogbn-products, ogbn-papers100M, friendster, mag240M",
    )
    argparser.add_argument("--root", type=str, default="/data")
    argparser.add_argument(
        "--num-trainers",
        type=int,
        default="8",
        help=
        "number of trainers participated in the compress, no greater than available GPUs num"
    )
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--dropout", type=float, default=0.2)
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--batch-size-eval", type=int, default=100000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=21)
    argparser.add_argument("--fan-out", type=str, default="12,12,12")
    argparser.add_argument("--num-hidden", type=int, default=32)
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--breakdown", action="store_true", default=False)
    argparser.add_argument("--nfeat-budget", type=float, default=0)
    argparser.add_argument("--adj-budget", type=float, default=0)
    args = argparser.parse_args()
    print(args)
    main(args)
