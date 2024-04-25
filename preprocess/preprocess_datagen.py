import argparse
import numpy as np
import pandas as pd
import torch
import os
from scipy.sparse import coo_matrix
from ogb.lsc import MAG240MDataset
import dgl
import dgl.function as fn
import tqdm
import time


def process_products_with_reorder(args, dataset_path, save_path):
    print(f"Process {args.dataset}...")

    print("Read data...")
    meta_data = torch.load(os.path.join(dataset_path, "metadata.pt"))
    labels = torch.load(os.path.join(dataset_path, "labels.pt"))
    indptr = torch.load(os.path.join(dataset_path, "indptr.pt"))
    indices = torch.load(os.path.join(dataset_path, "indices.pt"))
    train_idx = torch.load(os.path.join(dataset_path, "train_idx.pt"))

    print("Covert data...")

    graph = dgl.graph(('csc', (indptr, indices, torch.tensor([]))),
                      num_nodes=meta_data["num_nodes"])
    src, dst = graph.adj_tensors(fmt='coo')
    graph = graph.formats(['csc'])
    num_nodes = graph.num_nodes()

    print("Perform sampling...")
    graph.ndata.clear()
    graph.edata.clear()
    graph.pin_memory_()
    train_idx = train_idx.cuda()
    adj_counts, nfeat_counts = generate_stats(graph, train_idx)
    graph.unpin_memory_()
    train_idx = train_idx.cpu()
    torch.cuda.empty_cache()

    print("Reorder graph...")
    degs = graph.in_degrees() + 1
    priority = adj_counts / degs
    adj_order = priority.argsort(descending=True)
    graph = fast_reorder((src, dst), adj_order)
    del src, dst
    indptr, indices, _ = graph.adj_tensors(fmt='csc')

    # prepare other ndata, reorder accordingly and save ndata
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    train_idx = train_mask[adj_order]
    adj_counts = adj_counts[adj_order]
    nfeat_counts = nfeat_counts[adj_order]
    labels = labels[adj_order]

    print("Save data...")
    torch.save(indptr.long(), os.path.join(save_path, "indptr.pt"))
    torch.save(indices.long(), os.path.join(save_path, "indices.pt"))
    torch.save(train_idx.long(), os.path.join(save_path, "train_idx.pt"))
    torch.save(labels, os.path.join(save_path, "labels.pt"))
    torch.save(adj_counts, os.path.join(save_path, "adj_hotness.pt"))
    torch.save(nfeat_counts, os.path.join(save_path, "feat_hotness.pt"))

    num_train_nodes = train_idx.shape[0]

    torch.save(meta_data, os.path.join(save_path, "metadata.pt"))


def fast_reorder(graph, nodes_perm):
    if isinstance(graph, tuple):
        src, dst = graph
    else:
        assert isinstance(graph, dgl.DGLHeteroGraph)
        src, dst = graph.adj_sparse(fmt='coo')
    mmap = torch.zeros(nodes_perm.shape[0], dtype=torch.int64)
    mmap[nodes_perm] = torch.arange(nodes_perm.shape[0])
    src = mmap[src]
    dst = mmap[dst]
    new_graph = dgl.graph((src, dst))
    del src, dst
    return new_graph


def my_iter(train_idx):
    pm = torch.randperm(train_idx.shape[0]).to(train_idx.device)
    local_train_idx = train_idx[pm]
    length = train_idx.shape[0] // 1000
    for i in range(length):
        st = i * 1000
        ed = (i + 1) * 1000
        yield local_train_idx[st:ed]


def generate_stats(graph, train_idx):
    #mlog("start calculate counts")
    fanouts = [12, 12, 12]
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    nfeat_counts = torch.zeros(graph.num_nodes()).cuda()
    adj_counts = torch.zeros(graph.num_nodes()).cuda()
    tic = time.time()
    for _ in range(2):
        it = my_iter(train_idx)
        for seeds in it:
            input_nodes, output_nodes, blocks = sampler.sample(graph, seeds)
            # for nfeat, each iteration we only need to prepare the input layer
            nfeat_counts[input_nodes] += 1
            # for adj, each iteration we need to access each block's dst nodes
            for block in blocks:
                dst_num = block.dstnodes().shape[0]
                cur_touched_adj = block.ndata[dgl.NID]['_N'][:dst_num]
                adj_counts[cur_touched_adj] += 1
    #mlog(f"pre-sampling {args.pre_epochs} epochs time: {time.time()-tic:.3f}s")
    #mlog(f"adj counts' min, max, mean, nnz ratio: {adj_counts.min()}, {adj_counts.max()}, {adj_counts.mean():.2f}, {(adj_counts>0).sum()/adj_counts.shape[0]:.2f}")
    #mlog(f"nfeat counts' min, max, mean, nnz ratio: {nfeat_counts.min()}, {nfeat_counts.max()}, {nfeat_counts.mean():.2f}, {(nfeat_counts>0).sum()/nfeat_counts.shape[0]:.2f}")
    adj_counts = adj_counts.cpu()
    nfeat_counts = nfeat_counts.cpu()
    return adj_counts, nfeat_counts


def process_papers100M(dataset_path, save_path):
    None


def process_papers_from_dgl_graph(dataset_path, save_path):
    None


def process_mag240M(dataset_path, save_path, gen_feat=False):
    None


def process_mag240M_from_dgl_graph(dataset_path, save_path):
    None


def process_friendster(dataset_path, save_path):
    None


def process_friendster_from_dgl_graph(dataset_path, save_path):
    None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="ogbn-papers100M",
        choices=["ogbn-products", "ogbn-papers100M", "mag240M", "friendster"])
    parser.add_argument("--root", help="Path of the dataset.")
    parser.add_argument("--save-path", help="Path to save the processed data.")
    parser.add_argument("--dgl-graph", action="store_true")
    args = parser.parse_args()
    print(args)

    process_products_with_reorder(args, args.root, args.save_path)
