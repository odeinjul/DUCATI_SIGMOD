import dgl
import time
import torch

from common import set_random_seeds, get_seeds_list
from mylog import get_logger

mlog = get_logger()


def form_nfeat_cache(args, graph):
    if args.nfeat_budget == 0:
        return None, None, [None, None]

    # get probs and order
    nfeat_counts = graph["feat_hotness"]
    nfeat_probs = nfeat_counts / nfeat_counts.sum()
    nfeat_probs, nfeat_order = nfeat_probs.sort(descending=True)

    # calculate current cache
    features = graph["features"]
    labels = graph["labels"]
    single_line_size = features.shape[1] * features.element_size(
    ) + labels.element_size()

    cache_nums = int(args.nfeat_budget * (1024**3) / single_line_size)
    cache_nids = nfeat_order[:cache_nums]

    # prepare flag
    gpu_flag = torch.zeros(labels.shape[0], dtype=torch.bool)
    gpu_flag[cache_nids] = True
    gpu_flag = gpu_flag.cuda()

    # prepare cache
    cache = [features[cache_nids].to("cuda"), labels[cache_nids].to("cuda")]

    # prepare map in GPU
    # for gpu feature retrieve, input -(gpu_flag)-> gpu_mask --> gpu_nids -(gpu_map)-> gpu_local_id -> features
    gpu_map = torch.zeros(nfeat_probs.shape[0], dtype=torch.int32).fill_(-1)
    gpu_map[cache_nids] = torch.arange(cache_nids.shape[0]).int()
    gpu_map = gpu_map.cuda()

    return gpu_flag, gpu_map, cache


def form_adj_cache(args, graph):
    # given cache budget (in GB), derive the number of adj lists to be saved
    cache_bytes = args.adj_budget * (1024**3)
    cache_elements = cache_bytes // 8

    # search break point
    indptr = graph["indptr"]
    indices = graph["indices"]
    num_nodes = indptr.shape[0] - 1
    acc_size = indptr[1:] + torch.arange(
        1, num_nodes + 1) + 1  # accumulated cache size in theory
    cache_size = torch.searchsorted(acc_size, cache_elements).item()

    # prepare cache tensor
    cached_indptr = indptr[:cache_size + 1].cuda()
    cached_indices = indices[:indptr[cache_size]].cuda()

    return cached_indptr, cached_indices, cache_size


def separate_features_idx(args, graph):
    separate_tic = time.time()
    train_idx = torch.nonzero(graph.ndata.pop("train_mask")).reshape(-1)
    adj_counts = graph.ndata.pop('adj_counts')
    nfeat_counts = graph.ndata.pop('nfeat_counts')

    # cleanup
    graph.ndata.clear()
    graph.edata.clear()

    # we prepare fake input for all datasets
    fake_nfeat = dgl.contrib.UnifiedTensor(torch.rand(
        (graph.num_nodes(), args.fake_dim), dtype=torch.float),
                                           device='cuda')
    fake_label = dgl.contrib.UnifiedTensor(torch.randint(args.n_classes,
                                                         (graph.num_nodes(), ),
                                                         dtype=torch.long),
                                           device='cuda')

    mlog(
        f'finish generating random features with dim={args.fake_dim}, time elapsed: {time.time()-separate_tic:.2f}s'
    )
    return graph, [fake_nfeat,
                   fake_label], train_idx, [adj_counts, nfeat_counts]
