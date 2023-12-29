import torch
import torch.distributed as dist
import argparse
from shm import ShmManager

torch.manual_seed(25)


def cache_idx_select(
    hotness_feat_list,
    hotness_adj,
    feat_slope_list,
    adj_slope,
    feat_space_list,
    adj_space_tensor,
    gpu_capacity,
):
    num_feat_type = len(hotness_feat_list)
    unified_hotness_list = []
    unified_space_list = []
    type_range = [0]
    for i in range(num_feat_type):
        num_idx = hotness_feat_list[i].shape[0]
        unified_hotness_list.append(hotness_feat_list[i] * feat_slope_list[i] /
                                    feat_space_list[i])
        unified_space_list.append(torch.full((num_idx, ), feat_space_list[i]))
        range_max = num_idx + type_range[i]
        type_range.append(range_max)
    unified_hotness_list.append(hotness_adj * adj_slope / adj_space_tensor)
    unified_space_list.append(adj_space_tensor)

    unified_hotness = torch.cat(unified_hotness_list)
    unified_space = torch.cat(unified_space_list)
    # valid_mask = unified_hotness > 0
    sorted_index = torch.argsort(unified_hotness, descending=True)
    del unified_hotness
    # sorted_index = sorted_index[valid_mask[sorted_index]]
    # del valid_mask
    sorted_space = unified_space[sorted_index]
    del unified_space
    space_prefix_sum = torch.cumsum(sorted_space, 0)
    del sorted_space
    cached_index = sorted_index[space_prefix_sum <= gpu_capacity]
    del space_prefix_sum

    cached_index_list = []
    for i in range(num_feat_type):
        this_type_cached_index = cached_index[
            (cached_index >= type_range[i])
            & (cached_index < type_range[i + 1])] - type_range[i]
        cached_index_list.append(this_type_cached_index)
    adj_cached_index = cached_index[cached_index >=
                                    type_range[-1]] - type_range[-1]

    return cached_index_list, adj_cached_index


def dtype_sizeof(input):
    if isinstance(input, str):
        if input == "int32":
            return 4
        elif input == "int64":
            return 8
        elif input == "float32":
            return 4
        elif input == "float64":
            return 8
        elif input == "int16" or input == "float16":
            return 2
        elif input == "bool" or input == "uint8" or input == "int8":
            return 1
    else:
        if input == torch.int32:
            return 4
        elif input == torch.int64:
            return 8
        elif input == torch.float32:
            return 4
        elif input == torch.float64:
            return 8
        elif input == torch.int16 or input == torch.float16:
            return 2
        elif input == torch.bool or input == torch.uint8 or input == torch.int8:
            return 1


def compute_feat_sapce(feat_dim, feat_dtype):
    space = feat_dim * dtype_sizeof(feat_dtype)
    return space


def compute_adj_space_tensor(indptr, indptr_dtype, indices_dtype):
    degree = indptr[1:] - indptr[:-1]
    space = degree * dtype_sizeof(indices_dtype) + dtype_sizeof(indptr_dtype)
    return space


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
    argparser.add_argument("--feat-slope", type=float, default=None)
    argparser.add_argument("--adj-slope", type=float, default=None)
    argparser.add_argument("--gpu-budget", type=float, default=10.0)
    args = argparser.parse_args()
    print(args)

    dist.init_process_group(backend="nccl")

    shm_manager = ShmManager(0, 1, args.root, args.dataset, pin_memory=False)
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
    feat_hotness = [g["feat_hotness"]]
    feat_space = [compute_feat_sapce(feat_dim, feat_dtype)]
    feat_slope = [args.feat_slope / 4 * dtype_sizeof(feat_dtype) * feat_dim]
    adj_hotness = g["adj_hotness"]

    adj_space = compute_adj_space_tensor(g["indptr"], torch.int64, torch.int64)
    adj_slope = args.adj_slope
    mem_capacity = int(args.gpu_budget * 1024 * 1024 * 1024)
    feat_cache_nids, adj_cache_nids = cache_idx_select(feat_hotness,
                                                       adj_hotness, feat_slope,
                                                       adj_slope, feat_space,
                                                       adj_space, mem_capacity)

    feat_cache_size = feat_cache_nids[0].shape[0] * feat_space[0]
    feat_total_size = metadata["num_nodes"] * feat_dim * dtype_sizeof(
        feat_dtype)
    adj_cache_size = torch.sum(adj_space[adj_cache_nids.cpu()]).item()
    adj_total_size = g["indptr"].numel() * g["indices"].element_size(
    ) + g["indices"].numel() * g["indices"].element_size()

    print(
        "GPU capacity {:.3f} GB, adj cache size {:.3f} GB ratio {:.3f}, feat cache size {:.3f} ratio {:.3f}"
        .format(mem_capacity / (1024 * 1024 * 1024),
                adj_cache_size / (1024 * 1024 * 1024),
                adj_cache_size / adj_total_size,
                feat_cache_size / (1024 * 1024 * 1024),
                feat_cache_size / feat_total_size))
