CUDA_VISIBLE_DEVICES=0 python ../run_ducati_modified.py --dataset "ogbn-products" --fanouts "12,12,12" --root "../preprocess/ogbn-products" --fake-dim 100 --adj-budget 0.64 --nfeat-budget 0.32