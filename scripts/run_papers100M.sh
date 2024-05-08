# torchrun --nproc_per_node 8 train_ducati.py --num-trainers 8 --num-epochs 6 --root /home/ubuntu/ducati_workspace/dataset/ogbn-papers100M --dataset ogbn-papers100M --nfeat-budget 1.973 --adj-budget 0.027 --breakdown --num-hidden 128 --model sage --fan-out 20,20,20 --batch-size 1536
# torchrun --nproc_per_node 4 train_ducati.py --num-trainers 4 --num-epochs 6 --root /home/ubuntu/ducati_workspace/dataset/ogbn-papers100M --dataset ogbn-papers100M --nfeat-budget 1.973 --adj-budget 0.027 --breakdown --num-hidden 128 --model sage --fan-out 20,20,20 --batch-size 1536
torchrun --nproc_per_node 2 train_ducati.py --num-trainers 2 --num-epochs 6 --root /home/ubuntu/ducati_workspace/dataset/ogbn-papers100M --dataset ogbn-papers100M --nfeat-budget 1.473 --adj-budget 0.027 --breakdown --num-hidden 128 --model sage --fan-out 20,20,20 --batch-size 1536
torchrun --nproc_per_node 1 train_ducati.py --num-trainers 1 --num-epochs 6 --root /home/ubuntu/ducati_workspace/dataset/ogbn-papers100M --dataset ogbn-papers100M --nfeat-budget 1.473 --adj-budget 0.027 --breakdown --num-hidden 128 --model sage --fan-out 20,20,20 --batch-size 1536
# torchrun --nproc_per_node 8 train_ducati.py --num-trainers 8 --num-epochs 6 --root /home/ubuntu/ducati_workspace/dataset/ogbn-papers100M --dataset ogbn-papers100M --nfeat-budget 1.973 --adj-budget 0.027 --breakdown --num-hidden 128 --model gat --num-heads 8 --fan-out 20,20,20 --batch-size 1536
# torchrun --nproc_per_node 4 train_ducati.py --num-trainers 4 --num-epochs 6 --root /home/ubuntu/ducati_workspace/dataset/ogbn-papers100M --dataset ogbn-papers100M --nfeat-budget 1.973 --adj-budget 0.027 --breakdown --num-hidden 128 --model gat --num-heads 8 --fan-out 20,20,20 --batch-size 1536
# torchrun --nproc_per_node 2 train_ducati.py --num-trainers 2 --num-epochs 6 --root /home/ubuntu/ducati_workspace/dataset/ogbn-papers100M --dataset ogbn-papers100M --nfeat-budget 1.973 --adj-budget 0.027 --breakdown --num-hidden 128 --model gat --num-heads 8 --fan-out 20,20,20 --batch-size 1536
# torchrun --nproc_per_node 1 train_ducati.py --num-trainers 1 --num-epochs 6 --root /home/ubuntu/ducati_workspace/dataset/ogbn-papers100M --dataset ogbn-papers100M --nfeat-budget 1.973 --adj-budget 0.027 --breakdown --num-hidden 128 --model gat --num-heads 8 --fan-out 20,20,20 --batch-size 1536