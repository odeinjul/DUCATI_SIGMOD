# cd preprocess
# mkdir ogbn-papers100M
# mkdir ogbn-products
# mkdir mag240M
# mkdir friendster

python ../preprocess/preprocess_datagen.py --dataset "ogbn-products" \
 --root "../../processed_dataset/ogbn-products" --save-path  "../preprocess/ogbn-products/" --dgl-graph

python ../preprocess/preprocess_datagen.py --dataset "ogbn-papers100M" \
--root "../../processed_dataset/ogbn-papers100M" --save-path  "../preprocess/ogbn-papers100M/" --dgl-graph

python ../preprocess/preprocess_datagen.py --dataset "mag240M" \
--root "../../processed_dataset/mag240M" --save-path  "../preprocess/mag240M/" --dgl-graph

python ../preprocess/preprocess_datagen.py --dataset "friendster" \
--root "../../processed_dataset/friendster" --save-path  "../preprocess/friendster/" --dgl-graph