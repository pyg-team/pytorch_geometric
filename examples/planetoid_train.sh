#!/bin/bash

# make save dir
log_dir="logs"
if [ -d "$log_dir" ]; then
  rm -rf "${log_dir:?}/"*
else
  mkdir "$log_dir"
fi

# config
max_jobs=6
current_jobs=0

models=('agnn' 'arma' 'dna' 'gat' 'gcn' 'graphunet' 'mixhop' 'node2vec' 'pmlp' 'sgc' 'splinegnn' 'super_gat' 'tagcn' 'gcn2')
# script1 ===============================================

# datasets=('Cora' 'CiteSeer' 'PubMed')

# for dataset in "${datasets[@]}"; do
#   for model in "${models[@]}"; do
#     while [ $current_jobs -ge $max_jobs ]; do
#       current_jobs=$(jobs -r | wc -l)
#       sleep 1
#     done

#     log_file="$log_dir/${dataset}_${model}.log"
#     python examples/planetoid_train.py --gnn_choice "$model" --dataset "$dataset" > "$log_file" 2>&1 &

#     ((current_jobs++))
#   done
# done

# script2 ================================================

dataset='Cora'
params=(
  '--hidden_channels 16 --lr 0.01 --wd 5e-4 -e 200 --use_normalize_features' # agnn
  '--hidden_channels 16 --lr 0.01 --wd 5e-4 -e 400 --use_normalize_features' # arma
  '--hidden_channels 128 --lr 0.005 --wd 5e-4 -e 200' # dna
  '--hidden_channels 8 --lr 0.005 --wd 5e-4 -e 200 --use_normalize_features' # gat
  '--hidden_channels 16 --lr 0.01 --wd 5e-4 -e 200 --use_normalize_features' # gcn
  '--hidden_channels 32 --lr 0.01 --wd 1e-3 -e 200' # graphunet
  '--hidden_channels 60 --lr 0.5 --wd 5e-3 -e 100' # mixhop
  '--lr 0.01 -e 100' # node2vec
  '--hidden_channels 16 --lr 0.01 --wd 5e-4 -e 200 --use_normalize_features' # pmlp
  '--lr 0.2 --wd 5e-3 -e 100' # sgc
  '--hidden_channels 16 --lr 0.01 --wd 5e-3 -e 200 --num_val 500 --num_test 500 --use_target_indegree' # splinegnn
  '--lr 0.005 --wd 5e-4 -e 500 --use_normalize_features' # super_gat
  '--hidden_channels 16 --lr 0.01 --wd 5e-4 -e 200 --use_normalize_features' # tagcn
  '--hidden_channels 64 --lr 0.01 --wd 5e-4 -e 1000 --use_normalize_features' # gcn2
)

for i in "${!models[@]}"; do
  model="${models[$i]}"
  param="${params[$i]}"

  while [ $current_jobs -ge $max_jobs ]; do
    current_jobs=$(jobs -r | wc -l)
    sleep 1
  done

  log_file="$log_dir/${dataset}_${model}.log"
  echo "python examples/planetoid_train.py --gnn_choice $model --dataset $dataset $param > $log_file 2>&1 &"
  python examples/planetoid_train.py --gnn_choice "$model" --dataset "$dataset" $param > "$log_file" 2>&1 &

  ((current_jobs++))
done
# ========================================================

wait
echo "All training tasks have been completed."
