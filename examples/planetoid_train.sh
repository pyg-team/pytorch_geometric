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

# script1 ===============================================

# datasets=('Cora' 'CiteSeer' 'PubMed')
# models=('agnn' 'arma' 'mixhop' 'sgc' 'tagcn' 'graphunet' 'pmlp' 'splinegnn' 'gcn2' 'super_gat' 'dna' 'gat' 'gcn' 'node2vec')

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
models=('agnn' 'arma' 'dna' 'gat' 'gcn' 'graphunet' 'mixhop' 'node2vec' 'pmlp' 'sgc' 'splinegnn' 'super_gat' 'tagcn' 'gcn2')
params=(
  '--hidden_channels 16 --lr 0.01 --wd 5e-4 -e 200 --default_split --use_normalize_features' # agnn
  '--hidden_channels 16 --lr 0.01 --wd 5e-4 -e 400 --default_split --use_normalize_features' # arma
  '--hidden_channels 128 --lr 0.005 --wd 5e-4 -e 200 --num_val 542 --num_test 1624' # dna
  '--hidden_channels 8 --lr 0.005 --wd 5e-4 -e 200 --default_split --use_normalize_features' # gat
  '--hidden_channels 16 --lr 0.01 --wd 5e-4 -e 200 --default_split --use_normalize_features' # gcn
  '--hidden_channels 32 --lr 0.01 --wd 1e-3 -e 200 --default_split' # graphunet
  '--hidden_channels 60 --lr 0.5 --wd 5e-3 -e 100 --default_split' # mixhop
  '--lr 0.01 -e 100 --default_split' # node2vec
  '--hidden_channels 16 --lr 0.01 --wd 5e-4 -e 200 --default_split --use_normalize_features' # pmlp
  '--lr 0.2 --wd 5e-3 -e 100 --default_split' # sgc
  '--hidden_channels 16 --lr 0.01 --wd 5e-3 -e 200 --num_val 500 --num_test 500 --use_target_indegree' # splinegnn
  '--lr 0.005 --wd 5e-4 -e 500 --default_split --use_normalize_features' # super_gat
  '--hidden_channels 16 --lr 0.01 --wd 5e-4 -e 200 --default_split --use_normalize_features' # tagcn
  '--hidden_channels 64 --lr 0.01 --wd 5e-4 -e 1000 --default_split --use_normalize_features' # gcn2
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
