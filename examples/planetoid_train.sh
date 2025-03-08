#!/bin/bash

datasets=('Cora' 'CiteSeer' 'PubMed')
models=('agnn' 'arma' 'mixhop' 'sgc' 'tagcn' 'graphunet' 'pmlp' 'splinegnn' 'gcn2' 'super_gat' 'dna' 'gat' 'gcn' 'node2vec')

# make save dir
log_dir="logs"
if [ -d "$log_dir" ]; then
  rm -rf "${log_dir:?}/"*
else
  mkdir "$log_dir"
fi

# config
max_jobs=4
current_jobs=0

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    while [ $current_jobs -ge $max_jobs ]; do
      current_jobs=$(jobs -r | wc -l)
      sleep 1
    done

    log_file="$log_dir/${dataset}_${model}.log"
    python examples/planetoid_train.py --gnn_choice "$model" --dataset "$dataset" > "$log_file" 2>&1 &

    ((current_jobs++))
  done
done

wait
echo "All training tasks have been completed."
