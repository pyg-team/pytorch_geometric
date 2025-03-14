#!/bin/bash

scripts=(
  "agnn.py"
  "arma.py"
  "cora.py"
  "dna.py"
  "gcn2_cora.py"
  "graph_unet.py"
  "mixhop.py"
  "pmlp.py"
  "sgc.py"
  "super_gat.py"
  "tagcn.py"
  "gat.py"
  "gcn.py"
  "node2vec.py"
)

log_dir="logs"
if [ -d "$log_dir" ]; then
  rm -rf "${log_dir:?}/"*
else
  mkdir "$log_dir"
fi

max_jobs=4
current_jobs=0

for script in "${scripts[@]}"; do
  while [ $current_jobs -ge $max_jobs ]; do
    current_jobs=$(jobs -r | wc -l)
    sleep 1
  done

  log_file="$log_dir/${script%.py}.log"
  python "examples/$script" > "$log_file" 2>&1 &

  ((current_jobs++))
done

wait
echo "All Python scripts have been executed."
