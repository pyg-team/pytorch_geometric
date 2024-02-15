#!/bin/bash

CONDA_ENV=/home/XXX/anaconda3/envs/pyg
PYG_WORKSPACE=$PWD
PY_EXEC=${CONDA_ENV}/bin/python
EXEC_SCRIPT=${PYG_WORKSPACE}/node_ogb_cpu.py

# Node number:
NUM_NODES=2

# Dataset folder:
DATASET_ROOT_DIR="/home/XXX/mag/2-parts"

# Dataset name:
DATASET=ogbn-mag

# Number of epochs:
NUM_EPOCHS=3

# The batch size:
BATCH_SIZE=1024

# Number of workers for sampling:
NUM_WORKERS=2
CONCURRENCY=2

# Partition data directory:
PART_CONFIG="/home/XXX/mag/2-parts/ogbn-products-partitions/META.json"
NUM_PARTS=2

DDP_PORT=12351

# Fanout per layer:
NUM_NEIGHBORS="15,10,5"

# IP configuration path:
IP_CONFIG=${PYG_WORKSPACE}/ip_config.yaml

# Folder and filename to place logs:
logdir="logs"
mkdir -p "logs"
logname=log_${DATASET}_${NUM_PARTS}_$RANDOM
echo $logname
set -x

# stdout stored in `/logdir/logname.out`.
python launch.py --workspace "${PYG_WORKSPACE}" --num_nodes ${NUM_NODES} --num_neighbors ${NUM_NEIGHBORS} --dataset_root_dir ${DATASET_ROOT_DIR} --dataset ${DATASET}  --num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --concurrency ${CONCURRENCY} --ddp_port ${DDP_PORT} --part_config ${PART_CONFIG} --ip_config "${IP_CONFIG}" "cd /home/XXX; source ${CONDA_ENV}/bin/activate; cd ${PYG_WORKSPACE}; ${PY_EXEC} ${EXEC_SCRIPT}" |& tee ${logdir}/${logname}.txt
set +x
