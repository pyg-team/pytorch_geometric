#!/bin/bash

CONDA_ENV=/home/XXX/anaconda3/envs/pyg24
PYG_WORKSPACE=$PWD
PY_EXEC=${CONDA_ENV}/bin/python
EXEC_SCRIPT=${PYG_WORKSPACE}/distributed_cpu.py

# node number
NUM_NODES=2

# dataset folder
DATASET_ROOT_DIR="/home/XXX/mag/2-parts"

# process number for training
NUM_TRAINING_PROCS=1

# dataset name
DATASET=ogbn-mag

# num epochs to run for
NUM_EPOCHS=3

BATCH_SIZE=1024

# number of workers for sampling
NUM_WORKERS=2
CONCURRENCY=2

#partition data directory
PART_CONFIG="/home/XXX/mag/2-parts/ogbn-products-partitions/META.json"
NUMPART=2

DDP_PORT=12351
# fanout per layer
NUM_NEIGHBORS="15,10,5"

#ip_config path
IP_CONFIG=${PYG_WORKSPACE}/ip_config.yaml


# Folder and filename where you want your logs.
logdir="logs"
mkdir -p "logs"
logname=log_${DATASET}_${NUMPART}_$RANDOM
echo $logname
set -x

# stdout stored in /logdir/logname.out
python launch.py --workspace ${PYG_WORKSPACE} --num_nodes ${NUM_NODES} --num_neighbors ${NUM_NEIGHBORS} --dataset_root_dir ${DATASET_ROOT_DIR} --dataset ${DATASET}  --num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --concurrency ${CONCURRENCY} --ddp_port ${DDP_PORT} --part_config ${PART_CONFIG} --ip_config ${IP_CONFIG} "cd /home/XXX; source ${CONDA_ENV}/bin/activate; cd ${PYG_WORKSPACE}; ${PY_EXEC} ${EXEC_SCRIPT}" |& tee ${logdir}/${logname}.txt
set +x
