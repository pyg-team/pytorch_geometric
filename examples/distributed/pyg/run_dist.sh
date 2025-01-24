#!/bin/bash

PYG_WORKSPACE=$PWD
USER=user
CONDA_ENV=pygenv
CONDA_DIR="/home/${USER}/anaconda3"
PY_EXEC="${CONDA_DIR}/envs/${CONDA_ENV}/bin/python"
EXEC_SCRIPT="${PYG_WORKSPACE}/node_ogb_cpu.py"
CMD="cd ${PYG_WORKSPACE}; ${PY_EXEC} ${EXEC_SCRIPT}"

# Node number:
NUM_NODES=2

# Dataset name:
DATASET=ogbn-products

# Dataset folder:
DATASET_ROOT_DIR="../../../data/partitions/${DATASET}/${NUM_NODES}-parts"

# Number of epochs:
NUM_EPOCHS=10

# The batch size:
BATCH_SIZE=1024

# Fanout per layer:
NUM_NEIGHBORS="5,5,5"

# Number of workers for sampling:
NUM_WORKERS=2
CONCURRENCY=4

# DDP Port
DDP_PORT=11111

# IP configuration path:
IP_CONFIG=${PYG_WORKSPACE}/ip_config.yaml

# Folder and filename to place logs:
logdir="logs"
mkdir -p "logs"
logname=log_${DATASET}_${NUM_PARTS}_$RANDOM
echo "stdout stored in ${PYG_WORKSPACE}/${logdir}/${logname}"
set -x

# stdout stored in `/logdir/logname.out`.
python launch.py --workspace ${PYG_WORKSPACE} --ip_config ${IP_CONFIG} --ssh_username ${USER} --num_nodes ${NUM_NODES} --num_neighbors ${NUM_NEIGHBORS} --dataset_root_dir ${DATASET_ROOT_DIR} --dataset ${DATASET}  --num_epochs ${NUM_EPOCHS} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --concurrency ${CONCURRENCY} --ddp_port ${DDP_PORT} "${CMD}" |& tee ${logdir}/${logname} &
pid=$!
echo "started launch.py: ${pid}"
# kill processes at script exit (Ctrl + C)
trap "kill -2 $pid" SIGINT
wait $pid
set +x
