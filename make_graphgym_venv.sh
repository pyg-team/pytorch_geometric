#!/usr/bin/env bash

# generate graphgym-ready venv and dependencies

# conda update conda
# conda create -n graphgym_v0
# conda activate graphgym_v0

# # Mac (CPU only)
conda install pytorch torchvision torchaudio -c pytorch
# # Verges, Linux
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# # install PyG with pip editable, locally from own files
pip install -e /Users/beng/Documents/pytorch_geometric
# # install PyG from my own github
# pip install git+https://github.com/bengutteridge/pytorch_geometric.git
pip install torchmetrics

# for running graphgym
pip install pytorch_lightning
pip install yacs
pip install ogb
pip install tensorboard
pip install tensorboardX