# conda create a new environment
# make sure system cuda version is the same with pytorch cuda
# follow the instruction of Pyotrch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
export PATH=/usr/local/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH

conda create --name deeper_gcn_env python=3.7
# activate this enviroment
conda activate deeper_gcn_env

conda install -y pytorch=1.5.0 torchvision cudatoolkit=10.1 -c pytorch
# test if pytorch is installed successfully
python -c "import torch; print(torch.__version__)"
nvcc --version # should be same with that of torch_version_cuda (they should be the same)
python -c "import torch; print(torch.version.cuda)"

CUDA=cu101
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
## install torch-sparse
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
## install torch-cluster
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
## install torch-spline-conv
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
#pip install torch-geometric

pip install tqdm
pip install ogb
### check the version of ogb installed, if it is not the latest
python -c "import ogb; print(ogb.__version__)"
# please update the version by running
pip install -U ogb
