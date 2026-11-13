#!/bin/bash
# Installs the dependencies of kumo_gretriever.py and kumo_openclaw.py inside
# the NGC PyG container, the only tested environment. From examples/llm:
#
#   docker run --gpus all -it --name kumo -v $PWD:/workspace/kumo \
#       -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#       nvcr.io/nvidia/pyg:26.09-py3
#   cd /workspace/kumo && bash kumo_gretriever_setup.sh
#
# The installs live in that container (`docker start -ai kumo` returns to it),
# and the mounted cache keeps the model and RelBench downloads.
#
# Nemotron-H needs the mamba-ssm and causal-conv1d CUDA kernels, which have no
# prebuilt wheels for this container and do not build from PyPI as is: their
# setup.py compiles with -std=c++17, but the container's PyTorch 2.14 headers
# require C++20 ("#error C++20 or later compatible compiler is required to use
# ATen"). This script therefore downloads both source distributions, patches
# -std=c++17 to -std=c++20 and builds them (about 10 minutes). The patch is
# needed until a newer container ships C++20-ready builds.
set -euo pipefail

pip install "transformers==4.57.6" accelerate \
    "git+https://github.com/NVIDIA/structured-data-models@d4c82d7e720553114e54962d66dfddfad9730107"

WORK=$(mktemp -d)
cd "$WORK"
pip download --no-deps --no-binary :all: --no-build-isolation \
    mamba-ssm==2.2.6.post3 causal-conv1d==1.7.0
tar xzf mamba_ssm-2.2.6.post3.tar.gz
tar xzf causal_conv1d-1.7.0.tar.gz
sed -i 's/-std=c++17/-std=c++20/g' mamba_ssm-2.2.6.post3/setup.py \
    causal_conv1d-1.7.0/setup.py
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE MAX_JOBS=${MAX_JOBS:-16} \
    pip install --no-build-isolation --no-deps ./causal_conv1d-1.7.0 \
    ./mamba_ssm-2.2.6.post3
cd - > /dev/null
rm -rf "$WORK"

python -c "import causal_conv1d, mamba_ssm; print('mamba-ssm', mamba_ssm.__version__)"
# On Arm64 (e.g., GB200), also run `export HF_HUB_DISABLE_XET=1` before
# downloading the models, as hf_xet does not load there.
