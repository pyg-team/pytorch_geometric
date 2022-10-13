#!/bin/bash

export PYTHON_VERSION=$1
export TORCH_VERSION=$2
export CUDA_VERSION=$3

export CONDA_PYTORCH_CONSTRAINT="pytorch==${TORCH_VERSION%.*}.*"

if [ "${CUDA_VERSION}" = "cpu" ]; then
  export CONDA_CUDATOOLKIT_CONSTRAINT="cpuonly  # [not osx]"
else
  case $CUDA_VERSION in
    cu116)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==11.6.*"
      ;;
    cu115)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==11.5.*"
      ;;
    cu113)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==11.3.*"
      ;;
    cu111)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==11.1.*"
      ;;
    cu102)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==10.2.*"
      ;;
    cu101)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==10.1.*"
      ;;
    *)
      echo "Unrecognized CUDA_VERSION=$CUDA_VERSION"
      exit 1
      ;;
  esac
fi

echo "PyTorch $TORCH_VERSION+$CUDA_VERSION"
echo "- $CONDA_PYTORCH_CONSTRAINT"
echo "- $CONDA_CUDATOOLKIT_CONSTRAINT"

if [ "${CUDA_VERSION}" = "cu116" ]; then
  conda build . -c pytorch -c rusty1s -c default -c nvidia -c conda-forge --output-folder "$HOME/conda-bld"
else
  conda build . -c pytorch -c rusty1s -c default -c nvidia --output-folder "$HOME/conda-bld"
fi
