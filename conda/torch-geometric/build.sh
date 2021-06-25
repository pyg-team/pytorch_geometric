TORCH=$($PYTHON -c "import torch; print(torch.__version__)")
TORCH=${TORCH%+*}  # Get substring before the first occurence of `+`.
CUDA=$($PYTHON -c "import torch; print(torch.version.cuda)")
if [ "${CUDA}" = "None" ]; then
  CUDA="cpu"
else
  CUDA="cu${CUDA/./}"
fi

echo "Building for PyTorch $TORCH+$CUDA:"
# $PYTHON -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
$PYTHON -m pip install --no-deps --ignore-installed .
