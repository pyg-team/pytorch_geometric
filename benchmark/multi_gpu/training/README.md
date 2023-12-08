# Training Benchmark

## Running benchmark on CUDA GPU

Run benchmark, e.g. assuming you have `n` NVIDIA GPUs:

```
python training_benchmark_cuda.py --dataset ogbn-products --model edge_cnn --num-epochs 3 --n_gpus <n>
```

## Running benchmark on Intel GPU

## Environment setup

```
install intel_extension_for_pytorch
install oneccl_bindings_for_pytorch
```

Run benchmark, e.g. assuming you have `n` XPUs:

```
mpirun -np <n> python training_benchmark_xpu.py --dataset ogbn-products --model edge_cnn --num-epochs 3
```
