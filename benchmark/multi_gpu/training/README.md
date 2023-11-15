# Training Benchmark

## Environment setup

Optional, XPU only:
```
install intel_extension_for_pytorch
install oneccl_bindings_for_pytorch
```

## Running benchmark

Run benchmark, e.g. assuming you have 3 NVIDIA GPUs:
```
mpirun -np 3 python training_benchmark.py --dataset ogbn-products --model edge_cnn --num-epochs 3 --device cuda
```


Run benchmark, e.g. assuming you have 2 XPUs:
```
mpirun -np 2 python training_benchmark.py --dataset ogbn-products --model edge_cnn --num-epochs 3 --device xpu
```

