# Benchmark with Airspeed Velocity


## Note
- **--launch-method forkserver vs. spawn**: forkserver is faster but CUDA context cannot be created in the main process.
- **skip benchmarks**: Raising `NotImplementedError` in `setup()` to skip benchmarks is too slow. Instead just remove parameters from `params` to skip where possible.

## TODO
- **measure performance ratio**: This increases benchmark time as it requires benchmarking pure PyTorch implementation.
