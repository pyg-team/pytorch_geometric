# Benchmark with Airspeed Velocity


## Note
- **--launch-method forkserver vs. spawn**: forkserver is faster but CUDA context cannot be created in the main process.
- **skip benchmarks**: Raising `NotImplementedError` in `setup()` to skip benchmarks is too slow. Instead just remove parameters from `params` to skip where possible.

```sh
pip install git+https://github.com/akihironitta/asv.git@bugfix/mamba-deadlock
```

```sh
export ASV_OPTION='-e -v -m codespace-v100 --launch-method spawn --skip-existing-successful'
asv run $ASV_OPTION \
--bench Softmax.track_fwd \
--steps 3 \
2.3.0..master
```

## TODO
- **measure performance ratio**: This increases benchmark time as it requires benchmarking pure PyTorch implementation.
- **clean up**: Reuse code across benchmarks
- **adjust unit**: e.g. ms, us, ns
