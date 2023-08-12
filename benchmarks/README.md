# Benchmark with Airspeed Velocity


## Note
- **--launch-method forkserver vs. spawn**: forkserver is faster but CUDA context cannot be created in the main process.
- **skip benchmarks**: Raising `NotImplementedError` in `setup()` to skip benchmarks is too slow. Instead just remove parameters from `params` to skip where possible.

```sh
pip install git+https://github.com/akihironitta/asv.git@master
```

```sh
export ASV_OPTIONS='-e -v -m codespace-v100 --launch-method spawn --skip-existing-successful'
asv run $ASV_OPTIONS \
--bench Softmax.track_fwd \
--steps 3 \
2.3.0..master
```
