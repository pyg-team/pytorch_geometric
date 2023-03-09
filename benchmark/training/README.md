# Training Benchmark

## Environment setup

1. Confirm that PyG is properly installed.
2. Install dataset package:
   ```bash
   pip install ogb
   ```
3. Install `jemalloc` for performance benchmark:
   ```bash
   cd ${workspace}
   git clone https://github.com/jemalloc/jemalloc.git
   cd jemalloc
   git checkout 5.2.1
   ./autogen.sh
   ./configure --prefix=${workspace}/jemalloc-bin
   make
   make install
   ```

## Running benchmark

1. Set environment variables:
   ```bash
   source activate env_name
   export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
   export KMP_BLOCKTIME=1
   export KMP_AFFINITY=granularity=fine,compact,1,0

   jemalloc_lib=${workspace}/jemalloc-bin/lib/libjemalloc.so
   export LD_PRELOAD="$jemalloc_lib"
   export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
   ```
2. Core binding, *e.g.*, single socket / single core / 4 cores per instance:
   ```bash
   OMP_NUM_THREADS=${CORES} numactl -C 0-${LAST_CORE} -m 0 CMD......
   ```
3. Execute benchmarks, *e.g.*:
   ```bash
   python training_benchmark.py --models=gcn --datasets=Reddit --num-workers=0 --batch-sizes=512 --num-layers=2 --num-hidden-channels=64 --num-steps=50
   python training_benchmark.py --models=gcn --datasets=Reddit --num-workers=0 --batch-sizes=512 --num-layers=2 --num-hidden-channels=64 --num-steps=50 --use-sparse-tensor
   python training_benchmark.py --models=sage --datasets=ogbn-products --num-workers=0 --batch-sizes=512 --num-layers=2 --num-hidden-channels=64 --num-steps=50
   python training_benchmark.py --models=sage --datasets=ogbn-products --num-workers=0 --batch-sizes=512 --num-layers=2 --num-hidden-channels=64 --num-steps=50 --use-sparse-tensor
   ```
