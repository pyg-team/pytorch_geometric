# Inference Benchmark

## Environment setup

* Create conda environment
    ```bash
    conda create -n env_name python=3.9
    ```
* Install PyTorch
* Install pytorch-geometric and related libraries
    ```bash
    pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-geometric
    ```
* Install dataset package
    ```bash
    pip install ogb
    ```
* Install jemalloc for performance benchmark
    ```bash
    cd $workspace
    git clone https://github.com/jemalloc/jemalloc.git
    cd jemalloc
    git checkout 5.2.1
    ./autogen.sh
    ./configure --prefix=$workspace/jemalloc-bin
    make
    make install
    ```

## Running benchmark

* Set environment variables
    ```bash
    source activate env_name
    export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
    export KMP_BLOCKTIME=1
    export KMP_AFFINITY=granularity=fine,compact,1,0

    jemalloc_lib=${workspace}/jemalloc-bin/lib/libjemalloc.so
    export LD_PRELOAD="$jemalloc_lib"
    export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
    ```
* Core binding: single socket / single core/ 4 cores per instance
    ```bash
    OMP_NUM_THREADS=${CORES} numactl -C 0-${LAST_CORE} -m 0 CMD......
    ```
* Execute benchmarks
    For example:
    ```bash
    python -u inference_benchmark.py --datasets=Reddit --models=gcn --eval-batch-sizes=512 --num-layers=2 --num-hidden-channels=64
    python -u inference_benchmark.py --datasets=Reddit --models=gcn --eval-batch-sizes=1024 --num-layers=3 --num-hidden-channels=128 --use-sparse-tensor
    python -u inference_benchmark.py --datasets=Reddit --models=gcn --eval-batch-sizes=1024 --num-layers=3 --num-hidden-channels=128 --full-batch
    python -u inference_benchmark.py --datasets=ogbn-products --models=sage --eval-batch-sizes=512 --num-layers=2 --num-hidden-channels=64
    python -u inference_benchmark.py --datasets=ogbn-products --models=sage --eval-batch-sizes=1024 --num-layers=3 --num-hidden-channels=128 --use-sparse-tensor
    ```

## Reminder
torch.compile can only support full-batch mode in inference now.
