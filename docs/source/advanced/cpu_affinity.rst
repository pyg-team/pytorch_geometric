CPU affinity settings for PyG workloads
==========================================

The performance of :pyg:`PyG` workloads using CPU can be significantly improved by setting a proper affinity mask. Processor affinity, or core binding is a modification of native OS queue scheduling algorithm, that enables an application to assign a specific set of cores to processes or threads launched during its execution on the CPU.
In consequence, it is possible to increase the overall effective hardware utilisation, by minimizing core stalls and memory bounds. It also secures the CPU resources to critical processes or threads, even if the system is under a heavy load. The affinity targets the two main performance-critical regions:

* Execution bind: indicates a  core where process/thread will run.

* Memory bind: indicates a preferred memory area where memory pages will be bound (local areas in NUMA machine).

The following article discusses readily available tools and environment settings that one can use to maximize the performance of Intel CPUs with :pyg:`PyG`.

**Disclaimer:** Overall, CPU affinity can be a useful tool for improving the performance and predictability of certain types of applications, but but one configuration does not fit all cases: it is important to carefully consider whether CPU affinity is appropriate for your use case, and to test and measure the impact of any changes you make.

Using :attr:`cpu-affinity` and :attr:`filter-per-worker` setting
-------------------------------------------------------------------
Each :pyg:`PyG` workload can be parallelized using a base-class Pytorch iterator :class:`MultiProcessingDataLoaderIter`, which is automatically enabled, when the :attr:`num_workers` > 0 on :class:`~torch_geometric.loader.DataLoader` initialization. It creates number of sub-processes equal to :attr:`num_workers` that will run in parallel to the main process.
Setting CPU affinity mask for the data loading processes places :class:`~torch_geometric.loader.DataLoader` workers threads on specific CPU cores. In effect, it allows for more efficient data batch preparation by allocating pre-fetched batches in local memory. Every time a process or thread moves from one core to another, registers and caches need to be flushed and reloaded. This can become very costly if it happens often, and our threads may also no longer be close to their data, or be able to share data in a cache.

Since :pyg:`PyG` 2.3 release :class:`~torch_geometric.loader.NeighborLoader` and :class:`~torch_geometric.loader.LinkLoader` officially support native solution for CPU affinity using context manager :class:`~torch_geometric.loader.AffinityMixin` class. The affinity can be enabled with :func:`enable_cpu_affinity()` method only for multi-process :class:`~torch_geometric.loader.DataLoader` (only when :attr:`num_workers` > 0 ).
This will assign a separate core to each worker at initialization. User-defined list of core IDs may be assinged using :attr:`loader_cores` argument. Else, :class:`~torch_geometric.loader.DataLoader` cores will be assigned automatically, starting at core ID 0. As of now, only a signle core can be assigned to a worker, hence by default multi-threading is disabled in workers' processes.
The recommended number of workers to start with is in range [2,4], the optimum may vary based on workload characteristics.

.. code-block:: python

    loader = NeigborLoader(data,
        num_workers=3,
        filter_per_worker=True,
        **kwargs)
    with loader.enable_cpu_affinity(loader_cores=[0, 1, 2]):
        for batch in loader:
            pass

It is generally adivisable to use :attr:`filter-per-worker=True`, when enabling multi-process dataloader.
The workers prepare each :obj:`input_data` tensor: first by sampling the node indices using pre-defined sampler in :func:`collate_fn()` and secondly triggering :func:`filter_fn()`.
Filtering function selects node feature vectors from the complete input :class:`~torch_geometric.data.Data` tensor loaded into DRAM. This is a memory-expensive call which takes a significant time of each DataLoader iteration.
By default :attr:`filter-per-worker` is set to :attr:`False`, which causes that :func:`filter_fn()` execution is sent back to the main process. This can cause performance issues, because the main process will not be able to process all requests efficiently, especially with larger number of workers.
When :attr:`filter-per-worker=True`, each worker's subprocess performs the filtering within it's CPU resource in effect main process resources are relieved and can be secured only for Message-Passing computation.

Binding processes to physical cores
------------------------------------

Following general performance tuning principles it is advisable to use only physical cores for deep learning workloads. While 2 logical threads run GEMM at the same time, they will be sharing the same core resources causing front end bound, such that the overhead from this front end bound is greater than the gain from running both logical threads at the same time, because OpenMP threads will contend for the same GEMM execution units [1]_.

The binding can be done in many ways, however the most common tools are:

* numactl (only on Linux)

    .. code-block:: console

        --physcpubind=<cpus>, -C <cpus>  or --cpunodebind=<nodes>, -N <nodes>

* Intel OMP libiomp [4]_

    .. code-block:: console

        export KMP_AFFINITY=granularity=fine,proclist=[0-<physical_cores_num-1>],explicit

* GNU libgomp

    .. code-block:: console

        export GOMP_CPU_AFFINITY="0-<physical_cores_num-1>"


Isolating :class:`~torch_geometric.loader.DataLoader` process
---------------------------------------------------------------

For the best performance, it is required combine main process affinity using the tools listed above, with the multi-process :class:`~torch_geometric.loader.DataLoader` affinity settings.
In each parallelized :pyg:`PyG` workload execution, the main process performs Message-Passing updates over GNN layers, while the :class:`~torch_geometric.loader.DataLoader` workers sub-processes take care of fetching and pre-processing data to be passed to a GNN model.
It is advisable to isolate the CPU resources made available to these two processes to achieve the best results. To do this, CPUs assigned to each affinity mask should be mutually exclusive. For example, if 4 :class:`~torch_geometric.loader.DataLoader` workers are assigned to CPUs [0,1,2,3], the main process should use the rest of available cores, i.e. by calling:

.. code-block:: console

    numactl -C 4-(N-1) --localalloc python …

where N is the total number of physical cores, with the last CPU having core ID N-1. Adding "--localalloc" improves local memory allocation and keeps cache closer to active cores.

.. _Dual socket CPU separation:

Dual socket CPU separation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With dual-socket CPUs it might be beneficial to further isolate the processes between the sockets. This leads to decreased frequency of remote memory calls for the main process. The goal is to utilize high speed cache on local memory and reduces memory bound caused by migrating cached data between NUMA nodes [1]_. This can be achieved by using :class:`~torch_geometric.loader.DataLoader` affinity and launching main process on the cores of the second socket, i.e. with:

.. code-block:: console

    numactl -C M-(N-1) -m 1 python …

where M is the cpuid of the first core of the second CPU socket. Adding a complementary memory-allocation flag "-m 1" prioritizes cache allocation on the same NUMA node, where the main process is running (alternatively for less strict memory allocation use "--preferred 1").  This makes the data readily available on the same socket where the computation takes place. Using this setting is very workload-specific and may require some fine-tuning, as one needs to manage a trade-off between using more OMP threads versus limiting the number of remote memory calls.

Improving memory bound by using non-default memory allocator (jemalloc or TCMalloc)
------------------------------------------------------------------------------------
On the final note, following CPU performanc optimization for Pytorch, also for :pyg:`PyG` it is advised to use jemalloc or TCMalloc that can generally get better memory usage than the default PyTorch memory allocator, PTMalloc [2]_. A non-default memory allocator can be specified using LD_PRELOAD prior to script execution [3]_.


Quick start guidelines
--------------------------
The general guidelines for achieving the best performance with CPU Affinity can be summarized in the following steps:

#.  Test if your dataset benefits from using parallel Dataloader. For some datasets it might be more beneficial to use plain serial Dataloader, especially when the dimensions of the input :class:`~torch_geometric.data.Data` are relatively small.
#.  Enable multi-process Dataloder by setting :attr:`num_workers` > 0. A good estimate for initial num_workers is in range [2,4], however for more complex datasets you might want to experiment with larger number of workers. Enable :class:`~torch_geometric.loader.DataLoader` with :obj:`filter_per_worker=True` and use :obj:`enable_cpu_affinity()` feature to affinitize :class:`~torch_geometric.loader.DataLoader` cores. 
#.	Bind execution to physical cores. Alternatively, hyperthreading can be disabled completely at a system-level.
#.	Separate the cores used for main process from the DL workers' cores by using `numactl`, `KMP_AFFINITY` of `libiomp5` library, or `GOMP_CPU_AFFINITY` of `libgomp` library.
#.	Find the optimum number of OMP threads for your workload. The good starting point would be (N - `num_workers`). Generally well-parallelized models will benefit from many OMP threads, however if your model computation flow has interlaced parallel & serial regions, the performance will decrease due to resource allocation needed for spawning and maintaining threads between parallel regions.
#.	Using a dual-socket CPU you might want to experiment with assigning data loading to one socket and main process to another socket with memory allocation (`numactl -m`) on the same socket where main process is executed. This leads to best cache-allocation and often overweighs the benefit of using more OMP threads.
#.	An additional boost in performance can be obtained by using non-default memory allocator, such as `jemalloc` or `TCMalloc`.
#.	Finding an optimal setup for CPU Affinity mask is a problem of managing the proportion of CPU time spent in each iteration for loading and preparing the data, versus time spent in computing the message-passing step. Different results may be obtained by changing model hyperparamethers, such as: batch size, number of sampled neighbors and number of layers. As a general rule, workloads which require sampling a complex graph may benefit more from reserving some CPU resources just for the data preparation step.

Example results
-----------------
The figure below presents the outcome of applying CPU affinity mask to :py:obj:`training_benchmark.py`.
Measurements were taken for the variable number of workers, while other hyperparameters for each benchmark were constant: `--warmup 0 --use-sparse-tensor --num-layers 3 --num-hidden-channels 128 --batch-sizes 2048`.
3 different affinity configurations are presented:

* "Baseline" - only `OMP_NUM_THREADS` changes

.. code-block:: console

    OMP_NUM_THREADS=(N-num_workers) python training_benchmark.py --num-workers …

* "Aff" - DataLoader Process 1st socket, Main Process on 1st & 2nd socket, 98-110 threads

.. code-block:: console

    LD_PRELOAD=(path)/libjemalloc.so (path)/libiomp5.so MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto OMP_NUM_THREADS=(N-num_workers) KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 numactl -C <num_workers-(N-1)> --localalloc python training_benchmark.py --cpu-affinity --filter_per_worker --num-workers …


* "Aff+SocketSep" - Dataloader on 1st socket, Main on 2nd socket, 60 threads

.. code-block:: console

    LD_PRELOAD=(path)/libjemalloc.so (path)/libiomp5.so MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto OMP_NUM_THREADS=(N-M) KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 numactl -C <M-(N-1)> -m 1 python training_benchmark.py --cpu-affinity --filter_per_worker --num-workers ...

Training times for each model+dataset combination were obtained by taking a mean of results  at variable number of dataloader workers: [0,2,4,8,16] for the Baseline and [2,4,8,16] workers for each affinity configuration. 
Then the affinity means were normalized with respect to the mean Baseline measurement. This value is denoted on the y-axis. The labels above each result indicate the end-to-end performance gain from using the discussed config.
Taking the average over all model+dataset samples, the average training time is increased by: 1.53x for plain affinity and 1.85x for the affinity with socket seapration discussed in `Dual socket CPU separation`_.

.. figure:: ../_figures/training_affinity.png
    :align: center
    :width: 1600px

    Generated on pre-production dual-socket Intel(R) Xeon(R) Platinum 8481C @ 2.0Ghz (2 x 56) cores CPU.


.. [1] Grokking PyTorch Intel CPU Performance From First Principles, PyTorch Tutorials 2.0.0+cu117 Documentation, https://pytorch.org/tutorials/intermediate/torchserve_with_ipex.html

.. [2] Grokking PyTorch Intel CPU Performance From First Principles (Part 2), PyTorch Tutorials 2.0.0+cu117 Documentation, https://pytorch.org/tutorials/intermediate/torchserve_with_ipex_2.html

.. [3] Performance Tuning Guide, PyTorch Tutorials 2.0.0+cu117 Documentation, https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

.. [4] Maximize Performance of Intel® Software Optimization for PyTorch* on..., Intel, https://www.intel.com/content/www/us/en/developer/articles/technical/how-to-get-better-performance-on-pytorchcaffe2-with-intel-acceleration.html
