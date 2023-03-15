CPU affinity mask setting for performance
=============================================
The performance of :pyg:`PyG` workloads using CPU can be significantly improved by setting a proper affinity mask. By modifying the way that native OS scheduler is distributing work between CPU cores we can effectively increase overall hardware utilisation and system performance, by minimizing core stalls and memory bounds.
The following article discusses readily available tools and environment settings that one can use to squeeze out maximum performance of Intel CPUs with :pyg:`PyG`. 

**Disclaimer:** Overall, CPU affinity can be a useful tool for improving the performance and predictability of certain types of applications, but but one configuration does not fit all cases: it is important to carefully consider whether CPU affinity is appropriate for your use case, and to test and measure the impact of any changes you make.

Using :obj:`cpu-affinity` and :obj:`filter-per-worker` setting
---------------------------------------------------------------
Setting CPU affinity mask places data loader workers threads on specific CPU cores. In effect, it allows for more efficient local memory allocation and reduces remote memory calls. Every time a process or thread moves from one core to another, registers and caches need to be flushed and reloaded. This can become very costly if it happens often, and our threads may also no longer be close to their data, or be able to share data in a cache.

Since :pyg:`PyG` 2.3 release :class:`~torch_geometric.loader.NeighborLoader` and :class:`~torch_geometric.loader.LinkLoader` officially support native solution for CPU affinity using context manager :class:`~torch_geometric.loader.AffinityMixin` class. The affinity can be enabled with :func:`enable_cpu_affinity()` method only for multi-process :class:`~torch_geometric.loader.DataLoader`, when :obj:`num_workers` > 0.
This will assign a separate CPU to each :class:`~torch_geometric.loader.DataLoader` worker at initialization. If :obj:`loader_cores` are not provided, :class:`~torch_geometric.loader.DataLoader` cores will be assigned automatically. The recommended number of workers to start with is in range [2,4], the optimum may vary based on workload characteristics.

.. code-block:: python

    loader = NeigborLoader(data, 
        num_workers=3, 
        filter_per_worker=True,
        **kwargs)
    with loader.enable_cpu_affinity(loader_cores=[0, 1, 2]):
        for batch in loader:
            pass

Binding processes to physical cores
------------------------------------

Following general performance tuning principles it is advisable to use only physical cores for deep learning workloads. While 2 logical threads run GEMM at the same time, they will be sharing the same core resources causing front end bound, such that the overhead from this front end bound is greater than the gain from running both logical threads at the same time, OpenMP threads will contend for the same GEMM execution units [1]. 

The binding can be done in many ways, however the most common tools are:

* numactl (only on Linux)
  
    .. code-block:: console

        --physcpubind=<cpus>, -C <cpus>  or --cpunodebind=<nodes>, -N <nodes>

* Intel OMP libiomp [4]
  
    .. code-block:: console

        export KMP_AFFINITY=granularity=fine,proclist=[0-<physical_cores_num-1>],explicit

* GNU libgomp

    .. code-block:: console

        export GOMP_CPU_AFFINITY="0-<physical_cores_num-1>"


Isolating :class:`~torch_geometric.loader.DataLoader` process
---------------------------------------------------------------

For the best performance combine main process affinity using the tool of your choice with the multi-process :class:`~torch_geometric.loader.DataLoader` affinity settings. The main process will perform calculations in GNN layers, while the :class:`~torch_geometric.loader.DataLoader` workers sub-processes will take care of fetching and pre-processing data to be passed to a GNN model. 
It is advisable to isolate the CPU resources made available to these two processes to achieve the best results. To do this, CPUs assigned to each affinity mask should be mutually exclusive. For example, if 4 :class:`~torch_geometric.loader.DataLoader` workers are assigned to CPUs [0,1,2,3], the main process should use the rest of available cores, i.e. by calling:

.. code-block:: console

    numactl -C 4-(N-1) --localalloc python …

where N is the total number of physical cores, with the last CPU having ID N-1. Adding “—localalloc” improves local memory allocation and keeps cache closer to active cores. 
Dual socket CPU separation
With dual-socker CPUs it might be beneficial to further isolate the processes between the sockets. This leads to decreased frequency of remote memory calls for the main process. The goal is to utilize high speed cache on local memory and reduces memory bound caused by migrating cached data between NUMA nodes. [1] This can be achieved by using :class:`~torch_geometric.loader.DataLoader` affinity and launching main process on the cores of the second socket, i.e. with:

.. code-block:: console

    numactl -C M-(N-1) -m 1 python …

where M is the cpuid of the first core of the second CPU socket. Adding a complementary memory-allocation flag “-m 1” prioritizes cache allocation on the same NUMA node, where the main process is running (alternatively for less strict memory allocation use “—preferred 1”).  This makes the data readily available on the same socket where the computation takes place. Using this setting is very workload-specific and may require some fine-tuning, as one needs to manage a trade-off between using more OMP threads versus limiting the number of remote memory calls. 
 
Improving memory bound by using non-default memory allocator (jemalloc or TCMalloc)
------------------------------------------------------------------------------------
Finally, it has been proved that for Pytorch it is advisable to use jemalloc or TCMalloc  that can generally get better memory usage than the default PyTorch memory allocator, PTMalloc. [2] You can specify non-default memory allocator by using LD_PRELOAD prior to script execution.


Quick start guidelines:	
--------------------------
The general guidelines for achieving the best performance with CPU Affinity can be summarized in the following steps:

#.	Bind execution to physical cores. Alternatively, hyperthreading can be disabled completely at a system-level.
#.	Enable multi-process Dataloder by setting :obj:`num_workers` > 0. A good estimate for initial num_workers is in range [2,4], however for more complex datasets you might want to experiment with larger number of workers. Enable :class:`~torch_geometric.loader.DataLoader` with :obj:`filter_per_worker=True` and use :obj:`enable_cpu_affinity()` feature to affinitize :class:`~torch_geometric.loader.DataLoader` cores. 
#.	Separate the cores used for main process from the DL workers' cores by using numactl, KMP_AFFINITY of libiomp5 library of GOMP_CPU_AFFINITY of libgomp library.
#.	Find the optimum number of OMP threads for your workload. The good starting point would be N-num_workers. Generally well-parallelized models will benefit from many OMP threads, however if your model computation flow has interlaced parallel & serial regions, the performance will decrease due to resource allocation needed for spawning and maintaining threads between parallel regions.
#.	Using a dual-socket CPU you might want to experiment with assigning data loading to one socket and main process to another socket with memory allocation (numactl -m) on the same socket where main process is executed. This leads to best cache-allocation and often overweighs the benefit of using more OMP threads.
#.	An additional boost in performance can be obtained by using non-default memory allocator, such as jemalloc or TCMalloc.
#.	Finding an optimal setup for CPU Affinity mask is a problem of managing the proportion of CPU time spent in each iteration for loading and preparing the data, versus time spent in computing the message-passing step. Different results may be obtained by changing model hyperparamethers, such as: batch size, number of sampled neighbors and number of layers. As a general rule, workloads which require sampling a complex graph may benefit more from reserving some CPU resources just for the data preparation step. 
