Distributed Training for PyG
=================================================

In real life applications graphs often consists of billions of nodes that can't be fitted into a single system memory. This is when the distributed training comes in handy. By allocating a number of partitions of the large graph into a cluster of CPUs one can deploy a synchronized model training on the whole database at once, by making use of `PyTorch Distributed Data Parallel (DDP) <https://pytorch.org/docs/stable/notes/ddp.html>`_ training. The architecture seamlessly distributes graph neural network training across multiple nodes by integrating `Remote Procedure Call (RPC) <https://pytorch.org/docs/stable/rpc.html>`_ for efficient sampling and retrieval of non-local features into standard DDP for model training. This distributed training implementation doesn't require any additonal packages to be installed on top of a default PyG stack. In the future the solution will also be available for Intel's GPUs.

Key Advantages
-----------------
.. (TODO: add links)
#. Balanced graph partitioning with METIS for large graph databases, using ``Partitoner``
#. Utilizing DDP for model training in conjunction with RPC for remote sampling and feature calls, with TCP and the 'gloo' backend specifically tailored for CPU-based sampling, enhances the efficiency and scalability of the training process.
#. The implementation of a custom ``GraphStore``/``FeatureStore`` API provides a flexible and tailored interface for large graph structure information and feature storage.
#. ``DistNeighborSampler`` capable of node neighborhood sampling in both local and remote partitions, through RPC communication channel with other samplers, maintaining a consistent data structure ``Data``/``HeteroData`` at the output
#. The ``DistLoader``/``DistNeighborLoader``/``DistLinkLoader`` offers a high-level abstraction for managing sampler processes, ensuring simplicity and seamless integration with standard PyG Loaders. This facilitates easier development and harnesses the robustness of the torch dataloader
#. Incorporating Python's `asyncio` library for asynchronous processing on top of torch RPC further enhances the system's responsiveness and overall performance. This solution follows originally from the GLT distributed library.
#. Furthermore we provide homomgenous and heretogenous graph support with code examples, used in both edge and node-level prediction tasks.

The purpose of this manual is to guide you through the most important steps of deploying your distributed training application. For the code examples, please refer to:

* `partition_graph.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/distributed/pyg/partition_graph.py>`_ for graph partitioning
* `distributed_cpu.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/distributed/pyg/distributed_cpu.py>`_ for end-to-end GNN (GraphSAGE) model training with homogenous or heterogenous data


1. Graph Partitioning
------------------------

The first step for distributed training is to split the graph into multiple smaller partitions, which can then be loaded into nodes of the cluster. This is a pre-processing step that can be done once as the resulting dataset ``.pt`` files can be reused. The ``Partitoner`` build on top of ``ClusterData``, uses pyg-lib implementation of METIS `pyg_lib.partition <https://pyg-lib.readthedocs.io/en/latest/modules/partition.html>`_ algorithm to perform graph partitioning in an efficient way, even on very large graphs. By default METIS always tries to balance the number of nodes of each type in each partition and minimize the amount of edges between the partitions. This guarantees that the partition provides accessibility to all neighboring local vertices, enabling samplers to perform local computations without the need for inter-communication. Through this partitioning approach, every edge receives a distinct assignment, although certain vertices may be replicated. The vertices shared between partitions are so called "halo nodes".
Please note that METIS requires undirected, homogenous graph as input, but ``Partitioner`` performs necessary processing steps to parition heterogenous data objects with correct distribution and indexing.

.. figure:: ../_figures/DGL_metis.png
  :align: center
  :width: 50%
  :alt: Example of graph partitioning with METIS algorithm.

**Figure:** Generate graph partitions with HALO vertices (the vertices with different colors from majority of the vertices in the partition). Source: `DistDGL paper. <https://arxiv.org/pdf/2010.05337.pdf>`_

Provided example script `partition_graph.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/distributed/pyg/partition_graph.py>`_ demonstrates the partitioning for homogenous ``ogbn-products``, ``Reddit`` , and heterogenous: ``ogbn-mag``, ``Movielens`` datasets.
The ``Partitioner`` can also process temporal attributes of the nodes which is presented in the ``Movielens`` dataset partitioning.
** Important note: **
As result of METIS is non-deterministic, the resulting partitions differ between iterations. To perform training, make sure that each node has an access to the same data partition. Use a shared drive or remote storage, i.e. a docker volume or manually copy the dataset to each node of the cluster!

The result of partitioning, for a two-part split of homogenous ``ogbn-products`` is as follows:

#. ogbn-products-labels:
    * label.pt:   target node/edge labels
#. ogbn-products-partitions:
    * edge_map.pt:   mapping (partition book) between edge_id and partition_id
    * node_map.pt:   mapping (partition book) between node_id and partition_id
    * META.json:  {"num_parts": 2, "is_hetero": false, "node_types": null, "edge_types": null, "is_sorted": true}
    * part0:      partition 0
        * graph.pt:     graph topo
        * node_feats.pt:   node features
        * edge_feats.pt:   edge features (if present)
    * part1:      partition 1
        * ...
#. ogbn-products-train-partitions:
    * partion0.pt:  training node indices for partition0
    * partion1.pt:  training node indices for partition1
#. ogbn-products-test-partitions:
    * partion0.pt:  test node indices for partition0
    * partion0.pt:  test node indices for partition1


In distributed training, each node in the cluster holds a partition of the graph. Before the training starts, we will need partition the graph dataset into multiple partitions, each of which corresponds to a specific training node.

2. LocalGraphStore and LocalFeatureStore
-------------------------------------------

.. figure:: ../_static/thumbnails/distribute_graph_feature_store.png
  :align: center
  :width: 90%

2.1 LocalGraphStore
~~~~~~~~~~~~~~~~~~~~~

:class:`torch_geometric.distributed.LocalGraphStore` is a class designed to act as a local graph store for distributed training. It implements the :class:`torch_geometric.data.GraphStore` interface, providing features for efficient management of partition-related information and support for both homogeneous and heterogeneous :pyg:`PyG` graphs.

Key Features
~~~~~~~~~~~~~~~

1. **Partition Edge Index Storage:**

* Stores information about local graph connections within partition.

2. **Global Node and Edge Identifiers:**

* Maintains global identifiers for nodes and edges, allowing for consistent mapping across partitions.

3. **Homogeneous and Heterogeneous Graph Support:**

* Supports both homogeneous and heterogeneous :pyg:`PyG` graphs.

4. **Edge Attribute Storage:**

* Stores edge attributes and global identifiers.

5. **Initialization Methods:**

* Provides convenient methods for initializing the graph store from data or partition.

Initialization and Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Instances of :class:`torch_geometric.distributed.LocalGraphStore` can be created using the provided initialization methods.

- Edge indices, edge attributes, edge ids and other relevant information can be added or retrieved using the provided methods.

Example Usage
~~~~~~~~~~~~~~~~~

Below is an example of creating an instance of :class:`torch_geometric.distributed.LocalGraphStore` and using it for distributed training:

.. code-block:: python

    import torch
    from torch_geometric.distributed import LocalGraphStore

    # Create an instance of LocalGraphStore
    graph_store = LocalGraphStore()

    edge_id = torch.tensor([0, 1, 2, 3])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    # Access or modify attributes
    graph_store.num_partitions = 2
    graph_store.partition_idx = 1

    # Store edge information
    graph_store.put_edge_index(
        edge_index,
        edge_type=None,
        layout='coo',
        size=(100, 100),
    )
    graph_store.put_edge_id(
        edge_id,
        edge_type=None,
        layout='coo',
        size=(100, 100),
    )

    # Retrieve edge information
    edge_attr = graph_store.get_all_edge_attrs()[0]
    retrieved_edge_index = graph_store.get_edge_index(edge_attr)
    retrieved_edge_id = graph_store.get_edge_id(edge_attr)

    # ...

    # Remove edge information
    graph_store.remove_edge_index(edge_attr)
    graph_store.remove_edge_id(edge_attr)

    # ...


Initialization from Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`torch_geometric.distributed.LocalGraphStore` provides class methods for creating instances from homogeneous and heterogeneous graph data:

* :func:`torch_geometric.distributed.LocalGraphStore.from_data`: Creates a local graph store from homogeneous data.

.. code-block:: python

    import torch
    from torch_geometric.distributed import LocalGraphStore

    # Example data for homogeneous graph:
    edge_id = torch.tensor([0, 1, 2, 3])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = 3

    # Create a LocalGraphStore from homogeneous data:
    graph_store = LocalGraphStore.from_data(edge_id, edge_index, num_nodes)


* :func:`torch_geometric.distributed.LocalGraphStore.from_hetero_data`: Creates a local graph store from heterogeneous data.

.. code-block:: python

    import torch
    from torch_geometric.distributed import LocalGraphStore

    # Example data for heterogeneous graph:
    edge_id_dict = {
        ('v0', 'e0', 'v1'): torch.tensor([0, 1, 2, 3]),
    }
    edge_index_dict = {
        ('v0', 'e0', 'v1'): torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
    }
    num_nodes_dict = {'v0': 2, 'v1': 2}

    # Create a LocalGraphStore from heterogeneous data:
    graph_store = LocalGraphStore.from_hetero_data(edge_id_dict, edge_index_dict, num_nodes_dict)


2.2 LocalFeatureStore
~~~~~~~~~~~~~~~~~~~~~~~~

:class:`torch_geometric.distributed.LocalFeatureStore` is a class that implements the :class:`~torch_geometric.data.FeatureStore` interface. It serves as a local feature store for distributed training in Graph Neural Networks (GNNs). The local feature store is responsible for managing and distributing node and edge features across different partitions and machines during the training process.

Key Features
~~~~~~~~~~~~~~~~

1. **Node and Edge Feature Storage:**

* It extends the :class:`~torch_geometric.data.FeatureStore` class and provides functionalities for storing, retrieving, and distributing node and edge features. Features are stored locally for the nodes or edges within the partition managed by each machine or device.

2. **Global Node and Edge Identifiers:**

* Maintains global identifiers for nodes and edges, allowing for consistent mapping across partitions.

3. **Homogeneous and Heterogeneous Graph Support:**

* Supports both homogeneous and heterogeneous :pyg:`PyG` graphs.

4. **Remote Feature Lookup:**

* Implements mechanisms for looking up features in both local and remote nodes during distributed training.

5. **Initialization Methods:**

* Provides convenient methods for initializing the graph store from data or partition.

Initialization and Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Instances of :class:`torch_geometric.distributed.LocalFeatureStore` can be created using the provided initialization methods.

* Features, global identifiers, and other relevant information can be added or retrieved using the provided methods.

* The class is designed to work seamlessly in distributed training scenarios, allowing for efficient feature handling across partitions.

Example Usage
~~~~~~~~~~~~~~~~~

Below is an example of creating an instance of :class:`torch_geometric.distributed.LocalFeatureStore` and using it for distributed training:

.. code-block:: python

    import torch
    from torch_geometric.distributed import LocalFeatureStore
    from torch_geometric.distributed.event_loop import to_asyncio_future

    async def get_node_features():
        # Create a LocalFeatureStore instance:
        feature_store = LocalFeatureStore()

        # Add global node identifiers and node features:
        node_ids = torch.tensor([0, 1, 2])
        node_features = torch.randn((3, 64))  # Assuming 64-dimensional node features
        feature_store.put_global_id(node_ids, group_name=None)
        feature_store.put_tensor(node_features, group_name=None, attr_name='x')

        feature_store.num_partitions = 2
        feature_store.node_feat_pb = torch.tensor([0, 0, 1])
        feature_store.meta = {'is_hetero': False}

        # Retrieve node features for a specific node ID:
        node_id_to_lookup = torch.tensor([1])
        future = feature_store.lookup_features(node_id_to_lookup)

        nfeat = await to_asyncio_future(future)

        return nfeat

    # Use the retrieved features in the GNN training process
    # ...


Initialization from Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`torch_geometric.distributed.LocalFeatureStore` provides class methods for creating instances from homogeneous and heterogeneous graph data:

* :func:`torch_geometric.distributed.LocalFeatureStore.from_data`: Creates a local feature store from homogeneous data.

.. code-block:: python

    import torch
    from torch_geometric.distributed import LocalFeatureStore

    # Example data for homogeneous graph:
    node_id = torch.tensor([0, 1, 2])
    x = torch.rand((3, 4))
    y = torch.tensor([1, 0, 1])
    edge_id = torch.tensor([0, 1, 2])
    edge_attr = torch.rand((3, 5))

    # Create a LocalFeatureStore from homogeneous data:
    feature_store = LocalFeatureStore.from_data(
        node_id=node_id,
        x=x,
        y=y,
        edge_id=edge_id,
        edge_attr=edge_attr
    )

* :func:`torch_geometric.distributed.LocalFeatureStore.from_hetero_data`: Creates a local feature store from heterogeneous data.

.. code-block:: python

    import torch
    from torch_geometric.distributed import LocalFeatureStore

    # Example data for heterogeneous graph:
    node_id_dict = {
        'v0': torch.tensor([0, 1]),
        'v1': torch.tensor([2, 3, 4]),
    }

    x_dict = {
        'v0': torch.rand((2, 4)),
        'v1': torch.rand((3, 4)),
    }

    y_dict = {
        'v0': torch.tensor([1, 0]),
        'v1': torch.tensor([1, 0, 1]),
    }

    edge_id_dict = {
        ('v0', 'e0', 'v1'): torch.tensor([0, 1, 2]),
    }

    edge_attr_dict = {
        ('v0', 'e0', 'v1'): torch.rand((3, 5)),
    }

    # Create a LocalFeatureStore from heterogeneous data:
    feature_store = LocalFeatureStore.from_hetero_data(
        node_id_dict=node_id_dict,
        x_dict=x_dict,
        y_dict=y_dict,
        edge_id_dict=edge_id_dict,
        edge_attr_dict=edge_attr_dict
    )

Initialization of LocalFeatureStore and LocalGraphStore from Partition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`torch_geometric.distributed.LocalFeatureStore` and :class:`torch_geometric.distributed.LocalGraphStore` provide a class methods for creating instances from a specified partition:

* :func:`torch_geometric.distributed.from_partition`: Creates a local feature store / local graph store from a partition.

.. code-block:: python

    # Load partition into graph:
    graph_store = LocalGraphStore.from_partition(
        osp.join(root_dir, f'{dataset_name}-partitions'), node_rank)

    # Load partition into feature:
    feature_store = LocalFeatureStore.from_partition(
        osp.join(root_dir, f'{dataset_name}-partitions'), node_rank)

    # Load partition information:
     (
         meta,
         num_partitions,
         partition_idx,
         node_pb,
         edge_pb,
     ) = load_partition_info(osp.join(root_dir, f'{dataset}-partitions'),
                             node_rank)

    # Setup the partition information in graph store:
    graph_store.num_partitions = num_partitions
    graph_store.partition_idx = partition_idx
    graph_store.node_pb = node_pb
    graph_store.edge_pb = edge_pb
    graph_store.meta = meta

    # Setup the partition information in feature store:
    feature_store.num_partitions = num_partitions
    feature_store.partition_idx = partition_idx
    feature_store.node_feat_pb = node_pb
    feature_store.edge_feat_pb = edge_pb
    feature_store.feature_pb = node_pb
    feature_store.meta = meta

    # Load the label file and put into graph as labels:
    if node_label_file is not None:
        if isinstance(node_label_file, dict):
            whole_node_labels = {}
            for ntype, file in node_label_file.items():
                whole_node_labels[ntype] = torch.load(file)
        else:
            whole_node_labels = torch.load(node_label_file)
    node_labels = whole_node_labels
    graph_store.labels = node_labels

    partition_data = (feature_store, graph_store)


1. Setting up communication using DDP & RPC
---------------------------------------------------

In this distributed training implementation two `torch.distributed` communication technologies are used:

* ``torch.distributed.ddp`` for data parallel model training
* ``torch.distributed.rpc`` for remote sampling calls & feature retrieval from distributed database

In this context, we opted for ``torch.distributed.rpc`` over alternatives such as gRPC because PyTorch RPC inherently comprehends tensor-type data. Unlike some other RPC methods like gRPC, which require the serialization or digitization of JSON or other user data into tensor types, using this method helps avoid additional serialization/digitization overhead during loss backward for gradient communication.

The DDP group is initialzied in a standard way in the main training script.

.. code-block:: python

    # Initialize DDP training process group.
    torch.distributed.init_process_group(
        backend='gloo', rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method='tcp://{}:{}'.format(master_addr, ddp_port))
**Note:** For CPU-based sampling the recommended backed is `gloo`.

The RPC group initialization is more complicated as it needs to happen in each sampler process. This can be done my modifying ``worker_init_fn`` that is called at initialization of worker processes by torch base class ``_MultiProcessingDataLoaderIter``. We provide a customized init function:

.. code-block:: python

    def worker_init_fn(self, worker_id: int):
        try:
            num_sampler_proc = self.num_workers if self.num_workers > 0 else 1
            self.current_ctx_worker = DistContext(
                world_size=self.current_ctx.world_size * num_sampler_proc,
                rank=self.current_ctx.rank * num_sampler_proc + worker_id,
                global_world_size=self.current_ctx.world_size *
                num_sampler_proc,
                global_rank=self.current_ctx.rank * num_sampler_proc +
                worker_id,
                group_name='mp_sampling_worker',
            )

            init_rpc(
                current_ctx=self.current_ctx_worker,
                master_addr=self.master_addr,
                master_port=self.master_port,
                num_rpc_threads=self.num_rpc_threads,
                rpc_timeout=self.rpc_timeout,
            )
            logging.info(
                f"RPC initiated in worker-{worker_id} "
                f"(current_ctx_worker={self.current_ctx_worker.worker_name})")
            self.dist_sampler.init_sampler_instance()
            self.dist_sampler.register_sampler_rpc()
            global_barrier(timeout=10)  # Wait for all workers to initialize.

            # close RPC & worker group at exit:
            atexit.register(shutdown_rpc, self.current_ctx_worker.worker_name)

        except RuntimeError:
            raise RuntimeError(f"`{self}.init_fn()` could not initialize the "
                               f"worker loop of the neighbor sampler")

This functions first sets a unique ``DistContext`` for each worker and assigns its group and rank, subsequently it initializes a standard PyG ``NeighborSampler`` that provides basic functionality also for distributed data processing, and finally registers a new RPC worker within worker's sub-process.

4. Distributed Loader
------------------------------------

.. figure:: ../_static/thumbnails/distribute_neighborloader.png
  :align: center
  :width: 90%

Distributed class ``DistNeighborLoader`` is used to provide batch-sized data for distributed trainer. This class will have the input of data partition, num_neighbors, train_idx, batch_size, shuffle flag, device, number of sampler workers, master addr/port for ddp, context and rpc_worker_names, etc.

As the DistNeighborLoader architecture shown above there are the separate processes for sampler and trainer.

+ **Main process**:   cover the loading of data partition, distloader and model training, etc
+ **Sampler process**: cover the distNeighborSampler and message queue like here we used ``torch.mp.queue`` to send the sampler message from one process to another.

The working flow is from load partition into graphstore/featurestore, distNeighborSampler with local and remote sampling,  sampled nodes/features to be formed into PyG data for dataloader and finally into trainer for training.

.. figure:: ../_static/thumbnails/distribute_distloader.png
  :align: center
  :width: 90%

Distributed class ``DistLoader`` is used to create distributed data loading routines like initializing the parameters of current_ctx, rpc_worker_names, master_addr/port, channel, num_rpc_threads, num_workers, etc and then at the same time will initialize the context/rpc for distributed sampling based on ``worker_init_fn``.

Distributed class ``NodeLoader`` is used to do the distributed node sampling and feature collection from local/remotely based on the function of ``collate_fn`` and ``filter_fn`` in ``NodeLoader`` and finally formed sampled results into PyG data for dataloader output.


There are several key features for ``DistNeighborLoader`` and  ``DistLoader``:

+ ``DistNeighborLoader`` inherits all basic functionality from PyG Loaders and rely on PyTorch multiprocessing backend with modified ``_worker_loop`` arguments.
+ Modified args passed to the ``worker_init_fn`` control process initialization and closing behaviors, i.e. establish RPC and close it at process exit.
+ Each loader handles a number (num_workers) of spawned sampler subprocesses that exchange data through RPC.
+ RPC requests can be executed in synchronous or asynchronous manner with asyncio module.
+ ``DistLoader`` consumes input in custom format (``LocalFeatureStore``, ``LocalGraphStore``) and outputs standard Data\HeteroData object.
+ The same principles apply to ``DistLinkNeighborLoader``

5. Distributed Sampling
------------------------------------

:class:`torch_geometric.distributed.DistNeighborSampler` is a module designed for efficient distributed training of Graph Neural Networks. It addresses the challenges of sampling neighbors in a distributed environment, where graph data is partitioned across multiple machines or devices. The sampler ensures that GNNs can effectively learn from large-scale graphs, maintaining scalability and performance.

5.1 Asynchronous Neighbor Sampling and Feature Collection:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Asynchronous neighbor sampling: Asynchronous sampling is implemented using asynchronous ``torch.distributed.RPC`` calls. It allows machines to independently sample neighbors without strict synchronization. Each machine autonomously selects neighbors from its local graph partition, without waiting for others to complete their sampling processes. This approach enhances parallelism, as machines can progress asynchronously leading to faster training. In addition to asynchronous sampling, Distributed Neighbor Sampler also provides asynchronous feature collection.

5.2 Customizable Sampling Strategies:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can customize neighbor sampling strategies based on their specific requirements. The module provides flexibility in defining sampling techniques, such as:

* Node sampling
* Edge sampling
* Disjoint sampling
* Node-based temporal sampling
* Edge-based temporal sampling

Additionally, each of these methods is supported for both homogeneous and heterogeneous graph sampling.

5.3 Distributed Neighbor Sampling Workflow Key Steps:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Distributed node sampling: Utilizing the training seeds provided by the loader, the neighbor sampling procedure is executed. These training seeds may originate from either local or remote partitions. For nodes within a local partition, the neighbor sampling occurs on the local machine. Conversely, for nodes associated with a remote partition, the neighbor sampling is conducted on the machine responsible for storing the respective partition.

2) Distributed feature lookup: Each partition stores the features of its nodes and edges. Consequently, if the output of a sampler on a specific machine includes sampled nodes or edges that do not pertain to its partition, the machine must initiate an RPC request to the machine to which these nodes (or edges) belong in order to retrieve information about their features.

3) Form into PyG data format: Based on the sampler output and the acquired node (or edge) features, a Data/HeteroData object is created. This object forms a batch used in subsequent computational operations of the model. Note that this step occurs within the loader.

5.4 Algorithm Overview:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section outlines the Distributed Neighbor Sampling Algorithm. The algorithm focuses on efficiently sampling neighbors across distributed nodes to facilitate effective learning on large-scale graph-structured data.

.. figure:: ../_static/thumbnails/distribute_neighborsampler.png
  :align: center
  :width: 90%

While the mechanism is analogous, the distributed sampling process diverges from single-machine sampling. In distributed training, seed nodes can belong to different partitions, leading to simultaneous sampling on multiple machines for a single batch. Consequently, synchronization of sampling results across machines is necessary to obtain seed nodes for the subsequent layer, requiring modifications to the basic algorithm.

The accompanying image illustrates a graph divided into two partitions, each associated with a distinct machine. For nodes `[0, 1, 5, 6]` in the batch, the objective is to sample all neighbors within a single layer. The process unfolds as follows:

1) In the initial step, the algorithm checks whether the seed nodes belong to the local partition. If affirmative, sampling is executed on the local machine.

2) If the seed nodes belong to a remote partition, an RPC request is dispatched from the local machine to the remote machine to initiate sampling.

3) Upon completion of the neighbor sampling process, results from remote machines are transmitted to the local machine, where they are merged and arranged based on the sampling order (seed nodes first, followed by sampled neighbors in the order of individual seed node sampling). The final step involves removing duplicate nodes.

4)
  * If all layers have been sampled, as is the case in this example, the features of the sampled nodes (or edges in the case of edge sampling) are obtained, and the results are passed to the message channel.

  * If not, new input nodes for the next layer are acquired. In the context of the image example, these nodes would be `[2, 4, 3, 10, 7]`, and the entire process starts from the beginning.

5.5. Distributed Neighbor Sampler Code Structure:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section provides an overview of the key code structure elements of the Distributed Neighbor Sampler.

* :func:`torch_geomeric.distribued.DistNeighborSampler.node_sample`:

  * :func:`torch_geomeric.distribued.DistNeighborSampler.node_sample`, is responsible for performing layer-by-layer distributed sampling from either a :class:`torch_geomeric.sampler.NodeSamplerInput` or :class:`torch_geomeric.distributed.utils.DistEdgeHeteroSamplerInput` object.

  * It supports both homogeneous and heterogeneous graphs, adapting its behavior accordingly.

  * The sampling procedure takes into account temporal aspects.

  * Following the sampling of a single layer, the :func:`torch_geometric.distributed.utils.remove_duplicates` function is utilized to remove duplicates among the sampled nodes in the result.

  * Upon completion of the sampling process, the :func:`torch.ops.pyg.relabel_neighborhood` (or in the case of hetero graphs: :func:`torch.ops.pyg.hetero_relabel_neighborhood`) function is employed to perform mappings from global to local node indices.

  * The output of the sampling procedure is returned, encapsulated in either a :class:`torch_geomeric.sampler.SamplerOutput` or :class:`torch_geomeric.sampler.HeteroSamplerOutput` object.

.. code-block:: python

    async def node_sample(
        self,
        inputs: Union[NodeSamplerInput, DistEdgeHeteroSamplerInput],
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        # ...
        # Loop over the layers and perform sampling:
        for i, one_hop_num in enumerate(self.num_neighbors):
            out = await self.sample_one_hop(src, one_hop_num,
                                                src_seed_time, src_batch)
            # Remove duplicates:
            src, node, src_batch, batch = remove_duplicates(
                out, node, batch, self.disjoint)
        # ...
        row, col = torch.ops.pyg.relabel_neighborhood(
                seed,
                torch.cat(node_with_dupl),
                sampled_nbrs_per_node,
                self._sampler.num_nodes,
                torch.cat(batch_with_dupl) if self.disjoint else None,
                self.csc,
                self.disjoint,
        )

        sampler_output = SamplerOutput(
                node=node,
                row=row,
                col=col,
                edge=torch.cat(edge),
                batch=batch if self.disjoint else None,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
                metadata=metadata,
        )
        return sampler_output

* :func:`torch_geometric.distributed.DistNeighborSampler.sample_one_hop`:

  * This function is designed to sample one-hop neighbors for a given set of source nodes (:obj:`srcs`).

  * Using the input data, which consists of the indices of the source nodes :obj:`srcs` and their node type :obj:`src_node_type`, the assignment of these nodes to specific partitions is determined by invoking the :func:`torch_geometric.distributed.LocalGraphStore.get_partition_ids_from_nids` function.

  * Based on the :obj:`partition_ids` values produced by :func:`torch_geometric.distributed.LocalGraphStore.get_partition_ids_from_nids` it handles scenarios where the source nodes may be located on either local or remote partitions and executes the sampling accordingly using :func:`torch_geomeric.distributed.DistNeighborSampler._sample_one_hop` function.

  * In scenarios where nodes are associated with a local partition, sampling occurs on the local machine. Conversely, if the nodes belong to a remote partition, the local machine, utilizing ``torch.disributed.RPC``, sends a request to the remote machine for conducting sampling. The outcome of this sampling procedure is stored in the `torch.Futures` object.

  * The results from local and remote machines are merged in a :func:`torch_geometric.distributed.DistNeighborSampler.merge_sampler_outputs` to provide a comprehensive output.

.. code-block:: python

    async def sample_one_hop(
        self,
        srcs: Tensor,
        one_hop_num: int,
        seed_time: Optional[Tensor] = None,
        src_batch: Optional[Tensor] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> SamplerOutput:

        # ...
        partition_ids = self.graph_store.get_partition_ids_from_nids(
            srcs, src_node_type)
        # ...
        for i in range(self.graph_store.num_partitions):
            p_id = (self.graph_store.partition_idx +
                    i) % self.graph_store.num_partitions
            p_mask = partition_ids == p_id
            p_srcs = torch.masked_select(srcs, p_mask)
            # ...
            if p_srcs.shape[0] > 0:
                if p_id == self.graph_store.partition_idx:
                    # Sample for one hop on a local machine:
                    p_nbr_out = self._sample_one_hop(p_srcs, one_hop_num,
                                                     p_seed_time, edge_type)
                    p_outputs.pop(p_id)
                    p_outputs.insert(p_id, p_nbr_out)

                else:  # Sample on a remote machine:
                    local_only = False
                    to_worker = self.rpc_router.get_to_worker(p_id)
                    futs.append(
                        rpc_async(
                            to_worker,
                            self.rpc_sample_callee_id,
                            args=(p_srcs, one_hop_num, p_seed_time, edge_type),
                        ))
        # ...
        return self.merge_sampler_outputs(partition_ids, partition_orders,
                                          p_outputs, one_hop_num, src_batch)

* :func:`torch_geometric.distributed.DistNeighborSampler._sample_one_hop`

  * The primary objective of this function is to invoke the :pyg:`PyG` native neighbor sampling function :func:`torch.ops.pyg.neighbor_sample`, using a :func:`torch.ops.pyg.dist_neighbor_sample` wrapper specifically tailored for distributed behavior.

  * The function is designed to perform one-hop neighbor sampling.

  * The function produces a :class:`torch_geomeric.sampler.SamplerOutput`` as its output, encapsulating three key pieces of information: the identifiers of the sampled nodes (:obj:`node`), the identifiers of the sampled edges (:obj:`edge`), and the cumulative sum of neighbors per node (:obj:`cumsum_neighbors_per_node`). :obj:`cumsum_neighbors_per_node` stores information about the cumulated sum of the sampled neighbors by each sorce node, that is further needed to relabel global nodes indices into local within a subgraph. This argument is specific for distributed training.

.. code-block:: python

    def _sample_one_hop(
        self,
        input_nodes: Tensor,
        num_neighbors: int,
        seed_time: Optional[Tensor] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> SamplerOutput:
        # ...
        out = torch.ops.pyg.dist_neighbor_sample(
            colptr,
            row,
            input_nodes.to(colptr.dtype),
            num_neighbors,
            node_time,
            None,  # edge_time
            seed_time,
            None,  # TODO: edge_weight
            True,  # csc
            self.replace,
            self.subgraph_type != SubgraphType.induced,
            self.disjoint and node_time is not None,
            self.temporal_strategy,
        )
        node, edge, cumsum_neighbors_per_node = out

        # ...
        return SamplerOutput(
            node=node,
            row=None,
            col=None,
            edge=edge,
            batch=None,
            metadata=(cumsum_neighbors_per_node, ),
        )

5.6. Edge Sampling
~~~~~~~~~~~~~~~~~~~~~~~

* Edge sampling in the context of distributed training closely mirrors the methodology employed on a single machine. This process is facilitated by invoking the :func:`torch_geometric.distributed.edge_sample` function, a mechanism designed for distributed asynchronous sampling from an edge sampler input. Similarly to the single machine case, the :func:`torch_geometric.distributed.edge_sample` function invokes the :func:`torch_geometric.distributed.node_sample` function (but from the distributed package).

* The :class:`torch_geometric.distributed.utils.DistEdgeHeteroSamplerInput` class has been designed to hold the input parameters required for the distributed heterogeneous link sampling process within the :func:`torch_geometric.distributed.DistNeighborSampler.node_sample` method. This scenario specifically applies when dealing with edges where the source and target node types are distinct. In other cases, the :class:`torch_geomeric.sampler.NodeSamplerInput` objetc is used as input to the :func:`torch_geometric.distributed.DistNeighborSampler.node_sample` function.

.. code-block:: python

        async def edge_sample(
        self,
        inputs: EdgeSamplerInput,
        sample_fn: Callable,
        num_nodes: Union[int, Dict[NodeType, int]],
        disjoint: bool,
        node_time: Optional[Union[Tensor, Dict[str, Tensor]]] = None,
        neg_sampling: Optional[NegativeSampling] = None,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        # ...

        # Heterogeneus Neighborhood Sampling ##################################

        if input_type is not None:
            if input_type[0] != input_type[-1]:  # Two distinct node types:

            # ...
                out = await sample_fn(
                    DistEdgeHeteroSamplerInput(
                        input_id=inputs.input_id,
                        node_dict=seed_dict,
                        time_dict=seed_time_dict,
                        input_type=input_type,
                    ))

            else:
                # Only a single node type: Merge both source and destination.
                # ...

                out = await sample_fn(
                    NodeSamplerInput(
                        input_id=inputs.input_id,
                        node=seed,
                        time=seed_time,
                        input_type=input_type[0],  # csc
                    ))
        # ...

        # Homogeneus Neighborhood Sampling ####################################

        else:
        # ...

            out = await sample_fn(
                NodeSamplerInput(
                    input_id=inputs.input_id,
                    node=seed,
                    time=seed_time,
                    input_type=None,
                ))


1. Installation & Run for Homo/Hetero Example
---------------------------------------------

7.1 Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Requirement:

- latest PyG
- environment
        (1) Password-less ssh needs to be set up on all the nodes that you are using.

        (2) A network file system (NFS) is set up for all the nodes to access.

        (3) To perform distributed sampling, files and codes need to be accessed across multiple machines. A distributed file system (i.e., NFS, SSHFS, Ceph, ...) is required to allow you for synchnonizing files such as partition information.


7.2 Run for Homo Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1) Prepare and partition the data

In distributed training, each node in the cluster holds a partition of the graph. Before the training starts, we partition the ``ogbn-products`` dataset into multiple partitions, each of which corresponds to a specific training node.

Here, we use ``ogbn-products`` and partition it into two partitions (in default) by the `[partition example] <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/distributed/pyg/partition_graph.py>`__

.. code-block:: python

    python partition_graph.py --dataset=ogbn-products --root_dir=./data/products --num_partitions=2

The generated partition will have the folder below.

.. figure:: ../_static/thumbnails/distribute_homo_partition.png
  :align: center
  :width: 40%

You can put/move the products partition folder into one public folder that each node can access this shared folder.



2) Run the example in each training node

For example, running the example in two nodes:

.. code-block:: python

    # Node 0:
    python dist_train_sage_for_homo.py \
      --dataset_root_dir=your partition folder \
      --num_nodes=2 --node_rank=0 --num_training_procs=1 \
      --master_addr= master ip

    # Node 1:
    python dist_train_sage_for_homo.py \
      --dataset_root_dir=your partition folder \
      --num_nodes=2 --node_rank=1 --num_training_procs=1 \
      --master_addr= master ip


**Notes:**

1. You should change the `master_addr` to the IP of `node#0`.
2. In default this example will use the num_workers = 2 for number of sampling workers and concurrency=2 for mp.queue. you can also add these argument to speed up the training like "--num_workers=8 --concurrency=8"
3. All nodes need to use the same partitioned data when running `dist_train_sage_for_homo.py`.


7.3 Run for Hetero Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Prepare and partition the data


Here, we use ``ogbn-mags`` and partition it into two partitions (in default) by the [`partition example <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/distributed/pyg/partition_hetero_graph.py>`__] :

.. code-block:: python

    python partition_hetero_graph.py --dataset=ogbn-mag --root_dir=./data/mag --num_partitions=2

The generated partition will have the folder below.

.. figure:: ../_static/thumbnails/distribute_hetero_partition.png
  :align: center
  :width: 40%


You can put/move the products partition folder into one public folder that each node can access this shared folder.


2) Run the example in each training node

For example, running the example in two nodes:

.. code-block:: python

    # Node 0:
    python dist_train_sage_for_hetero.py \
      --dataset_root_dir=your partition folder \
      --dataset=ogbn-mags \
      --num_nodes=2 --node_rank=0 --num_training_procs=1 \
      --master_addr= master ip

    # Node 1:
    python dist_train_sage_for_hetero.py \
      --dataset_root_dir=your partition folder \
      --dataset=ogbn-mags \
      --num_nodes=2 --node_rank=1 --num_training_procs=1 \
      --master_addr= master ip



8. Run with Launch.py
------------------------------------

As you can see the run in previous paragraph we need run the script in separate nodes which is not easy for the case of big partition numbers. So in this chapter we will use one script to run just in one node for multiple partitions.

The requirement for this single-script run is that you still need multiple nodes with NFS supported & ssh with password-less.

In the followings we will show the files to run with single-scripts.

1) **ip_config.yaml**

There are the 2 ip and 2 ports list for 2 partitions inside this file as example below.

+ x.x.x.10 1234
+ x.x.x.12 1234

The node with first IP address will be the host node to run with launch.py as below.


2) **launch.py**

In the launch.py you need setup the parameters as below

+ workspace
+ parameters used in e2e example
+ part_config:  "partition config"
+ ip_config:  "ip_config.yaml"
+ remote cmd & "e2e_xxx.py" in remote nodes



.. code-block:: python

    python launch.py --workspace ./distributed_pyg/pytorch_geometric --num_nodes 2 --num_neighbors 15,10,5 --num_training_procs 1 --dataset_root_dir ./partition_ds/products --dataset ogbn-product --epochs 20 --batch_size 1024 --num_workers 2 --concurrency 2 --part_config ./partition_ds/products/ogbn-products-partitions/META.json --ip_config ./distributed_pyg/pytorch_geometric/ip_config.yaml 'cd /home/userXXX; source anaconda3/envs/PyGDistributed/bin/activate; cd /home/userXXX/distributed_pyg/pytorch_geometric; /home/userXXX/anaconda3/envs/PyGDistributed/bin/python /home/userXXX/distributed_pyg/pytorch_geometric/e2e_homo.py'


3) **run_dist.sh**

You also create one .sh file to run this distributed script with all parameters inside of this .sh file and if you need run another setting you just need change a little settting in this .sh file.

The below .sh example is assume that you have the anaconda virtual environment in all nodes.

.. code-block:: python

    #!/bin/bash

    CONDA_ENV=/home/userXXX/anaconda3/envs/PyGDistributed
    PYG_WORKSPACE=$PWD    #/home/userXXX/distributed_pyg/pytorch_geometric
    PY_EXEC=${CONDA_ENV}/bin/python
    EXEC_SCRIPT=${PYG_WORKSPACE}/e2e_homo.py

    # node number
    NUM_NODES=2

    # dataset folder
    DATASET_ROOT_DIR="/home/userXXX/partition_ds/products"

    # process number for training
    NUM_TRAINING_PROCS=1

    # dataset name
    DATASET=ogbn-product

    # num epochs to run for
    EPOCHS=20

    BATCH_SIZE=1024

    # number of workers for sampling
    NUM_WORKERS=2
    CONCURRENCY=2

    #partition data directory
    PART_CONFIG="/home/userXXX/partition_ds/products/ogbn-products-partitions/META.json"
    NUMPART=2

    # fanout per layer
    NUM_NEIGHBORS="15,10,5"

    #ip_config path
    IP_CONFIG=${PYG_WORKSPACE}/ip_config.yaml


    # Folder and filename where you want your logs.
    logdir="logs"
    mkdir -p "logs"
    logname=log_${DATASET}_${NUMPART}_$RANDOM
    echo $logname
    set -x

    # stdout stored in /logdir/logname.out
    python launch.py --workspace ${PYG_WORKSPACE} --num_nodes ${NUM_NODES} --num_neighbors ${NUM_NEIGHBORS} --num_training_procs ${NUM_TRAINING_PROCS} --dataset_root_dir ${DATASET_ROOT_DIR} --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --concurrency ${CONCURRENCY} --part_config ${PART_CONFIG} --ip_config ${IP_CONFIG} "cd /home/userXXX; source anaconda3/envs/PyGDistributed/bin/activate; cd ${PYG_WORKSPACE}; ${PY_EXEC} ${EXEC_SCRIPT}" |& tee ${logdir}/${logname}.txt
    set +x
