Scaling Up GNNs with Remote Backend Support
============================================

PyG (2.2 and beyond) includes numerous primitives to easily integrate with simple paradigms for scalable graph machine learning, enabling users to train GNNs on graphs far larger than the size of their machine's available memory.
It does so by introducing simple, easy-to-use, and extensible abstractions of a feature store and graph store that plug directly into existing familiar PyG interfaces.
Defining a feature store allows users to leverage node (and soon, edge) features stored remotely, and defining a graph store allows users to leverage graph structure information stored remotely.
Together, they allow for powerful GNN scalability with low developer friction.

.. warning::

    The remote backend APIs discussed here may change in the future as we continuously work to improve their ease-of-use and generalizability.

.. warning::

    Currently, the feature store and graph store only support *heterogeneous* graphs, and do not support edge features. Homogeneous graph and edge feature support is coming soon.


Background
----------


An instantiated graph neural network consists of two types of data:

- Node and/or edge features: dense vectors corresponding to attributes of the nodes and edges in a graph
- Graph layout information: the nodes in the graph and the edges that connect them

Message passing performs scatter/gather operations on feature tensors between node space and edge space in the defined graph.
An immediate observation is that scaling to data larger than the available memory of a chosen accelerator requires training on sampled subgraphs (which form minibatches), instead of the full graph at once.
While this method adds stochasticity to the learning process, it reduces the memory requirements of the accelerator to those of the sampled subgraphs.

.. figure:: ../_figures/remote_1.png
  :align: center
  :width: 100%

  **Figure 1:** The classical minibatch GNN training paradigm.

However, while minibatch training reduces the memory requirements of the chosen accelerator, it is not a silver bullet for all graph learning scalability problems.
In particular, since one must sample subgraphs to pass to the accelerator at each iteration of the learning process, the graph and features are traditionally required to be stored in the CPU DRAM of a user's machine.
At large scale, this requirement can become quite burdensome:

- Acquiring instances with enough CPU DRAM to store a graph and features is challenging
- Training with data parallelism requires replicating the graph and features in each compute node, which is burdensome
- Graphs and features can easily be much larger than the memory of a single machine, which imposes a hard limit in the current setup

Scalability to very large graphs and features beyond the memory requirements of a single machine thus requires moving these data structures out-of-core and only processing sampled subgraphs on a node that performs computation.
In order to achieve this goal, PyG relies on two primary abstractions to store feature data and graph layout data.
Features are stored in a key-value **feature store**, which must support efficient random access.
Graph information is stored in a **graph store**, which must support efficient sampling for the samplers defined on the graph store.

.. figure:: ../_figures/remote_2.png
  :align: center
  :width: 100%

  **Figure 2:** Graph data storage layout between remote storage and a training instance.

In PyG 2.2 and beyond, the separation of graph data into its features and layout information, storage of this information in locations potentially remote to the actual training node, and interaction between these components are all completely abstracted from the end user.
So long as the feature store and graph store are defined appropriately (keeping in mind the aforementioned performance requirements), PyG handles the rest!



Feature Store
--------------

A feature store holds features for the nodes and edges of a graph.
Feature storage is often the primary storage bottleneck in graph learning applications, as storing a graph's layout information (the :obj:`edge_index`) is relatively cheap (~24 bytes per edge).
PyG provides a common interface for various feature store implementations to interface with its core learning API.

The implementation details of a feature store are abstracted from PyG through a CRUD-like interface.
In particular, implementors of the feature store abstraction are expected to primarily override :obj:`put_tensor`, :obj:`get_tensor`, and :obj:`remove_tensor` functionalities (for a full list, see the abstract methods in :obj:`data/feature_store.py`).
Doing so both enables PyG to leverage the features stored in the inmplementation and allows a user to employ a Pythonic dict-like interface to inspect and modify the feature store elements:

.. code-block:: python

    feature_store = CustomFeatureStore()
    paper_features = ... # [num_papers, num_features_paper]
    author_features = ... # [num_authors, num_features_author]

    # Add features:
    feature_store['paper', 'x', None] = paper_features
    feature_store['author', 'x', None] = author_features

    # Get features, with a dict-like interface:
    assert torch.equal(feature_store['paper', 'x'], paper_features)
    assert torch.equal(feature_store['paper'].x, paper_features)
    assert torch.equal(feature_store['author', 'x', 0:20], author_features[0:20])

Common implementations of feature stores are key-value stores: backends such as `memcached`, `LevelDB`, `RocksDB`, etc. are all performant options.

Graph Store and Sampler
-----------------------

**Graph Store.** A graph store holds the edge indices that define relationships between nodes in a graph.
The goal of the graph store is to store graph information in a manner that allows for efficient sampling from root nodes, according to a sampling algorithm of the developer's choice.

Similar to the feature store, PyG provides a common interface for various graph store implementations to interface with its core learning API; this interface is defined in :obj:`data/graph_store.py`.
However, unlike the feature store, the graph store does not need to provide efficient random access for all its elements; rather, it needs to define a representation that provides efficient subgraph sampling.
Example usage of the interface is below:

.. code-block:: python

    graph_store = CustomGraphStore()
    edge_index = torch.LongTensor([[0, 1], [1, 2]])

    # Put:
    graph_store['edge', 'coo'] = coo

    # Get:
    row, col = graph_store['edge', 'coo']
    assert torch.equal(row, edge_index[0])
    assert torch.equal(col, edge_index[1])

Common implementations of graph stores are graph databases: Neo4j, TigerGraph, and others are all performant options.

**Graph Sampler.** A graph sampler is tightly tied with a graph store, and operates on the graph store to produce sampled subgraphs from input nodes.
Different sampling algorithms are implemented behind the base sampler interface, defined in :obj:`sampler/base.py`.
PyG's default in-memory sampler pulls all edge indices from the graph store into the training node memory, converts them to compressed sparse column (CSC) format, and leverages in-memory sampling routines defined in `pyg-lib <https://github.com/pyg-team/pyg-lib>`__.
However, custom sampler implementations may choose to call specialized graph store methods for efficiency reasons.

.. code-block:: python

    graph_store = ...  # as previous

    # CustomGraphSampler knows how to sample on CustomGraphStore:
    graph_sampler = CustomGraphSampler(
        graph_store=graph_store,
        num_neighbors=[10, 20],
        ...
    )

Data Loader
-----------------------

PyG does not define a domain-specific language for sampling that must be implemented by the graph store; rather, the sampler and graph store are tightly coupled through a data loader.

PyG provides two data loaders out-of-the-box: a loader that samples subgraphs from input nodes (:obj:`loader/node_loader.py`) for use in node classification tasks, and a loader that samples subgraphs from either side of an edge (:obj:`loader/link_loader.py`) for use in link prediction tasks.
These data loaders require a feature store, graph store, and graph sampler as input, and internally call the sampler's :obj:`sample_from_nodes` or :obj:`sample_from_edges` method to perform subgraph sampling:

.. code-block:: python

    feature_store = ...  # as previous
    graph_store = ...  # as previous
    graph_sampler = ...  # as previous

    # Instead of passing a `HeteroData` object, we now pass a tuple of the
    # feature store and graph store:
    loader = NodeLoader(
        data=(feature_store, graph_store),
        node_sampler=graph_sampler,
        batch_size=20,
        input_nodes='paper',
    )

    # Train:
    for batch in loader:
        ...

Putting it All Together
-----------------------

At a high level, the components listed above all work together to provide support for scaling up GNNs within PyG.

- The **data loader** (precisely, each worker) leverages a **graph sampler** to make a sampling request to the **graph store**.
- Upon receipt of a response, the data loader subsequently queries the **feature store** for features associated with the nodes and edges of the sampled subgraphs.
- The data loader subsequently constructs a final minibatch from graph structure and feature information to send to the accelerator for forward/backward.
- Repeat until convergence.

All of the outlined classes speak through common interfaces, making them extensible, generalizable, and easy to integrate with the PyG you use today:

.. figure:: ../_figures/remote_3.png
  :align: center
  :width: 80%

  **Figure 3:** The common interfaces (and data flow) uniting the feature store, graph store, graph sampler, and data loader.

To get started with scalability, we recommend inspecting the interfaces listed above and defining your own feature and graph store implementations behind them.
Once a feature store, graph store, and sampler are correctly implemented, simply pass them as parameters to a :obj:`NodeLoader` or :obj:`LinkLoader` (as above), and the rest of PyG will work seamlessly.

Since this feature is still undergoing heavy development, please feel free to reach out to the PyG Core Team either on Github or Slack if you have any questions, comments or concerns.
