Introduction by example
=======================

We shortly introduce the fundamental concepts of `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric>`_ through self-contained examples.
At its core, PyTorch Geometric provides the following main features:

.. contents::
    :local:

Data handling of graphs
------------------------------------

A single graph in PyTorch Geometric is decribed by :class:`torch_geometric.data.Data`.
This ``Data`` object holds all information needed to define an arbitrary graph.
There are already some predefined attributes:

- ``data.x``: Node feature matrix with shape ``[num_nodes, num_node_features]``
- ``data.edge_index``: Graph connectivity in COO format with shape ``[2, num_edges]`` and type ``torch.long``
- ``data.edge_attr``: Edge feature matrix with shape ``[num_edges, num_edge_features]``
- ``data.y``: Target to train against (may have arbitrary shape)
- ``data.pos``: Node position matrix with shape ``[num_nodes, num_dimensions]``

None of these attributes is required.
In fact, the ``Data`` object is not even restricted to these attributes.
We can, e.g., extend it by ``data.face`` to save the connectivity of triangles from a 3D mesh in a tensor with shape ``[3, num_faces]`` and type ``torch.long``.

.. Note::
    PyTorch and ``torchvision`` define an example as a tuple of an image and a target.
    In contrast, we omit this notation in PyTorch Geometric to allow for various data structures in a clean und understandable way.

We show a simple example of an unweighted and undirected graph with three nodes and four edges.
Each node is assigned exactly one feature:

.. code-block:: python

    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    >>> Data(x=[3, 1], edge_index=[2, 4])

.. tikz::
   :include: graph.tikz

.. Note::
    You can print out your data object anytime and receive a short information about its attributes and their shapes.

Besides of being a plain old python object, :class:`torch_geometric.data.Data` provides a number of utility functions, e.g.:

.. code-block:: python

    print(data.keys)
    >>> ['x', 'edge_index']

    print(data['x'])
    >>> tensor([[0.0],
                [1.0],
                [2.0]])

    for key, item in data:
        print('{} found in data)
    >>> x found in data
    >>> edge_index found in data

    'edge_attr' in data
    >>> False

    data.num_nodes
    >>> 3

    data.num_edges
    >>> 4

    data.num_features
    >>> 1

    data.contains_isolated_nodes()
    >>> False

    data.contains_self_loops()
    >>> False

    data.is_directed()
    >>> True

    # Transfer data object to GPU.
    device = torch.device('cuda')
    data = data.to(device)

You can find a complete list of all methods at :class:`torch_geometric.data.Data`.

Common benchmark datasets
-------------------------

PyTorch Geometric contains a large number of common benchmark datasets, e.g. all Planetoid datasets (Cora, Citeseer, Pubmed), all graph classification datasets from `http://graphkernels.cs.tu-dortmund.de/ <http://graphkernels.cs.tu-dortmund.de/>`_, the QM9 dataset, and a handful of 3D mesh/point cloud datasets (FAUST, ModelNet10/40, ShapeNet).

Initializing a dataset is straightforward.
The dataset will be automatically downloaded and process the graphs to the previously decribed ``Data`` format.
E.g., to load the ENZYMES dataset (consisting of 600 graphs within 6 classes), type:

.. code-block:: python

    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    >>> ENZYMES(600)

    len(dataset)
    >>> 600

    dataset.num_classes
    >>> 6

    dataset.num_features
    >>> 21

We now have access to all 600 graphs in the dataset:

.. code-block:: python

    data = dataset[0]
    >>> Data(x=[37, 21], edge_index=[2, 168], y=[1])

    data.is_undirected()
    >>> True

We can see that the first graph in the dataset contains 37 nodes, each one having 21 features.
There are 168/2 = 84 undirected edges and the graph is assigned to exactly one class.

We can even use slices, long or byte tensors to split the dataset.
E.g., to create a 90/10 train/test split, type:

.. code-block:: python

    train_dataset = dataset[:540]
    >>> ENZYMES(540)

    test_dataset = dataset[540:]
    >>> ENZYMES(60)

If you are unsure whether the dataset is already shuffled before you split, you can random permutate it by running:

.. code-block:: python

    dataset = dataset.shuffle()
    >>> ENZYMES(600)

This is equivalent of doing:

.. code-block:: python

    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    >> ENZYMES(600)

Let's try another one! Let's download Cora, the standard benchmark dataset for semi-supervised graph node classification:

.. code-block:: python

    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    >>> Cora()

    len(dataset)
    >>> 1

    dataset.num_classes
    >>> 7

    dataset.num_features
    >>> 1433

Here, the dataset contains only a single, undirected citation graph:

.. code-block:: python

    data = dataset[0]
    >>> Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])

    data.is_undirected()
    >>> True

    data.train_mask.sum()
    >>> 140

    data.val_mask.sum()
    >>> 500

    data.test_mask.sum()
    >>> 1000

This time, the ``Data`` objects holds additional attributes: ``train_mask``, ``val_mask`` and ``test_mask``:

- ``train_mask`` denotes against which nodes to train (140 nodes)
- ``val_mask`` denotes which nodes to use for validation, e.g., to perform early stopping (500 nodes)
- ``test_mask`` denotes against which nodes to test (1000 nodes)

Mini-Batches
------------

Neural networks are usually trained in a batch-wise fashion.
PyTorch Geometric achieves parallelization over a mini-batch by creating sparse block diagonal adjacency matrices (defined by ``edge_index`` and ``edge_attr``) and concatenating feature and target matrices in the node dimension.
This composition allows differing number of nodes and edges over examples in one batch:

.. math::

    \mathbf{A} = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_n \end{bmatrix}, \qquad \mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_n \end{bmatrix}, \qquad \mathbf{Y} = \begin{bmatrix} \mathbf{Y}_1 \\ \vdots \\ \mathbf{Y}_n \end{bmatrix}

PyTorch Geometric contains its own :class:`torch_geometric.data.DataLoader`, which already takes care of this concatenation process.
Let's learn about it in an example:

.. code-block:: python

    from torch_geometric.datasets import TUDataset
    from torch_geometric.data import DataLoader

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        batch
        >>> Batch(x=[1082, 21], edge_index=[2, 4066], y=[32], batch=[1082])

        batch.num_graphs
        >>> 32

:class:`torch_geometric.data.Batch` inherits from :class:`torch_geometric.data.Data` and contains an additional attribute: ``batch``.

``batch`` is a column vector of graph identifiers for all nodes of all graphs in the batch:

.. math::

    \mathrm{batch} = {\begin{bmatrix} 0 & \cdots & 0 & 1 & \cdots & n - 2 & n -1 & \cdots & n - 1 \end{bmatrix}}^{\top}

You can use it to, e.g., average node features for each graph individually:

.. code-block:: python

    from torch_scatter import scatter_mean
    from torch_geometric.datasets import TUDataset
    from torch_geometric.data import DataLoader

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in loader:
        data
        >>> Batch(x=[1082, 21], edge_index=[2, 4066], y=[32], batch=[1082])

        data.num_graphs
        >>> 32

        x = scatter_mean(data.x, data.batch, dim=0)
        x.size()
        >>> torch.Size([32, 21])

You can learn more about scatter operations in the `documentation <http://rusty1s.github.io/pytorch_scatter>`_ of ``torch_scatter``.

Data transformations
--------------------

Learning methods on graphs
--------------------------
