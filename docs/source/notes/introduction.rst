Introduction by example
=======================

We shortly introduce the fundamental concepts of `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric>`_ through self-contained examples.
At its core, PyTorch Geometric provides the following main features:

.. contents::
    :local:

Data handling of graphs
------------------------------------

A single graph in PyTorch Geometric is decribed by a ``Data`` object, which holds the graph data in its attributes.

- ``data.x``: Node feature matrix with shape ``[num_nodes, num_node_features]``
- ``data.edge_index``: Graph connectivity in COO format with shape ``[2, num_edges]`` and type ``torch.long``
- ``data.edge_attr``: Edge feature matrix with shape ``[num_edges, num_edge_features]``
- ``data.y``: Target features to train against with arbitrary shape
- ``data.pos``: Node position matrix with shape ``[num_nodes, num_dimensions]``

None of these attributes is required.
In fact, we can just extend the ``Data`` object by an arbitrary number of attributes, e.g. ``data.face`` to save the connectivity of triangles from a 3D mesh as a tensor with shape ``[3, num_faces]`` and type ``torch.long``.

.. Note::
    PyTorch defines an example as a tuple of an image and a target.
    We omit this notation in PyTorch Geometric to allow for various data structures in a clean und understandable way.

We show a simple example to save a unweighted, directed graph with three nodes and four edges. Each node is defined by one feature:

.. code-block:: python

    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[0], [1], [2]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    >>> Data(x=[3, 1], edge_index=[2, 4])

Note that you can print out your data object anytime and receive a short information of which attributes with which shapes its holding.

Besides of being a plain old python object, the ``Data`` object holds some utility functions, e.g.:

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

    # Bring data object onto the GPU.
    device = torch.device('cuda')
    data = data.to(device)

You can find a complete list of all ``Data`` methods at :class:`torch_geometric.data.Data`.

Common benchmark datasets
-------------------------

PyTorch Geometric contains a large number of common benchmark datasets, e.g. all Planetoid datasets (Cora, Citeseer, Pubmed), all graph classification datasets from `http://graphkernels.cs.tu-dortmund.de/ <http://graphkernels.cs.tu-dortmund.de/>`_ and a handful of 3D mesh/point cloud datasets (FAUST, ModelNet, ShapeNet).

Initializing datasets is straightforward, e.g. the ENZYMES dataset (consisting of 600 enzymes):

.. code-block:: python

    from torch_geometric.datasets import TUDataset

    dataset = TUDataset('/tmp/ENZYMES', name='ENZYMES')
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
There are 84 undirected edges and the graph belongs to exactly one out of six classes.

We can even use slices, long or byte tensors to split the dataset.
E.g., to create a 90/10 train/test split, type:

.. code-block:: python

    train_dataset = dataset[:540]
    >>> ENZYMES(540)

    test_dataset = dataset[540:]
    >>> ENZYMES(60)

If you are unsure whether the dataset is already shuffled, you can random permutate it by running:

.. code-block:: python

    dataset = dataset.shuffle()
    >>> ENZYMES(600)

Mini-Batches
------------

Data transformations
--------------------

Learning methods on graphs
--------------------------
