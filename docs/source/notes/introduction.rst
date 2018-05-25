Introduction by example
=======================

We shortly introduce the fundamental concepts of `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric>`_ through self-contained examples.
At its core, we provide four main features:

.. contents::
    :local:

Data handling of graphs
------------------------------------

A single graph in PyTorch Geometric is decribed by a ``Data`` object, which holds the graph data in its attributes.

- ``data.x``: Node feature matrix with shape ``[num_nodes, num_node_features]``
- ``data.edge_index``: Graph connectivity in COO format with shape ``[2, num_edges]`` and dtype ``torch.long``
- ``data.edge_attr``: Edge feature matrix with shape ``[num_edges, num_edge_features]``
- ``data.y``: Target features to train against with arbitrary shape
- ``data.pos``: Node position matrix with shape ``[num_nodes, num_dimensions]``

None of these attributes is required.
In fact, we can just extend the ``Data`` object by an arbitrary number of attributes, e.g. ``data.face`` to save the triangle connectivity of 3D meshes with shape ``[3, num_faces]`` and dtype ``torch.long``.

We show a simple example to save a unweighted, directed graph with three nodes and four edges:

.. code-block:: python

    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[0], [1], [2]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    print(data)
    >>> Data(x=[3, 1], edge_index=[2, 4])

Note that you can print out your data object anytime and receive a short information of which attributes with which shapes its holding.

Besides of being a plain old python obect, the ``Data`` object holds some utility functions:

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

    print('edge_attr' in data)
    >>> False

    print(data.num_nodes)
    >>> 3

    print(data.num_edges)
    >>> 4

    print(data.num_features)
    >>> 1

    print(data.contains_isolated_nodes())
    >>> False

    print(data.contains_self_loops())
    >>> False

    print(data.is_directed())
    >>> True

    # Bring data object onto the GPU.
    device = torch.device('cuda')
    data = data.to(device)

Common benchmark datasets
-------------------------

Data transformations
--------------------

Learning methods on graphs
--------------------------
