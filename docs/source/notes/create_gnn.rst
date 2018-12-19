Create your own graph neural network
====================================

.. We shortly introduce the fundamental concepts of `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric>`_ through self-contained examples.
.. At its core, PyTorch Geometric provides the following main features:

.. contents::
    :local:

Message Passing Graph Neural Networks
-------------------------------------

Generalizing the convolution operator to irregular domains is typically expressed as a *neighborhood aggregation* or *message passing* scheme.
Let therefore :math:`\mathbf{x}^{(k-1)}_i \in \mathbb{R}^F` denote node features of node :math:`i` in layer :math:`(k-1)` and let :math:`\mathbf{e}_{i,j} \in \mathbb{R}^D` denote (optional) edge features from node :math:`i` to node :math:`j`.
In its most general form, message passing graph neural networks can then be described as

.. math::
  \mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \, \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)},\mathbf{e}_{i,j}\right) \right),

where :math:`\square` denotes a differentiable, permutation invariant function, *e.g.*, sum, mean or max, and :math:`\gamma` and :math:`\phi` denote differentiable functions such as MLPs.

Nearly all of the already implemented methods in PyTorch Geometric follow this simple scheme.
Using the :class:`torch_geometric.nn.MessagePassing` base class, it is really simple to create them by yourself.
All you need to do is implement :math:`\gamma` (*.i.e.* the :meth:`update` method) and :math:`\phi` (*i.e.* the :meth:`message` method), and choose between various permutation invariant functions :math:`\square` (:attr:`aggr`).

Let us verify this by re-implementing two popular GNN variants, GCN and GraphSAGE:

.. math::

    \mathbf{x}_i^{(k)} = \mathbf{\Theta} \cdot \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{deg(j)}} \cdot \mathbf{x}_j^{(k-1)}

.. code-block:: python

    import torch
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, degree

    class GCNConv(MessagePassing):
        def __init__(in_channels, out_channels):
            super(GCNConv, self).__init__(aggr='add')
            self.lin = torch.nn.Linear(in_channels, out_channels)

        def prepare(x, edge_index):
            # We need to add self-loops to the adjaceny matrix.
            edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

            # Compute normalized adjacency matrix.
            row, col = edge_index
            deg = degree(row, num_nodes=x.size(0), dtype=x.dtype)
            deg_inv = deg.pow(-0.5)
            edge_attr = deg_inv[row] * deg_inv[col]

            return x, edge_index, edge_attr

        def message(x_i, x_j, edge_attr):
            return x_j * edge_attr

        def update(x_old, x):
            return self.lin(x)

Graph Nets
----------

Other Graph Neural Networks
---------------------------

The beauty of PyTorch Geometric is that one can easily create new message passing layer or more general graph nets, but is not limited to do.
