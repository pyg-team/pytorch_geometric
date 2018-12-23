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
This base class automatically takes care of message propagation, so that the user only has to define the methods :math:`\phi` , *i.e.* the :meth:`message` method, and :math:`\gamma` , *.i.e.* the :meth:`update` method, as well as the aggregation scheme to use, *.i.e.* :obj:`aggr='add'`, :obj:`aggr='mean'` or :obj:`aggr='max'`.

This is done with the help of the following methods:

* :obj:`torch_geometric.nn.MessagePassing.propagate(aggr, edge_index, **kwargs)`:
  The initial call to start propagating messages.
  Takes in an aggregation scheme, the edge indices, and all additional data which is needed to construct messages and to update node embeddings.
* :meth:`torch_geometric.nn.MessagePassing.message`: Constructs messages in analogy to :math:`\phi` for each edge in :math:`(i,j) \in \mathcal{E}`.
  Can take any argument which was initially passed to :meth:`propagate`.
  In addition, features can be lifted to the source node :math:`i` and target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
* :meth:`torch_geometric.nn.MessagePassing.update`: Updates node embeddings in analogy to :math:`\gamma` for each node :math:`i \in \mathcal{V}`.
  Takes in the output of aggregation as first argument and any argument which was initially passed to :meth:`propagate`.

Let us verify this by re-implementing two popular GNN variants, `GCN from Kipf and Welling <https://arxiv.org/abs/1609.02907>`_ and the `Message Passing Layer from Gilmer et al. <https://arxiv.org/abs/1704.01212>`_:

A GCN layer is mathematically defined as

.. math::

    \mathbf{x}_i^{(k)} = \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{deg(j)}} \cdot \mathbf{\Theta} \cdot \mathbf{x}_j^{(k-1)},

where input node features are first transformed by a weight matrix :math:`\mathbf{\Theta}`, neighborhood-normalized and finally aggregated together.
We can identify six steps to compute this layer:

1. Add self-loops to the adjacency matrix.
2. Normalize adjacency matrix based on in- and out-degree.
3. Linearly transform node feature matrix.
4. Normalize node features in :math:`\phi`.
5. Sum up neighboring node features (:obj:`"add"` aggregation)
6. Discard old node features and replace them with the new ones in :math:`\gamma`.

Steps 1-3 are typically computed before message passing takes place.
Steps 4-6 can be easily processed using the :class:`torch_geometric.nn.MessagePassing` base class.
The full layer implementation is shown below.

.. code-block:: python

    import torch
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, degree

    class GCNConv(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super(GCNLayer, self).__init__()
            self.lin = torch.nn.Linear(in_channels, out_channels)

        def forward(self, x, edge_index):
            # Step 1: Add self-loops to the adjacency matrix.
            edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

            # Step 2: Compute normalized adjacency coefficients.
            row, col = edge_index
            deg = degree(row, num_nodes=x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # Step 3: Linearly transform node feature matrix.
            x = self.lin(x)

            # Step 4-6: Start propagating messages.
            return self.propagate('add', edge_index, x=x, norm=norm)

        def message(self, norm, x_j):
            # Step 4: Normalize node features.
            return norm.view(-1, 1) * x_j

        def update(self, aggr_out):
            # Step 6: Update node embeddings according to the aggregation output.
            return aggr_out

:class:`GCNConv` inherits from :class:`torch_geometric.nn.MessagePassing`.
The :meth:`forward` function contains the logic of this layer.
We first add self-loops to our edge indices using the :meth:`torch_geometric.utils.add_self_loops` functionality, as well as
computing node degrees :math:`\deg(i)` for each node :math:`i` and saving :math:`1/(\sqrt{\deg(i)} \cdot \sqrt{\deg(j)})` in :obj:`norm` for each edge :math:`(i,j) \in \mathcal{E}`.

Graph Nets
----------

Other Graph Neural Networks
---------------------------

The beauty of PyTorch Geometric is that one can easily create new message passing layer or more general graph nets, but is not limited to do.
