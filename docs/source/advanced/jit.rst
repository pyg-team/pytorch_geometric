TorchScript Support
===================

TorchScript is a way to create serializable and optimizable models from :pytorch:`PyTorch` code.
Any TorchScript program can be saved from a :python:`Python` process and loaded in a process where there is no :python:`Python` dependency.
If you are unfamilar with TorchScript, we recommend to read the official "`Introduction to TorchScript <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_" tutorial first.

Converting GNN Models
---------------------

.. note::
    From :pyg:`PyG` 2.5 (and onwards), GNN layers are now fully compatible with :meth:`torch.jit.script` without any modification needed.
    If you are on an earlier version of :pyg:`PyG`, consider to convert your GNN layers into "jittable" instances first by calling :meth:`~torch_geometric.nn.conv.MessagePassing.jittable`.

Converting your :pyg:`PyG` model to a TorchScript program is straightforward and requires only a few code changes.
Let's consider the following model:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 64)
            self.conv2 = GCNConv(64, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    model = GNN(dataset.num_features, dataset.num_classes)

The instantiated model can now be directly passed into :meth:`torch.jit.script`:

.. code-block:: python

    model = torch.jit.script(model)

That is all you need to know on how to convert your :pyg:`PyG` models to TorchScript programs.
You can have a further look at our JIT examples that show-case how to obtain TorchScript programs for `node <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/jit/gat.py>`_ and `graph classification <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/jit/gin.py>`_ models.

Creating Jittable GNN Operators
--------------------------------

All :pyg:`PyG` :class:`~torch_geometric.nn.conv.MessagePassing` operators are tested to be convertible to a TorchScript program.
However, if you want your own GNN module to be compatible with :meth:`torch.jit.script`, you need to account for the following two things:

1. As one would expect, your :meth:`forward` code may need to be adjusted so that it passes the TorchScript compiler requirements, *e.g.*, by adding type notations.
2. You need to tell the :class:`~torch_geometric.nn.conv.MessagePassing` module the types that you pass to its :meth:`~torch_geometric.nn.conv.MessagePassing.propagate` function.
   This can be achieved in two different ways:

   1. Declaring the type of propagation arguments in a dictionary called :obj:`propagate_type`:

    .. code-block:: python

        from typing import Optional
        from torch import Tensor
        from torch_geometric.nn import MessagePassing

        class MyConv(MessagePassing):
            propagate_type = {'x': Tensor, 'edge_weight': Optional[Tensor] }

            def forward(
                self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: Optional[Tensor] = None,
            ) -> Tensor:
                return self.propagate(edge_index, x=x, edge_weight=edge_weight)

   2. Declaring the type of propagation arguments as a comment inside your module:

    .. code-block:: python

        from typing import Optional
        from torch import Tensor
        from torch_geometric.nn import MessagePassing

        class MyConv(MessagePassing):
            def forward(
                self,
                x: Tensor,
                edge_index: Tensor,
                edge_weight: Optional[Tensor] = None,
            ) -> Tensor:
                # propagate_type: (x: Tensor, edge_weight: Optional[Tensor])
                return self.propagate(edge_index, x=x, edge_weight=edge_weight)

   If none of these options are given, the :class:`~torch_geometric.nn.conv.MessagePassing` module will infer the arguments of :meth:`~torch_geometric.nn.conv.MessagePassing.propagate` to be of type :class:`torch.Tensor` (mimicking the default type that TorchScript is inferring for non-annotated arguments).
