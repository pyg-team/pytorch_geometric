TorchScript Support
===================

TorchScript is a way to create serializable and optimizable models from :pytorch:`PyTorch` code.
Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.
If you are unfamilar with TorchScript, we recommend to read the official "`Introduction to TorchScript <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_" tutorial first.

Converting GNN Models
---------------------

Converting your :pyg:`PyG` model to a TorchScript program is straightforward and requires only a few code changes.

As always, it is best understood by an example, so let's consider the following model for now:

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

For TorchScript support, we need to convert our GNN operators into "jittable" instances.
This is done by calling the :func:`~torch_geometric.nn.conv.message_passing.MessagePassing.jittable` function provided by the underlying :class:`~torch_geometric.nn.conv.message_passing.MessagePassing` interface:

.. code-block:: python

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64).jittable()
        self.conv2 = GCNConv(64, out_channels).jittable()

This will create temporary instances of the :class:`~torch_geometric.nn.conv.GCNConv` operator that can now be passed into :func:`torch.jit.script`:

.. code-block:: python

    model = torch.jit.script(model)

Under the hood, the :func:`~torch_geometric.nn.conv.message_passing.MessagePassing.jittable` call applies the following two modifications to the original class:

1. It parses and converts the arguments of the internal :func:`~torch_geometric.nn.conv.message_passing.MessagePassing.propagate` function into a :obj:`NamedTuple` which can be handled by the TorchScript compiler.
2. It replaces any :obj:`Union` arguments of the :func:`forward` function (*i.e.*, arguments that may contain different types) with :obj:`@torch.jit._overload_method` annotations.
   With this, we can do the following while everything remains jittable:

.. code-block:: python

    from typing import Union, Tuple
    from torch import Tensor

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], edge_index: Tensor) -> Tensor:
        pass

    conv(x, edge_index)
    conv((x_src, x_dst), edge_index)

This technique is, *e.g.*, applied in the :class:`~torch_geometric.nn.conv.SAGEConv` class, which can operate on both single node feature matrices and tuples of node feature matrices at the same time.

And that is all you need to know on how to convert your :pyg:`PyG` models to TorchScript programs.
You can have a further look at our JIT examples that show-case how to obtain TorchScript programs for `node <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/jit/gat.py>`_ and `graph classification <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/jit/gin.py>`_ models.

.. note::
    TorchScript support is still experimental.
    If you have problems converting your model to a TorchScript program, *e.g.*, because an internal function does not support TorchScript yet, please `let us know <https://github.com/pyg-team/pytorch_geometric/issues/new/choose>`_.

Creating Jittable GNN Operators
--------------------------------

All :pyg:`PyG` :class:`~torch_geometric.nn.conv.MessagePassing` operators are tested to be convertible to a TorchScript program.
However, if you want your own GNN module to be jittable, you need to account for the following two things:

1. As one would expect, your :meth:`forward` code may need to be adjusted so that it passes the TorchScript compiler requirements, *e.g.*, by adding type notations.
2. You need to tell the :class:`~torch_geometric.nn.conv.message_passing.MessagePassing` module the types that you pass to its :func:`~torch_geometric.nn.conv.message_passing.MessagePassing.propagate` function.
   This can be achieved in two different ways:

   1. Declaring the type of propagation arguments in a dictionary called :obj:`propagate_type`:

    .. code-block:: python

        from typing import Optional
        from torch import Tensor
        from torch_geometric.nn import MessagePassing

        class MyConv(MessagePassing):

            propagate_type = {'x': Tensor, 'edge_weight': Optional[Tensor] }

            def forward(self, x: Tensor, edge_index: Tensor,
                        edge_weight: Optional[Tensor]) -> Tensor:

                return self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                      size=None)

   2. Declaring the type of propagation arguments as a comment anywhere inside your module:

    .. code-block:: python

        from typing import Optional
        from torch import Tensor
        from torch_geometric.nn import MessagePassing

        class MyConv(MessagePassing):

            def forward(self, x: Tensor, edge_index: Tensor,
                        edge_weight: Optional[Tensor]) -> Tensor:

                # propagate_type: (x: Tensor, edge_weight: Optional[Tensor])
                return self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                      size=None)

.. warning::

   Importantly, due to TorchScript limitations, one also has to pass in the :obj:`size` attribute to :meth:`~torch_geometric.nn.conv.message_passing.MessagePassing.propagate`.
   In most cases, this can be simply set to :obj:`None` in which case it will be automatically inferred.
