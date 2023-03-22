Compiled Graph Neural Networks
==============================

:meth:`torch.compile` is the latest method to speed up your :pytorch:`PyTorch` code in :obj:`torch >= 2.0.0`!
:meth:`torch.compile` makes PyTorch code run faster by JIT-compiling it into opimized kernels, all while required minimal code changes.

Under the hood, :meth:`torch.compile` captures :pytorch:`PyTorch` programs via :obj:`TorchDynamo`, canonicalizes over 2,000 :pytorch:`PyTorch` operators via :obj:`PrimTorch`, and finally generates fast code out of it across multiple accelerators and backends via the deep learning compiler :obj:`TorchInductor`.

.. note::
    See `here <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__ for a general tutorial on how to leverage :meth:`torch.compile`, and `here <https://pytorch.org/docs/stable/generated/torch.compile.html>`__ for a description of its interface.

In this tutorial, we show how to optimize your custom :pyg:`PyG` model via :meth:`torch.compile`.

Introducing :meth:`torch_geometric.compile`
-------------------------------------------

By default, :meth:`torch.compile` struggles to optimize a custom :pyg:`PyG` model since its underlying :class:`~torch_geometric.nn.conv.MessagePassing` interface is JIT-unfriendly due to its generality.
As such, in :pyg:`PyG 2.3`, we introduce :meth:`torch_geometric.compile`, a wrapper around :meth:`torch.compile` with the same signature.

:meth:`torch_geometric.compile` applies further optimizations to make :pyg:`PyG` models more compiler-friendly.
Specifically, it:

#. Temporarily disables the usage of the extension packages :obj:`torch_scatter`, :obj:`torch_sparse` and :obj:`pyg_lib` during GNN execution workflows (since these are not *yet* directly optimizable by :pytorch:`PyTorch`).
   From :pyg:`PyG 2.3` onwards, these packages are purely optional and not required anymore for running :pyg:`PyG` models (but :obj:`pyg_lib` may be required for graph sampling routines).

#. Converts all instances of :class:`~torch_geometric.nn.conv.MessagePassing` modules into their jittable instances (see :meth:`torch_geometric.nn.conv.MessagePassing.jittable`)

Without these adjustments, :meth:`torch.compile` may currently fail to correctly optimize your :pyg:`PyG` model.
We are working on fully relying on :meth:`torch.compile` for future releases.

Basic Usage
-----------

Leveraging :meth:`torch_geometric.compile` is as simple as the usage of :meth:`torch.compile`.
Once you have a :pyg:`PyG` model defined, simply wrap it with :meth:`torch_geometric.compile` to obtain its optimized version:

.. code-block:: python

    import torch_geometric
    from torch_geometric.nn import GraphSAGE

    model = GraphSAGE(in_channels, hidden_channels, num_layers, out_channels)
    model = model.to(device)

    model = torch_geometric.compile(model)

and execute it as usual:

.. code-block:: python

    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root, name="Cora")
    data = dataset[0].to(device)

    out = model(data.x, data.edge_index)

We have incorporated multiple examples in :obj:`examples/compile` that further show the practical usage of :meth:`torch_geometric.compile`:

#. `Node Classification <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gcn.py>`__ via :class:`~torch_geometric.nn.models.GCN`
#. `Graph Classification <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gin.py>`__ via :class:`~torch_geometric.nn.models.GIN`

Note that :meth:`torch.compile(model, dynamic=True)` does sadly not yet work for :pyg:`PyG` models on :pytorch:`PyTorch 2.0`.
While static compilation via :meth:`torch.compile(model, dynamic=False)` works fine, it will re-compile the model everytime it sees an input with a different shape.
That currently does not play that nicely with the way :pyg:`PyG` performs mini-batching, and will hence lead to major slow-downs.
We are working with the :pytorch:`PyTorch` team to fix this limitation (see `this <https://github.com/pytorch/pytorch/issues/94640>`_ :github:`GitHub` issue).
A temporary workaround is to utilize the :class:`torch_geometric.transforms.Pad` transformation to ensure that all inputs are of equal shape.

If you notice that :meth:`~torch_geometric.compile` fails for a certain :pyg:`PyG` model, do not hesitate to reach out either on :github:`null` `GitHub <https://github.com/pyg-team/pytorch_geometric/issues>`_ or :slack:`null` `Slack <https://data.pyg.org/slack.html>`_.
We are very eager to improve :meth:`~torch_geometric.compile` support across the whole :pyg:`PyG` code base.

Benchmark
---------

:meth:`torch.compile` works **fantastically well** for many :pyg:`PyG` models.
**Overall, we observe runtime improvements of nearly up to 300%.**

Specifically, we benchmark :class:`~torch_geometric.nn.models.GCN`, :class:`~torch_geometric.nn.models.GraphSAGE` and :class:`~torch_geometric.nn.models.GIN` and compare runtimes obtained from traditional eager mode and :meth:`torch_geometric.compile`.
We use a synthetic graph with 10,000 nodes and 200,000 edges, and a hidden feature dimensionality of 64.
We report runtimes over 500 optimization steps:

.. list-table::
   :widths: 15 15 15 15 15 15
   :header-rows: 1

   * - Model
     - Mode
     - Forward
     - Backward
     - Total
     - Speedup
   * - :class:`~torch_geometric.nn.models.GCN`
     - Eager
     - 2.6396s
     - 2.1697s
     - 4.8093s
     -
   * - :class:`~torch_geometric.nn.models.GCN`
     - **Compiled**
     - **1.1082s**
     - **0.5896s**
     - **1.6978s**
     - **2.83x**
   * - :class:`~torch_geometric.nn.models.GraphSAGE`
     - Eager
     - 1.6023s
     - 1.6428s
     - 3.2451s
     -
   * - :class:`~torch_geometric.nn.models.GraphSAGE`
     - **Compiled**
     - **0.7033s**
     - **0.7465s**
     - **1.4498s**
     - **2.24x**
   * - :class:`~torch_geometric.nn.models.GIN`
     - Eager
     - 1.6701s
     - 1.6990s
     - 3.3690s
     -
   * - :class:`~torch_geometric.nn.models.GIN`
     - **Compiled**
     - **0.7320s**
     - **0.7407s**
     - **1.4727s**
     - **2.29x**

To reproduce these results, run

.. code-block:: console

    python test/nn/models/test_basic_gnn.py

from the root folder of your checked out :pyg:`PyG` repository from :github:`GitHub`.
