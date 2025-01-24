Compiled Graph Neural Networks
==============================

:meth:`torch.compile` is the latest method to speed up your :pytorch:`PyTorch` code in :obj:`torch >= 2.0.0`!
:meth:`torch.compile` makes PyTorch code run faster by JIT-compiling it into optimized kernels, all while required minimal code changes.

Under the hood, :meth:`torch.compile` captures :pytorch:`PyTorch` programs via :obj:`TorchDynamo`, canonicalizes over 2,000 :pytorch:`PyTorch` operators via :obj:`PrimTorch`, and finally generates fast code out of it across multiple accelerators and backends via the deep learning compiler :obj:`TorchInductor`.

.. note::
    See `here <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__ for a general tutorial on how to leverage :meth:`torch.compile`, and `here <https://pytorch.org/docs/stable/generated/torch.compile.html>`__ for a description of its interface.

In this tutorial, we show how to optimize your custom :pyg:`PyG` model via :meth:`torch.compile`.

.. note::
    From :pyg:`PyG` 2.5 (and onwards), :meth:`torch.compile` is now fully compatible with all :pyg:`PyG` GNN layers.
    If you are on an earlier version of :pyg:`PyG`, consider using :meth:`torch_geometric.compile` instead.

Basic Usage
-----------

Once you have a :pyg:`PyG` model defined, simply wrap it with :meth:`torch.compile` to obtain its optimized version:

.. code-block:: python

    import torch
    from torch_geometric.nn import GraphSAGE

    model = GraphSAGE(in_channels, hidden_channels, num_layers, out_channels)
    model = model.to(device)

    model = torch.compile(model)

and execute it as usual:

.. code-block:: python

    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root, name="Cora")
    data = dataset[0].to(device)

    out = model(data.x, data.edge_index)

Maximizing Performance
----------------------

The :meth:`torch.compile` method provides two important arguments to be aware of:

* Most of the mini-batches observed in :pyg:`PyG` are dynamic by nature, meaning that their shape varies across different mini-batches.
  For these scenarios, we can enforce dynamic shape tracing in :pytorch:`PyTorch` via the :obj:`dynamic=True` argument:

  .. code-block:: python

      torch.compile(model, dynamic=True)

  With this, :pytorch:`PyTorch` will up-front attempt to generate a kernel that is as dynamic as possible to avoid recompilations when sizes change across mini-batches changes.
  Note that when :obj:`dynamic` is set to :obj:`False`, :pytorch:`PyTorch` will *never* generate dynamic kernels, and thus only works when graph sizes are guaranteed to never change (*e.g.*, in full-batch training on small graphs).
  By default, :obj:`dynamic` is set to :obj:`None` in :pytorch:`PyTorch` :obj:`>= 2.1.0`, and :pytorch:`PyTorch` will automatically detect if dynamism has occured.
  Note that support for dynamic shape tracing requires :pytorch:`PyTorch` :obj:`>= 2.1.0` to be installed.

* In order to maximize speedup, graphs breaks in the compiled model should be limited.
  We can force compilation to raise an error upon the first graph break encountered by using the :obj:`fullgraph=True` argument:

  .. code-block:: python

      torch.compile(model, fullgraph=True)

  It is generally a good practice to confirm that your written model does not contain any graph breaks.
  Importantly, there exists a few operations in :pyg:`PyG` that will currently lead to graph breaks (but workaround exists), *e.g.*:

  1. :meth:`~torch_geometric.nn.pool.global_mean_pool` (and other pooling operators) perform device synchronization in case the batch size :obj:`size` is not passed, leading to a graph break.

  2. :meth:`~torch_geometric.utils.remove_self_loops` and :meth:`~torch_geometric.utils.add_remaining_self_loops` mask the given :obj:`edge_index`, leading to a device synchronization to compute its final output shape.
     As such, we recommend to augment your graph *before* inputting it into your GNN, *e.g.*, via the :class:`~torch_geometric.transforms.AddSelfLoops` or :class:`~torch_geometric.transforms.GCNNorm` transformations, and setting :obj:`add_self_loops=False`/:obj:`normalize=False` when initializing layers such as :class:`~torch_geometric.nn.conv.GCNConv`.

Exampe Scripts
--------------

We have incorporated multiple examples in :obj:`examples/compile` that further show the practical usage of :meth:`torch.compile`:

#. `Node Classification <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gcn.py>`__ via :class:`~torch_geometric.nn.models.GCN` (:obj:`dynamic=False`)
#. `Graph Classification <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gin.py>`__ via :class:`~torch_geometric.nn.models.GIN` (:obj:`dynamic=True`)

If you notice that :meth:`torch.compile` fails for a certain :pyg:`PyG` model, do not hesitate to reach out either on :github:`null` `GitHub <https://github.com/pyg-team/pytorch_geometric/issues>`_ or :slack:`null` `Slack <https://data.pyg.org/slack.html>`_.
We are very eager to improve :meth:`torch.compile` support across the whole :pyg:`PyG` code base.

Benchmark
---------

:meth:`torch.compile` works **fantastically well** for many :pyg:`PyG` models.
**Overall, we observe runtime improvements of nearly up to 300%.**

Specifically, we benchmark :class:`~torch_geometric.nn.models.GCN`, :class:`~torch_geometric.nn.models.GraphSAGE` and :class:`~torch_geometric.nn.models.GIN` and compare runtimes obtained from traditional eager mode and :meth:`torch.compile`.
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
