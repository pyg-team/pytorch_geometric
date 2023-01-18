torch_geometric.nn
==================

.. contents:: Contents
    :local:

.. autoclass:: torch_geometric.nn.sequential.Sequential

.. currentmodule:: torch_geometric.nn.dense

   {% for name in torch_geometric.nn.dense.lin_classes %}
.. autoclass:: {{ name }}
   :members:
   :undoc-members:
   :exclude-members: training, initialize_parameters
   {% endfor %}

Convolutional Layers
--------------------

.. currentmodule:: torch_geometric.nn.conv
.. autosummary::
   :nosignatures:
   {% for name in torch_geometric.nn.conv.classes %}
     {{ name }}
   {% endfor %}

.. autoclass:: torch_geometric.nn.conv.MessagePassing
   :members:

.. automodule:: torch_geometric.nn.conv
   :members:
   :undoc-members:
   :exclude-members: message, aggregate, message_and_aggregate, update, MessagePassing, training, initialize_parameters

.. automodule:: torch_geometric.nn.meta
   :members:
   :undoc-members:
   :exclude-members: training

Aggregation Operators
---------------------

.. currentmodule:: torch_geometric.nn.aggr

Aggregation functions play an important role in the message passing framework and the readout functions of Graph Neural Networks.
Specifically, many works in the literature (`Hamilton et al. (2017) <https://arxiv.org/abs/1706.02216>`__, `Xu et al. (2018) <https://arxiv.org/abs/1810.00826>`__, `Corso et al. (2020) <https://arxiv.org/abs/2004.05718>`__, `Li et al. (2020) <https://arxiv.org/abs/2006.07739>`__, `Tailor et al. (2021) <https://arxiv.org/abs/2104.01481>`__) demonstrate that the choice of aggregation functions contributes significantly to the representational power and performance of the model.
For example, **mean aggregation** captures the distribution (or proportions) of elements, **max aggregation** proves to be advantageous to identify representative elements, and **sum aggregation** enables the learning of structural graph properties (`Xu et al. (2018) <https://arxiv.org/abs/1810.00826>`__).
Recent works also show that using **multiple aggregations** (`Corso et al. (2020) <https://arxiv.org/abs/2004.05718>`__, `Tailor et al. (2021) <https://arxiv.org/abs/2104.01481>`__) and **learnable aggregations** (`Li et al. (2020) <https://arxiv.org/abs/2006.07739>`__) can potentially provide substantial improvements.
Another line of research studies optimization-based and implicitly-defined aggregations (`Bartunov et al. (2022) <https://arxiv.org/abs/2202.12795>`__).
Furthermore, an interesting discussion concerns the trade-off between representational power (usually gained through learnable functions implemented as neural networks) and the formal property of permutation invariance (`Buterez et al. (2022) <https://arxiv.org/abs/2211.04952>`__).

To facilitate further experimentation and unify the concepts of aggregation within GNNs across both :class:`~torch_geometric.nn.conv.MessagePassing` and global readouts, we have made the concept of :class:`~torch_geometric.nn.aggr.Aggregation` a first-class principle in :pyg:`PyG`.
As of now, :pyg:`PyG` provides support for various aggregations --- from rather simple ones (*e.g.*, :obj:`mean`, :obj:`max`, :obj:`sum`), to advanced ones (*e.g.*, :obj:`median`, :obj:`var`, :obj:`std`), learnable ones (*e.g.*, :class:`~torch_geometric.nn.aggr.SoftmaxAggregation`, :class:`~torch_geometric.nn.aggr.PowerMeanAggregation`, :class:`~torch_geometric.nn.aggr.SetTransformerAggregation`), and exotic ones (*e.g.*, :class:`~torch_geometric.nn.aggr.MLPAggregation`, :class:`~torch_geometric.nn.aggr.LSTMAggregation`, :class:`~torch_geometric.nn.aggr.SortAggregation`, :class:`~torch_geometric.nn.aggr.EquilibriumAggregation`):

.. code-block:: python

   from torch_geometric.nn import aggr

   # Simple aggregations:
   mean_aggr = aggr.MeanAggregation()
   max_aggr = aggr.MaxAggregation()

   # Advanced aggregations:
   median_aggr = aggr.MedianAggregation()

   # Learnable aggregations:
   softmax_aggr = aggr.SoftmaxAggregation(learn=True)
   powermean_aggr = aggr.PowerMeanAggregation(learn=True)

   # Exotic aggregations:
   lstm_aggr = aggr.LSTMAggregation(in_channels=..., out_channels=...)
   sort_aggr = aggr.SortAggregation(k=4)

We can then easily apply these aggregations over a batch of sets of potentially varying size.
For this, an :obj:`index` vector defines the mapping from input elements to their location in the output:

.. code-block:: python

   # Feature matrix holding 1000 elements with 64 features each:
   x = torch.randn(1000, 64)

   # Randomly assign elements to 100 sets:
   index = torch.randint(0, 100, (1000, ))

   output = mean_aggr(x, index)  #  Output shape: [100, 64]

Notably, all aggregations share the same set of forward arguments, as described in detail in the :class:`torch_geometric.nn.aggr.Aggregation` base class.

Each of the provided aggregations can be used within :class:`~torch_geometric.nn.conv.MessagePassing` as well as for hierachical/global pooling to obtain graph-level representations:

.. code-block:: python

   import torch
   from torch_geometric.nn import MessagePassing

   class MyConv(MessagePassing):
       def __init__(self, ...):
           # Use a learnable softmax neighborhood aggregation:
           super().__init__(aggr=aggr.SoftmaxAggregation(learn=True))

      def forward(self, x, edge_index):
          ....


   class MyGNN(torch.nn.Module)
       def __init__(self, ...):
           super().__init__()

           self.conv = MyConv(...)
           # Use a global sort aggregation:
           self.global_pool = aggr.SortAggregation(k=4)
           self.classifier = torch.nn.Linear(...)

        def foward(self, x, edge_index, batch):
            x = self.conv(x, edge_index).relu()
            x = self.global_pool(x, batch)
            x = self.classifier(x)
            return x

In addition, the aggregation package of :pyg:`PyG` introduces two new concepts:
First, aggregations can be **resolved from pure strings** via a lookup table, following the design principles of the `class-resolver <https://github.com/cthoyt/class-resolver>`__ library, *e.g.*, by simply passing in :obj:`"median"` to the :class:`~torch_geometric.nn.conv.MessagePassing` module.
This will automatically resolve to the :obj:`~torch_geometric.nn.aggr.MedianAggregation` class:

.. code-block:: python

   class MyConv(MessagePassing):
       def __init__(self, ...):
           super().__init__(aggr="median")

Secondly, **multiple aggregations** can be combined and stacked via the :class:`~torch_geometric.nn.aggr.MultiAggregation` module in order to enhance the representational power of GNNs (`Corso et al. (2020) <https://arxiv.org/abs/2004.05718>`__, `Tailor et al. (2021) <https://arxiv.org/abs/2104.01481>`__):

.. code-block:: python

   class MyConv(MessagePassing):
       def __init__(self, ...):
           # Combines a set of aggregations and concatenates their results,
           # i.e. its output will be `[num_nodes, 3 * out_channels]` here.
           # Note that the interface also supports automatic resolution.
           super().__init__(aggr=aggr.MultiAggregation(
               ['mean', 'std', aggr.SoftmaxAggregation(learn=True)]))

Importantly, :class:`~torch_geometric.nn.aggr.MultiAggregation` provides various options to combine the outputs of its underlying aggegations (*e.g.*, using concatenation, summation, attention, ...) via its :obj:`mode` argument.
The default :obj:`mode` performs concatenation (:obj:`"cat"`).
For combining via attention, we need to additionally specify the :obj:`in_channels` :obj:`out_channels`, and :obj:`num_heads`:

.. code-block:: python

   multi_aggr = aggr.MultiAggregation(['mean', 'std'], in_channels=64,
                                      out_channels=64, num_heads=4))

If aggregations are given as a list, they will be automatically resolved to a :class:`~torch_geometric.nn.aggr.MultiAggregation`, *e.g.*, :obj:`aggr=['mean', 'std', 'median']`.

Finally, we added full support for customization of aggregations into the :class:`~torch_geometric.nn.conv.SAGEConv` layer --- simply override its :obj:`aggr` argument and **utilize the power of aggregation within your GNN**.

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.nn.aggr.classes %}
     {{ name }}
   {% endfor %}

Normalization Layers
--------------------

.. currentmodule:: torch_geometric.nn.norm

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.nn.norm.classes %}
     {{ name }}
   {% endfor %}

Pooling Layers
--------------

.. currentmodule:: torch_geometric.nn.pool

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.nn.pool.classes %}
     {{ name }}
   {% endfor %}

Unpooling Layers
----------------

.. currentmodule:: torch_geometric.nn.unpool

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.nn.unpool.classes %}
     {{ name }}
   {% endfor %}

Models
------

.. currentmodule:: torch_geometric.nn.models
.. autosummary::
   :nosignatures:
   {% for name in torch_geometric.nn.models.classes %}
     {{ name }}
   {% endfor %}

.. automodule:: torch_geometric.nn.models
   :members:
   :undoc-members:
   :exclude-members: message, aggregate, message_and_aggregate, update, MessagePassing, training, init_conv

KGE Models
----------

.. currentmodule:: torch_geometric.nn.kge

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.nn.kge.classes %}
     {{ name }}
   {% endfor %}

Encodings
---------

.. automodule:: torch_geometric.nn.encoding
   :members:
   :undoc-members:
   :exclude-members: training

Functional
----------

.. py:currentmodule:: torch_geometric.nn.functional

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.nn.functional.classes %}
     {{ name }}
   {% endfor %}

Dense Convolutional Layers
--------------------------

.. currentmodule:: torch_geometric.nn.dense

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.nn.dense.conv_classes %}
     {{ name }}
   {% endfor %}

Dense Pooling Layers
--------------------

.. currentmodule:: torch_geometric.nn.dense

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.nn.dense.pool_classes %}
     {{ name }}
   {% endfor %}

Model Transformations
---------------------

.. autoclass:: torch_geometric.nn.fx.Transformer
   :members:
   :undoc-members:
   :exclude-members: graph, find_by_target, find_by_name

.. autofunction:: torch_geometric.nn.to_hetero_transformer.to_hetero

.. autofunction:: torch_geometric.nn.to_hetero_with_bases_transformer.to_hetero_with_bases

DataParallel Layers
-------------------

.. automodule:: torch_geometric.nn.data_parallel
   :members:
   :undoc-members:
   :exclude-members: training


Model Summary
-------------

.. autofunction:: torch_geometric.nn.summary.summary
