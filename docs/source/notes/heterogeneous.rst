Heterogeneous Graph Learning
============================

A large set of real-world datasets are stored as heterogeneous graphs, motivating the introduction of specialized functionality for them in Pytorch Geometric (PyG).
For example, most graphs in the area of recommendation, such as social graphs, are heterogeneous, as they store information about different types of entities and their different types of relations.
This tutorial introduces how heterogeneous graphs are mapped to PyG and how they can be used as input to Graph Neural Network models.

Heterogeneous graphs come with different types of information attached to nodes and edges.
Thus, a single node or edge feature tensor can not hold all node or edge features of the whole graph, due to differences in type and dimensionality.
Instead, a set of types need to be specified for nodes and edges, respectively, each having its own data tensors.
As a consequence of the different data structure, the message passing formulation changes accordingly, allowing the computation of message and update function conditioned on node or edge type.

Example
-------

As a guiding example, we take a look at the heterogenous `ogbn-mag <https://ogb.stanford.edu/docs/nodeprop>`__ network from the `OGB datasets <https://ogb.stanford.edu>`_:

.. image:: ../_figures/hg_example.svg
  :align: center
  :width: 500px

The given heterogeneous graph has 1,939,743 nodes, split between the four node types **author**, **paper**, **institution** and **field of study**.
It further has 21,111,007 edges, which also are of one of four types:

* **writes**: An author *writes* a specific paper
* **affiliated with**: An author is *affiliated with* a specific institution
* **cites**: A paper *cites* another paper
* **has topic**: A paper *has a topic* of a specific field of study

The task for this graph is to infer the venue of each paper (conference or journal) given the information stored in the graph.

Creating Heterogeneous Graphs
-----------------------------

First, we can create a data object of type :class:`torch_geometric.data.HeteroData`, for which we define node feature tensors, edge index tensors and edge feature tensors individually for each type:

.. code-block:: python

   from torch_geometric.data import HeteroData

   data = HeteroData()

   data['paper'].x = ... # [num_papers, num_features_paper]
   data['author'].x = ... # [num_authors, num_features_author]
   data['institution'].x = ... # [num_institutions, num_features_institution]
   data['field_of_study'].x = ... # [num_field, num_features_field]

   data['paper', 'cites', 'paper'].edge_index = ... # [2, num_edges_cites]
   data['author', 'writes', 'paper'].edge_index = ... # [2, num_edges_writes]
   data['author', 'affiliated_with', 'institution'].edge_index = ... # [2, num_edges_affiliated]
   data['author', 'has_topic', 'institution'].edge_index = ... # [2, num_edges_topic]

   data['paper', 'cites', 'paper'].edge_attr = ... # [num_edges_cites, num_features_cites]
   data['author', 'writes', 'paper'].edge_attr = ... # [num_edges_writes, num_features_writes]
   data['author', 'affiliated_with', 'institution'].edge_attr = ... # [num_edges_affiliated, num_features_affiliated]
   data['paper', 'has_topic', 'field_of_study'].edge_attr = ... # [num_edges_topic, num_features_topic]

Node or edge tensors will be automatically created upon first access and indexed by string keys.
Node types are identified by a single string while edge types are identified by using a triplet :obj:`(source_node_type, edge_type, destination_node_type)` of strings, the edge type identifier and the two node types, between which the edge type can exist.
The data object allows different feature dimensionalities for each type.

Dictionaries containing the heterogeneous information grouped by attribute names rather than by node or edge type can directly be accessed via :obj:`data.{attribute_name}_dict` and serve as input to GNN models later:

.. code-block:: python

   model = HeteroGNN(...)

   output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

If the dataset exists in the `list of Pytorch Geometric datasets <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html>`_, it can directly be imported and used.
In particular, it will be downloaded to :obj:`root` and processed automatically.

.. code-block:: python

   from torch_geometric.datasets import OGB_MAG

   dataset = OGB_MAG(root='./data', preprocess='metapath2vec')
   data = dataset[0]

The :obj:`data` object can be printed for verification.

.. code-block:: text

   HeteroData(
      paper={
         x=[736389, 256],
         y=[736389],
         train_mask=[736389],
         val_mask=[736389],
         test_mask=[736389]
      },
      author={ x=[1134649, 128] },
      institution={ x=[8740, 128] },
      field_of_study={ x=[59965, 128] },
      (author, affiliated_with, institution)={ edge_index=[2, 1043998] },
      (author, writes, paper)={ edge_index=[2, 7145660] },
      (paper, cites, paper)={ edge_index=[2, 5416271] },
      (paper, has_topic, field_of_study)={ edge_index=[2, 7505078] }
   )

.. note::

   The original `ogbn-mag <https://ogb.stanford.edu/docs/nodeprop>`__ network does only provide features for "paper" nodes.
   In :class:`~torch_geometric.datasets.OGB_MAG`, we provide the option to download a processed version of it in which structural features (obtained from either :obj:`"metapath2vec"` or :obj:`"TransE"`) are added to featureless nodes, as it is commonly done in the top ranked submissions to the `OGB leaderboards <https://ogb.stanford.edu/docs/leader_nodeprop>`_.

Utility Functions
~~~~~~~~~~~~~~~~~

The :class:`torch_geometric.data.HeteroData` class provides a number of useful utility functions to modify and analyze the given graph.

For example, single node or edge stores can be indiviually indexed:

.. code-block:: python

   paper_node_data = data['paper']
   cites_edge_data = data['paper', 'cites', 'paper']

In case the edge type can be uniquely identified by only the pair of source and destination node types or the edge type, the following operations work as well:

.. code-block:: python

   cites_edge_data = data['paper', 'paper']
   cites_edge_data = data['cites']

We can add new node types or tensors and remove them:

.. code-block:: python

   data['paper'].year = ...    # Setting a new paper attribute
   del data['field_of_study']  # Deleting 'field_of_study' node type
   del data['has_topic']       # Deleting 'has_topic' edge type

We can access the meta-data of the :obj:`data` object, holding information of all present node and edge types:

.. code-block:: python

   node_types, edge_types = data.metadata()
   print(node_types)
   >>> ['paper', 'author', 'institution']
   print(edge_types)
   >>> [('paper', 'cites', 'paper'),
        ('author', 'writes', 'paper'),
        ('author', 'affiliated_with', 'institution')]

The :obj:`data` object can be transferred between devices as usual:

.. code-block:: python

   data = data.to('cuda:0')
   data = data.cpu()

We further have access to additional helper functions to analyze the given graph:

.. code-block:: python

   data.has_isolated_nodes()
   data.has_self_loops()
   data.is_undirected()

Heterogeneous Graph Transformations
-----------------------------------

Most `transformations <https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html>`_ for preprocessing regular graphs work as well on the heterogeneous graph :obj:`data` object.

.. code-block:: python

   import torch_geometric.transforms as T

   data = T.ToUndirected()(data)
   data = T.AddSelfLoops()(data)
   data = T.NormalizeFeatures()(data)

Here, :meth:`~torch_geometric.transforms.ToUndirected` transforms a directed graph into (the PyG representation of) an undirected graph, by adding reverse edges for all edges in the graph.
Thus, future message passing is performed in both direction of all edges.
The function may add reverse edge types to the heterogeneous graph, if necessary.

For all nodes of type :obj:`'node_type'` and all existing edge types of the form :obj:`('node_type', 'edge_type', 'node_type')`, the function :meth:`~torch_geometric.transforms.AddSelfLoops` will add self-loop edges.
As a result, each node might receive one or more (one per appropriate edge type) messages from itself during message passing.

The transform :meth:`~torch_geometric.transforms.NormalizeFeatures` works like in the homogenous case, and normalizes all specified features (of all types) to sum up to one.

Creating Heterogeneous GNNs
---------------------------

Standard Message Passing GNNs (MP-GNNs) can not trivially be applied to heterogenous graph data, as node and edge features from different types can not be processed by the same functions due to differences in feature type.
A natural way to circumvent this is to implement message and update functions individually for each edge type.
During runtime, the MP-GNN algorithm would need to iterate over edge type dictionaries during message computation and over node type dictionaries during node updates.

To avoid unnecessary runtime overheads and to make the creation of heterogeneous MP-GNNs as simple as possible, Pytorch Geometric provides three ways for the user to create models on heterogeneous graph data:

#. Automatically convert a homogenous GNN model to a heterogeneous GNN model by making use of :meth:`torch_geometric.nn.to_hetero` or :meth:`torch_geometric.nn.to_hetero_with_bases`
#. Define inidividual functions for different types using PyGs wrapper :class:`torch_geometric.nn.conv.HeteroConv` for heterogeneous convolution
#. Deploy existing (or write your own) heterogeneous GNN operators

In the following, each option is introduced in detail.

Automatically Converting GNN Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pytorch Geometric allows to automatically convert any PyG GNN model to a model for heterogeneous input graphs, using the built in functions :meth:`torch_geometric.nn.to_hetero` or :meth:`torch_geometric.nn.to_hetero_with_bases`.
The following `example <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py>`__ shows how to apply it:

.. code-block:: python

   from torch_geometric.nn import SAGEConv, to_hetero

   class GNN(torch.nn.Module):
       def __init__(in_channels, hidden_channels, out_channels):
           super().__init__()
           self.conv1 = SAGEConv(in_channels, hidden_channels)
           self.conv2 = SAGEConv(hidden_channels, out_channels)

       def forward(self, x, edge_index):
           x = self.conv1(x, edge_index).relu()
           x = self.conv2(x, edge_index)
           return x


   model = GNN(in_channels=-1, hidden_channels=64, out_channels=dataset.num_classes)
   model = to_hetero(model, data.metadata(), aggr='sum')

The process takes an existing GNN model and duplicates the message functions to work on each edge type individually, as detailed in the following figure.

.. image:: ../_figures/to_hetero.svg
   :align: center
   :width: 90%

As a result, the model now expects dictionaries with node and edge types as keys as input arguments, rather than single tensors utilized in homogeneous graphs.

.. _lazyinit:

.. note::
   Since the number of input features and thus the size of tensors varies between different types, PyG can make use of **lazy initialization** to initialize parameters in heterogeneous GNNs (as denoted by :obj:`-1` as the :obj:`in_channels` argument).
   This allows us to avoid calculating and keeping track of all tensor sizes of the computation graph.
   Lazy initialization is supported for all existing PyG operators.
   We can initialize the model's parameters by calling it once:

   .. code-block:: python

      with torch.no_grad():  # Initialize lazy modules.
          out = model(data.x_dict, data.edge_index_dict)

Afterwards, the created model can be trained as usual:

.. _trainfunc:

.. code-block:: python

   def train():
       model.train()
       optimizer.zero_grad()
       out = model(data.x_dict, data.edge_index_dict)
       mask = data['author'].train_mask
       loss = F.cross_entropy(out[mask], data['author'].y[mask])
       loss.backward()
       optimizer.step()
       return float(loss)

Using the Heterogenous Convolution Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The heterogenous convolution wrapper :class:`torch_geometric.nn.conv.HeteroConv` allows to define custom heterogenous message and update functions to build arbitrary MP-GNNs for heterogeneous graphs from scratch.
While the automatic converter :meth:`~torch_geometric.nn.to_hetero` uses the same operator for all edge types, the wrapper allows to define different operators for different edge types.
Here, :class:`~torch_geometric.nn.conv.HeteroConv` takes a dictionary of submodules as input, one for each edge type in the graph data.
The following `example <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/hetero/hetero_conv_dblp.py>`__ shows how to apply it.

.. code-block:: python

   from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear

   class HeteroGNN(torch.nn.Module):
       def __init__(self, hidden_channels, out_channels, num_layers):
           super().__init__()

           self.convs = torch.nn.ModuleList()
           for _ in range(num_layers):
               conv = HeteroConv({
                   ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels)
                   ('author', 'writes', 'paper'): GATConv((-1, -1), hidden_channels)
                   ('author', 'affiliated_with', 'institution'): SAGEConv((-1, -1), hidden_channels)
               }, aggr='sum')
               self.convs.append(conv)

           self.lin = Linear(hidden_channels, out_channels)

      def forward(self, x_dict, edge_index_dict):
         for conv in self.convs:
             x_dict = conv(x_dict, edge_index_dict)
             x_dict = {key: x.relu() for key, x in x_dict.items()}
         return self.lin(x_dict['author'])

   model = HeteroGNN(hidden_channels=64, out_channels=dataset.num_classes,
                     num_layers=2)

We can initialize the model by calling it once (see :ref:`here<lazyinit>` for more details about lazy initialization)

.. code-block:: python

   with torch.no_grad():  # Initialize lazy modules.
       out = model(data.x_dict, data.edge_index_dict)

and run the standard training procedure as outlined :ref:`here<trainfunc>`.

Deploy Existing Heterogenous Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyG provides operators (*e.g.*, :class:`torch_geometric.nn.conv.HGTConv`), which are specifically designed for heterogeneous graphs.
These operators can be directly used to build heterogeneous GNN models as can be seen in the following `example <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/hetero/hgt_dblp.py>`__:

.. code-block:: python

   from torch_geometric.nn import HGTConv

   class HGT(torch.nn.Module):
       def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
           super().__init__()

           self.lin_dict = torch.nn.ModuleDict()
           for node_type in data.node_types:
               self.lin_dict[node_type] = Linear(-1, hidden_channels)

           self.convs = torch.nn.ModuleList()
           for _ in range(num_layers):
               conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                              num_heads, group='sum')
               self.convs.append(conv)

           self.lin = Linear(hidden_channels, out_channels)

       def forward(self, x_dict, edge_index_dict):
           x_dict = {key: lin(x).relu() for key, lin in self.lin_dict.items()}
           for conv in self.convs:
               x_dict = conv(x_dict, edge_index_dict)
           return self.lin(x_dict['author'])

   model = HGT(hidden_channels=64, out_channels=dataset.num_classes,
               num_heads=2, num_layers=2)

We can initialize the model by calling it once (see :ref:`here<lazyinit>` for more details about lazy initialization).

.. code-block:: python

   with torch.no_grad():  # Initialize lazy modules.
      out = model(data.x_dict, data.edge_index_dict)

and run the standard training procedure as outlined :ref:`here<trainfunc>`.

Heterogeneous Graph Samplers
----------------------------

PyG provides functionality for neighbor sampling on heterogeneous graphs in the standard :class:`torch_geometric.loader.NeighborLoader` class as well as in other samplers such as via :class:`torch_geometric.loader.HGTLoader`.
This is especially useful for efficient representation learning on large heterogeneous graphs, where processing the full number of neighbors is too computationally expensive.

Performing neighbor sampling using :class:`~torch_geometric.loader.NeighborLoader` works as outlined in the following `example <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py>`__:

.. code-block:: python

   from torch_geometric.loader import NeighborLoader

   train_loader = NeighborLoader(
       data,
       # Sample 15 neighbors for each node and each edge type for 2 iterations:
       num_neighbors=[15] * 2
       # Use a batch size of 128 for sampling training nodes of type "paper":
       batch_size=128,
       input_nodes: ('paper', data['paper'].train_mask),
   )

   sampled_data = next(iter(train_loader))

In contrast to sampling on homogenous graphs, we simply provide a heterogeneous :obj:`data` object instead of a homogeneous one.
Using the :obj:`input_nodes` parameter, we specify the type and indices of nodes, from which we want to sample local neighborhoods.
