torch_geometric.datasets
========================

Benchmark Datasets
------------------

.. currentmodule:: torch_geometric.datasets

.. autosummary::
    :nosignatures:
    {% for cls in torch_geometric.datasets.classes %}
      {{ cls }}
    {% endfor %}

.. automodule:: torch_geometric.datasets
    :members:
    :exclude-members: download, process, processed_file_names, raw_file_names, num_classes, get

Graph Generators
----------------

.. currentmodule:: torch_geometric.datasets.graph_generator

.. autosummary::
   :nosignatures:
   {% for cls in torch_geometric.datasets.graph_generator.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: torch_geometric.datasets.graph_generator
    :members:

Motif Generators
----------------

.. currentmodule:: torch_geometric.datasets.motif_generator

.. autosummary::
   :nosignatures:
   {% for cls in torch_geometric.datasets.motif_generator.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: torch_geometric.datasets.motif_generator
    :members:

Pandas DataFrame To HeteroData
------------------------------

.. currentmodule:: torch_geometric.datasets.pandas_heterodata

Where :class:`~torch_geometric.typing.HeteroDF` is a dictionary with keys:
    :class:`~torch_geometric.typing.NodeType` (str):
        keys are str that defines node types
        values are pd.Dataframes with at least an "id" column and other feature columns.
    :class:`~torch_geometric.typing.EdgeType` (Tuple[str,str,str]):
        Keys are EdgeType of the form (sourceType,edgeType,targetType)
        values are pd.Dataframes with source and target columns that contain the node ids
        that define the edge and other feature columns.

.. code-block:: python

    NodeType = str
    EdgeType = Tuple[str, str, str] # (source_type,edge_type,target_type)
    HeteroDF = Dict[Union[NodeType, EdgeType], pandas.DataFrame]

.. autosummary::
   :nosignatures:
   {% for cls in torch_geometric.datasets.pandas_heterodata.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: torch_geometric.datasets.pandas_heterodata
    :members:
