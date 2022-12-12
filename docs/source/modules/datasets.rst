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
