torch_geometric.datasets
========================

.. contents:: Contents
    :local:

Homogeneous Datasets
--------------------

.. currentmodule:: torch_geometric.datasets

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_geometric.datasets.homo_datasets %}
     {{ name }}
   {% endfor %}

Heterogeneous Datasets
----------------------

.. currentmodule:: torch_geometric.datasets

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_geometric.datasets.hetero_datasets %}
     {{ name }}
   {% endfor %}

Synthetic Datasets
------------------

.. currentmodule:: torch_geometric.datasets

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_geometric.datasets.synthetic_datasets %}
     {{ name }}
   {% endfor %}

Graph Generators
----------------

.. currentmodule:: torch_geometric.datasets.graph_generator

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_geometric.datasets.graph_generator.classes %}
     {{ name }}
   {% endfor %}

Motif Generators
----------------

.. currentmodule:: torch_geometric.datasets.motif_generator

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_geometric.datasets.motif_generator.classes %}
     {{ name }}
   {% endfor %}
