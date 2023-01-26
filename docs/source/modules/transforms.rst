torch_geometric.transforms
==========================

.. contents:: Contents
    :local:

Transforms are a general way to modify and customize :class:`~torch_geometric.data.Data` or :class:`~torch_geometric.data.HeteroData` objects, either by implicitly passing them as an argument to a :class:`~torch_geometric.data.Dataset`, or by applying them explicitly to individual :class:`~torch_geometric.data.Data` or :class:`~torch_geometric.data.HeteroData` objects:

.. code-block:: python

   import torch_geometric.transforms as T
   from torch_geometric.datasets import TUDataset

   transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

   dataset = TUDataset(path, name='MUTAG', transform=transform)
   data = dataset[0]  # Implicitly transform data on every access.

   data = TUDataset(path, name='MUTAG')[0]
   data = transform(data)  # Explicitly transform data.

General Transforms
------------------

.. currentmodule:: torch_geometric.transforms

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_geometric.transforms.general_transforms %}
     {{ name }}
   {% endfor %}

Graph Transforms
----------------

.. currentmodule:: torch_geometric.transforms

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_geometric.transforms.graph_transforms %}
     {{ name }}
   {% endfor %}

Vision Transforms
-----------------

.. currentmodule:: torch_geometric.transforms

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_geometric.transforms.vision_transforms %}
     {{ name }}
   {% endfor %}
