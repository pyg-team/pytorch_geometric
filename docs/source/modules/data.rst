torch_geometric.data
====================

.. contents:: Contents
    :local:

Data Objects
------------

.. currentmodule:: torch_geometric.data

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.data.data_classes %}
     {{ name }}
   {% endfor %}

Remote Backend Interfaces
-------------------------

.. currentmodule:: torch_geometric.data

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.data.remote_backend_classes %}
     {{ name }}
   {% endfor %}

PyTorch Lightning Wrappers
--------------------------

.. currentmodule:: torch_geometric.data.lightning

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for name in torch_geometric.data.lightning.classes %}
     {{ name }}
   {% endfor %}

Helper Functions
----------------

.. currentmodule:: torch_geometric.data

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.data.helper_functions %}
     {{ name }}
   {% endfor %}
