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

   {% for cls in torch_geometric.data.data_classes %}
     {{ cls }}
   {% endfor %}

Remote Backend Interfaces
-------------------------

.. currentmodule:: torch_geometric.data

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for cls in torch_geometric.data.remote_backend_classes %}
     {{ cls }}
   {% endfor %}

PyTorch Lightning Wrappers
--------------------------

.. currentmodule:: torch_geometric.data.lightning

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/only_class.rst

   {% for cls in torch_geometric.data.lightning.classes %}
     {{ cls }}
   {% endfor %}

Helper Functions
----------------

.. currentmodule:: torch_geometric.data

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for cls in torch_geometric.data.helper_functions %}
     {{ cls }}
   {% endfor %}
