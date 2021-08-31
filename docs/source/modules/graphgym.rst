torch_geometric.graphgym
========================

.. contents:: Contents
    :local:


GraphGym workflow modules
-------------------------

.. currentmodule:: torch_geometric.graphgym
.. autosummary::
   :nosignatures:
    {% for cls in torch_geometric.graphgym.classes %}
     {{ cls }}
    {% endfor %}


.. automodule:: torch_geometric.graphgym
    :members:
    :exclude-members:


GraphGym model modules
----------------------

.. currentmodule:: torch_geometric.graphgym.models
.. autosummary::
    :nosignatures:
    {% for cls in torch_geometric.graphgym.models.classes %}
      {{ cls }}
    {% endfor %}

.. automodule:: torch_geometric.graphgym.models
    :members:
    :exclude-members: forward



GraphGym register modules
-------------------------

.. currentmodule:: torch_geometric.graphgym
.. autosummary::
   :nosignatures:
    {% for cls in torch_geometric.graphgym.classes_reg %}
     {{ cls }}
    {% endfor %}

   {% for cls in torch_geometric.graphgym.classes_reg %}
.. autofunction:: {{ cls }}
   {% endfor %}
