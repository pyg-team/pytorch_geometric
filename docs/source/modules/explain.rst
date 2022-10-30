torch_geometric.explain
========================

.. contents:: Contents
    :local:


Explainer
---------

.. currentmodule:: torch_geometric.explain

.. autoclass:: torch_geometric.explain.explainer.Explainer


Explanations
------------

.. currentmodule:: torch_geometric.explain

.. autoclass:: torch_geometric.explain.explanations.Explanation


Explanations algorithms
-----------------------

.. currentmodule:: torch_geometric.explain.algorithm
.. autosummary::
   :nosignatures:
   {% for cls in torch_geometric.explain.algorithm.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: torch_geometric.explain.algorithm
   :members:
   :exclude-members:
