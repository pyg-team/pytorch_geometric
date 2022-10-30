torch_geometric.explain
========================



.. contents:: Contents
    :local:



Philoshopy
----------

The ref::`torch_geometric.explain` module provides a set of tools to explain the
predictions of a :class:`torch_geometric.nn.model`  or to explain the underlying
phenomenon of a dataset (see `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks"
<https://arxiv.org/abs/2206.09677>`_  for more details).

We represent explanations using the :class:`Explanation` class, which is a `Data` object containing masks for the nodes, edges, features and any attributes of the data.

The :class:`~torch_geometric.explain.Explainer` class is designed to habdle all the explainability parameters:

-  which algorithm from the :class:`~torch_geometric.explain.algorithm` module to use (e.g. :class:`~torch_geometric.explain.algorithm.GNNExplainer` or :class:`~torch_geometric.explain.algorithm.RandomExplainer`).
-  the :obj:`mask` type (e.g. :obj:`mask="node"` or :obj:`mask="edge"`).
-  any postprocessing of the masks (e.g. :obj:`threshold = "topk"` or :obj:`threshold = "hard"`).

This class allows the user to easily compare different explainability methods and to easily switch between different types of masks, while making sure the
high level framework stays the same.



Explainer
---------

.. currentmodule:: torch_geometric.explain

.. autoclass:: torch_geometric.explain.explainer.Explainer
   :members:


Explanations
------------

.. currentmodule:: torch_geometric.explain

.. autoclass:: torch_geometric.explain.explanations.Explanation
   :members:


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
