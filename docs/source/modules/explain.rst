torch_geometric.explain
=======================

.. currentmodule:: torch_geometric.explain

.. warning::

    This module is in active development and may not be stable.
    Access requires installing PyTorch Geometric from master.

.. contents:: Contents
    :local:

Philoshopy
----------

This module provides a set of tools to explain the predictions of a PyG model or to explain the underlying phenomenon of a dataset (see the `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper for more details).

We represent explanations using the :class:`torch_geometric.explain.Explanation` class, which is a :class:`~torch_geometric.data.Data` object containing masks for the nodes, edges, features and any attributes of the data.

The :class:`torch_geometric.explain.Explainer` class is designed to handle all explainability parameters:

- which algorithm from the :class:`torch_geometric.explain.algorithm` module to use (*e.g.*, :class:`~torch_geometric.explain.algorithm.GNNExplainer`)
- the different type of masks for node and edges (*e.g.*, :obj:`mask="object"` or :obj:`mask="attributes"`)
- any postprocessing of the masks (*e.g.*, :obj:`threshold="topk"` or :obj:`threshold="hard"`)

This class allows the user to easily compare different explainability methods and to easily switch between different types of masks, while making sure the high-level framework stays the same.

Explainer
---------

.. autoclass:: torch_geometric.explain.Explainer
   :members:

   .. automethod:: __call__

.. autoclass:: torch_geometric.explain.config.ExplainerConfig
   :members:

.. autoclass:: torch_geometric.explain.config.ModelConfig
   :members:

.. autoclass:: torch_geometric.explain.config.ThresholdConfig
   :members:

Explanations
------------

.. autoclass:: torch_geometric.explain.Explanation
   :members:

Explainer Algorithms
--------------------

.. currentmodule:: torch_geometric.explain.algorithm

.. autosummary::
   :nosignatures:
   {% for cls in torch_geometric.explain.algorithm.classes %}
     {{ cls }}
   {% endfor %}

.. autoclass:: torch_geometric.explain.algorithm.ExplainerAlgorithm
   :members:

.. automodule:: torch_geometric.explain.algorithm
   :members:
   :exclude-members: ExplainerAlgorithm, forward, loss, supports
