torch_geometric.explain
=======================

.. currentmodule:: torch_geometric.explain

.. warning::

    This module is in active development and may not be stable.
    Access requires installing :pyg:`PyG` from master.

.. contents:: Contents
    :local:

Philosophy
----------

This module provides a set of tools to explain the predictions of a PyG model or to explain the underlying phenomenon of a dataset (see the `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper for more details).

We represent explanations using the :class:`torch_geometric.explain.Explanation` class, which is a :class:`~torch_geometric.data.Data` object containing masks for the nodes, edges, features and any attributes of the data.

The :class:`torch_geometric.explain.Explainer` class is designed to handle all explainability parameters (see the :class:`torch_geometric.explain.config.ExplainerConfig` class for more details):

- which algorithm from the :class:`torch_geometric.explain.algorithm` module to use (*e.g.*, :class:`~torch_geometric.explain.algorithm.GNNExplainer`)
- the type of explanation to compute (*e.g.*, :obj:`explanation_type="phenomenon"` or :obj:`explanation_type="model"`)
- the different type of masks for node and edges (*e.g.*, :obj:`mask="object"` or :obj:`mask="attributes"`)
- any postprocessing of the masks (*e.g.*, :obj:`threshold_type="topk"` or :obj:`threshold_type="hard"`)

This class allows the user to easily compare different explainability methods and to easily switch between different types of masks, while making sure the high-level framework stays the same.

Explainer
---------

.. autoclass:: torch_geometric.explain.Explainer
   :show-inheritance:
   :members:
   :special-members: __call__

.. autoclass:: torch_geometric.explain.config.ExplainerConfig
   :members:

.. autoclass:: torch_geometric.explain.config.ModelConfig
   :members:

.. autoclass:: torch_geometric.explain.config.ThresholdConfig
   :members:

Explanations
------------

.. autoclass:: torch_geometric.explain.Explanation
   :show-inheritance:
   :members:

.. autoclass:: torch_geometric.explain.HeteroExplanation
   :show-inheritance:
   :members:

Explainer Algorithms
--------------------

.. currentmodule:: torch_geometric.explain.algorithm

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.explain.algorithm.classes %}
     {{ name }}
   {% endfor %}

Explanation Metrics
-------------------

The quality of an explanation can be judged by a variety of different methods.
PyG supports the following metrics out-of-the-box:

.. currentmodule:: torch_geometric.explain.metric

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   {% for name in torch_geometric.explain.metric.classes %}
     {{ name }}
   {% endfor %}
