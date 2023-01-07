GNN Explainability
===================================

Interpretting GNN models is crucial for many use cases. PyG (2.2 and beyond) includes:
1. Flexible interface to generate a variety of explanations :class:`~torch_geometric.explain.Explainer`.
2. Several explanation algorithms like :class:`~torch_geometric.explain.algorithm.GNNExplainer`, :class:`~torch_geometric.explain.algorithm.AttentionExplainer` etc.
3. Support to visualize explanations like .
4. Metrics to evalaute explanations like .

.. warning::

    The explanation APIs discussed here may change in the future as we continuously work to improve their ease-of-use and generalizability.

Background
----------


Explainer Interface and Explaienr Algorithms
---------------------------------------------

The :class:`~torch_geometric.explain.Explainer` class is designed to handle all explainability parameters (see the :class:`~torch_geometric.explain.config.ExplainerConfig` class for more details):

which algorithm from the torch_geometric.explain.algorithm module to use (e.g., GNNExplainer)

the type of explanation to compute (e.g., explanation_type="phenomenon" or explanation_type="model")

the different type of masks for node and edges (e.g., mask="object" or mask="attributes")

any postprocessing of the masks (e.g., threshold_type="topk" or threshold_type="hard")

Metrics and Visualization
--------------------------

Putting it All Together
-----------------------



Since this feature is still undergoing heavy development, please feel free to reach out to the PyG core team either on `GitHub <https://github.com/pyg-team/pytorch_geometric/discussions>`_ or `Slack <https://data.pyg.org/slack.html>`_ if you have any questions, comments or concerns.
