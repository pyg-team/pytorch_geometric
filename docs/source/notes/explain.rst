GNN Explainability
===================================

Interpretting GNN models is crucial for many use cases. PyG (2.2 and beyond) includes:
#. Flexible interface to generate a variety of explanations :class:`~torch_geometric.explain.Explainer`.
#. Several explanation algorithms like :class:`~torch_geometric.explain.algorithm.GNNExplainer`, :class:`~torch_geometric.explain.algorithm.AttentionExplainer` etc.
#. Support to visualize explanations like .
#. Metrics to evalaute explanations like .

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

Examples
--------

**Example 1 : Explaining node prediciton on a homogenous graph.

Assume you have a GNN `model` that does node classification on homogenous `data`. You can get `Explanation` with a edge mask and node feature mask, that explains a nodes prediction as follows:

.. code-block:: python
    from torch_geometric.explain import Explainer, GNNExplainer, Explanation
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs', # model returns log of probability
        ),
    )

    # Generate explanation for node 10s prediction.
    node_index = 10
    explanation: Explanation = explainer(data.x, data.edge_index, index=node_index)

Visulaize the generate explanation(node mask, edge mask):

.. code-block:: python
    path = 'feature_importance.png'
    explanation.visualize_feature_importance(path, top_k=10)
    path = 'subgraph.pdf'
    explanation.visualize_graph(path)

Sometimes we want to measure how good an explanation is

.. code-block:: python
    from torch_geometric.explain.metrics import unfaithfulness
    metric = unfaithfulness(explainer, explanation)

**Example 2 : Explaining node prediciton on a heterogenous graph.

Since this feature is still undergoing heavy development, please feel free to reach out to the PyG core team either on `GitHub <https://github.com/pyg-team/pytorch_geometric/discussions>`_ or `Slack <https://data.pyg.org/slack.html>`_ if you have any questions, comments or concerns.
