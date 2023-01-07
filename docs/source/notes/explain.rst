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


Explainer Interface
-------------------

The :class:`~torch_geometric.explain.Explainer` class is designed to handle all explainability parameters (see the :class:`~torch_geometric.explain.config.ExplainerConfig` class for more details):

#. which algorithm from the torch_geometric.explain.algorithm module to use (e.g., GNNExplainer)

#. the type of explanation to compute (e.g., explanation_type="phenomenon" or explanation_type="model")

#. the different type of masks for node and edges (e.g., mask="object" or mask="attributes")

#. any postprocessing of the masks (e.g., threshold_type="topk" or threshold_type="hard")

Metrics and Visualization
-------------------------



Examples
--------

**Example 1 : Explaining node classification on a homogenous graph.**

Assume we have a GNN `model` that does node classification on homogenous `data`. Lets use `GNNExplainer` to generate an `Explanation`. We configure the `Explainer` using `node_mask_type` and `edge_mask_type`, so that `Explanation` contains 1.) `edge_mask` indicating which edges are most important and 2.) a `node_mask` indicating which feature are crucial in explaining the nodes prediction

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
    print(explanation.edge_mask, explanation.node_mask)

To visulaize the explanation(node mask, edge mask):

.. code-block:: python

    path = 'feature_importance.png'
    explanation.visualize_feature_importance(path, top_k=10)
    path = 'subgraph.pdf'
    explanation.visualize_graph(path)

To evaluate the explanation from the `GNNExplainer`:

.. code-block:: python

    from torch_geometric.explain.metrics import unfaithfulness
    metric = unfaithfulness(explainer, explanation)

**Example 2 : Explaining graph regression on a homogenous graph.**

Assume we have a GNN `model` that does graph classification on homogenous `data`. Lets use `PGExplainer` to generate an `Explanation`. Since `PGExplainer` only explains which edges are crucial. We configure the `Explainer` using `node_mask_type` and `edge_mask_type`, so that `Explanation` contains only `edge_mask` indicating which edges are most important.

.. code-block:: python

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30, lr=0.003),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config = dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
            ),
        # Include only top 10 most important edges.
        threshold_config = ('top_k', 10)
    )

    # PGExplainer algorithm needs to be trained separately since its a
    # parametric explainer i.e it uses a neural network to generate explanation.
    for epoch in range(30):
        loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                         target=target)

    # Generate explanation for a particular graph.
    explanation: Explanation = explainer(data.x, data.edge_index)
    print(explanation.edge_mask)


Since this feature is still undergoing heavy development, please feel free to reach out to the PyG core team either on `GitHub <https://github.com/pyg-team/pytorch_geometric/discussions>`_ or `Slack <https://data.pyg.org/slack.html>`_ if you have any questions, comments or concerns.
