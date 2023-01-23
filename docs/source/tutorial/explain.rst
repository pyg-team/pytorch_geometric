GNN Explainability
==================

Interpreting GNN models is crucial for many use cases.
:pyg:`PyG` (2.3 and beyond) provides the :class:`torch_geometric.explain` package for first-class GNN explainability support that currently includes

#. a flexible interface to generate a variety of explanations via the :class:`~torch_geometric.explain.Explainer` class,

#. several underlying explanation algorithms including, *e.g.*, :class:`~torch_geometric.explain.algorithm.GNNExplainer`,  :class:`~torch_geometric.explain.algorithm.PGExplainer` and :class:`~torch_geometric.explain.algorithm.CaptumExplainer`,

#. support to visualize explanations via the :class:`~torch_geometric.explain.Explanation` or the :class:`~torch_geometric.explain.HeteroExplanation` class,

#. and metrics to evaluate explanations via the :class:`~torch_geometric.explain.metric` package.

.. warning::

    The explanation APIs discussed here may change in the future as we continuously work to improve their ease-of-use and generalizability.

Explainer Interface
-------------------

The :class:`torch_geometric.explain.Explainer` class is designed to handle all explainability parameters (see the :class:`~torch_geometric.explain.config.ExplainerConfig` class for more details):

#. which algorithm from the :class:`torch_geometric.explain.algorithm` module to use (*e.g.*, :class:`~torch_geometric.explain.algorithm.GNNExplainer`)

#. the type of explanation to compute, *i.e.* :obj:`explanation_type="phenomenon"` to explain the underlying phenomenon of a dataset, and :obj:`explanation_type="model"` to explain the prediction of a GNN model (see the `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper for more details).

#. the different type of masks for node and edges (*e.g.*, :obj:`mask="object"` or :obj:`mask="attributes"`)

#. any postprocessing of the masks (*e.g.*, :obj:`threshold_type="topk"` or :obj:`threshold_type="hard"`)

This class allows the user to easily compare different explainability methods and to easily switch between different types of masks, while making sure the high-level framework stays the same.
The :class:`~torch_geometric.explain.Explainer` generates an :class:`~torch_geometric.explain.Explanation` or :class:`~torch_geometric.explain.HeteroExplanation` object which contains the final information about which nodes, edges and features are crucial to explain a GNN model.

Examples
--------

Explaining node classification on a homogeneous graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume we have a GNN :obj:`model` that does node classification on a homogeneous graph.
We can use the :class:`torch_geometric.explain.algorithm.GNNExplainer` algorithm to generate an :class:`~torch_geometric.explain.Explanation`.
We configure the :class:`~torch_geometric.explain.Explainer` to use both a :obj:`node_mask_type` and an :obj:`edge_mask_type` such that the final :class:`~torch_geometric.explain.Explanation` object contains (1) a :obj:`node_mask` (indicating which nodes and features are crucial for prediction), and (2) an :obj:`edge_mask` (indicating which edges are crucial for prediction).

.. code-block:: python

    from torch_geometric.data import Data
    from torch_geometric.explain import Explainer, GNNExplainer

    data = Data(...)  # A homogeneous graph data object.

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',  # Model returns log probabilities.
        ),
    )

    # Generate explanation for the node at index `10`:
    explanation = explainer(data.x, data.edge_index, index=10)
    print(explanation.edge_mask)
    print(explanation.node_mask)

Finally, we can visualize both feature importance and the crucial subgraph of the explanation:

.. code-block:: python

    explanation.visualize_feature_importance(top_k=10)

    explanation.visualize_graph()

To evaluate the explanation from the :class:`~torch_geometric.explain.algorithm.GNNExplainer`, we can utilize the :class:`torch_geometric.explain.metric` module.
For example, to compute the :meth:`~torch_geometric.explain.metric.unfaithfulness` of an explanation, run:

.. code-block:: python

    from torch_geometric.explain import unfaithfulness

    metric = unfaithfulness(explainer, explanation)
    print(metric)

Explaining node classification on a heterogeneous graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume we have a heterogeneous GNN :obj:`model` that does node classification on a heterogeneous graph.
We can use the :class:`IntegratedGradient` attribution method from `Captum <https://captum.ai/docs/extension/integrated_gradients>`__ via the :class:`torch_geometric.explain.algorithm.CaptumExplainer` algorithm to generate a :class:`~torch_geometric.explain.HeteroExplanation`.

.. note::
    :class:`~torch_geometric.explain.algorithm.CaptumExplainer` is a wrapper around the `Captum <https://captum.ai>`__ library with support for most of attribution methods to explain *any* homogeneous or heterogeneous :pyg:`PyG` model.

We configure the :class:`~torch_geometric.explain.Explainer` to use both a :obj:`node_mask_type` and an :obj:`edge_mask_type` such that the final :class:`~torch_geometric.explain.HeteroExplanation` object contains (1) a :obj:`node_mask` for *each* node type (indicating which nodes and features for each node type are crucial for prediction), and (2) an :obj:`edge_mask` for *each* edge type (indicating which edges for each edge type are crucial for prediction).

.. code-block:: python

    from torch_geometric.data import HeteroData
    from torch_geometric.explain import Explainer, CaptumExplainer

    hetero_data = HeteroData(...)  # A heterogeneous graph data object.

    explainer = Explainer(
        model,  # It is assumed that model outputs a single tensor.
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config = dict(
            mode='multiclass_classification',
            task_level=task_level,
            return_type='probs',  # Model returns probabilities.
        ),
    )

    # Generate batch-wise heterogeneous explanations for
    # the nodes at index `1` and `3`:
    hetero_explanation = explainer(
        hetero_data.x_dict,
        hetero_data.edge_index_dict,
        index=torch.tensor([1, 3]),
    )
    print(hetero_explanation.edge_mask_dict)
    print(hetero_explanation.node_mask_dict)

Explaining graph regression on a homogeneous graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume we have a GNN :obj:`model` that does graph regression on a homogeneous graph.
We can use the :class:`torch_geometric.explain.algorithm.PGExplainer` algorithm to generate an :class:`~torch_geometric.explain.Explanation`.
We configure the :class:`~torch_geometric.explain.Explainer` to use an :obj:`edge_mask_type` such that the final :class:`~torch_geometric.explain.Explanation` object contains an :obj:`edge_mask` (indicating which edges are crucial for prediction).
Importantly, passing a :obj:`node_mask_type` to the :class:`~torch_geometric.explain.Explainer` will throw an error since :class:`~torch_geometric.explain.algorithm.PGExplainer` cannot explain the importance of nodes:

.. code-block:: python

    from torch_geometric.data import Data
    from torch_geometric.explain import Explainer, PGExplainer

    dataset = ...
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30, lr=0.003),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
        # Include only the top 10 most important edges:
        threshold_config=dict(type='topk', value=10),
    )

    # PGExplainer needs to be trained separately since it is a parametric
    # explainer i.e it uses a neural network to generate explanations:
    for epoch in range(30):
        for batch in loader:
            loss = explainer.algorithm.train(
                epoch, model, batch.x, batch.edge_index, target=batch.target)

    # Generate the explanation for a particular graph:
    explanation = explainer(dataset[0].x, dataset[0].edge_index)
    print(explanation.edge_mask)

Since this feature is still undergoing heavy development, please feel free to reach out to the :pyg:`PyG` core team either on `GitHub <https://github.com/pyg-team/pytorch_geometric/discussions>`_ or `Slack <https://data.pyg.org/slack.html>`_ if you have any questions, comments or concerns.
