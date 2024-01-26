# Examples for Generating Explanations of Graph Neural Networks

This directory contains examples demonstrating the use of `torch_geometric.explain` package. The `explain` package of PyTorch Geometric provides a set of tools to explain the predictions of a PyG model or to explain the underlying phenomenon of a dataset.

| Example                                                                | Description                                                                                                             |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [`captum_explainer.py`](./captum_explainer.py)                         | Demonstrates the use of Captum-based explainer for explaining Graph Neural Networks (GNN) model in node classification. |
| [`captum_explainer_hetero_link.py`](./captum_explainer_hetero_link.py) | Demonstrates the use of Captum-based explainer for explaining GNN model in heterogenous link prediction.                |
| [`gnn_explainer.py`](./gnn_explainer.py)                               | Shows GNNExplainer applied to the Cora dataset for understanding GNN model.                                             |
| [`gnn_explainer_ba_shapes.py`](./gnn_explainer_ba_shapes.py)           | Shows GNNExplainer applied to the BA-Shapes dataset for understanding GNN model.                                        |
| [`gnn_explainer_link_pred.py`](./gnn_explainer_link_pred.py)           | Demonstrates the use of GNNExplainer for explaining GNN model in link prediction.                                       |
| [`graphmask_explainer.py`](./graphmask_explainer.py)                   | Demonstrates the use of GraphMask-Explainer for explaining GNN model in node classification.                            |
