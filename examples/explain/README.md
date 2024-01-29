# Examples for Generating Explanations of Graph Neural Networks

This directory contains examples demonstrating the use of the `torch_geometric.explain` package.
The `explain` package of PyG provides a set of tools to explain the predictions of a GNN model or to explain the underlying phenomenon of a dataset.

| Example                                                                | Description                                             |
| ---------------------------------------------------------------------- | ------------------------------------------------------- |
| [`gnn_explainer.py`](./gnn_explainer.py)                               | `GNNExplainer` for node classification                  |
| [`gnn_explainer_link_pred.py`](./gnn_explainer_link_pred.py)           | `GNNExplainer` for link prediction                      |
| [`gnn_explainer_ba_shapes.py`](./gnn_explainer_ba_shapes.py)           | `GNNExplainer` applied on the `BAShapes` dataset        |
| [`captum_explainer.py`](./captum_explainer.py)                         | Captum-based explainer for node classification          |
| [`captum_explainer_hetero_link.py`](./captum_explainer_hetero_link.py) | Captum-based explainer for heterogenous link prediction |
| [`graphmask_explainer.py`](./graphmask_explainer.py)                   | `GraphMaskExplainer` for node classification            |
