Point Cloud Classification with Graph Neural Networks
================================

This tutorial guides you through how we can use Graph Neural Networks (GNNs) to perform node classification on point clouds, and introduces the PyG layers and pipelines for this task. In point cloud classification and segmentation, we have a dataset of objects or point sets, and we want to embed those objects in such a way so that they are linearly separable given a task at hand. We can do this through graph neural networks (GNN), in which the raw point cloud is used as input into GNN and the training process will help us to capture meaningful local structures in order to classify the entire point set.

3D Point Cloud Dataset
---------------------
PyG provides several point cloud datasets from the dataset module, such as the `PCPNetDataset <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.PCPNetDataset.html#torch_geometric.datasets.PCPNetDataset>`_ , `S3DIS <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.S3DIS.html#torch_geometric.datasets.S3DIS>`_ and `GeometricShapes <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GeometricShapes.html#torch_geometric.datasets.GeometricShapes>`_ datasets. For simplicity, in this tutorial, we will use the Geometric Shapes dataset, which is a synthetic dataset of various geometric shapes like cubes, spheres or pyramids.

PointNet++
-------------------------
`PointNet++ <https://arxiv.org/abs/1706.02413>`_ is a pioneering work that proposes a graph neural network architecture for point cloud classification and segmentation.

PointNet++ processes point clouds iteratively by following a simple grouping, neighborhood aggregation and downsampling scheme:

1. The grouping phase constructs a graph in which nearby points are connected. Typically, this is either done via  k-nearest neighbor search or via ball queries (which connects all points that are within a radius to the query point).

2. The neighborhood aggregation phase executes a Graph Neural Network layer that, for each point, aggregates information from its direct neighbors (given by the graph constructed in the previous phase). This allows PointNet++ to capture local context at different scales.

3. The downsampling phase implements a pooling scheme suitable for point clouds with potentially different sizes.

In the following code, we will reconstruct PointNet++ using :pyg:`PyG` layers to illustrate how you can reuse the functionality to build new models for point cloud classification and segmentation

Phase 1: Grouping via Dynamic Graph Generation
-------------------------------------------
PyTorch Geometric provides utilities for dynamic graph generation via its helper package `torch_cluster <https://github.com/rusty1s/pytorch_cluster>`_, in particular via the knn_graph and radius_graph functions for  k -nearest neighbor and ball query graph generation, respectively.


.. code-block:: python

    import torch
    from torch.nn import Linear, Parameter
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, degree

Some more notes
