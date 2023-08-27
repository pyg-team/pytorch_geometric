Point Cloud Classification with Graph Neural Networks
================================

This tutorial guides you through how we can use Graph Neural Networks (GNNs) to perform node classification on point clouds, and introduces the PyG layers and pipelines for this task.


Model: PointNet++

Classification/ SEgmentation



Training pipelinlines

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



.. code-block:: python

    import torch
    from torch.nn import Linear, Parameter
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops, degree

Some more notes
