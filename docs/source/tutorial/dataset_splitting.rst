Dataset Splitting
=================

Dataset splitting is a critical step in graph machine learning, where we divide our dataset into subsets for training, validation, and testing.
It ensures that our models are evaluated properly, preventing overfitting, and enabling generalization.
In this tutorial, we will explore the basics of dataset splitting, focusing on three fundamental tasks: node prediction, link prediction, and graph prediction.
We will introduce commonly used techniques, including :class:`~torch_geometric.transforms.RandomNodeSplit` and :class:`~torch_geometric.transforms.RandomLinkSplit` transformations.
Additionally, we will also cover how to create custom dataset splits beyond random ones.

Node Prediction
---------------

.. note::

    In this section, we'll learn how to use :class:`~torch_geometric.transforms.RandomNodeSplit` of :pyg:`PyG` to randomly divide nodes into training, validation, and test sets.
    A fully working example on dataset :class:`~torch_geometric.datasets.Planetoid` is available in `examples/cora.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/cora.py>`_.

The :class:`~torch_geometric.transforms.RandomNodeSplit` is initialized to split nodes for both a :pyg:`PyG` :class:`~torch_geometric.data.Data` and :class:`~torch_geometric.data.HeteroData` object.

* :obj:`split` defines the dataset's split type.
* :obj:`num_splits` defines the number of splits to add.
* :obj:`num_train_per_class` defines the number of training nodes per class.
* :obj:`num_val` defines the number of validation nodes after data splitting.
* :obj:`num_test` defines the number of test nodes after data splitting.
* :obj:`key` defines the name of the ground-truth labels.

.. code-block:: python

    import torch
    from torch_geometric.data import Data
    from torch_geometric.transforms import RandomNodeSplit

    x = torch.randn(8, 32)  # Node features of shape [num_nodes, num_features]
    y = torch.randint(0, 4, (8, ))  # Node labels of shape [num_nodes]
    edge_index = torch.tensor([
        [2, 3, 3, 4, 5, 6, 7],
        [0, 0, 1, 1, 2, 3, 4]],
    )

    #   0  1
    #  / \/ \
    # 2  3  4
    # |  |  |
    # 5  6  7

    data = Data(x=x, y=y, edge_index=edge_index)
    node_transform = RandomNodeSplit(num_val=2, num_test=3)
    node_splits = node_transform(data)

Here, we initialize a :class:`~torch_geometric.transforms.RandomNodeSplit` transformation to split the graph data by nodes.
After the transformation, :obj:`train_mask`, :obj:`valid_mask` and :obj:`test_mask` will be attached to the graph data.

.. code-block:: python

    node_splits.train_mask
    >>> tensor([ True, False, False, False, True, True, False, False])
    node_splits.val_mask
    >>> tensor([False, False, False, False, False, False, True, True])
    node_splits.test_mask
    >>> tensor([False, True, True, True, False, False, False, False])

In this example, there are 8 nodes, we want to sample 2 nodes for validation, 3 nodes for testing, and the rest for training.
Finally, we got node :obj:`0, 4, 5` as training set, node :obj:`6, 7` as validation set, and node :obj:`1, 2, 3` as test set.

Link Prediction
---------------

.. note::

    In this section, we'll learn how to use :class:`~torch_geometric.transforms.RandomLinkSplit` of :pyg:`PyG` to randomly divide edges into training, validation, and test sets.
    A fully working example on dataset :class:`~torch_geometric.datasets.Planetoid` is available in `examples/link_pred.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py>`_.

The :class:`~torch_geometric.transforms.RandomLinkSplit` is initialized to split edges for both a :pyg:`PyG` :class:`~torch_geometric.data.Data` and :class:`~torch_geometric.data.HeteroData` object.

* :obj:`num_val` defines the number of validation edges after data splitting.
* :obj:`num_test` defines the number of test edges after data splitting.
* :obj:`is_undirected` defines whether the graph is assumed as undirected.

.. code-block:: python

    import torch
    from torch_geometric.data import Data
    from torch_geometric.transforms import RandomLinkSplit

    x = torch.randn(8, 32)  # Node features of shape [num_nodes, num_features]
    y = torch.randint(0, 4, (8, ))  # Node labels of shape [num_nodes]
    edge_index = torch.tensor([
        [2, 3, 3, 4, 5, 6, 7],
        [0, 0, 1, 1, 2, 3, 4]],
    )

    edge_y = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    #   0  1
    #  / \/ \
    # 2  3  4
    # |  |  |
    # 5  6  7

    data = Data(x=x, y=y, edge_index=edge_index, edge_y=edge_y)
    edge_transform = RandomLinkSplit(num_val=0.2, num_test=0.2, key='edge_y',
                                    is_undirected=False, add_negative_train_samples=False)
    train_data, val_data, test_data = edge_transform(data)

Similar to node splitting, we initialize a :class:`~torch_geometric.transforms.RandomLinkSplit` transformation to split the graph data by edges.
Below, we can see the splitting results.

.. code-block:: python

    train_data
    >>> Data(x=[8, 32], edge_index=[2, 5], y=[8], edge_y=[5], edge_y_index=[2, 5])
    val_data
    >>> Data(x=[8, 32], edge_index=[2, 5], y=[8], edge_y=[2], edge_y_index=[2, 2])
    test_data
    >>> Data(x=[8, 32], edge_index=[2, 6], y=[8], edge_y=[2], edge_y_index=[2, 2])

:obj:`train_data.edge_index` and :obj:`val_data.edge_index` refers to the edges that are used for message passing.
As such, during training and validation, we are allowed to propagate information based on the training edges.
While during testing, we can propagate information based on the union of training and validation edges.
For evaluation and testing, :obj:`val_data.edge_label_index` and :obj:`test_data.edge_label_index` hold a batch of positive and negative samples that should be used to evaluate and test our model on.

Graph Prediction
----------------

.. note::

    In this section, we'll learn how to randomly divide graphs into training, validation, and test sets.
    A fully working example on dataset :class:`~torch_geometric.datasets.PPI` is available in `examples/ppi.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ppi.py>`_.

In graph prediction task, each graph is an independent sample.
Usually we need to divide a graph dataset according to a certain ratio.
:pyg:`PyG` has provided some datasets that already contain corresponding indexes for training, validation and test, such as :class:`~torch_geometric.datasets.PPI`.

.. code-block:: python

    from torch_geometric.datasets import PPI

    path = './data/PPI'
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')

In addition, we can also use :obj:`scikit-learn` or :obj:`numpy` to randomly divide :pyg:`PyG` dataset.

Creating Custom Splits
----------------------

If random splitting doesn't suit our specific use case, then we can create custom node splits.
This requirement generally occurs in real business scenarios.
For example, there are large-scale heterogeneous graphs in e-commerce scenarios, and nodes can be used to represent users, products, merchants, etc.
We may divide new and old users to evaluate the performance of the model on new users.
Therefore, we'll not post specific examples here for reference.
