import matplotlib.pyplot as plt
import torch
from GraphMask import GraphMaskExplainer

import torch_geometric.transforms as T
from torch_geometric.datasets import Entities, Planetoid, TUDataset
from torch_geometric.loader import DataLoader

if __name__ == "__main__":
    # GCN Node Classification=================================================
    transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
    edge_y = torch.randint(low=0, high=30,
                           size=(dataset[0].edge_index.size(1), ))
    edge_weights = [dataset[0].edge_weight, None, dataset[0].edge_weight]
    add_self_loops = [True, False, True]
    normalize = [True, False, True]
    explainer = GraphMaskExplainer([1433, 7, 9], [7, 9, 7], 3,
                                   allow_multiple_explanations=True,
                                   type_task='homogeneous')
    model = explainer.train_node_model(dataset[0].x, dataset[0].y,
                                       dataset[0].edge_index,
                                       dataset[0].train_mask,
                                       edge_weight=edge_weights,
                                       add_self_loops=add_self_loops,
                                       normalize=normalize)

    explainer.model_to_explain = model
    feat_mask = explainer.__train_node_explainer__([1, 6, 3, 5, 12],
                                                   dataset[0].x,
                                                   dataset[0].edge_index,
                                                   edge_weight1=edge_weights)
    edge_mask = explainer.__explain_node__([1, 6, 3, 5, 12], dataset[0].x,
                                           dataset[0].edge_index)
    ax, G = explainer.visualize_subgraph([1, 6, 3, 5, 12],
                                         dataset[0].edge_index, edge_mask,
                                         y=dataset[0].y, edge_y=edge_y,
                                         node_alpha=feat_mask)
    plt.show()

    # R-GCN Node Classification===============================================
    dataset = Entities(root='/tmp/AIFB', name='AIFB')
    edge_index, edge_type, train_labels, test_labels, train_mask, test_mask, \
        num_nodes = dataset[0].edge_index, dataset[0].edge_type, \
        dataset[0].train_y, dataset[0].test_y, \
        dataset[0].train_idx, dataset[0].test_idx, dataset[0].num_nodes
    x = torch.randn(8285, 4)
    edge_y = torch.randint(low=0, high=30, size=(edge_index.size(1), ))
    aggr = ['mean', 'add', 'mean', 'mean', 'add']
    num_bases = [None] * 5
    num_blocks = [2, 4, 2, 2, 4]

    explainer = GraphMaskExplainer([4, 8, 12, 4, 20], [8, 12, 4, 20, 4], 5,
                                   400, num_bases=num_bases,
                                   num_blocks=num_blocks, num_hops=None)
    model = explainer.train_node_model(x, train_labels, edge_index, train_mask,
                                       edge_type, aggr=aggr)
    explainer.allow_multiple_explanations, explainer.model_to_explain = \
        True, model
    feat_mask = explainer.__train_node_explainer__(
        [1, 6, 3, 5, 12, 15, 20, 23, 25, 28, 30, 35, 40, 50, 65], x,
        edge_index, edge_type)
    edge_mask = explainer.__explain_node__(
        [1, 6, 3, 5, 12, 15, 20, 23, 25, 28, 30, 35, 40, 50, 65], x,
        edge_index)
    ax, G = explainer.visualize_subgraph(
        [1, 6, 3, 5, 12, 15, 20, 23, 25, 28, 30, 35, 40, 50, 65], edge_index,
        edge_mask, y=None, edge_y=edge_y, node_alpha=feat_mask)
    plt.show()

    # GCN-Graph Classification================================================
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    edge_y = torch.randint(low=0, high=30,
                           size=(dataset[4].edge_index.size(1), ))
    add_self_loops = [True, False, True, False, True]
    normalize = [False, False, False, False, False]
    edge_weight = torch.rand(dataset[4].edge_index.size(1))
    edge_weights = [edge_weight, None, edge_weight, edge_weight, None]

    explainer = GraphMaskExplainer([3, 7, 9, 5, 2], [7, 9, 5, 2, 6], 5,
                                   pooling='mean', type_task='homogeneous')
    model = explainer.train_graph_model(loader, add_self_loops=add_self_loops,
                                        normalize=normalize)

    explainer.model_to_explain = model
    feat_mask = explainer.__train_graph_explainer__([4], dataset[4].x,
                                                    dataset[4].edge_index,
                                                    edge_weight=edge_weights)
    edge_mask = explainer.__explain_graph__([4])
    ax, G = explainer.visualize_subgraph([4], dataset[4].edge_index, edge_mask,
                                         y=dataset[4].y, edge_y=edge_y,
                                         node_alpha=feat_mask, task='graph')
    plt.show()

    # RGCN-Graph Classification===============================================
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    edge_type = torch.randint(low=0, high=92,
                              size=(dataset[4].edge_index.size(1), ))
    edge_y = torch.randint(low=0, high=30,
                           size=(dataset[4].edge_index.size(1), ))
    aggr = ['mean', 'add', 'mean', 'mean', 'add']
    num_bases = [None] * 5
    num_blocks = [3] * 5

    explainer = GraphMaskExplainer([3, 6, 3, 9, 3], [6, 3, 9, 3, 6], 5, 400,
                                   num_bases=num_bases, num_blocks=num_blocks,
                                   pooling='mean')
    model = explainer.train_graph_model(loader, aggr=aggr)

    explainer.model_to_explain = model
    feat_mask = explainer.__train_graph_explainer__([4], dataset[4].x,
                                                    dataset[4].edge_index,
                                                    edge_type=edge_type)
    edge_mask = explainer.__explain_graph__([4])
    ax, G = explainer.visualize_subgraph([4], dataset[4].edge_index, edge_mask,
                                         y=dataset[4].y, edge_y=edge_y,
                                         node_alpha=feat_mask, task='graph')
    plt.show()
