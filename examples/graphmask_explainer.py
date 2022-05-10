import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Entities, Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GraphMaskExplainer,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.nn.conv import FastRGCNConv, GATConv, GCNConv

if __name__ == "__main__":
    # GCN Node Classification=================================================
    transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
    data = dataset[0]

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(dataset.num_features, 16, normalize=False)
            self.conv2 = GCNConv(16, dataset.num_classes, normalize=False)

        def forward(self, x, edge_index, edge_weight):
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
    explainer = GraphMaskExplainer(2, epochs=4,
                                   allow_multiple_explanations=True)
    explainer.__set_flags__(model)

    for epoch in range(1, 5):
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index, edge_weight)
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print('Epoch {:03d}'.format(epoch), 'train_loss: {:.4f}'.format(loss))
    print("Optimization Finished!")

    # [1, 6, 3, 5, 12]
    explainer = GraphMaskExplainer(2, model_to_explain=model, epochs=4,
                                   allow_multiple_explanations=True)
    edge_y = torch.randint(low=0, high=30, size=(data.edge_index.size(1), ))
    feat_mask = explainer.train_node_explainer([6, 12], data.x,
                                               data.edge_index,
                                               edge_weight=data.edge_weight)
    edge_mask = explainer.explain_node([6, 12], data.x, data.edge_index)
    ax, G = explainer.visualize_subgraph([6, 12], data.edge_index, edge_mask,
                                         y=data.y, edge_y=edge_y,
                                         node_alpha=feat_mask)
    plt.show()

    # GAT Node Classification=================================================
    transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
    data = dataset[0]

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(dataset.num_features, 8, heads=8)
            self.conv2 = GATConv(64, dataset.num_classes, heads=1,
                                 concat=False)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
    explainer = GraphMaskExplainer(2, epochs=4,
                                   allow_multiple_explanations=True,
                                   layer_type='GAT')
    explainer.__set_flags__(model)

    for epoch in range(1, 5):
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index)
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print('Epoch {:03d}'.format(epoch), 'train_loss: {:.4f}'.format(loss))
    print("Optimization Finished!")

    explainer = GraphMaskExplainer(2, model_to_explain=model, epochs=4,
                                   allow_multiple_explanations=True,
                                   layer_type='GAT')
    edge_y = torch.randint(low=0, high=30, size=(data.edge_index.size(1), ))
    feat_mask = explainer.train_node_explainer([1, 6, 3, 5, 12], data.x,
                                               data.edge_index)
    edge_mask = explainer.explain_node([1, 6, 3, 5, 12], data.x,
                                       data.edge_index)
    ax, G = explainer.visualize_subgraph([1, 6, 3, 5, 12], data.edge_index,
                                         edge_mask, y=data.y, edge_y=edge_y,
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

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = FastRGCNConv(4, 8, 90, num_blocks=2, aggr='mean')
            self.conv2 = FastRGCNConv(8, 12, 90, num_blocks=4, aggr='add')
            self.conv3 = FastRGCNConv(12, 4, 90, num_blocks=2, aggr='mean')

        def forward(self, x, edge_index, edge_type):
            x = F.relu(self.conv1(x, edge_index, edge_type))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.conv2(x, edge_index, edge_type))
            x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index, edge_type)
            return F.log_softmax(x, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)
    explainer = GraphMaskExplainer(3, epochs=4)
    explainer.__set_flags__(model)

    for epoch in range(1, 5):
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index, edge_type)
        loss = F.nll_loss(log_logits[train_mask], train_labels)
        loss.backward()
        optimizer.step()
        print('Epoch {:03d}'.format(epoch), 'train_loss: {:.4f}'.format(loss))
    print("Optimization Finished!")

    explainer = GraphMaskExplainer(3, epochs=4, model_to_explain=model,
                                   allow_multiple_explanations=True,
                                   layer_type='FastRGCN')
    feat_mask = explainer.train_node_explainer([1, 5, 12], x, edge_index,
                                               edge_type=edge_type)
    edge_mask = explainer.explain_node([1, 5, 12], x, edge_index)
    ax, G = explainer.visualize_subgraph([1, 5, 12], edge_index, edge_mask,
                                         y=None, edge_y=edge_y,
                                         node_alpha=feat_mask)
    plt.show()

    # GCN Graph Classification================================================
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    edge_y = torch.randint(low=0, high=30,
                           size=(dataset[4].edge_index.size(1), ))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(3, 7, add_self_loops=True, normalize=False)
            self.conv2 = GCNConv(7, 9, add_self_loops=False, normalize=True)
            self.conv3 = GCNConv(9, 6, add_self_loops=True, normalize=False)

        def forward(self, x, edge_index, batch, edge_weight):
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.conv2(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.conv3(x, edge_index, edge_weight))
            x = global_add_pool(x, batch)
            return F.log_softmax(x, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)
    explainer = GraphMaskExplainer(3, epochs=4)
    explainer.__set_flags__(model)

    for epoch in range(1, 5):
        loss_all = 0
        for data in loader:
            model.train()
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch,
                           data.edge_weight)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
            optimizer.step()
            loss_all += loss.item() * data.num_graphs
        print('Epoch {:03d}'.format(epoch),
              'train_loss: {:.4f}'.format(loss_all / len(loader.dataset)))
    print("Optimization Finished!")

    explainer = GraphMaskExplainer(3, model_to_explain=model, epochs=4)
    edge_weight = torch.rand(dataset[4].edge_index.size(1))
    feat_mask = explainer.train_graph_explainer(4, dataset[4].x,
                                                dataset[4].edge_index,
                                                edge_weight=edge_weight)
    edge_mask = explainer.explain_graph(4)
    ax, G = explainer.visualize_subgraph(None, dataset[4].edge_index,
                                         edge_mask, y=dataset[4].y,
                                         edge_y=edge_y, node_alpha=feat_mask,
                                         task='graph')
    plt.show()

    # GAT Graph Classification================================================
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    edge_y = torch.randint(low=0, high=30,
                           size=(dataset[4].edge_index.size(1), ))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(3, 8, heads=8)
            self.conv2 = GATConv(64, 6, heads=1, concat=False)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            return F.log_softmax(x, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)
    explainer = GraphMaskExplainer(2, epochs=4, layer_type='GAT')
    explainer.__set_flags__(model)

    for epoch in range(1, 5):
        loss_all = 0
        for data in loader:
            model.train()
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
            optimizer.step()
            loss_all += loss.item() * data.num_graphs
        print('Epoch {:03d}'.format(epoch),
              'train_loss: {:.4f}'.format(loss_all / len(loader.dataset)))
    print("Optimization Finished!")

    explainer = GraphMaskExplainer(2, epochs=4, model_to_explain=model,
                                   layer_type='GAT')
    feat_mask = explainer.train_graph_explainer(4, dataset[4].x,
                                                dataset[4].edge_index)
    edge_mask = explainer.explain_graph(4)
    ax, G = explainer.visualize_subgraph(None, dataset[4].edge_index,
                                         edge_mask, y=dataset[4].y,
                                         edge_y=edge_y, node_alpha=feat_mask,
                                         task='graph')
    plt.show()

    # RGCN Graph Classification===============================================
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    edge_type = torch.randint(low=0, high=90,
                              size=(dataset[4].edge_index.size(1), ))
    edge_y = torch.randint(low=0, high=30,
                           size=(dataset[4].edge_index.size(1), ))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = FastRGCNConv(3, 6, 90, num_blocks=3, aggr='mean')
            self.conv2 = FastRGCNConv(6, 3, 90, num_blocks=3, aggr='add')
            self.conv3 = FastRGCNConv(3, 6, 90, num_blocks=3, aggr='mean')

        def forward(self, x, edge_index, batch, edge_type):
            x = F.relu(self.conv1(x, edge_index, edge_type))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.conv2(x, edge_index, edge_type))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.conv3(x, edge_index, edge_type))
            x = global_mean_pool(x, batch)
            return F.log_softmax(x, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=5e-4)
    explainer = GraphMaskExplainer(3, epochs=4)
    explainer.__set_flags__(model)

    for epoch in range(1, 5):
        loss_all = 0
        for data in loader:
            model.train()
            optimizer.zero_grad()
            data.edge_type = torch.randint(low=0, high=90,
                                           size=(data.edge_index.size(1), ))
            output = model(data.x, data.edge_index, data.batch, data.edge_type)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
            optimizer.step()
            loss_all += loss.item() * data.num_graphs
        print('Epoch {:03d}'.format(epoch),
              'train_loss: {:.4f}'.format(loss_all / len(loader.dataset)))
    print("Optimization Finished!")

    explainer = GraphMaskExplainer(3, model_to_explain=model, epochs=4,
                                   layer_type='FastRGCN')
    feat_mask = explainer.train_graph_explainer(4, dataset[4].x,
                                                dataset[4].edge_index,
                                                edge_type=edge_type)
    edge_mask = explainer.explain_graph(4)
    ax, G = explainer.visualize_subgraph(None, dataset[4].edge_index,
                                         edge_mask, y=dataset[4].y,
                                         edge_y=edge_y, node_alpha=feat_mask,
                                         task='graph')
    plt.show()
