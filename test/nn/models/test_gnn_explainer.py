import torch
from torch_geometric.nn import GNNExplainer, GCNConv


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


def test_gnn_explainer():
    model = Net()
    explainer = GNNExplainer(model, log=False)
    assert explainer.__repr__() == 'GNNExplainer()'

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    y = torch.randint(0, 6, (8, ), dtype=torch.long)

    node_feat_mask, edge_mask = explainer.explain_node(2, x, edge_index)
    assert node_feat_mask.size() == (x.size(1), )
    assert node_feat_mask.min() >= 0 and node_feat_mask.max() <= 1
    assert edge_mask.size() == (edge_index.size(1), )
    assert edge_mask.min() >= 0 and edge_mask.max() <= 1

    explainer.visualize_subgraph(2, edge_index, edge_mask, threshold=None)
    explainer.visualize_subgraph(2, edge_index, edge_mask, threshold=0.5)
    explainer.visualize_subgraph(2, edge_index, edge_mask, y=y, threshold=None)
    explainer.visualize_subgraph(2, edge_index, edge_mask, y=y, threshold=0.5)
