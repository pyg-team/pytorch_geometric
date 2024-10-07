import torch
from torch.nn import ModuleList
from torch_geometric.nn import GCN
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.models.cognn import CoGNN


def test_cognn() -> None:
    num_nodes = 10
    num_features = 128
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    ])

    env_layer1 = GCNConv(
        in_channels=num_features,
        out_channels=num_features,
    )

    env_layer2 = GCNConv(
        in_channels=num_features,
        out_channels=num_features,
    )

    env_layer3 = GCNConv(
        in_channels=num_features,
        out_channels=1,
    )

    action_net = GCN(
        in_channels=num_features,
        out_channels=4,
        hidden_channels=num_features,
        num_layers=2,
    )

    model = CoGNN(
        env_net=ModuleList([env_layer1, env_layer2, env_layer3]),
        action_net=action_net,
        env_activation='relu',
        temp=0.1,
        dropout=0.1,
    )
    assert str(model) == '''CoGNN(
  (env_net): ModuleList(
    (0-1): 2 x GCNConv(128, 128)
    (2): GCNConv(128, 1)
  )
  (action_net): GCN(128, 4, num_layers=2)
  (activation): ReLU()
  (dropout): Dropout(p=0.1, inplace=False)
)'''

    # Test train:
    model.train()
    out = model(x, edge_index)
    assert out.size() == (num_nodes, 1)
    assert out.min().item() >= -0.1 and out.max().item() < 1.3

    # Test train:
    model.eval()
    out = model(x, edge_index)
    assert out.size() == (num_nodes, 1)
    assert out.min().item() >= 0 and out.max().item() < 1.5
