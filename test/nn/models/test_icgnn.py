import torch

from torch_geometric.nn.models.icgnn import ICGNN


def test_icgnn() -> None:
    num_nodes = 10
    num_features = 128
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    ])

    model = ICGNN(
        num_nodes=num_nodes,
        K=10,
        feature_scale=1,
        in_channel=num_features,
        hidden_channel=num_features,
        out_channel=1,
        num_layers=2,
        dropout=0.0,
        activation='relu',
    )
    assert str(model) == '''ICGNN(
  (icg_fitter): ICGFitter()
  (nn): ICGNNSequence(
    (layers): ModuleList(
      (0): ICGNNLayer(
        (lin_node): Linear(in_features=128, out_features=128, bias=True)
        (lin_community): Linear(in_features=128, out_features=128, bias=True)
      )
      (1): ICGNNLayer(
        (lin_node): Linear(in_features=128, out_features=1, bias=True)
        (lin_community): Linear(in_features=1, out_features=1, bias=True)
      )
    )
    (dropout): Dropout(p=0.0, inplace=False)
    (activation): ReLU()
  )
)'''

    # ICG loss
    out = model.icg_fit_loss(x, edge_index)
    assert out.size() == ()
    assert out.min().item() >= 10 and out.max().item() < 12.0

    # Train:
    model.train()
    out = model(x)
    assert out.size() == (num_nodes, 1)
    assert out.min().item() >= -2 and out.max().item() < 1.2

    # Test:
    model.eval()
    out = model(x)
    assert out.size() == (num_nodes, 1)
    assert out.min().item() >= -0.5 and out.max().item() < 1.2
