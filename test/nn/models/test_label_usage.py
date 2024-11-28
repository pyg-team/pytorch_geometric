import torch

from torch_geometric.nn.models import LabelUsage
from torch_geometric.nn.models import GCN


def test_label_usage():
    x = torch.rand(6, 4) # 6 nodes, 4 features
    y = torch.tensor([1, 0, 0, 2, 1, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    train_idx = torch.tensor([0, 2, 3, 5])

    num_classes = len(torch.unique(y))
    base_model = GCN(in_channels=x.size(1) + num_classes, hidden_channels=8, num_layers=3, out_channels=num_classes)
    label_usage = LabelUsage(
        split_ratio=0.5, 
        num_recycling_iterations=10, 
        return_tuple=True, 
        base_model=base_model,
    )
    
    output, train_labels_idx, train_pred_idx = label_usage(x, edge_index, y, train_idx)

    # Check output shapes
    assert output.size(0) == x.size(0)
    assert output.size(1) == num_classes

    # Check split ratio masking
    actual_ratio = len(train_labels_idx) / len(train_idx)
    assert torch.isclose(torch.tensor(actual_ratio), torch.tensor(label_usage.split_ratio), atol=0.1)

    # Test zero recycling iterations
    label_usage = LabelUsage(
        split_ratio=0.5, 
        num_recycling_iterations=0, 
        return_tuple=False, 
        base_model=base_model,
    )
    output = label_usage(x, edge_index, y, train_idx)
    assert output.size(0) == x.size(0)
    assert output.size(1) == num_classes