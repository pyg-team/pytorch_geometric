import torch

from torch_geometric.nn.models import GCN, LabelUsage


def test_label_usage():
    # Test mask index tensor
    x = torch.rand(6, 4)  # 6 nodes, 4 features
    y = torch.tensor([1, 0, 0, 2, 1, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = torch.tensor([0, 2, 3, 5])

    num_classes = len(torch.unique(y))
    base_model = GCN(in_channels=x.size(1) + num_classes, hidden_channels=8,
                     num_layers=3, out_channels=num_classes)
    label_usage = LabelUsage(base_model=base_model, num_classes=num_classes,
                             split_ratio=0.6, num_recycling_iterations=10,
                             return_tuple=True)

    output, train_labels_idx, train_pred_idx = label_usage(
        feat=x, edge_index=edge_index, y=y, mask=mask)

    # Check output shapes
    assert output.size(0) == x.size(0)
    assert output.size(1) == num_classes

    # Test mask bool tensor
    num_nodes = x.size(0)
    mask_bool = torch.zeros(num_nodes, dtype=torch.bool)
    mask_bool[mask] = True

    label_usage_bool = LabelUsage(base_model=base_model,
                                  num_classes=num_classes, split_ratio=0.6,
                                  num_recycling_iterations=10,
                                  return_tuple=True)

    output, train_labels_idx, train_pred_idx = label_usage_bool(
        feat=x, edge_index=edge_index, y=y, mask=mask_bool)

    # Check output shapes
    assert output.size(0) == x.size(0)
    assert output.size(1) == num_classes

    # Test zero recycling iterations
    label_usage_zero_recycling = LabelUsage(
        base_model=base_model,
        num_classes=num_classes,
    )

    output = label_usage_zero_recycling(feat=x, edge_index=edge_index, y=y,
                                        mask=mask)
    assert output.size(0) == x.size(0)
    assert output.size(1) == num_classes

    # Test 2D label tensor
    y = torch.tensor([[1], [0], [0], [2], [1], [1]])  # Node labels (N, 1)
    label_usage_2d = LabelUsage(base_model=base_model, num_classes=num_classes,
                                split_ratio=0.6, num_recycling_iterations=10,
                                return_tuple=False)
    output = label_usage_2d(feat=x, edge_index=edge_index, y=y, mask=mask)
    assert output.size(0) == x.size(0)
    assert output.size(1) == num_classes
