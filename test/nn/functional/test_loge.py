import torch

from torch_geometric.nn import functional as F


def test_loge():
    inputs = torch.tensor([
        [-3.0, -2.0],
        [-4.0, -5.0],
    ])
    targets = torch.tensor([1, 0])

    # Test no reduction
    assert torch.allclose(
        F.LogELoss(reduction="none")(inputs, targets),
        torch.tensor([2.0173, 2.6416]), atol=1e-4)

    # Test epsilon
    assert torch.allclose(
        F.LogELoss(reduction="none", epsilon=0.1)(inputs, targets),
        torch.tensor([3.0445, 3.7136]), atol=1e-4)

    # Test weighting
    assert torch.allclose(
        F.LogELoss(reduction="none", weight=torch.tensor([1, 2]))(inputs,
                                                                  targets),
        torch.tensor([4.0345, 2.6416]), atol=1e-4)

    # Test sum reduction
    assert torch.allclose(
        F.LogELoss(reduction="sum")(inputs, targets), torch.tensor(4.6589),
        atol=1e-4)

    # Test mean reduction
    assert torch.allclose(
        F.LogELoss(reduction="mean")(inputs, targets), torch.tensor(2.3294),
        atol=1e-4)

    # Test weighted multidimensional
    inputs = inputs.unsqueeze(-1).expand(-1, -1, 2)
    targets = targets.unsqueeze(-1).expand(-1, 2)
    assert torch.allclose(
        F.LogELoss(reduction="none", weight=torch.tensor([1, 2]))(inputs,
                                                                  targets),
        torch.tensor([[4.0345, 4.0345], [2.6416, 2.6416]]), atol=1e-4)


def test_loge_with_logits():
    inputs = torch.tensor([
        [-2.0, 1.0],
        [-3.0, 2.0],
    ])
    targets = torch.tensor([1, 0])

    # Test no reduction
    assert torch.allclose(
        F.LogEWithLogitsLoss(reduction="none")(inputs, targets),
        torch.tensor([0.1470, 2.8517]), atol=1e-4)

    # Test epsilon
    assert torch.allclose(
        F.LogEWithLogitsLoss(reduction="none", epsilon=0.1)(inputs, targets),
        torch.tensor([0.3960, 3.9331]), atol=1e-4)

    # Test weighting
    assert torch.allclose(
        F.LogEWithLogitsLoss(reduction="none",
                             weight=torch.tensor([1, 2]))(inputs, targets),
        torch.tensor([0.2940, 2.8517]), atol=1e-4)

    # Test sum reduction
    assert torch.allclose(
        F.LogEWithLogitsLoss(reduction="sum")(inputs, targets),
        torch.tensor(2.9986), atol=1e-4)

    # Test mean reduction
    assert torch.allclose(
        F.LogEWithLogitsLoss(reduction="mean")(inputs, targets),
        torch.tensor(1.4993), atol=1e-4)

    # Test weighted multidimensional
    inputs = inputs.unsqueeze(-1).expand(-1, -1, 2)
    targets = targets.unsqueeze(-1).expand(-1, 2)
    assert torch.allclose(
        F.LogEWithLogitsLoss(reduction="none",
                             weight=torch.tensor([1, 2]))(inputs, targets),
        torch.tensor([[0.2940, 0.2940], [2.8517, 2.8517]]), atol=1e-4)


def test_binary_loge():
    inputs = torch.tensor([0.6, 0.1])
    targets = torch.tensor([0.0, 1.0])

    # Test no reduction
    assert torch.allclose(
        F.BinaryLogELoss(reduction="none")(inputs, targets),
        torch.tensor([1.3828, 2.1405]),
        atol=1e-4,
    )

    # Test epsilon
    assert torch.allclose(
        F.BinaryLogELoss(reduction="none", epsilon=0.1)(inputs, targets),
        torch.tensor([2.3187, 3.1791]), atol=1e-4)

    # Test weighting
    assert torch.allclose(
        F.BinaryLogELoss(reduction="none",
                         weight=torch.tensor([1, 2]))(inputs, targets),
        torch.tensor([1.3828, 4.2810]),
        atol=1e-4,
    )

    # Test sum reduction
    assert torch.allclose(
        F.BinaryLogELoss(reduction="sum")(inputs, targets),
        torch.tensor(3.5233),
        atol=1e-4,
    )

    # Test mean reduction
    assert torch.allclose(
        F.BinaryLogELoss(reduction="mean")(inputs, targets),
        torch.tensor(1.7617),
        atol=1e-4,
    )


def test_binary_loge_with_logits():
    inputs = torch.tensor([-1.0, 2.0])
    targets = torch.tensor([0.0, 1.0])

    # Test no reduction
    assert torch.allclose(
        F.BinaryLogEWithLogitsLoss(reduction="none")(inputs, targets),
        torch.tensor([0.7035, 0.3462]),
        atol=1e-4,
    )

    # Test epsilon
    assert torch.allclose(
        F.BinaryLogEWithLogitsLoss(reduction="none", epsilon=0.1)(inputs,
                                                                  targets),
        torch.tensor([1.4189, 0.8195]), atol=1e-4)

    # Test weighting
    assert torch.allclose(
        F.BinaryLogEWithLogitsLoss(reduction="none",
                                   weight=torch.tensor([1, 2]))(inputs,
                                                                targets),
        torch.tensor([0.7035, 0.6923]),
        atol=1e-4,
    )

    # Test sum reduction
    assert torch.allclose(
        F.BinaryLogEWithLogitsLoss(reduction="sum")(inputs, targets),
        torch.tensor(1.0497),
        atol=1e-4,
    )

    # Test mean reduction
    assert torch.allclose(
        F.BinaryLogEWithLogitsLoss(reduction="mean")(inputs, targets),
        torch.tensor(0.5249),
        atol=1e-4,
    )
