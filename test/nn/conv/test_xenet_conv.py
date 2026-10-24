import pytest
import torch

from torch_geometric.nn import XENetConv


def test_xenet_conv():
    x = torch.randn(4, 8)

    # Every edge has its reverse.
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])

    edge_attr = torch.randn(6, 4)

    conv = XENetConv(
        stack_channels=16,
        node_channels=12,
        edge_channels=6,
    )

    out_x, out_edge_attr = conv(
        x,
        edge_index,
        edge_attr,
    )

    assert out_x.size() == (4, 12)
    assert out_edge_attr.size() == (6, 6)


def test_xenet_conv_without_attention():
    x = torch.randn(4, 8)

    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ])

    edge_attr = torch.randn(4, 4)

    conv = XENetConv(
        stack_channels=16,
        node_channels=12,
        edge_channels=6,
        attention=False,
    )

    out_x, out_edge_attr = conv(
        x,
        edge_index,
        edge_attr,
    )

    assert out_x.size() == (4, 12)
    assert out_edge_attr.size() == (4, 6)


def test_xenet_conv_requires_reverse_edges():
    x = torch.randn(3, 8)

    edge_index = torch.tensor([
        [0, 1],
        [1, 2],
    ])

    edge_attr = torch.randn(2, 4)

    conv = XENetConv(
        stack_channels=16,
        node_channels=12,
        edge_channels=6,
    )

    with pytest.raises(ValueError, match="both directions"):
        conv(x, edge_index, edge_attr)


def test_xenet_conv_gradients():
    x = torch.randn(4, 8, requires_grad=True)

    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ])

    edge_attr = torch.randn(4, 4, requires_grad=True)

    conv = XENetConv(
        stack_channels=16,
        node_channels=12,
        edge_channels=6,
    )

    out_x, out_edge_attr = conv(
        x,
        edge_index,
        edge_attr,
    )

    loss = out_x.sum() + out_edge_attr.sum()
    loss.backward()

    assert x.grad is not None
    assert edge_attr.grad is not None


def test_xenet_conv_bias():
    conv = XENetConv(
        stack_channels=8,
        node_channels=4,
        edge_channels=6,
        bias=False,
    )

    assert conv.node_model.bias is None
    assert conv.edge_model.bias is None
    assert conv.stack_models[0].bias is None
    assert conv.incoming_attention.bias is None
    assert conv.outgoing_attention.bias is None


def test_xenet_conv_reset_parameters():
    conv = XENetConv(
        stack_channels=8,
        node_channels=4,
        edge_channels=6,
    )

    conv.reset_parameters()


def test_xenet_conv_multiple_stack_layers():
    x = torch.randn(4, 8)

    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])

    edge_attr = torch.randn(6, 4)

    conv = XENetConv(
        stack_channels=[8, 16, 32],
        node_channels=12,
        edge_channels=6,
    )

    out_x, out_edge_attr = conv(
        x,
        edge_index,
        edge_attr,
    )

    assert out_x.size() == (4, 12)
    assert out_edge_attr.size() == (6, 6)


def test_xenet_conv_self_loops():
    x = torch.randn(3, 8)

    edge_index = torch.tensor([
        [0, 1, 2],
        [0, 1, 2],
    ])

    edge_attr = torch.randn(3, 4)

    conv = XENetConv(
        stack_channels=16,
        node_channels=12,
        edge_channels=6,
    )

    out_x, out_edge_attr = conv(
        x,
        edge_index,
        edge_attr,
    )

    assert out_x.size() == (3, 12)
    assert out_edge_attr.size() == (3, 6)


def test_xenet_conv_empty_stack():
    with pytest.raises(
            ValueError,
            match="stack_channels.*at least one layer",
    ):
        XENetConv(
            stack_channels=[],
            node_channels=4,
            edge_channels=6,
        )
