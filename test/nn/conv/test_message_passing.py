import torch
from torch_geometric.nn import MessagePassing


def test_message_passing():
    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2],
    ])

    x = torch.Tensor([[1], [2]])
    out = MessagePassing(flow='source_to_target').propagate(
        edge_index, x=x, size=(2, 3))
    assert out.tolist() == [[3], [3], [3]]

    x = torch.Tensor([[1], [2], [3]])
    out = MessagePassing(flow='source_to_target').propagate(edge_index, x=x)
    assert out.tolist() == [[3], [3], [3]]

    x = torch.Tensor([[1], [2], [3]])
    out = MessagePassing(flow='target_to_source').propagate(
        edge_index, x=x, size=(2, 3))
    assert out.tolist() == [[6], [6]]

    x = torch.Tensor([[1], [2], [3]])
    out = MessagePassing(flow='target_to_source').propagate(edge_index, x=x)
    assert out.tolist() == [[6], [6], [0]]

    x = (torch.Tensor([[1], [2]]), torch.Tensor([[1], [2], [3]]))
    out = MessagePassing(flow='source_to_target').propagate(edge_index, x=x)
    assert out.tolist() == [[3], [3], [3]]

    x = (torch.Tensor([[1], [2]]), torch.Tensor([[1], [2], [3]]))
    out = MessagePassing(flow='target_to_source').propagate(edge_index, x=x)
    assert out.tolist() == [[6], [6]]
