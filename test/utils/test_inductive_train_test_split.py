import pytest
import torch

import torch_geometric as tg
from torch_geometric.utils import inductive_train_test_split, split_graph

edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 3, 0, 4, 2],
    [1, 0, 2, 1, 3, 2, 4, 3, 0, 3, 2, 4],
])
edge_attr = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9],
                          [10], [11]])


def test_split():
    expected_output1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    expected_output2 = torch.tensor([[3, 4], [4, 3]])
    expected_output3 = torch.tensor([[2, 3, 3, 0, 4, 2], [3, 2, 0, 3, 2, 4]])
    expected_output4 = torch.tensor([[0], [1], [2], [3]])
    expected_output5 = torch.tensor([[6], [7]])
    expected_output6 = torch.tensor([[4], [5], [8], [9], [10], [11]])

    output1, output2, output3, output4, output5, output6 = split_graph(
        [0, 1, 2], [3, 4], edge_index, edge_attr, bridge=True)
    assert torch.equal(output1, expected_output1)
    assert torch.equal(output2, expected_output2)
    assert torch.equal(output3, expected_output3)
    assert torch.equal(output4, expected_output4)
    assert torch.equal(output5, expected_output5)
    assert torch.equal(output6, expected_output6)

    output7, output8, output9, output10 = split_graph([0, 1, 2], [3, 4],
                                                      edge_index, edge_attr,
                                                      bridge=False)
    assert torch.equal(output7, expected_output1)
    assert torch.equal(output8, expected_output2)
    assert torch.equal(output9, expected_output4)
    assert torch.equal(output10, expected_output5)


input_graph = tg.data.Data(
    edge_index=torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 3, 0, 4, 2],
        [1, 0, 2, 1, 3, 2, 4, 3, 0, 3, 2, 4],
    ]),
    x=torch.tensor([[0], [1], [2], [3], [4]]),
    edge_attr=torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9],
                            [10], [11]]),
    y=torch.tensor([0, 1, 2, 3, 4]),
)


def test_inductive_train_test_split():
    expected_output1 = tg.data.Data(
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.tensor([[0], [1], [2], [3]]),
        x=torch.tensor([[0], [1], [2]]),
        y=torch.tensor([0, 1, 2]),
    )
    expected_output2 = tg.data.Data(
        edge_index=torch.tensor([[3, 4], [4, 3]]),
        edge_attr=torch.tensor([[6], [7]]),
        x=torch.tensor([[3], [4]]),
        y=torch.tensor([3, 4]),
    )
    expected_output3 = tg.data.Data(
        edge_index=torch.tensor([[3, 4], [4, 3]]),
        edge_attr=torch.tensor([[6], [7]]),
        x=torch.tensor([[3], [4]]),
        y=torch.tensor([3, 4]),
        bridge_edge_index=torch.tensor([[2, 3, 3, 0, 4, 2], [3, 2, 0, 3, 2,
                                                             4]]),
        bridge_edge_attr=torch.tensor([[4], [5], [8], [9], [10], [11]]),
    )
    output1, output2 = inductive_train_test_split(input_graph, [0, 1, 2],
                                                  [3, 4], bridge=False)
    output3, output4 = inductive_train_test_split(input_graph, [0, 1, 2],
                                                  [3, 4])
    output5, output6 = inductive_train_test_split(
        input_graph,
        [True, True, True, False, False],
        [False, False, False, True, True],
        bridge=False,
    )

    assert torch.equal(output1.edge_index, expected_output1.edge_index)
    assert torch.equal(output1.edge_attr, expected_output1.edge_attr)
    assert torch.equal(output1.x, expected_output1.x)
    assert torch.equal(output1.y, expected_output1.y)
    assert torch.equal(output2.edge_index, expected_output2.edge_index)
    assert torch.equal(output2.edge_attr, expected_output2.edge_attr)
    assert torch.equal(output2.x, expected_output2.x)
    assert torch.equal(output2.y, expected_output2.y)
    assert torch.equal(output2.y, expected_output2.y)
    assert torch.equal(output3.edge_index, expected_output1.edge_index)
    assert torch.equal(output3.edge_attr, expected_output1.edge_attr)
    assert torch.equal(output3.x, expected_output1.x)
    assert torch.equal(output3.y, expected_output1.y)
    assert torch.equal(output4.bridge_edge_index,
                       expected_output3.bridge_edge_index)
    assert torch.equal(output4.bridge_edge_attr,
                       expected_output3.bridge_edge_attr)
    assert torch.equal(output4.edge_index, expected_output3.edge_index)
    assert torch.equal(output4.edge_attr, expected_output3.edge_attr)
    assert torch.equal(output4.x, expected_output3.x)
    assert torch.equal(output4.y, expected_output3.y)
    assert torch.equal(output4.bridge_edge_index,
                       expected_output3.bridge_edge_index)
    assert torch.equal(output4.bridge_edge_attr,
                       expected_output3.bridge_edge_attr)
    assert torch.equal(output5.edge_index, expected_output1.edge_index)
    assert torch.equal(output5.edge_attr, expected_output1.edge_attr)
    assert torch.equal(output5.x, expected_output1.x)
    assert torch.equal(output5.y, expected_output1.y)
    assert torch.equal(output6.edge_index, expected_output2.edge_index)
    assert torch.equal(output6.edge_attr, expected_output2.edge_attr)
    assert torch.equal(output6.x, expected_output2.x)
    assert torch.equal(output6.y, expected_output2.y)

    with pytest.raises(Exception) as e:
        inductive_train_test_split(input_graph, [0, 1, 2], [1, 3, 4])
    assert str(e.value) == "train/test node must be disjoint"

    with pytest.raises(Exception) as e:
        inductive_train_test_split(input_graph, [0, 1], [3, 4])
    assert str(e.value) == "train/test node must cover all nodes"
