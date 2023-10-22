from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_geometric.utils import subgraph
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes


def split_graph(
    subset1: Union[Tensor, List[int]],
    subset2: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    num_nodes: Optional[int] = None,
    bridge: bool = True,
) -> Tuple[Tensor, Tensor, OptTensor, OptTensor, OptTensor, OptTensor]:
    r"""
    Given a graph and two node subsets, this function returns the subgraphs
    edges induced by the two subsets, as well as the bridge edges between them
    (if :obj:`bridge` = True).

    Args:
        edge_index (Tensor): The edge indices of the graph to be split.
        edge_attr (OptTensor, optional): The edge attributes of the graph
                    to be split. (default: :obj:`None`)
        subset1 (Union[Tensor, List[int]]): the first subset of nodes to keep.
                    It should be a boolean mask (Tensor) or
                    a list of indices (List[int]).
        subset2 (Union[Tensor, List[int]]): the second subset of nodes to keep.
                    It should be a boolean mask (Tensor) or
                    a list of indices (List[int]).
        num_nodes (Optional[int], optional): The number of nodes in the graph.
                    (default: :obj:`None`)
        bridge (bool, optional): whether to return the bridge edges between
                    the two subsets. (default: :obj:`True`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`OptTensor`,
        :class:`OptTensor`, :class:`OptTensor`, :class:`OptTensor`)
        if :obj:`bridge` = True,
            (:class:`Tensor`, :class:`Tensor`, :class:`OptTensor`,
        :class:`OptTensor`) otherwise.

    Examples:

        >>> edge_index=torch.tensor(
        ...     [[0, 1, 1, 2, 2, 3, 3, 4, 3, 0, 4, 2],
        ...     [1, 0, 2, 1, 3, 2, 4, 3, 0, 3, 2, 4]]
        ... )
        ... edge_attr=torch.tensor([[0], [1], [2], [3], [4], [5],
        ...                        [6], [7], [8], [9], [10], [11]])
        >>> split_graph([0, 1, 2], [3, 4], edge_index, edge_attr, bridge=True)
        (
            torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            torch.tensor([[3, 4], [4, 3]]),
            torch.tensor([[2, 3, 3, 0, 4, 2], [3, 2, 0, 3, 2, 4]]),
            torch.tensor([[0], [1], [2], [3]]),
            torch.tensor([[6], [7]]),
            torch.tensor([[4], [5], [8], [9], [10], [11]]),
        )
        >>> split_graph([0, 1, 2], [3, 4], edge_index, edge_attr, bridge=False)
        (
            torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            torch.tensor([[3, 4], [4, 3]]),
            torch.tensor([[0], [1], [2], [3]]),
            torch.tensor([[6], [7]]),
        )

    """

    device = edge_index.device

    if isinstance(subset1, (list, tuple)):
        subset1 = torch.tensor(subset1, dtype=torch.long, device=device)

    if isinstance(subset2, (list, tuple)):
        subset2 = torch.tensor(subset2, dtype=torch.long, device=device)

    if subset1.dtype != torch.bool:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask1 = index_to_mask(subset1, size=num_nodes)
    else:
        num_nodes = subset1.size(0)
        node_mask1 = subset1

    if subset2.dtype != torch.bool:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask2 = index_to_mask(subset2, size=num_nodes)
    else:
        num_nodes = subset2.size(0)
        node_mask2 = subset2

    edge_index1, edge_attr1 = subgraph(node_mask1, edge_index, edge_attr,
                                       return_edge_mask=False)
    edge_index2, edge_attr2 = subgraph(node_mask2, edge_index, edge_attr,
                                       return_edge_mask=False)
    if bridge:
        bridge_mask = (node_mask1[edge_index[0]] & node_mask2[edge_index[1]]
                       ) | (node_mask2[edge_index[0]]
                            & node_mask1[edge_index[1]])
        bridge_edge_index = edge_index[:, bridge_mask]
        bridge_edge_attr = edge_attr[bridge_mask]
        return (
            edge_index1,
            edge_index2,
            bridge_edge_index,
            edge_attr1,
            edge_attr2,
            bridge_edge_attr,
        )
    else:
        return (
            edge_index1,
            edge_index2,
            edge_attr1,
            edge_attr2,
        )


def inductive_train_test_split(data, train_node, test_node,
                               bridge=True) -> Tuple[Data, Data]:
    r"""
    Given a graph and two node subsets, this function returns the train and
    test subgraphs data induced by the two subsets, as well as the bridge edges
    between them (if :obj:`bridge` = True) and also splits the node features
    and labels. train_node and test_node must be disjoint and cover all nodes.

    Args:
        data (Data): Data object to be split.
        train_node (_type_): the train subset of nodes to keep. It should be a
                boolean mask (Tensor) or a list of indices (List[int]).
        test_node (_type_): the test subset of nodes to keep. It should be a
                boolean mask (Tensor) or a list of indices (List[int]).
                train_node and test_node must be disjoint and cover all nodes.
        bridge (bool, optional): whether to return the bridge edges between
                the two subsets in test data features
                as bridge_edge_index and bridge_edge_attr.
                (default: :obj:`True`)

    Raises:
        ValueError: train/test node must be disjoint
        ValueError: train/test node must cover all nodes

    :rtype: (:class:`Data`, :class:`Data`)

    Examples:

        >>> input_graph = Data(
        ...     edge_index=torch.tensor(
        ...         [[0, 1, 1, 2, 2, 3, 3, 4, 3, 0, 4, 2],
        ...         [1, 0, 2, 1, 3, 2, 4, 3, 0, 3, 2, 4]]
        ...     ),
        ...     x=torch.tensor([[0], [1], [2], [3], [4]]),
        ...     edge_attr=torch.tensor([[0], [1], [2], [3], [4], [5],
        ...                         [6], [7], [8], [9], [10], [11]]),
        ...     y=torch.tensor([0, 1, 2, 3, 4]),
        ... )
        >>> inductive_train_test_split(input_graph, [0, 1, 2], [3, 4])
        (Data(
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_attr=torch.tensor([[0], [1], [2], [3]]),
            x=torch.tensor([[0], [1], [2]]),
            y=torch.tensor([0, 1, 2]),
        ),
        Data(
            edge_index=torch.tensor([[3, 4], [4, 3]]),
            edge_attr=torch.tensor([[6], [7]]),
            x=torch.tensor([[3], [4]]),
            y=torch.tensor([3, 4]),
            bridge_edge_index=torch.tensor([[2, 3, 3, 0, 4, 2],
                                            [3, 2, 0, 3, 2, 4]]),
            bridge_edge_attr=torch.tensor([[4], [5], [8], [9], [10], [11]]),
        ))
        >>> inductive_train_test_split(input_graph, [0, 1, 2],
                                            [3, 4], bridge=False)
        (Data(
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_attr=torch.tensor([[0], [1], [2], [3]]),
            x=torch.tensor([[0], [1], [2]]),
            y=torch.tensor([0, 1, 2]),
        ),
        Data(
            edge_index=torch.tensor([[3, 4], [4, 3]]),
            edge_attr=torch.tensor([[6], [7]]),
            x=torch.tensor([[3], [4]]),
            y=torch.tensor([3, 4]),
        ))
    """

    device = data.edge_index.device

    if isinstance(train_node, (list, tuple)):
        train_node = torch.tensor(train_node, dtype=torch.long, device=device)

    if isinstance(test_node, (list, tuple)):
        test_node = torch.tensor(test_node, dtype=torch.long, device=device)

    if train_node.dtype != torch.bool:
        num_nodes = maybe_num_nodes(data.edge_index, data.num_nodes)
        node_mask1 = index_to_mask(train_node, size=num_nodes)
    else:
        num_nodes = train_node.size(0)
        node_mask1 = train_node

    if test_node.dtype != torch.bool:
        num_nodes = maybe_num_nodes(data.edge_index, data.num_nodes)
        node_mask2 = index_to_mask(test_node, size=num_nodes)
    else:
        num_nodes = test_node.size(0)
        node_mask2 = test_node

    if (node_mask1 & node_mask2).any():
        raise ValueError("train/test node must be disjoint")
    if not (node_mask1 | node_mask2).all():
        raise ValueError("train/test node must cover all nodes")

    if bridge:
        (
            train_edge_index,
            test_edge_index,
            bridge_edge_index,
            train_edge_attr,
            test_edge_attr,
            bridge_edge_attr,
        ) = split_graph(
            node_mask1,
            node_mask2,
            data.edge_index,
            data.edge_attr,
            num_nodes=data.num_nodes,
            bridge=True,
        )
        train_data = Data(
            edge_index=train_edge_index,
            edge_attr=train_edge_attr,
        )
        test_data = Data(
            edge_index=test_edge_index,
            edge_attr=test_edge_attr,
            bridge_edge_index=bridge_edge_index,
            bridge_edge_attr=bridge_edge_attr,
        )

    else:
        (
            train_edge_index,
            test_edge_index,
            train_edge_attr,
            test_edge_attr,
        ) = split_graph(
            node_mask1,
            node_mask2,
            data.edge_index,
            data.edge_attr,
            num_nodes=data.num_nodes,
            bridge=False,
        )
        train_data = Data(
            edge_index=train_edge_index,
            edge_attr=train_edge_attr,
        )
        test_data = Data(
            edge_index=test_edge_index,
            edge_attr=test_edge_attr,
        )

    if data.x is not None:
        train_data.x = data.x[train_node]
        test_data.x = data.x[test_node]
    if data.y is not None:
        train_data.y = data.y[train_node]
        test_data.y = data.y[test_node]

    return train_data, test_data
