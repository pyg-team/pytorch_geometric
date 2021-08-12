from typing import Union, Tuple

import copy

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import BaseTransform


class RandomLinkSplit(BaseTransform):
    r"""Performs an edge-level random split into training, validation and test
    sets.
    The split is performed such that the training split does not include edges
    in validation and test splits; and the validation split does not include
    edges in the test split.

    .. code-block::

        from torch_geometric.transforms import RandomLinkSplit

        transform = RandomLinkSplit(is_undirected=True)
        train_data, val_data, test_data = transform(data)

    Args:
        num_val (int or float, optional): The number of validation edges.
            If set to a floating-point value in :math:` [0, 1]`, it represents
            the ratio of edges to include in the validation set.
            (default: :obj:`0.1`)
        num_test (int or float, optional): The number of test edges.
            If set to a floating-point value in :math:`[0, 1]`, it represents
            the ratio of edges to include in the test set.
            (default: :obj:`0.2`)
        is_undirected (bool): If set to :obj:`True`, the graph is assumed to be
            undirected. (default: :obj:`False`)
        key (str, optional): The name of the attribute holding
            ground-truth labels.
            If :obj:`data[key]` does not exist, it will be automatically
            created and represents a binary classification task
            (:obj:`1` = edge, :obj:`0` = no edge).
            If :obj:`data[key]` exists, it has to be a categorical label from
            :obj:`0` to :obj:`num_classes - 1`.
            After negative sampling, label :obj:`0` represents negative edges,
            and labels :obj:`1` to :obj:`num_classes` represent the labels of
            positive edges. (default: :obj:`"edge_label"`)
        add_negative_train_samples (bool, optional): Whether to add negative
            training samples for link prediction.
            If the model already performs negative sampling, then the option
            should be set to :obj:`False`.
            Otherwise, the added negative samples will be the same across
            training iterations unless negative sampling is performed again.
            (default: :obj:`True`)
        neg_sampling_ratio: (float, optional): The ratio of sampled negative
            edges to the number of positive edges. (default: :obj:`1.0`)
    """
    def __init__(
        self,
        num_val: Union[int, float] = 0.1,
        num_test: Union[int, float] = 0.2,
        is_undirected: bool = False,
        key: str = 'edge_label',
        add_negative_train_samples: bool = True,
        neg_sampling_ratio: float = 1.0,
    ):
        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = is_undirected
        self.key = key
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio

    def __call__(self, data: Data) -> Tuple[Data, Data, Data]:
        perm = torch.randperm(data.num_edges)
        if self.is_undirected:
            perm = perm[data.edge_index[0] <= data.edge_index[1]]

        num_val, num_test = self.num_val, self.num_test
        if isinstance(num_val, float):
            num_val = int(num_val * perm.numel())
        if isinstance(num_test, float):
            num_test = int(num_test * perm.numel())

        num_train = perm.numel() - num_val - num_test
        if num_train <= 0:
            raise ValueError("Insufficient number of edges for training.")

        train_edges = perm[:num_train]
        val_edges = perm[num_train:num_train + num_val]
        test_edges = perm[num_train + num_val:]
        train_val_edges = perm[:num_train + num_val]

        # Create data splits:
        train_data = self._split_data(data, train_edges)
        val_data = self._split_data(data, train_edges)
        test_data = self._split_data(data, train_val_edges)

        # Create negative samples:
        num_neg_train = 0
        if self.add_negative_train_samples:
            num_neg_train = int(num_train * self.neg_sampling_ratio)
        num_neg_val = int(num_val * self.neg_sampling_ratio)
        num_neg_test = int(num_test * self.neg_sampling_ratio)

        num_neg = num_neg_train + num_neg_val + num_neg_test
        neg_edge_index = negative_sampling(data.edge_index,
                                           num_nodes=data.num_nodes,
                                           num_neg_samples=num_neg,
                                           method='sparse')

        # Create labels:
        key_index = f'{self.key}_index'
        assert not hasattr(data, key_index)
        train_data[key_index], train_data[self.key] = self._create_label(
            data, train_edges, self.key,
            neg_edge_index[:, num_neg_val + num_neg_test:])
        val_data[key_index], val_data[self.key] = self._create_label(
            data, val_edges, self.key,
            neg_edge_index=neg_edge_index[:, :num_neg_val])
        test_data[key_index], test_data[self.key] = self._create_label(
            data, test_edges, self.key,
            neg_edge_index[:, num_neg_val:num_neg_val + num_neg_test])

        return train_data, val_data, test_data

    def _split(self, edge_index: Tensor, index: Tensor) -> Tensor:
        edge_index = edge_index[:, index]

        if self.is_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)

        return edge_index

    def _split_data(self, data: Data, index: Tensor) -> Data:
        num_edges = data.num_edges

        data = copy.copy(data)
        data.edge_index = self._split(data.edge_index, index)

        for key, value in data.items():
            if isinstance(value, Tensor) and value.size(0) == num_edges:
                value = value[index]
                if self.is_undirected:
                    value = torch.cat([value, value], dim=0)
                data[key] = value

        return data

    def _create_label(self, data: Data, index: Tensor, key: str,
                      neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:

        edge_index = data.edge_index[:, index]

        if hasattr(data, key):
            edge_label = data[key]
            assert edge_label.dtype == torch.long and edge_label.dim() == 1
            edge_label = edge_label[index] + 1
        else:
            edge_label = torch.ones(index.numel())

        if neg_edge_index is not None and neg_edge_index.numel() > 0:
            edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)
            neg_edge_label = torch.zeros(neg_edge_index.size(1),
                                         dtype=edge_label.dtype)
            edge_label = torch.cat([edge_label, neg_edge_label], dim=0)

        return edge_index, edge_label

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_val={self.num_val}, '
                f'num_test={self.num_test})')
