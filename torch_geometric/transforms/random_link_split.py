from typing import Union, Tuple

import copy

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, negative_sampling
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
            If set to a floating-point value in :math:`[0, 1]`, it represents
            the ratio of edges to include in the validation set.
            (default: :obj:`0.1`)
        num_test (int or float, optional): The number of test edges.
            If set to a floating-point value in :math:`[0, 1]`, it represents
            the ratio of edges to include in the test set.
            (default: :obj:`0.2`)
        is_undirected (bool): If set to :obj:`True`, the graph is assumed to be
            undirected, and positive and negative samples will not leak
            (reverse) edge connectivity across different splits.
            (default: :obj:`False`)
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
        split_labels (bool, optional): If set to :obj:`True`, will split
            positive and negative labels and save them in distinct attributes
            :obj:`"pos_edge_label"` and :obj:`"neg_edge_label"`, respectively.
            (default: :obj:`False`)
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
        split_labels: bool = False,
        add_negative_train_samples: bool = True,
        neg_sampling_ratio: float = 1.0,
    ):
        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = is_undirected
        self.key = key
        self.split_labels = split_labels
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio

    def __call__(self, data: Data) -> Tuple[Data, Data, Data]:
        perm = torch.randperm(data.num_edges, device=data.edge_index.device)
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
        neg_edge_index = negative_sampling(
            add_self_loops(data.edge_index)[0], num_nodes=data.num_nodes,
            num_neg_samples=num_neg, method='sparse')

        # Create labels:
        self._create_label(
            data,
            train_edges,
            neg_edge_index[:, num_neg_val + num_neg_test:],
            out=train_data,
        )
        self._create_label(
            data,
            val_edges,
            neg_edge_index[:, :num_neg_val],
            out=val_data,
        )
        self._create_label(
            data,
            test_edges,
            neg_edge_index[:, num_neg_val:num_neg_val + num_neg_test],
            out=test_data,
        )

        return train_data, val_data, test_data

    def _split(self, edge_index: Tensor, index: Tensor) -> Tensor:
        edge_index = edge_index[:, index]

        if self.is_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)

        return edge_index

    def _split_data(self, data: Data, index: Tensor) -> Data:
        splitted_data = copy.copy(data)
        splitted_data.edge_index = self._split(data.edge_index, index)

        for key, value in data.items():
            if key == 'edge_index':
                continue
            if isinstance(value, Tensor) and data.is_edge_attr(key):
                value = value[index]
                if self.is_undirected:
                    value = torch.cat([value, value], dim=0)
                splitted_data[key] = value

        return splitted_data

    def _create_label(self, data: Data, index: Tensor, neg_edge_index: Tensor,
                      out: Data):

        edge_index = data.edge_index[:, index]

        if hasattr(data, self.key):
            edge_label = data[self.key]
            assert edge_label.dtype == torch.long and edge_label.dim() == 1
            edge_label = edge_label[index].add_(1)
            delattr(data, self.key)
        else:
            edge_label = torch.ones(index.numel(), device=index.device)

        if neg_edge_index.numel() > 0:
            neg_edge_label = edge_label.new_zeros(neg_edge_index.size(1))

        if self.split_labels:
            out[f'pos_{self.key}'] = edge_label
            out[f'pos_{self.key}_index'] = edge_index
            if neg_edge_index.numel() > 0:
                out[f'neg_{self.key}'] = neg_edge_label
                out[f'neg_{self.key}_index'] = neg_edge_index

        else:
            if neg_edge_index.numel() > 0:
                edge_label = torch.cat([edge_label, neg_edge_label], dim=0)
                edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)
            out[self.key] = edge_label
            out[f'{self.key}_index'] = edge_index

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_val={self.num_val}, '
                f'num_test={self.num_test})')
