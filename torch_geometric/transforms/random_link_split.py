from copy import copy
from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType
from torch_geometric.utils import negative_sampling


@functional_transform('random_link_split')
class RandomLinkSplit(BaseTransform):
    r"""Performs an edge-level random split into training, validation and test
    sets of a :class:`~torch_geometric.data.Data` or a
    :class:`~torch_geometric.data.HeteroData` object
    (functional name: :obj:`random_link_split`).
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
        neg_sampling_ratio (float, optional): The ratio of sampled negative
            edges to the number of positive edges. (default: :obj:`1.0`)
        disjoint_train_ratio (int or float, optional): If set to a value
            greater than :obj:`0.0`, training edges will not be shared for
            message passing and supervision. Instead,
            :obj:`disjoint_train_ratio` edges are used as ground-truth labels
            for supervision during training. (default: :obj:`0.0`)
        edge_types (Tuple[EdgeType] or List[EdgeType], optional): The edge
            types used for performing edge-level splitting in case of
            operating on :class:`~torch_geometric.data.HeteroData` objects.
            (default: :obj:`None`)
        rev_edge_types (Tuple[EdgeType] or List[Tuple[EdgeType]], optional):
            The reverse edge types of :obj:`edge_types` in case of operating
            on :class:`~torch_geometric.data.HeteroData` objects.
            This will ensure that edges of the reverse direction will be
            splitted accordingly to prevent any data leakage.
            Can be :obj:`None` in case no reverse connection exists.
            (default: :obj:`None`)
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
        disjoint_train_ratio: Union[int, float] = 0.0,
        edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
        rev_edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
    ):
        if isinstance(edge_types, list):
            if rev_edge_types is None:
                rev_edge_types = [None] * len(edge_types)

            assert isinstance(rev_edge_types, list)
            assert len(edge_types) == len(rev_edge_types)

        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = is_undirected
        self.key = key
        self.split_labels = split_labels
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio
        self.disjoint_train_ratio = disjoint_train_ratio
        self.edge_types = edge_types
        self.rev_edge_types = rev_edge_types

    def __call__(self, data: Union[Data, HeteroData]):
        edge_types = self.edge_types
        rev_edge_types = self.rev_edge_types

        train_data, val_data, test_data = copy(data), copy(data), copy(data)

        if isinstance(data, HeteroData):
            if edge_types is None:
                raise ValueError(
                    "The 'RandomLinkSplit' transform expects 'edge_types' to"
                    "be specified when operating on 'HeteroData' objects")

            if not isinstance(edge_types, list):
                edge_types = [edge_types]
                rev_edge_types = [rev_edge_types]

            stores = [data[edge_type] for edge_type in edge_types]
            train_stores = [train_data[edge_type] for edge_type in edge_types]
            val_stores = [val_data[edge_type] for edge_type in edge_types]
            test_stores = [test_data[edge_type] for edge_type in edge_types]
        else:
            rev_edge_types = [None]
            stores = [data._store]
            train_stores = [train_data._store]
            val_stores = [val_data._store]
            test_stores = [test_data._store]

        for item in zip(stores, train_stores, val_stores, test_stores,
                        rev_edge_types):
            store, train_store, val_store, test_store, rev_edge_type = item

            is_undirected = self.is_undirected
            is_undirected &= not store.is_bipartite()
            is_undirected &= rev_edge_type is None

            edge_index = store.edge_index
            if is_undirected:
                mask = edge_index[0] <= edge_index[1]
                perm = mask.nonzero(as_tuple=False).view(-1)
                perm = perm[torch.randperm(perm.size(0), device=perm.device)]
            else:
                device = edge_index.device
                perm = torch.randperm(edge_index.size(1), device=device)

            num_val = self.num_val
            if isinstance(num_val, float):
                num_val = int(num_val * perm.numel())
            num_test = self.num_test
            if isinstance(num_test, float):
                num_test = int(num_test * perm.numel())

            num_train = perm.numel() - num_val - num_test
            if num_train <= 0:
                raise ValueError("Insufficient number of edges for training")

            train_edges = perm[:num_train]
            val_edges = perm[num_train:num_train + num_val]
            test_edges = perm[num_train + num_val:]
            train_val_edges = perm[:num_train + num_val]

            num_disjoint = self.disjoint_train_ratio
            if isinstance(num_disjoint, float):
                num_disjoint = int(num_disjoint * train_edges.numel())
            if num_train - num_disjoint <= 0:
                raise ValueError("Insufficient number of edges for training")

            # Create data splits:
            self._split(train_store, train_edges[num_disjoint:], is_undirected,
                        rev_edge_type)
            self._split(val_store, train_edges, is_undirected, rev_edge_type)
            self._split(test_store, train_val_edges, is_undirected,
                        rev_edge_type)

            # Create negative samples:
            num_neg_train = 0
            if self.add_negative_train_samples:
                if num_disjoint > 0:
                    num_neg_train = int(num_disjoint * self.neg_sampling_ratio)
                else:
                    num_neg_train = int(num_train * self.neg_sampling_ratio)
            num_neg_val = int(num_val * self.neg_sampling_ratio)
            num_neg_test = int(num_test * self.neg_sampling_ratio)

            num_neg = num_neg_train + num_neg_val + num_neg_test

            size = store.size()
            if store._key is None or store._key[0] == store._key[-1]:
                size = size[0]
            neg_edge_index = negative_sampling(edge_index, size,
                                               num_neg_samples=num_neg,
                                               method='sparse')

            # Create labels:
            if num_disjoint > 0:
                train_edges = train_edges[:num_disjoint]
            self._create_label(
                store,
                train_edges,
                neg_edge_index[:, num_neg_val + num_neg_test:],
                out=train_store,
            )
            self._create_label(
                store,
                val_edges,
                neg_edge_index[:, :num_neg_val],
                out=val_store,
            )
            self._create_label(
                store,
                test_edges,
                neg_edge_index[:, num_neg_val:num_neg_val + num_neg_test],
                out=test_store,
            )

        return train_data, val_data, test_data

    def _split(self, store: EdgeStorage, index: Tensor, is_undirected: bool,
               rev_edge_type: EdgeType):

        for key, value in store.items():
            if key == 'edge_index':
                continue

            if store.is_edge_attr(key):
                value = value[index]
                if is_undirected:
                    value = torch.cat([value, value], dim=0)
                store[key] = value

        edge_index = store.edge_index[:, index]
        if is_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)
        store.edge_index = edge_index

        if rev_edge_type is not None:
            rev_store = store._parent()[rev_edge_type]
            for key in rev_store.keys():
                if key not in store:
                    del rev_store[key]  # We delete all outdated attributes.
                elif key == 'edge_index':
                    rev_store.edge_index = store.edge_index.flip([0])
                else:
                    rev_store[key] = store[key]

        return store

    def _create_label(self, store: EdgeStorage, index: Tensor,
                      neg_edge_index: Tensor, out: EdgeStorage):

        edge_index = store.edge_index[:, index]

        if hasattr(store, self.key):
            edge_label = store[self.key]
            edge_label = edge_label[index]
            # Increment labels by one. Note that there is no need to increment
            # in case no negative edges are added.
            if neg_edge_index.numel() > 0:
                assert edge_label.dtype == torch.long
                assert edge_label.size(0) == store.edge_index.size(1)
                edge_label.add_(1)
            if hasattr(out, self.key):
                delattr(out, self.key)
        else:
            edge_label = torch.ones(index.numel(), device=index.device)

        if neg_edge_index.numel() > 0:
            neg_edge_label = edge_label.new_zeros((neg_edge_index.size(1), ) +
                                                  edge_label.size()[1:])

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
