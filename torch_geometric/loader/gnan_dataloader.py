from collections.abc import Sequence
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader as PyTorchDataLoader

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter


def _create_block_diagonal_matrix(
        matrix_list: List[torch.Tensor]) -> torch.Tensor:
    r"""Create a block diagonal matrix from a list of matrices."""
    if not matrix_list:
        return torch.empty(0, 0)

    total_size = sum(matrix.size(0) for matrix in matrix_list)
    result = torch.zeros(
        total_size,
        total_size,
        dtype=matrix_list[0].dtype,
        device=matrix_list[0].device,
    )

    offset = 0
    for matrix in matrix_list:
        size = matrix.size(0)
        result[offset:offset + size, offset:offset + size] = matrix
        offset += size

    return result


class GNANCollater:
    def __init__(
        self,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, data_list: List[BaseData]) -> Batch:
        node_distances_list = []
        normalization_matrix_list = []

        has_node_distances = hasattr(data_list[0], 'node_distances')
        has_normalization_matrix = hasattr(data_list[0],
                                           'normalization_matrix')

        if has_node_distances:
            for data in data_list:
                node_distances_list.append(data.node_distances)
                delattr(data, 'node_distances')

        if has_normalization_matrix:
            for data in data_list:
                normalization_matrix_list.append(data.normalization_matrix)
                delattr(data, 'normalization_matrix')

        batch = Batch.from_data_list(
            data_list,
            follow_batch=self.follow_batch,
            exclude_keys=self.exclude_keys,
        )

        for i, data in enumerate(data_list):
            if has_node_distances:
                data.node_distances = node_distances_list[i]
            if has_normalization_matrix:
                data.normalization_matrix = normalization_matrix_list[i]

        if node_distances_list:
            batch.node_distances = _create_block_diagonal_matrix(
                node_distances_list)
        if normalization_matrix_list:
            batch.normalization_matrix = _create_block_diagonal_matrix(
                normalization_matrix_list)

        return batch


class GNANDataLoader(PyTorchDataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch, specifically for
    use with the :class:`torch_geometric.nn.models.TensorGNAN` model.

    This loader will batch the :obj:`node_distances` and
    :obj:`normalization_matrix` attributes of
    :class:`~torch_geometric.data.Data` objects by creating large block-
    diagonal matrices.

    For this to work, every data object in the dataset needs to have the
    attributes :obj:`node_distances` and :obj:`normalization_matrix`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=GNANCollater(follow_batch, exclude_keys),
            **kwargs,
        )
