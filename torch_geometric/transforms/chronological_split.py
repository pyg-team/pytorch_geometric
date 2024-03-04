import copy
from typing import List, Tuple, Union

import numpy as np

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import NodeOrEdgeType


@functional_transform('chronological_split')
class ChronologicalSplit(BaseTransform):
    r"""Performs a chronological split into training, validation and test sets
    of a :class:`~torch_geometric.data.Data` or a
    :class:`~torch_geometric.data.HeteroData` object
    (functional name: :obj:`chronological_split`).

    Chronological split satisfies the following relationship:
    max(train_time) < min(val_time) && max(val_time) < min(test_time)

    Args:
        val_ratio (float, optional): The proportion (in percents) of the
            dataset to include in the validation split.
            (default: :obj:`0.15`)
        test_ratio (float, optional): The proportion (in percents) of the
            dataset to include in the test split. (default: :obj:`0.15`)
    """
    def __init__(self, val_ratio: float = 0.15,
                 test_ratio: float = 0.15) -> None:
        assert 0. <= val_ratio <= 1.
        assert 0. <= test_ratio <= 1.
        assert val_ratio + test_ratio < 1.
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.key = 'time'

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Tuple[
            Union[Data, HeteroData],
            Union[Data, HeteroData],
            Union[Data, HeteroData],
    ]:
        if isinstance(data, Data):
            return self._forward_data(data)
        else:  # HeteroData
            return self._forward_hetero_data(data)

    def _forward_data(self, data: Data) -> Tuple[Data, Data, Data]:
        assert hasattr(data, self.key)

        val_time, test_time = np.quantile(
            data[self.key].numpy(),
            [1 - self.val_ratio - self.test_ratio, 1 - self.test_ratio])
        train_data = data.snapshot(0, val_time)
        val_data = data.snapshot(val_time, test_time)
        test_data = data.snapshot(test_time, data[self.key].max().item() + 1)

        return train_data, val_data, test_data

    def _forward_hetero_data(
            self,
            data: HeteroData) -> Tuple[HeteroData, HeteroData, HeteroData]:
        train_data = copy.copy(data)
        val_data = copy.copy(data)
        test_data = copy.copy(data)

        all_types: List[NodeOrEdgeType] = data.node_types + data.edge_types
        for key in all_types:
            if hasattr(data[key], self.key):
                val_time, test_time = np.quantile(data[key].time.numpy(), [
                    1 - self.val_ratio - self.test_ratio, 1 - self.test_ratio
                ])
                train_data[key].snapshot(0, val_time)
                val_data[key].snapshot(val_time, test_time)
                test_data[key].snapshot(test_time,
                                        data[key].time.max().item() + 1)

        return train_data, val_data, test_data
