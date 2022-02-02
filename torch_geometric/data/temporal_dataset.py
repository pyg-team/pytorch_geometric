import copy

import numpy as np

from torch_geometric.data import InMemoryDataset


class TemporalDataset(InMemoryDataset):
    r"""Dataset base class for creating temporal graph datasets which easily 
    fit into CPU memory.
    Inherits from :class:`torch_geometric.data.InMemoryDataset`.

    To create a temporal graph please inherits from this class. An example
    can be seem in :class:`torch_geometric.dataset.JODIEDataset`.
    """
    def _get_partial_dataset(self, idx):
        dataset = copy.copy(self)
        dataset.data = self.data[idx]
        return dataset

    def train_val_test_split(self, val_ratio=0.15, test_ratio=0.15):
        r"""Split the dataset in 3 parts new datasets: training, validation
        and test.

        Args:
            val_ratio (float, optional): The proportion (in percents) of the
                dataset to include in the validation split. (default: 
                :float:`0.15`)
            test_ratio (float, optional): The proportion (in percents) of the
                dataset to include in the test split. (default: :float:`0.15`)
        """
        val_time, test_time = np.quantile(
            self.data.t.cpu().numpy(),
            [1. - val_ratio - test_ratio, 1. - test_ratio])

        val_idx = int((self.data.t <= val_time).sum())
        test_idx = int((self.data.t <= test_time).sum())

        train = self._get_partial_dataset(slice(None, val_idx, None))
        val = self._get_partial_dataset(slice(val_idx, test_idx, None))
        test = self._get_partial_dataset(slice(test_idx, None, None))
        return train, val, test

    def __len__(self):
        return len(self.data.t)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        return f'{self.name.capitalize()}({len(self)})'
