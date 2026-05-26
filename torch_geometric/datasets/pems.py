import os.path as osp
from typing import List

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class PEMS(InMemoryDataset):
    r"""Curated graph from the PEMS road network dataset;
    Review `"Matérn Gaussian Processes on Graphs"
    <https://arxiv.org/abs/2010.15538>`_ for details and
    `"Do Deep Ensembles Actually Capture Uncertainty
    in Graph Neural Networks?" <https://arxiv.org/abs/2605.22593>`_ for
    extended usage with GNNs.

    The graph is homogeneous (1016 nodes, 1172 edges),
    topologically challenging (low average degree, elongated structure)
    and commonly used for uncertainty quantification.

    Nodes correspond to traffic sensors recording vehicle speeds.
    The task is to predict speeds at designated nodes (node regression).
    The graph carries no natural node features; training targets are instead
    used as input features, with test nodes masked to zero.
    Using :obj:`validation_percentage` equal to zero (default) reproduces
    the splits from the referenced papers. If a validation set is needed
    (e.g., for hyperparameter tuning), we advise retraining on the full
    training data before reporting test performance.

    .. note::
        The targets are standardised by default. Access the original standard
        deviation and mean with properties :obj:`original_std` and
        :obj:`original_mean`.

    .. note::
        Targets are stored as :obj:`torch.float64` for consistency with
        previous works and to better support numerical methods common
        in uncertainty quantification (e.g., Gaussian processes) where
        :obj:`torch.float32` precision may be insufficient.
        Users can cast via :obj:`data.y.float()` if needed.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        validation_percentage (float, optional): Controls the amount of
            train data used for validation (default: :obj:`0`)

    """

    NUM_TRAIN = 250
    _url = "https://zenodo.org/records/20394854/files"

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        force_reload=False,
        validation_percentage=0.0,
    ) -> None:
        self.validation_percentage = validation_percentage

        super().__init__(root, transform, pre_transform, force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "edge_index.pt",
            "edge_features.pt",
            "node_coordinates.pt",
            "valid_label_index_data.npz",
        ]

    @property
    def processed_file_names(self) -> str:
        return f"transformed_data_{self.validation_percentage}.pt"

    @property
    def original_std(self) -> float:
        """The standard deviation of the original (unstandardised) targets."""
        return self._data.original_std

    @property
    def original_mean(self) -> float:
        """The mean of the original (unstandardised) targets."""
        return self._data.original_mean

    def download(self) -> None:
        for filename in self.raw_file_names:
            download_url(f"{self._url}/{filename}", self.raw_dir)

    def process(self) -> None:
        loaded_data = np.load(
            osp.join(self.raw_dir, "valid_label_index_data.npz"))
        data = tuple(loaded_data[f"arr_{i}"]
                     for i in range(len(loaded_data.files)))

        # Self contained legacy rng to match curated random split
        legacy_rng = np.random.RandomState(1111)

        # data[0] contains the index of the labels that are valid (non NaN)
        random_perm = legacy_rng.permutation(np.arange(data[0].shape[0]))
        num_val = int(self.NUM_TRAIN * self.validation_percentage)
        num_train_final = self.NUM_TRAIN - num_val

        train_vertex = random_perm[:num_train_final]
        validation_vertex = random_perm[num_train_final:num_train_final +
                                        num_val]
        test_vertex = random_perm[self.NUM_TRAIN:]

        # For index purposes only, not real data
        xs_train = torch.tensor(data[0][train_vertex], dtype=torch.int64)
        ys_train = torch.tensor(data[1][train_vertex], dtype=torch.float64)

        xs_validation = torch.tensor(data[0][validation_vertex],
                                     dtype=torch.int64)
        ys_validation = torch.tensor(data[1][validation_vertex],
                                     dtype=torch.float64)

        xs_test = torch.tensor(data[0][test_vertex], dtype=torch.int64)
        ys_test = torch.tensor(data[1][test_vertex], dtype=torch.float64)

        num_nodes = int(np.max(data[0])) + 1

        # Note: since there is lots of _actually_ unknown y-s, the `ys`
        # array will have lots of NaN-s.
        ys = torch.full((num_nodes, ), np.nan, dtype=torch.float64)
        ys[xs_train.squeeze()] = ys_train.squeeze()
        ys[xs_validation.squeeze()] = ys_validation.squeeze()
        ys[xs_test.squeeze()] = ys_test.squeeze()
        ys = ys[:, None]

        edge_index = torch.load(osp.join(self.raw_dir, "edge_index.pt"),
                                weights_only=True)
        edge_features = torch.load(osp.join(self.raw_dir, "edge_features.pt"),
                                   weights_only=True)

        train_mask = torch.full((num_nodes, ), False, dtype=torch.bool)
        validation_mask = torch.full((num_nodes, ), False, dtype=torch.bool)
        test_mask = torch.full((num_nodes, ), False, dtype=torch.bool)

        test_mask[xs_test.squeeze()] = True
        train_mask[xs_train.squeeze()] = True
        validation_mask[xs_validation.squeeze()] = True

        original_mean = torch.mean(ys[train_mask], dim=0)
        original_std = torch.std(ys[train_mask], dim=0)
        standard_ys = (ys - original_mean) / original_std

        masked_standard_ys = torch.nan_to_num(standard_ys)
        masked_standard_ys[test_mask] = 0.0
        masked_standard_ys[validation_mask] = 0.0

        spatial_position = torch.load(
            osp.join(self.raw_dir, "node_coordinates.pt"), weights_only=True)

        pyg_data = Data(
            x=masked_standard_ys,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=standard_ys,
            pos=spatial_position,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=validation_mask,
            original_std=original_std,
            original_mean=original_mean,
        )

        pyg_data = (pyg_data if self.pre_transform is None else
                    self.pre_transform(pyg_data))
        self.save([pyg_data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"
