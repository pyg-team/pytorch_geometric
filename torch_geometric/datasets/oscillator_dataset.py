import os
import zipfile

import h5py
import numpy as np
import torch

from torch_geometric.data import Data as gData
from torch_geometric.data import InMemoryDataset as InMemoryDataset
from torch_geometric.data import download_url

zenodo_url = "https://zenodo.org/record/8204334/files/"
dataset020_url = zenodo_url + "ds20.zip"
dataset100_url = zenodo_url + "ds100.zip"
datasettexas_url = zenodo_url + "texas.zip"


class oscillatorDataset(InMemoryDataset):
    r"""Oscillator networks from
    <https://doi.org/10.1063/5.0160915>.

    The data has been published in the following papers:
    Toward dynamic stability assessment of power grid topologies using graph
    neural networks <https://doi.org/10.1063/5.0160915>
    Towards dynamic stability analysis of sustainable power grids using graph
    neural networks <https://www.climatechange.ai/papers/neurips2022/16>

    The data is available on Zenodo: <https://zenodo.org/records/8204334>

    There are three datasets: osc20, osc100, and osctexas. osc20 consists of
    10,000 grids with 20 nodes per grid, osc100 consists of 10,000 grids with
    100 nodes each. osc20 and osc100 are split 70:15:15.

    The task involves nodal regression, where the goal is to predict the
    probabilistic measure of single-node basin stability for each node. This
    measure represents the likelihood that the entire grid will return to a
    synchronized state after being perturbed.

    Besides considering osc20 or osc100 individually, training on osc20 and
    evaluating on osc100 is an interesting out-of-distribution generalization
    task. For evaluation purposes only, osctexas is a single grid with 1,910
    nodes inspired by Birchfield et al.
    <https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg2000/>
    to further test the out-of-distribution generalization.

    Synchronization of non-linear oscillators is a crucial phenomenon in many
    real-world systems, including cognitive functions of brains, pacemaker
    cells in a beating heart, and the stable operation of power grids.
    However, exact numerical simulations of large systems of coupled
    oscillators are exceedingly expensive.
    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (one of :obj:`"osc20"`,
            :obj:`"osc100"`, :obj:`"osctexas"`).
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"valid"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`).
        transform (callable, optional): A function/transform that takes in a
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`).
        pre_transform (callable, optional): A function/transform that takes in
            a :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`).
        pre_filter (callable, optional): A function that takes in a
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`).
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`).
    """

    datasets = {
        "osc20": dataset020_url,
        "osc100": dataset100_url,
        "osctexas": datasettexas_url,
    }

    def __init__(
        self,
        root,
        name,
        split="train",
        normalize_targets=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload=False,
    ):
        self.name = name.lower()
        assert self.name in self.datasets
        self.single_grid = self.name in ["osctexas"]
        if self.single_grid:
            split = "test"
        assert split in ["train", "valid", "test"]
        self.split = split
        self.normalize_targets = normalize_targets
        self.task_name = "snbs"
        super().__init__(root, transform, pre_transform, pre_filter,
            force_reload=force_reload)
        path = os.path.join(self.processed_dir, f"{split}.pt")
        if not os.path.exists(path) or force_reload:
            self.process()
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ["input_data.h5", f"{self.task_name}.h5"]

    @property
    def processed_file_names(self):
        return ["train.pt", "valid.pt", "test.pt"]

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    def download(self):
        raw_dir = self.raw_dir
        if all(
            os.path.exists(os.path.join(raw_dir, split, file))
            for split in ["train", "valid", "test"]
            for file in self.raw_file_names
        ):
            print("Files already exist, skipping download.")
            return
        url = self.datasets[self.name]
        os.makedirs(raw_dir, exist_ok=True)
        print(f"Downloading from {url} to {raw_dir}")
        path = download_url(url, raw_dir)
        dataset_zip = os.path.join(raw_dir, f"{self.name}.zip")
        os.rename(path, dataset_zip)
        print(f"Downloaded zip file to {dataset_zip}")
        self.unzip_datasets(dataset_zip, raw_dir)
        os.unlink(dataset_zip)
        print(f"Unzipped and removed {dataset_zip}")

    def unzip_datasets(self, zip_path, extract_to):
        print(f"Unzipping {zip_path} to {extract_to}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Unzipped {zip_path}")
        # Move files from nested directories to the correct location
        nested_dirs = [
            d
            for d in os.listdir(extract_to)
            if os.path.isdir(os.path.join(extract_to, d))
        ]
        for nested_dir in nested_dirs:
            nested_dir_path = os.path.join(extract_to, nested_dir)
            for filename in os.listdir(nested_dir_path):
                os.rename(
                    os.path.join(nested_dir_path, filename),
                    os.path.join(extract_to, filename),
                )
                if self.single_grid:
                    if filename in [
                        "train",
                        "test",
                        "tm.h5",
                        "mfd.h5",
                        "input_features.csv",
                    ]:
                        os.remove(os.path.join(extract_to, filename))
                else:
                    for other_file in os.listdir(os.path.join(extract_to, filename)):
                        if other_file in ["tm.h5", "mfd.h5", "input_features.csv"]:
                            os.remove(os.path.join(extract_to, filename, other_file))
            os.rmdir(nested_dir_path)

    def read_targets(self):
        file_targets = os.path.join(self.raw_dir, self.split, f"{self.task_name}.h5")
<<<<<<< HEAD
=======
        file_targets = os.path.join(self.raw_dir, self.split, f"{self.task_name}.h5")
>>>>>>> 9d130f8af (update format)
        if self.single_grid:
            file_targets = os.path.join(self.raw_dir, f"{self.task_name}.h5")
        print(f"Reading targets from {file_targets}")
        hf = h5py.File(file_targets, "r")
        int_keys = [int(x) for x in list(hf.keys())]
        self.slice_index = slice(
            min(int_keys), max(int_keys)
        )  # Automatically set slice_index based on the file
        return {
            index_grid: np.array(hf.get(str(index_grid)), dtype="float32")
            for index_grid in range(self.slice_index.start, self.slice_index.stop + 1)
        }

    def process(self):
        print("Processing data")
        targets = self.read_targets()
        file_to_read = os.path.join(self.raw_dir, self.split, "input_data.h5")
        if self.single_grid:
            file_to_read = os.path.join(self.raw_dir, "input_data.h5")
        print(f"Reading input data from {file_to_read}")
        f = h5py.File(file_to_read, "r")
        dset_grids = f["grids"]
        data_list = []
        for index_grid in range(self.slice_index.start, self.slice_index.stop + 1):
            node_features = np.array(
                dset_grids[str(index_grid)].get("node_features"), dtype="float32"
            ).transpose()
            edge_index = (
                np.array(dset_grids[str(index_grid)].get("edge_index"), dtype="int64")
                - 1
            ).transpose()
            edge_attr = np.array(
                dset_grids[str(index_grid)].get("edge_attr"), dtype="float32"
            )
            y = torch.tensor(targets[index_grid])
            data = gData(
                x=(torch.tensor(node_features).unsqueeze(-1)),
                edge_index=torch.tensor(edge_index),
                edge_attr=torch.tensor(edge_attr).unsqueeze(-1),
                y=y,
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), os.path.join(self.processed_dir, f"{self.split}.pt"))
        print(
            f"Processed data saved to "
            f"{os.path.join(self.processed_dir, f'{self.split}.pt')}"
        )
