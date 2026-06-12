import os
import tempfile

import h5py
import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.io import fs
from torch_geometric.io.fs import torch_load, torch_save

zenodo_url = "https://zenodo.org/record/8204334/files/"
dataset020_url = zenodo_url + "ds20.zip"
dataset100_url = zenodo_url + "ds100.zip"
datasettexas_url = zenodo_url + "texas.zip"


class OscillatorDataset(InMemoryDataset):
    r"""A dataset of oscillator networks for studying dynamic stability in
    power grids, as introduced in the paper `"Toward dynamic stability
    assessment of power grid topologies using graph neural networks"
    <https://doi.org/10.1063/5.0160915>`_.

    Synchronization of oscillators is a crucial phenomenon in many
    real-world systems, including cognitive functions in brains, pacemaker
    cells in a beating heart, and the stable operation of power grids.
    Exact numerical simulations of large coupled-oscillator systems are
    exceedingly expensive, motivating the use of graph neural networks as a
    surrogate.

    The task is **nodal regression**: predicting the probabilistic *single-node
    basin stability* (SNBS) for each node, i.e. the likelihood that the entire
    grid returns to a synchronized state after a perturbation at that node.

    Three datasets are provided:

    - :obj:`"osc20"` — 10,000 synthetic grids with 20 nodes each
      (train/valid/test split of 70/15/15).
    - :obj:`"osc100"` — 10,000 synthetic grids with 100 nodes each
      (train/valid/test split of 70/15/15).
    - :obj:`"osctexas"` — a single large grid with 1,910 nodes inspired by a
      real US transmission network of `"Texas"
      <https://electricgrids.engr.tamu.edu/electric-grid-test-cases/
      activsg2000/>`_, provided as a test-only out-of-distribution benchmark.

    Training on :obj:`"osc20"` or :obj:`"osc100"` and evaluating on the
    other or :obj:`"osctexas"` constitutes a challenging
    out-of-distribution generalization task.

    Raw data is available on `Zenodo <https://zenodo.org/records/8204334>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"osc20"`, :obj:`"osc100"`,
            or :obj:`"osctexas"`).
        split (str, optional): Which split to load: :obj:`"train"`,
            :obj:`"valid"`, or :obj:`"test"`. Ignored for :obj:`"osctexas"`,
            which is test-only. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` object and returns a
            transformed version. Applied before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` object and returns a
            transformed version. Applied once before saving to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~torch_geometric.data.Data` object and returns a boolean
            indicating whether to retain it. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes
          - #edges
          - #features
        * - osc20
          - 10,000
          - 20
          - ~variable~
          - 1
        * - osc100
          - 10,000
          - 100
          - ~variable~
          - 1
        * - osctexas
          - 1
          - 1,910
          - 5154
          - 1
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
        self.task_name = "snbs"
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = os.path.join(self.processed_dir, f"{split}.pt")
        if not fs.exists(path) or force_reload:
            self.process()
        self.data, self.slices = torch_load(path)

    @property
    def raw_file_names(self):
        return ["input_data.h5", f"{self.task_name}.h5"]

    @property
    def processed_file_names(self):
        return ["train.pt", "valid.pt", "test.pt"]

    @property
    def raw_dir(self):
        return self.root + '/' + self.name + '/raw'

    @property
    def processed_dir(self):
        return self.root + '/' + self.name + '/processed'

    def download(self):
        raw_dir = self.raw_dir
        if self.single_grid:
            # For single grid datasets, files are directly in raw_dir
            if all(
                    fs.exists(os.path.join(raw_dir, file))
                    for file in self.raw_file_names):
                print("Files already exist, skipping download.")
                return
        else:
            # For multi-grid datasets, files are in split subdirectories
            if all(
                    fs.exists(os.path.join(raw_dir, split, file))
                    for split in ["train", "valid", "test"]
                    for file in self.raw_file_names):
                print("Files already exist, skipping download.")
                return
        url = self.datasets[self.name]
        fs.makedirs(raw_dir, exist_ok=True)
        print(f"Downloading from {url} to {raw_dir}")
        path = download_url(url, raw_dir)
        dataset_zip = os.path.join(raw_dir, f"{self.name}.zip")
        zip_path = path
        if path != dataset_zip:
            fs.mv(path, dataset_zip)
            zip_path = dataset_zip
        print(f"Downloaded zip file to {zip_path}")
        self.unzip(zip_path, raw_dir)
        if fs.exists(zip_path):
            fs.rm(zip_path)

    def unzip(self, zip_path, extract_to):
        print(f"Unzipping {zip_path} to {extract_to}")
        extract_fs = fs.get_fs(extract_to)

        with tempfile.TemporaryDirectory() as tmpdir:
            fs.cp(zip_path, tmpdir, extract=True, log=False)
            tmp_fs = fs.get_fs(tmpdir)

            # Find the single top-level directory the zip extracted into
            top_level = tmp_fs.ls(tmpdir, detail=False)
            source_dir = (top_level[0] if len(top_level) == 1
                          and tmp_fs.isdir(top_level[0]) else tmpdir)

            for item in tmp_fs.ls(source_dir, detail=False):
                item_name = item.split('/')[-1]
                if self.single_grid:
                    is_file = tmp_fs.isfile(item)
                    is_valid = is_file and item_name in self.raw_file_names
                    if is_valid:
                        src_path = f"{tmpdir}/{item}"
                        dst_path = f"{extract_to}/{item_name}"
                        self._copy_file(src_path, dst_path, tmp_fs, extract_fs)
                else:
                    if tmp_fs.isdir(item) and item_name in ("train", "valid",
                                                            "test"):
                        extract_fs.makedirs(f"{extract_to}/{item_name}",
                                            exist_ok=True)
                        for subitem in tmp_fs.ls(item, detail=False):
                            subitem_name = subitem.split('/')[-1]
                            if subitem_name in self.raw_file_names:
                                src_path = f"{tmpdir}/{item}/{subitem}"
                                dst_path = f"{extract_to}/{subitem_name}"
                                self._copy_file(src_path, dst_path, tmp_fs,
                                                extract_fs)

        print(f"Unzipped {zip_path}")

    def _copy_file(self, src, dst, src_fs, dst_fs):
        with src_fs.open(src, 'rb') as f_in:
            with dst_fs.open(dst, 'wb') as f_out:
                f_out.write(f_in.read())

    def read_targets(self):
        file_targets = os.path.join(self.raw_dir, self.split,
                                    f"{self.task_name}.h5")
        if self.single_grid:
            file_targets = os.path.join(self.raw_dir, f"{self.task_name}.h5")
        print(f"Reading targets from {file_targets}")
        with self.open_h5(file_targets) as hf:
            grid_keys = sorted(int(x) for x in hf.keys())

            targets = {
                index_grid: np.array(hf.get(str(index_grid)), dtype="float32")
                for index_grid in grid_keys
            }
        return targets, grid_keys

    def open_h5(self, file_path, mode='r'):
        """Open h5py file, handling both local and memory filesystems."""
        fs_obj = fs.get_fs(file_path)
        f = fs_obj.open(file_path, 'rb')
        return h5py.File(f, mode)

    def process(self):
        print("Processing data")
        targets, grid_keys = self.read_targets()
        file_to_read = os.path.join(self.raw_dir, self.split, "input_data.h5")
        if self.single_grid:
            file_to_read = os.path.join(self.raw_dir, "input_data.h5")
        print(f"Reading input data from {file_to_read}")
        with self.open_h5(file_to_read) as f:
            dset_grids = f["grids"]
            data_list = []
            for index_grid in grid_keys:
                node_features = np.array(
                    dset_grids[str(index_grid)].get("node_features"),
                    dtype="float32").transpose()
                edge_index = np.array(
                    dset_grids[str(index_grid)].get("edge_index"),
                    dtype="int64") - 1
                edge_index = edge_index.transpose()
                edge_attr = np.array(
                    dset_grids[str(index_grid)].get("edge_attr"),
                    dtype="float32")
                y = torch.tensor(targets[index_grid])
                data = Data(
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
        fs.makedirs(self.processed_dir, exist_ok=True)
        data, slices = self.collate(data_list)
        save_path = os.path.join(self.processed_dir, f"{self.split}.pt")
        torch_save((data, slices), save_path)
        print(f"Processed data saved to {save_path}")
