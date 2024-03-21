import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data.hypergraph_data import HyperGraphData


class NDCClasses25(InMemoryDataset):
    r"""This is a temporal higher-order network dataset from the
    `"Simplicial Closure and higher-order link prediction"
    <https://arxiv.org/abs/1802.06916>`_ paper, which here means
    a sequence of timestamped simplices where each simplex is a set of nodes.
    Under the Drug Listing Act of 1972, the U.S. Food and Drug Administration
    releases information on all commercial drugs going through the regulation
    of the agency, forming the National Drug Code (NDC) Directory.

    See the original `dataset page
    <https://www.cs.cornell.edu/~arb/data/NDC-classes>`_ for more details.

    In this dataset, each simplex corresponds to a drug and the nodes are
    class labels applied to the drugs. Timestamps are in days and
    represent when the drug was first marketed. This dataset is restricted
    to simplices that consist of at most 25 nodes.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        setting (str, optional): If :obj:`"transductive"`, loads the dataset
            for Transductive training. If :obj:`"inductive"`, loads the
            dataset for Inductive training. (default: :obj:`"transductive"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = ("https://huggingface.co/datasets/SauravMaheshkar/NDC-classes-25/"
           "raw/main/processed")
    settings = ["transductive", "inductive"]

    def __init__(
        self,
        root: Optional[str] = None,
        split: Optional[str] = "train",
        setting: Optional[str] = "transductive",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        assert split in ["train", "val", "test"]

        if setting not in self.settings:
            raise ValueError(
                f"Invalid 'setting' argument must be one of {self.settings}")

        self.setting = setting
        self.url = osp.join(self.url, self.setting)

        super().__init__(root, transform, pre_transform, pre_filter, log,
                         force_reload)
        path = osp.join(self.processed_dir, f"{self.setting}_{split}.pt")
        self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ["train_df.csv", "val_df.csv", "test_df.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            f"{self.setting}_train.pt",
            f"{self.setting}_val.pt",
            f"{self.setting}_test.pt",
        ]

    def download(self) -> None:
        download_url(osp.join(self.url, self.raw_file_names[0]), self.raw_dir)
        download_url(osp.join(self.url, self.raw_file_names[1]), self.raw_dir)
        download_url(osp.join(self.url, self.raw_file_names[2]), self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        for split in ["train", "val", "test"]:
            # Read CSV and create dataframe
            df = pd.read_csv(f"{self.raw_dir}/{split}_df.csv")

            data_list = []
            for index, row in df.iterrows():
                # Parse nodes and timestamp
                nodes = eval(row["nodes"])  # str(List) -> List
                timestamp = row["timestamp"]

                # Construct the node features tensor
                node_features = [[timestamp]]

                # Add edges to edge_index
                edge_index_container: List[List[int]] = [[], []]
                for node in nodes:
                    edge_index_container[0].append(node)
                    edge_index_container[1].append(
                        index)  # Use index as hyperedge index

                # Convert node_features and edge_index lists to tensors
                x = torch.tensor(node_features, dtype=torch.float)
                edge_index = torch.tensor(edge_index_container)

                data = HyperGraphData(x=x, edge_index=edge_index)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            path = osp.join(self.processed_dir, f"{self.setting}_{split}.pt")
            self.save(data_list, path)
