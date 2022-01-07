# from typing import Optional, Callable, List
import os.path as osp
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset, download_url, Data


class EllipticBitcoinDataset(InMemoryDataset):
    r"""The Elliptic Data Set, a graph network of Bitcoin transactions
    with handcrafted features.
    <https://arxiv.org/pdf/1908.02591.pdf>`_ paper for reference.

    The Elliptic Data Set maps Bitcoin transactions to real entities
    belonging to licit categories (exchanges, wallet providers, miners,
    licit services, etc.) versus illicit ones (scams, malware, terrorist
    organizations, ransomware, Ponzi schemes, etc.)

    There are 203,769 node transactions and 234,355 directed edge
    payments flows.
    In the Elliptic Data Set, two percent (4,545) are labelled
    class1 (illicit).
    Twenty-one percent (42,019) are labelled class2 (licit).
    The remaining transactions are not labelled with regard to
    licit versus illicit,
    but have other features.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            osp.join(self.processed_dir, self.processed_paths[0])
        )

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def raw_file_names(self):
        return [
            f"{self.raw_dir}/elliptic_txs_classes.csv",
            f"{self.raw_dir}/elliptic_txs_edgelist.csv",
            f"{self.raw_dir}/elliptic_txs_features.csv",
        ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        print("Downloading...")
        features_csv_url = "https://tinyurl.com/9b7f8efe"
        edgelist_csv_url = "https://tinyurl.com/mr3v9d3f"
        classes_csv_url = "https://tinyurl.com/2p8up25z"
        download_url(features_csv_url, self.raw_dir)
        download_url(edgelist_csv_url, self.raw_dir)
        download_url(classes_csv_url, self.raw_dir)
        print("Done")

    def process(self):
        # Read data from `raw_path`.
        # self.download() # to download the files
        print(self.raw_file_names)
        classes_csv, edgelist_csv, features_csv = self.raw_file_names
        # classes_csv = f"{self.root}/elliptic_txs_classes.csv"
        # edgelist_csv = f"{self.root}/elliptic_txs_edgelist.csv"
        # features_csv = f"{self.root}/elliptic_txs_features.csv"

        # %%time
        df_features = pd.read_csv(features_csv, header=None)
        df_classes = pd.read_csv(classes_csv)
        df_edges = pd.read_csv(edgelist_csv)
        # creating column names for the the features
        # Reference: https://arxiv.org/abs/1908.02591
        colNames1 = {"0": "txId", 1: "Time step"}
        colNames2 = {str(ii + 2): "LF_" + str(ii + 1) for ii in range(93)}
        colNames3 = {str(ii + 95): "AF_" + str(ii + 1) for ii in range(72)}

        colNames = dict(colNames1, **colNames2, **colNames3)
        colNames = {int(jj): item_kk for jj, item_kk in colNames.items()}
        # local_features = colNames2
        # all_features = colNames
        df_features = df_features.rename(columns=colNames)

        # There are 3 different classes in the dataset
        # 0 - Licit
        # 1 - Illicit
        # 2 - Unknown
        df_classes["class"] = df_classes["class"]\
            .map({"unknown": 2, "1": 1, "2": 0})
        # merging dataframes
        df_merge = df_features.merge(
            df_classes, how="left", right_on="txId", left_on="txId"
        )
        df_merge = df_merge.sort_values("Time step").reset_index(drop=True)

        # training index edges mapping
        # original edge values to 0 based index edge values
        index_map = {}
        for i, row in df_merge[["txId"]].iterrows():
            index_map[row["txId"]] = i
        df_edges.txId1 = df_edges.txId1.map(index_map)
        df_edges.txId2 = df_edges.txId2.map(index_map)

        # convert edges to 2xN format
        df_edges = df_edges.astype(int)
        edge_index = np.array(df_edges.values).T
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

        # convert original transaction ids (nodes) to their respective mappings
        df_merge["txId"] = df_merge["txId"].map(index_map)
        weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)

        # timestamp based splits
        # training data 1 - 34 timestamps
        # testing data 35-49 timestamps
        # unknowns are taken from 1 - 49 timestamps

        train_idx = df_merge["Time step"].isin(range(1, 35)) & \
            df_merge["class"].isin([0, 1])
        test_idx = df_merge["Time step"].isin(range(35, 50)) & \
            df_merge["class"].isin([0, 1])
        unknown_idx = df_merge["Time step"].isin(range(1, 50)) & \
            df_merge["class"].isin([2])

        # _train_mask = torch.tensor(train_idx, dtype=torch.bool)
        # _test_mask = torch.tensor(test_idx, dtype=torch.bool)
        _unknown_mask = torch.tensor(unknown_idx, dtype=torch.bool)

        df_class_feature_selected_train_test = df_merge.copy()
        X_features = df_class_feature_selected_train_test.drop(
            columns=["txId", "Time step", "class"]
        )
        X_features = torch.tensor(
            np.array(X_features.values, dtype=np.double), dtype=torch.double
        )

        y_label = df_class_feature_selected_train_test[["class"]]
        y_labels = y_label["class"].values

        # converting data to PyGeometric graph data format

        y_labels[_unknown_mask] = 0
        target = torch.tensor(y_labels, dtype=torch.long)
        y = target
        # In case labels needs one-hot encoding format
        # y = torch.zeros(y_labels.shape[0], 2)
        # y[range(y.shape[0]), target]=1

        data = Data(
            x=X_features,
            edge_index=edge_index,
            edge_attr=weights,
            y=torch.tensor(y, dtype=torch.double),
        )

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True
        data.unknown_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.unknown_mask[unknown_idx] = True

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_classes(self):
        return 2

    def len(self):
        return len(self.processed_file_names)
