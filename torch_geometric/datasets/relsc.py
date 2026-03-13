import json
import math
import os
import urllib.request
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import Data, HeteroData, InMemoryDataset, extract_zip

SEMANTIC_DICT = {
    'declarations':
    [6, 11, 12, 13, 17, 55, 32, 33, 43, 27, 65, 72, 53, 42, 4, 68, 5, 0],
    'control_flow': [3, 16, 21, 23, 38, 40, 10, 50, 61, 63, 35, 69, 46, 18],
    'types_and_references': [1, 45, 15, 36, 49, 37, 9, 66, 28, 26, 29],
    'expressions_and_operations': [2, 7, 67, 39, 52, 54, 64, 70, 30, 58, 57],
    'code_structure': [8, 19, 25, 24, 48, 71, 14, 47, 62, 31, 22],
    'exceptions': [60, 41, 56, 34],
    'literals_and_constants': [59, 44, 51, 20]
}

ZENODO_URLS = [
    'https://zenodo.org/records/18598713/files/data.zip?download=1',
    'https://zenodo.org/records/18598713/files/y_labels.zip?download=1'
]


def _find_file(directory: str, filename: str) -> str:
    target_lower = filename.lower()
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower() == target_lower:
                return os.path.join(root, f)
    raise FileNotFoundError(
        f"File {filename} (case-insensitive check) not found in {directory}")


def _download_with_tqdm(url: str, filepath: str) -> None:
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    filename_desc = url.split('/')[-1].split('?')[0]
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                             desc=f"Downloading {filename_desc}") as t:
        urllib.request.urlretrieve(url, filename=filepath,
                                   reporthook=t.update_to)


class RelSCH(InMemoryDataset):
    r"""The homogeneous variant (RelSC-H) of the RelSC
    benchmark dataset from the `"A Benchmark Dataset for
    Graph Regression with Homogeneous and
    Multi-Relational Variants" <https://arxiv.org/pdf/2505.23875>`_ paper.

    RelSC is a graph-regression dataset built from
    program graphs that combine syntactic and semantic information extracted
    from source code. This variant (RelSC-H) supplies rich node features under
    a single edge type.

    For more information, tutorials, and scripts to reproduce paper results or
    build your own dataset, please visit the `Official Project Page
    <https://github.com/MarcusVukojevic/graph_regression_datasets>`_.

    .. note::
        The target values are normalized execution
        times in the range :obj:`[0, 1]`.

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - Avg. Nodes
          - Avg. Edges
        * - hadoop
          - 2,895
          - ~1,490
          - ~5,731
        * - H2
          - 194
          - ~2,091
          - ~8,020
        * - dubbo
          - 123
          - ~616
          - ~2,354
        * - rdf
          - 478
          - ~450
          - ~1,740
        * - systemds
          - 127
          - ~871
          - ~3,321
        * - ossbuilds
          - 922
          - ~875
          - ~3,361

    Args:
        root (str): Root directory where the dataset should be saved.
        project_name (str, optional): The name of the software project to load.
            Available options are :obj:`"rdf"`, :obj:`"dubbo"`, :obj:`"H2"`,
            :obj:`"hadoop"`, :obj:`"systemds"`, and :obj:`"ossbuilds"`.
            (default: :obj:`"rdf"`)
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
    """
    urls = ZENODO_URLS

    def __init__(
        self,
        root: str = './data',
        project_name: str = 'rdf',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.project_name = project_name
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=Data)
        self.split_idx = torch.load(self.processed_paths[1],
                                    weights_only=False)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'RelSC', 'raw')

    @property
    def raw_file_names(self):
        return [f"{self.project_name}.json", f"y_{self.project_name}.csv"]

    @property
    def processed_file_names(self):
        return [
            f'data_homogeneous_{self.project_name}.pt',
            f'split_homogeneous_{self.project_name}.pt'
        ]

    def download(self):
        files_exist = True
        for f in self.raw_file_names:
            try:
                _find_file(self.raw_dir, f)
            except FileNotFoundError:
                files_exist = False
                break

        if files_exist:
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        for url in self.urls:
            filename = url.split('/')[-1].split('?')[0]
            filepath = os.path.join(self.raw_dir, filename)

            _download_with_tqdm(url, filepath)
            extract_zip(filepath, self.raw_dir)
            os.remove(filepath)

    def get_idx_split(self):
        return self.split_idx

    def _aggregate_multi_edges(self, edge_index, edge_attr):
        if edge_attr is not None:
            unique_edges, inverse_indices = torch.unique(
                edge_index, dim=1, return_inverse=True)
            num_edges = unique_edges.size(1)
            aggregated_edge_attr = torch.zeros((num_edges, edge_attr.size(1)),
                                               dtype=torch.float)

            for i in range(len(inverse_indices)):
                aggregated_edge_attr[inverse_indices[i]] = torch.logical_or(
                    aggregated_edge_attr[inverse_indices[i]].bool(),
                    edge_attr[i].bool()).float()
            return unique_edges, aggregated_edge_attr
        return edge_index, edge_attr

    def process(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        json_path = _find_file(self.raw_dir, self.raw_file_names[0])
        csv_path = _find_file(self.raw_dir, self.raw_file_names[1])

        with open(json_path) as f:
            dataset = json.load(f)

        labels_file = pd.read_csv(csv_path)
        labels_file['Value'] = labels_file['Value'] + 1
        labels_file['LogValue'] = labels_file['Value'].apply(math.log)
        max_val = labels_file['LogValue'].max()
        labels_file['NormalizedValue'] = labels_file['LogValue'] / max_val
        labels = labels_file.set_index('Key')['NormalizedValue'].to_dict()

        sorted_keys = sorted([k for k in dataset.keys() if k in labels])

        train_keys, temp_keys = train_test_split(sorted_keys, train_size=0.7,
                                                 shuffle=False)
        dev_keys, test_keys = train_test_split(temp_keys, test_size=0.5,
                                               shuffle=False)

        all_edge_features = []
        for key in sorted_keys:
            all_edge_features.extend(dataset[key][0][2])

        unique_edge_features = np.unique(all_edge_features)
        num_edge_features = len(unique_edge_features)

        max_nodes = 73
        unique_node_features = np.arange(0, max_nodes + 1)

        data_list = []

        for key in sorted_keys:
            graph_data = dataset[key]
            num_nodes = graph_data[1]
            node_features = graph_data[0][0]
            edge_features = graph_data[0][2]

            node_attr = torch.zeros((len(node_features), max_nodes),
                                    dtype=torch.float)
            for i, feature in enumerate(node_features):
                feature_index = np.where(
                    unique_node_features == feature[0])[0][0]
                node_attr[i, feature_index] = 1.0

            edge_attr = torch.zeros((len(edge_features), num_edge_features),
                                    dtype=torch.int32)
            for i, feature in enumerate(edge_features):
                feature_index = np.where(
                    unique_edge_features == feature[0])[0][0]
                edge_attr[i, feature_index] = 1.0

            adj_list = np.array(graph_data[0][1])
            edge_index = torch.tensor(adj_list,
                                      dtype=torch.long).t().contiguous()

            edge_index, edge_attr = self._aggregate_multi_edges(
                edge_index, edge_attr)

            node_edge_info = torch.zeros((num_nodes, num_edge_features),
                                         dtype=torch.float)
            for i in range(edge_index.size(1)):
                src_node = edge_index[0, i]
                node_edge_info[src_node] += edge_attr[i]

            node_attr = torch.cat([node_attr, node_edge_info], dim=1)
            label = torch.tensor([labels[key]], dtype=torch.float)

            data = Data(x=node_attr, edge_index=edge_index, y=label)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        split_dict = {
            'train':
            torch.tensor([sorted_keys.index(k) for k in train_keys],
                         dtype=torch.long),
            'val':
            torch.tensor([sorted_keys.index(k) for k in dev_keys],
                         dtype=torch.long),
            'test':
            torch.tensor([sorted_keys.index(k) for k in test_keys],
                         dtype=torch.long)
        }

        self.save(data_list, self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])


class RelSCM(InMemoryDataset):
    r"""The multi-relational variant (RelSC-M)
    of the RelSC benchmark dataset from
    the `"A Benchmark Dataset for Graph Regression with Homogeneous and
    Multi-Relational Variants" <https://arxiv.org/pdf/2505.23875>`_ paper.

    RelSC is a multi-relational graph dataset where nodes
    preserve their original structure, connecting
    nodes through multiple edge types
    that encode distinct semantic relationships.

    For more information, tutorials, and scripts to reproduce paper results or
    build your own dataset, please visit the `Official Project Page
    <https://github.com/MarcusVukojevic/graph_regression_datasets>`_.

    .. note::
        The target values are normalized execution
        times in the range :obj:`[0, 1]`.

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - Avg. Nodes
          - Avg. Edges
        * - hadoop
          - 2,895
          - ~1,490
          - ~11,764
        * - H2
          - 194
          - ~2,091
          - ~16,518
        * - dubbo
          - 123
          - ~616
          - ~4,812
        * - rdf
          - 478
          - ~450
          - ~3,574
        * - systemds
          - 127
          - ~871
          - ~6,805
        * - ossbuilds
          - 922
          - ~875
          - ~6,907

    Args:
        root (str): Root directory where the dataset should be saved.
        project_name (str, optional): The name of the software project to load.
            Available options are :obj:`"rdf"`, :obj:`"dubbo"`, :obj:`"H2"`,
            :obj:`"hadoop"`, :obj:`"systemds"`, and :obj:`"ossbuilds"`.
            (default: :obj:`"rdf"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    urls = ZENODO_URLS

    def __init__(
        self,
        root: str = './data',
        project_name: str = 'rdf',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.project_name = project_name
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)

        self.data_list = torch.load(self.processed_paths[0],
                                    weights_only=False)
        self.split_idx = torch.load(self.processed_paths[1],
                                    weights_only=False)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'RelSC', 'raw')

    @property
    def raw_file_names(self):
        return [f"{self.project_name}.json", f"y_{self.project_name}.csv"]

    @property
    def processed_file_names(self):
        return [
            f'data_heterogeneous_{self.project_name}.pt',
            f'split_heterogeneous_{self.project_name}.pt'
        ]

    def download(self):
        files_exist = True
        for f in self.raw_file_names:
            try:
                _find_file(self.raw_dir, f)
            except FileNotFoundError:
                files_exist = False
                break

        if files_exist:
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        for url in self.urls:
            filename = url.split('/')[-1].split('?')[0]
            filepath = os.path.join(self.raw_dir, filename)

            _download_with_tqdm(url, filepath)
            extract_zip(filepath, self.raw_dir)
            os.remove(filepath)

    def get_idx_split(self):
        return self.split_idx

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def _get_category(self, number):
        for category, lst in SEMANTIC_DICT.items():
            if number in lst:
                return category
        return 'others'

    def _find_index(self, dictionary, target_number):
        for _, lst in dictionary.items():
            if target_number in lst:
                return lst.index(target_number)
        return None

    def _add_reverse_edges(self, graph):
        new_edges = defaultdict(list)
        for key, edges in graph.items():
            src_type, rel_id, dst_type = key
            if src_type == dst_type:
                for edge in edges:
                    new_edges[key].append(edge)
                    new_edges[key].append(edge[::-1])
            else:
                new_key = (dst_type, rel_id + "_rev", src_type)
                for edge in edges:
                    new_edges[key].append(edge)
                    new_edges[new_key].append(edge[::-1])
        return new_edges

    def process(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        json_path = _find_file(self.raw_dir, self.raw_file_names[0])
        csv_path = _find_file(self.raw_dir, self.raw_file_names[1])

        with open(json_path) as f:
            dataset = json.load(f)

        labels_file = pd.read_csv(csv_path)
        labels_file['Value'] = labels_file['Value'] + 1
        labels_file['LogValue'] = labels_file['Value'].apply(math.log)
        max_val = labels_file['LogValue'].max()
        labels_file['NormalizedValue'] = labels_file['LogValue'] / max_val
        labels = labels_file.set_index('Key')['NormalizedValue'].to_dict()

        sorted_keys = sorted([k for k in dataset.keys() if k in labels])

        train_keys, temp_keys = train_test_split(sorted_keys, train_size=0.7,
                                                 shuffle=False)
        dev_keys, test_keys = train_test_split(temp_keys, test_size=0.5,
                                               shuffle=False)

        all_edge_features = []
        for key in sorted_keys:
            all_edge_features.extend(dataset[key][0][2])

        unique_edge_features = np.unique(all_edge_features)
        num_edge_features = len(unique_edge_features)

        max_nodes = 73
        unique_node_features = np.arange(0, max_nodes + 1)

        data_list = []

        for key in sorted_keys:
            graph_data = dataset[key]
            num_nodes = graph_data[1]
            node_features = graph_data[0][0]
            edge_features = graph_data[0][2]

            adj_list = np.array(graph_data[0][1])
            edge_index = torch.tensor(adj_list,
                                      dtype=torch.long).t().contiguous()
            or_edge_attr = torch.tensor(edge_features, dtype=torch.int32)

            node_attr = torch.zeros((len(node_features), max_nodes),
                                    dtype=torch.float)
            for i, feature in enumerate(node_features):
                feature_index = np.where(
                    unique_node_features == feature[0])[0][0]
                node_attr[i, feature_index] = 1.0

            edge_attr = torch.zeros((len(edge_features), num_edge_features),
                                    dtype=torch.int32)
            for i, feature in enumerate(edge_features):
                feature_index = np.where(
                    unique_edge_features == feature[0])[0][0]
                edge_attr[i, feature_index] = 1.0

            node_types = [i[0] for i in node_features]
            data = HeteroData()

            node_type_dict = {}
            for idx, node_type in enumerate(node_types):
                cat = self._get_category(int(node_type))
                if cat not in node_type_dict:
                    node_type_dict[cat] = []
                node_type_dict[cat].append(idx)

            node_edge_info = torch.zeros((num_nodes, num_edge_features),
                                         dtype=torch.float)
            for i in range(edge_index.size(1)):
                src_node = edge_index[0, i]
                node_edge_info[src_node] += edge_attr[i]

            new_node_attr = torch.zeros((num_nodes, 84), dtype=torch.float)
            for i in range(num_nodes):
                new_node_attr[i] = torch.cat((node_attr[i], node_edge_info[i]),
                                             dim=0)

            for category, indices in node_type_dict.items():
                if 'x' in data[category]:
                    data[category].x = torch.cat(
                        [data[category].x, new_node_attr[indices]], dim=0)
                else:
                    data[category].x = new_node_attr[indices]

            edge_index_dict = defaultdict(list)
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dest = edge_index[1, i].item()

                src_type = self._get_category(node_types[src])
                dest_type = self._get_category(node_types[dest])
                edge_type = or_edge_attr[i].item()

                src_local = self._find_index(node_type_dict, src)
                dest_local = self._find_index(node_type_dict, dest)

                k = (str(src_type), str(edge_type), str(dest_type))
                edge_index_dict[k].append([src_local, dest_local])

            edge_index_dict = self._add_reverse_edges(edge_index_dict)

            for k_edge in edge_index_dict:
                data[k_edge].edge_index = torch.tensor(
                    edge_index_dict[k_edge],
                    dtype=torch.long).t().contiguous()

            data.y = torch.tensor([labels[key]], dtype=torch.float)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        split_dict = {
            'train':
            torch.tensor([sorted_keys.index(k) for k in train_keys],
                         dtype=torch.long),
            'val':
            torch.tensor([sorted_keys.index(k) for k in dev_keys],
                         dtype=torch.long),
            'test':
            torch.tensor([sorted_keys.index(k) for k in test_keys],
                         dtype=torch.long)
        }

        torch.save(data_list, self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])
