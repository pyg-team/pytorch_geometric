import os
import os.path as osp
import pickle
import shutil
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class LRGBDataset(InMemoryDataset):
    r"""The `"Long Range Graph Benchmark (LRGB)"
    <https://arxiv.org/abs/2206.08164>`_
    datasets which is a collection of 5 graph learning datasets with tasks
    that are based on long-range dependencies in graphs. See the original
    `source code <https://github.com/vijaydwivedi75/lrgb>`_ for more details
    on the individual datasets.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (one of :obj:`"PascalVOC-SP"`,
            :obj:`"COCO-SP"`, :obj:`"PCQM-Contact"`, :obj:`"Peptides-func"`,
            :obj:`"Peptides-struct"`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
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

    Stats:
        .. list-table::
            :widths: 15 15 10 10 10
            :header-rows: 1

            * - Name
              - Task
              - #graphs
              - #nodes
              - #edges
            * - PascalVOC-SP
              - Node Prediction
              - 11,355
              - ~479.40
              - ~2,710.48
            * - COCO-SP
              - Node Prediction
              - 123,286
              - ~476.88
              - ~2,693.67
            * - PCQM-Contact
              - Link Prediction
              - 529,434
              - ~30.14
              - ~61.09
            * - Peptides-func
              - Graph Classification
              - 15,535
              - ~150.94
              - ~307.30
            * - Peptides-struct
              - Graph Regression
              - 15,535
              - ~150.94
              - ~307.30

    """

    names = [
        'pascalvoc-sp', 'coco-sp', 'pcqm-contact', 'peptides-func',
        'peptides-struct'
    ]

    urls = {
        'pascalvoc-sp':
        'https://www.dropbox.com/s/8x722ai272wqwl4/voc_superpixels_edge_wt_region_boundary.zip?dl=1',
        'coco-sp':
        'https://www.dropbox.com/s/r6ihg1f4pmyjjy0/coco_superpixels_edge_wt_region_boundary.zip?dl=1',
        'pcqm-contact':
        'https://www.dropbox.com/s/qdag867u6h6i60y/pcqmcontact.zip?dl=1',
        'peptides-func':
        'https://www.dropbox.com/s/ycsq37q8sxs1ou8/peptidesfunc.zip?dl=1',
        'peptides-struct':
        'https://www.dropbox.com/s/zgv4z8fcpmknhs8/peptidesstruct.zip?dl=1'
    }

    dwnld_file_name = {
        'pascalvoc-sp': 'voc_superpixels_edge_wt_region_boundary',
        'coco-sp': 'coco_superpixels_edge_wt_region_boundary',
        'pcqm-contact': 'pcqmcontact',
        'peptides-func': 'peptidesfunc',
        'peptides-struct': 'peptidesstruct'
    }

    def __init__(self, root: str, name: str, split: str = "train",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in self.names
        assert split in ['train', 'val', 'test']

        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        if self.name.split('-')[1] == 'sp':
            return ['train.pickle', 'val.pickle', 'test.pickle']
        else:
            return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, self.dwnld_file_name[self.name]),
                  self.raw_dir)
        os.unlink(path)

    def process(self):
        if self.name.split('-')[1] == 'sp':
            # PascalVOC-SP and COCO-SP
            self.process_sp()
        elif self.name.split('-')[0] == 'peptides':
            # Peptides-func and Peptides-struct
            self.process_peptides()
        else:
            # PCQM-Contact
            self.process_pcqm_contact()

    def process_sp(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]
                """
                Each `graph` is a tuple (x, edge_attr, edge_index, y)
                    Shape of x : [num_nodes, 14]
                    Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
                    Shape of edge_index : [2, num_edges]
                    Shape of y : [num_nodes]
                """

                x = graph[0].to(torch.float)
                edge_attr = graph[1].to(torch.float)
                edge_index = graph[2]
                y = torch.LongTensor(graph[3])

                if self.name == 'coco-sp':
                    # Label remapping for coco. See self.label_remap_coco() func
                    label_map = self.label_remap_coco()
                    for i, label in enumerate(y):
                        y[i] = label_map[label.item()]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

    def label_remap_coco(self):
        # Util function for name 'COCO-SP'
        # to remap the labels as the original label idxs are not contiguous

        original_label_ix = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]
        label_map = {}
        for i, key in enumerate(original_label_ix):
            label_map[key] = i

        return label_map

    def process_peptides(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pt'), 'rb') as f:
                graphs = torch.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]
                """
                Each `graph` is a tuple (x, edge_attr, edge_index, y)
                    Shape of x : [num_nodes, 9]
                    Shape of edge_attr : [num_edges, 3]
                    Shape of edge_index : [2, num_edges]
                    Shape of y : [1, 10] for Peptides-func,  or
                                 [1, 11] for Peptides-struct
                """

                x = graph[0]
                edge_attr = graph[1]
                edge_index = graph[2]
                y = graph[3]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))

    def process_pcqm_contact(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pt'), 'rb') as f:
                graphs = torch.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]
                """
                Each `graph` is a tuple (x, edge_attr, edge_index,
                                        edge_index_labeled, edge_label)
                    Shape of x : [num_nodes, 9]
                    Shape of edge_attr : [num_edges, 3]
                    Shape of edge_index : [2, num_edges]
                    Shape of edge_index_labeled: [2, num_labeled_edges]
                    Shape of edge_label : [num_labeled_edges]

                    where,
                    num_labeled_edges are negative edges and link pred labels,
                    https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/loader/dataset/pcqm4mv2_contact.py#L192
                """

                x = graph[0]
                edge_attr = graph[1]
                edge_index = graph[2]
                edge_index_labeled = graph[3]
                edge_label = graph[4]
                y = None

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            edge_index_labeled=edge_index_labeled,
                            edge_label=edge_label, y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
