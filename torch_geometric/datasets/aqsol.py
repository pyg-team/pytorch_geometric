import os
import os.path as osp
import pickle
import shutil

import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class AQSOL(InMemoryDataset):
    r"""The AQSOL dataset from `Benchmarking GNNs
    <http://arxiv.org/abs/2003.00982>`_,
    based on `AqSolDB <https://www.nature.com/articles/s41597-019-0151-1>`_
    which is a standardized database of 9,982 molecular graphs with
    their aqueous solubility values, collected from 9 different data
    sources.

    The aqueous solubility targets are collected from experimental
    measurements and standardized to LogS units in AqSolDB. These final
    values as the property to regress in the AQSOL dataset which is the
    resultant collection in Benchmarking GNNs after filtering out few graphs
    with no bonds/edges and a small number of graphs with missing node
    feature values.

    Thus, the total molecular graphs are 9,823. For each molecular graph,
    the node features are the types of heavy atoms and the edge features
    are the types of bonds between them, similar as ZINC.

    Size of Dataset: 9,982 molecules
    Split: Scaffold split (8:1:1) following same code as OGB
    After cleaning: 7,831 train / 996 val / 996 test
    Number of (unique) atoms: 65
    Number of (unique) bonds: 5
    Performance Metric: MAE, same as ZINC

    Atom Dict: {'Br': 0, 'C': 1, 'N': 2, 'O': 3, 'Cl': 4, 'Zn': 5, 'F': 6,
    'P': 7, 'S': 8, 'Na': 9, 'Al': 10, 'Si': 11, 'Mo': 12, 'Ca': 13, 'W': 14,
    'Pb': 15, 'B': 16, 'V': 17, 'Co': 18, 'Mg': 19, 'Bi': 20, 'Fe': 21,
    'Ba': 22, 'K': 23, 'Ti': 24, 'Sn': 25, 'Cd': 26, 'I': 27, 'Re': 28,
    'Sr': 29, 'H': 30, 'Cu': 31, 'Ni': 32, 'Lu': 33, 'Pr': 34, 'Te': 35,
    'Ce': 36, 'Nd': 37, 'Gd': 38, 'Zr': 39, 'Mn': 40, 'As': 41, 'Hg': 42,
    'Sb': 43, 'Cr': 44, 'Se': 45, 'La': 46, 'Dy': 47, 'Y': 48, 'Pd': 49,
    'Ag': 50, 'In': 51, 'Li': 52, 'Rh': 53, 'Nb': 54, 'Hf': 55, 'Cs': 56,
    'Ru': 57, 'Au': 58, 'Sm': 59, 'Ta': 60, 'Pt': 61, 'Ir': 62, 'Be': 63,
    'Ge': 64}

    Bond Dict: {'NONE': 0, 'SINGLE': 1, 'DOUBLE': 2, 'AROMATIC': 3,
    'TRIPLE': 4}

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
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in
            the final dataset. (default: :obj:`None`)
    """

    url = 'https://www.dropbox.com/s/lzu9lmukwov12kt/aqsol_graph_raw.zip?dl=1'

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = "AQSOL"
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'asqol_graph_raw'), self.raw_dir)
        os.unlink(path)

    def process(self):
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)

            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx]

                # Each `graph` is a tuple (x, edge_attr, edge_index, y)
                #     Shape of x : [num_nodes, 1]
                #     Shape of edge_attr : [num_edges]
                #     Shape of edge_index : [2, num_edges]
                #     Shape of y : [1]

                x = torch.LongTensor(graph[0]).unsqueeze(-1)
                edge_attr = torch.LongTensor(graph[1])
                edge_index = torch.LongTensor(graph[2])
                y = torch.tensor(graph[3])

                data = Data(edge_index=edge_index)

                if edge_index.shape[1] == 0:
                    continue
                    # skipping for graphs with no bonds/edges

                if data.num_nodes != len(x):
                    continue
                    # cleaning <10 graphs with this discrepancy

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
