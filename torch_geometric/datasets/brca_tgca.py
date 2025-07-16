import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs


class BrcaTcga(InMemoryDataset):
    r"""The breast cancer (BRCA TCGA Pan-Cancer Atlas) dataset consisting of
    patients with survival information and gene expression data from
    `cBioPortal <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4160307/>`_
    and a network of biological interactions between those nodes from
    `Pathway Commons <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7145667/>`_.
    The dataset contains the gene features of 1,082 patients, and the overall
    survival time (in months) of each patient as label.

    Pre-processing and example model codes on how to use this dataset can be
    found `here <https://github.com/cannin/pyg_pathway_commons_cbioportal>`_.

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
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
        * - 1,082
          - 9,288
          - 271,771
          - 1,082
    """
    url = 'https://zenodo.org/record/8251328/files/brca_tcga.zip?download=1'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['graph_idx.csv', 'graph_labels.csv', 'edge_index.pt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        fs.rm(self.raw_dir)
        os.rename(osp.join(self.root, 'brca_tcga'), self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        graph_feat = pd.read_csv(self.raw_paths[0], index_col=0).values
        graph_feat = torch.from_numpy(graph_feat).to(torch.float)
        graph_labels = np.loadtxt(self.raw_paths[1], delimiter=',')
        graph_label = torch.from_numpy(graph_labels).to(torch.float)
        edge_index = fs.torch_load(self.raw_paths[2])

        data_list = []
        for x, y in zip(graph_feat, graph_label):
            data = Data(x=x.view(-1, 1), edge_index=edge_index, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
