import os
import os.path as osp
import re
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import torch

from torch_geometric.data import InMemoryDataset, download_url, extract_gz
from torch_geometric.utils import from_smiles as _from_smiles


class MoleculeNet(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.org/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the :ogb:`null`
    `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"ESOL"`, :obj:`"FreeSolv"`,
            :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`, :obj:`"HIV"`,
            :obj:`"BACE"`, :obj:`"BBBP"`, :obj:`"Tox21"`, :obj:`"ToxCast"`,
            :obj:`"SIDER"`, :obj:`"ClinTox"`).
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
        from_smiles (callable, optional): A custom function that takes a SMILES
            string and outputs a :obj:`~torch_geometric.data.Data` object.
            If not set, defaults to :meth:`~torch_geometric.utils.from_smiles`.
            (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - ESOL
          - 1,128
          - ~13.3
          - ~27.4
          - 9
          - 1
        * - FreeSolv
          - 642
          - ~8.7
          - ~16.8
          - 9
          - 1
        * - Lipophilicity
          - 4,200
          - ~27.0
          - ~59.0
          - 9
          - 1
        * - PCBA
          - 437,929
          - ~26.0
          - ~56.2
          - 9
          - 128
        * - MUV
          - 93,087
          - ~24.2
          - ~52.6
          - 9
          - 17
        * - HIV
          - 41,127
          - ~25.5
          - ~54.9
          - 9
          - 1
        * - BACE
          - 1513
          - ~34.1
          - ~73.7
          - 9
          - 1
        * - BBBP
          - 2,050
          - ~23.9
          - ~51.6
          - 9
          - 1
        * - Tox21
          - 7,831
          - ~18.6
          - ~38.6
          - 9
          - 12
        * - ToxCast
          - 8,597
          - ~18.7
          - ~38.4
          - 9
          - 617
        * - SIDER
          - 1,427
          - ~33.6
          - ~70.7
          - 9
          - 27
        * - ClinTox
          - 1,484
          - ~26.1
          - ~55.5
          - 9
          - 2
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: (display_name, url_name, csv_name, smiles_idx, y_idx)
    names: Dict[str, Tuple[str, str, str, int, Union[int, slice]]] = {
        'esol': ('ESOL', 'delaney-processed.csv', 'delaney-processed', -1, -2),
        'freesolv': ('FreeSolv', 'SAMPL.csv', 'SAMPL', 1, 2),
        'lipo': ('Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity', 2, 1),
        'pcba': ('PCBA', 'pcba.csv.gz', 'pcba', -1, slice(0, 128)),
        'muv': ('MUV', 'muv.csv.gz', 'muv', -1, slice(0, 17)),
        'hiv': ('HIV', 'HIV.csv', 'HIV', 0, -1),
        'bace': ('BACE', 'bace.csv', 'bace', 0, 2),
        'bbbp': ('BBBP', 'BBBP.csv', 'BBBP', -1, -2),
        'tox21': ('Tox21', 'tox21.csv.gz', 'tox21', -1, slice(0, 12)),
        'toxcast':
        ('ToxCast', 'toxcast_data.csv.gz', 'toxcast_data', 0, slice(1, 618)),
        'sider': ('SIDER', 'sider.csv.gz', 'sider', 0, slice(1, 28)),
        'clintox': ('ClinTox', 'clintox.csv.gz', 'clintox', 0, slice(1, 3)),
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        from_smiles: Optional[Callable] = None,
    ) -> None:
        self.name = name.lower()
        assert self.name in self.names.keys()
        self.from_smiles = from_smiles or _from_smiles
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        with open(self.raw_paths[0]) as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            values = line.split(',')

            smiles = values[self.names[self.name][3]]
            labels = values[self.names[self.name][4]]
            labels = labels if isinstance(labels, list) else [labels]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in labels]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            data = self.from_smiles(smiles)
            data.y = y

            if data.num_nodes == 0:
                warnings.warn(
                    f"Skipping molecule '{smiles}' since it "
                    f"resulted in zero atoms", stacklevel=2)
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.names[self.name][0]}({len(self)})'
