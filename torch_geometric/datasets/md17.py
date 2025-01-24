import os
import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
    extract_zip,
)


class MD17(InMemoryDataset):
    r"""A variety of ab-initio molecular dynamics trajectories from the authors
    of `sGDML <http://quantum-machine.org/gdml>`_.
    This class provides access to the original MD17 datasets, their revised
    versions, and the CCSD(T) trajectories.

    For every trajectory, the dataset contains the Cartesian positions of atoms
    (in Angstrom), their atomic numbers, as well as the total energy
    (in kcal/mol) and forces (kcal/mol/Angstrom) on each atom.
    The latter two are the regression targets for this collection.

    .. note::

        Data objects contain no edge indices as these are most commonly
        constructed via the :obj:`torch_geometric.transforms.RadiusGraph`
        transform, with its cut-off being a hyperparameter.

    The `original MD17 dataset <https://arxiv.org/abs/1611.04678>`_ contains
    ten molecule trajectories.
    This version of the dataset was found to suffer from high numerical noise.
    The `revised MD17 dataset <https://arxiv.org/abs/2007.09593>`_ contains the
    same molecules, but the energies and forces were recalculated at the
    PBE/def2-SVP level of theory using very tight SCF convergence and very
    dense DFT integration grid.
    The third version of the dataset contains fewer molecules, computed at the
    CCSD(T) level of theory.
    The benzene molecule at the DFT FHI-aims level of theory was
    `released separately <https://arxiv.org/abs/1802.09238>`_.

    Check the table below for detailed information on the molecule, level of
    theory and number of data points contained in each dataset.
    Which trajectory is loaded is determined by the :attr:`name` argument.
    For the coupled cluster trajectories, the dataset comes with pre-defined
    training and testing splits which are loaded separately via the
    :attr:`train` argument.

    +--------------------+--------------------+-------------------------------+-----------+
    | Molecule           | Level of Theory    | Name                          | #Examples |
    +====================+====================+===============================+===========+
    | Benzene            | DFT                | :obj:`benzene`                | 627,983   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Uracil             | DFT                | :obj:`uracil`                 | 133,770   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Naphthalene        | DFT                | :obj:`napthalene`             | 326,250   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Aspirin            | DFT                | :obj:`aspirin`                | 211,762   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Salicylic acid     | DFT                | :obj:`salicylic acid`         | 320,231   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Malonaldehyde      | DFT                | :obj:`malonaldehyde`          | 993,237   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Ethanol            | DFT                | :obj:`ethanol`                | 555,092   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Toluene            | DFT                | :obj:`toluene`                | 442,790   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Paracetamol        | DFT                | :obj:`paracetamol`            | 106,490   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Azobenzene         | DFT                | :obj:`azobenzene`             | 99,999    |
    +--------------------+--------------------+-------------------------------+-----------+
    | Benzene (R)        | DFT (PBE/def2-SVP) | :obj:`revised benzene`        | 100,000   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Uracil (R)         | DFT (PBE/def2-SVP) | :obj:`revised uracil`         | 100,000   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Naphthalene (R)    | DFT (PBE/def2-SVP) | :obj:`revised napthalene`     | 100,000   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Aspirin (R)        | DFT (PBE/def2-SVP) | :obj:`revised aspirin`        | 100,000   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Salicylic acid (R) | DFT (PBE/def2-SVP) | :obj:`revised salicylic acid` | 100,000   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Malonaldehyde (R)  | DFT (PBE/def2-SVP) | :obj:`revised malonaldehyde`  | 100,000   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Ethanol (R)        | DFT (PBE/def2-SVP) | :obj:`revised ethanol`        | 100,000   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Toluene (R)        | DFT (PBE/def2-SVP) | :obj:`revised toluene`        | 100,000   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Paracetamol (R)    | DFT (PBE/def2-SVP) | :obj:`revised paracetamol`    | 100,000   |
    +--------------------+--------------------+-------------------------------+-----------+
    | Azobenzene (R)     | DFT (PBE/def2-SVP) | :obj:`revised azobenzene`     | 99,988    |
    +--------------------+--------------------+-------------------------------+-----------+
    | Benzene            | CCSD(T)            | :obj:`benzene CCSD(T)`        | 1,500     |
    +--------------------+--------------------+-------------------------------+-----------+
    | Aspirin            | CCSD               | :obj:`aspirin CCSD`           | 1,500     |
    +--------------------+--------------------+-------------------------------+-----------+
    | Malonaldehyde      | CCSD(T)            | :obj:`malonaldehyde CCSD(T)`  | 1,500     |
    +--------------------+--------------------+-------------------------------+-----------+
    | Ethanol            | CCSD(T)            | :obj:`ethanol CCSD(T)`        | 2,000     |
    +--------------------+--------------------+-------------------------------+-----------+
    | Toluene            | CCSD(T)            | :obj:`toluene CCSD(T)`        | 1,501     |
    +--------------------+--------------------+-------------------------------+-----------+
    | Benzene            | DFT FHI-aims       | :obj:`benzene FHI-aims`       | 49,863    |
    +--------------------+--------------------+-------------------------------+-----------+

    .. warning::

        It is advised to not train a model on more than 1,000 samples from the
        original or revised MD17 dataset.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): Keyword of the trajectory that should be loaded.
        train (bool, optional): Determines whether the train or test split
            gets loaded for the coupled cluster trajectories.
            (default: :obj:`None`)
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
        :widths: 20 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - Benzene
          - 627,983
          - 12
          - 0
          - 1
          - 2
        * - Uracil
          - 133,770
          - 12
          - 0
          - 1
          - 2
        * - Naphthalene
          - 326,250
          - 10
          - 0
          - 1
          - 2
        * - Aspirin
          - 211,762
          - 21
          - 0
          - 1
          - 2
        * - Salicylic acid
          - 320,231
          - 16
          - 0
          - 1
          - 2
        * - Malonaldehyde
          - 993,237
          - 9
          - 0
          - 1
          - 2
        * - Ethanol
          - 555,092
          - 9
          - 0
          - 1
          - 2
        * - Toluene
          - 442,790
          - 15
          - 0
          - 1
          - 2
        * - Paracetamol
          - 106,490
          - 20
          - 0
          - 1
          - 2
        * - Azobenzene
          - 99,999
          - 24
          - 0
          - 1
          - 2
        * - Benzene (R)
          - 100,000
          - 12
          - 0
          - 1
          - 2
        * - Uracil (R)
          - 100,000
          - 12
          - 0
          - 1
          - 2
        * - Naphthalene (R)
          - 100,000
          - 10
          - 0
          - 1
          - 2
        * - Aspirin (R)
          - 100,000
          - 21
          - 0
          - 1
          - 2
        * - Salicylic acid (R)
          - 100,000
          - 16
          - 0
          - 1
          - 2
        * - Malonaldehyde (R)
          - 100,000
          - 9
          - 0
          - 1
          - 2
        * - Ethanol (R)
          - 100,000
          - 9
          - 0
          - 1
          - 2
        * - Toluene (R)
          - 100,000
          - 15
          - 0
          - 1
          - 2
        * - Paracetamol (R)
          - 100,000
          - 20
          - 0
          - 1
          - 2
        * - Azobenzene (R)
          - 99,988
          - 24
          - 0
          - 1
          - 2
        * - Benzene CCSD-T
          - 1,500
          - 12
          - 0
          - 1
          - 2
        * - Aspirin CCSD-T
          - 1,500
          - 21
          - 0
          - 1
          - 2
        * - Malonaldehyde CCSD-T
          - 1,500
          - 9
          - 0
          - 1
          - 2
        * - Ethanol CCSD-T
          - 2000
          - 9
          - 0
          - 1
          - 2
        * - Toluene CCSD-T
          - 1,501
          - 15
          - 0
          - 1
          - 2
        * - Benzene FHI-aims
          - 49,863
          - 12
          - 0
          - 1
          - 2
    """  # noqa: E501

    gdml_url = 'http://quantum-machine.org/gdml/data/npz'
    revised_url = ('https://archive.materialscloud.org/record/'
                   'file?filename=rmd17.tar.bz2&record_id=466')

    file_names = {
        'benzene': 'md17_benzene2017.npz',
        'uracil': 'md17_uracil.npz',
        'naphtalene': 'md17_naphthalene.npz',
        'aspirin': 'md17_aspirin.npz',
        'salicylic acid': 'md17_salicylic.npz',
        'malonaldehyde': 'md17_malonaldehyde.npz',
        'ethanol': 'md17_ethanol.npz',
        'toluene': 'md17_toluene.npz',
        'paracetamol': 'paracetamol_dft.npz',
        'azobenzene': 'azobenzene_dft.npz',
        'revised benzene': 'rmd17_benzene.npz',
        'revised uracil': 'rmd17_uracil.npz',
        'revised naphthalene': 'rmd17_naphthalene.npz',
        'revised aspirin': 'rmd17_aspirin.npz',
        'revised salicylic acid': 'rmd17_salicylic.npz',
        'revised malonaldehyde': 'rmd17_malonaldehyde.npz',
        'revised ethanol': 'rmd17_ethanol.npz',
        'revised toluene': 'rmd17_toluene.npz',
        'revised paracetamol': 'rmd17_paracetamol.npz',
        'revised azobenzene': 'rmd17_azobenzene.npz',
        'benzene CCSD(T)': 'benzene_ccsd_t.zip',
        'aspirin CCSD': 'aspirin_ccsd.zip',
        'malonaldehyde CCSD(T)': 'malonaldehyde_ccsd_t.zip',
        'ethanol CCSD(T)': 'ethanol_ccsd_t.zip',
        'toluene CCSD(T)': 'toluene_ccsd_t.zip',
        'benzene FHI-aims': 'benzene2018_dft.npz',
    }

    def __init__(
        self,
        root: str,
        name: str,
        train: Optional[bool] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if name not in self.file_names:
            raise ValueError(f"Unknown dataset name '{name}'")

        self.name = name
        self.revised = 'revised' in name
        self.ccsd = 'CCSD' in self.name

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        if len(self.processed_file_names) == 1 and train is not None:
            raise ValueError(
                f"'{self.name}' dataset does not provide pre-defined splits "
                f"but the 'train' argument is set to '{train}'")
        elif len(self.processed_file_names) == 2 and train is None:
            raise ValueError(
                f"'{self.name}' dataset does provide pre-defined splits but "
                f"the 'train' argument was not specified")

        idx = 0 if train is None or train else 1
        self.load(self.processed_paths[idx])

    def mean(self) -> float:
        assert isinstance(self._data, Data)
        return float(self._data.energy.mean())

    @property
    def raw_dir(self) -> str:
        if self.revised:
            return osp.join(self.root, 'raw')
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        name = self.file_names[self.name]
        if self.revised:
            return osp.join('rmd17', 'npz_data', name)
        elif self.ccsd:
            return [name[:-4] + '-train.npz', name[:-4] + '-test.npz']
        return name

    @property
    def processed_file_names(self) -> List[str]:
        if self.ccsd:
            return ['train.pt', 'test.pt']
        else:
            return ['data.pt']

    def download(self) -> None:
        if self.revised:
            path = download_url(self.revised_url, self.raw_dir)
            extract_tar(path, self.raw_dir, mode='r:bz2')
            os.unlink(path)
        else:
            url = f'{self.gdml_url}/{self.file_names[self.name]}'
            path = download_url(url, self.raw_dir)
            if self.ccsd:
                extract_zip(path, self.raw_dir)
                os.unlink(path)

    def process(self) -> None:
        it = zip(self.raw_paths, self.processed_paths)
        for raw_path, processed_path in it:
            raw_data = np.load(raw_path)

            if self.revised:
                z = torch.from_numpy(raw_data['nuclear_charges']).long()
                pos = torch.from_numpy(raw_data['coords']).float()
                energy = torch.from_numpy(raw_data['energies']).float()
                force = torch.from_numpy(raw_data['forces']).float()
            else:
                z = torch.from_numpy(raw_data['z']).long()
                pos = torch.from_numpy(raw_data['R']).float()
                energy = torch.from_numpy(raw_data['E']).float()
                force = torch.from_numpy(raw_data['F']).float()

            data_list = []
            for i in range(pos.size(0)):
                data = Data(z=z, pos=pos[i], energy=energy[i], force=force[i])
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            self.save(data_list, processed_path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"
