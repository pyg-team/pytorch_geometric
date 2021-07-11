from typing import Optional, Callable

import os

from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset,
                                  download_url,
                                  extract_zip,
                                  Data)

base_url = "http://quantum-machine.org/gdml/data/npz/"

# FHI-aims
benzene_aims_url = base_url + "benzene_dft.npz"

# Original MD17 (DFT)
benzene_url = base_url + "benzene_old_dft.npz"
uracil_url = base_url + "uracil_dft.npz"
naphtalene_url = base_url + "naphthalene_dft.npz"
aspirin_url = base_url + "aspirin_dft.npz"
salicylic_url = base_url + "salicylic_dft.npz"
malonaldehyde_url = base_url + "malonaldehyde_dft.npz"
ethanol_url = base_url + "ethanol_dft.npz"
toluene_url = base_url + "toluene_dft.npz"

# Other DFT trajectories
paracetamol_url = base_url + "paracetamol_dft.npz"
azobenzene_url = base_url + "azobenzene_dft.npz"

# CCSD
aspirin_ccsd_url = base_url + "aspirin_ccsd.zip"

# CCSD(T)
benzene_ccsdt_url = base_url + "benzene_ccsd_t.zip"
malonaldehyde_ccsdt_url = base_url + "malonaldehyde_ccsd_t.zip"
toluene_ccsdt_url = base_url + "toluene_ccsd_t.zip"
ethanol_ccsdt_url = base_url + "ethanol_ccsd_t.zip"


dataset_dict = {"benzene FHI-aims"      : benzene_url,
                "benzene"               : benzene_url,
                "uracil"                : uracil_url,
                "napthalene"            : naphtalene_url,
                "aspirin"               : aspirin_url,
                "salicylic acid"        : salicylic_url,
                "malonaldehyde"         : malonaldehyde_url,
                "ethanol"               : ethanol_url,
                "toluene"               : toluene_url,
                "paracetamol"           : paracetamol_url,
                "azobenzene"            : azobenzene_url,
                "aspirin CCSD"          : aspirin_ccsd_url,
                "benzene CCSD(T)"       : benzene_ccsdt_url,
                "malonaldehyde CCSD(T)" : malonaldehyde_ccsdt_url,
                "toluene CCSD(T)"       : toluene_ccsdt_url,
                "ethanol CCSD(T)"       : ethanol_ccsdt_url,
                }


class MD17(InMemoryDataset):
    r"""A variety of ab-initio molecular dynamics trajectories
    from the authors of sGDML. This class provides access
    to the original MD17 datasets as well as all other datsets
    released by sGDML since then (16 in total).

    For every trajectory, the dataset contains the cartesian positions
    of the atoms (Angstrom), their atomic numbers, the total energy (kcal/mol)
    and the forces (kcal/mol/Angstrom) on each atom.
    The latter two are the regression targets for this collection.

    .. note::

        Data object contains no edge indices as these are most commonly
        constructed from a :obj:`torch_geometric.nn.pool.radius_graph`
        where the cut off is a hyperparameter.

    Some of the trajectories were computed at different levels of theory and for
    most there are two versions: generally a long trajectory on DFT level of theory
    and a short trajectory on coupled cluster level of theory.
    Check the table below for detailed information on the molecule,
    level of theory and number of data points contained in each dataset.
    For the coupled cluster trajectories the dataset comes with pre-defined
    training and testing splits which can be loaded with the :obj:`train` variable.
    For the other datasets, this variable does nothing..
    Which trajectory is loaded is determined by the :obj:`name` variable.

    When using these datasets, make sure to cite the following publications:
    `"Machine Learning of Accurate Energy-conserving Molecular Force Fields"
    <http://advances.sciencemag.org/content/3/5/e1603015>`_
    `"Towards Exact Molecular Dynamics Simulations with Machine-Learned
    Force Fields" <https://www.nature.com/articles/s41467-018-06169-2>`_
    `"sGDML: Constructing Accurate and Data Efficient Molecular Force Fields
    Using Machine Learning" <https://doi.org/10.1016/j.cpc.2019.02.007>`_
    `"Molecular Force Fields with Gradient-Domain Machine Learning:
    Construction and Application to Dynamics of Small Molecules with
    Coupled Cluster Forces" <https://doi.org/10.1016/j.cpc.2019.02.007>`_

    +----------------+-----------------+-----------------------+--------------+
    | Molecule       | Level of Theory | :obj:`name`           | #data points |
    +================+=================+=======================+==============+
    +----------------+-----------------+-----------------------+--------------+
    | Benzene        | DFT             | benzene               | 49 863       |
    +----------------+-----------------+-----------------------+--------------+
    | Benzene        | DFT FHI-aims    | benzene FHI-aims      | 627 983      |
    +----------------+-----------------+-----------------------+--------------+
    | Uracil         | DFT             | uracil                | 133 770      |
    +----------------+-----------------+-----------------------+--------------+
    | Naphthalene    | DFT             | naphthalene           | 326 250      |
    +----------------+-----------------+-----------------------+--------------+
    | Aspirin        | DFT             | aspirin               | 627 983      |
    +----------------+-----------------+-----------------------+--------------+
    | Aspirin        | CCSD            | aspirin CCSD          | 1 500        |
    +----------------+-----------------+-----------------------+--------------+
    | Salicylic acid | DFT             | salicylic acid        | 320 231      |
    +----------------+-----------------+-----------------------+--------------+
    | Malonaldehyde  | DFT             | malonaldehyde         | 993 237      |
    +----------------+-----------------+-----------------------+--------------+
    | Malonaldehyde  | CCSD(T)         | malonaldehyde CCSD(T) | 1 500        |
    +----------------+-----------------+-----------------------+--------------+
    | Ethanol        | DFT             | ethanol               | 555 092      |
    +----------------+-----------------+-----------------------+--------------+
    | Ethanol        | CCSD(T)         | ethanol CCSD(T)       | 2 000        |
    +----------------+-----------------+-----------------------+--------------+
    | Toluene        | DFT             | toluene               | 442 790      |
    +----------------+-----------------+-----------------------+--------------+
    | Toluene        | CCSD(T)         | toluene CCSD(T)       | 1 500        |
    +----------------+-----------------+-----------------------+--------------+
    | Paracetamol    | DFT             | paracetamol           | 106 490      |
    +----------------+-----------------+-----------------------+--------------+
    | Azobenzene     | DFT             | azobenzene            | 99 999       |
    +----------------+-----------------+-----------------------+--------------+

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): Keyword of the trajectory that should be loaded.
        train (bool, optional): Determines whether the train or test split
            gets loaded for the coupled cluster trajectories. For the DFT
            trajectories this variable can be ignored. (default: :obj:`True`)
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
    """

    def __init__(self, root: str, name: str, train: Optional[bool] = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name
        self.url = dataset_dict[name]
        self.train = train

        self.file_name, self.file_type = self.url.split("/")[-1].split(".")

        self.raw_file_name = self.file_name + "." + self.file_type
        self.processed_file_name = self.file_name + ".pt"

        super().__init__(root, transform, pre_transform, pre_filter)

        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    def mean(self) -> float:
        E_tot = torch.cat([self.get(i).E for i in range(len(self))], dim=0)
        return float(E_tot.mean())

    @property
    def raw_file_names(self) -> list:
        if self.file_type == "zip":
            return [self.file_name + "-train.npz", self.file_name + "-test.npz"]
        else:
            return [self.raw_file_name]

    @property
    def processed_file_names(self) -> list:
        if self.file_type == "zip":
            return [self.file_name + "-train.pt", self.file_name + "-test.pt"]
        else:
            return [self.processed_file_name]

    def download(self):
        path = download_url(self.url, self.raw_dir)
        if self.file_type == "zip":
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process_npz(self, raw_data) -> list:
        data_list = []
        z = torch.tensor(raw_data["z"], dtype=torch.int64)
        pos = torch.tensor(raw_data["R"], dtype=torch.float32)
        energy = torch.tensor(raw_data["E"], dtype=torch.float32)
        force = torch.tensor(raw_data["F"], dtype=torch.float32)

        for ii in tqdm(range(len(raw_data["E"]))):
            data = Data(z=z, pos=pos[ii], E=energy[ii], F=force[ii], idx=ii)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        return data_list

    def process(self):

        # Training data
        raw_data = np.load(self.raw_paths[0])

        data_list = self.process_npz(raw_data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # Testing data
        if len(self.raw_paths) > 1:
            raw_data = np.load(self.raw_paths[1])

            data_list = self.process_npz(raw_data)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[1])
