import csv
import os
import shutil
import sys
from typing import Callable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    download_google_url,
    extract_zip,
)
from torch_geometric.io import fs
from torch_geometric.utils import one_hot, scatter

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    HAR2EV,
    HAR2EV,
    HAR2EV,
    HAR2EV,
    1.0,
    1.0,
    1.0,
    KCALMOL2EV,
    1.0,
    1.0,
    1.0,
    HAR2EV,
    HAR2EV,
    HAR2EV,
    KCALMOL2EV,
    KCALMOL2EV,
])


def rename_files(root: str) -> None:
    # move files within the root/'QM40 dataset' directory to current directory
    for file in os.listdir(os.path.join(root, "QM40 dataset")):
        if file.startswith("QM40") and file.endswith(".csv"):
            os.rename(
                os.path.join(root, "QM40 dataset", file),
                os.path.join(root, file),
            )
    # remove excess 'QM40 dataset' directory
    os.rmdir(os.path.join(root, "QM40 dataset"))

    # remove excess __MACOSX directory and all its contents
    shutil.rmtree(os.path.join(root, "__MACOSX"))


class QM40(InMemoryDataset):
    r"""The QM40 dataset consisting of molecules with 16 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | Internal_E(0K)                   | Internal energy at 0K                                                             | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | HOMO                             | Energy of HOMO                                                                    | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | LUMO                             | Energy of LUMO                                                                    | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | HL_gap                           | Energy difference of (HOMO - LUMO)                                                | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | Polarizability                   | Isotropic polarizability                                                          | :math:`a_0^3`                               |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | spatial_extent                   | Electronic spatial extent                                                         | :math:`a_0^2`                               |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | dipol_moment                     | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | ZPE                              | Zero point energy                                                                 | :math:`\textrm{Kcal/mol}`                   |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | rot1                             | Rotational constant 1                                                             | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | rot2                             | Rotational constant 2                                                             | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | rot3                             | Rotational constant 3                                                             | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | Inter_E(298)                     | Internal energy at 298.15K                                                        | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | Enthalpy                         | Enthalpy at 298.15K                                                              | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | Free_E                           | Free energy at 298.15K                                                            | :math:`\textrm{Ha}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | CV                               | Heat capacity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | Entropy                          | Entropy at 298.15K                                                                | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    """

    raw_url = "https://figshare.com/ndownloader/files/47535647"
    # processed_url = "https://drive.google.com/file/d/1d4NACk0KIxwuPpSxq-fNiqIU3v9s6BLE"
    processed_url = "1d4NACk0KIxwuPpSxq-fNiqIU3v9s6BLE"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa

            return ["QM40_main.csv", "QM40_xyz.csv", "QM40_bond.csv"]
        except ImportError:
            return ["qm40.pt"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        try:
            import rdkit  # noqa

            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)
            rename_files(self.raw_dir)

        except ImportError:
            path = download_google_url(self.processed_url, self.raw_dir, filename="qm40.zip")
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType

            RDLogger.DisableLog("rdApp.*")
            WITH_RDKIT = True
        except ImportError:
            WITH_RDKIT = False

        if not WITH_RDKIT:
            print(
                ("Using a pre-processed version of the dataset. Please "
                 "install 'rdkit' to alternatively process the raw data."),
                file=sys.stderr,
            )

            data_list = fs.torch_load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_paths[0])
            return

        types = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}
        type_idx_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        print("Loading raw data...")
        main_data = {}
        xyz_data = {}
        lmfc_data = {}

        # Read main CSV
        with open(self.raw_paths[0]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ID = row["Zinc_id"]
                main_data[ID] = {
                    "smile":
                    row["smile"],
                    "properties": [
                        float(row[key]) for key in row.keys()
                        if key not in ["Zinc_id", "smile"]
                    ],
                }

        # Read xyz CSV
        with open(self.raw_paths[1]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ID = row["Zinc_id"]
                if ID not in xyz_data:
                    xyz_data[ID] = []
                xyz_data[ID].append({
                    "charge": float(row["charge"]),
                    "final_x": float(row["final_x"]),
                    "final_y": float(row["final_y"]),
                    "final_z": float(row["final_z"]),
                })

        # Read bond CSV
        with open(self.raw_paths[2]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ID = row["Zinc_id"]
                if ID not in lmfc_data:
                    lmfc_data[ID] = []
                lmfc_data[ID].append(
                    (int(row["atom1"]), int(row["atom2"]), float(row["lmod"])))

        data_list = []
        print("Processing raw data...")
        for mol_idx, ID in enumerate(tqdm(main_data.keys())):
            if ID not in lmfc_data or ID not in xyz_data:
                continue  # Skip molecules with missing data

            mol_data = main_data[ID]
            SMILES = mol_data["smile"]
            y = torch.tensor(mol_data["properties"], dtype=torch.float)

            mol_xyz = xyz_data[ID]

            mol = Chem.MolFromSmiles(SMILES)
            mol = Chem.AddHs(mol)
            N = mol.GetNumAtoms()

            # Vectorize atom property extraction
            atoms = mol.GetAtoms()
            atomic_numbers = np.array([atom.GetAtomicNum() for atom in atoms])
            type_idx = np.array([type_idx_map[num] for num in atomic_numbers])
            aromatic = np.array([int(atom.GetIsAromatic()) for atom in atoms])
            hybridizations = np.array(
                [atom.GetHybridization() for atom in atoms])
            sp = (hybridizations == HybridizationType.SP).astype(int)
            sp2 = (hybridizations == HybridizationType.SP2).astype(int)
            sp3 = (hybridizations == HybridizationType.SP3).astype(int)

            # Set charges and positions
            conf = Chem.Conformer(N)
            for i, xyz in enumerate(mol_xyz):
                atoms[i].SetFormalCharge(int(round(xyz["charge"])))
                conf.SetAtomPosition(
                    i, (xyz["final_x"], xyz["final_y"], xyz["final_z"]))

            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
            z = torch.tensor(atomic_numbers, dtype=torch.long)

            # Process bonds
            bond_data = [(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bonds[bond.GetBondType()],
            ) for bond in mol.GetBonds()]
            rows, cols, edge_types = zip(*bond_data)
            rows, cols = rows + cols, cols + rows  # Add reverse edges
            edge_types = edge_types + edge_types

            # local mode force constants
            lmfc = torch.stack([
                torch.stack([
                    torch.tensor([bond[0] - 1, bond[1] - 1, bond[2]],
                                 dtype=torch.float),
                    torch.tensor([bond[1] - 1, bond[0] - 1, bond[2]],
                                 dtype=torch.float),
                ]) for bond in lmfc_data[ID]
            ]).reshape(-1, 3)

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            # Sort edges
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            # Sort lmfc
            perm = (lmfc[:, 0] * N + lmfc[:, 1]).argsort()
            lmfc = lmfc[perm]

            # Count hydrogens
            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

            # Create node features
            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = (torch.tensor(
                np.array([aromatic, sp, sp2, sp3, num_hs]),
                dtype=torch.float,
            ).t().contiguous())
            x = torch.cat([x1, x2], dim=-1)

            data = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                smiles=SMILES,
                edge_attr=edge_attr,
                edge_attr2=lmfc[:, 2],
                y=y * conversion.view(1, -1),
                name=ID,
                idx=mol_idx,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
