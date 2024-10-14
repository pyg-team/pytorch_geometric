import os
import os.path as osp
import sys
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs
from torch_geometric.utils import one_hot, scatter

import pandas as pd

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor(
    [
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
    ]
)

class QM40(InMemoryDataset):

    raw_url = None
    processed_url = None

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
        try:
            import rdkit  # noqa

            return ["QM40_main.csv", "QM40_xyz.csv", "QM40_bond.csv"]
        except ImportError:
            return ["qm40.pt"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    # def download(self) -> None:
    #     try:
    #         import rdkit  # noqa

    #         file_path = download_url(self.raw_url, self.raw_dir)
    #         extract_zip(file_path, self.raw_dir)
    #         os.unlink(file_path)

    #     except ImportError:
    #         path = download_url(self.processed_url, self.raw_dir)
    #         extract_zip(path, self.raw_dir)
    #         os.unlink(path)
    def download(self) -> None:
        pass

    def process(self) -> None:
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType

            RDLogger.DisableLog("rdApp.*")  # type: ignore
            WITH_RDKIT = True

        except ImportError:
            WITH_RDKIT = False

        if not WITH_RDKIT:
            print(
                (
                    "Using a pre-processed version of the dataset. Please "
                    "install 'rdkit' to alternatively process the raw data."
                ),
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
        main_df = pd.read_csv(self.raw_paths[0])
        xyz_df = pd.read_csv(self.raw_paths[1])
        bond_df = pd.read_csv(self.raw_paths[2])

        # Pre-process xyz_df and bond_df
        xyz_df_grouped = xyz_df.groupby('Zinc_id')
        bond_df_grouped = bond_df.groupby('Zinc_id')

        data_list = []
        print("Processing raw data...")
        for mol_idx, row in tqdm(main_df.iterrows(), total=len(main_df)):
            ID = row['Zinc_id']
            SMILES = row['smile']
            y = torch.tensor(row.iloc[2:].values.astype(np.float32), dtype=torch.float)

            mol_xyz = xyz_df_grouped.get_group(ID).reset_index(drop=True)
            mol_bonds = bond_df_grouped.get_group(ID).reset_index(drop=True)

            mol = Chem.MolFromSmiles(SMILES)
            mol = Chem.AddHs(mol)
            N = mol.GetNumAtoms()

            # Vectorize atom property extraction
            atoms = mol.GetAtoms()
            atomic_numbers = np.array([atom.GetAtomicNum() for atom in atoms])
            type_idx = np.array([type_idx_map[num] for num in atomic_numbers])
            aromatic = np.array([int(atom.GetIsAromatic()) for atom in atoms])
            hybridizations = np.array([atom.GetHybridization() for atom in atoms])
            sp = (hybridizations == HybridizationType.SP).astype(int)
            sp2 = (hybridizations == HybridizationType.SP2).astype(int)
            sp3 = (hybridizations == HybridizationType.SP3).astype(int)

            # Set charges and positions
            conf = Chem.Conformer(N)
            for i, row in mol_xyz.iterrows():
                atoms[i].SetFormalCharge(int(round(row["charge"])))
                conf.SetAtomPosition(i, (row["final_x"], row["final_y"], row["final_z"]))

            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
            z = torch.tensor(atomic_numbers, dtype=torch.long)

            # Process bonds
            bond_data = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bonds[bond.GetBondType()]) 
                        for bond in mol.GetBonds()]
            rows, cols, edge_types = zip(*bond_data)
            rows, cols = rows + cols, cols + rows  # Add reverse edges
            edge_types = edge_types + edge_types

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))
            edge_attr2 = torch.tensor(mol_bonds['lmod'].tolist(), dtype=torch.float)

            # Sort edges
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            # Count hydrogens
            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

            # Create node features
            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor(np.array([aromatic, sp, sp2, sp3, num_hs]), dtype=torch.float).t().contiguous()
            x = torch.cat([x1, x2], dim=-1)

            data = Data(
                x=x, z=z, pos=pos, edge_index=edge_index, smiles=SMILES,
                edge_attr=edge_attr, edge_attr2=edge_attr2, y=y * conversion.view(1, -1), name=ID, idx=mol_idx,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    # def process(self) -> None:
    #     try:
    #         from rdkit import Chem, RDLogger
    #         from rdkit.Chem.rdchem import BondType as BT
    #         from rdkit.Chem.rdchem import HybridizationType

    #         RDLogger.DisableLog("rdApp.*")  # type: ignore
    #         WITH_RDKIT = True

    #     except ImportError:
    #         WITH_RDKIT = False

    #     if not WITH_RDKIT:
    #         print(
    #             (
    #                 "Using a pre-processed version of the dataset. Please "
    #                 "install 'rdkit' to alternatively process the raw data."
    #             ),
    #             file=sys.stderr,
    #         )

    #         data_list = fs.torch_load(self.raw_paths[0])
    #         data_list = [Data(**data_dict) for data_dict in data_list]

    #         if self.pre_filter is not None:
    #             data_list = [d for d in data_list if self.pre_filter(d)]

    #         if self.pre_transform is not None:
    #             data_list = [self.pre_transform(d) for d in data_list]

    #         self.save(data_list, self.processed_paths[0])
    #         return

    #     types = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}
    #     type_idx_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}
    #     bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    #     print("Loading raw data...")
    #     main_df = pd.read_csv(self.raw_paths[0])
    #     xyz_df = pd.read_csv(self.raw_paths[1])
    #     bond_df = pd.read_csv(self.raw_paths[2])

    #     data_list = []
    #     print("Processing raw data...")
    #     for mol_idx, row in tqdm(main_df.iterrows(), total=len(main_df)):

    #         ID = row['Zinc_id']
    #         SMILES = row['smile']

    #         y = row.iloc[2:].values.astype(np.float32)

    #         mol_xyz = xyz_df.loc[xyz_df['Zinc_id'] == ID].reset_index(drop=True)
    #         mol_bonds = bond_df.loc[bond_df['Zinc_id'] == ID].reset_index(drop=True)

    #         mol = Chem.MolFromSmiles(SMILES)
    #         mol = Chem.AddHs(mol)
    #         conf = Chem.Conformer(len(mol_xyz))
    #         N = mol.GetNumAtoms()

    #         type_idx = []
    #         atomic_number = []
    #         aromatic = []
    #         sp = []
    #         sp2 = []
    #         sp3 = []
    #         num_hs = []

    #         for i, row in mol_xyz.iterrows():
    #             atom = mol.GetAtomWithIdx(i)
    #             atom.SetFormalCharge(int(round(row["charge"])))
    #             conf.SetAtomPosition(i, (row["final_x"], row["final_y"], row["final_z"]))

    #             type_idx.append(type_idx_map[atom.GetAtomicNum()])
    #             atomic_number.append(atom.GetAtomicNum())
    #             aromatic.append(1 if atom.GetIsAromatic() else 0)
    #             hybridization = atom.GetHybridization()
    #             sp.append(1 if hybridization == HybridizationType.SP else 0)
    #             sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
    #             sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    #         pos = conf.GetPositions()
    #         pos = torch.tensor(pos, dtype=torch.float)

    #         z = torch.tensor(atomic_number, dtype=torch.long)

    #         rows, cols, edge_types, lmods = [], [], [], []
    #         for bond, bond_row in zip(mol.GetBonds(), mol_bonds.iterrows()):
    #             start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    #             rows += [start, end]
    #             cols += [end, start]
    #             edge_types += 2 * [bonds[bond.GetBondType()]]
    #             lmods.append(bond_row[1]['lmod'])

    #         edge_index = torch.tensor([rows, cols], dtype=torch.long)
    #         edge_type = torch.tensor(edge_types, dtype=torch.long)
    #         edge_attr = one_hot(edge_type, num_classes=len(bonds))
    #         edge_attr2 = torch.tensor(lmods, dtype=torch.float)

    #         perm = (edge_index[0] * N + edge_index[1]).argsort()
    #         edge_index = edge_index[:, perm]
    #         edge_type = edge_type[perm]
    #         edge_attr = edge_attr[perm]

    #         row, col = edge_index
    #         # count hydrogens
    #         hs = (z == 1).to(torch.float)
    #         num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

    #         # one hot encoding of atom types
    #         x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
    #         # other properties
    #         x2 = (
    #             torch.tensor(
    #                 [atomic_number, aromatic, sp, sp2, sp3, num_hs],
    #                 dtype=torch.float,
    #             )
    #             .t()
    #             .contiguous()
    #         )
    #         # concat
    #         x = torch.cat([x1, x2], dim=-1)

    #         data = Data(
    #             x=x,
    #             z=z,
    #             pos=pos,
    #             edge_index=edge_index,
    #             smiles=SMILES,
    #             edge_attr=edge_attr,
    #             edge_attr2=edge_attr2,
    #             y=y,
    #             name=ID,
    #             idx=mol_idx,
    #         )

    #         if self.pre_filter is not None and not self.pre_filter(data):
    #             continue
    #         if self.pre_transform is not None:
    #             data = self.pre_transform(data)

    #         data_list.append(data)

    #     self.save(data_list, self.processed_paths[0])
