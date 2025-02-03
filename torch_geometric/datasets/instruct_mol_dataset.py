import json
import sys
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from torch_geometric.utils import one_hot


class InstructMolDataset(InMemoryDataset):
    r"""The dataset from the `"InstructMol: Multi-Modal Integration for
    Building a Versatile and Reliable Molecular Assistant in Drug Discovery"
    <https://arxiv.org/pdf/2311.16208>`_ paper.

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
    """
    raw_url = 'https://huggingface.co/datasets/OpenMol/PubChemSFT/resolve/main'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['all_clean.json']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def download(self) -> None:
        print('downloading dataset...')
        fs.cp(f'{self.raw_url}/all_clean.json', self.raw_dir)

    def process(self) -> None:
        try:
            from rdkit import Chem
            from rdkit.Chem.rdchem import BondType as BT
            WITH_RDKIT = True

        except ImportError:
            WITH_RDKIT = False

        if not WITH_RDKIT:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = fs.torch_load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_paths[0])
            return

        # types of atom and bond
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Unknow': 5}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        # load data
        mols = json.load(open(f'{self.raw_dir}/all_clean.json'))

        data_list = []
        for smiles, qa_pairs in tqdm(mols.items(), total=len(mols)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            x: torch.Tensor = torch.tensor([
                types[atom.GetSymbol()] if atom.GetSymbol() in types else 5
                for atom in mol.GetAtoms()
            ])
            x = one_hot(x, num_classes=len(types), dtype=torch.float)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_types += [bonds[bond.GetBondType()]] * 2
                rows += [i, j]
                cols += [j, i]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            for question, answer in qa_pairs:
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    smiles=smiles,
                    instruction=question,
                    y=answer,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        self.save(data_list, self.processed_paths[0])
