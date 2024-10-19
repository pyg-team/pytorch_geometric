import sys
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from torch_geometric.utils import one_hot

samples = [
    {
        'smiles': 'CCCCC(=O)N',
        'instruction': 'What is ∼ functional related to?',
        'ground_truth': '∼ is functionally related to a valeric acid'
    },
    {
        'smiles':
        'CC(C)C(=O)C(=O)O',
        'instruction':
        'What is the chemical structure of the molecule represented by ∼?',
        'ground_truth':
        '∼ is a 2-oxo monocarboxylic acid and a branched-chain keto acid'
    },
    {
        'smiles': 'CCCCC(=O)N',
        'instruction': 'What is ∼ functional related to?',
        'ground_truth': '∼ is functionally related to a valeric acid'
    },
    {
        'smiles':
        'CC(C)C(=O)C(=O)O',
        'instruction':
        'What is the chemical structure of the molecule represented by ∼?',
        'ground_truth':
        '∼ is a 2-oxo monocarboxylic acid and a branched-chain keto acid'
    },
    {
        'smiles': 'CCCCC(=O)N',
        'instruction': 'What is ∼ functional related to?',
        'ground_truth': '∼ is functionally related to a valeric acid'
    },
    {
        'smiles':
        'CC(C)C(=O)C(=O)O',
        'instruction':
        'What is the chemical structure of the molecule represented by ∼?',
        'ground_truth':
        '∼ is a 2-oxo monocarboxylic acid and a branched-chain keto acid'
    },
    {
        'smiles': 'CCCCC(=O)N',
        'instruction': 'What is ∼ functional related to?',
        'ground_truth': '∼ is functionally related to a valeric acid'
    },
    {
        'smiles':
        'CC(C)C(=O)C(=O)O',
        'instruction':
        'What is the chemical structure of the molecule represented by ∼?',
        'ground_truth':
        '∼ is a 2-oxo monocarboxylic acid and a branched-chain keto acid'
    },
    {
        'smiles': 'CCCCC(=O)N',
        'instruction': 'What is ∼ functional related to?',
        'ground_truth': '∼ is functionally related to a valeric acid'
    },
    {
        'smiles':
        'CC(C)C(=O)C(=O)O',
        'instruction':
        'What is the chemical structure of the molecule represented by ∼?',
        'ground_truth':
        '∼ is a 2-oxo monocarboxylic acid and a branched-chain keto acid'
    },
]


class MoleculeGPTDataset(InMemoryDataset):
    r"""The dataset from the `"MoleculeGPT: Instruction Following Large
    Language Models for Molecular Property Prediction"
    <https://ai4d3.github.io/papers/34.pdf>`_ paper.

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
        return ['pubchem.csv']

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self) -> None:
        pass

    def process(self) -> None:
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            RDLogger.DisableLog('rdApp.*')  # type: ignore
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

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        data_list = []
        for i, mol in enumerate(samples):
            smiles = mol['smiles']
            instruction = mol['instruction']
            ground_truth = mol['ground_truth']

            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)

            x = [types[atom.GetSymbol()] for atom in mol.GetAtoms()]
            x = one_hot(torch.tensor(x), num_classes=len(types))
            x = x.to(torch.float)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_types += [bonds[bond.GetBondType()]] * 2
                rows += [i, j]
                cols += [j, i]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                smiles=smiles,
                instruction=instruction,
                y=ground_truth,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
