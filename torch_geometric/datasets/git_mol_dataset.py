import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_google_url,
    extract_zip,
)
from torch_geometric.io import fs


def safe_index(lst: List[Any], e: int) -> int:
    return lst.index(e) if e in lst else len(lst) - 1


class GitMolDataset(InMemoryDataset):
    r"""The dataset from the `"GIT-Mol: A Multi-modal Large Language Model
    for Molecular Science with Graph, Image, and Text"
    <https://arxiv.org/pdf/2308.06911>`_ paper.

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
        split (int, optional): Datasets split, train/valid/test=0/1/2.
            (default: :obj:`0`)
    """

    raw_url_id = '1loBXabD6ncAFY-vanRsVtRUSFkEtBweg'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        split: int = 0,
    ):
        from torchvision import transforms

        self.split = split

        if self.split == 0:
            self.img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['train_3500.pkl', 'valid_450.pkl', 'test_450.pkl']

    @property
    def processed_file_names(self) -> str:
        return ['train.pt', 'valid.pt', 'test.pt'][self.split]

    def download(self) -> None:
        file_path = download_google_url(
            self.raw_url_id,
            self.raw_dir,
            'gitmol.zip',
        )
        extract_zip(file_path, self.raw_dir)

    def process(self) -> None:
        import pandas as pd
        from PIL import Image

        try:
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog('rdApp.*')
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

        allowable_features: Dict[str, List[Any]] = {
            'possible_atomic_num_list':
            list(range(1, 119)) + ['misc'],
            'possible_formal_charge_list':
            [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
            'possible_chirality_list': [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ],
            'possible_hybridization_list': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.UNSPECIFIED, 'misc'
            ],
            'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
            'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
            'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
            'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
            'possible_is_aromatic_list': [False, True],
            'possible_is_in_ring_list': [False, True],
            'possible_bond_type_list': [
                Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
                Chem.rdchem.BondType.ZERO
            ],
            'possible_bond_dirs': [  # only for double bond stereo information
                Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT
            ],
            'possible_bond_stereo_list': [
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE,
                Chem.rdchem.BondStereo.STEREOCIS,
                Chem.rdchem.BondStereo.STEREOTRANS,
                Chem.rdchem.BondStereo.STEREOANY,
            ],
            'possible_is_conjugated_list': [False, True]
        }

        data = pd.read_pickle(
            f'{self.raw_dir}/igcdata_toy/{self.raw_file_names[self.split]}')

        data_list = []
        for _, r in tqdm(data.iterrows(), total=data.shape[0]):
            smiles = r['isosmiles']
            mol = Chem.MolFromSmiles(smiles.strip('\n'))
            if mol is not None:
                # text
                summary = r['summary']
                # image
                cid = r['cid']
                img_file = f'{self.raw_dir}/igcdata_toy/imgs/CID_{cid}.png'
                img = Image.open(img_file).convert('RGB')
                img = self.img_transform(img).unsqueeze(0)
                # graph
                atom_features_list = []
                for atom in mol.GetAtoms():
                    atom_feature = [
                        safe_index(
                            allowable_features['possible_atomic_num_list'],
                            atom.GetAtomicNum()),
                        allowable_features['possible_chirality_list'].index(
                            atom.GetChiralTag()),
                        safe_index(allowable_features['possible_degree_list'],
                                   atom.GetTotalDegree()),
                        safe_index(
                            allowable_features['possible_formal_charge_list'],
                            atom.GetFormalCharge()),
                        safe_index(allowable_features['possible_numH_list'],
                                   atom.GetTotalNumHs()),
                        safe_index(
                            allowable_features[
                                'possible_number_radical_e_list'],
                            atom.GetNumRadicalElectrons()),
                        safe_index(
                            allowable_features['possible_hybridization_list'],
                            atom.GetHybridization()),
                        allowable_features['possible_is_aromatic_list'].index(
                            atom.GetIsAromatic()),
                        allowable_features['possible_is_in_ring_list'].index(
                            atom.IsInRing()),
                    ]
                    atom_features_list.append(atom_feature)
                x = torch.tensor(np.array(atom_features_list),
                                 dtype=torch.long)

                edges_list = []
                edge_features_list = []
                for bond in mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edge_feature = [
                        safe_index(
                            allowable_features['possible_bond_type_list'],
                            bond.GetBondType()),
                        allowable_features['possible_bond_stereo_list'].index(
                            bond.GetStereo()),
                        allowable_features['possible_is_conjugated_list'].
                        index(bond.GetIsConjugated()),
                    ]
                    edges_list.append((i, j))
                    edge_features_list.append(edge_feature)
                    edges_list.append((j, i))
                    edge_features_list.append(edge_feature)

                edge_index = torch.tensor(
                    np.array(edges_list).T,
                    dtype=torch.long,
                )
                edge_attr = torch.tensor(
                    np.array(edge_features_list),
                    dtype=torch.long,
                )

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    smiles=smiles,
                    edge_attr=edge_attr,
                    image=img,
                    caption=summary,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        self.save(data_list, self.processed_paths[0])
