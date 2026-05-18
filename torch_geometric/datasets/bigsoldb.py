from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import from_smiles


class BigSolDB(InMemoryDataset):
    r"""The BigSolDB 2.1 dataset from the `"BigSolDB 2.0, dataset of
    solubility values for organic compounds in different solvents at
    various temperatures"
    <https://www.nature.com/articles/s41597-025-05559-8>`_ paper
    (Zenodo: `10.5281/zenodo.18552681
    <https://doi.org/10.5281/zenodo.18552681>`_), containing 112,465
    experimental solubility values for 1,525 organic compounds measured
    in 218 individual solvents, extracted from 1,687 peer-reviewed
    articles.

    Each data point represents a (solute, solvent, temperature) triplet
    with an experimental LogS value. Both solute and solvent are encoded
    as molecular graphs using standard atom and bond features from
    :func:`torch_geometric.utils.from_smiles`.

    Solvent information is stored alongside the solute graph as
    :obj:`x_solvent`, :obj:`edge_index_solvent`, :obj:`edge_attr_solvent`.
    Measurement temperature (in K) is stored as :obj:`temperature`.
    The regression target :obj:`y` is LogS in mol/L.

    No official train/val/test split is provided. Use
    :class:`torch.utils.data.random_split` or a scaffold-based splitter
    (e.g. from `DeepChem <https://deepchem.io>`_) depending on your
    evaluation protocol.

    Args:
        root (str): Root directory where the dataset should be saved.
        solvent (str, optional): If given, only load measurements for
            this solvent (case-insensitive, e.g. :obj:`"ethanol"`).
            If :obj:`None`, all solvents are included.
            (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes
            in a :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed
            before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that
            takes in a :obj:`torch_geometric.data.Data` object and
            returns a transformed version. The data object will be
            transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included
            in the final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 15 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes (solute)
          - #edges (solute)
          - #solvents
          - #features
          - #tasks
        * - 112,465
          - ~17.2
          - ~35.6
          - 218
          - 9
          - 1
    """

    url = ('https://zenodo.org/records/18552681/files/'
           'BigSolDBv2.1.csv?download=1')

    def __init__(
        self,
        root: str,
        solvent: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.solvent = solvent
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['BigSolDBv2.1.csv']

    @property
    def processed_file_names(self) -> str:
        suffix = f'_{self.solvent.lower()}' if self.solvent else ''
        return [f'data{suffix}.pt']

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0])

        # Columns: SMILES_Solute, Temperature_K, Solvent, SMILES_Solvent,
        # Solubility(mole_fraction), Solubility(mol/L), LogS(mol/L),
        # Compound_Name, CAS, PubChem_CID, FDA_Approved, Source

        if self.solvent is not None:
            df = df[df['Solvent'].str.lower() == self.solvent.lower()]
            df = df.reset_index(drop=True)

        data_list: List[Data] = []
        for _, row in df.iterrows():
            try:
                sol_data = from_smiles(str(row['SMILES_Solute']))
                solv_data = from_smiles(str(row['SMILES_Solvent']))
            except Exception:
                continue

            if sol_data.edge_index.numel() == 0:
                continue  # skip single-atom entries

            data = Data(
                # solute graph
                x=sol_data.x,
                edge_index=sol_data.edge_index,
                edge_attr=sol_data.edge_attr,
                # solvent graph
                x_solvent=solv_data.x,
                edge_index_solvent=solv_data.edge_index,
                edge_attr_solvent=solv_data.edge_attr,
                # measurement conditions
                temperature=torch.tensor([row['Temperature_K']],
                                         dtype=torch.float),
                solvent_name=str(row['Solvent']),
                solute_smiles=str(row['SMILES_Solute']),
                compound_name=str(row['Compound_Name']),
                fda_approved=str(row['FDA_Approved']),
                # target
                y=torch.tensor([row['LogS(mol/L)']], dtype=torch.float),
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        solvent_str = (f", solvent='{self.solvent}'"
                       if self.solvent is not None else '')
        return f'{self.__class__.__name__}({len(self)}{solvent_str})'
