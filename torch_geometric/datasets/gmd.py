import os.path as osp
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import Data, InMemoryDataset, download_url

# Homologous series: family -> (formula, extra hydrogens, first n, last n),
# so that a member with n carbons is called `formula` with 2 * n + extra H.
SERIES: Dict[str, Tuple[str, int, int, int]] = {
    'Alkanes': ('C{n}H{h}', 2, 2, 30),
    'Alcohols': ('OC{n}H{h}', 2, 2, 15),
    'Aldehydes': ('OC{n}H{h}', 0, 2, 15),
    'CarboxylicAcids': ('O2C{n}H{h}', 0, 2, 15),
    'DiCarboxylicAcids': ('O4C{n}H{h}', -2, 2, 16),
    'Amines': ('C{n}H{h}N', 3, 4, 8),
    'DiAmines': ('N2C{n}H{h}', 4, 2, 11),
    'Amides': ('C{n}H{h}NO', 1, 4, 8),
    'AminoAcids': ('NC{n}O2H{h}', 1, 2, 10),
}


def _series(family: str, lo: int, hi: int) -> List[str]:
    formula, extra = SERIES[family][:2]
    return [
        f'{family}/' + formula.format(n=n, h=2 * n + extra)
        for n in range(lo, hi + 1)
    ]


# Complex polyols and polycarbonyls (training set of Fragment Fusion):
COMPLEX = [
    f'Polyols/{name}' for name in [
        'C3OC3OC4H22', 'C3OC6OCH22', 'OC3OC3OC3OCH22', 'OC3OC6OCOH22',
        'OC6OC3OCH22', 'OC6OC4OH22', 'OC7OH16', 'OC8OH18', 'OC9OH20',
        'OC10OH22', 'OC11OH24'
    ]
] + [
    f'KetoAldehydes/{name}' for name in [
        'C4OC5OC2H20', 'COC10OH20', 'COC2OC5OC2OH14', 'COC2OC6OC2OH16',
        'COC2OCOC3OC2OH10', 'COC2OCOC5OC2OH14', 'COC2OCOC7H18', 'COC8OC2OH18'
    ]
]

# Cyclic molecules that are part of the release but of no benchmark task:
CYCLO = [
    f'CycloComplexes/{name}' for name in
    ['O2C7H12', 'O4C8H12', 'O6C9H12', 'O8C10H12', 'O10C11H12', 'O12C12H12']
]

MOLECULES = [
    name for family, (_, _, lo, hi) in SERIES.items()
    for name in _series(family, lo, hi)
] + COMPLEX + CYCLO

# task -> (training molecules, out-of-distribution test molecules):
TASKS: Dict[str, Tuple[List[str], List[str]]] = {
    'fragment_chain_extension': (
        _series('Alkanes', 2, 6),
        _series('Alkanes', 7, 13),
    ),
    'fragment_fusion': (
        _series('Alcohols', 4, 10) + _series('Aldehydes', 4, 10) + COMPLEX,
        _series('CarboxylicAcids', 4, 10),
    ),
    'fragment_duplication': (
        _series('CarboxylicAcids', 5, 10),
        _series('DiCarboxylicAcids', 5, 10),
    ),
    'fragment_combination': (
        _series('DiAmines', 2, 9) + _series('DiCarboxylicAcids', 2, 9),
        _series('AminoAcids', 2, 9),
    ),
    'fragment_fission': (
        _series('CarboxylicAcids', 5, 10),
        _series('Alcohols', 5, 10) + _series('Aldehydes', 5, 10),
    ),
    'fragment_chain_extension_augmented': (
        _series('Alcohols', 2, 3) + _series('Alcohols', 9, 15) +
        _series('CarboxylicAcids', 4, 8),
        _series('Alcohols', 4, 8) + _series('CarboxylicAcids', 2, 3) +
        _series('CarboxylicAcids', 9, 15),
    ),
    'fragment_fusion_augmented': (
        _series('Alcohols', 4, 10) + _series('Aldehydes', 4, 10) + COMPLEX +
        _series('Amines', 4, 8) + _series('Amides', 4, 8),
        _series('CarboxylicAcids', 4, 10),
    ),
}


class GMD(InMemoryDataset):
    r"""The GMD (*Generalisation for Molecular Dynamics*) benchmark from the
    `"Benchmarking Compositional Generalisation for Machine Learning
    Interatomic Potentials" <https://arxiv.org/abs/2605.08988>`_ paper,
    which tests whether machine learning interatomic potentials generalise to
    unseen molecules that differ from the training molecules by a single
    compositional operation.

    GMD contains molecular dynamics trajectories of 140 organic molecules
    (H, C, N, O) sampled at 300 K, labelled with total energies (in eV) and
    forces (in eV/Angstrom) at the PBE/def2-TZVP+D3BJ level of theory.
    Each benchmark task defines a set of training molecules and a disjoint set
    of out-of-distribution (OOD) test molecules:

    .. list-table::
        :widths: 30 40 30
        :header-rows: 1

        * - Task
          - Training molecules
          - OOD test molecules
        * - :obj:`fragment_chain_extension`
          - C2-C6 alkanes
          - C7-C13 alkanes
        * - :obj:`fragment_fusion`
          - C4-C10 alcohols and aldehydes, complex polyols and polycarbonyls
          - C4-C10 carboxylic acids
        * - :obj:`fragment_duplication`
          - C5-C10 carboxylic acids
          - C5-C10 dicarboxylic acids
        * - :obj:`fragment_combination`
          - C2-C9 diamines and dicarboxylic acids
          - C2-C9 amino acids
        * - :obj:`fragment_fission`
          - C5-C10 carboxylic acids
          - C5-C10 alcohols and aldehydes
        * - :obj:`fragment_chain_extension_augmented`
          - C2-C3 and C9-C15 alcohols, C4-C8 carboxylic acids
          - C4-C8 alcohols, C2-C3 and C9-C15 carboxylic acids
        * - :obj:`fragment_fusion_augmented`
          - :obj:`fragment_fusion` training set, C4-C8 amines and amides
          - C4-C10 carboxylic acids

    Only the molecules of the requested :obj:`task` are downloaded.
    Each task comes with the pre-defined splits of the paper, selected via
    the :obj:`split` argument: frames 0-999 (:obj:`"train"`), 1000-1199
    (:obj:`"val"`) and 1400-1599 (:obj:`"test_id"`) of every training
    molecule, and frames 1400-1599 of every OOD test molecule
    (:obj:`"test_ood"`).
    Setting :obj:`task="all"` downloads every frame of all 140 molecules
    without pre-defined splits.

    Every graph holds the atomic numbers :obj:`z`, positions :obj:`pos`, the
    total :obj:`energy` and the atomic forces :obj:`force`, together with the
    trajectory :obj:`frame` index and the :obj:`molecule` and :obj:`family`
    names, *e.g.*, for grouping frames by molecule.
    :meth:`atomref` returns the energies of the isolated atoms at the same
    level of theory, *e.g.*, for predicting atomisation energies or for
    :class:`~torch_geometric.nn.models.SchNet`.

    .. note::

        Data objects contain no edge indices as these are most commonly
        constructed via the :obj:`torch_geometric.transforms.RadiusGraph`
        transform, with its cut-off being a hyperparameter.

    Args:
        root (str): Root directory where the dataset should be saved.
            Downloaded molecules are shared by all tasks in the same
            :obj:`root`.
        task (str): The benchmark task to load (see the table above), or
            :obj:`"all"` for the whole dataset.
        split (str, optional): If :obj:`"train"`, :obj:`"val"`,
            :obj:`"test_id"` or :obj:`"test_ood"`, loads the corresponding
            split of :obj:`task`.
            Must be :obj:`None` for :obj:`task="all"`.
            (default: :obj:`None`, *i.e.*, :obj:`"train"` for a task)
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
        :widths: 30 10 10 10 10 10 10
        :header-rows: 1

        * - Task
          - #graphs (train)
          - #graphs (val)
          - #graphs (test_id)
          - #graphs (test_ood)
          - #features
          - #tasks
        * - fragment_chain_extension
          - 5,000
          - 1,000
          - 1,000
          - 1,400
          - 1
          - 2
        * - fragment_fusion
          - 33,000
          - 6,600
          - 6,600
          - 1,400
          - 1
          - 2
        * - fragment_duplication
          - 6,000
          - 1,200
          - 1,200
          - 1,200
          - 1
          - 2
        * - fragment_combination
          - 16,000
          - 3,200
          - 3,200
          - 1,600
          - 1
          - 2
        * - fragment_fission
          - 6,000
          - 1,200
          - 1,200
          - 2,400
          - 1
          - 2
        * - fragment_chain_extension_augmented
          - 14,000
          - 2,800
          - 2,800
          - 2,800
          - 1
          - 2
        * - fragment_fusion_augmented
          - 43,000
          - 8,600
          - 8,600
          - 1,400
          - 1
          - 2
        * - all
          - 281,984
          - \-
          - \-
          - \-
          - 1
          - 2
    """

    url = 'https://huggingface.co/datasets/AmirMasoud/GMD/resolve/main/npz'

    tasks = TASKS
    molecules = MOLECULES

    # Isolated-atom energies (in eV) at the level of theory of the labels:
    atomrefs = {
        1: -13.595298,
        6: -1028.481145,
        7: -1483.895483,
        8: -2041.117735,
    }

    # split -> [first frame, last frame + 1) of each molecule's trajectory:
    splits = {
        'train': (0, 1000),
        'val': (1000, 1200),
        'test_id': (1400, 1600),
        'test_ood': (1400, 1600),
    }

    def __init__(
        self,
        root: str,
        task: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if task != 'all' and task not in self.tasks:
            raise ValueError(f"Unknown task '{task}' (choose 'all' or one of "
                             f"{list(self.tasks)})")
        if task == 'all' and split is not None:
            raise ValueError(f"The 'all' task does not provide pre-defined "
                             f"splits but the 'split' argument is set to "
                             f"'{split}'")
        if task != 'all':
            split = split or 'train'
            if split not in self.splits:
                raise ValueError(f"Unknown split '{split}' (choose one of "
                                 f"{list(self.splits)})")

        self.task = task
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        idx = 0 if split is None else list(self.splits).index(split)
        self.load(self.processed_paths[idx])

    def atomref(self) -> Tensor:
        r"""Returns the isolated-atom energies (in eV) indexed by atomic
        number, as a tensor of shape :obj:`[100, 1]`.
        """
        out = torch.zeros(100)
        out[list(self.atomrefs)] = torch.tensor(list(self.atomrefs.values()))
        return out.view(-1, 1)

    @property
    def task_molecules(self) -> List[str]:
        if self.task == 'all':
            return self.molecules
        train, ood = self.tasks[self.task]
        return train + ood

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.task, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{name}.npz' for name in self.task_molecules]

    @property
    def processed_file_names(self) -> List[str]:
        if self.task == 'all':
            return ['data.pt']
        return [f'{split}.pt' for split in self.splits]

    def download(self) -> None:
        for name in self.raw_file_names:  # Already downloaded files are kept:
            download_url(f'{self.url}/{name}',
                         osp.join(self.raw_dir, osp.dirname(name)))

    def process(self) -> None:
        train = set() if self.task == 'all' else set(self.tasks[self.task][0])
        names = [osp.splitext(f)[0] for f in self.processed_file_names]
        data_lists: Dict[str, List[Data]] = {name: [] for name in names}

        for name, raw_path in zip(self.task_molecules, self.raw_paths):
            raw_data = np.load(raw_path)
            z = torch.from_numpy(raw_data['z']).long()
            pos = torch.from_numpy(raw_data['pos']).float()
            energy = torch.from_numpy(raw_data['energy']).float()
            force = torch.from_numpy(raw_data['forces']).float()
            frame = torch.from_numpy(raw_data['frame']).long()
            family, molecule = name.split('/')

            if self.task == 'all':
                windows = {'data': (0, int(frame.max()) + 1)}
            elif name in train:
                windows = {
                    s: self.splits[s]
                    for s in ['train', 'val', 'test_id']
                }
            else:
                windows = {'test_ood': self.splits['test_ood']}

            for split, (lo, hi) in windows.items():
                mask = (frame >= lo) & (frame < hi)
                for i in mask.nonzero().view(-1).tolist():
                    data = Data(z=z, pos=pos[i], energy=energy[i],
                                force=force[i], frame=frame[i],
                                molecule=molecule, family=family)
                    if self.pre_filter is not None and not self.pre_filter(
                            data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_lists[split].append(data)

        for name, processed_path in zip(names, self.processed_paths):
            self.save(data_lists[name], processed_path)

    def __repr__(self) -> str:
        split = '' if self.split is None else f", split='{self.split}'"
        return (f"{self.__class__.__name__}({len(self)}, "
                f"task='{self.task}'{split})")
