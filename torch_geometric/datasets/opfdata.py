import json
import os
import shutil
from functools import cached_property
from typing import Callable, List, Literal, Optional, Tuple

import torch

from torch_geometric.data import Dataset, HeteroData, download_url, extract_tar


def read_json(filename: str) -> dict:
    """Reads json file."""
    # with gfile.GFile(filename, 'r') as f:
    with open(filename, 'r') as f:
        data = json.load(f)
        return data


def extract_edge_index(json_file: dict, edge_name: str) -> torch.Tensor:
    return torch.tensor([
        json_file['grid']['edges'][edge_name]['senders'],
        json_file['grid']['edges'][edge_name]['receivers'],
    ])


def extract_edge_index_rev(json_file: dict, edge_name: str) -> torch.Tensor:
    return torch.tensor([
        json_file['grid']['edges'][edge_name]['receivers'],
        json_file['grid']['edges'][edge_name]['senders'],
    ])


class OPFData(Dataset):
    """Heterogeneous OPFData dataset from arxiv...pdf.  # TODO

    OPFData is a large dataset of solved Optimal Power Flow problems,
    derived from the
    [pglib-opf](https://github.com/power-grid-lib/pglib-opf) dataset.

    The physical topology of the grid is represented by the 'bus' node type,
    and the connecting AC lines and transformers. Additionally, 'generator',
    'load', and 'shunt' nodes are connected to 'bus' nodes using a dedicated
    edge type each, for example 'generator_link'.

    Edge direction corresponds to the properties of the line, e.g. `b_fr` is
    the line charging susceptance at the `from` (source/sender) bus.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str): 'train', 'test', or 'valid' split.
        case_name (str): The name of the original pglib opf case.
        topological_perturbations (bool): Whether to use the dataset with
            added topological perturbations or not.
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes
            in an :obj:`torch_geometric.data.HeteroData` object and returns
            a transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        root: str,
        split: Literal['train', 'test', 'valid'] = 'train',
        case_name: Literal[
            'pglib_opf_case14_ieee',
            'pglib_opf_case30_ieee',
            'pglib_opf_case57_ieee',
            'pglib_opf_case118_ieee',
            'pglib_opf_case500_goc',
            'pglib_opf_case2000_goc',
            'pglib_opf_case6470_rte',
            'pglib_opf_case4661_sdet'
            'pglib_opf_case10000_goc',
            'pglib_opf_case13659_pegase',
        ] = 'pglib_opf_case14_ieee',
        topological_perturbations: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ) -> None:
        # From the constructor.
        self.split = split
        self.case_name = case_name
        self._release = 'dataset_release_1'
        if topological_perturbations:
            self._release += '_nminusone'

        # Magic values (known in advance).
        self._n_examples = 300000

        # Initialise.
        self._base_url = 'https://storage.googleapis.com/gridopt-dataset'
        disambiguated_root = os.path.join(root, self._release, self.case_name)
        super().__init__(disambiguated_root, transform, pre_transform,
                         pre_filter)

    def file_indices(self) -> Tuple[int, int]:
        if self.split == 'train':
            return 0, int(0.9 * self._n_examples)
        elif self.split == 'valid':
            return int(0.9 * self._n_examples), int(0.95 * self._n_examples)
        elif self.split == 'test':
            return int(0.95 * self._n_examples), self._n_examples

    def _group_indices(self) -> range:
        if self.split == 'train':
            return range(0, 18)
        elif self.split == 'valid':
            return range(18, 19)
        elif self.split == 'test':
            return range(19, 20)

    @cached_property
    def raw_file_names(self) -> List[str]:
        return [f'{self.case_name}_{i}.tar.gz' for i in self._group_indices()]

    @cached_property
    def processed_file_names(self) -> List[str]:
        return [f'example_{idx}.pt' for idx in range(*self.file_indices())]

    def download(self) -> None:
        # Download.
        url_path = f'{self._base_url}/{self._release}'
        for group in self._group_indices():
            url = f'{url_path}/{self.case_name}_{group}.tar.gz'
            download_url(url, self.raw_dir)

    def process(self) -> None:
        # Create a temporary untarred directory of json files.
        group_indices = self._group_indices()
        tarfiles = [
            os.path.join(self.raw_dir, f'{self.case_name}_{i}.tar.gz')
            for i in group_indices
        ]
        for tarfile in tarfiles:
            extract_tar(tarfile, self.root)

        for group_index in group_indices:
            tmp_dir = os.path.join(
                self.root,
                'gridopt-dataset-tmp',
                self._release,
                self.case_name,
                f'group_{group_index}',
            )
            for fname in os.listdir(tmp_dir):
                path = os.path.join(tmp_dir, fname)

                # Read data from `raw_path`.
                json_file = read_json(path)
                grid = json_file['grid']
                solution = json_file['solution']

                # Graph-level properties.
                data = HeteroData()
                data.x = torch.tensor(grid['context']).reshape(-1)

                # Nodes. Note that only some have a target.
                data['bus'].x = torch.tensor(grid['nodes']['bus'])
                data['bus'].y = torch.tensor(solution['nodes']['bus'])

                data['generator'].x = torch.tensor(grid['nodes']['generator'])
                data['generator'].y = torch.tensor(
                    solution['nodes']['generator'])

                data['load'].x = torch.tensor(grid['nodes']['load'])

                data['shunt'].x = torch.tensor(grid['nodes']['shunt'])

                # Edges. Only ac lines and transformers have features.
                data['bus', 'ac_line', 'bus'].x = torch.tensor(
                    grid['edges']['ac_line']['features'])
                data['bus', 'ac_line', 'bus'].y = torch.tensor(
                    solution['edges']['ac_line']['features'])
                data['bus', 'ac_line', 'bus'].edge_index = (extract_edge_index(
                    json_file, 'ac_line'))

                data['bus', 'transformer', 'bus'].x = torch.tensor(
                    grid['edges']['transformer']['features'])
                data['bus', 'transformer', 'bus'].y = torch.tensor(
                    solution['edges']['transformer']['features'])
                data['bus', 'transformer',
                     'bus'].edge_index = (extract_edge_index(
                         json_file, 'transformer'))

                data['generator', 'generator_link',
                     'bus'].edge_index = (extract_edge_index(
                         json_file, 'generator_link'))
                data['bus', 'generator_link',
                     'generator'].edge_index = (extract_edge_index_rev(
                         json_file, 'generator_link'))

                data['load', 'load_link',
                     'bus'].edge_index = (extract_edge_index(
                         json_file, 'load_link'))
                data['bus', 'load_link',
                     'load'].edge_index = (extract_edge_index_rev(
                         json_file, 'load_link'))

                data['shunt', 'shunt_link',
                     'bus'].edge_index = (extract_edge_index(
                         json_file, 'shunt_link'))
                data['bus', 'shunt_link',
                     'shunt'].edge_index = (extract_edge_index_rev(
                         json_file, 'shunt_link'))

                # Apply pre-transform if needed.
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                # Save data to `self.processed_dir`.
                stem = fname.split(".")[0]
                pt_path = os.path.join(self.processed_dir, f'{stem}.pt')
                torch.save(data, pt_path)

            # Clean up: delete the temporary untarred directory.
            shutil.rmtree(tmp_dir)

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> HeteroData:
        offset_idx = idx + self.file_indices()[0]
        fname = f'example_{offset_idx}.pt'
        return torch.load(os.path.join(self.processed_dir, fname))
