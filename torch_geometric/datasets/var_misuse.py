import json
import os
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, Dataset, download_url, extract_gz


class VarMisuse(Dataset):
    r"""The VarMisuse dataset from the `"Learning to Represent Programs
    with Graphs." <https://arxiv.org/abs/1711.00740>`_ paper, consisting of
    code from files from different open source projects represented as
    graphs for learning to identify misuse of variable and naming of
    variables.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"valid"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
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
    url = ('https://github.com/amitamb-ml/'
           'varmisuse-data/raw/main/graphs-{}/{}.jsonl.gz')

    windows_count = {'train': 20, 'valid': 4, 'test': 4}

    chunks_per_window = 10

    graph_edge_types = [
        "Child", "NextToken", "LastUse", "LastWrite", "LastLexicalUse",
        "ComputedFrom", "GuardedByNegation", "GuardedBy", "FormalArgName",
        "ReturnsTo"
    ]

    def __init__(self, root: str, split: str = 'train',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{chunk_name}.jsonl' for chunk_name in self.chunk_names]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{chunk_name}.pt' for chunk_name in self.chunk_names]

    @property
    def chunk_names(self) -> List[str]:
        chunk_names = []
        total_windows = self.windows_count[self.split]
        for window in range(total_windows):
            for chunk in range(self.chunks_per_window):
                chunk_names.append(f'chunk_{window}-{chunk}')
        return chunk_names

    def download(self):
        for chunk_name in self.chunk_names:
            full_url = self.url.format(self.split, chunk_name)
            path = download_url(full_url, self.raw_dir)
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        for idx, raw_path in enumerate(self.raw_paths):
            self.process_path(self.raw_paths[idx], self.processed_paths[idx])

    def process_path(self, raw_path, processed_path):
        with open(raw_path, 'r') as f:
            graphs_json = f.read().split('\n')[:-1]
            data_list = []
            for graph_json in graphs_json:
                raw_sample = json.loads(graph_json)

                slot_node_id = raw_sample['SlotDummyNode']
                distractor_candidate_ids = []
                for candidate in raw_sample['SymbolCandidates']:
                    if candidate['IsCorrect']:
                        correct_candidate_id = candidate['SymbolDummyNode']
                    else:
                        distractor_candidate_ids.append(
                            candidate['SymbolDummyNode'])
                assert correct_candidate_id is not None
                # 0th candidate_id will always be correct
                # candidate_node_ids = [correct_candidate_id] +
                #                      distractor_candidate_ids

                edge_indexes = []
                edge_types = []

                raw_edges = raw_sample['ContextGraph']['Edges']
                for e_type, e_type_edges in raw_edges.items():
                    if len(e_type_edges) > 0:
                        e_type_idx = self.graph_edge_types[e_type] * 2
                        e_type_bkwd_idx = (e_type_idx * 2) + 1

                        fwd_edges = torch.tensor(
                            e_type_edges, dtype=torch.int32).t().contiguous()
                        fwd_edge_type = torch.tensor(
                            [e_type_idx] * fwd_edges.size(1), dtype=torch.int8)
                        bkwd_edges = torch.flip(fwd_edges, [0])
                        bkwd_edge_type = torch.tensor([e_type_bkwd_idx] *
                                                      bkwd_edges.size(1),
                                                      dtype=torch.int8)

                        edge_indexes.append(fwd_edges)
                        edge_types.append(fwd_edge_type)
                        edge_indexes.append(bkwd_edges)
                        edge_types.append(bkwd_edge_type)

                edge_index = torch.cat(edge_indexes, dim=1)
                edge_type = torch.cat(edge_types, dim=-1)

                num_nodes = len(raw_sample['ContextGraph']['NodeLabels'])
                node_labels = [""] * num_nodes
                node_labels = raw_sample['ContextGraph']['NodeLabels']
                for (node, label) in node_labels.items():
                    node_labels[int(node)] = label

                data = Data(edge_index=edge_index, edge_type=edge_type,
                            node_labels=node_labels, slot_node_id=slot_node_id,
                            correct_candidate_id=correct_candidate_id,
                            distractor_candidate_ids=distractor_candidate_ids)

                data_list.append(data)

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            torch.save(data_list, processed_path)

    def len(self):
        return len(self.processed_file_names) * 100

    def get(self, idx):
        file_idx, data_idx = divmod(idx, 100)
        window, chunk = divmod(file_idx, 10)
        data_list = torch.load(
            os.path.join(self.processed_dir, f'chunk_{window}-{chunk}.pt'))
        data = data_list[data_idx]
        return data
