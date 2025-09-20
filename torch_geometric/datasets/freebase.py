from typing import Callable, Dict, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class FB15k_237(InMemoryDataset):
    r"""The FB15K237 dataset from the `"Translating Embeddings for Modeling
    Multi-Relational Data"
    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling
    -multi-relational-data>`_ paper,
    containing 14,541 entities, 237 relations and 310,116 fact triples.

    .. note::

        The original :class:`FB15k` dataset suffers from major test leakage
        through inverse relations, where a large number of test triples could
        be obtained by inverting triples in the training set.
        In order to create a dataset without this characteristic, the
        :class:`~torch_geometric.datasets.FB15k_237` describes a subset of
        :class:`FB15k` where inverse relations are removed.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    url = ('https://raw.githubusercontent.com/villmow/'
           'datasets_knowledge_embedding/master/FB15k-237')

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)

        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid 'split' argument (got {split})")

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.txt', 'valid.txt', 'test.txt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self) -> None:
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process(self) -> None:
        data_list: List[Data] = []
        node_dict: Dict[str, int] = {}
        rel_dict: Dict[str, int] = {}

        for path in self.raw_paths:
            with open(path) as f:
                lines = [x.split('\t') for x in f.read().split('\n')[:-1]]

            edge_index = torch.empty((2, len(lines)), dtype=torch.long)
            edge_type = torch.empty(len(lines), dtype=torch.long)
            for i, (src, rel, dst) in enumerate(lines):
                if src not in node_dict:
                    node_dict[src] = len(node_dict)
                if dst not in node_dict:
                    node_dict[dst] = len(node_dict)
                if rel not in rel_dict:
                    rel_dict[rel] = len(rel_dict)

                edge_index[0, i] = node_dict[src]
                edge_index[1, i] = node_dict[dst]
                edge_type[i] = rel_dict[rel]

            data = Data(edge_index=edge_index, edge_type=edge_type)
            data_list.append(data)

        for data, path in zip(data_list, self.processed_paths):
            data.num_nodes = len(node_dict)
            self.save([data], path)
