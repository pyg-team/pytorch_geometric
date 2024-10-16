import os
import os.path as osp
from collections.abc import Sequence
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.data import InMemoryDataset, download_google_url
from torch_geometric.data.data import BaseData

try:
    from pandas import DataFrame, read_csv
    WITH_PANDAS = True
except ImportError:
    WITH_PANDAS = False

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class TAGDataset(InMemoryDataset):
    r"""The Text Attributed Graph datasets from the
    `"Learning on Large-scale Text-attributed Graphs via Variational Inference
    " <https://arxiv.org/abs/2210.14709>`_ paper.
    This dataset is aiming on transform `ogbn products`, `ogbn arxiv`
    into Text Attributed Graph that each node in graph is associate with a
    raw text, that dataset can be adapt to DataLoader (for LM training) and
    NeighborLoader(for GNN training). In addition, this class can be use as a
    wrapper class by convert a InMemoryDataset with Tokenizer and text into
    Text Attributed Graph.

    Args:
        root (str): Root directory where the dataset should be saved.
        dataset (InMemoryDataset): The name of the dataset
            (:obj:`"ogbn-products"`, :obj:`"ogbn-arxiv"`).
        tokenizer_name (str): The tokenizer name for language model,
            Be sure to use same tokenizer name as your `model id` of model repo
            on huggingface.co.
        text (List[str]): list of raw text associate with node, the order of
            list should be align with node list
        split_idx (Optional[Dict[str, torch.Tensor]]): Optional dictionary,
            for saving split index, it is required that if your dataset doesn't
            have get_split_idx function
        tokenize_batch_size (int): batch size of tokenizing text, the
            tokenizing process will run on cpu, default: 256
        token_on_disk (bool): save token as .pt file on disk or not,
            default: False
        text_on_disk (bool): save given text(list of str) as dataframe on disk
            or not, default: False
        force_reload (bool): default: False
    .. note::
        See `example/llm_plus_gnn/glem.py` for example usage
    """
    raw_text_id = {
        'ogbn-arxiv': '1g3OOVhRyiyKv13LY6gbp8GLITocOUr_3',
        'ogbn-products': '1I-S176-W4Bm1iPDjQv3hYwQBtxE0v8mt'
    }

    def __init__(self, root: str, dataset: InMemoryDataset,
                 tokenizer_name: str, text: Optional[List[str]] = None,
                 split_idx: Optional[Dict[str, Tensor]] = None,
                 tokenize_batch_size: int = 256, token_on_disk: bool = False,
                 text_on_disk: bool = False,
                 force_reload: bool = False) -> None:
        # list the vars you want to pass in before run download & process
        self.name = dataset.name
        self.text = text
        self.tokenizer_name = tokenizer_name
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dir_name = '_'.join(dataset.name.split('-'))
        self.root = osp.join(root, self.dir_name)
        missing_str_list = []
        if not WITH_PANDAS:
            missing_str_list.append('pandas')
        if len(missing_str_list) > 0:
            missing_str = ' '.join(missing_str_list)
            error_out = f"`pip install {missing_str}` to use this dataset."
            raise ImportError(error_out)
        if hasattr(dataset, 'get_idx_split'):
            self.split_idx = dataset.get_idx_split()
        elif split_idx is not None:
            self.split_idx = split_idx
        else:
            raise ValueError("TAGDataset need split idx for generating "
                             "is_gold mask, please pass splited index "
                             "in format of dictionaty with 'train', 'valid' "
                             "'test' index tensor to 'split_idx'")
        if text is not None and text_on_disk:
            self.save_node_text(text)
        self.text_on_disk = text_on_disk
        # init will call download and process
        super().__init__(self.root, transform=None, pre_transform=None,
                         pre_filter=None, force_reload=force_reload)
        # after processing and download
        # Dataset has to have BaseData as _data
        assert dataset._data is not None
        self._data = dataset._data  # reassign reference
        assert self._data is not None
        assert dataset._data.y is not None
        assert isinstance(self._data, BaseData)
        assert self._data.num_nodes is not None
        assert isinstance(dataset._data.num_nodes, int)
        assert isinstance(self._data.num_nodes, int)
        self._n_id = torch.arange(self._data.num_nodes)
        is_good_tensor = self.load_gold_mask()
        self._is_gold = is_good_tensor.squeeze()
        self._data['is_gold'] = is_good_tensor
        if self.text is not None and len(self.text) != self._data.num_nodes:
            raise ValueError("The number of text sequence in 'text' should be "
                             "equal to number of nodes!")
        self.token_on_disk = token_on_disk
        self.tokenize_batch_size = tokenize_batch_size
        self._token = self.tokenize_graph(self.tokenize_batch_size)
        self.__num_classes__ = dataset.num_classes

    @property
    def num_classes(self) -> int:
        return self.__num_classes__

    @property
    def raw_file_names(self) -> List[str]:
        file_names = []
        for root, _, files in os.walk(osp.join(self.root, 'raw')):
            for file in files:
                file_names.append(file)
        return file_names

    @property
    def processed_file_names(self) -> List[str]:
        return [
            'geometric_data_processed.pt', 'pre_filter.pt',
            'pre_transformed.pt'
        ]

    @property
    def token(self) -> Dict[str, Tensor]:
        if self._token is None:  # lazy load
            self._token = self.tokenize_graph()
        return self._token

    # load is_gold after init
    @property
    def is_gold(self) -> Tensor:
        if self._is_gold is None:
            print('lazy load is_gold!!')
            self._is_gold = self.load_gold_mask()
        return self._is_gold

    def get_n_id(self, node_idx: IndexType) -> Tensor:
        if self._n_id is None:
            assert self._data is not None
            assert self._data.num_nodes is not None
            assert isinstance(self._data.num_nodes, int)
            self._n_id = torch.arange(self._data.num_nodes)
        return self._n_id[node_idx]

    def load_gold_mask(self) -> Tensor:
        r"""Use original train split as gold split, generating is_gold mask
        for picking ground truth labels and pseudo labels.
        """
        train_split_idx = self.get_idx_split()['train']
        assert self._data is not None
        assert self._data.num_nodes is not None
        assert isinstance(self._data.num_nodes, int)
        is_good_tensor = torch.zeros(self._data.num_nodes,
                                     dtype=torch.bool).view(-1, 1)
        is_good_tensor[train_split_idx] = True
        return is_good_tensor

    def get_gold(self, node_idx: IndexType) -> Tensor:
        r"""Get gold mask for given node_idx.

        Args:
            node_idx (torch.tensor): a tensor contain node idx
        """
        if self._is_gold is None:
            self._is_gold = self.is_gold
        return self._is_gold[node_idx]

    def get_idx_split(self) -> Dict[str, Tensor]:
        return self.split_idx

    def download(self) -> None:
        print('downloading raw text')
        raw_text_path = download_google_url(id=self.raw_text_id[self.name],
                                            folder=f'{self.root}/raw',
                                            filename='node-text.csv.gz',
                                            log=True)
        text_df = read_csv(raw_text_path)
        self.text = list(text_df['text'])

    def process(self) -> None:
        if osp.exists(osp.join(self.root, 'raw', 'node-text.csv.gz')):
            text_df = read_csv(osp.join(self.root, 'raw', 'node-text.csv.gz'))
            self.text = list(text_df['text'])
        elif self.name in self.raw_text_id:
            self.download()
        else:
            print('The dataset is not ogbn-products nor ogbn-arxiv,'
                  'please pass in your raw text string list to `text`')
        if self.text is None:
            raise ValueError("The TAGDataset only have ogbn-products and "
                             "ogbn-arxiv raw text in default "
                             "The raw text of each node is not specified"
                             "Please pass in 'text' when convert your dataset "
                             "to Text Attribute Graph Dataset")

    def save_node_text(self, text: List[str]) -> None:
        node_text_path = osp.join(self.root, 'raw', 'node-text.csv.gz')
        if osp.exists(node_text_path):
            print(f'The raw text is existed at {node_text_path}')
        else:
            print(f'Saving raw text file at {node_text_path}')
            os.makedirs(f'{self.root}/raw', exist_ok=True)
            text_df = DataFrame(text, columns=['text'])
            text_df.to_csv(osp.join(node_text_path), compression='gzip',
                           index=False)

    def tokenize_graph(self, batch_size: int = 256) -> Dict[str, Tensor]:
        r"""Tokenizing the text associate with each node, running in cpu.

        Args:
            batch_size (Optional[int]): batch size of list of text for
                generating emebdding
        Returns:
            Dict[str, torch.Tensor]: tokenized graph
        """
        data_len = 0
        if self.text is not None:
            data_len = len(self.text)
        else:
            raise ValueError("The TAGDataset need text for tokenization")
        token_keys = ['input_ids', 'token_type_ids', 'attention_mask']
        path = os.path.join(self.processed_dir, 'token', self.tokenizer_name)
        # Check if the .pt files already exist
        token_files_exist = any(
            os.path.exists(os.path.join(path, f'{k}.pt')) for k in token_keys)

        if token_files_exist and self.token_on_disk:
            print('Found tokenized file, loading may take several minutes...')
            all_encoded_token = {
                k: torch.load(os.path.join(path, f'{k}.pt'), weights_only=True)
                for k in token_keys
                if os.path.exists(os.path.join(path, f'{k}.pt'))
            }
            return all_encoded_token

        all_encoded_token = {k: [] for k in token_keys}
        pbar = tqdm(total=data_len)

        pbar.set_description('Tokenizing Text Attributed Graph')
        for i in range(0, data_len, batch_size):
            end_index = min(data_len, i + batch_size)
            token = self.tokenizer(self.text[i:min(i + batch_size, data_len)],
                                   padding='max_length', truncation=True,
                                   max_length=512, return_tensors="pt")
            for k in token.keys():
                all_encoded_token[k].append(token[k])
            pbar.update(end_index - i)
        pbar.close()

        all_encoded_token = {
            k: torch.cat(v)
            for k, v in all_encoded_token.items() if len(v) > 0
        }
        if self.token_on_disk:
            os.makedirs(path, exist_ok=True)
            print('Saving tokens on Disk')
            for k, tensor in all_encoded_token.items():
                torch.save(tensor, os.path.join(path, f'{k}.pt'))
                print('Token saved:', os.path.join(path, f'{k}.pt'))
        os.environ["TOKENIZERS_PARALLELISM"] = 'true'  # supressing warning
        return all_encoded_token

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    class TextDataset(torch.utils.data.Dataset):
        r"""This nested dataset provides textual data for each node in
        the graph. Factory method to create TextDataset from TAGDataset.

        Args:
            tag_dataset (TAGDataset): the parent dataset
        """
        def __init__(self, tag_dataset: 'TAGDataset') -> None:
            self.tag_dataset = tag_dataset
            self.token = tag_dataset.token
            assert tag_dataset._data is not None
            self._data = tag_dataset._data

            assert tag_dataset._data.y is not None
            self.labels = tag_dataset._data.y

        def get_token(self, node_idx: IndexType) -> Dict[str, Tensor]:
            r"""This function will be called in __getitem__().

            Args:
                node_idx (IndexType): selected node idx in each batch
            Returns:
                items (Dict[str, Tensor]): input for LM
            """
            items = {k: v[node_idx] for k, v in self.token.items()}
            return items

        # for LM training
        def __getitem__(
                self, node_id: IndexType
        ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
            r"""This function will override the function in
            torch.utils.data.Dataset, and will be called when you
            iterate batch in the dataloader, make sure all following
            key value pairs are present in the return dict.

            Args:
                node_id (List[int]): list of node idx for selecting tokens,
                    labels etc. when iterating data loader for LM
            Returns:
                items (dict): input k,v pairs for Language model training and
                    inference
            """
            item: Dict[str, Union[Tensor, Dict[str, Tensor]]] = {}
            item['input'] = self.get_token(node_id)
            item['labels'] = self.labels[node_id]
            item['is_gold'] = self.tag_dataset.get_gold(node_id)
            item['n_id'] = self.tag_dataset.get_n_id(node_id)
            return item

        def __len__(self) -> int:
            assert self._data.num_nodes is not None
            return self._data.num_nodes

        def get(self, idx: int) -> BaseData:
            return self._data

        def __repr__(self) -> str:
            return f'{self.__class__.__name__}()'

    def to_text_dataset(self) -> TextDataset:
        r"""Factory Build text dataset from Text Attributed Graph Dataset
        each data point is node's associated text token.
        """
        return TAGDataset.TextDataset(self)
