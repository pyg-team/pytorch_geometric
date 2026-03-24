
import os
import torch
import os.path as osp
from tqdm import tqdm
from typing import Callable, Optional
from torch_geometric.data import InMemoryDataset, download_url

urls = {
    'charge_train': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/charge/train_data.pt',
    'charge_val': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/charge/val_data.pt',
    'charge_test': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/charge/test_data.pt',

    'energy_train': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/energy/train_data.pt',
    'energy_val': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/energy/val_data.pt',
    'energy_test': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-chem/energy/test_data.pt',

    'synth_train': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-synth/train_data.pt',
    'synth_val': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-synth/val_data.pt',
    'synth_test': 'https://huggingface.co/datasets/lucamiglior/echo-benchmark/resolve/main/echo-synth/test_data.pt',
}

backup_urls = {
    'charge_train': 'https://zenodo.org/records/19185560/files/charge-train.pt',
    'charge_val': 'https://zenodo.org/records/19185560/files/charge-val.pt',
    'charge_test': 'https://zenodo.org/records/19185560/files/charge-test.pt',

    'energy_train': 'https://zenodo.org/records/19185560/files/energy-train.pt',
    'energy_val': 'https://zenodo.org/records/19185560/files/energy-val.pt',
    'energy_test': 'https://zenodo.org/records/19185560/files/energy-test.pt',

    'synth_train': 'https://zenodo.org/records/19185560/files/synth-train.pt',
    'synth_val': 'https://zenodo.org/records/19185560/files/synth-val.pt',
    'synth_test': 'https://zenodo.org/records/19185560/files/synth-test.pt',
}



NODE_LVL_TASKS = ['sssp', 'ecc', 'charge']
GRAPH_LVL_TASKS = ['diam', 'energy']
TASKS = NODE_LVL_TASKS + GRAPH_LVL_TASKS



class ECHOBenchmark(InMemoryDataset):
    r"""The **ECHOBenchmark** dataset introduced in
    `"Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation"
    <https://openreview.net/forum?id=DgkWFPZMPp>`_.

    ECHO is a benchmark for evaluating long-range graph propagation
    capabilities of graph neural networks. It contains five tasks:

    - ``sssp`` (Single-Source Shortest Path)
    - ``diam`` (Graph Diameter)
    - ``ecc`` (Node Eccentricity)
    - ``energy`` (Molecular Energy)
    - ``charge`` (Molecular Partial Charge)

    Each task provides three predefined splits (``train``, ``val``,
    ``test``), where every split consists of a list of homogeneous
    :class:`torch_geometric.data.Data` objects.

    The dataset is hosted on HuggingFace at:
    https://huggingface.co/datasets/lucamiglior/echo-benchmark

    See the original `source code <https://github.com/Graph-ECHO-Benchmark/ECHO>` for more details on the individual datasets.

    Args:
        root (str): Root directory where the dataset should be saved.
        task (str): One of
            ``{'sssp', 'diam', 'ecc', 'energy', 'charge'}``.
        split (str): One of ``{'train', 'val', 'test'}``.
        transform (callable, optional): A function/transform that takes in a
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    Example:
        >>> from torch_geometric.datasets import ECHOBenchmark
        >>> train = ECHOBenchmark(root='data/ECHOBenchmark', task='sssp', split='train')
        >>> len(train)

        
    **Citation:**

    .. code-block:: bibtex

        @inproceedings{echobenchmark,
        title={Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation},
        author={Miglior, Luca and Tolloso, Matteo and Gravina, Alessio and Bacciu, Davide},
        booktitle={The Fourteenth International Conference on Learning Representations},
        year={2026},
        url={https://openreview.net/forum?id=DgkWFPZMPp}
        }
    """

    def __init__(self, 
                 root: str, 
                 task: str, 
                 split: str = 'train', 
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None, 
                 pre_filter: Optional[Callable] = None,
                 force_reload: bool = False, 
                 **kwargs):
        """_summary_

        Parameters
        ----------
        root : str
            _description_
        task : str
            _description_
        split : str, optional
            _description_, by default 'train'
        transform : Optional[Callable], optional
            _description_, by default None
        pre_transform : Optional[Callable], optional
            _description_, by default None
        pre_filter : Optional[Callable], optional
            _description_, by default None
        force_reload : bool, optional
            _description_, by default False
        """
        assert task in TASKS, f'{task} is not in {TASKS}'
        assert split in ['train', 'val', 'test']

        self.split = split
        self.task = task
        super().__init__(root, pre_transform=pre_transform, pre_filter=pre_filter,
                         transform=transform, force_reload=force_reload, **kwargs)
        
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def num_classes(self) -> int:
            return 1 # NOTE: all tasks are regression tasks with a single target value per node/graph
    
    @property
    def is_node_level_task(self) -> bool:
        return self.task in NODE_LVL_TASKS
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.task, self.split, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.task, self.split, 'processed')
    
    @property
    def processed_file_names(self):
        return [f"{self.task}_{self.split}.pt"] 
    
    @property
    def processed_paths(self):
        return [osp.join(self.processed_dir, file_name) for file_name in self.processed_file_names]

    @property 
    def raw_file_names(self):
        return [f'{self.split}_data.pt']

    def download(self):
        print(f'Downloading {self.task} data for split {self.split}')
        print(f'Raw dir: {self.raw_dir}')
        prefix = 'synth' if self.task in ['diam', 'ecc', 'sssp'] else self.task         
        url = urls[f'{prefix}_{self.split}']
        try:
            download_url(url, self.raw_dir)
        except:
            b_url = backup_urls[f'{prefix}_{self.split}']
            print(f'Failed to download from {url}, trying {b_url} instead')
            downloaded_file = download_url(b_url, self.raw_dir)
            # rename file to match expected raw_file_names
            expected_file = osp.join(self.raw_dir, f'{self.split}_data.pt')
            os.rename(downloaded_file, expected_file)

    def process(self):
        data_list = torch.load(self.raw_paths[0], weights_only=False)

        processed_data_list = []
        for i, data in tqdm(enumerate(data_list), total=len(data_list), desc=f'Processing {self.task} data'):
            
            if self.task == 'diam':
                data_list[i].y = data.y[:, 1][0] / 40.0 # We normalize the target wrt the max value, which by design is 40 
            elif self.task == 'ecc':    
                data_list[i].y = data.y[:, 0] / 40.0 # We normalize the target wrt the max value, which by design is 40 
            elif self.task == 'sssp':
                data_list[i].y = data.y[:, 2] / 40.0 # We normalize the target wrt the max value, which by design is 40 
            elif self.task == 'charge':
                data_list[i].y = data.y[:, 0]
            elif self.task == 'energy':
                data_list[i].y = data.y[:, 0] # This target is already normalized in the original dataset as the log10(original_graph_energy)
            
            data_list[i].y = data_list[i].y.unsqueeze(-1) # shape [num_nodes, 1] for node-level tasks, shape [1] for graph-level tasks
            if self.task == 'diam':
                data_list[i].y = data_list[i].y.unsqueeze(-1) # shape [1, 1] for graph-level tasks

            data_list[i].x = torch.tensor(data.x).float()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            if self.pre_transform is not None:
                data_list[i] = self.pre_transform(data_list[i])

            processed_data_list.append(data_list[i])
   
        data, slices = self.collate(processed_data_list)
        torch.save((data, slices), self.processed_paths[0])
