from typing import Optional, Callable, List

import os
import json
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_tar


class DeepNets1M(InMemoryDataset):
    url = 'https://dl.fbaipublicfiles.com/deepnets1m/deepnets1m'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        filenames = ['train', 'search', 'eval']
        json_filenames = [f'deepnets1m_{name}_meta.json' for name in filenames]
        hdf5_filenames = [f'deepnets1m_{name}.hdf5' for name in filenames]
        return json_filenames + hdf5_filenames

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(f'{self.url}_train.hdf5', self.raw_dir)
        download_url(f'{self.url}_search.hdf5', self.raw_dir)
        download_url(f'{self.url}_eval.hdf5', self.raw_dir)
        path = download_url(f'{self.url}_meta.tar.gz', self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.remove(path)

    def process(self):
        import h5py

        raw_names = self.raw_file_names
        for json_name, h5_name in zip(raw_names[:3], raw_names[3:]):
            split = h5_name[11:-5]
            split = 'val' if split == 'eval' else split
            print(split, h5_name, json_name)

            h5_data = h5py.File(osp.join(self.raw_dir, h5_name), mode='r')

            adj = torch.from_numpy(h5_data[split]['0']['adj'][()])
            x = torch.from_numpy(h5_data[split]['0']['nodes'][()])
            print(adj.shape, adj.dtype)
            print(x.shape, x.dtype)
            print(adj[:5, :5])
            print(x[:5, :5])
            raise NotImplementedError

            # keys = list(h5_data[split].keys())
            # print(len(keys))
            # print(keys[:5])
            # print(keys[-5:])
            # print(len(set(keys)))
            # print(min(keys), max(keys))

            # with open(osp.join(self.raw_dir, json_name), 'r') as f:
            #     meta = json.load(f)['train']

            #     nets = meta['nets']
            #     primitives = meta['meta']['primitives_ext']
            #     op_names_net = meta['meta']['unique_op_names']

            # print(nets)
            # print('-------')
            # print(primitives)
            # print('--------')
            # print(op_names_net)

            # print(list(meta.keys()))

            # [split]
            # n_all = len(meta['nets'])
            # self.nets = meta[
            #     'nets'][:n_all if num_nets is None else num_nets]
            # self.primitives_ext = to_int_dict(
            #     meta['meta']['primitives_ext'])
            # self.op_names_net = to_int_dict(
            #     meta['meta']['unique_op_names'])
            # self.h5_idx = [arch] if arch is not None else None
            # self.nodes = torch.tensor([net['num_nodes'] for net in self.nets])

        raise NotImplementedError

        # self.primitives_dict = {
        #     op: i
        #     for i, op in enumerate(PRIMITIVES_DEEPNETS1M)
        # }

        raise NotImplementedError
