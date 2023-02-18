import os.path as osp

import torch
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, Reddit
from torch_geometric.nn import GAT, GCN, PNA, EdgeCNN, GraphSAGE
from torch_geometric.utils import index_to_mask

from .hetero_gat import HeteroGAT
from .hetero_sage import HeteroGraphSAGE

try:
    from torch.autograd.profiler import emit_itt
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def emit_itt(*args, **kwargs):
        yield


models_dict = {
    'edge_cnn': EdgeCNN,
    'gat': GAT,
    'gcn': GCN,
    'pna': PNA,
    'sage': GraphSAGE,
    'rgat': HeteroGAT,
    'rgcn': HeteroGraphSAGE,
}


def get_dataset(name, root, use_sparse_tensor=False, bf16=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), root, name)
    transform = T.ToSparseTensor(
        remove_edge_index=False) if use_sparse_tensor else None
    if name == 'ogbn-mag':
        if transform is None:
            transform = T.ToUndirected(merge=True)
        else:
            transform = T.Compose([T.ToUndirected(merge=True), transform])
        dataset = OGB_MAG(root=path, preprocess='metapath2vec',
                          transform=transform)
    elif name == 'ogbn-products':
        if transform is None:
            transform = T.RemoveDuplicatedEdges()
        else:
            transform = T.Compose([T.RemoveDuplicatedEdges(), transform])

        dataset = PygNodePropPredDataset('ogbn-products', root=path,
                                         transform=transform)

    elif name == 'Reddit':
        dataset = Reddit(root=path, transform=transform)

    data = dataset[0]

    if name == 'ogbn-products':
        split_idx = dataset.get_idx_split()
        data.train_mask = index_to_mask(split_idx['train'],
                                        size=data.num_nodes)
        data.val_mask = index_to_mask(split_idx['valid'], size=data.num_nodes)
        data.test_mask = index_to_mask(split_idx['test'], size=data.num_nodes)
        data.y = data.y.squeeze()

    if bf16:
        data.x = data.x.to(torch.bfloat16)

    return data, dataset.num_classes


def get_model(name, params, metadata=None):
    Model = models_dict.get(name, None)
    assert Model is not None, f'Model {name} not supported!'

    if name == 'rgat':
        return Model(metadata, params['hidden_channels'], params['num_layers'],
                     params['output_channels'], params['num_heads'])

    if name == 'rgcn':
        return Model(metadata, params['hidden_channels'], params['num_layers'],
                     params['output_channels'])

    if name == 'gat':
        return Model(params['inputs_channels'], params['hidden_channels'],
                     params['num_layers'], params['output_channels'],
                     heads=params['num_heads'])

    if name == 'pna':
        return Model(params['inputs_channels'], params['hidden_channels'],
                     params['num_layers'], params['output_channels'],
                     aggregators=['mean', 'min', 'max', 'std'],
                     scalers=['identity', 'amplification',
                              'attenuation'], deg=params['degree'])

    return Model(params['inputs_channels'], params['hidden_channels'],
                 params['num_layers'], params['output_channels'])


def get_split_masks(data, dataset_name):
    if dataset_name == 'ogbn-mag':
        train_mask = ('paper', data['paper'].train_mask)
        test_mask = ('paper', data['paper'].test_mask)
        val_mask = ('paper', data['paper'].val_mask)
    else:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    return train_mask, val_mask, test_mask
