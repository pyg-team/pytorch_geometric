import os.path as osp

from edgeconv import EdgeConvNet
from gat import GATNet
from gcn import GCN
from graphsage import SAGE_HETERO
from ogb.nodeproppred import PygNodePropPredDataset
from pna import PNANet
from rgat import GAT_HETERO

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG, Reddit

models_dict = {
    'edge_conv': EdgeConvNet,
    'gat': GATNet,
    'gcn': GCN,
    'pna_conv': PNANet,
    'rgat': GAT_HETERO,
    'rgcn': SAGE_HETERO,
}


def get_dataset(name, root):
    path = osp.dirname(osp.realpath(__file__))

    if name == 'ogbn-mag':
        transform = T.ToUndirected(merge=True)
        dataset = OGB_MAG(root=osp.join(path, root, 'mag'),
                          preprocess='metapath2vec', transform=transform)
    elif name == 'ogbn-products':
        dataset = PygNodePropPredDataset('ogbn-products',
                                         root=osp.join(path, root, 'products'))
    elif name == 'reddit':
        dataset = Reddit(root=osp.join(path, root, 'reddit'))

    return dataset


def get_model(name, params, metadata=None):
    try:
        model_type = models_dict[name]
    except KeyError:
        print(f'Model {name} not supported!')

    if name in ['rgat', 'rgcn']:
        if name == 'rgat':
            model = model_type(params['hidden_channels'],
                               params['output_channels'], params['num_layers'],
                               params['num_heads'])
        elif name == 'rgcn':
            model = model_type(params['hidden_channels'],
                               params['output_channels'], params['num_layers'])
        model.create_hetero(metadata)

    elif name == 'gat':
        model = model_type(params['inputs_channels'],
                           params['hidden_channels'],
                           params['output_channels'], params['num_heads'],
                           params['num_layers'])

    elif name == 'pna_conv':
        model = model_type(params['inputs_channels'],
                           params['hidden_channels'],
                           params['output_channels'], params['num_layers'],
                           params['degree'])

    else:
        model = model_type(params['inputs_channels'],
                           params['hidden_channels'],
                           params['output_channels'], params['num_layers'])
    return model
