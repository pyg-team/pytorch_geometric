import os
import os.path as osp
from datetime import datetime

import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
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


def get_dataset_with_transformation(name, root, use_sparse_tensor=False,
                                    bf16=False):
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
        if isinstance(data, HeteroData):
            for node_type in data.node_types:
                data[node_type].x = data[node_type].x.to(torch.bfloat16)
        else:
            data.x = data.x.to(torch.bfloat16)

    return data, dataset.num_classes, transform


def get_dataset(name, root, use_sparse_tensor=False, bf16=False):
    data, num_classes, _ = get_dataset_with_transformation(
        name, root, use_sparse_tensor, bf16)
    return data, num_classes


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


def save_benchmark_data(csv_data, batch_size, layers, num_neighbors,
                        hidden_channels, total_time, model_name, dataset_name,
                        use_sparse_tensor):
    config = f'Batch size={batch_size}, ' \
             f'#Layers={layers}, ' \
             f'#Neighbors={num_neighbors}, ' \
             f'#Hidden features={hidden_channels}'
    csv_data['DATE'].append(datetime.now().date())
    csv_data['TIME (s)'].append(round(total_time, 2))
    csv_data['MODEL'].append(model_name)
    csv_data['DATASET'].append(dataset_name)
    csv_data['CONFIG'].append(config)
    csv_data['SPARSE'].append(use_sparse_tensor)


def write_to_csv(csv_data, training=False):
    results_path = osp.join(osp.dirname(osp.realpath(__file__)), '../results/')
    os.makedirs(results_path, exist_ok=True)

    name = 'training' if training else 'inference'
    csv_path = osp.join(results_path, f'TOTAL_{name}_benchmark.csv')
    with_header = not osp.exists(csv_path)
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, mode='a', index_label='TEST_ID', header=with_header)


@torch.no_grad()
def test(model, loader, device, hetero, progress_bar=True,
         desc="Evaluation") -> None:
    if progress_bar:
        loader = tqdm(loader, desc=desc)
    total_examples = total_correct = 0
    if hetero:
        for batch in loader:
            batch = batch.to(device)
            if len(batch.adj_t_dict) > 0:
                edge_index_dict = batch.adj_t_dict
            else:
                edge_index_dict = batch.edge_index_dict
            out = model(batch.x_dict, edge_index_dict)
            batch_size = batch['paper'].batch_size
            out = out['paper'][:batch_size]
            pred = out.argmax(dim=-1)

            total_examples += batch_size
            total_correct += int((pred == batch['paper'].y[:batch_size]).sum())
    else:
        for batch in loader:
            batch = batch.to(device)
            if hasattr(batch, 'adj_t'):
                edge_index = batch.adj_t
            else:
                edge_index = batch.edge_index
            out = model(batch.x, edge_index)
            batch_size = batch.batch_size
            out = out[:batch_size]
            pred = out.argmax(dim=-1)

            total_examples += batch_size
            total_correct += int((pred == batch.y[:batch_size]).sum())
    return total_correct / total_examples
