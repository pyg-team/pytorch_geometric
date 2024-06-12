"""Implementation of MPLP from the `Pure Message Passing Can Estimate Common Neighbor
 for Link Prediction <https://arxiv.org/abs/2309.00976>`_ paper.
Based on the code https://github.com/Barcavin/efficient-node-labelling
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn.models import MLP, MPLP, MPLP_GCN
from torch_geometric.transforms import ToSparseTensor, ToUndirected
from torch_geometric.utils import degree

########################
######## Utils #########
########################


def get_dataset(root, name: str):
    dataset = PygLinkPropPredDataset(name=name, root=root)
    data = dataset[0]
    """
        SparseTensor's value is NxNx1 for collab. due to edge_weight is |E|x1
        NeuralNeighborCompletion just set edge_weight=None
        ELPH use edge_weight
    """

    split_edge = dataset.get_edge_split()
    if 'edge_weight' in data:
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    if 'edge' in split_edge['train']:
        key = 'edge'
    else:
        key = 'source_node'
    print("-" * 20)
    print(f"train: {split_edge['train'][key].shape[0]}")
    print(f"{split_edge['train'][key]}")
    print(f"valid: {split_edge['valid'][key].shape[0]}")
    print(f"test: {split_edge['test'][key].shape[0]}")
    print(f"max_degree:{degree(data.edge_index[0], data.num_nodes).max()}")
    data = ToUndirected()(data)
    data = ToSparseTensor(remove_edge_index=False)(data)
    data.full_adj_t = data.adj_t
    # make node feature as float
    if data.x is not None:
        data.x = data.x.to(torch.float)
    if name != 'ogbl-ddi':
        del data.edge_index
    return data, split_edge


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def data_summary(name: str, data: Data):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    n_degree = data.adj_t.sum(dim=1).to(torch.float)
    avg_degree = n_degree.mean().item()
    degree_std = n_degree.std().item()
    max_degree = n_degree.max().long().item()
    density = num_edges / (num_nodes * (num_nodes - 1) / 2)
    if data.x is not None:
        attr_dim = data.x.shape[1]
    else:
        attr_dim = '-'  # no attribute

    print("-" * 30 + 'Dataset and Features' + "-" * 60)
    print("{:<10}|{:<10}|{:<10}|{:<15}|{:<15}|{:<15}|{:<10}|{:<15}"\
        .format('Dataset','#Nodes','#Edges','Avg. node deg.','Std. node deg.','Max. node deg.', 'Density','Attr. Dimension'))
    print("-" * 110)
    print("{:<10}|{:<10}|{:<10}|{:<15.2f}|{:<15.2f}|{:<15}|{:<9.4f}%|{:<15}"\
        .format(name, num_nodes, num_edges, avg_degree, degree_std, max_degree, density*100, attr_dim))
    print("-" * 110)


def initial_embedding(data, hidden_channels, device):
    embedding = torch.nn.Embedding(data.num_nodes, hidden_channels).to(device)
    torch.nn.init.xavier_uniform_(embedding.weight)
    return embedding


def create_input(data):
    if hasattr(data, 'emb') and data.emb is not None:
        x = data.emb.weight
    else:
        x = data.x
    return x


########################
##### Train utils ######
########################


def __elem2spm(element: torch.Tensor, sizes: List[int],
               val: torch.Tensor = None) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    if val is None:
        sp_tensor = SparseTensor(row=row, col=col,
                                 sparse_sizes=sizes).to_device(
                                     element.device).fill_value_(1.0)
    else:
        sp_tensor = SparseTensor(row=row, col=col, value=val,
                                 sparse_sizes=sizes).to_device(element.device)
    return sp_tensor


def __spm2elem(spm: SparseTensor) -> torch.Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    val = spm.storage.value()
    return elem, val


def __spmdiff(adj1: SparseTensor, adj2: SparseTensor,
              keep_val=False) -> Tuple[SparseTensor, SparseTensor]:
    """Return elements in adj1 but not in adj2 and in adj2 but not adj1
    """
    element1, val1 = __spm2elem(adj1)
    element2, val2 = __spm2elem(adj2)

    if element1.shape[0] == 0:
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]

    if keep_val and val1 is not None:
        retval1 = val1[maskelem1]
        return __elem2spm(retelem1, adj1.sizes(), retval1)
    else:
        return __elem2spm(retelem1, adj1.sizes())


def get_train_test(args):
    if args.dataset == "ogbl-citation2":
        evaluator = Evaluator(name='ogbl-citation2')
        return train_mrr, test_mrr, evaluator
    else:
        evaluator = Evaluator(name='ogbl-ddi')
        return train_hits, test_hits, evaluator


def train_hits(encoder, predictor, data, split_edge, optimizer, batch_size,
               mask_target, num_neg):
    encoder.train()
    predictor.train()
    device = data.adj_t.device()
    criterion = BCEWithLogitsLoss(reduction='mean')
    pos_train_edge = split_edge['train']['edge'].to(device)

    optimizer.zero_grad()
    total_loss = total_examples = 0
    num_pos_max = max(data.adj_t.nnz() // 2, pos_train_edge.size(0))
    neg_edge_epoch = torch.randint(0, data.adj_t.size(0),
                                   size=(2, num_pos_max * num_neg),
                                   dtype=torch.long, device=device)
    for perm in tqdm(
            DataLoader(range(pos_train_edge.size(0)), batch_size,
                       shuffle=True), desc='Train'):
        edge = pos_train_edge[perm].t()
        if mask_target:
            adj_t = data.adj_t
            undirected_edges = torch.cat((edge, edge.flip(0)), dim=-1)
            target_adj = SparseTensor.from_edge_index(
                undirected_edges, sparse_sizes=adj_t.sizes())
            adj_t = __spmdiff(adj_t, target_adj, keep_val=True)
        else:
            adj_t = data.adj_t

        h = encoder(data.x, adj_t)

        neg_edge = neg_edge_epoch[:, perm]
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat(
            (torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])),
            dim=0).to(device)
        out = predictor(h, adj_t, train_edges).squeeze()
        loss = criterion(out, train_label)

        loss.backward()

        if data.x is not None:
            torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_examples += train_label.size(0)
        total_loss += loss.item() * train_label.size(0)

    return total_loss / total_examples


@torch.no_grad()
def test_hits(encoder, predictor, data, split_edge, evaluator, batch_size,
              fast_inference):
    encoder.eval()
    predictor.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    h = encoder(data.x, adj_t)

    def test_split(split, cache_mode=None):
        pos_test_edge = split_edge[split]['edge'].to(device)
        neg_test_edge = split_edge[split]['edge_neg'].to(device)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()
            out = predictor(h, adj_t, edge, cache_mode=cache_mode)
            pos_test_preds += [out.squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            neg_test_preds += [
                predictor(h, adj_t, edge,
                          cache_mode=cache_mode).squeeze().cpu()
            ]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)
        return pos_test_pred, neg_test_pred

    pos_valid_pred, neg_valid_pred = test_split('valid')

    start_time = time.perf_counter()
    if fast_inference:
        # caching
        predictor(h, adj_t, None, cache_mode='build')
        cache_mode = 'use'
    else:
        cache_mode = None

    pos_test_pred, neg_test_pred = test_split('test', cache_mode=cache_mode)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Inference for one epoch Took {total_time:.4f} seconds')
    if fast_inference:
        # delete cache
        predictor(h, adj_t, None, cache_mode='delete')

    results = {}
    K = 100
    evaluator.K = K
    valid_hits = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })[f'hits@{K}']
    test_hits = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })[f'hits@{K}']

    results[f'Hits@{K}'] = (valid_hits, test_hits)

    valid_result = torch.cat((torch.ones(
        pos_valid_pred.size()), torch.zeros(neg_valid_pred.size())), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    test_result = torch.cat(
        (torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())),
        dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    results['AUC'] = (roc_auc_score(valid_result.cpu().numpy(),
                                    valid_pred.cpu().numpy()),
                      roc_auc_score(test_result.cpu().numpy(),
                                    test_pred.cpu().numpy()))

    return results


def make_symmetric(sparse_tensor, reduce='sum'):
    # Extract COO format
    indices = sparse_tensor.coalesce().indices()
    row, col = indices[0], indices[1]
    value = sparse_tensor.coalesce().values()

    # Concatenate the original and transposed entries
    all_row = torch.cat([row, col])
    all_col = torch.cat([col, row])
    all_value = torch.cat([value, value])

    # Create a new COO matrix with these entries
    new_indices = torch.stack([all_row, all_col])
    new_value = all_value

    # Remove duplicates by summing the values for symmetric entries
    unique_indices, inverse_indices = torch.unique(new_indices, dim=1,
                                                   return_inverse=True)
    unique_value = torch.zeros(unique_indices.size(1),
                               device=value.device).scatter_reduce_(
                                   0, inverse_indices, new_value,
                                   reduce="amax")

    # Create the symmetric sparse tensor
    symmetric_tensor = torch.sparse_coo_tensor(unique_indices, unique_value,
                                               sparse_tensor.size())

    return symmetric_tensor


def train_mrr(encoder, predictor, data, split_edge, optimizer, batch_size,
              mask_target, num_neg):
    encoder.train()
    predictor.train()
    device = data.adj_t.device()
    criterion = BCEWithLogitsLoss(reduction='mean')
    source_edge = split_edge['train']['source_node'].to(device)
    target_edge = split_edge['train']['target_node'].to(device)
    adjmask = torch.ones_like(source_edge, dtype=torch.bool)

    optimizer.zero_grad()
    total_loss = total_examples = 0
    for perm in tqdm(
            DataLoader(range(source_edge.size(0)), batch_size, shuffle=True),
            desc='Train'):
        if mask_target:
            adjmask[perm] = 0
            tei = torch.stack(
                (source_edge[adjmask], target_edge[adjmask]),
                dim=0)  # TODO: check if both direction is removed

            adj_t = SparseTensor.from_edge_index(
                tei, sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                    source_edge.device, non_blocking=True)
            adjmask[perm] = 1

            adj_t = adj_t.to_symmetric()

            #adj_t = torch.sparse_coo_tensor(tei, torch.ones((tei.size(1)), device=tei.device), (data.num_nodes, data.num_nodes))
            #adj_t = adj_t.coalesce()
            #adj_t = make_symmetric(adj_t).coalesce()
        else:
            adj_t = data.adj_t

        h = encoder(data.x, adj_t)
        dst_neg = torch.randint(0, data.num_nodes,
                                perm.size() * num_neg, dtype=torch.long,
                                device=device)

        edge = torch.stack((source_edge[perm], target_edge[perm]), dim=0)
        neg_edge = torch.stack((source_edge[perm].repeat(num_neg), dst_neg),
                               dim=0)
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat(
            (torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])),
            dim=0).to(device)
        out = predictor(h, adj_t, train_edges).squeeze()
        loss = criterion(out, train_label)

        loss.backward()

        if data.x is not None:
            torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_examples += train_label.size(0)
        total_loss += loss.item() * train_label.size(0)

    return total_loss / total_examples


@torch.no_grad()
def test_mrr(encoder, predictor, data, split_edge, evaluator, batch_size,
             fast_inference):
    encoder.eval()
    predictor.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    h = encoder(data.x, adj_t)

    def test_split(split, cache_mode=None):
        source = split_edge[split]['source_node'].to(device)
        target = split_edge[split]['target_node'].to(device)
        target_neg = split_edge[split]['target_node_neg'].to(device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [
                predictor(h, adj_t, torch.stack((src, dst)),
                          cache_mode=cache_mode).squeeze().cpu()
            ]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [
                predictor(h, adj_t, torch.stack((src, dst_neg)),
                          cache_mode=cache_mode).squeeze().cpu()
            ]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return pos_pred, neg_pred

    pos_valid_pred, neg_valid_pred = test_split('valid')

    start_time = time.perf_counter()
    if fast_inference:
        # caching
        predictor(h, adj_t, None, cache_mode='build')
        cache_mode = 'use'
    else:
        cache_mode = None

    pos_test_pred, neg_test_pred = test_split('test', cache_mode=cache_mode)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Inference for one epoch Took {total_time:.4f} seconds')
    if fast_inference:
        # delete cache
        predictor(h, adj_t, None, cache_mode='delete')

    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })['mrr_list'].mean().item()
    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results = {
        "MRR": (valid_mrr, test_mrr),
    }

    return results


########################
######## Main  #########
########################


def main():
    parser = argparse.ArgumentParser(description='MPLP')
    # dataset setting
    parser.add_argument('--dataset', type=str, default='collab')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--dataset_dir', type=str, default='./data')

    # MPLP settings
    parser.add_argument('--signature_dim', type=int, default=1024,
                        help="the node signature dimension `F` in MPLP")
    parser.add_argument(
        '--minimum_degree_onehot', type=int, default=-1, help=
        'the minimum degree of hubs with onehot encoding to reduce variance')
    parser.add_argument(
        '--use_degree', type=str, default='none',
        choices=["none", "mlp", "AA", "RA"],
        help="rescale vector norm to facilitate weighted count")
    parser.add_argument('--signature_sampling', type=str, default='torchhd',
                        help='whether to use torchhd to randomize vectors',
                        choices=["torchhd", "gaussian", "onehot"])
    parser.add_argument(
        '--fast_inference', type=str2bool, default='False',
        help='whether to enable a faster inference by caching the node vectors'
    )
    parser.add_argument(
        '--mask_target', type=str2bool, default='True',
        help='whether to mask the target edges to remove the shortcut')

    # model setting
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--xdp', type=float, default=0.2)
    parser.add_argument('--feat_dropout', type=float, default=0.5)
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_feature', type=str2bool, default='True',
                        help='whether to use node features as input')
    parser.add_argument('--feature_combine', type=str, default='hadamard',
                        choices=['hadamard', 'plus_minus'],
                        help='how to represent a link with two nodes features')
    parser.add_argument('--jk', type=str2bool, default='True',
                        help='whether to use Jumping Knowledge')
    parser.add_argument('--batchnorm_affine', type=str2bool, default='True',
                        help='whether to use Affine in BatchNorm')
    parser.add_argument('--use_embedding', type=str2bool, default='False',
                        help='whether to train node embedding')

    # training setting
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--test_batch_size', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--patience', type=int, default=100,
                        help='number of patience steps for early stopping')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--metric', type=str, default='Hits@100',
                        help='main evaluation metric')

    # misc
    parser.add_argument('--data_split_only', type=str2bool, default='False')
    parser.add_argument('--print_summary', type=str, default='')

    args = parser.parse_args()
    # start time
    start_time = time.time()
    set_random_seeds(234)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data, split_edge = get_dataset(args.dataset_dir, args.dataset)
    if args.dataset == "ogbl-citation2":
        args.metric = "MRR"
    if data.x is None:
        args.use_feature = False

    if args.print_summary:
        data_summary(args.dataset, data)
        exit(0)
    else:
        print(args)

    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    print('Command line input: ' + cmd_input + ' is saved.')

    train, test, evaluator = get_train_test(args)

    val_max = 0.0
    for run in range(args.runs):
        if args.minimum_degree_onehot > 0:
            d_v = data.adj_t.sum(dim=0).to_dense()
            nodes_to_one_hot = d_v >= args.minimum_degree_onehot
            one_hot_dim = nodes_to_one_hot.sum()
            print(f"number of nodes to onehot: {int(one_hot_dim)}")
        data = data.to(device)
        if args.use_embedding:
            emb = initial_embedding(data, args.hidden_channels, device)
        else:
            emb = None
        if 'gcn' in args.encoder:
            encoder = MPLP_GCN(data.num_features, args.hidden_channels,
                               args.hidden_channels, args.num_layers,
                               args.feat_dropout, args.xdp, args.use_feature,
                               args.jk, args.encoder, emb).to(device)
        elif args.encoder == 'mlp':
            encoder = MLP(num_layers=args.num_layers,
                          in_channels=data.num_features,
                          hidden_channels=args.hidden_channels,
                          out_channels=args.hidden_channels,
                          dropout=args.feat_dropout, act=None).to(device)

        predictor_in_dim = args.hidden_channels * int(args.use_feature
                                                      or args.use_embedding)

        predictor = MPLP(predictor_in_dim, args.hidden_channels,
                         args.num_layers, args.feat_dropout,
                         args.label_dropout, args.num_hops,
                         signature_sampling=args.signature_sampling,
                         use_degree=args.use_degree,
                         signature_dim=args.signature_dim,
                         minimum_degree_onehot=args.minimum_degree_onehot,
                         batchnorm_affine=args.batchnorm_affine,
                         feature_combine=args.feature_combine)

        predictor = predictor.to(device)

        encoder.reset_parameters()
        predictor.reset_parameters()
        parameters = list(encoder.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr,
                                     weight_decay=args.weight_decay)
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f'Total number of parameters is {total_params}')

        cnt_wait = 0
        best_val = 0.0

        for epoch in range(1, 1 + args.epochs):
            loss = train(encoder, predictor, data, split_edge, optimizer,
                         args.batch_size, args.mask_target,
                         num_neg=args.num_neg)

            results = test(encoder, predictor, data, split_edge, evaluator,
                           args.test_batch_size, args.fast_inference)

            if results[args.metric][0] >= best_val:
                best_val = results[args.metric][0]
                cnt_wait = 0
            else:
                cnt_wait += 1

            if epoch % args.log_steps == 0:
                for key, result in results.items():
                    valid_hits, test_hits = result
                    to_print = (f'Run: {run + 1:02d}, ' +
                                f'Epoch: {epoch:02d}, ' +
                                f'Loss: {loss:.4f}, ' +
                                f'Valid: {100 * valid_hits:.2f}%, ' +
                                f'Test: {100 * test_hits:.2f}%')
                    print(key)
                    print(to_print)
                print('---')

            if cnt_wait >= args.patience:
                break
        print(f'Highest Valid: {best_val}')
    # end time
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f}s")


if __name__ == "__main__":
    main()
