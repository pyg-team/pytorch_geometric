import argparse
import datetime
import os
from copy import copy
from resource import *

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Parameter, ParameterDict
from torch_sparse import SparseTensor
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.datasets import KGBen_banchmark_dataset
from torch_geometric.loader import (
    BiasedRandomWalkSampler,
    GraphSAINTRandomWalkSampler,
)
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.hetero import group_hetero_graph

subject_node = None
subject_node_code = -1


def eval_acc(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        correct = correct.tolist()
        acc_list.append(float(np.sum(correct)) / len(correct))

    return {'acc': sum(acc_list) / len(acc_list)}


def print_memory_usage():
    print('used virtual memory GB:',
          psutil.virtual_memory().used / (1024.0**3), " percent",
          psutil.virtual_memory().percent)


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types):
        super(RGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])

        self.root_lins = ModuleList([
            Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, node_type):
        out = x.new_zeros(x.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = node_type == i
            out[mask] += self.root_lins[i](x[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(RGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}':
            Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx):
        # Create global node feature matrix.
        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, x_dict, edge_index, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)

    def inference(self, x_dict, edge_index_dict, key2int):
        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)

        # paper_count=len(x_dict[2])
        # paper_count = len(x_dict[3])
        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb
            # print(key," size=",x_dict[int(key)].size())

        # print(key2int)
        # print("x_dict keys=",x_dict.keys())

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)
            # print(key,adj_t_dict[key].size)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                # print("keys=",keys)
                # print("adj_t=",adj_t)
                # print("key2int[src_key]=",key2int[src_key])
                # print("x_dict[key2int[src_key]]=",x_dict[key2int[src_key]].size())
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                # print("out size=",out.size())
                # print("tmp size=",conv.rel_lins[key2int[keys]](tmp).size())
                ################## fill missed rows hsh############################
                tmp = conv.rel_lins[key2int[keys]](tmp).resize_(
                    [out.size()[0], out.size()[1]])
                out.add_(tmp)

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict


def graphSaint():
    def train(epoch):
        model.train()
        # tqdm.monitor_interval = 0
        pbar = tqdm(total=args.num_steps * args.batch_size)
        pbar.set_description(f'Epoch {epoch:02d}')
        total_loss = total_examples = 0
        data_suff_lst = []
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            subject_nodes_count = int(
                (data.node_type == subject_node_code).float().sum().item())
            all_nodes_count = data.num_nodes
            data_suff = "batch_idx=" + str(
                idx) + ",subject_nodes_count=" + str(
                    subject_nodes_count) + ",nodes_count=" + str(
                        all_nodes_count) + ",subject nodes(%)=" + str(
                            (subject_nodes_count / all_nodes_count) * 100)
            data_suff_lst.append(data_suff)
            optimizer.zero_grad()
            out = model(x_dict, data.edge_index, data.edge_attr,
                        data.node_type, data.local_node_idx)
            out = out[data.train_mask]
            y = data.y[data.train_mask].squeeze()
            loss = F.nll_loss(out, y)
            # print("loss=",loss)
            loss.backward()
            optimizer.step()
            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            pbar.update(args.batch_size)

        # pbar.refresh()  # force print final state
        pbar.close()
        print(data_suff_lst)
        # pbar.reset()
        return total_loss / total_examples

    @torch.no_grad()
    def test():
        model.eval()
        out = model.inference(x_dict, edge_index_dict, key2int)
        out = out[key2int[subject_node]]
        y_pred = out.argmax(dim=-1, keepdim=True).cpu()
        y_true = data.y_dict[subject_node]
        train_acc = eval_acc(y_true[split_idx['train'][subject_node]],
                             y_pred[split_idx['train'][subject_node]])
        valid_acc = eval_acc(y_true[split_idx['valid'][subject_node]],
                             y_pred[split_idx['valid'][subject_node]])
        test_acc = eval_acc(y_true[split_idx['test'][subject_node]],
                            y_pred[split_idx['test'][subject_node]])
        return train_acc, valid_acc, test_acc

    parser = argparse.ArgumentParser(description='KGTOSA (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--dataset_name', type=str, default='MAG42M_PV_FG')
    parser.add_argument('--n_class', type=int, default=349)
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--use_BRW_sampler', type=int, default=1)
    parser.add_argument('--split_type', type=str, default='time')
    args = parser.parse_args()
    to_remove_pedicates = []
    to_remove_subject_object = []
    include_reverse_edge = True
    print(args)
    ###################################Delete Folder if exist #############################
    # dir_path = root_path + GA_dataset_name
    # try:
    #     shutil.rmtree(dir_path)
    #     print("Folder Deleted")
    # except OSError as e:
    #     print("Error Deleting : %s : %s" % (dir_path, e.strerror))
    #####################
    dataset = KGBen_banchmark_dataset(
        root=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../data/")),
        name=args.dataset_name, numofClasses=str(args.n_class))
    print(getrusage(RUSAGE_SELF))
    start_t = datetime.datetime.now()
    data = dataset[0]
    global subject_node
    subject_node = list(data.y_dict.keys())[0]
    split_idx = dataset.get_idx_split(args.split_type)
    data.node_year_dict = None
    data.edge_reltype_dict = None
    to_remove_rels = []
    for keys, (row, col) in data.edge_index_dict.items():
        if (keys[2]
                in to_remove_subject_object) or (keys[0]
                                                 in to_remove_subject_object):
            to_remove_rels.append(keys)
    for keys, (row, col) in data.edge_index_dict.items():
        if (keys[1] in to_remove_pedicates):
            to_remove_rels.append(keys)
            to_remove_rels.append((keys[2], '_inv_' + keys[1], keys[0]))
    for elem in to_remove_rels:
        data.edge_index_dict.pop(elem, None)
        data.edge_reltype.pop(elem, None)

    for key in to_remove_subject_object:
        data.num_nodes_dict.pop(key, None)
    ##############add inverse edges ###################
    if include_reverse_edge:
        edge_index_dict = data.edge_index_dict
        key_lst = list(edge_index_dict.keys())
        for key in key_lst:
            r, c = edge_index_dict[(key[0], key[1], key[2])]
            edge_index_dict[(key[2], 'inv_' + key[1],
                             key[0])] = torch.stack([c, r])

    out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
    edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
    subject_node_code = key2int[subject_node]
    homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                     node_type=node_type, local_node_idx=local_node_idx,
                     num_nodes=node_type.size(0))
    homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
    homo_data.y[local2global[subject_node]] = data.y_dict[subject_node]
    homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
    homo_data.train_mask[local2global[subject_node][split_idx['train']
                                                    [subject_node]]] = True
    print("dataset.processed_dir", dataset.processed_dir)
    ############### Configure the Sampler ####################
    if args.use_BRW_sampler:
        train_loader = BiasedRandomWalkSampler(
            data=homo_data, batch_size=args.batch_size,
            walk_length=args.walk_length,
            target_indices=local2global[subject_node],
            num_steps=args.num_steps, sample_coverage=0, target_node_ratio=1,
            save_dir=dataset.processed_dir)
    else:
        train_loader = GraphSAINTRandomWalkSampler(
            data=homo_data, batch_size=args.batch_size,
            walk_length=args.walk_length, num_steps=args.num_steps,
            sample_coverage=0, save_dir=dataset.processed_dir)
    #######################intialize random features ###############################
    feat = torch.Tensor(data.num_nodes_dict[subject_node], args.emb_size)
    torch.nn.init.xavier_uniform_(feat)
    feat_dic = {subject_node: feat}
    ################################################################
    x_dict = {}
    for key, x in feat_dic.items():
        x_dict[key2int[key]] = x
    num_nodes_dict = {}
    for key, N in data.num_nodes_dict.items():
        num_nodes_dict[key2int[key]] = N
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    model = RGCN(args.emb_size, args.hidden_channels, dataset.num_classes,
                 args.num_layers, args.dropout, num_nodes_dict,
                 list(x_dict.keys()), len(edge_index_dict.keys())).to(device)
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    print("x_dict=", x_dict.keys())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("start test")
    test()  # Test if inference succeeds.
    ################## start Training #######################
    for run in range(args.runs):
        model.reset_parameters()
        for epoch in range(1, 1 + args.epochs):
            loss = train(epoch)
            torch.cuda.empty_cache()
            result = test()
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f"Train: {100 * train_acc['acc']:.2f}%, "
                  f"Valid: {100 * valid_acc['acc']:.2f}%, "
                  f"Test: {100 * test_acc['acc']:.2f}%")
            print("model # Total  parameters ",
                  sum(p.numel() for p in model.parameters()))
            print(
                "model # trainable paramters ",
                sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':
    graphSaint()
