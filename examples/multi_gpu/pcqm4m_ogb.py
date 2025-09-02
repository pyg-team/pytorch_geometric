# Code adapted from OGB.
# https://github.com/snap-stanford/ogb/tree/master/examples/lsc/pcqm4m-v2
import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from torch_geometric.data import Data
from torch_geometric.datasets import PCQM4Mv2
from torch_geometric.io import fs
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GlobalAttention,
    MessagePassing,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import degree

try:
    from ogb.lsc import PCQM4Mv2Evaluator, PygPCQM4Mv2Dataset
except ImportError as e:
    raise ImportError(
        "`PygPCQM4Mv2Dataset` requires rdkit (`pip install rdkit`)") from e

from ogb.utils import smiles2graph


def ogb_from_smiles_wrapper(smiles, *args, **kwargs):
    """Returns `torch_geometric.data.Data` object from smiles while
    `ogb.utils.smiles2graph` returns a dict of np arrays.
    """
    data_dict = smiles2graph(smiles, *args, **kwargs)
    return Data(
        x=torch.from_numpy(data_dict['node_feat']),
        edge_index=torch.from_numpy(data_dict['edge_index']),
        edge_attr=torch.from_numpy(data_dict['edge_feat']),
        smiles=smiles,
    )


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        r"""GINConv.

        Args:
            emb_dim (int): node embedding dimensionality
        """
        super().__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        return self.mlp(
            (1 + self.eps) * x +
            self.propagate(edge_index, x=x, edge_attr=edge_embedding))

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr='add')
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNNNode(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK="last",
                 residual=False, gnn_type='gin'):
        r"""GNN Node.

        Args:
            emb_dim (int): node embedding dimensionality.
            num_layers (int): number of GNN message passing layers.
            residual (bool): whether to add residual connection.
            drop_ratio (float): dropout ratio.
            JK (str): "last" or "sum" to choose JK concat strat.
            residual (bool): Whether or not to add the residual
            gnn_type (str): Type of GNN to use.
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.atom_encoder = AtomEncoder(emb_dim)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError(f'Undefined GNN type called {gnn_type}')

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x = batched_data.x
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr

        # compute input node embedding
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio,
                              training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation


class GNNNodeVirtualNode(torch.nn.Module):
    """Outputs node representations."""
    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK="last",
                 residual=False, gnn_type='gin'):
        super().__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.atom_encoder = AtomEncoder(emb_dim)

        # set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {gnn_type}')

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for _ in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU(),
                ))

    def forward(self, batched_data):
        x = batched_data.x
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr
        batch = batched_data.batch

        # virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(
                edge_index.device))

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            # add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio,
                              training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            # update the virtual nodes
            if layer < self.num_layers - 1:
                # add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(
                    h_list[layer], batch) + virtualnode_embedding
                # transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer]
                        (virtualnode_embedding_temp), self.drop_ratio,
                        training=self.training)
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](
                            virtualnode_embedding_temp), self.drop_ratio,
                        training=self.training)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_tasks=1,
        num_layers=5,
        emb_dim=300,
        gnn_type='gin',
        virtual_node=True,
        residual=False,
        drop_ratio=0,
        JK="last",
        graph_pooling="sum",
    ):
        r"""GNN.

        Args:
            num_tasks (int): number of labels to be predicted
            num_layers (int): number of gnn layers.
            emb_dim (int): embedding dim to use.
            gnn_type (str): Type of GNN to use.
            virtual_node (bool): whether to add virtual node or not.
            residual (bool): Whether or not to add the residual
            drop_ratio (float): dropout ratio.
            JK (str): "last" or "sum" to choose JK concat strat.
            graph_pooling (str): Graph pooling strat to use.
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        if virtual_node:
            self.gnn_node = GNNNodeVirtualNode(
                num_layers,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )
        else:
            self.gnn_node = GNNNode(
                num_layers,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, 1),
            ))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        output = self.graph_pred_linear(h_graph)
        if self.training:
            return output
        else:
            # At inference time, we clamp the value between 0 and 20
            return torch.clamp(output, min=0, max=20)


def train(model, rank, device, loader, optimizer):
    model.train()
    reg_criterion = torch.nn.L1Loss()
    loss_accum = 0.0
    for step, batch in enumerate(  # noqa: B007
            tqdm(loader, desc="Training", disable=(rank > 0))):
        batch = batch.to(device)
        pred = model(batch).view(-1, )
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()
    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch).view(-1, )

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader):
    model.eval()
    y_pred = []
    for batch in tqdm(loader, desc="Testing"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch).view(-1, )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    return y_pred


def run(rank, dataset, args):
    num_devices = args.num_devices
    device = torch.device(
        "cuda:" + str(rank)) if num_devices > 0 else torch.device("cpu")

    if num_devices > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=num_devices)

    if args.on_disk_dataset:
        train_idx = torch.arange(len(dataset.indices()))
    else:
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]

    if num_devices > 1:
        num_splits = math.ceil(train_idx.size(0) / num_devices)
        train_idx = train_idx.split(num_splits)[rank]

    if args.train_subset:
        subset_ratio = 0.1
        n = len(train_idx)
        subset_idx = torch.randperm(n)[:int(subset_ratio * n)]
        train_dataset = dataset[train_idx[subset_idx]]
    else:
        train_dataset = dataset[train_idx]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    if rank == 0:
        if args.on_disk_dataset:
            valid_dataset = PCQM4Mv2(root='on_disk_dataset/', split="val",
                                     from_smiles_func=ogb_from_smiles_wrapper)
            test_dev_dataset = PCQM4Mv2(
                root='on_disk_dataset/', split="test",
                from_smiles_func=ogb_from_smiles_wrapper)
            test_challenge_dataset = PCQM4Mv2(
                root='on_disk_dataset/', split="holdout",
                from_smiles_func=ogb_from_smiles_wrapper)
        else:
            valid_dataset = dataset[split_idx["valid"]]
            test_dev_dataset = dataset[split_idx["test-dev"]]
            test_challenge_dataset = dataset[split_idx["test-challenge"]]

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if args.save_test_dir != '':
            testdev_loader = DataLoader(
                test_dev_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            testchallenge_loader = DataLoader(
                test_challenge_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

        if args.checkpoint_dir != '':
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        evaluator = PCQM4Mv2Evaluator()

    gnn_type, virtual_node = args.gnn.split('-')
    model = GNN(
        gnn_type=gnn_type,
        virtual_node=virtual_node,
        num_layers=args.num_layers,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
        graph_pooling=args.graph_pooling,
    )
    if num_devices > 0:
        model = model.to(rank)
    if num_devices > 1:
        model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    current_epoch = 1

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    if os.path.isfile(checkpoint_path):
        checkpoint = fs.torch_load(checkpoint_path)
        current_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_valid_mae = checkpoint['best_val_mae']
        print(f"Found checkpoint, resume training at epoch {current_epoch}")

    for epoch in range(current_epoch, args.epochs + 1):
        train_mae = train(model, rank, device, train_loader, optimizer)

        if num_devices > 1:
            dist.barrier()

        if rank == 0:
            valid_mae = eval(
                model.module if isinstance(model, DistributedDataParallel) else
                model, device, valid_loader, evaluator)

            print(f"Epoch {epoch:02d}, "
                  f"Train MAE: {train_mae:.4f}, "
                  f"Val MAE: {valid_mae:.4f}")

            if args.log_dir != '':
                writer.add_scalar('valid/mae', valid_mae, epoch)
                writer.add_scalar('train/mae', train_mae, epoch)

            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                if args.checkpoint_dir != '':
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_mae': best_valid_mae,
                    }
                    torch.save(checkpoint, checkpoint_path)

                if args.save_test_dir != '':
                    test_model = model.module if isinstance(
                        model, DistributedDataParallel) else model

                    testdev_pred = test(test_model, device, testdev_loader)
                    evaluator.save_test_submission(
                        {'y_pred': testdev_pred.cpu().detach().numpy()},
                        args.save_test_dir,
                        mode='test-dev',
                    )

                    testchallenge_pred = test(test_model, device,
                                              testchallenge_loader)
                    evaluator.save_test_submission(
                        {'y_pred': testchallenge_pred.cpu().detach().numpy()},
                        args.save_test_dir,
                        mode='test-challenge',
                    )

            print(f'Best validation MAE so far: {best_valid_mae}')

        if num_devices > 1:
            dist.barrier()

        scheduler.step()

    if rank == 0 and args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='GNN baselines on pcqm4m with Pytorch Geometrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        choices=['gin', 'gin-virtual', 'gcn',
                                 'gcn-virtual'], help='GNN architecture')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default='',
                        help='directory to save test submission file')
    parser.add_argument('--num_devices', type=int, default='1',
                        help="Number of GPUs, if 0 runs on the CPU")
    parser.add_argument('--on_disk_dataset', action='store_true')
    args = parser.parse_args()

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available(
    ) else 0
    if args.num_devices > available_gpus:
        if available_gpus == 0:
            print("No GPUs available, running w/ CPU...")
        else:
            raise ValueError(f"Cannot train with {args.num_devices} GPUs: "
                             f"available GPUs count {available_gpus}")

    # automatic dataloading and splitting
    if args.on_disk_dataset:
        dataset = PCQM4Mv2(root='on_disk_dataset/', split='train',
                           from_smiles_func=ogb_from_smiles_wrapper)
    else:
        dataset = PygPCQM4Mv2Dataset(root='dataset/')

    if args.num_devices > 1:
        mp.spawn(run, args=(dataset, args), nprocs=args.num_devices, join=True)
    else:
        run(0, dataset, args)
