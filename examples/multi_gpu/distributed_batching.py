import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from ogb.graphproppred import PygGraphPropPredDataset as Dataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=3,
                 dropout=0.5):
        super().__init__()

        self.dropout = dropout

        self.atom_encoder = AtomEncoder(hidden_channels)
        self.bond_encoder = BondEncoder(hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                ReLU(),
            )
            self.convs.append(GINEConv(nn, train_eps=True))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, adj_t, batch):
        x = self.atom_encoder(x)
        edge_attr = adj_t.coo()[2]
        adj_t = adj_t.set_value(self.bond_encoder(edge_attr), layout='coo')

        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


def run(rank, world_size: int, dataset_name: str, root: str):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    dataset = Dataset(dataset_name, root,
                      pre_transform=T.ToSparseTensor(attr='edge_attr'))
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(dataset_name)

    train_dataset = dataset[split_idx['train']]
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=128,
                              sampler=train_sampler)

    torch.manual_seed(12345)
    model = GIN(128, dataset.num_tasks, num_layers=3, dropout=0.5).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    if rank == 0:
        val_loader = DataLoader(dataset[split_idx['valid']], batch_size=256)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=256)

    for epoch in range(1, 51):
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(rank)
            optimizer.zero_grad()
            logits = model(data.x, data.adj_t, data.batch)
            loss = criterion(logits, data.y.to(torch.float))
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * logits.size(0)
        loss = total_loss / len(train_loader.dataset)

        dist.barrier()

        if rank == 0:  # We evaluate on a single GPU for now.
            model.eval()

            y_pred, y_true = [], []
            for data in val_loader:
                data = data.to(rank)
                with torch.no_grad():
                    y_pred.append(model.module(data.x, data.adj_t, data.batch))
                    y_true.append(data.y)
            val_rocauc = evaluator.eval({
                'y_pred': torch.cat(y_pred, dim=0),
                'y_true': torch.cat(y_true, dim=0),
            })['rocauc']

            y_pred, y_true = [], []
            for data in test_loader:
                data = data.to(rank)
                with torch.no_grad():
                    y_pred.append(model.module(data.x, data.adj_t, data.batch))
                    y_true.append(data.y)
            test_rocauc = evaluator.eval({
                'y_pred': torch.cat(y_pred, dim=0),
                'y_true': torch.cat(y_true, dim=0),
            })['rocauc']

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Val: {val_rocauc:.4f}, Test: {test_rocauc:.4f}')

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    dataset_name = 'ogbg-molhiv'
    root = '../../data/OGB'

    # Download and process the dataset on main process.
    Dataset(dataset_name, root,
            pre_transform=T.ToSparseTensor(attr='edge_attr'))

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    args = (world_size, dataset_name, root)
    mp.spawn(run, args=args, nprocs=world_size, join=True)
