# An Multi GPU implementation of unsupervised bipartite GraphSAGE
# using the Alibaba Taobao dataset.
import argparse
import os
import os.path as osp

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score
from torch.nn import Embedding, Linear
from torch.nn.parallel import DistributedDataParallel

import torch_geometric.transforms as T
from torch_geometric.datasets import Taobao
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils.convert import to_scipy_sparse_matrix


class ItemGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


class UserGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        item_x = self.conv1(
            x_dict['item'],
            edge_index_dict[('item', 'to', 'item')],
        ).relu()

        user_x = self.conv2(
            (x_dict['item'], x_dict['user']),
            edge_index_dict[('item', 'rev_to', 'user')],
        ).relu()

        user_x = self.conv3(
            (item_x, user_x),
            edge_index_dict[('item', 'rev_to', 'user')],
        ).relu()

        return self.lin(user_x)


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = Embedding(num_users, hidden_channels)
        self.item_emb = Embedding(num_items, hidden_channels)
        self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        x_dict['user'] = self.user_emb(x_dict['user'])
        x_dict['item'] = self.item_emb(x_dict['item'])
        z_dict['item'] = self.item_encoder(
            x_dict['item'],
            edge_index_dict[('item', 'to', 'item')],
        )
        z_dict['user'] = self.user_encoder(x_dict, edge_index_dict)

        return self.decoder(z_dict['user'], z_dict['item'], edge_label_index)


def run_train(rank, data, train_data, val_data, test_data, args, world_size):
    if rank == 0:
        print("Setting up Data Loaders...")
    train_edge_label_idx = train_data[('user', 'to', 'item')].edge_label_index
    train_edge_label_idx = train_edge_label_idx.split(
        train_edge_label_idx.size(1) // world_size, dim=1)[rank].clone()
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[8, 4],
        edge_label_index=(('user', 'to', 'item'), train_edge_label_idx),
        neg_sampling='binary',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[8, 4],
        edge_label_index=(
            ('user', 'to', 'item'),
            val_data[('user', 'to', 'item')].edge_label_index,
        ),
        edge_label=val_data[('user', 'to', 'item')].edge_label,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[8, 4],
        edge_label_index=(
            ('user', 'to', 'item'),
            test_data[('user', 'to', 'item')].edge_label_index,
        ),
        edge_label=test_data[('user', 'to', 'item')].edge_label,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    def train():
        model.train()

        total_loss = total_examples = 0
        for batch in tqdm.tqdm(train_loader, disable=rank != 0):
            batch = batch.to(rank)
            optimizer.zero_grad()

            pred = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch['user', 'item'].edge_label_index,
            )
            loss = F.binary_cross_entropy_with_logits(
                pred, batch['user', 'item'].edge_label)

            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            total_examples += pred.numel()

        return total_loss / total_examples

    @torch.no_grad()
    def test(loader):
        model.eval()
        preds, targets = [], []
        for batch in tqdm.tqdm(loader, disable=rank != 0):
            batch = batch.to(rank)

            pred = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch['user', 'item'].edge_label_index,
            ).sigmoid().view(-1).cpu()
            target = batch['user', 'item'].edge_label.long().cpu()

            preds.append(pred)
            targets.append(target)

        pred = torch.cat(preds, dim=0).numpy()
        target = torch.cat(targets, dim=0).numpy()

        return roc_auc_score(target, pred)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    model = Model(
        num_users=data['user'].num_nodes,
        num_items=data['item'].num_nodes,
        hidden_channels=64,
        out_channels=64,
    ).to(rank)
    # Initialize lazy modules
    for batch in train_loader:
        batch = batch.to(rank)
        _ = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch['user', 'item'].edge_label_index,
        )
        break
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_auc = 0
    for epoch in range(1, args.epochs):
        print("Train")
        loss = train()
        if rank == 0:
            print("Val")
            val_auc = test(val_loader)
            best_val_auc = max(best_val_auc, val_auc)
        if rank == 0:
            print(
                f'Epoch: {epoch:02d}, Loss: {loss:4f}, Val AUC: {val_auc:.4f}')
    if rank == 0:
        print("Test")
        test_auc = test(test_loader)
        print(f'Total {args.epochs:02d} epochs: Final Loss: {loss:4f}, '
              f'Best Val AUC: {best_val_auc:.4f}, '
              f'Test AUC: {test_auc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=16,
                        help="Number of workers per dataloader")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=21)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument(
        '--dataset_root_dir', type=str,
        default=osp.join(osp.dirname(osp.realpath(__file__)),
                         '../../data/Taobao'))
    args = parser.parse_args()

    def pre_transform(data):
        # Compute sparsified item<>item relationships through users:
        print('Computing item<>item relationships...')
        mat = to_scipy_sparse_matrix(data['user', 'item'].edge_index).tocsr()
        mat = mat[:data['user'].num_nodes, :data['item'].num_nodes]
        comat = mat.T @ mat
        comat.setdiag(0)
        comat = comat >= 3.
        comat = comat.tocoo()
        row = torch.from_numpy(comat.row).to(torch.long)
        col = torch.from_numpy(comat.col).to(torch.long)
        data['item', 'item'].edge_index = torch.stack([row, col], dim=0)
        return data

    dataset = Taobao(args.dataset_root_dir, pre_transform=pre_transform)
    data = dataset[0]

    data['user'].x = torch.arange(0, data['user'].num_nodes)
    data['item'].x = torch.arange(0, data['item'].num_nodes)

    # Only consider user<>item relationships for simplicity:
    del data['category']
    del data['item', 'category']
    del data['user', 'item'].time
    del data['user', 'item'].behavior

    # Add a reverse ('item', 'rev_to', 'user') relation for message passing:
    data = T.ToUndirected()(data)

    # Perform a link-level split into training, validation, and test edges:
    print('Computing data splits...')
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=False,
        edge_types=[('user', 'to', 'item')],
        rev_edge_types=[('item', 'rev_to', 'user')],
    )(data)
    print('Done!')

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run_train,
             args=(data, train_data, val_data, test_data, args, world_size),
             nprocs=world_size, join=True)
