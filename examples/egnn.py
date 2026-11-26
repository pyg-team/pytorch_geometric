import argparse
import copy
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import EGNN, global_max_pool


class Net(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels,
                 backbone):
        super().__init__()
        self.encoder = EGNN(in_channels, hidden_channels, hidden_channels,
                            num_layers=2, edge_dim=edge_dim, backbone=backbone)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.encoder(x, edge_index, edge_attr)
        return self.lin(global_max_pool(x, batch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['gcn', 'gat'], default='gat')
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
    dataset = TUDataset(path, name='MUTAG').shuffle()
    train_loader = DataLoader(dataset[:0.8], args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[0.8:0.9], args.batch_size)
    test_loader = DataLoader(dataset[0.9:], args.batch_size)

    model = Net(dataset.num_node_features, dataset.num_edge_features,
                args.hidden_channels, dataset.num_classes,
                args.backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005,
                                 weight_decay=0.0001)

    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * data.num_graphs
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            correct += int((out.argmax(-1) == data.y).sum())
        return correct / len(loader.dataset)

    best_val, best_state = -1., copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        loss = train()
        val_acc = test(val_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')

    model.load_state_dict(best_state)
    print(f'Test: {test(test_loader):.4f}')


if __name__ == '__main__':
    main()
