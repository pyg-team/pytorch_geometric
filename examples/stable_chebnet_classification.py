import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import StableChebConv, global_mean_pool


class StableChebGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, K=8, epsilon=0.3, gamma=0.05, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(
            StableChebConv(in_channels, hidden_channels, K, epsilon=epsilon,
                           gamma=gamma))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(
                StableChebConv(hidden_channels, hidden_channels, K,
                               epsilon=epsilon, gamma=gamma))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = F.dropout(F.relu(self.lin1(x)), p=self.dropout,
                      training=self.training)
        return F.log_softmax(self.lin2(x), dim=-1)


def train_graph(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data.x, data.edge_index, data.batch), data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test_graph(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        correct += int((model(data.x, data.edge_index,
                              data.batch).argmax(dim=-1) == data.y).sum())
    return correct / len(loader.dataset)


def run_graph_classification(epochs=100, hidden=64, num_layers=3, K=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG').shuffle()
    n = int(len(dataset) * 0.8)
    train_loader = DataLoader(dataset[:n], batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset[n:], batch_size=32)

    model = StableChebGraphClassifier(dataset.num_node_features, hidden,
                                      dataset.num_classes,
                                      num_layers=num_layers, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,
                                                gamma=0.5)

    best_test = 0.0
    for epoch in range(1, epochs + 1):
        loss = train_graph(model, train_loader, optimizer, device)
        test_acc = test_graph(model, test_loader, device)
        scheduler.step()
        if test_acc > best_test:
            best_test = test_acc
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | loss {loss:.4f} | test {test_acc:.3f}")

    print(f"\n[MUTAG] best test {best_test:.4f}")


def sanity_check():
    torch.manual_seed(0)

    graph_model = StableChebGraphClassifier(7, 32, 2, num_layers=3, K=6)
    x2 = torch.randn(20, 7)
    ei2 = torch.randint(0, 20, (2, 60))
    batch = torch.cat(
        [torch.zeros(10, dtype=torch.long),
         torch.ones(10, dtype=torch.long)])
    assert graph_model(x2, ei2, batch).shape == (2, 2)
    print(f"GraphClassifier → {graph_model(x2, ei2, batch).shape}  ✓")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['sanity', 'node', 'graph', 'all'],
                        default='sanity')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--K', type=int, default=10)
    args, _ = parser.parse_known_args()

    sanity_check()
    run_graph_classification(100, args.hidden, 3, 8)
