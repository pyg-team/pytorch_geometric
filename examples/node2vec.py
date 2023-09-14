import os.path as osp
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec


def main():
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset)
    data = dataset[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
    ).to(device)

    num_workers = 0 if sys.platform.startswith('win') else 4
    loader = model.loader(batch_size=128, shuffle=True,
                          num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc

    for epoch in range(1, 101):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    @torch.no_grad()
    def plot_points(colors):
        model.eval()
        z = model(torch.arange(data.num_nodes, device=device))
        z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
        y = data.y.cpu().numpy()

        plt.figure(figsize=(8, 8))
        for i in range(dataset.num_classes):
            plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
        plt.axis('off')
        plt.show()

    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700'
    ]
    plot_points(colors)


if __name__ == "__main__":
    main()
