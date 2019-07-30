import os.path as osp

import torch
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset)
data = dataset[0]
loader = DataLoader(torch.arange(data.num_nodes), batch_size=128, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(data.num_nodes, embedding_dim=2, walk_length=20,
                 context_size=10, walks_per_node=10)
model, data = model.to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for subset in loader:
        optimizer.zero_grad()
        loss = model.loss(data.edge_index, subset.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test():
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask], max_iter=150)
    return acc


for epoch in range(1, 1001):
    loss = train()
    print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss))
acc = test()
print('Accuracy: {:.4f}'.format(acc))


def plot_points():
    X = model(torch.arange(data.num_nodes, device=device))
    X = X.cpu().detach().numpy()

    #X = data.x.cpu().detach().numpy()
    y = data.y.cpu().detach().numpy()
    colors = ['red','blue','green','purple','cyan','magenta','yellow']

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,8))
    for color, i in zip(colors, [0, 1, 2, 3, 4, 5, 6]):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, linewidth=0.5)

    plt.axis('off')
    plt.savefig('plot_1000_dim2.png', bbox_inches='tight')

plot_points()
