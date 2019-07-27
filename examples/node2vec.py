import os.path as osp

import torch
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]
loader = DataLoader(torch.arange(data.num_nodes), batch_size=128, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Node2Vec(data.num_nodes, embedding_dim=128, walk_length=80,
                 context_size=10, walks_per_node=10)
model, data = model.to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    for subset in loader:
        optimizer.zero_grad()
        loss = model.loss(data.edge_index, subset.to(device))
        loss.backward()
        optimizer.step()
    return loss.item()


for epoch in range(1, 51):
    loss = train()

# class Classifier(torch.nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.lin1 = torch.nn.Linear(feature_dim, feature_dim)
#         self.lin2 = torch.nn.Linear(feature_dim, num_labels)

#     def forward(self, input):
#         x = self.lin1(input)
#         x = F.relu(x)
#         x = self.lin2(x)
#         return F.log_softmax(x, dim=1)

# classifier = Classifier().to(dev)
# optimizer_c = torch.optim.Adam(classifier.parameters(), lr=0.001)

# def train_classifier():
#     classifier.train()
#     optimizer_c.zero_grad()
#     train_indices = data.train_mask.nonzero().squeeze().to(dev)
#     output = classifier(embedding(train_indices).detach())
#     label = data.y[data.train_mask].to(dev)
#     loss = F.nll_loss(output, label)
#     loss.backward()
#     optimizer_c.step()
#     print('Classifier Loss: {:.6f}'.format(loss))

# def test_classifier():
#     classifier.eval()
#     accs = []
#     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#         indices = mask.nonzero().squeeze().to(dev)
#         output = classifier(embedding(indices).detach())
#         pred = output.max(1)[1]
#         label = data.y[mask].to(dev)
#         acc = pred.eq(label).sum().item() / mask.sum().item()
#         accs.append(acc)
#     return accs

# # train node2vec model
# try:

#     print('Start Train Embedding')
#     for epoch in range(1, 50):
#         loss = train_embedding()
#         print('Epoch: {:3d}, Negative Sampling Loss: {:.6f}'.format(
#             epoch, loss.item()))

# except (KeyboardInterrupt):
#     print('Stopped train_embedding()')

# # train classifier with node embedding and test
# best_val_acc = test_acc = 0
# for epoch in range(1, 1000):
#     train_classifier()
#     train_acc, val_acc, tmp_test_acc = test_classifier()
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         test_acc = tmp_test_acc
#     log = 'Epoch: {:3d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
#     print(log.format(epoch, train_acc, best_val_acc, test_acc))
