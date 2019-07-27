import math
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec, DataParallel

torch.backends.cudnn.benchmark = False
torch.manual_seed(762019)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'CiteSeer'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]
feature_dim = 128

p = 1
q = 1
walks_per_node = 10
walk_length = 80
context_size = 10
neg_samples = 10

num_nodes = data.num_nodes
num_labels = data.y.max().item() + 1

node2vec = Node2Vec(data, feature_dim, p, q, walks_per_node, walk_length, context_size, neg_samples)
embedding = node2vec.embedding
optimizer_e = torch.optim.Adam(embedding.parameters(), lr=0.01, weight_decay=0.0)


def train_embedding():
    embedding.train()
    optimizer_e.zero_grad()
    random_walks = node2vec.random_walks()
    batch_size = math.ceil(random_walks.size(0) / 8)
    runs = random_walks.size(0) / batch_size
    for i in range(math.ceil(runs)):
        # compute size for last batch
        if not runs.is_integer() and i == math.floor(runs):
            size = (random_walks.size(0) - batch_size * math.floor(runs)).__int__()
            j = random_walks.size(0) - size
            walks = random_walks[j:]
        else:
            walks = random_walks[i * batch_size:(i + 1) * batch_size]
        loss = node2vec.neg_loss(walks)
        loss.backward()
        optimizer_e.step()
        print('Negative Sampling Loss:', loss.item())
    return loss

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lin1 = torch.nn.Linear(feature_dim, feature_dim)
        self.lin2 = torch.nn.Linear(feature_dim, num_labels)

    def forward(self, input):
        x = self.lin1(input)
        x = F.relu(x)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

classifier = Classifier().to(dev)
optimizer_c = torch.optim.Adam(classifier.parameters(), lr=0.001)

def train_classifier():
    classifier.train()
    optimizer_c.zero_grad()
    train_indices = data.train_mask.nonzero().squeeze().to(dev)
    output = classifier(embedding(train_indices).detach())
    label = data.y[data.train_mask].to(dev)
    loss = F.nll_loss(output, label)
    loss.backward()
    optimizer_c.step()
    print('Classifier Loss: {:.6f}'.format(loss))

def test_classifier():
    classifier.eval()
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        indices = mask.nonzero().squeeze().to(dev)
        output = classifier(embedding(indices).detach())
        pred = output.max(1)[1]
        label = data.y[mask].to(dev)
        acc = pred.eq(label).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


# train node2vec model
try:

    print('Start Train Embedding')
    for epoch in range(1, 50):
        loss = train_embedding()
        print('Epoch: {:3d}, Negative Sampling Loss: {:.6f}'.format(epoch, loss.item()))

except(KeyboardInterrupt):
    print('Stopped train_embedding()')


# train classifier with node embedding and test
best_val_acc = test_acc = 0
for epoch in range(1, 1000):
    train_classifier()
    train_acc, val_acc, tmp_test_acc = test_classifier()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:3d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))


