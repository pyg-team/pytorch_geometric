import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
import os
import scipy.io as sio
import numpy as np
from optparse import OptionParser
import time
from torch_geometric.nn import PANConv, PANPooling


class PAN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, nhid, ratio,
                 filter_size):
        super(PAN, self).__init__()
        self.conv1 = PANConv(num_node_features, nhid, filter_size)
        self.pool1 = PANPooling(nhid, filter_size=filter_size)
        self.conv2 = PANConv(nhid, nhid, filter_size=2)
        self.pool2 = PANPooling(nhid)
        self.conv3 = PANConv(nhid, nhid, filter_size=2)
        self.pool3 = PANPooling(nhid)

        self.lin1 = torch.nn.Linear(nhid, nhid // 2)
        self.lin2 = torch.nn.Linear(nhid // 2, nhid // 4)
        self.lin3 = torch.nn.Linear(nhid // 4, num_classes)

        self.mlp = torch.nn.Linear(nhid, num_classes)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        perm_list = list()
        edge_mask_list = None

        x = self.conv1(x, edge_index)
        M = self.conv1.m
        x, edge_index, _, batch, perm, score_perm = self.pool1(
            x, edge_index, batch=batch, M=M)
        perm_list.append(perm)

        x = self.conv2(x, edge_index, edge_mask_list=edge_mask_list)
        M = self.conv2.m
        x, edge_index, _, batch, perm, score_perm = self.pool2(
            x, edge_index, batch=batch, M=M)
        perm_list.append(perm)

        x = self.conv3(x, edge_index, edge_mask_list=edge_mask_list)
        M = self.conv3.m
        x, edge_index, _, batch, perm, score_perm = self.pool3(
            x, edge_index, batch=batch, M=M)
        perm_list.append(perm)

        mean = scatter_mean(x, batch, dim=0)
        x = mean

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x, perm_list


def train(model, train_loader, device):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        for name, param in model.named_parameters():
            if 'panconv_filter_weight' in name:
                param.data = param.data.clamp(0, 1)
            if 'panpool_filter_weight' in name:
                param.data = param.data.clamp(0, 1)
    return loss_all / len(train_loader.dataset)


def test(model, loader, device):
    model.eval()

    correct = 0
    loss = 0.0
    for data in loader:
        data = data.to(device)
        out, _ = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y).item() * data.num_graphs
    return correct / len(loader.dataset), loss / len(loader.dataset)


parser = OptionParser()
parser.add_option(
    "--dataset_name", dest="dataset_name", default='PROTEINS', help=
    "the name of dataset from Pytorch Geometric, other options include PROTEINS_full, NCI1, AIDS, Mutagenicity"
)
parser.add_option("--phi", dest="phi", default=0.3, type=np.float,
                  help="type of dataset dataset")
parser.add_option("--runs", dest="runs", default=1, type=np.int,
                  help="number of runs")
parser.add_option("--batch_size", type=np.int, dest="batch_size", default=32,
                  help="batch size")
parser.add_option("--L", dest="L", default=4, type=np.int,
                  help="order L in MET")
parser.add_option("--learning_rate", type=np.float, dest="learning_rate",
                  default=0.005, help="learning rate")
parser.add_option("--weight_decay", type=np.float, dest="weight_decay",
                  default=1e-3, help="weight decay")
parser.add_option("--pool_ratio", type=np.float, dest="pool_ratio",
                  default=0.5, help="proportion of nodes to be pooled")
parser.add_option("--nhid", type=np.int, dest="nhid", default=64,
                  help="number of each hidden-layer neurons")
parser.add_option("--epochs", type=np.int, dest="epochs", default=100,
                  help="number of epochs each run")
options, argss = parser.parse_args()
datasetname = options.dataset_name
phi = options.phi
runs = options.runs
batch_size = options.batch_size
filter_size = options.L + 1
learning_rate = options.learning_rate
weight_decay = options.weight_decay
pool_ratio = options.pool_ratio
nhid = options.nhid
epochs = options.epochs

train_loss = np.zeros((runs, epochs), dtype=np.float)
val_loss = np.zeros((runs, epochs), dtype=np.float)
val_acc = np.zeros((runs, epochs), dtype=np.float)
test_acc = np.zeros(runs, dtype=np.float)
min_loss = 1e10 * np.ones(runs)

# dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sv_dir = 'data/'
if not os.path.exists(sv_dir):
    os.makedirs(sv_dir)
path = os.path.join(os.path.abspath(''), 'data', datasetname)
dataset = TUDataset(path, name=datasetname)

print(len(dataset))
print(dataset.num_classes)
print(dataset.num_node_features)

num_classes = dataset.num_classes
num_node_features = dataset.num_node_features
num_edge = 0
num_node = 0
num_graph = len(dataset)

dataset1 = list()
for i in range(len(dataset)):
    data1 = Data(x=dataset[i].x, edge_index=dataset[i].edge_index,
                 y=dataset[i].y)
    data1.num_node = dataset[i].num_nodes
    data1.num_edge = dataset[i].edge_index.size(1)
    num_node = num_node + data1.num_node
    num_edge = num_edge + data1.num_edge
    dataset1.append(data1)
dataset = dataset1

num_edge = num_edge * 1.0 / num_graph
num_node = num_node * 1.0 / num_graph

# generate training, validation and test data sets
num_training = int(num_graph * 0.8)
num_val = int(num_graph * 0.1)
num_test = num_graph - (num_training + num_val)

## train model
for run in range(runs):

    training_set, val_set, test_set = random_split(
        dataset, [num_training, num_val, num_test])
    train_loader = DataLoader(training_set, batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print('***** PAN for {}, phi {} *****'.format(datasetname, phi))
    print('#training data: {}, #test data: {}'.format(num_training, num_test))
    print('Mean #nodes: {:.1f}, mean #edges: {:.1f}'.format(
        num_node, num_edge))
    print('Network architectur: PC-PA')
    print(
        'filter_size: {:d}, pool_ratio: {:.2f}, learning rate: {:.2e}, weight decay: {:.2e}, nhid: {:d}'
        .format(filter_size, pool_ratio, learning_rate, weight_decay, nhid))
    print('batchsize: {:d}, epochs: {:d}, runs: {:d}'.format(
        batch_size, epochs, runs))
    print('Device: {}'.format(device))

    ## train model
    model = PAN(num_node_features, num_classes, nhid=nhid, ratio=pool_ratio,
                filter_size=filter_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    for epoch in range(epochs):
        # training
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
            for name, param in model.named_parameters():
                if 'panconv_filter_weight' in name:
                    param.data = param.data.clamp(0, 1)
                if 'panpool_filter_weight' in name:
                    param.data = param.data.clamp(0, 1)
        loss = loss_all / len(train_loader.dataset)
        train_loss[run, epoch] = loss
        # validation
        val_acc_1, val_loss_1 = test(model, val_loader, device)
        val_loss[run, epoch] = val_loss_1
        val_acc[run, epoch] = val_acc_1
        print('Run: {:02d}, Epoch: {:03d}, Val loss: {:.4f}, Val acc: {:.4f}'.
              format(run + 1, epoch + 1, val_loss[run, epoch], val_acc[run,
                                                                       epoch]))
        if val_loss_1 < min_loss[run]:
            # save the model and reuse later in test
            torch.save(model.state_dict(), 'latest.pth')
            min_loss[run] = val_loss_1

    # test
    model.load_state_dict(torch.load('latest.pth'))
    test_acc[run], _ = test(model, test_loader, device)
    print('==Test Acc: {:.4f}'.format(test_acc[run]))

print('==Mean Test Acc: {:.4f}'.format(np.mean(test_acc)))

t1 = time.time()
sv = datasetname + '_pcpa_runs' + str(runs) + '_phi' + str(
    phi) + '_time' + str(t1) + '.mat'
sio.savemat(
    sv, mdict={
        'test_acc': test_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'filter_size': filter_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'nhid': nhid,
        'batch_size': batch_size,
        'epochs': epochs
    })
