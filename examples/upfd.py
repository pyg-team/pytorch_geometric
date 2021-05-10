import argparse
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.metrics import average_precision_score

import torch
import torch.nn.functional as F
from torch.utils.data import random_split, Subset
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from torch_geometric.datasets import UPFD
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.transforms import ToUndirected

"""

The example implementation of UPFD-GCN, UPFD-GAT, and UPFD-SAGE
based on the UPFD graph classification dataset

Paper: User Preference-aware Fake News Detection
Link: https://arxiv.org/abs/2104.12259
Source Code: https://github.com/safe-graph/GNN-FakeNews

"""


class Model(torch.nn.Module):
    def __init__(self, args, concat=False):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.model = args.model
        self.concat = concat

        if self.model == 'gcn':
            self.conv1 = GCNConv(self.num_features, self.nhid)
        elif self.model == 'sage':
            self.conv1 = SAGEConv(self.num_features, self.nhid)
        elif self.model == 'gat':
            self.conv1 = GATConv(self.num_features, self.nhid)

        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = gmp(x, batch)

        if self.concat:
            news = [data.x[(data.batch == idx).nonzero().squeeze()[0]]
                    for idx in range(data.num_graphs)]
            news = torch.stack(news)
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))

        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


def eval_deep(log, loader):
    """
    Evaluating the classification performance given mini-batch data
    """

    # get the empirical batch_size for each mini-batch
    data_size = len(loader.dataset.indices)
    batch_size = loader.batch_size
    if data_size % batch_size == 0:
        size_list = [batch_size] * (data_size // batch_size)
    else:
        size_list = [batch_size] * (data_size // batch_size) + \
                    [data_size % batch_size]

    assert len(log) == len(size_list)

    accuracy, f1_macro, f1_micro, precision, recall = 0, 0, 0, 0, 0

    prob_log, label_log = [], []

    for batch, size in zip(log, size_list):
        pred_y = batch[0].data.cpu().numpy().argmax(axis=1)
        y = batch[1].data.cpu().numpy().tolist()
        prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
        label_log.extend(y)

        accuracy += accuracy_score(y, pred_y) * size
        f1_macro += f1_score(y, pred_y, average='macro') * size
        f1_micro += f1_score(y, pred_y, average='micro') * size
        precision += precision_score(y, pred_y, zero_division=0) * size
        recall += recall_score(y, pred_y, zero_division=0) * size

    auc = roc_auc_score(label_log, prob_log)
    ap = average_precision_score(label_log, prob_log)

    accuracy /= data_size
    f1_macro /= data_size
    f1_micro /= data_size
    precision /= data_size
    recall /= data_size

    return accuracy, f1_macro, f1_micro, precision, recall, auc, ap


@torch.no_grad()
def compute_test(loader, verbose=False):
    model.eval()
    loss_test = 0.0
    out_log = []
    for data in loader:
        if not args.multi_gpu:
            data = data.to(args.device)
        out = model(data)
        if args.multi_gpu:
            y = torch.cat([d.y.unsqueeze(0) for d in data]) \
                .squeeze().to(out.device)
        else:
            y = data.y
        if verbose:
            print(F.softmax(out, dim=1).cpu().numpy())
        out_log.append([F.softmax(out, dim=1), y])
        loss_test += F.nll_loss(out, y).item()
    return eval_deep(out_log, loader), loss_test


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='specify cuda devices')

# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact',
                    help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden layer size')
parser.add_argument('--epochs', type=int, default=60,
                    help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=True,
                    help='whether concat news embedding and graph embedding')
parser.add_argument('--multi_gpu', type=bool, default=False,
                    help='multi-gpu mode')
parser.add_argument('--benchmark', type=bool, default=True,
                    help='whether using fixed data split')
parser.add_argument('--feature', type=str, default='spacy',
                    help='feature type, [profile, spacy, bert, content]')
parser.add_argument('--model', type=str, default='gcn',
                    help='model type, [gcn, gat, sage]')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = UPFD(root='data', feature=args.feature, name=args.dataset,
               transform=ToUndirected())

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

if args.benchmark:
    training_set = Subset(dataset, dataset.train_idx.numpy().tolist())
    validation_set = Subset(dataset, dataset.val_idx.numpy().tolist())
    test_set = Subset(dataset, dataset.test_idx.numpy().tolist())
else:
    num_training = int(len(dataset) * 0.2)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = \
        random_split(dataset, [num_training, num_val, num_test])

if args.multi_gpu:
    loader = DataListLoader
else:
    loader = DataLoader

train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)

model = Model(args, concat=args.concat)
if args.multi_gpu:
    model = DataParallel(model)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)

if __name__ == '__main__':

    t = time.time()
    model.train()
    for epoch in tqdm(range(args.epochs)):
        loss_train = 0.0
        out_log = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if not args.multi_gpu:
                data = data.to(args.device)
            out = model(data)
            if args.multi_gpu:
                y = torch.cat([d.y.unsqueeze(0) for d in data]) \
                    .squeeze().to(out.device)
            else:
                y = data.y
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), y])
        acc_train, _, _, _, recall_train, auc_train, _ = \
            eval_deep(out_log, train_loader)
        [acc_val, _, _, _, recall_val, auc_val, _], loss_val = \
            compute_test(val_loader)
        print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
              f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
              f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
              f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

    [acc_test, f1_macro_test, f1_micro_test, precision_test, recall_test,
     auc_test, ap_test], test_loss = compute_test(test_loader, verbose=False)
    print(f'Test set results: acc: {acc_test:.4f},'
          f'f1_macro: {f1_macro_test:.4f}, f1_micro: {f1_micro_test:.4f}, '
          f'precision: {precision_test:.4f}, recall: {recall_test:.4f},'
          f' auc: {auc_test:.4f}, ap: {ap_test:.4f}')
