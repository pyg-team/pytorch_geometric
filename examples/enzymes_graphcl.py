import argparse
import os.path as osp

import torch
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

import numpy as np
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
#from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_scatter import segment_csr

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.seed import seed_everything


from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, adj_t, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, adj_t)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [segment_csr(z, batch,reduce=args.global_pool) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, adj_t, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, adj_t)
        x2, edge_index2, edge_weight2 = aug2(x, adj_t)
        z, g = self.encoder(x, adj_t, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.adj_t, data.ptr)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

@torch.no_grad()
def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.adj_t, data.ptr)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0).cpu()
    y = torch.cat(y, dim=0).cpu()

    return evaluate_graph_embeddings_using_svm(x,y)

    
def evaluate_graph_embeddings_using_svm(embeddings, labels):
    micro_f1_result = []
    macro_f1_result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in tqdm(kf.split(embeddings, labels), total=kf.get_n_splits(), desc="(E) 10-folds"):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}

        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params, n_jobs=8)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)

        micro_f1 = f1_score(y_test, preds, average="micro")
        macro_f1 = f1_score(y_test, preds, average="macro")
        micro_f1_result.append(micro_f1)
        macro_f1_result.append(macro_f1)
    test_micro_f1 = np.mean(micro_f1_result)
    test_micro_std = np.std(micro_f1_result)
    test_macro_f1 = np.mean(macro_f1_result)
    test_macro_std = np.std(macro_f1_result)

    return test_micro_f1, test_micro_std, test_macro_f1,test_macro_std

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GCL')
    parser.add_argument('--dataset', type=str, default='ENZYMES')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--aug1', type=str, help="aug1")
    parser.add_argument('--aug2', type=str, help="aug2")
    parser.add_argument('--global_pool', type=str, default="sum")
    parser.add_argument('--wandb', action='store_true', help='Track experiment',default='True')
    
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = osp.join('data', 'TU')
    transform = T.Compose([T.RemoveIsolatedNodes(),T.ToUndirected(), T.AddSelfLoops(), T.LocalDegreeProfile(), T.NormalizeFeatures(), T.ToSparseTensor()])
    dataset = TUDataset(path, name=args.dataset, pre_transform=transform)

    train_loader = DataLoader(dataset, args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset, args.batch_size, shuffle=False)
    
    ## different combination of augmentation

    # 10-folds (R) Micro-F1:0.4100±0.0507, Macro-F1:0.4120±0.0462
    # aug1 = A.Identity()
    # aug2 = A.Identity()

    # aug1 = A.EdgeRemoving(pe=0.1)
    # aug2 = A.NodeDropping(pn=0.1)

    # aug1 = A.FeatureMasking(pf=0.1)
    # aug2 = A.NodeDropping(pn=0.1)

    # best 10-folds (R) Micro-F1:0.4317±0.0603, Macro-F1:0.4260±0.0656
    aug1 = A.FeatureMasking(pf=0.1)
    aug2 = A.EdgeRemoving(pe=0.1)

    # aug1 = A.RWSampling(num_seeds=1000, walk_length=10)
    # aug2 = A.NodeDropping(pn=0.1)

    # aug1 = A.RWSampling(num_seeds=1000, walk_length=10)
    # aug2 = A.FeatureMasking(pf=0.1)



    gconv = GConv(input_dim=dataset.num_features, hidden_dim=args.hidden_channels, num_layers=args.num_layers).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    epoch_iter = tqdm(range(args.epochs))
    for epoch in epoch_iter:
        loss = train(encoder_model, contrast_model, train_loader, optimizer)
        epoch_iter.set_description(f"(T) Epoch {epoch+1}, Train Loss: {loss:.4f}")

    test_micro_f1, test_micro_std, test_macro_f1,test_macro_std = test(encoder_model, val_loader)
    print(f'(R) Micro-F1:{test_micro_f1:.4f}±{test_micro_std:.4f}, Macro-F1:{test_macro_f1:.4f}±{test_macro_std:.4f}')