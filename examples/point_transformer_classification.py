import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN, ReLU

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import knn
from torch_geometric.nn.conv import PointTransformerConv
from torch_geometric.nn import global_mean_pool

from torch_cluster import knn_graph
from torch_cluster import fps
from torch_scatter import scatter_max

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
# In its current state, PointTransformerConv does not support the absence of
# features, so we create dummy nodes features with the 'Constant' Transform
pre_transform, transform = T.NormalizeScale(), T.Compose(
    [T.SamplePoints(1024), T.Constant(1)])
train_dataset = ModelNet(path, '10', True, transform, pre_transform)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(TransformerBlock, self).__init__()
        self.lin_in = Lin(in_channels, hidden_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = Seq(
            Lin(3, 64),
            ReLU(),
            Lin(64, out_channels)
        )

        self.attn_nn = Seq(
            Lin(out_channels, 64),
            ReLU(),
            Lin(64, out_channels)
        )

        self.transformer = PointTransformerConv(
            in_channels, #TODO : fix hidden_channels
            out_channels,
            pos_nn=self.pos_nn,
            attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x_skip = x.clone()
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        x = x + x_skip
        return x


def transition_down(x, pos, mlp, batch=None, k=16):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''

    # FPS sampling
    id_clusters = fps(pos, ratio=0.25, batch=batch)

    # compute for each cluster the k nearest points
    sub_batch = batch[id_clusters] if batch is not None else None
    # beware of self loop
    id_k_neighbor = knn(pos, pos[id_clusters], k=k,
                        batch_x=batch, batch_y=sub_batch)

    # transformation of features through a simple MLP
    x = mlp(x)

    # Max pool onto each cluster the features from knn in points
    x_out, _ = scatter_max(x[id_k_neighbor[1]],
                           id_k_neighbor[0],
                           dim_size=id_clusters.size(0),
                           dim=0)

    # keep only the clusters and their max-pooled features
    new_pos, new_x = pos[id_clusters], x_out
    return new_x, new_pos, sub_batch


class Net(torch.nn.Module):
    def __init__(self, NUM_CLASSES, NUM_FEATURES, dim_model, k=16):
        super(Net, self).__init__()
        self.k = k

        # first block
        self.mlp_input = Seq(
            Lin(NUM_FEATURES, dim_model[0]),
            BN(dim_model[0]),
            ReLU()
        )
        
        self.transformer_input = TransformerBlock(in_channels=dim_model[0],
                                                  out_channels=dim_model[0],
                                                  hidden_channels=dim_model[0])

        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.mlp_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block 
            self.mlp_down.append(Seq(
                Lin(dim_model[i], dim_model[i + 1]),
                BN(dim_model[i + 1]),
                ReLU())
            )
            
            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1],
                                 hidden_channels=dim_model[i + 1])
            )

        # class score computation
        self.lin = Seq(Lin(dim_model[-1], 64),
                       ReLU(),
                       Lin(64, 64),
                       ReLU(),
                       Lin(64, NUM_CLASSES))

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos,edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = transition_down(
                x, pos, self.mlp_down[i], batch=batch, k=self.k)
            
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)
            

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.lin(x)

        return F.log_softmax(out, dim=-1)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(train_dataset.num_classes, 1, dim_model=[
                32, 64, 128, 256, 512], k=16).to(device)
    # NB : learning parameters are different from the paper
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=20,
                                                gamma=0.5)

    for epoch in range(1, 201):
        loss = train()
        test_acc = test(test_loader)
        print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
            epoch, loss, test_acc))
        scheduler.step()
