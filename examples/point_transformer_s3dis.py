import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN, ReLU
from torch.nn import Identity

import torch_geometric.transforms as T
from torch_geometric.datasets import S3DIS
from torch_geometric.loader import DataLoader
from torch_geometric.utils import intersection_and_union as i_and_u
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn.pool import knn
from torch_geometric.nn.conv import PointTransformerConv


from torch_cluster import knn_graph
from torch_cluster import fps
from torch_scatter import scatter_max


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], batch_norm=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], batch_norm=False)

        self.transformer = PointTransformerConv(
            in_channels,
            out_channels,
            pos_nn=self.pos_nn,
            attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''

    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels])

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k,
                            batch_x=batch, batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x[id_k_neighbor[1]],
                               id_k_neighbor[0],
                               dim_size=id_clusters.size(0),
                               dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(
            Lin(channels[i - 1], channels[i]),
            BN(channels[i]) if batch_norm else Identity(),
            ReLU()
        )
        for i in range(1, len(channels))
    ])

class TransitionUp(torch.nn.Module):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels])
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                         batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]])

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )

        # backbone layers
        self.encoders = torch.nn.ModuleList()
        n = len(dim_model) - 1
        for i in range(0, n):

            # Add Transition Down block followed by a Point Transformer block
            self.encoders.append(Seq(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k),
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))
            )
        
        # summit layers
        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], batch_norm=False)

        self.transformer_summit = TransformerBlock(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
        )
        
        self.decoders = torch.nn.ModuleList()
        for i in range(0, n):
            # Add Transition Up block followed by Point Transformer block
            self.decoders.append(Seq(
                TransitionUp(in_channels=dim_model[n - i],
                              out_channels=dim_model[n - i -1]),
                TransformerBlock(in_channels=dim_model[n - i -1],
                                  out_channels=dim_model[n - i -1]))
            )

        # class score computation
        self.mlp_output = Seq(Lin(dim_model[0], 64), ReLU(), Lin(64, 64),
                              ReLU(), Lin(64, out_channels))

    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1)).to(pos.get_device())

        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.encoders)):
            x, pos, batch = self.encoders[i][0](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.encoders[i][1](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.encoders)
        for i in range(n):
            x = self.decoders[i][0](x=out_x[-i - 2], x_sub=x,
                                           pos=out_pos[-i - 2],
                                           pos_sub=out_pos[-i - 1],
                                           batch_sub=out_batch[-i - 1],
                                           batch=out_batch[-i - 2])

            edge_index = knn_graph(out_pos[-i - 2], k=self.k,
                                   batch=out_batch[-i - 2])
            x = self.decoders[i][1](x, out_pos[-i - 2], edge_index)

        # Class score
        out = self.mlp_output(x)

        return F.log_softmax(out, dim=-1)


def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loss_history.append(total_loss)
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 100 == 0:
            print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f}')
            total_loss = correct_nodes = total_nodes = 0


@torch.no_grad()
def test(loader):
    model.eval()

    preds = []  #to compute IoU on the full dataset instead of a per-batch basis
    labels = [] # we will stack the predictions and labels
    for i,data in enumerate(loader):
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).argmax(dim=1)
        preds.append(pred)
        labels.append(data.y)


    preds_tensor = torch.hstack(preds).type(torch.LongTensor).cpu()
    labels_tensor = torch.hstack(labels).type(torch.LongTensor).cpu()
    
    i, u = torch.zeros((13)), torch.zeros((13))
    # intersection_and_union does one-hot encoding, making the full labels
    # matrix too large to fit all at once so we do it in two times
    
    i_sub, u_sub = i_and_u(preds_tensor[:2000000], labels_tensor[:2000000], 13)#, data.batch)
    i += i_sub
    u += u_sub
    
    i_sub, u_sub = i_and_u(preds_tensor[2000000:], labels_tensor[2000000:], 13)#, data.batch)
    i += i_sub
    u += u_sub
    
    iou = i / u
    iou = iou[ torch.isnan(iou) == False].mean()
    # Compute mean IoU.
    return iou


if __name__ == '__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'S3DIS')
    print(path)
    transform = T.Compose([
        T.RandomTranslate(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2),
    ])
    pre_transform = T.NormalizeScale()
    train_dataset = S3DIS(path, test_area=5, train=True, transform=transform,
                             pre_transform=pre_transform)
    test_dataset = S3DIS(path, test_area=5, train=False,
                            pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(6, train_dataset.num_classes, dim_model=[
        32, 64, 128, 256, 512], k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    
    loss_history = []
    test_iou_history = []
    for epoch in range(1, 100):
        train()
        iou = test(test_loader)
        test_iou_history.append(iou)
        print('Epoch: {:02d}, Test IoU: {:.4f}'.format(epoch, iou))
        scheduler.step()
    
    
    
    
    
    
