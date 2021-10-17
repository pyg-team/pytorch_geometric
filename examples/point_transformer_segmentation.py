from point_transformer_classification import transition_down, TransformerBlock

import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN, ReLU

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import intersection_and_union as i_and_u
from torch_geometric.nn.unpool import knn_interpolate

from torch_cluster import knn_graph

category = 'Airplane'  # Pass in `None` to train on all categories.
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ShapeNet')
transform = T.Compose([
    T.RandomTranslate(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2),
])
pre_transform = T.NormalizeScale()
train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
                         pre_transform=pre_transform)
test_dataset = ShapeNet(path, category, split='test',
                        pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


def transition_up(x_sub, pos_sub, mlp_sub, x, pos, mlp, batch_sub=None, batch=None):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''

    # transform low-res features
    x_sub = mlp_sub(x_sub)

    # interpolate low-res feats to high-res points
    x_interpolated = knn_interpolate(
        x_sub, pos_sub, pos, k=3, batch_x=batch_sub, batch_y=batch)

    x = mlp(x) + x_interpolated

    return x, pos


class Net(torch.nn.Module):
    def __init__(self, NUM_CLASSES, NUM_FEATURES, dim_model, k=16):
        super(Net, self).__init__()
        self.k = k

        # first block
        self.mlp_input = Seq(
            Lin(NUM_FEATURES, 32),
            BN(32),
            ReLU()
        )

        # backbone layers
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.mlp_down = torch.nn.ModuleList()
        self.mlp_up_sub = torch.nn.ModuleList()
        self.mlp_up = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):

            # Add Point Transformer block followed by a Transition Down block
            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i],
                                 out_channels=dim_model[i + 1],
                                 hidden_channels=dim_model[i])
            )
            self.mlp_down.append(Seq(
                Lin(dim_model[i + 1], dim_model[i + 1]),
                BN(dim_model[i + 1]),
                ReLU())
            )

            # Add Transition Up block followed by Point Transformer block
            self.mlp_up_sub.append(Seq(
                Lin(dim_model[i], dim_model[i]),
                BN(dim_model[i]),
                ReLU())
            )

            self.mlp_up.append(Seq(
                Lin(dim_model[i], dim_model[i]),
                BN(dim_model[i]),
                ReLU())
            )

            self.transformers_up.append(TransformerBlock(
                in_channels=dim_model[i + 1],
                out_channels=dim_model[i],
                hidden_channels=dim_model[i + 1]))

        # summit layers
        self.transformer = TransformerBlock(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
            hidden_channels=dim_model[-1]
        )

        self.mlp_summit = Seq(
            Lin(dim_model[-1], dim_model[-1]),
            BN(dim_model[-1]),
            ReLU()
        )

        # end block
        self.end_transformer = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
            hidden_channels=dim_model[0]
        )

        # class score computation
        self.lin = Seq(Lin(dim_model[0], 64),
                       ReLU(),
                       Lin(64, 64),
                       ReLU(),
                       Lin(64, NUM_CLASSES))

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.transformers_down)):
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)
            x, pos, batch = transition_down(
                x, pos, self.mlp_down[i], batch=batch, k=self.k)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        # summit
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer(x, pos, edge_index)
        x = self.mlp_summit(x)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transformers_down)
        for i in range(n):
            edge_index = knn_graph(pos, k=self.k, batch=out_batch[- i - 1])
            x = self.transformers_up[- i - 1](x, pos, edge_index)
            x, pos = transition_up(x,
                                   pos,
                                   self.mlp_up_sub[- i - 1],
                                   out_x[- i - 2],
                                   out_pos[- i - 2],
                                   self.mlp_up[- i - 1],
                                   batch_sub=out_batch[- i - 1],
                                   batch=out_batch[- i - 2])

        # end transformer
        edge_index = knn_graph(pos, k=self.k, batch=out_batch[0])
        x = self.end_transformer(x, pos, edge_index)

        # Class score
        out = self.lin(x)

        return F.log_softmax(out, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_classes, 3, dim_model=[
    32, 64, 128, 256, 512], k=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f}')
            total_loss = correct_nodes = total_nodes = 0


@torch.no_grad()
def test(loader):
    model.eval()

    y_mask = loader.dataset.y_mask
    ious = [[] for _ in range(len(loader.dataset.categories))]

    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1)

        i, u = i_and_u(pred, data.y, loader.dataset.num_classes, data.batch)
        iou = i.cpu().to(torch.float) / u.cpu().to(torch.float)
        iou[torch.isnan(iou)] = 1

        # Find and filter the relevant classes for each category.
        for iou, category in zip(iou.unbind(), data.category.unbind()):
            ious[category.item()].append(iou[y_mask[category]])

    # Compute mean IoU.
    ious = [torch.stack(iou).mean(0).mean(0) for iou in ious]
    return torch.tensor(ious).mean().item()


for epoch in range(1, 100):
    train()
    iou = test(test_loader)
    print('Epoch: {:02d}, Test IoU: {:.4f}'.format(epoch, iou))
    scheduler.step()
