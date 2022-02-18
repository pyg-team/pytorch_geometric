import os.path as osp

import torch
import torch.nn.functional as F
from point_transformer_classification import (MLP, TransformerBlock,
                                              TransitionDown)
from point_transformer_segmentation import TransitionSummit, TransitionUp
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq
from torch_cluster import knn_graph

import torch_geometric.transforms as T
from torch_geometric.datasets import S3DIS
from torch_geometric.loader import DataLoader
from torch_geometric.utils import intersection_and_union as i_and_u


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], bias=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )

        blocks = [1, 2, 3, 5, 2]

        # backbone layers
        self.encoders = torch.nn.ModuleList()
        n = len(dim_model) - 1
        for i in range(0, n):

            # Add Transition Down block followed by a Point Transformer block
            self.encoders.append(
                Seq(
                    TransitionDown(in_channels=dim_model[i],
                                   out_channels=dim_model[i + 1], k=self.k),
                    *[
                        TransformerBlock(in_channels=dim_model[i + 1],
                                         out_channels=dim_model[i + 1])
                        for k in range(blocks[1:][i])
                    ]))

        # summit layers
        self.mlp_summit = TransitionSummit(dim_model[-1])

        self.transformer_summit = Seq(*[
            TransformerBlock(
                in_channels=dim_model[-1],
                out_channels=dim_model[-1],
            ) for i in range(1)
        ])

        self.decoders = torch.nn.ModuleList()
        for i in range(0, n):
            # Add Transition Up block followed by Point Transformer block
            self.decoders.append(
                Seq(
                    TransitionUp(in_channels=dim_model[n - i],
                                 out_channels=dim_model[n - i - 1]),
                    *[
                        TransformerBlock(in_channels=dim_model[n - i - 1],
                                         out_channels=dim_model[n - i - 1])
                        for k in range(1)
                    ]))

        # class score computation
        self.mlp_output = Seq(MLP([dim_model[0], dim_model[0]]),
                              Lin(dim_model[0], out_channels))

    def forward(self, x, pos, batch=None):
        # add dummy features in case there is none
        x = pos if x is None else torch.cat((pos, x), 1)

        out_x = []
        out_pos = []
        out_batch = []
        edges_index = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)
        edges_index.append(edge_index)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.encoders)):

            x, pos, batch = self.encoders[i][0](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
            for layer in self.encoders[i][1:]:
                x = layer(x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)
            edges_index.append(edge_index)

        # summit
        x = self.mlp_summit(x, batch=batch)
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        for layer in self.transformer_summit:
            x = layer(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.encoders)
        for i in range(n):
            x = self.decoders[i][0](x=out_x[-i - 2], x_sub=x,
                                    pos=out_pos[-i - 2],
                                    pos_sub=out_pos[-i - 1],
                                    batch_sub=out_batch[-i - 1],
                                    batch=out_batch[-i - 2])

            edge_index = edges_index[-i - 2]

            for layer in self.decoders[i][1:]:
                x = layer(x, out_pos[-i - 2], edge_index)

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

    preds = [
    ]  # to compute IoU on the full dataset instead of a per-batch basis
    labels = []  # we will stack the predictions and labels
    for i, data in enumerate(loader):
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).argmax(dim=1)
        preds.append(pred)
        labels.append(data.y)

    preds_tensor = torch.hstack(preds).type(torch.LongTensor).cpu()
    labels_tensor = torch.hstack(labels).type(torch.LongTensor).cpu()

    i, u = torch.zeros((13)), torch.zeros((13))
    # intersection_and_union does one-hot encoding, making the full labels
    # matrix too large to fit all at once so we do it in two times

    i_sub, u_sub = i_and_u(preds_tensor[:2000000], labels_tensor[:2000000], 13)
    i += i_sub
    u += u_sub

    i_sub, u_sub = i_and_u(preds_tensor[2000000:], labels_tensor[2000000:], 13)
    i += i_sub
    u += u_sub

    iou = i / u
    iou = iou[~torch.isnan(iou)].mean()
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
    print('done')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(6 + 3, train_dataset.num_classes,
                dim_model=[32, 64, 128, 256, 512], k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40,
                                                gamma=0.1)

    loss_history = []
    test_iou_history = []
    for epoch in range(1, 100):
        train()
        iou = test(test_loader)
        test_iou_history.append(iou)
        print('Epoch: {:02d}, Test IoU: {:.4f}'.format(epoch, iou))
        scheduler.step()
