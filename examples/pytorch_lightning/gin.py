import os.path as osp

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torchmetrics import Accuracy
import pytorch_lightning as pl

import torch_geometric.transforms as T
from torch_geometric.nn import global_add_pool
from torch_geometric.data import LightningDataset
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset


class Model(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()

        self.convs = ModuleList()
        for _ in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(inplace=True),
                Linear(2 * hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(inplace=True),
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            in_channels = hidden_channels

        self.classifier = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=dropout),
            Linear(hidden_channels, out_channels),
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_add_pool(x, batch)
        return self.classifier(x)

    def training_step(self, data: Data, batch_idx: int):
        y_hat = self(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(y_hat, data.y)
        self.train_acc(y_hat.softmax(dim=-1), data.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=data.y.size(0))
        return loss

    def validation_step(self, data: Data, batch_idx: int):
        y_hat = self(data.x, data.edge_index, data.batch)
        self.val_acc(y_hat.softmax(dim=-1), data.y)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=data.y.size(0))

    def test_step(self, data: Data, batch_idx: int):
        y_hat = self(data.x, data.edge_index, data.batch)
        self.test_acc(y_hat.softmax(dim=-1), data.y)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=data.y.size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main():
    seed_everything(42)

    root = osp.join('data', 'TUDataset')
    dataset = TUDataset(root, 'IMDB-BINARY', pre_transform=T.OneHotDegree(135))

    dataset = dataset.shuffle()
    test_dataset = dataset[:len(dataset) // 10]
    val_dataset = dataset[len(dataset) // 10:2 * len(dataset) // 10]
    train_dataset = dataset[2 * len(dataset) // 10:]

    datamodule = LightningDataset(train_dataset, val_dataset, test_dataset,
                                  batch_size=64, num_workers=1)

    model = Model(dataset.num_node_features, dataset.num_classes)

    gpus = torch.cuda.device_count()
    strategy = pl.plugins.DDPSpawnPlugin(find_unused_parameters=False)
    trainer = pl.Trainer(gpus=gpus, strategy=strategy, max_epochs=5,
                         log_every_n_steps=5)

    trainer.fit(model, datamodule)
    trainer.test(ckpt_path='best', datamodule=datamodule)


if __name__ == '__main__':
    main()
