import os.path as osp

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy

import torch_geometric.transforms as T
from torch_geometric import seed_everything
from torch_geometric.data import LightningDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GIN, MLP, global_add_pool


class Model(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.gnn = GIN(in_channels, hidden_channels, num_layers,
                       dropout=dropout, jk='cat')

        self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                              batch_norm=True, dropout=dropout)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.classifier(x)
        return x

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(y_hat, data.y)
        self.train_acc(y_hat.softmax(dim=-1), data.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        self.val_acc(y_hat.softmax(dim=-1), data.y)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        self.test_acc(y_hat.softmax(dim=-1), data.y)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))

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
                                  batch_size=64, num_workers=4)

    model = Model(dataset.num_node_features, dataset.num_classes)

    devices = torch.cuda.device_count()
    strategy = pl.strategies.DDPSpawnStrategy(find_unused_parameters=False)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1)
    trainer = pl.Trainer(strategy=strategy, accelerator='gpu', devices=devices,
                         max_epochs=50, log_every_n_steps=5,
                         callbacks=[checkpoint])

    trainer.fit(model, datamodule)
    trainer.test(ckpt_path='best', datamodule=datamodule)


if __name__ == '__main__':
    main()
