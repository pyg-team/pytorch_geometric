import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics import Accuracy

from torch_geometric.nn import global_mean_pool
from torch_geometric.data import LightningDataset
from torch_geometric.datasets import Planetoid, TUDataset


class LinearModule(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x, batch):
        x = self.lin1(x).relu()
        x = global_mean_pool(x, batch)
        x = self.lin2(x)
        return x

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.batch)
        loss = F.cross_entropy(y_hat, data.y)
        self.train_acc(y_hat.softmax(dim=-1), data.y)
        self.log('loss', loss, batch_size=data.num_graphs)
        self.log('train_acc', self.train_acc, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.batch)
        self.val_acc(y_hat.softmax(dim=-1), data.y)
        self.log('val_acc', self.val_acc, batch_size=data.num_graphs)

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.batch)
        self.test_acc(y_hat.softmax(dim=-1), data.y)
        self.log('test_acc', self.test_acc, batch_size=data.num_graphs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def test_lightning_dataset():
    dataset = Planetoid('/tmp/Planetoid', name='Cora')
    # data_module = LightningDataModule(dataset, loader='NeighborLoader')

    dataset = TUDataset('/tmp/TUDataset', name='MUTAG').shuffle()
    train_dataset = dataset[:50]
    val_dataset = dataset[50:100]
    test_dataset = dataset[100:150]

    data_module = LightningDataset(train_dataset, val_dataset, test_dataset,
                                   num_workers=2, persistent_workers=True)
    model = LinearModule(dataset.num_features, 64, dataset.num_classes)

    trainer = pl.Trainer(
        gpus=2,
        max_epochs=5,
        strategy='ddp_spawn',
        # strategy=pl.plugins.DDPSpawnPlugin(find_unused_parameters=False),
    )
    trainer.fit(model, data_module)

    pass
