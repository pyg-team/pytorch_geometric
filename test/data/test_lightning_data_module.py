import pytest

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import LightningDataset

try:
    from pytorch_lightning import LightningModule
    no_pytorch_lightning = False
except (ImportError, ModuleNotFoundError):
    LightningModule = torch.nn.Module
    no_pytorch_lightning = True


class LinearModule(LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        from torchmetrics import Accuracy

        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


@pytest.mark.skipif(no_pytorch_lightning, reason='PL not available')
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_lightning_dataset():
    import pytorch_lightning as pl

    dataset = TUDataset('/tmp/TUDataset', name='MUTAG').shuffle()
    train_dataset = dataset[:50]
    val_dataset = dataset[50:60]
    test_dataset = dataset[60:70]

    data_module = LightningDataset(train_dataset, val_dataset, test_dataset,
                                   batch_size=5, num_workers=2)
    model = LinearModule(dataset.num_features, 64, dataset.num_classes)

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=1,
        log_every_n_steps=1,
        strategy=pl.plugins.DDPSpawnPlugin(find_unused_parameters=False),
    )
    trainer.fit(model, data_module)
