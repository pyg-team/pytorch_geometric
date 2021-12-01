import sys
import random
import shutil
import os.path as osp
import pytest

import torch
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset, Planetoid, DBLP
from torch_geometric.data import LightningDataset, LightningNodeData

try:
    from pytorch_lightning import LightningModule
    no_pytorch_lightning = False
except (ImportError, ModuleNotFoundError):
    LightningModule = torch.nn.Module
    no_pytorch_lightning = True


class LinearGraphModule(LightningModule):
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


class LinearNodeModule(LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        from torchmetrics import Accuracy

        self.lin = torch.nn.Linear(in_channels, out_channels)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.lin(x)

    def training_step(self, data, batch_idx):
        y_hat = self(data.x)[data.train_mask]
        y = data.y[data.train_mask]
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log('loss', loss, batch_size=y.size(0))
        self.log('train_acc', self.train_acc, batch_size=y.size(0))
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x)[data.val_mask]
        y = data.y[data.val_mask]
        self.val_acc(y_hat.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, batch_size=y.size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


@pytest.mark.skipif(no_pytorch_lightning, reason='PL not available')
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_lightning_dataset():
    return
    import pytorch_lightning as pl

    root = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    dataset = TUDataset(root, name='MUTAG').shuffle()
    train_dataset = dataset[:50]
    val_dataset = dataset[50:60]
    test_dataset = dataset[60:70]
    shutil.rmtree(root)

    model = LinearGraphModule(dataset.num_features, 64, dataset.num_classes)

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=1,
        log_every_n_steps=1,
        strategy=pl.plugins.DDPSpawnPlugin(find_unused_parameters=False),
    )

    data_module = LightningDataset(train_dataset, val_dataset, test_dataset,
                                   batch_size=5, num_workers=2)
    assert str(data_module) == ('LightningDataset(train_dataset=MUTAG(50), '
                                'val_dataset=MUTAG(10), '
                                'test_dataset=MUTAG(10), batch_size=5, '
                                'num_workers=2, pin_memory=True, '
                                'persistent_workers=True)')
    trainer.fit(model, data_module)
    assert trainer._data_connector._val_dataloader_source.is_defined()
    assert trainer._data_connector._test_dataloader_source.is_defined()

    data_module = LightningDataset(train_dataset, batch_size=5, num_workers=2)
    assert str(data_module) == ('LightningDataset(train_dataset=MUTAG(50), '
                                'batch_size=5, num_workers=2, '
                                'pin_memory=True, persistent_workers=True)')
    trainer.fit(model, data_module)
    assert not trainer._data_connector._val_dataloader_source.is_defined()
    assert not trainer._data_connector._test_dataloader_source.is_defined()


@pytest.mark.skipif(no_pytorch_lightning, reason='PL not available')
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('loader', ['full'])
def test_lightning_node_data(loader):
    import pytorch_lightning as pl
    print('==================================')
    print('loader', loader)

    # root = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    root = '/tmp/dawdhhaiuad'
    dataset = Planetoid(root, name='Cora')
    data = dataset[0]
    data_repr = ('Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], '
                 'train_mask=[2708], val_mask=[2708], test_mask=[2708])')
    # shutil.rmtree(root)

    model = LinearNodeModule(dataset.num_features, dataset.num_classes)

    trainer = pl.Trainer(
        gpus=1 if loader == 'full' else torch.cuda.device_count(),
        max_epochs=1,
        log_every_n_steps=1,
        strategy=pl.plugins.DDPSpawnPlugin(find_unused_parameters=False),
    )

    data_module = LightningNodeData(data, loader=loader)
    print(data_module)
    assert str(data_module) == (f'LightningDataset(data={data_repr}, '
                                f'loader={loader}, batch_size=1, '
                                f'num_workers=0, pin_memory=False, '
                                f'persistent_workers=False)')

    trainer.fit(model, data_module)

    print(data.x.device)


# @pytest.mark.skipif(no_pytorch_lightning, reason='PL not available')
# @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
# def test_lightning_hetero_node_data():
#     import pytorch_lightning as pl

#     # root = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
#     root = '/tmp/dawdhhaiuad2323'
#     dataset = DBLP(root)
#     data = dataset[0]
#     # shutil.rmtree(root)

#     data_module = LightningNodeData(data, batch_size=5, num_workers=2)
#     # print(data_module)
