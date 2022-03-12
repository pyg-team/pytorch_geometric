import math
import warnings

import pytest
import torch
import torch.nn.functional as F

from torch_geometric.data import LightningDataset, LightningNodeData
from torch_geometric.nn import global_mean_pool

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
        # Basic test to ensure that the dataset is not replicated:
        self.trainer.datamodule.train_dataset.data.x.add_(1)

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
@pytest.mark.parametrize('strategy', [None, 'ddp_spawn'])
def test_lightning_dataset(get_dataset, strategy):
    import pytorch_lightning as pl

    dataset = get_dataset(name='MUTAG').shuffle()
    train_dataset = dataset[:50]
    val_dataset = dataset[50:80]
    test_dataset = dataset[80:90]

    gpus = 1 if strategy is None else torch.cuda.device_count()
    if strategy == 'ddp_spawn':
        strategy = pl.plugins.DDPSpawnPlugin(find_unused_parameters=False)

    model = LinearGraphModule(dataset.num_features, 64, dataset.num_classes)

    trainer = pl.Trainer(strategy=strategy, gpus=gpus, max_epochs=1,
                         log_every_n_steps=1)
    datamodule = LightningDataset(train_dataset, val_dataset, test_dataset,
                                  batch_size=5, num_workers=3)
    old_x = train_dataset.data.x.clone()
    assert str(datamodule) == ('LightningDataset(train_dataset=MUTAG(50), '
                               'val_dataset=MUTAG(30), '
                               'test_dataset=MUTAG(10), batch_size=5, '
                               'num_workers=3, pin_memory=True, '
                               'persistent_workers=True)')
    trainer.fit(model, datamodule)
    new_x = train_dataset.data.x
    offset = 10 + 6 + 2 * gpus  # `train_steps` + `val_steps` + `sanity`
    assert torch.all(new_x > (old_x + offset - 4))  # Ensure shared data.
    assert trainer._data_connector._val_dataloader_source.is_defined()
    assert trainer._data_connector._test_dataloader_source.is_defined()

    # Test with `val_dataset=None` and `test_dataset=None`:
    warnings.filterwarnings('ignore', '.*Skipping val loop.*')
    trainer = pl.Trainer(strategy=strategy, gpus=gpus, max_epochs=1,
                         log_every_n_steps=1)
    datamodule = LightningDataset(train_dataset, batch_size=5, num_workers=3)
    assert str(datamodule) == ('LightningDataset(train_dataset=MUTAG(50), '
                               'batch_size=5, num_workers=3, '
                               'pin_memory=True, persistent_workers=True)')
    trainer.fit(model, datamodule)
    assert not trainer._data_connector._val_dataloader_source.is_defined()
    assert not trainer._data_connector._test_dataloader_source.is_defined()


class LinearNodeModule(LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        from torchmetrics import Accuracy

        self.lin = torch.nn.Linear(in_channels, out_channels)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        # Basic test to ensure that the dataset is not replicated:
        self.trainer.datamodule.data.x.add_(1)

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
@pytest.mark.parametrize('loader', ['full', 'neighbor'])
@pytest.mark.parametrize('strategy', [None, 'ddp_spawn'])
def test_lightning_node_data(get_dataset, strategy, loader):
    import pytorch_lightning as pl

    dataset = get_dataset(name='Cora')
    data = dataset[0]
    data_repr = ('Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], '
                 'train_mask=[2708], val_mask=[2708], test_mask=[2708])')

    model = LinearNodeModule(dataset.num_features, dataset.num_classes)

    if strategy is None or loader == 'full':
        gpus = 1
    else:
        gpus = torch.cuda.device_count()

    if strategy == 'ddp_spawn' and loader == 'full':
        data = data.cuda()  # This is necessary to test sharing of data.

    if strategy == 'ddp_spawn':
        strategy = pl.plugins.DDPSpawnPlugin(find_unused_parameters=False)

    batch_size = 1 if loader == 'full' else 32
    num_workers = 0 if loader == 'full' else 3
    kwargs, kwargs_repr = {}, ''
    if loader == 'neighbor':
        kwargs['num_neighbors'] = [5]
        kwargs_repr += 'num_neighbors=[5], '

    trainer = pl.Trainer(strategy=strategy, gpus=gpus, max_epochs=5,
                         log_every_n_steps=1)
    datamodule = LightningNodeData(data, loader=loader, batch_size=batch_size,
                                   num_workers=num_workers, **kwargs)
    old_x = data.x.clone().cpu()
    assert str(datamodule) == (f'LightningNodeData(data={data_repr}, '
                               f'loader={loader}, batch_size={batch_size}, '
                               f'num_workers={num_workers}, {kwargs_repr}'
                               f'pin_memory={loader != "full"}, '
                               f'persistent_workers={loader != "full"})')
    trainer.fit(model, datamodule)
    new_x = data.x.cpu()
    if loader == 'full':
        offset = 5 + 5 + 1  # `train_steps` + `val_steps` + `sanity`
    else:
        offset = 0
        offset += gpus * 2  # `sanity`
        offset += 5 * gpus * math.ceil(140 / (gpus * batch_size))  # `train`
        offset += 5 * gpus * math.ceil(500 / (gpus * batch_size))  # `val`
    assert torch.all(new_x > (old_x + offset - 4))  # Ensure shared data.
    assert trainer._data_connector._val_dataloader_source.is_defined()
    assert trainer._data_connector._test_dataloader_source.is_defined()


class LinearHeteroNodeModule(LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        from torchmetrics import Accuracy

        self.lin = torch.nn.Linear(in_channels, out_channels)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        # Basic test to ensure that the dataset is not replicated:
        self.trainer.datamodule.data['author'].x.add_(1)

        return self.lin(x)

    def training_step(self, data, batch_idx):
        y_hat = self(data['author'].x)[data['author'].train_mask]
        y = data['author'].y[data['author'].train_mask]
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log('loss', loss, batch_size=y.size(0))
        self.log('train_acc', self.train_acc, batch_size=y.size(0))
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data['author'].x)[data['author'].val_mask]
        y = data['author'].y[data['author'].val_mask]
        self.val_acc(y_hat.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, batch_size=y.size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


@pytest.mark.skipif(no_pytorch_lightning, reason='PL not available')
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_lightning_hetero_node_data(get_dataset):
    import pytorch_lightning as pl

    dataset = get_dataset(name='DBLP')
    data = dataset[0]

    model = LinearHeteroNodeModule(data['author'].num_features,
                                   int(data['author'].y.max()) + 1)

    gpus = torch.cuda.device_count()
    strategy = pl.plugins.DDPSpawnPlugin(find_unused_parameters=False)

    trainer = pl.Trainer(strategy=strategy, gpus=gpus, max_epochs=5,
                         log_every_n_steps=1)
    datamodule = LightningNodeData(data, loader='neighbor', num_neighbors=[5],
                                   batch_size=32, num_workers=3)
    old_x = data['author'].x.clone()
    trainer.fit(model, datamodule)
    new_x = data['author'].x
    offset = 0
    offset += gpus * 2  # `sanity`
    offset += 5 * gpus * math.ceil(400 / (gpus * 32))  # `train`
    offset += 5 * gpus * math.ceil(400 / (gpus * 32))  # `val`
    assert torch.all(new_x > (old_x + offset - 4))  # Ensure shared data.
    assert trainer._data_connector._val_dataloader_source.is_defined()
    assert trainer._data_connector._test_dataloader_source.is_defined()
