import math

import pytest
import torch
import torch.nn.functional as F

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.lightning import (
    LightningDataset,
    LightningLinkData,
    LightningNodeData,
)
from torch_geometric.nn import global_mean_pool
from torch_geometric.sampler import BaseSampler, NeighborSampler
from torch_geometric.testing import (
    MyFeatureStore,
    MyGraphStore,
    onlyCUDA,
    onlyFullTest,
    withPackage,
)

try:
    from pytorch_lightning import LightningModule
except ImportError:
    LightningModule = torch.nn.Module


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


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


@onlyCUDA
@onlyFullTest
@withPackage('pytorch_lightning')
@pytest.mark.parametrize('strategy_type', [None, 'ddp_spawn'])
def test_lightning_dataset(get_dataset, strategy_type):
    import pytorch_lightning as pl

    dataset = get_dataset(name='MUTAG').shuffle()
    train_dataset = dataset[:50]
    val_dataset = dataset[50:80]
    test_dataset = dataset[80:90]

    devices = 1 if strategy_type is None else torch.cuda.device_count()
    if strategy_type == 'ddp_spawn':
        strategy = pl.strategies.DDPSpawnStrategy(find_unused_parameters=False)
    else:
        strategy = None

    model = LinearGraphModule(dataset.num_features, 64, dataset.num_classes)

    trainer = pl.Trainer(strategy=strategy, accelerator='gpu', devices=devices,
                         max_epochs=1, log_every_n_steps=1)
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
    offset = 10 + 6 + 2 * devices  # `train_steps` + `val_steps` + `sanity`
    assert torch.all(new_x > (old_x + offset - 4))  # Ensure shared data.
    if strategy_type is None:
        assert trainer._data_connector._val_dataloader_source.is_defined()
        assert trainer._data_connector._test_dataloader_source.is_defined()

    # Test with `val_dataset=None` and `test_dataset=None`:
    if strategy_type == 'ddp_spawn':
        strategy = pl.strategies.DDPSpawnStrategy(find_unused_parameters=False)
    else:
        strategy = None

    trainer = pl.Trainer(strategy=strategy, accelerator='gpu', devices=devices,
                         max_epochs=1, log_every_n_steps=1)
    datamodule = LightningDataset(train_dataset, batch_size=5, num_workers=3)
    assert str(datamodule) == ('LightningDataset(train_dataset=MUTAG(50), '
                               'batch_size=5, num_workers=3, '
                               'pin_memory=True, persistent_workers=True)')
    trainer.fit(model, datamodule)
    if strategy_type is None:
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


@onlyCUDA
@onlyFullTest
@withPackage('pytorch_lightning')
@pytest.mark.parametrize('loader', ['full', 'neighbor'])
@pytest.mark.parametrize('strategy_type', [None, 'ddp_spawn'])
def test_lightning_node_data(get_dataset, strategy_type, loader):
    import pytorch_lightning as pl

    dataset = get_dataset(name='Cora')
    data = dataset[0]
    data_repr = ('Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], '
                 'train_mask=[2708], val_mask=[2708], test_mask=[2708])')

    model = LinearNodeModule(dataset.num_features, dataset.num_classes)

    if strategy_type is None or loader == 'full':
        devices = 1
    else:
        devices = torch.cuda.device_count()

    if strategy_type == 'ddp_spawn' and loader == 'full':
        data = data.cuda()  # This is necessary to test sharing of data.

    if strategy_type == 'ddp_spawn':
        strategy = pl.strategies.DDPSpawnStrategy(find_unused_parameters=False)
    else:
        strategy = None

    batch_size = 1 if loader == 'full' else 32
    num_workers = 0 if loader == 'full' else 3
    kwargs, kwargs_repr = {}, ''
    if loader == 'neighbor':
        kwargs['num_neighbors'] = [5]
        kwargs_repr += 'num_neighbors=[5], '

    trainer = pl.Trainer(strategy=strategy, accelerator='gpu', devices=devices,
                         max_epochs=5, log_every_n_steps=1)
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
        offset += devices * 2  # `sanity`
        offset += 5 * devices * math.ceil(140 / (devices * batch_size))
        offset += 5 * devices * math.ceil(500 / (devices * batch_size))
    assert torch.all(new_x > (old_x + offset - 4))  # Ensure shared data.
    if strategy_type is None:
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


@onlyCUDA
@onlyFullTest
@withPackage('pytorch_lightning')
def test_lightning_hetero_node_data(get_dataset):
    import pytorch_lightning as pl

    dataset = get_dataset(name='DBLP')
    data = dataset[0]

    model = LinearHeteroNodeModule(data['author'].num_features,
                                   int(data['author'].y.max()) + 1)

    devices = torch.cuda.device_count()
    strategy = pl.strategies.DDPSpawnStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(strategy=strategy, accelerator='gpu', devices=devices,
                         max_epochs=5, log_every_n_steps=1)
    datamodule = LightningNodeData(data, loader='neighbor', num_neighbors=[5],
                                   batch_size=32, num_workers=3)
    assert isinstance(datamodule.neighbor_sampler, NeighborSampler)
    old_x = data['author'].x.clone()
    trainer.fit(model, datamodule)
    new_x = data['author'].x
    offset = 0
    offset += devices * 2  # `sanity`
    offset += 5 * devices * math.ceil(400 / (devices * 32))  # `train`
    offset += 5 * devices * math.ceil(400 / (devices * 32))  # `val`
    assert torch.all(new_x > (old_x + offset - 4))  # Ensure shared data.


@withPackage('pytorch_lightning')
def test_lightning_data_custom_sampler():
    class DummySampler(BaseSampler):
        def sample_from_edges(self, *args, **kwargs):
            pass

        def sample_from_nodes(self, *args, **kwargs):
            pass

    data = Data(num_nodes=2, edge_index=torch.tensor([[0, 1], [1, 0]]))

    datamodule = LightningNodeData(data, node_sampler=DummySampler(),
                                   input_train_nodes=torch.arange(2))
    assert isinstance(datamodule.neighbor_sampler, DummySampler)

    datamodule = LightningLinkData(
        data, link_sampler=DummySampler(),
        input_train_edges=torch.tensor([[0, 1], [0, 1]]))
    assert isinstance(datamodule.neighbor_sampler, DummySampler)


@onlyCUDA
@onlyFullTest
@withPackage('pytorch_lightning')
def test_lightning_hetero_link_data():
    torch.manual_seed(12345)

    data = HeteroData()

    data['paper'].x = torch.arange(10)
    data['author'].x = torch.arange(10)
    data['term'].x = torch.arange(10)

    data['paper', 'author'].edge_index = get_edge_index(10, 10, 10)
    data['author', 'paper'].edge_index = get_edge_index(10, 10, 10)
    data['paper', 'term'].edge_index = get_edge_index(10, 10, 10)

    datamodule = LightningLinkData(
        data,
        input_train_edges=('author', 'paper'),
        loader='neighbor',
        num_neighbors=[5],
        batch_size=32,
        num_workers=0,
    )
    assert isinstance(datamodule.neighbor_sampler, NeighborSampler)
    for batch in datamodule.train_dataloader():
        assert 'edge_label_index' in batch['author', 'paper']

    data['author'].time = torch.arange(data['author'].num_nodes)
    data['paper'].time = torch.arange(data['paper'].num_nodes)
    data['term'].time = torch.arange(data['term'].num_nodes)

    datamodule = LightningLinkData(
        data,
        input_train_edges=('author', 'paper'),
        input_train_time=torch.arange(data['author', 'paper'].num_edges),
        loader='neighbor',
        num_neighbors=[5],
        batch_size=32,
        num_workers=0,
        time_attr='time',
    )

    for batch in datamodule.train_dataloader():
        assert 'edge_label_index' in batch['author', 'paper']
        assert 'edge_label_time' in batch['author', 'paper']


@withPackage('pytorch_lightning')
def test_lightning_hetero_link_data_custom_store():
    torch.manual_seed(12345)

    feature_store = MyFeatureStore()
    graph_store = MyGraphStore()

    x = torch.arange(10)
    feature_store.put_tensor(x, group_name='paper', attr_name='x', index=None)
    feature_store.put_tensor(x, group_name='author', attr_name='x', index=None)
    feature_store.put_tensor(x, group_name='term', attr_name='x', index=None)

    edge_index = get_edge_index(10, 10, 10)
    graph_store.put_edge_index(edge_index=(edge_index[0], edge_index[1]),
                               edge_type=('paper', 'to', 'author'),
                               layout='coo', size=(10, 10))
    graph_store.put_edge_index(edge_index=(edge_index[0], edge_index[1]),
                               edge_type=('author', 'to', 'paper'),
                               layout='coo', size=(10, 10))
    graph_store.put_edge_index(edge_index=(edge_index[0], edge_index[1]),
                               edge_type=('paper', 'to', 'term'), layout='coo',
                               size=(10, 10))

    datamodule = LightningLinkData(
        (feature_store, graph_store),
        input_train_edges=('author', 'to', 'paper'),
        loader='neighbor',
        num_neighbors=[5],
        batch_size=32,
        num_workers=0,
    )

    batch = next(iter(datamodule.train_dataloader()))
    assert 'edge_label_index' in batch['author', 'paper']


@withPackage('pytorch_lightning')
def test_eval_loader_kwargs(get_dataset):
    data = get_dataset(name='Cora')[0]

    node_sampler = NeighborSampler(data, num_neighbors=[5])

    datamodule = LightningNodeData(
        data,
        node_sampler=node_sampler,
        batch_size=32,
        eval_loader_kwargs=dict(num_neighbors=[-1], batch_size=64),
    )

    assert datamodule.loader_kwargs['batch_size'] == 32
    assert datamodule.neighbor_sampler.num_neighbors == [5]
    assert datamodule.eval_loader_kwargs['batch_size'] == 64
    assert datamodule.eval_neighbor_sampler.num_neighbors == [-1]

    train_loader = datamodule.train_dataloader()
    assert math.ceil(int(data.train_mask.sum()) / 32) == len(train_loader)

    val_loader = datamodule.val_dataloader()
    assert math.ceil(int(data.val_mask.sum()) / 64) == len(val_loader)
