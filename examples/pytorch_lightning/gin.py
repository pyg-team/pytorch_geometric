from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import (Trainer, LightningDataModule, LightningModule,
                               seed_everything)

from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_add_pool


class IMDBBinary(LightningDataModule):
    def __init__(self, data_dir: str = './data/TUDataset'):
        super().__init__()
        self.data_dir = data_dir

    @property
    def num_features(self):
        return 136

    @property
    def num_classes(self):
        return 2

    def prepare_data(self):
        transform = T.Compose([
            T.OneHotDegree(self.num_features - 1),
            T.ToSparseTensor(),
        ])
        self.dataset = TUDataset(self.data_dir, name='IMDB-BINARY',
                                 pre_transform=transform).shuffle()

    def setup(self, stage: Optional[str] = None):
        dataset = self.dataset
        self.test_dataset = dataset[:len(dataset) // 10]
        self.val_dataset = dataset[len(dataset) // 10:len(dataset) // 5]
        self.train_dataset = dataset[len(dataset) // 5:]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=128, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=256)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=256)


class GIN(LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()

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
            self.convs.append(GINConv(mlp, train_eps=True))
            in_channels = hidden_channels

        self.classifier = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=dropout, inplace=True),
            Linear(hidden_channels, out_channels),
        )

        self.acc = Accuracy(out_channels)

    def forward(self, x: Tensor, adj_t: SparseTensor, batch: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x, adj_t)
        x = global_add_pool(x, batch)
        return self.classifier(x)

    def training_step(self, data: Data, batch_idx: int):
        y_hat = self(data.x, data.adj_t, data.batch)
        train_loss = F.cross_entropy(y_hat, data.y)
        self.log('train_loss', train_loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('train_acc', self.acc(y_hat, data.y), prog_bar=True,
                 on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, data: Data, batch_idx: int):
        y_hat = self(data.x, data.adj_t, data.batch)
        self.log('val_acc', self.acc(y_hat, data.y), on_step=False,
                 on_epoch=True, prog_bar=True)

    def test_step(self, data: Data, batch_idx: int):
        y_hat = self(data.x, data.adj_t, data.batch)
        self.log('test_acc', self.acc(y_hat, data.y), on_epoch=True,
                 prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main():
    seed_everything(42)
    data_module = IMDBBinary()
    model = GIN(data_module.num_features, data_module.num_classes)
    trainer = Trainer(gpus=1)
    trainer.fit(model, data_module)
    trainer.test()
    # model.to_torchscript(file_path='GIN_IMDB.pt', method='script')


if __name__ == "__main__":
    main()
