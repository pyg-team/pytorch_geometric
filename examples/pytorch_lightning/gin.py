import os
from typing import Optional, Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from torch_sparse import SparseTensor
from torch_scatter import segment_csr
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Batch, DataLoader
from torch_geometric.nn import GINConv
from pytorch_lightning.core.decorators import auto_move_data


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

    def setup(self, stage: Optional[str] = None):
        transform = T.Compose([
            T.OneHotDegree(self.num_features - 1),
            T.ToSparseTensor(),
        ])
        dataset = TUDataset(self.data_dir, name='IMDB-BINARY',
                            pre_transform=transform).shuffle()

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
            conv = GINConv(mlp, train_eps=True)  # .jittable()
            self.convs.append(conv)
            in_channels = hidden_channels

        self.classifier = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=dropout),
            Linear(hidden_channels, out_channels),
        )

        self.acc = Accuracy(out_channels)

    def move_data(self, batch):
        return batch.to(self.device)

    def transfer_batch_to_device(self, batch: Any, device: torch.device):
        return (batch[0].to(device), batch[1])

    def forward(self, x: Tensor, adj_t: SparseTensor, ptr: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x, adj_t)
        x = segment_csr(x, ptr, reduce='sum')  # Global add pooling.
        return self.classifier(x)

    def training_step(self, batch: Batch, batch_idx: int):
        y_hat = self(batch.x, batch.adj_t, batch.ptr)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.log('train_loss', train_loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        self.log('train_acc', self.acc(y_hat, batch.y), prog_bar=True,
                 on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch: Batch, batch_idx: int):
        y_hat = self(batch.x, batch.adj_t, batch.ptr)
        self.log('val_acc', self.acc(y_hat, batch.y), on_step=False,
                 on_epoch=True, prog_bar=True)

    def test_step(self, batch: Batch, batch_idx: int):
        y_hat = self(batch.x, batch.adj_t, batch.ptr)
        self.log('test_acc', self.acc(y_hat, batch.y), on_epoch=True,
                 prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

def main():
    seed_everything(42)
    datamodule = IMDBBinary()
    model = GIN(datamodule.num_features, datamodule.num_classes)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=-1, save_last=True)
    trainer = Trainer(
        gpus=2, 
        accelerator='ddp', 
        max_epochs=20,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0
        )

    trainer.fit(model, datamodule=datamodule)
    trainer.test()

    # if os.environ["LOCAL_RANK"] == '0':
        # model = GIN.load_from_checkpoint(checkpoint_callback.best_model_path)
        # model.to_torchscript(file_path='GIN_IMDB-BINARY.pt', method='script')


if __name__ == "__main__":
    main()
