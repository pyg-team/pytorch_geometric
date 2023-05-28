import os.path as osp
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torchmetrics import Accuracy

import torch_geometric.transforms as T
from torch_geometric.data import Batch
from torch_geometric.data.lightning import LightningNodeData
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import Linear, SAGEConv, to_hetero
from torch_geometric.typing import EdgeType, NodeType


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int,
                 dropout: float):
        super().__init__()
        self.dropout = dropout

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(-1, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class RelationalGNN(LightningModule):
    def __init__(
        self,
        metadata: Tuple[List[NodeType], List[EdgeType]],
        hidden_channels: int,
        out_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        model = GNN(hidden_channels, out_channels, dropout)
        # Convert the homogeneous GNN model to a heterogeneous variant in
        # which distinct parameters are learned for each node and edge type.
        self.model = to_hetero(model, metadata, aggr='sum')

        self.train_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.val_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.test_acc = Accuracy(task='multiclass', num_classes=out_channels)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        return self.model(x_dict, edge_index_dict)

    def common_step(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        batch_size = batch['paper'].batch_size
        y_hat = self(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        y = batch['paper'].y[:batch_size]
        return y_hat, y

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        y_hat, y = self.common_step(batch)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        y_hat, y = self.common_step(batch)
        self.val_acc(y_hat.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True)

    def test_step(self, batch: Batch, batch_idx: int):
        y_hat, y = self.common_step(batch)
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main():
    dataset = OGB_MAG(osp.join('data', 'OGB'), preprocess='metapath2vec',
                      transform=T.ToUndirected(merge=False))
    data = dataset[0]

    datamodule = LightningNodeData(
        data,
        input_train_nodes=('paper', data['paper'].train_mask),
        input_val_nodes=('paper', data['paper'].val_mask),
        input_test_nodes=('paper', data['paper'].test_mask),
        loader='neighbor',
        num_neighbors=[10, 10],
        batch_size=1024,
        num_workers=8,
    )

    model = RelationalGNN(data.metadata(), hidden_channels=64,
                          out_channels=349, dropout=0.0)

    with torch.no_grad():  # Run a dummy forward pass to initialize lazy model
        loader = datamodule.train_dataloader()
        batch = next(iter(loader))
        model.common_step(batch)

    strategy = pl.strategies.SingleDeviceStrategy('cuda:0')
    checkpoint = ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
    trainer = Trainer(strategy=strategy, devices=1, max_epochs=20,
                      log_every_n_steps=5, callbacks=[checkpoint])

    trainer.fit(model, datamodule)
    trainer.test(ckpt_path='best', datamodule=datamodule)


if __name__ == "__main__":
    main()
