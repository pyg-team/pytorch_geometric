from typing import Dict, List, Tuple, Optional
from torch_geometric.typing import NodeType, EdgeType

import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

import torch_geometric.transforms as T
from torch_geometric.data import Batch
from torch_geometric import seed_everything
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, Linear, to_hetero


class DataModule(LightningDataModule):
    def __init__(self, root: str):
        super().__init__()
        self.data = OGB_MAG(root, preprocess='metapath2vec',
                            transform=T.ToUndirected(merge=False))[0]

    @property
    def num_classes(self) -> int:
        return 349

    def metadata(self) -> Tuple[List[NodeType], List[EdgeType]]:
        node_types = ['paper', 'author', 'institution', 'field_of_study']
        edge_types = [('author', 'affiliated_with', 'institution'),
                      ('author', 'writes', 'paper'),
                      ('paper', 'cites', 'paper'),
                      ('paper', 'has_topic', 'field_of_study'),
                      ('institution', 'rev_affiliated_with', 'author'),
                      ('paper', 'rev_writes', 'author'),
                      ('paper', 'rev_cites', 'paper'),
                      ('field_of_study', 'rev_has_topic', 'paper')]
        return node_types, edge_types

    def train_dataloader(self):
        return NeighborLoader(
            self.data, num_neighbors=[10] * 2, shuffle=True, batch_size=1024,
            input_nodes=('paper', self.data['paper'].train_mask),
            num_workers=6, persistent_workers=True)

    def val_dataloader(self):
        return NeighborLoader(
            self.data, num_neighbors=[10] * 2, shuffle=False, batch_size=1024,
            input_nodes=('paper', self.data['paper'].val_mask), num_workers=6,
            persistent_workers=True)

    def test_dataloader(self):
        return NeighborLoader(
            self.data, num_neighbors=[10] * 2, shuffle=False, batch_size=1024,
            input_nodes=('paper', self.data['paper'].test_mask), num_workers=6,
            persistent_workers=True)


class GAT(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)

        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


class RelationalGAT(LightningModule):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        metadata: Tuple[List[NodeType], List[EdgeType]],
        dropout: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = GAT(hidden_channels, out_channels, dropout)
        self.model = to_hetero(self.model, metadata, aggr='sum')

        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        return self.model(x_dict, edge_index_dict)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        batch_size = batch['paper'].batch_size
        y_hat = self(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        loss = F.cross_entropy(y_hat, batch['paper'].y[:batch_size])
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        batch_size = batch['paper'].batch_size
        y_hat = self(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        self.val_acc(y_hat.softmax(dim=-1), batch['paper'].y[:batch_size])
        self.log('val_acc', self.val_acc)

    def test_step(self, batch: Batch, batch_idx: int):
        batch_size = batch['paper'].batch_size
        y_hat = self(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        self.test_acc(y_hat.softmax(dim=-1), batch['paper'].y[:batch_size])
        self.log('test_acc', self.test_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main():
    seed_everything(42)
    datamodule = DataModule('../../data/OGB')
    model = RelationalGAT(64, datamodule.num_classes, datamodule.metadata())
    with torch.no_grad():  # Initialize parameters.
        for batch in datamodule.val_dataloader():
            model(batch.x_dict, batch.edge_index_dict)
            break
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1)
    trainer = Trainer(gpus=1, max_epochs=20, callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule)
    trainer.test()


if __name__ == "__main__":
    main()
