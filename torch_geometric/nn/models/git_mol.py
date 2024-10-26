from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential

from torch_geometric.nn import GINEConv
from torch_geometric.nn.cv import SwinTransformer
from torch_geometric.nn.nlp import SentenceTransformer
from torch_geometric.utils import to_dense_batch


class GraphEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        dropout: float = 0.,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.gnns = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.gnns.append(
                GINEConv(
                    nn=Sequential(
                        Linear(in_channels, in_channels * 2),
                        ReLU(),
                        Linear(in_channels * 2, in_channels),
                    ),
                    train_eps=True,
                    edge_dim=in_channels,
                ))
            self.batch_norms.append(BatchNorm1d(in_channels))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        for i, (gnn, bn) in enumerate(zip(self.gnns, self.batch_norms)):
            x = gnn(x, edge_index, edge_attr)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x, _ = to_dense_batch(x, batch)
        return x


class GITMol(torch.nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.graph_encoder = GraphEncoder(num_layers=2, in_channels=16)
        self.text_encoder = SentenceTransformer(
            model_name='allenai/scibert_scivocab_uncased',
            pooling_strategy='last_hidden_state',
        )
        self.vision_encoder = SwinTransformer()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor],
        smiles: List[str],
        captions: List[str],
        images: Tensor,
    ) -> Tensor:
        # TODO: add atom and bond embedding
        x_graph = self.graph_encoder(x, edge_index, batch,
                                     edge_attr)  # [bs, node_len, 16]
        x_smiles = self.text_encoder.encode(smiles)  # [bs, seq_len, 768]
        x_captions = self.text_encoder.encode(captions)  # [bs, seq_len, 768]
        x_vision = self.vision_encoder(images)  # [bs, patch_len, 1536]
        print(x_graph.size(), x_smiles.size(), x_captions.size(),
              x_vision.size())
        import pdb
        pdb.set_trace()

    def pretrain(
        self,
        task: str,
    ) -> None:
        pass

    def finetune(
        self,
        task: str,
    ) -> None:
        pass

    def inference(self, ) -> Tensor:
        pass
