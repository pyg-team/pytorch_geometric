from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import conv as conv_layers


@dataclass
class NodeClassifierArgs:
    conv_layer: str
    hidden_channels: int
    num_layers: int
    dropout: Optional[float] = 0.0
    in_channels: Optional[int] = None  # Inferred from the dataset
    out_channels: Optional[int] = None  # Inferred from the dataset
    use_predictions: Optional[bool] = None  # Inferred from the dataset


class NodeClassifier(torch.nn.Module):
    def __init__(self, args: NodeClassifierArgs) -> None:

        super().__init__()
        self.use_predictions = args.use_predictions
        if self.use_predictions:
            # Embedding lookup for each class (out_channels == num_classes)
            self.encoder = nn.Embedding(args.out_channels + 1,
                                        args.hidden_channels)

        self.convs = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        assert (conv_cls := getattr(conv_layers, args.conv_layer, None))
        self.convs.append(conv_cls(args.in_channels, args.hidden_channels))
        for _ in range(args.num_layers - 2):
            self.convs.append(
                conv_cls(args.hidden_channels, args.hidden_channels))
            self.batch_norm.append(nn.BatchNorm1d(args.hidden_channels))
        self.convs.append(conv_cls(args.hidden_channels, args.out_channels))
        self.batch_norm.append(nn.BatchNorm1d(args.hidden_channels))

        self.dropout = args.dropout

    def reset_parameters(self) -> None:

        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norm:
            bn.reset_parameters()

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:

        if self.use_predictions:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norm[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)

        return x
