import torch
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.layer import (
    BatchNorm1dNode,
    GeneralLayer,
    GeneralMultiLayer,
    new_layer_config,
)
from torch_geometric.graphgym.register import register_stage


def GNNLayer(dim_in: int, dim_out: int, has_act: bool = True) -> GeneralLayer:
    r"""Creates a GNN layer, given the specified input and output dimensions
    and the underlying configuration in :obj:`cfg`.

    Args:
        dim_in (int): The input dimension
        dim_out (int): The output dimension.
        has_act (bool, optional): Whether to apply an activation function
            after the layer. (default: :obj:`True`)
    """
    return GeneralLayer(
        cfg.gnn.layer_type,
        layer_config=new_layer_config(
            dim_in,
            dim_out,
            1,
            has_act=has_act,
            has_bias=False,
            cfg=cfg,
        ),
    )


def GNNPreMP(dim_in: int, dim_out: int, num_layers: int) -> GeneralMultiLayer:
    r"""Creates a NN layer used before message passing, given the specified
    input and output dimensions and the underlying configuration in :obj:`cfg`.

    Args:
        dim_in (int): The input dimension
        dim_out (int): The output dimension.
        num_layers (int): The number of layers.
    """
    return GeneralMultiLayer(
        'linear',
        layer_config=new_layer_config(
            dim_in,
            dim_out,
            num_layers,
            has_act=False,
            has_bias=False,
            cfg=cfg,
        ),
    )


@register_stage('stack')
@register_stage('skipsum')
@register_stage('skipconcat')
class GNNStackStage(torch.nn.Module):
    r"""Stacks a number of GNN layers.

    Args:
        dim_in (int): The input dimension
        dim_out (int): The output dimension.
        num_layers (int): The number of layers.
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.add_module(f'layer{i}', layer)

    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif (cfg.gnn.stage_type == 'skipconcat'
                  and i < self.num_layers - 1):
                batch.x = torch.cat([x, batch.x], dim=1)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch


class FeatureEncoder(torch.nn.Module):
    r"""Encodes node and edge features, given the specified input dimension and
    the underlying configuration in :obj:`cfg`.

    Args:
        dim_in (int): The input feature dimension.
    """
    def __init__(self, dim_in: int):
        super().__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via `torch.nn.Embedding`:
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    ))
            # Update `dim_in` to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via `torch.nn.Embedding`:
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    ))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class GNN(torch.nn.Module):
    r"""A general Graph Neural Network (GNN) model.

    The GNN model consists of three main components:

    1. An encoder to transform input features into a fixed-size embedding
       space.
    2. A processing or message passing stage for information exchange between
       nodes.
    3. A head to produce the final output features/predictions.

    The configuration of each component is determined by the underlying
    configuration in :obj:`cfg`.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
        **kwargs (optional): Additional keyword arguments.
    """
    def __init__(self, dim_in: int, dim_out: int, **kwargs):
        super().__init__()
        GNNStage = register.stage_dict[cfg.gnn.stage_type]
        GNNHead = register.head_dict[cfg.gnn.head]

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp > 0:
            self.mp = GNNStage(dim_in=dim_in, dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
