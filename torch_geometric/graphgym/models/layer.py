import copy
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

import torch_geometric as pyg
import torch_geometric.graphgym.models.act
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.contrib.layer.generalconv import (
    GeneralConvLayer,
    GeneralEdgeConvLayer,
)
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg


@dataclass
class LayerConfig:
    # batchnorm parameters.
    has_batchnorm: bool = False
    bn_eps: float = 1e-5
    bn_mom: float = 0.1

    # mem parameters.
    mem_inplace: bool = False

    # gnn parameters.
    dim_in: int = -1
    dim_out: int = -1
    edge_dim: int = -1
    dim_inner: int = None
    num_layers: int = 2
    has_bias: bool = True
    # regularizer parameters.
    has_l2norm: bool = True
    dropout: float = 0.0
    # activation parameters.
    has_act: bool = True
    final_act: bool = True
    act: str = 'relu'

    # other parameters.
    keep_edge: float = 0.5


def new_layer_config(
    dim_in: int,
    dim_out: int,
    num_layers: int,
    has_act: bool,
    has_bias: bool,
    cfg,
) -> LayerConfig:
    r"""Createa a layer configuration for a GNN layer.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
        num_layers (int): The number of hidden layers
        has_act (bool): Whether to apply an activation function after the
            layer.
        has_bias (bool): Whether to apply a bias term in the layer.
        cfg (ConfigNode): The underlying configuration.
    """
    return LayerConfig(
        has_batchnorm=cfg.gnn.batchnorm,
        bn_eps=cfg.bn.eps,
        bn_mom=cfg.bn.mom,
        mem_inplace=cfg.mem.inplace,
        dim_in=dim_in,
        dim_out=dim_out,
        edge_dim=cfg.dataset.edge_dim,
        has_l2norm=cfg.gnn.l2norm,
        dropout=cfg.gnn.dropout,
        has_act=has_act,
        final_act=True,
        act=cfg.gnn.act,
        has_bias=has_bias,
        keep_edge=cfg.gnn.keep_edge,
        dim_inner=cfg.gnn.dim_inner,
        num_layers=num_layers,
    )


class GeneralLayer(torch.nn.Module):
    r"""A general wrapper for layers.

    Args:
        name (str): The registered name of the layer.
        layer_config (LayerConfig): The configuration of the layer.
        **kwargs (optional): Additional keyword arguments.
    """
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.has_l2norm = layer_config.has_l2norm
        has_bn = layer_config.has_batchnorm
        layer_config.has_bias = not has_bn
        self.layer = register.layer_dict[name](layer_config, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                torch.nn.BatchNorm1d(
                    layer_config.dim_out,
                    eps=layer_config.bn_eps,
                    momentum=layer_config.bn_mom,
                ))
        if layer_config.dropout > 0:
            layer_wrapper.append(
                torch.nn.Dropout(
                    p=layer_config.dropout,
                    inplace=layer_config.mem_inplace,
                ))
        if layer_config.has_act:
            layer_wrapper.append(register.act_dict[layer_config.act]())
        self.post_layer = torch.nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch


class GeneralMultiLayer(torch.nn.Module):
    r"""A general wrapper class for a stacking multiple NN layers.

    Args:
        name (str): The registered name of the layer.
        layer_config (LayerConfig): The configuration of the layer.
        **kwargs (optional): Additional keyword arguments.
    """
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        if layer_config.dim_inner:
            dim_inner = layer_config.dim_out
        else:
            dim_inner = layer_config.dim_inner

        for i in range(layer_config.num_layers):
            d_in = layer_config.dim_in if i == 0 else dim_inner
            d_out = layer_config.dim_out \
                if i == layer_config.num_layers - 1 else dim_inner
            has_act = layer_config.final_act \
                if i == layer_config.num_layers - 1 else True
            inter_layer_config = copy.deepcopy(layer_config)
            inter_layer_config.dim_in = d_in
            inter_layer_config.dim_out = d_out
            inter_layer_config.has_act = has_act
            layer = GeneralLayer(name, inter_layer_config, **kwargs)
            self.add_module(f'Layer_{i}', layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


# ---------- Core basic layers. Input: batch; Output: batch ----------------- #


@register_layer('linear')
class Linear(torch.nn.Module):
    r"""A basic Linear layer.

    Args:
        layer_config (LayerConfig): The configuration of the layer.
        **kwargs (optional): Additional keyword arguments.
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = Linear_pyg(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class BatchNorm1dNode(torch.nn.Module):
    r"""A batch normalization layer for node-level features.

    Args:
        layer_config (LayerConfig): The configuration of the layer.
    """
    def __init__(self, layer_config: LayerConfig):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(
            layer_config.dim_in,
            eps=layer_config.bn_eps,
            momentum=layer_config.bn_mom,
        )

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(torch.nn.Module):
    r"""A batch normalization layer for edge-level features.

    Args:
        layer_config (LayerConfig): The configuration of the layer.
    """
    def __init__(self, layer_config: LayerConfig):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(
            layer_config.dim_in,
            eps=layer_config.bn_eps,
            momentum=layer_config.bn_mom,
        )

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch


@register_layer('mlp')
class MLP(torch.nn.Module):
    """A basic MLP model.

    Args:
        layer_config (LayerConfig): The configuration of the layer.
        **kwargs (optional): Additional keyword arguments.
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        if layer_config.dim_inner is None:
            dim_inner = layer_config.dim_in
        else:
            dim_inner = layer_config.dim_inner

        layer_config.has_bias = True
        layers = []
        if layer_config.num_layers > 1:
            sub_layer_config = LayerConfig(
                num_layers=layer_config.num_layers - 1,
                dim_in=layer_config.dim_in, dim_out=dim_inner,
                dim_inner=dim_inner, final_act=True)
            layers.append(GeneralMultiLayer('linear', sub_layer_config))
            layer_config = replace(layer_config, dim_in=dim_inner)
            layers.append(Linear(layer_config))
        else:
            layers.append(Linear(layer_config))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


@register_layer('gcnconv')
class GCNConv(torch.nn.Module):
    r"""A Graph Convolutional Network (GCN) layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GCNConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('sageconv')
class SAGEConv(torch.nn.Module):
    r"""A GraphSAGE layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.SAGEConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('gatconv')
class GATConv(torch.nn.Module):
    r"""A Graph Attention Network (GAT) layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GATConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('ginconv')
class GINConv(torch.nn.Module):
    r"""A Graph Isomorphism Network (GIN) layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = torch.nn.Sequential(
            Linear_pyg(layer_config.dim_in, layer_config.dim_out),
            torch.nn.ReLU(),
            Linear_pyg(layer_config.dim_out, layer_config.dim_out),
        )
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('splineconv')
class SplineConv(torch.nn.Module):
    r"""A SplineCNN layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.SplineConv(
            layer_config.dim_in,
            layer_config.dim_out,
            dim=1,
            kernel_size=2,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


@register_layer('generalconv')
class GeneralConv(torch.nn.Module):
    r"""A general GNN layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralConvLayer(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('generaledgeconv')
class GeneralEdgeConv(torch.nn.Module):
    r"""A general GNN layer with edge feature support."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralEdgeConvLayer(
            layer_config.dim_in,
            layer_config.dim_out,
            layer_config.edge_dim,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index,
                             edge_feature=batch.edge_attr)
        return batch


@register_layer('generalsampleedgeconv')
class GeneralSampleEdgeConv(torch.nn.Module):
    r"""A general GNN layer that supports edge features and edge sampling."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralEdgeConvLayer(
            layer_config.dim_in,
            layer_config.dim_out,
            layer_config.edge_dim,
            bias=layer_config.has_bias,
        )
        self.keep_edge = layer_config.keep_edge

    def forward(self, batch):
        edge_mask = torch.rand(batch.edge_index.shape[1]) < self.keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_attr[edge_mask, :]
        batch.x = self.model(batch.x, edge_index, edge_feature=edge_feature)
        return batch
