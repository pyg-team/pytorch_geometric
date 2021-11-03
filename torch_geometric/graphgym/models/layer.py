import copy
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

import torch_geometric.graphgym.models.act
from torch_geometric.graphgym.contrib.layer.generalconv import (
    GeneralConvLayer, GeneralEdgeConvLayer)

import torch_geometric.graphgym.register as register


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


def new_layer_config(dim_in, dim_out, num_layers, has_act, has_bias, cfg):
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
        num_layers=num_layers)


# General classes
class GeneralLayer(nn.Module):
    """
    General wrapper for layers

    Args:
        name (string): Name of the layer in registered :obj:`layer_dict`
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation after the layer
        has_bn (bool):  Whether has BatchNorm in the layer
        has_l2norm (bool): Wheter has L2 normalization after the layer
        **kwargs (optional): Additional args
    """
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = layer_config.has_l2norm
        has_bn = layer_config.has_batchnorm
        layer_config.has_bias = not has_bn
        self.layer = register.layer_dict[name](layer_config, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                nn.BatchNorm1d(layer_config.dim_out,
                               eps=layer_config.bn_eps,
                               momentum=layer_config.bn_mom))
        if layer_config.dropout > 0:
            layer_wrapper.append(
                nn.Dropout(p=layer_config.dropout,
                           inplace=layer_config.mem_inplace))
        if layer_config.has_act:
            layer_wrapper.append(register.act_dict[layer_config.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

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


class GeneralMultiLayer(nn.Module):
    """
    General wrapper for a stack of multiple layers

    Args:
        name (string): Name of the layer in registered :obj:`layer_dict`
        num_layers (int): Number of layers in the stack
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        dim_inner (int): The dimension for the inner layers
        final_act (bool): Whether has activation after the layer stack
        **kwargs (optional): Additional args
    """
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = layer_config.dim_in \
            if layer_config.dim_inner is None \
            else layer_config.dim_inner
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
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


# ---------- Core basic layers. Input: batch; Output: batch ----------------- #


class Linear(nn.Module):
    """
    Basic Linear layer.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        bias (bool): Whether has bias term
        **kwargs (optional): Additional args
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super(Linear, self).__init__()
        self.model = nn.Linear(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class BatchNorm1dNode(nn.Module):
    """
    BatchNorm for node feature.

    Args:
        dim_in (int): Input dimension
    """
    def __init__(self, layer_config: LayerConfig):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(layer_config.dim_in,
                                 eps=layer_config.bn_eps,
                                 momentum=layer_config.bn_mom)

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(nn.Module):
    """
    BatchNorm for edge feature.

    Args:
        dim_in (int): Input dimension
    """
    def __init__(self, layer_config: LayerConfig):
        super(BatchNorm1dEdge, self).__init__()
        self.bn = nn.BatchNorm1d(layer_config.dim_in,
                                 eps=layer_config.bn_eps,
                                 momentum=layer_config.bn_mom)

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch


class MLP(nn.Module):
    """
    Basic MLP model.
    Here 1-layer MLP is equivalent to a Liner layer.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        bias (bool): Whether has bias term
        dim_inner (int): The dimension for the inner layers
        num_layers (int): Number of layers in the stack
        **kwargs (optional): Additional args
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super(MLP, self).__init__()
        dim_inner = layer_config.dim_in \
            if layer_config.dim_inner is None \
            else layer_config.dim_inner
        layer_config.has_bias = True
        layers = []
        if layer_config.num_layers > 1:
            sub_layer_config = LayerConfig(
                num_layers=layer_config.num_layers - 1,
                dim_in=layer_config.dim_in,
                dim_out=dim_inner,
                dim_inner=dim_inner,
                final_act=True)
            layers.append(GeneralMultiLayer('linear', sub_layer_config))
            layer_config = replace(layer_config, dim_in=dim_inner)
            layers.append(Linear(layer_config))
        else:
            layers.append(Linear(layer_config))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch


class GCNConv(nn.Module):
    """
    Graph Convolutional Network (GCN) layer
    """

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super(GCNConv, self).__init__()
        self.model = pyg.nn.GCNConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class SAGEConv(nn.Module):
    """
    GraphSAGE Conv layer
    """

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = pyg.nn.SAGEConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class GATConv(nn.Module):
    """
    Graph Attention Network (GAT) layer
    """

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super(GATConv, self).__init__()
        self.model = pyg.nn.GATConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class GINConv(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super(GINConv, self).__init__()
        gin_nn = nn.Sequential(
            nn.Linear(layer_config.dim_in, layer_config.dim_out),
            nn.ReLU(),
            nn.Linear(layer_config.dim_out, layer_config.dim_out))
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class SplineConv(nn.Module):
    """
    SplineCNN layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super(SplineConv, self).__init__()
        self.model = pyg.nn.SplineConv(layer_config.dim_in,
                                       layer_config.dim_out,
                                       dim=1,
                                       kernel_size=2,
                                       bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


class GeneralConv(nn.Module):
    """A general GNN layer"""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super(GeneralConv, self).__init__()
        self.model = GeneralConvLayer(layer_config.dim_in,
                                      layer_config.dim_out,
                                      bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


class GeneralEdgeConv(nn.Module):
    """A general GNN layer that supports edge features as well"""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super(GeneralEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(layer_config.dim_in,
                                          layer_config.dim_out,
                                          layer_config.edge_dim,
                                          bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index,
                             edge_feature=batch.edge_attr)
        return batch


class GeneralSampleEdgeConv(nn.Module):
    """A general GNN layer that supports edge features and edge sampling"""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super(GeneralSampleEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(layer_config.dim_in,
                                          layer_config.dim_out,
                                          layer_config.edge_dim,
                                          bias=layer_config.has_bias)
        self.keep_edge = layer_config.keep_edge

    def forward(self, batch):
        edge_mask = torch.rand(batch.edge_index.shape[1]) < self.keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_attr[edge_mask, :]
        batch.x = self.model(batch.x, edge_index, edge_feature=edge_feature)
        return batch


layer_dict = {
    'linear': Linear,
    'mlp': MLP,
    'gcnconv': GCNConv,
    'sageconv': SAGEConv,
    'gatconv': GATConv,
    'splineconv': SplineConv,
    'ginconv': GINConv,
    'generalconv': GeneralConv,
    'generaledgeconv': GeneralEdgeConv,
    'generalsampleedgeconv': GeneralSampleEdgeConv,
}

# register additional convs
register.layer_dict = {**register.layer_dict, **layer_dict}
