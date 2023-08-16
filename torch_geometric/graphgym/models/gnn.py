import torch
import torch.nn as nn
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


def GNNLayer(dim_in, dim_out, has_act=True):
    """
    Wrapper for a GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    Returns:
        GeneralLayer: A GNN layer configured according to the provided parameters.

    This function creates a GNN layer based on the specified input and output dimensions,
    with an optional activation function. The layer configuration is determined using the
    `new_layer_config` function, considering the provided settings and configurations.

    Example:
        To create a GNN layer with input dimension 16 and output dimension 32:
        >>> layer = GNNLayer(dim_in=16, dim_out=32)

    Note:
        Make sure the `new_layer_config` function is properly configured before using this
        function.

    """
    return GeneralLayer(
        cfg.gnn.layer_type,
        layer_config=new_layer_config(dim_in, dim_out, 1, has_act=has_act,
                                      has_bias=False, cfg=cfg),
    )


def GNNPreMP(dim_in, dim_out, num_layers):
    """
    Wrapper for NN layer before GNN message passing

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of layers

    Returns:
        GeneralMultiLayer: A stack of neural network layers for preprocessing before GNN
        message passing.

    This function creates a sequence of neural network layers intended to preprocess input
    features before GNN message passing. The number of layers, input dimension, and output
    dimension are specified, and the layer configuration is determined using the
    `new_layer_config` function.

    Example:
        To create a stack of 3 linear layers with input dimension 16 and output dimension 32:
        >>> pre_mp_layers = GNNPreMP(dim_in=16, dim_out=32, num_layers=3)

    Note:
        Make sure the `new_layer_config` function is properly configured before using this
        function.
    """
    return GeneralMultiLayer(
        "linear",
        layer_config=new_layer_config(dim_in, dim_out, num_layers,
                                      has_act=False, has_bias=False, cfg=cfg),
    )


@register_stage("stack")
@register_stage("skipsum")
@register_stage("skipconcat")
class GNNStackStage(nn.Module):
    """
    Simple Stage that stack GNN layers

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if cfg.gnn.stage_type == "skipconcat":
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.add_module("layer{}".format(i), layer)

    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.stage_type == "skipsum":
                batch.x = x + batch.x
            elif cfg.gnn.stage_type == "skipconcat" and i < self.num_layers - 1:
                batch.x = torch.cat([x, batch.x], dim=1)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch


class FeatureEncoder(nn.Module):
    """
    Encodes node and edge features based on the provided configurations.

    Args:
        dim_in (int): Input feature dimension.

    Attributes:
        dim_in (int): The current input feature dimension after applying encoders.
        node_encoder (nn.Module): Node feature encoder module, if enabled.
        edge_encoder (nn.Module): Edge feature encoder module, if enabled.
        node_encoder_bn (BatchNorm1dNode): Batch normalization for node encoder output,
            if enabled.
        edge_encoder_bn (BatchNorm1dNode): Batch normalization for edge encoder output,
            if enabled.

    The FeatureEncoder module encodes node and edge features based on the configurations
    specified in the provided `cfg` object. It supports encoding integer node and edge
    features using embeddings, optionally followed by batch normalization. The output
    dimension of the encoded features is determined by the `cfg.gnn.dim_inner` parameter.

    If `cfg.dataset.node_encoder` or `cfg.dataset.edge_encoder` is enabled, the respective
    encoder modules are created based on the provided encoder names. If batch
    normalization is enabled for either encoder, the corresponding batch normalization
    layer is added after the encoder.

    Example:
        Given an instance of FeatureEncoder:
        >>> encoder = FeatureEncoder(dim_in=16)
        >>> encoded_features = encoder(batch)

    Note:
        Make sure to set up the configuration (`cfg`) appropriately before creating an
        instance of FeatureEncoder.
    """
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
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
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
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


class GNN(nn.Module):
    """
    General Graph Neural Network (GNN) model composed of an encoder, processing stage,
    and head.

    Args:
        dim_in (int): Input feature dimension.
        dim_out (int): Output feature dimension.
        **kwargs (optional): Optional additional keyword arguments.

    Attributes:
        encoder (FeatureEncoder): Node and edge feature encoder.
        pre_mp (GNNPreMP, optional): Pre-message-passing processing layers, if any.
        mp (GNNStage, optional): Message-passing stage, if any.
        post_mp (GNNHead): Post-message-passing processing layers.

    The GNN model consists of three main components: an encoder to transform input
    features, a message-passing stage for information exchange, and a head to produce
    final output features. The processing layers in each component are determined by
    the provided configurations.

    Example:
        Given an instance of GNN:
        >>> gnn = GNN(dim_in=16, dim_out=32)
        >>> output = gnn(batch)

    Note:
        Make sure to set up the configuration (`cfg`) and any required module registrations
        (`register`) before creating an instance of GNN.
    """
    def __init__(self, dim_in, dim_out, **kwargs):
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
