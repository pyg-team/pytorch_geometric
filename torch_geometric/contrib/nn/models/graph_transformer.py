from typing import Callable, Literal, Optional, Sequence

import torch
import torch.nn as nn
from torch.nn import ModuleList

from torch_geometric.contrib.nn.bias.base import BaseBiasProvider
from torch_geometric.contrib.nn.layers.transformer import (
    GraphTransformerEncoder,
    GraphTransformerEncoderLayer,
)
from torch_geometric.contrib.nn.positional.base import BasePositionalEncoder
from torch_geometric.contrib.utils.mask_utils import build_key_padding
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


class GraphTransformer(torch.nn.Module):
    r"""Graph Transformer model.

    Implements the transformer for graphs as described in
    "Transformer for Graphs: An Overview from Architecture Perspective"
    (https://arxiv.org/pdf/2202.08455).

    Args:
        hidden_dim (int): Dimension of hidden representations.
        num_class (int): Number of output classes.
        use_super_node (bool, optional): Use learnable class token as a
            super node. Defaults to False.
        node_feature_encoder (nn.Module, optional): Module to encode node
            features. Defaults to nn.Identity().
        num_encoder_layers (int, optional): Number of encoder layers.
            Defaults to 0 (single layer).
        num_heads (int, optional): Number of attention heads.
            Defaults to 4.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        ffn_hidden_dim (Optional[int], optional): Hidden dim of feedforward.
            Defaults to 4 * hidden_dim.
        activation (str, optional): Activation function.
            Defaults to 'gelu'.
        attn_bias_providers (Sequence[BaseBiasProvider], optional): Sequence of
            bias providers that return attention bias masks.
            Defaults to empty tuple.
        gnn_block (Optional[Callable[[Data, torch.Tensor], torch.Tensor]],
            optional): Function applying a GNN block. Defaults to None.
        gnn_position (Literal['pre', 'post', 'parallel'], optional): Position
            to apply the GNN block. Defaults to 'pre'.
        positional_encoders (Sequence[BasePositionalEncoder], optional):
        Sequence of positional encoders. Defaults to empty tuple.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        num_class: int = 2,
        use_super_node: bool = False,
        node_feature_encoder=None,
        num_encoder_layers: int = 0,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_hidden_dim: Optional[int] = None,
        activation: str = 'gelu',
        attn_bias_providers: Sequence[BaseBiasProvider] = (),
        gnn_block: Optional[Callable[[Data, torch.Tensor],
                                     torch.Tensor]] = None,
        gnn_position: Literal['pre', 'post', 'parallel'] = 'pre',
        positional_encoders: Sequence[BasePositionalEncoder] = ()
    ) -> None:
        super().__init__()

        if num_heads is None and attn_bias_providers:
            num_heads = attn_bias_providers[0].num_heads

        if node_feature_encoder is None:
            node_feature_encoder = nn.Identity()

        num_heads = num_heads if num_heads is not None else 4
        self._validate_init_args(
            hidden_dim, num_class, num_encoder_layers, num_heads, dropout,
            ffn_hidden_dim, activation, attn_bias_providers, gnn_block,
            gnn_position, positional_encoders
        )

        self.classifier = nn.Linear(hidden_dim, num_class)
        self.use_super_node = use_super_node
        if self.use_super_node:
            self.cls_token = nn.Parameter(torch.zeros(1, hidden_dim))

        self.node_feature_encoder = node_feature_encoder
        self.dropout = dropout
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim or 4 * hidden_dim
        self.activation = activation

        encoder_layer = GraphTransformerEncoderLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden_dim,
            activation=activation
        )
        self.encoder = (
            GraphTransformerEncoder(encoder_layer, num_encoder_layers)
            if num_encoder_layers > 0 else encoder_layer
        )
        self.is_encoder_stack = num_encoder_layers > 0
        self.attn_bias_providers = ModuleList(attn_bias_providers)
        self.gnn_block = gnn_block
        self.gnn_position = gnn_position
        self.positional_encoders = ModuleList(positional_encoders)

    def _validate_init_args(
        self,
        hidden_dim: int,
        num_class: int,
        num_encoder_layers: int,
        num_heads: int,
        dropout: float,
        ffn_hidden_dim: Optional[int],
        activation: str,
        attn_bias_providers: Sequence[BaseBiasProvider],
        gnn_block: Optional[Callable],
        gnn_position: str,
        positional_encoders: Sequence[BasePositionalEncoder],
    ):
        # ---- shape & type checks ----
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be a positive int (got " + f"{hidden_dim})"
            )
        if not isinstance(num_class, int) or num_class <= 0:
            raise ValueError(
                "num_class must be a positive int (got " + f"{num_class})"
            )
        if not isinstance(num_encoder_layers, int) or num_encoder_layers < 0:
            raise ValueError(
                "num_encoder_layers must be ≥ 0 (got " +
                f"{num_encoder_layers})"
            )

        # ---- transformer hyper-params ----
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                "Invalid configuration: embed_dim and num_heads must be"
                f"positive (got num_heads={num_heads})"
            )
        if not (isinstance(dropout, float) and 0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0,1) (got " + f"{dropout})")
        if ffn_hidden_dim is not None:
            if not (isinstance(ffn_hidden_dim, int)
                    and ffn_hidden_dim >= hidden_dim):
                raise ValueError(
                    "ffn_hidden_dim must be ≥ hidden_dim (" +
                    f"{hidden_dim}), got {ffn_hidden_dim}"
                )
        allowed_acts = {
            'relu', 'leakyrelu', 'prelu', 'tanh', 'selu', 'elu', 'linear',
            'gelu'
        }
        if activation not in allowed_acts:
            raise ValueError(
                "activation must be one of " +
                f"{allowed_acts} (got '{activation}', not supported)"
            )

        # ---- bias providers ----
        for prov in attn_bias_providers:
            if not isinstance(prov, BaseBiasProvider):
                raise TypeError(f"{prov!r} is not a BaseBiasProvider")
            if prov.num_heads != num_heads:
                msg = (
                    f"BiasProvider {prov.__class__.__name__} has "
                    f"num_heads={prov.num_heads}, but GraphTransformer is "
                    f"configured with num_heads={num_heads}"
                )
                raise ValueError(msg)

        # ---- GNN block & position ----
        valid_positions = {'pre', 'post', 'parallel'}
        if gnn_position not in valid_positions:
            raise ValueError(
                "gnn_position must be one of " +
                f"{valid_positions}, got '{gnn_position}'"
            )
        if gnn_block is None and gnn_position != 'pre':
            raise ValueError(
                "Cannot set gnn_position to 'post' or 'parallel' when " +
                "gnn_block is None"
            )

        # ---- positional encoders ----
        for enc in positional_encoders:
            if not callable(getattr(enc, "forward", None)):
                raise TypeError(
                    f"{enc!r} does not have a callable forward method"
                )

    @torch.jit.ignore
    def _readout(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Readout node features.

        If use_super_node is True, return the CLS token; else perform global
        mean pooling.

        Args:
            x (torch.Tensor): Node features.
            batch (torch.Tensor): Batch indices.

        Returns:
            torch.Tensor: Readout features.
        """
        if self.use_super_node:
            graph_sizes = torch.bincount(batch)
            first_idx = torch.cumsum(graph_sizes, 0) - graph_sizes
            return x[first_idx]
        else:
            return global_mean_pool(x, batch)

    def _prepend_cls_token_flat(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """Prepend a learnable CLS token to each graph's nodes.

        Inserts one classification token per graph without loops or
        non-TorchScript features.

        Args:
            x (torch.Tensor): Node features of shape (N, C).
            batch (torch.Tensor): Graph indices of shape (N,).

        Returns:
            tuple[torch.Tensor, torch.LongTensor]: New node features of shape
            (N + B, C) and updated batch indices.
        """
        B = batch.max() + 1
        N, C = x.size(0), x.size(1)
        lengths = torch.bincount(batch, minlength=B)
        new_lengths = lengths + 1
        graph_ids = torch.arange(B, device=x.device)
        new_batch = graph_ids.repeat_interleave(new_lengths)
        offsets = new_lengths.cumsum(0) - new_lengths
        cls_positions = offsets
        node_positions = torch.arange(N, device=x.device) + batch + 1
        all_positions = torch.cat([cls_positions, node_positions], dim=0)
        cls_tokens = self.cls_token.expand(B, C)
        all_features = torch.cat([cls_tokens, x], dim=0)
        total = N + B
        new_x = x.new_zeros((total, C))
        new_x = new_x.scatter(
            0,
            all_positions.unsqueeze(1).expand(-1, C),
            all_features,
        )

        return new_x, new_batch

    @staticmethod
    def _find_in_features(module):
        """Recursively finds the in_features attribute in a module."""
        if hasattr(module, 'in_features'):
            return module.in_features
        if hasattr(module, 'children'):
            for child in module.children():
                in_feat = GraphTransformer._find_in_features(child)
                if in_feat is not None:
                    return in_feat
        return None

    @torch.jit.ignore
    def _encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes node features using the node feature encoder.

        Args:
            x (torch.Tensor): Input node features.

        Returns:
            torch.Tensor: Encoded node features.

        Raises:
            ValueError: If input features are None or their dimension does not
                match the encoder's expected input.
        """
        encoder = self.node_feature_encoder
        if x is None:
            raise ValueError(
                "Input node features are None. Please ensure your dataset "
                "provides node features or supply a suitable "
                "node_feature_encoder that can handle None input."
            )
        if isinstance(encoder, nn.Identity):
            expected_dim = self.classifier.in_features
            if x.size(-1) != expected_dim:
                raise ValueError(
                    f"Node feature dimension mismatch: got {x.size(-1)}, "
                    f"expected {expected_dim} for nn.Identity encoder. "
                    "Please ensure your input features match hidden_dim or "
                    "use a node_feature_encoder that projects to hidden_dim."
                )
            return x
        in_features = self._find_in_features(encoder)
        if in_features is not None and x.size(-1) != in_features:
            raise ValueError(
                f"Node feature dimension mismatch: got {x.size(-1)}, "
                f"expected {in_features}. Please check your dataset and "
                f"node_feature_encoder configuration."
            )
        return encoder(x)

    def forward(self, data: Data) -> torch.Tensor:
        """Perform a forward pass of GraphTransformer.

        Encodes node features, applies optional GNN blocks, runs the
        transformer encoder, and performs readout to produce logits.

        Args:
            data (torch_geometric.data.Data): Input graph data.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self._encode_and_apply_structural(data)
        x = self._apply_gnn_if(position="pre", data=data, x=x)
        x, batch_vec = self._prepare_batch(x, data.batch)
        struct_mask = self._collect_attn_bias(data)
        x_parallel_in = x if self._is_parallel() else None
        x = self._run_encoder(x, batch_vec, struct_mask)
        x = self._apply_gnn_if(position="post", data=data, x=x)

        if x_parallel_in is not None:
            x = x + self.gnn_block(data, x_parallel_in)

        x = self._readout(x, batch_vec)
        return self.classifier(x)

    def _encode_and_apply_structural(self, data: Data) -> torch.Tensor:
        """Encode node features and apply positional encodings.

        Args:
            data (Data): Input graph data.

        Returns:
            torch.Tensor: Node features with positional encodings.
        """
        x = data.x

        # Handle when node features are None (common in regression datasets)
        if x is None and self.node_feature_encoder is not None:
            # Get input dimension from the first layer of the encoder
            if hasattr(self.node_feature_encoder, '__getitem__'):
                first_layer = self.node_feature_encoder[0]
            else:
                first_layer = self.node_feature_encoder
            input_dim = getattr(
                first_layer, 'in_features', self.classifier.in_features
            )

            num_nodes = data.num_nodes
            device = next(self.parameters()).device
            x = torch.zeros(num_nodes, input_dim, device=device)

        x = self._encode_nodes(x)
        if self.positional_encoders:
            x = x + sum(encoder(data) for encoder in self.positional_encoders)
        return x

    def _apply_gnn_if(
        self, position: Literal['pre', 'post', 'parallel'], data: Data,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply the GNN block at a specified position, if available.

        Args:
            position (Literal['pre', 'post', 'parallel']): Position to apply
                the GNN block.
            data (Data): Input graph data.
            x (torch.Tensor): Current node features.

        Returns:
            torch.Tensor: Node features after applying the GNN block.
        """
        if self.gnn_block is not None and self.gnn_position == position:
            x = self.gnn_block(data, x)
        return x

    def _prepare_batch(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the batch vector and optionally prepend the CLS token.

        Args:
            x (torch.Tensor): Node features.
            batch (torch.Tensor): Batch indices for each node.

        Returns:
            x (torch.Tensor): Node features, potentially with prepended CLS.
            batch (torch.Tensor): Updated batch indices.
        """
        if self.use_super_node:
            x, batch = self._prepend_cls_token_flat(x, batch)
        return x, batch

    def _is_parallel(self) -> bool:
        """Check if the GNN block is applied in parallel.

        Returns:
            bool: True if a parallel GNN block is used, else False.
        """
        return self.gnn_block is not None and self.gnn_position == 'parallel'

    def _run_encoder(
        self, x: torch.Tensor, batch_vec: torch.Tensor,
        struct_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Run the transformer encoder.

        Iterates over encoder layers and rebuilds key padding masks as needed.

        Args:
            x (torch.Tensor): Node features.
            batch_vec (torch.Tensor): Batch indices.
            struct_mask (Optional[torch.Tensor]): Structural attention mask.

        Returns:
            torch.Tensor: Node features after encoding.
        """
        layers = self.encoder if self.is_encoder_stack else [self.encoder]
        key_pad = None
        current_heads = None
        for layer in layers:
            if key_pad is None or layer.num_heads != current_heads:
                key_pad = build_key_padding(
                    batch_vec, num_heads=layer.num_heads
                )
                current_heads = layer.num_heads

            x = layer(x, batch_vec, struct_mask, key_pad)
        return x

    @torch.jit.ignore
    def _collect_attn_bias(self, data: Data) -> Optional[torch.Tensor]:
        """Collect and aggregate attention bias masks from providers.

        Args:
            data (Data): Input graph data.

        Returns:
            Optional[torch.Tensor]: Aggregated attention bias mask or None.
        """
        masks = []
        if getattr(data, "bias", None) is not None:
            legacy = data.bias
            # cast bool→float, leave floats as-is
            masks.append(
                legacy.to(torch.float32)
                if not torch.is_floating_point(legacy) else legacy
            )

        for prov in self.attn_bias_providers:
            m = prov(data)
            if m is not None:
                masks.append(m.to(torch.float32))

        if not masks:
            return None
        return torch.stack(
            masks, dim=0
        ).sum(dim=0).to(data.x.dtype).to(data.x.device)

    def __repr__(self):
        """Return a string representation of GraphTransformer.

        Includes details like hidden dims, num classes, encoder layers,
        bias providers, positional encoders, dropout, num heads, feedforward
        dim, activation function, and GNN block config.

        Returns:
            str: String representation.
        """
        n_layers = len(self.encoder) if self.is_encoder_stack else 0
        providers = [
            prov.__class__.__name__ for prov in self.attn_bias_providers
        ]
        if self.gnn_block is not None:
            gnn_name = self.gnn_block.__name__
        else:
            gnn_name = None

        pos_encoders = [
            enc.__class__.__name__ for enc in self.positional_encoders
        ]

        return (
            "GraphTransformer("
            f"hidden_dim={self.classifier.in_features}, "
            f"num_class={self.classifier.out_features}, "
            f"use_super_node={self.use_super_node}, "
            f"num_encoder_layers={n_layers}, "
            f"bias_providers={providers}, "
            f"pos_encoders={pos_encoders}, "
            f"dropout={self.dropout}, "
            f"num_heads={self.num_heads}, "
            f"ffn_hidden_dim={self.ffn_hidden_dim}, "
            f"activation='{self.activation}', "
            f"gnn_hook={gnn_name}@'{self.gnn_position}'"
            ")"
        )
