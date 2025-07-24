from typing import Callable, Dict, Literal, Optional, Sequence

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
from torch_geometric.nn.inits import reset

DEFAULT_ENCODER = {
    "num_encoder_layers": 0,
    "num_heads": 4,
    "dropout": 0.1,
    "ffn_hidden_dim": None,
    "activation": 'gelu',
    "attn_bias_providers": (),
    "positional_encoders": (),
    "node_feature_encoder": None,
    "use_super_node": False,
}

DEFAULT_GNN = {
    "gnn_block": None,
    "gnn_position": 'pre',
}


class GraphTransformer(torch.nn.Module):
    r"""Graph Transformer model.

    Implements the transformer for graphs as described in
    "Transformer for Graphs: An Overview from Architecture Perspective"
    (https://arxiv.org/pdf/2202.08455).

    Attributes:
        hidden_dim (int): Dimension of hidden representations.
        out_channels (int | None): Number of output channels.
        output_projection (nn.Module | None): Output projection layer.
        node_feature_encoder (nn.Module): Module to encode node features.
        dropout (float): Dropout rate.
        num_heads (int): Number of attention heads.
        ffn_hidden_dim (int): Hidden dimension of the feedforward network.
        encoder_cfg (dict, optional): Encoder configuration dictionary. Keys:
            - use_super_node (bool, optional): Use learnable class token as a
              super node. Defaults to False.
            - node_feature_encoder (nn.Module, optional): Module to encode
              node features. Defaults to nn.Identity().
            - num_encoder_layers (int, optional): Number of encoder layers.
              Defaults to 0 (single layer).
            - num_heads (int, optional): Number of attention heads. Defaults
              to 4.
            - dropout (float, optional): Dropout rate. Defaults to 0.1.
            - ffn_hidden_dim (Optional[int], optional): Hidden dim of
              feedforward. Defaults to 4 * hidden_dim.
            - activation (str, optional): Activation function. Defaults to
              'gelu'.
            - attn_bias_providers (Sequence[BaseBiasProvider], optional):
              Sequence of bias providers that return attention bias masks.
              Defaults to empty tuple.
            - positional_encoders (Sequence[BasePositionalEncoder], optional):
              Sequence of positional encoders. Defaults to empty tuple.
        gnn_cfg (dict, optional): GNN configuration dictionary. Keys:
            - gnn_block(Optional[Callable[[Data, torch.Tensor], torch.Tensor]],
              optional): Function applying a GNN block. Defaults to None.
            - gnn_position (Literal['pre', 'post', 'parallel'], optional):
              Position to apply the GNN block. Defaults to 'pre'.
        cache_masks (bool, optional): If True, caches masks for reuse.
            This can improve performance for large batches with the same
            number of nodes and heads. Defaults to False.
        cast_bias (bool, optional): If True, casts attention bias to the
            same dtype as node features. Defaults to False.
    """
    def __init__(
        self,
        hidden_dim: int,
        out_channels: int | None = None,
        *,
        encoder_cfg: dict | None = None,
        gnn_cfg: dict | None = None,
        cache_masks: bool = False,
        cast_bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.cast_bias = cast_bias
        self._user_supplied_encoder = (
            encoder_cfg is not None and "node_feature_encoder" in encoder_cfg
            and encoder_cfg["node_feature_encoder"] is not None)
        self.encoder_cfg = encoder_cfg or {}
        self.gnn_cfg = gnn_cfg or {}
        # batch-aware mask cache keyed by (num_graphs, num_nodes, num_heads)
        self.cache_masks = cache_masks
        if cache_masks:
            self._mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        else:
            self._mask_cache = None

        cfg, gnn = self._parse_cfg(hidden_dim, encoder_cfg, gnn_cfg)
        self._validate_cfg(hidden_dim, out_channels, cfg, gnn)
        self._build_modules(hidden_dim, out_channels, cfg, gnn)
        self.reset_parameters()

    def _parse_cfg(self, hidden_dim: int, encoder_cfg: dict | None,
                   gnn_cfg: dict | None) -> tuple[dict, dict]:
        """Parse and merge configuration dictionaries, fill in defaults."""
        cfg = {**DEFAULT_ENCODER, **(encoder_cfg or {})}
        gnn = {**DEFAULT_GNN, **(gnn_cfg or {})}
        if cfg["num_heads"] is None and cfg["attn_bias_providers"]:
            cfg["num_heads"] = cfg["attn_bias_providers"][0].num_heads
        if cfg["node_feature_encoder"] is None:
            cfg["node_feature_encoder"] = nn.Identity()
        if cfg["ffn_hidden_dim"] is None:
            cfg["ffn_hidden_dim"] = 4 * hidden_dim
        return cfg, gnn

    def forward(self, data: Data) -> torch.Tensor:
        """Perform a forward pass of GraphTransformer.

        Encodes node features, applies optional GNN blocks, runs the
        transformer encoder, and (optionally) projects to output channels.

        Args:
            data (torch_geometric.data.Data): Input graph data.

        Returns:
            torch.Tensor: Output tensor of shape (N_total, C), where:
                - C = hidden_dim if out_channels is None,
                - C = out_channels otherwise.
                - N_total = sum of nodes in all graphs in the batch.
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

        if self.output_projection is not None:
            x = self.output_projection(x)
        return x

    def _validate_cfg(self, hidden_dim: int, out_channels: int | None,
                      cfg: dict, gnn: dict) -> None:
        """Validate configuration dictionaries.

        Args:
            hidden_dim (int): Hidden representation dimension.
            out_channels (int | None): Number of output channels.
            cfg (dict): Encoder configuration dictionary.
            gnn (dict): GNN configuration dictionary.
        """
        self._validate_init_args(
            hidden_dim,
            out_channels,
            cfg["num_encoder_layers"],
            cfg["num_heads"],
            cfg["dropout"],
            cfg["ffn_hidden_dim"],
            cfg["activation"],
            cfg["attn_bias_providers"],
            gnn["gnn_block"],
            gnn["gnn_position"],
            cfg["positional_encoders"],
        )

    def _build_modules(
        self,
        hidden_dim: int,
        out_channels: int | None,
        cfg: Dict,
        gnn: Dict,
    ) -> None:
        """Instantiate model modules from config.

        Args:
            hidden_dim (int): Hidden representation dimension.
            out_channels (int | None): Number of output channels.
            cfg (dict): Encoder configuration dictionary.
            gnn (dict): GNN configuration dictionary.
        """
        if out_channels is None:
            self.output_projection: nn.Module | None = None
        else:
            self.output_projection = nn.Linear(hidden_dim, out_channels)
        self.use_super_node = cfg["use_super_node"]
        if self.use_super_node:
            self.cls_token = nn.Parameter(torch.zeros(1, hidden_dim))

        self.node_feature_encoder = cfg["node_feature_encoder"]
        self.dropout = cfg["dropout"]
        self.num_heads = cfg["num_heads"]
        self.ffn_hidden_dim = cfg["ffn_hidden_dim"]
        self.activation = cfg["activation"]

        layer = GraphTransformerEncoderLayer(
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            ffn_hidden_dim=self.ffn_hidden_dim,
            activation=self.activation,
        )
        self.encoder = (GraphTransformerEncoder(layer,
                                                cfg["num_encoder_layers"])
                        if cfg["num_encoder_layers"] > 0 else layer)
        self.is_encoder_stack = cfg["num_encoder_layers"] > 0

        self.attn_bias_providers = ModuleList(cfg["attn_bias_providers"])
        self.positional_encoders = ModuleList(cfg["positional_encoders"])
        self.gnn_block = gnn["gnn_block"]
        self.gnn_position = gnn["gnn_position"]

    def _validate_init_args(
        self,
        hidden_dim: int,
        out_channels: int,
        num_encoder_layers: int,
        num_heads: int,
        dropout: float,
        ffn_hidden_dim: Optional[int],
        activation: str,
        attn_bias_providers: Sequence[BaseBiasProvider],
        gnn_block: Optional[Callable],
        gnn_position: str,
        positional_encoders: Sequence[BasePositionalEncoder],
    ) -> None:
        """Validates initialization arguments via specialized validators.

        Args:
            hidden_dim (int): Hidden representation dimension.
            out_channels (int): Number of output classes.
            num_encoder_layers (int): Number of encoder layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            ffn_hidden_dim (Optional[int]): FFN hidden dimension.
            activation (str): Name of activation function.
            attn_bias_providers (Sequence[BaseBiasProvider]): Bias providers.
            gnn_block (Optional[Callable]): GNN block function.
            gnn_position (str): GNN block position.
            positional_encoders (Sequence[BasePositionalEncoder]):
                Positional encoders.
        """
        self._validate_dimensions(hidden_dim, out_channels, num_encoder_layers)
        self._validate_transformer_params(num_heads, dropout, ffn_hidden_dim,
                                          hidden_dim, activation)
        self._validate_bias_providers(attn_bias_providers, num_heads)
        self._validate_gnn_config(gnn_block, gnn_position)
        self._validate_positional_encoders(positional_encoders)

    def _validate_dimensions(self, hidden_dim: int, out_channels: int,
                             num_encoder_layers: int) -> None:
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive int (got {hidden_dim})")
        if not isinstance(out_channels, (int, type(None))):
            raise ValueError(
                f"out_channels must be int or None (got {type(out_channels)})")
        if isinstance(out_channels, int) and out_channels <= 0:
            raise ValueError(
                f"out_channels must be a positive int (got {out_channels})")

        if not isinstance(num_encoder_layers, int) or num_encoder_layers < 0:
            raise ValueError(
                f"num_encoder_layers must be ≥ 0 (got {num_encoder_layers})")

    def _validate_transformer_params(self, num_heads: int, dropout: float,
                                     ffn_hidden_dim: Optional[int],
                                     hidden_dim: int, activation: str) -> None:
        self._validate_num_heads(num_heads)
        self._validate_dropout(dropout)
        self._validate_ffn_dim(ffn_hidden_dim, hidden_dim)
        self._validate_activation(activation)

    def _validate_num_heads(self, num_heads: int) -> None:
        r"""Validates number of attention heads.

        Args:
            num_heads (int): Number of attention heads

        Raises:
            ValueError: If num_heads is not a positive integer
        """
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("Invalid configuration: num_heads must be"
                             f"positive (got num_heads={num_heads})")

    def _validate_dropout(self, dropout: float) -> None:
        r"""Validates dropout rate.

        Args:
            dropout (float): Dropout rate

        Raises:
            ValueError: If dropout is not a float in [0,1)
        """
        if not isinstance(dropout, float):
            raise ValueError(f"dropout must be float (got {type(dropout)})")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0,1) (got {dropout})")

    def _validate_ffn_dim(self, ffn_hidden_dim: Optional[int],
                          hidden_dim: int) -> None:
        if ffn_hidden_dim is None:
            return
        if not isinstance(ffn_hidden_dim, int):
            raise ValueError(
                f"ffn_hidden_dim must be int (got {type(ffn_hidden_dim)})")
        if ffn_hidden_dim < hidden_dim:
            raise ValueError(
                f"ffn_hidden_dim ({ffn_hidden_dim}) must be ≥ hidden_dim "
                f"({hidden_dim})")

    def _validate_activation(self, activation: str) -> None:
        r"""Validates activation function name.

        Args:
            activation (str): Activation function name

        Raises:
            ValueError: If activation is not one of the supported functions
        """
        allowed_acts = {
            'relu', 'leakyrelu', 'prelu', 'tanh', 'selu', 'elu', 'linear',
            'gelu'
        }
        if activation not in allowed_acts:
            raise ValueError(
                "activation must be one of " +
                f"{allowed_acts} (got '{activation}', not supported)")

    def _validate_bias_providers(self, providers: Sequence[BaseBiasProvider],
                                 num_heads: int) -> None:
        """Validates attention bias providers.
        Ensures all providers are instances of BaseBiasProvider and that
        their num_heads match the model's num_heads.

        Args:
            providers (Sequence[BaseBiasProvider]): Sequence of bias providers.
            num_heads (int): Number of attention heads.

        Returns:
            None
        Raises:
            TypeError: If any provider is not a BaseBiasProvider.
            ValueError: If any provider's num_heads does not match num_heads.
        """
        for prov in providers:
            if not isinstance(prov, BaseBiasProvider):
                raise TypeError(f"{prov!r} is not a BaseBiasProvider")
            if prov.num_heads != num_heads:
                raise ValueError(
                    f"BiasProvider {prov.__class__.__name__} has "
                    f"num_heads={prov.num_heads}, but GraphTransformer is "
                    f"configured with num_heads={num_heads}")

    def _validate_gnn_config(self, gnn_block: Optional[Callable],
                             gnn_position: str) -> None:
        """Validates GNN configuration.
        Ensures gnn_block is callable if provided, and gnn_position is valid.

        Args:
            gnn_block (Optional[Callable]): GNN block function.
            gnn_position (str): Position to apply the GNN block.

        Returns:
            None
        Raises:
            ValueError: If gnn_block is None and gnn_position is not 'pre'.
            ValueError: If gnn_position is not one of
            'pre', 'post', or 'parallel'.
        """
        valid_positions = {'pre', 'post', 'parallel'}
        if gnn_position not in valid_positions:
            raise ValueError(f"gnn_position must be one of {valid_positions}, "
                             f"got '{gnn_position}'")
        if gnn_block is None and gnn_position != 'pre':
            raise ValueError("Cannot set gnn_position to 'post' or 'parallel' "
                             "when gnn_block is None")

    def _validate_positional_encoders(
            self, encoders: Sequence[BasePositionalEncoder]) -> None:
        """Validates positional encoders.
        Ensures all encoders are callable and have a forward method.

        Args:
            encoders (Sequence[BasePositionalEncoder]): Sequence of positional
                encoders.

        Returns:
            None
        Raises:
            TypeError: If any encoder is not callable or does not have a
                forward method.
        """
        for enc in encoders:
            if not callable(getattr(enc, "forward", None)):
                raise TypeError(
                    f"{enc!r} does not have a callable forward method")

    def _prepend_cls_token_flat(
            self, x: torch.Tensor,
            batch: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
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
        """Recursively finds the in_features attribute in a module.
        This is used to determine the expected input dimension for the
        node feature encoder.

        Args:
            module (torch.nn.Module): The module to search.

        Returns:
            Optional[int]: The in_features attribute if found, else None.
        """
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
                "node_feature_encoder that can handle None input.")

        if isinstance(encoder, nn.Identity):
            expected_dim = self.hidden_dim
            if x.size(-1) == expected_dim:
                return x

            if self._user_supplied_encoder:
                raise ValueError(
                    f"Node feature dimension mismatch: got {x.size(-1)}, "
                    f"expected {expected_dim} for nn.Identity encoder.")
            if not hasattr(self, "_input_proj"):
                self._input_proj = nn.Linear(
                    x.size(-1),
                    self.hidden_dim,
                    bias=False,
                ).to(x.device)
            return self._input_proj(x)

        in_features = self._find_in_features(encoder)
        if in_features is not None and x.size(-1) != in_features:
            raise ValueError(
                f"Node feature dimension mismatch: got {x.size(-1)}, "
                f"expected {in_features}. Please check your dataset and "
                f"node_feature_encoder configuration.")
        return encoder(x)

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
            input_dim = getattr(first_layer, 'in_features',
                                self.output_projection.in_features)

            num_nodes = data.num_nodes
            device = next(self.parameters()).device
            x = torch.ones(num_nodes, input_dim, device=device)

        x = self._encode_nodes(x)
        if self.positional_encoders:
            x = x + sum(encoder(data) for encoder in self.positional_encoders)
        return x

    def _apply_gnn_if(self, position: Literal['pre', 'post', 'parallel'],
                      data: Data, x: torch.Tensor) -> torch.Tensor:
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
            self, x: torch.Tensor,
            batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _run_encoder(self, x: torch.Tensor, batch_vec: torch.Tensor,
                     struct_mask: Optional[torch.Tensor]) -> torch.Tensor:
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
                key_pad = self._get_key_pad(batch_vec, layer.num_heads)
                current_heads = layer.num_heads

            x = layer(x, batch_vec, struct_mask, key_pad)
        return x

    @torch.no_grad()
    def _get_key_pad(self, batch_vec: torch.Tensor,
                     num_heads: int) -> torch.Tensor:
        """Get or build the key padding mask for the given batch vector.

        Args:
            batch_vec (torch.Tensor): Batch indices for each node.
            num_heads (int): Number of attention heads.

        Returns:
            torch.Tensor: Key padding mask of shape (B, num_heads, L, L).

        Note:
        This mask is used to prevent attention to padding tokens in the input.
        If the mask is already cached, it is returned; otherwise, it is built
        using the `build_key_padding` function and cached for future use.
        The mask is a square tensor where the first dimension is the batch size
        and the second dimension is the number of attention heads. The mask
        is expanded to match the number of heads and is used to ensure that
        attention is only applied to valid nodes in the batch.

        """
        if not self.cache_masks:
            return build_key_padding(batch_vec, num_heads=num_heads)

        if batch_vec.numel() == 0:
            return torch.empty((0, num_heads, 0, 0), dtype=torch.bool,
                               device=batch_vec.device)
        B, L = int(batch_vec.max()) + 1, batch_vec.size(0)
        key = (B, L, num_heads)
        cached = self._mask_cache.get(key)
        if cached is None or cached.device != batch_vec.device \
                or cached.dtype != torch.bool:
            self._mask_cache[key] = build_key_padding(batch_vec,
                                                      num_heads=num_heads)
        return self._mask_cache[key]

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
                legacy.to(torch.float32
                          ) if not torch.is_floating_point(legacy) else legacy)

        for prov in self.attn_bias_providers:
            m = prov(data)
            if m is not None:
                masks.append(m.to(torch.float32))

        if not masks:
            return None

        bias = torch.stack(masks, dim=0).sum(dim=0)
        if data.x.dtype == torch.float32 or self.cast_bias:
            bias = bias.to(data.x.dtype)
        return bias.to(data.x.device)

    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters.

        This method resets the parameters of the classifier, encoder,
        node feature encoder, attention bias providers, and positional
        encoders. It also resets the parameters of the GNN block if it exists.

        Args:
            None

        Returns:
            None
        """
        for module in [
                self.output_projection, self.encoder,
                self.node_feature_encoder, *self.attn_bias_providers,
                *self.positional_encoders
        ]:
            reset(module)

        if isinstance(self.gnn_block, nn.Module):
            reset(self.gnn_block)
        if self.use_super_node:
            nn.init.zeros_(self.cls_token)

        self._mask_cache.clear() if self.cache_masks else None

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

        return ("GraphTransformer("
                f"hidden_dim={self.hidden_dim}, "
                f"out_channels={self.out_channels}, "
                f"use_super_node={self.use_super_node}, "
                f"num_encoder_layers={n_layers}, "
                f"bias_providers={providers}, "
                f"pos_encoders={pos_encoders}, "
                f"dropout={self.dropout}, "
                f"num_heads={self.num_heads}, "
                f"ffn_hidden_dim={self.ffn_hidden_dim}, "
                f"activation='{self.activation}', "
                f"gnn_hook={gnn_name}@'{self.gnn_position}'"
                ")")
