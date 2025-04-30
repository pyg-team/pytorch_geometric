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
    r"""The graph transformer model from the "Transformer for Graphs:
    An Overview from Architecture Perspective"
    <https://arxiv.org/pdf/2202.08455>_ paper.

    Args:
        hidden_dim (int): The dimension of the hidden representations.
        num_class (int): The number of output classes.
        use_super_node (bool, optional): Whether to use a learnable class
            token as a super node. Defaults to False.
        node_feature_encoder (nn.Module, optional): A module to encode
            node features. Defaults to nn.Identity().
        num_encoder_layers (int, optional): The number of encoder layers
            in the transformer. Defaults to 0 (single layer).
        num_heads (int, optional): The number of attention heads used in
            in the transformer. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        ffn_hidden_dim (Optional[int], optional): The hidden dimension
            of the feedforward network. If None, defaults to 4 * hidden_dim.
        activation (str, optional): The activation function to use in the
            feedforward network. Defaults to 'gelu'.
        attn_bias_providers (Sequence[BaseBiasProvider], optional): A
            sequence of bias providers that return attention bias masks.
            Defaults to an empty sequence.
        gnn_block (Optional[Callable[[Data, torch.Tensor],
            torch.Tensor]], optional): A function that takes a Data object
            and node features, and returns updated node features after
            applying a GNN block. Defaults to None.
        gnn_position (Literal['pre', 'post', 'parallel'], optional):
            Where to apply the GNN block. Can be 'pre', 'post', or
            'parallel'. Defaults to 'pre'.
        positional_encoders (Sequence[BasePositionalEncoder], optional): A
            sequence of positional encoders to apply to node features.
            Defaults to an empty sequence.

    """

    def __init__(
        self,
        hidden_dim: int = 16,
        num_class: int = 2,
        use_super_node: bool = False,
        node_feature_encoder=nn.Identity(),
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
            self.register_parameter('cls_token', self.cls_token)

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
        if self.use_super_node:
            graph_sizes = torch.bincount(batch)
            first_idx = torch.cumsum(graph_sizes, 0) - graph_sizes
            return x[first_idx]
        else:
            return global_mean_pool(x, batch)

    @torch.jit.ignore
    def _add_cls_token(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Prepends the learnable class token to every graph's node embeddings.

        Args:
            x (torch.Tensor): The input features.
            batch (torch.Tensor): The batch vector.

        Returns:
            torch.Tensor: A stacked tensor of shape
            (num_graphs, num_nodes_i+1, hidden_dim) where
            the first token of every graph corresponds to the cls token.
        """
        num_graphs = batch.max().item() + 1
        x_list = []
        for i in range(num_graphs):
            mask = batch == i
            x_i = x[mask]
            x_i = torch.cat([self.cls_token.expand(1, -1), x_i], dim=0)
            x_list.append(x_i)
        return torch.stack(x_list, dim=0)

    @torch.jit.ignore
    def _prepend_cls_token_flat(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend a learnable CLS token to each graph’s nodes in a batch.

        Given:
            x (Tensor[N, C]): Node features for N total nodes.
            batch (LongTensor[N]): Graph assignment indices in [0..B-1].

        Returns:
            new_x (Tensor[N + B, C]): Features with one CLS token
            prepended per graph.
            new_batch (LongTensor[N + B]): Updated batch vector, length N + B.
        """
        device = x.device
        B = int(batch.max()) + 1
        C = x.size(1)

        lengths = torch.bincount(batch, minlength=B)
        new_lengths = lengths + 1
        new_batch = torch.repeat_interleave(
            torch.arange(B, device=device), new_lengths
        )

        new_N = x.size(0) + B
        new_x = x.new_empty((new_N, C))

        offsets = new_lengths.cumsum(0) - new_lengths
        cls_tokens = self.cls_token.expand(B, -1)
        new_x[offsets] = cls_tokens

        orig_positions = torch.arange(x.size(0), device=device)
        new_positions = orig_positions + batch + 1
        new_x[new_positions] = x

        return new_x, new_batch

    @torch.jit.ignore
    def _encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes node features using the node feature encoder.

        Args:
            x (torch.Tensor): The input features.

        Returns:
            torch.Tensor: The encoded node features.
        """
        if self.node_feature_encoder is not None:
            x = self.node_feature_encoder(x)
        return x

    def forward(self, data):
        r"""Applies the graph transformer model to the input data.

        Args:
            data (torch_geometric.data.Data): The input data.

        Returns:
            torch.Tensor: The output of the model.
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
        return {"logits": self.classifier(x)}

    def _encode_and_apply_structural(self, data: Data) -> torch.Tensor:
        r"""Encodes node features and applies structural encodings.

        Args:
            data (Data): The input graph data.

        Returns:
            torch.Tensor: The encoded node features.
        """
        x = data.x
        x = self._encode_nodes(x)
        if self.positional_encoders:
            x = x + sum(encoder(data) for encoder in self.positional_encoders)
        return x

    def _apply_gnn_if(
        self, position: Literal['pre', 'post', 'parallel'], data: Data,
        x: torch.Tensor
    ) -> torch.Tensor:
        r"""Applies the GNN block if specified and at the correct position.

        Args:
            position (Literal['pre', 'post', 'parallel']): Where to apply
                the GNN block.
            data (Data): The input graph data.
            x (torch.Tensor): The current node features.

        Returns:
            torch.Tensor: The updated node features after applying GNN.
        """
        if self.gnn_block is not None and self.gnn_position == position:
            x = self.gnn_block(data, x)
        return x

    def _prepare_batch(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Prepares the batch vector and optionally adds a class token.

        Args:
            x (torch.Tensor): The current node features.
            batch (torch.Tensor): The batch vector.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated
            node features and the batch vector.
        """
        if self.use_super_node:
            x, batch = self._prepend_cls_token_flat(x, batch)
        return x, batch

    def _is_parallel(self) -> bool:
        r"""Checks if the GNN block is applied in parallel.

        Returns:
            bool: True if GNN block is applied in parallel, False otherwise.
        """
        return self.gnn_block is not None and self.gnn_position == 'parallel'

    def _run_encoder(
        self, x: torch.Tensor, batch_vec: torch.Tensor,
        struct_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""Runs the encoder on the node features in sequence.
        Rebuilds key_pad only on head‐count changes.

        Args:
            x (torch.Tensor): The current node features.
            batch_vec (torch.Tensor): The batch vector.
            struct_mask (Optional[torch.Tensor]): The attention bias mask.

        Returns:
            torch.Tensor: The updated node features after encoding.
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
        """Combine legacy and all provider masks into one.

        Returns:
            FloatTensor of shape (B, H, L, L) or None if no mask was provided.
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
            f"gnn_hook={gnn_name}@'{self.gnn_position}'"
            ")"
        )
