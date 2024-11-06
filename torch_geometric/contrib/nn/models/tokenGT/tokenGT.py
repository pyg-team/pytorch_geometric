import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from torch_geometric.nn.models.tokengt_layers import GraphFeatureTokenizer

class TokenGT(nn.Module):
    r"""
    Tokenized Graph Transformer (TokenGT) model for graph representation learning.

    Args:
        node_feat_dim (int): Dimension of node features.
        edge_feat_dim (int): Dimension of edge features.
        hidden_dim (int): Input dimension of the transformer.
        num_layers (int): Number of transformer encoder layers.
        num_heads (int): Number of attention heads.
        num_classes (int): Number of classes for classification.
        method (str): Method to generate node identifiers ('orf' or 'laplacian').
        d_p (int): Dimension of node identifiers.
        d_e (int): Dimension of type identifiers.
        use_graph_token (bool): Whether to include the [graph] token.
        dropout (float): Dropout rate.
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 hidden_dim: int, num_layers: int, num_heads: int,
                 num_classes: int, method: str, d_p: int, d_e: int,
                 use_graph_token: bool = True, dropout: float = 0.1):
        super(TokenGT, self).__init__()
        self.tokenizer = GraphFeatureTokenizer(
            node_feat_dim, edge_feat_dim, hidden_dim, method, d_p, d_e,
            use_graph_token
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, num_classes)  # For classification
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Batch) -> torch.Tensor:
        r"""
        Forward pass of the TokenGT model.

        Args:
            data (Batch): PyG Batch object containing graph data.

        Returns:
            logits (torch.Tensor): Output logits, shape [batch_size, num_classes]
        """
        tokens, attention_masks = self.tokenizer(data)
        output = self.encoder(tokens, src_key_padding_mask=~attention_masks)

        if self.tokenizer.use_graph_token:
            graph_embeddings = output[:, 0, :]  
        else:
            graph_embeddings = global_add_pool(output, data.batch)
        graph_embeddings = self.dropout(graph_embeddings)

        logits = self.fc_out(graph_embeddings) 

        return logits