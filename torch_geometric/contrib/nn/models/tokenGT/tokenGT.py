import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from tokenizer import GraphFeatureTokenizer

class TokenGT(nn.Module):
    r"""
    Tokenized Graph Transformer (TokenGT) model from the "Pure Transformers are Powerful Graph Learners"
    https://arxiv.org/abs/2207.02505 paper.

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
    def __init__(self, input_feat_dim: int,
                 hidden_dim: int, num_layers: int, num_heads: int,
                 num_classes: int, method: str, d_p: int, d_e: int,
                 use_graph_token: bool = True, dropout: float = 0.1):
        super(TokenGT, self).__init__()

        # Graph tokenizer: converts graph structure into sequence of tokens
        self.tokenizer = GraphFeatureTokenizer(
            input_feat_dim, hidden_dim, method, d_p, d_e,
            use_graph_token
        )

        # Standard transformer encoder layer with multi-head attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )

        # Stack multiple encoder layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, num_classes) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Process a batch of graphs through the TokenGT model.

        Workflow:
        1. Convert graphs into token sequences using tokenizer
        2. Process tokens through transformer encoder
        3. Extract graph-level embeddings (either from [graph] token or via pooling)
        4. Apply dropout and classification layer

        Args:
            data (Batch): PyG batch object containing graphs

        Returns:
            logits (torch.Tensor): Classification logits [batch_size, num_classes]
        """

        # Convert graphs into token sequences and attention masks
        tokens, attention_masks = self.tokenizer(data)
        output = self.encoder(tokens, src_key_padding_mask=~attention_masks)

        if self.tokenizer.use_graph_token:
            graph_embeddings = output[:, 0, :]  
        else:
            graph_embeddings = global_add_pool(output, data.batch)

        # Apply dropout for regularization
        graph_embeddings = self.dropout(graph_embeddings)

        logits = self.fc_out(graph_embeddings) 

        return logits
