import torch
import torch.nn.functional as F


class ARLinkPredictor(torch.nn.Module):
    r"""Link predictor using Attract-Repel embeddings from the paper
    `"Pseudo-Euclidean Attract-Repel Embeddings for Undirected Graphs"
    <https://arxiv.org/abs/2106.09671>`_.

    This model splits node embeddings into: attract and
    repel.
    The edge prediction score is computed as the dot product of attract
    components minus the dot product of repel components.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of hidden embeddings.
        out_channels (int, optional): Size of output embeddings.
            If set to :obj:`None`, will default to :obj:`hidden_channels`.
            (default: :obj:`None`)
        num_layers (int): Number of message passing layers.
            (default: :obj:`2`)
        dropout (float): Dropout probability. (default: :obj:`0.0`)
        attract_ratio (float): Ratio to use for attract component.
            Must be between 0 and 1. (default: :obj:`0.5`)
    """
    def __init__(self, in_channels, hidden_channels, out_channels=None,
                 num_layers=2, dropout=0.0, attract_ratio=0.5):
        super().__init__()

        if out_channels is None:
            out_channels = hidden_channels

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        if not 0 <= attract_ratio <= 1:
            raise ValueError(
                f"attract_ratio must be between 0 and 1, got {attract_ratio}")

        self.attract_ratio = attract_ratio
        self.attract_dim = int(out_channels * attract_ratio)
        self.repel_dim = out_channels - self.attract_dim

        # Create model layers
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

        # Final layer splits into attract and repel components
        self.lin_attract = torch.nn.Linear(hidden_channels, self.attract_dim)
        self.lin_repel = torch.nn.Linear(hidden_channels, self.repel_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for lin in self.lins:
            lin.reset_parameters()
        self.lin_attract.reset_parameters()
        self.lin_repel.reset_parameters()

    def encode(self, x, *args, **kwargs):
        """Encode node features into attract-repel embeddings.

        Args:
            x (torch.Tensor): Node feature matrix of shape
                :obj:`[num_nodes, in_channels]`.
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        """
        for lin in self.lins:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Split into attract and repel components
        attract_x = self.lin_attract(x)
        repel_x = self.lin_repel(x)

        return attract_x, repel_x

    def decode(self, attract_z, repel_z, edge_index):
        """Decode edge scores from attract-repel embeddings.

        Args:
            attract_z (torch.Tensor): Attract embeddings of shape
                :obj:`[num_nodes, attract_dim]`.
            repel_z (torch.Tensor): Repel embeddings of shape
                :obj:`[num_nodes, repel_dim]`.
            edge_index (torch.Tensor): Edge indices of shape
                :obj:`[2, num_edges]`.

        Returns:
            torch.Tensor: Edge prediction scores.
        """
        # Get node embeddings for edges
        row, col = edge_index
        attract_z_row = attract_z[row]
        attract_z_col = attract_z[col]
        repel_z_row = repel_z[row]
        repel_z_col = repel_z[col]

        # Compute attract-repel scores
        attract_score = torch.sum(attract_z_row * attract_z_col, dim=1)
        repel_score = torch.sum(repel_z_row * repel_z_col, dim=1)

        return attract_score - repel_score

    def forward(self, x, edge_index):
        """Forward pass for link prediction.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Edge indices to predict.

        Returns:
            torch.Tensor: Predicted edge scores.
        """
        # Encode nodes into attract-repel embeddings
        attract_z, repel_z = self.encode(x)

        # Decode target edges
        return torch.sigmoid(self.decode(attract_z, repel_z, edge_index))

    def calculate_r_fraction(self, attract_z, repel_z):
        """Calculate the R-fraction (proportion of energy in repel space).

        Args:
            attract_z (torch.Tensor): Attract embeddings.
            repel_z (torch.Tensor): Repel embeddings.

        Returns:
            float: R-fraction value.
        """
        attract_norm_squared = torch.sum(attract_z**2)
        repel_norm_squared = torch.sum(repel_z**2)

        r_fraction = repel_norm_squared / (attract_norm_squared +
                                           repel_norm_squared + 1e-10)

        return r_fraction.item()
