from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# Import these at runtime to avoid circular imports
from torch_geometric.typing import Adj, OptTensor


class AttractRepel(nn.Module):
    r"""A wrapper for Graph Neural Networks that implements Attract-Repel embeddings.

    This wrapper splits the embedding space into "attract" and "repel" components,
    which enables better representation of non-transitive relationships in graphs
    as described in the paper "Pseudo-Euclidean Attract-Repel Embeddings for
    Undirected Graphs" (Peysakhovich et al., 2023).

    Args:
        model (Union[str, nn.Module]): The base GNN model to wrap. Can be either a string
            ('GCN', 'GAT', 'GraphSAGE') or an instance of a PyG model.
        in_channels (Optional[int]): Size of each input sample. Required if `model` is a string.
        hidden_channels (Optional[int]): Size of hidden layers. Required if `model` is a string.
        out_channels (int): Size of each output sample.
        attract_ratio (float, optional): Ratio of dimensions to allocate to the attract component.
            (Default: 0.5)
        **kwargs: Additional arguments for the base model if `model` is a string.
    """
    def __init__(self, model: Union[str, nn.Module],
                 in_channels: Optional[int] = None,
                 hidden_channels: Optional[int] = None, out_channels: int = 64,
                 attract_ratio: float = 0.5, **kwargs):
        super().__init__()

        # Process dimensions
        self.out_channels = out_channels
        self.attract_ratio = attract_ratio
        self.attract_dim = int(out_channels * attract_ratio)
        self.repel_dim = out_channels - self.attract_dim

        # Set up base model
        if isinstance(model, str):
            if in_channels is None or hidden_channels is None:
                raise ValueError(
                    "in_channels and hidden_channels must be provided when model is a string"
                )

            # Handle num_layers parameter
            model_kwargs = kwargs.copy()
            if 'num_layers' not in model_kwargs:
                model_kwargs['num_layers'] = 2  # Default to 2 layers

            # Import specific models here to avoid circular imports
            if model == 'GCN':
                from torch_geometric.nn import GCN
                self.base_model = GCN(in_channels=in_channels,
                                      hidden_channels=hidden_channels,
                                      out_channels=out_channels,
                                      **model_kwargs)
            elif model == 'GAT':
                from torch_geometric.nn import GAT
                self.base_model = GAT(in_channels=in_channels,
                                      hidden_channels=hidden_channels,
                                      out_channels=out_channels,
                                      **model_kwargs)
            elif model == 'GraphSAGE':
                from torch_geometric.nn import GraphSAGE
                self.base_model = GraphSAGE(in_channels=in_channels,
                                            hidden_channels=hidden_channels,
                                            out_channels=out_channels,
                                            **model_kwargs)
            else:
                raise ValueError(f"Unknown model type: {model}")
        else:
            # User provided a model instance
            self.base_model = model

            # Check if we need to modify the output dimension
            if hasattr(self.base_model, 'out_channels'
                       ) and self.base_model.out_channels != out_channels:
                # Replace last layer
                if hasattr(self.base_model, 'lin') and isinstance(
                        self.base_model.lin, nn.Linear):
                    in_features = self.base_model.lin.in_features
                    self.base_model.lin = nn.Linear(in_features, out_channels)
                    self.base_model.out_channels = out_channels
                elif hasattr(self.base_model, 'convs') and isinstance(
                        self.base_model.convs, nn.ModuleList):
                    # Assume the last layer in convs is the output layer
                    self._replace_last_layer(self.base_model.convs,
                                             out_channels)
                    self.base_model.out_channels = out_channels

    def _replace_last_layer(self, module_list: nn.ModuleList,
                            out_channels: int):
        """Helper to replace the last layer in a module list."""
        if len(module_list) == 0:
            return

        last_module = module_list[-1]
        if hasattr(last_module, 'lin_rel') and isinstance(
                last_module.lin_rel, nn.Linear):
            # For GAT-like models
            in_features = last_module.lin_rel.in_features
            last_module.lin_rel = nn.Linear(in_features, out_channels)
        elif hasattr(last_module, 'lin') and isinstance(
                last_module.lin, nn.Linear):
            # For GCN-like models
            in_features = last_module.lin.in_features
            last_module.lin = nn.Linear(in_features, out_channels)
        elif hasattr(last_module, 'out_channels'):
            # Direct attribute
            last_module.out_channels = out_channels

    def encode(self, x: torch.Tensor, edge_index: Adj,
               edge_weight: OptTensor = None,
               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input data into attract and repel embeddings.

        Args:
            x: The node features.
            edge_index: The edge indices.
            edge_weight: One-dimensional edge weights.
            **kwargs: Additional arguments forwarded to the base model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attract and repel embeddings.
        """
        # Get node embeddings from base model
        z = self.base_model(x, edge_index, edge_weight, **kwargs)

        # Split into attract and repel components
        attract_z = z[:, :self.attract_dim]
        repel_z = z[:, self.attract_dim:]

        return attract_z, repel_z

    def calculate_scores(self, attract_z_src: torch.Tensor,
                         attract_z_dst: torch.Tensor,
                         repel_z_src: torch.Tensor,
                         repel_z_dst: torch.Tensor) -> torch.Tensor:
        """Calculate the attract-repel scores for pairs of nodes.

        Args:
            attract_z_src: Attract embeddings for source nodes.
            attract_z_dst: Attract embeddings for destination nodes.
            repel_z_src: Repel embeddings for source nodes.
            repel_z_dst: Repel embeddings for destination nodes.

        Returns:
            torch.Tensor: Attract-repel scores.
        """
        # Calculate attract score (dot product of attract components)
        attract_score = (attract_z_src * attract_z_dst).sum(dim=1)

        # Calculate repel score (dot product of repel components)
        repel_score = (repel_z_src * repel_z_dst).sum(dim=1)

        # AR score = attract score - repel score
        return attract_score - repel_score

    def calculate_r_fraction(self, attract_z: torch.Tensor,
                             repel_z: torch.Tensor) -> torch.Tensor:
        """Calculate the R-fraction of the embeddings.

        The R-fraction measures how much of the embedding is explained by the repel component.
        Higher values indicate more heterophily in the network.

        Args:
            attract_z: Attract embeddings.
            repel_z: Repel embeddings.

        Returns:
            torch.Tensor: R-fraction value.
        """
        attract_norm_squared = torch.sum(attract_z**2)
        repel_norm_squared = torch.sum(repel_z**2)

        return repel_norm_squared / (attract_norm_squared + repel_norm_squared)

    def forward(
            self, x: torch.Tensor, edge_index: Adj,
            edge_weight: OptTensor = None, edge_label_index: OptTensor = None,
            **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: The node features.
            edge_index: The edge indices.
            edge_weight: One-dimensional edge weights.
            edge_label_index: Indices of edges for which to compute scores.
            **kwargs: Additional arguments forwarded to the base model.

        Returns:
            If edge_label_index or edge_index is provided as the target for prediction:
                Attract-repel scores for the provided edges.
            Otherwise:
                The concatenated attract-repel node embeddings.
        """
        # Get attract and repel embeddings
        attract_z, repel_z = self.encode(x, edge_index, edge_weight, **kwargs)

        # If edge_label_index is provided, compute scores for those edges
        if edge_label_index is not None:
            src, dst = edge_label_index
            return self.calculate_scores(attract_z[src], attract_z[dst],
                                         repel_z[src], repel_z[dst])

        # If edge_index is specified again specifically for prediction
        elif 'edge_index' in kwargs:
            src, dst = kwargs['edge_index']
            return self.calculate_scores(attract_z[src], attract_z[dst],
                                         repel_z[src], repel_z[dst])

        # Otherwise return full node embeddings (concatenated)
        else:
            return torch.cat([attract_z, repel_z], dim=1)
