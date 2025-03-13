import torch
import torch.nn as nn
from torch_geometric.nn.models.basic_gnn import GCN, GAT, GraphSAGE

class AttractRepel(torch.nn.Module):
    def __init__(self, base_model, in_channels=None, hidden_channels=None, 
                 out_channels=None, num_layers=2, attract_ratio=0.5, **kwargs):
        super(AttractRepel, self).__init__()
        
        # Store parameters
        self.attract_ratio = attract_ratio
        
        # Initialize base model
        if isinstance(base_model, str):
            # Verify required parameters
            if in_channels is None or hidden_channels is None:
                raise ValueError("in_channels and hidden_channels must be provided when base_model is a string")
            
            # Create model based on string name
            if base_model == 'GCN':
                from torch_geometric.nn.models.basic_gnn import GCN
                self.base_model = GCN(in_channels=in_channels, 
                                     hidden_channels=hidden_channels,
                                     num_layers=num_layers,
                                     out_channels=hidden_channels,  # Set output to hidden_channels
                                     **kwargs)
            elif base_model == 'GAT':
                from torch_geometric.nn.models.basic_gnn import GAT
                self.base_model = GAT(in_channels=in_channels, 
                                     hidden_channels=hidden_channels,
                                     num_layers=num_layers,
                                     out_channels=hidden_channels,  # Set output to hidden_channels
                                     **kwargs)
            elif base_model == 'GraphSAGE':
                from torch_geometric.nn.models.basic_gnn import GraphSAGE
                self.base_model = GraphSAGE(in_channels=in_channels, 
                                           hidden_channels=hidden_channels,
                                           num_layers=num_layers,
                                           out_channels=hidden_channels,  # Set output to hidden_channels
                                           **kwargs)
            else:
                raise ValueError(f"Unknown model: {base_model}. "
                                f"Supported models: 'GCN', 'GAT', 'GraphSAGE'")
        else:
            self.base_model = base_model
        
        # Set output dimensions
        base_out_channels = hidden_channels  # Use hidden_channels as safe default
        self.out_channels = out_channels if out_channels is not None else base_out_channels
        
        # Calculate dimensions for attract and repel components
        self.attract_dim = int(self.out_channels * attract_ratio)
        self.repel_dim = self.out_channels - self.attract_dim
        
        # Create projection layers
        self.proj_attract = nn.Linear(base_out_channels, self.attract_dim)
        self.proj_repel = nn.Linear(base_out_channels, self.repel_dim)
        
        print(f"Debug: Model initialized with: out_channels={self.out_channels}, "
              f"attract_dim={self.attract_dim}, repel_dim={self.repel_dim}")
    def encode(self, *args, **kwargs):
        """
        Encode inputs using the base model and project to attract-repel space.
        
        Args:
            *args: Arguments to pass to the base model
            **kwargs: Keyword arguments to pass to the base model
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Attract and repel embeddings
        """
        # Get embeddings from base model
        x = self.base_model(*args, **kwargs)
        
        # Project to attract and repel spaces
        attract_emb = self.proj_attract(x)
        repel_emb = self.proj_repel(x)
        
        return attract_emb, repel_emb
    
    def decode(self, attract_z, repel_z, edge_index):
        """
        Compute link prediction scores for given edges.
        
        Args:
            attract_z (torch.Tensor): Attract embeddings of shape [num_nodes, attract_dim]
            repel_z (torch.Tensor): Repel embeddings of shape [num_nodes, repel_dim]
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges]
            
        Returns:
            torch.Tensor: Link prediction scores for each edge
        """
        # Get node pairs
        row, col = edge_index
        
        # Get embeddings for the node pairs
        attract_z_i, attract_z_j = attract_z[row], attract_z[col]
        repel_z_i, repel_z_j = repel_z[row], repel_z[col]
        
        # Compute the attract-repel dot products
        attract_score = (attract_z_i * attract_z_j).sum(dim=1)
        repel_score = (repel_z_i * repel_z_j).sum(dim=1)
        
        # Return the final score
        return attract_score - repel_score
    
    def forward(self, *args, edge_index=None, **kwargs):
        """
        Forward pass for link prediction.
        
        Args:
            *args: Arguments to pass to the base model
            edge_index (torch.Tensor, optional): Edge indices to predict.
                If not provided, returns node embeddings.
            **kwargs: Keyword arguments to pass to the base model
            
        Returns:
            torch.Tensor or tuple: If edge_index is provided, returns link prediction
                scores for each edge. Otherwise, returns node embeddings.
        """
        # Get node embeddings
        attract_z, repel_z = self.encode(*args, **kwargs)
        
        # If edge_index is provided, decode edges
        if edge_index is not None:
            return self.decode(attract_z, repel_z, edge_index)
        
        # Otherwise, return embeddings
        return torch.cat([attract_z, repel_z], dim=1)
    
    def calculate_r_fraction(self, attract_z, repel_z):
        """
        Calculate the R-fraction of the embeddings.
        
        Args:
            attract_z (torch.Tensor): Attract embeddings
            repel_z (torch.Tensor): Repel embeddings
            
        Returns:
            float: R-fraction value
        """
        attract_norm_squared = torch.sum(attract_z ** 2)
        repel_norm_squared = torch.sum(repel_z ** 2)
        
        r_fraction = repel_norm_squared / (attract_norm_squared + repel_norm_squared + 1e-10)
        
        return r_fraction.item()