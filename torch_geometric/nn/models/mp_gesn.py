"""
Multi-Perspective Graph Echo State Network (MP-GESN) for PyTorch Geometric.

This module implements a novel architecture that combines graph-structured
reservoir computing with multi-perspective attention mechanisms for neural
signal processing and multivariate time series prediction.

Author: Venkiteswaran, K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple, Dict, Any
import math


class GraphReservoirCell(MessagePassing):
    """
    Graph-structured reservoir computing cell with specialized neuron populations.
    
    This cell implements a reservoir with multiple neuron populations, each
    specialized for different types of dynamics (e.g., low-frequency, high-frequency,
    memory, adaptation).
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden state
        num_populations (int): Number of specialized neuron populations
        spectral_radius (float): Target spectral radius for reservoir weights
        leaking_rate (float): Leaking rate for reservoir dynamics
        input_scaling (float): Scaling factor for input weights
        bias_scaling (float): Scaling factor for bias terms
        population_ratios (Optional[List[float]]): Ratio of neurons in each population
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_populations: int = 4,
        spectral_radius: float = 0.9,
        leaking_rate: float = 0.1,
        input_scaling: float = 1.0,
        bias_scaling: float = 0.1,
        population_ratios: Optional[list] = None,
        **kwargs
    ):
        super().__init__(aggr='add', **kwargs)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_populations = num_populations
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        
        # Default population ratios
        if population_ratios is None:
            population_ratios = [1.0 / num_populations] * num_populations
        self.population_ratios = population_ratios
        
        # Calculate population sizes
        self.population_sizes = [
            int(hidden_size * ratio) for ratio in population_ratios
        ]
        # Adjust for rounding errors
        diff = hidden_size - sum(self.population_sizes)
        if diff > 0:
            self.population_sizes[0] += diff
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize reservoir weights with proper scaling."""
        # Input weights
        self.W_in = nn.Parameter(
            torch.randn(self.hidden_size, self.input_size) * self.input_scaling,
            requires_grad=False
        )
        
        # Reservoir weights (recurrent)
        W_reservoir = torch.randn(self.hidden_size, self.hidden_size)
        
        # Apply population-specific connectivity patterns
        self._apply_population_structure(W_reservoir)
        
        # Normalize to target spectral radius
        eigenvalues = torch.linalg.eigvals(W_reservoir)
        current_radius = torch.max(torch.abs(eigenvalues)).real
        W_reservoir = W_reservoir * (self.spectral_radius / current_radius)
        
        self.W_reservoir = nn.Parameter(W_reservoir, requires_grad=False)
        
        # Bias terms
        self.bias = nn.Parameter(
            torch.randn(self.hidden_size) * self.bias_scaling,
            requires_grad=False
        )
        
    def _apply_population_structure(self, W: torch.Tensor):
        """Apply population-specific connectivity patterns."""
        start_idx = 0
        
        for i, pop_size in enumerate(self.population_sizes):
            end_idx = start_idx + pop_size
            
            # Population-specific connectivity
            if i == 0:  # Low-frequency population
                # Stronger self-connections, weaker cross-connections
                W[start_idx:end_idx, start_idx:end_idx] *= 1.2
                W[start_idx:end_idx, :start_idx] *= 0.5
                W[start_idx:end_idx, end_idx:] *= 0.5
            elif i == 1:  # High-frequency population
                # Weaker self-connections, stronger cross-connections
                W[start_idx:end_idx, start_idx:end_idx] *= 0.8
                W[start_idx:end_idx, :start_idx] *= 1.3
                W[start_idx:end_idx, end_idx:] *= 1.3
            elif i == 2:  # Memory population
                # Strong self-connections, moderate cross-connections
                W[start_idx:end_idx, start_idx:end_idx] *= 1.5
                W[start_idx:end_idx, :start_idx] *= 0.7
                W[start_idx:end_idx, end_idx:] *= 0.7
            else:  # Adaptation population
                # Moderate all connections
                W[start_idx:end_idx, :] *= 1.0
                
            start_idx = end_idx
    
    def _get_num_nodes_from_edge_index(self, edge_index: torch.Tensor) -> int:
        """Get the number of nodes from edge_index."""
        if edge_index.numel() == 0:
            return 1
        return int(edge_index.max().item()) + 1
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the graph reservoir cell.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            edge_index (torch.Tensor): Graph edge indices
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, input_size = x.shape
        
        # Get number of nodes from edge_index
        num_nodes = self._get_num_nodes_from_edge_index(edge_index)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        
        # Create projection layer for graph processing if needed
        if not hasattr(self, 'graph_proj'):
            self.graph_proj = nn.Linear(self.hidden_size, num_nodes).to(x.device)
            self.graph_unproj = nn.Linear(num_nodes, self.hidden_size).to(x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Current input
            x_t = x[:, t, :]  # (batch_size, input_size)
            
            # Input transformation
            input_contrib = torch.matmul(x_t, self.W_in.t())  # (batch_size, hidden_size)
            
            # Reservoir dynamics
            reservoir_contrib = torch.matmul(h, self.W_reservoir.t())  # (batch_size, hidden_size)
            
            # Graph message passing
            if edge_index.numel() > 0 and num_nodes > 1:
                # Project hidden state to graph nodes
                h_graph = self.graph_proj(h)  # (batch_size, num_nodes)
                
                # Reshape for message passing: (batch_size * num_nodes,)
                h_flat = h_graph.reshape(-1)  # (batch_size * num_nodes,)
                
                # Adjust edge_index for batched processing
                batch_edge_index = edge_index.clone()
                for b in range(1, batch_size):
                    batch_offset = b * num_nodes
                    batch_edge_offset = torch.tensor([[batch_offset], [batch_offset]], 
                                                   device=edge_index.device, dtype=edge_index.dtype)
                    batch_edge_index = torch.cat([
                        batch_edge_index, 
                        edge_index + batch_edge_offset
                    ], dim=1)
                
                # Apply message passing
                h_graph_flat = self.propagate(batch_edge_index, x=h_flat.unsqueeze(-1))
                
                # Reshape back to (batch_size, num_nodes)
                h_graph_new = h_graph_flat.squeeze(-1).reshape(batch_size, num_nodes)
                
                # Project back to hidden size
                graph_contrib = self.graph_unproj(h_graph_new)  # (batch_size, hidden_size)
            else:
                graph_contrib = torch.zeros_like(h)
            
            # Update hidden state
            h_new = torch.tanh(input_contrib + reservoir_contrib + graph_contrib + self.bias)
            
            # Apply leaking rate
            h = (1 - self.leaking_rate) * h + self.leaking_rate * h_new
            
            outputs.append(h.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Message function for graph neural network."""
        return x_j
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update function for graph neural network."""
        return aggr_out


class MultiPerspectiveAttention(nn.Module):
    """
    Multi-perspective attention mechanism for channel-specific learning.
    
    This attention mechanism learns different perspectives for each input channel
    while maintaining shared temporal dynamics.
    
    Args:
        hidden_size (int): Size of hidden representations
        num_channels (int): Number of input channels
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Channel-specific transformations
        self.channel_queries = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_channels)
        ])
        
        self.channel_keys = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_channels)
        ])
        
        self.channel_values = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_channels)
        ])
        
        # Shared output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-perspective attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Assume channels are embedded in the sequence dimension
        # This is a simplification - in practice, you'd need to handle
        # channel information more explicitly
        
        # For now, we'll use a simple multi-head self-attention
        # You can extend this to be truly channel-specific
        
        # Multi-head self-attention
        q = k = v = x
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        # Output projection
        output = self.output_proj(attended)
        
        return output


class MPGESNEncoder(nn.Module):
    """
    Multi-layer encoder using Graph Reservoir Cells.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden representations
        num_layers (int): Number of encoder layers
        num_populations (int): Number of neuron populations per layer
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        num_populations: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create reservoir layers
        self.reservoir_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.reservoir_layers.append(
                GraphReservoirCell(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_populations=num_populations,
                    spectral_radius=0.9 - i * 0.1,  # Decrease spectral radius with depth
                    leaking_rate=0.1 + i * 0.05,   # Increase leaking rate with depth
                )
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            edge_index (torch.Tensor): Graph edge indices
            
        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, seq_len, hidden_size)
        """
        hidden = x
        
        for layer in self.reservoir_layers:
            hidden = layer(hidden, edge_index)
            hidden = self.dropout(hidden)
        
        return hidden


class MultiPerspectiveGraphESN(nn.Module):
    """
    Multi-Perspective Graph Echo State Network for neural signal processing.
    
    This model combines graph-structured reservoir computing with multi-perspective
    attention mechanisms to learn from neural signals and multivariate time series.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden representations
        output_size (int): Size of output predictions
        num_channels (int): Number of input channels
        num_layers (int): Number of encoder layers
        num_populations (int): Number of neuron populations per layer
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        use_attention (bool): Whether to use multi-perspective attention
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_channels: int,
        num_layers: int = 2,
        num_populations: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Encoder
        self.encoder = MPGESNEncoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_populations=num_populations,
            dropout=dropout
        )
        
        # Multi-perspective attention
        if use_attention:
            self.attention = MultiPerspectiveAttention(
                hidden_size=hidden_size,
                num_channels=num_channels,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MP-GESN model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            edge_index (torch.Tensor): Graph edge indices
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, seq_len, output_size)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Encode through graph reservoir
        encoded = self.encoder(x, edge_index)
        
        # Apply attention if enabled
        if self.use_attention:
            encoded = self.attention(encoded)
        
        # Layer normalization
        encoded = self.layer_norm(encoded)
        
        # Output projection
        output = self.output_proj(encoded)
        
        return output
    
    def get_reservoir_states(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get reservoir states for analysis.
        
        Args:
            x (torch.Tensor): Input tensor
            edge_index (torch.Tensor): Graph edge indices
            
        Returns:
            torch.Tensor: Reservoir states
        """
        x = self.input_proj(x)
        return self.encoder(x, edge_index)


class MPGESNLoss(nn.Module):
    """
    Multi-component loss function for MP-GESN training.
    
    This loss combines time-domain prediction error, frequency-domain consistency,
    and temporal consistency regularization.
    
    Args:
        time_weight (float): Weight for time-domain loss
        freq_weight (float): Weight for frequency-domain loss
        consistency_weight (float): Weight for consistency loss
        freq_bands (List[Tuple[float, float]]): Frequency bands for analysis
    """
    
    def __init__(
        self,
        time_weight: float = 1.0,
        freq_weight: float = 0.1,
        consistency_weight: float = 0.05,
        freq_bands: Optional[list] = None
    ):
        super().__init__()
        
        self.time_weight = time_weight
        self.freq_weight = freq_weight
        self.consistency_weight = consistency_weight
        
        if freq_bands is None:
            freq_bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100)]
        self.freq_bands = freq_bands
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-component loss.
        
        Args:
            pred (torch.Tensor): Predicted values of shape (batch_size, seq_len, output_size)
            target (torch.Tensor): Target values of shape (batch_size, seq_len, output_size)
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Ensure pred and target have same shape
        if pred.shape != target.shape:
            target = F.interpolate(
                target.transpose(1, 2), 
                size=pred.size(1)
            ).transpose(1, 2)
        
        total_loss = 0.0
        
        # Time-domain loss
        if self.time_weight > 0:
            time_loss = F.mse_loss(pred, target)
            total_loss += self.time_weight * time_loss
        
        # Frequency-domain loss  
        if self.freq_weight > 0:
            freq_loss = self._compute_frequency_loss(pred, target)
            total_loss += self.freq_weight * freq_loss
            
        # Consistency loss
        if self.consistency_weight > 0:
            consistency_loss = self._compute_consistency_loss(pred, target)
            total_loss += self.consistency_weight * consistency_loss
            
        return total_loss
    
    def _compute_frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency-domain loss using FFT.
        
        Args:
            pred (torch.Tensor): Predicted values
            target (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Frequency-domain loss
        """
        # Ensure minimum sequence length for FFT
        seq_len = pred.size(1)
        if seq_len < 8:
            # Pad sequences to minimum length
            pad_size = 8 - seq_len
            pred = F.pad(pred, (0, 0, 0, pad_size))
            target = F.pad(target, (0, 0, 0, pad_size))
        
        # Compute FFT
        pred_fft = torch.fft.fft(pred, dim=1)
        target_fft = torch.fft.fft(target, dim=1)
        
        # Compute magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Compute frequency-domain MSE
        freq_loss = F.mse_loss(pred_mag, target_mag)
        
        return freq_loss
    
    def _compute_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            pred (torch.Tensor): Predicted values
            target (torch.Tensor): Target values
            
        Returns:
            torch.Tensor: Consistency loss
        """
        # Compute temporal differences
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        # Compute consistency loss
        consistency_loss = F.mse_loss(pred_diff, target_diff)
        
        return consistency_loss


def create_electrode_graph(num_electrodes: int, electrode_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Create electrode connectivity graph for EEG/neural data.
    
    Args:
        num_electrodes (int): Number of electrodes
        electrode_positions (Optional[torch.Tensor]): 3D positions of electrodes
        
    Returns:
        torch.Tensor: Edge index tensor for graph connectivity
    """
    if electrode_positions is not None:
        # Create graph based on spatial proximity
        distances = torch.cdist(electrode_positions, electrode_positions)
        
        # Connect each electrode to its k nearest neighbors
        k = min(4, num_electrodes - 1)
        _, indices = torch.topk(distances, k + 1, dim=1, largest=False)
        
        # Remove self-connections
        indices = indices[:, 1:]
        
        # Create edge index
        source = torch.arange(num_electrodes).repeat_interleave(k)
        target = indices.flatten()
        
        edge_index = torch.stack([source, target], dim=0)
    else:
        # Create simple ring topology
        source = torch.arange(num_electrodes)
        target = (torch.arange(num_electrodes) + 1) % num_electrodes
        
        # Add reverse edges for undirected graph
        edge_index = torch.stack([
            torch.cat([source, target]),
            torch.cat([target, source])
        ], dim=0)
    
    return edge_index


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for MP-GESN.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        'model': {
            'input_size': 64,
            'hidden_size': 128,
            'output_size': 32,
            'num_channels': 64,
            'num_layers': 2,
            'num_populations': 4,
            'num_heads': 8,
            'dropout': 0.1,
            'use_attention': True
        },
        'reservoir': {
            'spectral_radius': 0.9,
            'leaking_rate': 0.1,
            'input_scaling': 1.0,
            'bias_scaling': 0.1,
            'population_ratios': [0.3, 0.3, 0.2, 0.2]
        },
        'loss': {
            'time_weight': 1.0,
            'freq_weight': 0.1,
            'consistency_weight': 0.05,
            'freq_bands': [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100)]
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'max_epochs': 100,
            'patience': 10
        }
    }


# Example usage and utility functions
def example_usage():
    """
    Example usage of the MP-GESN model.
    """
    # Create model
    model = MultiPerspectiveGraphESN(
        input_size=64,
        hidden_size=128,
        output_size=32,
        num_channels=64,
        num_layers=2
    )
    
    # Create sample data
    batch_size, seq_len, input_size = 8, 100, 64
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Create electrode graph
    edge_index = create_electrode_graph(num_electrodes=8)
    
    # Forward pass
    output = model(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Create loss function
    loss_fn = MPGESNLoss()
    target = torch.randn_like(output)
    loss = loss_fn(output, target)
    print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    example_usage()