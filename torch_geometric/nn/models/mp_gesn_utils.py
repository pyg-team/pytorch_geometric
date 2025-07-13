"""
Utility functions for Multi-Perspective Graph Echo State Networks.

This module provides helper functions for data preprocessing, graph construction,
and model analysis specifically designed for MP-GESN applications.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import warnings


def preprocess_eeg_data(
    data: torch.Tensor,
    sampling_rate: float = 250.0,
    bandpass_freq: Optional[Tuple[float, float]] = (0.5, 50.0),
    normalize: bool = True,
    artifact_threshold: float = 5.0
) -> torch.Tensor:
    """
    Preprocess EEG data for MP-GESN input.
    
    Args:
        data (torch.Tensor): Raw EEG data of shape (batch, time, channels)
        sampling_rate (float): Sampling rate in Hz
        bandpass_freq (Optional[Tuple]): Bandpass filter frequencies (low, high)
        normalize (bool): Whether to normalize the data
        artifact_threshold (float): Z-score threshold for artifact detection
        
    Returns:
        torch.Tensor: Preprocessed EEG data
    """
    processed_data = data.clone()
    
    # Artifact removal based on amplitude
    if artifact_threshold > 0:
        # Compute z-scores across time dimension
        mean_vals = processed_data.mean(dim=1, keepdim=True)
        std_vals = processed_data.std(dim=1, keepdim=True)
        z_scores = torch.abs((processed_data - mean_vals) / (std_vals + 1e-8))
        
        # Mask artifacts
        artifact_mask = z_scores > artifact_threshold
        if artifact_mask.any():
            warnings.warn(f"Found {artifact_mask.sum().item()} artifact samples")
            # Interpolate artifacts (simple linear interpolation)
            processed_data[artifact_mask] = 0  # Simple approach
    
    # Bandpass filtering (simplified - in practice use proper filters)
    if bandpass_freq is not None:
        # This is a simplified filtering - for real applications use scipy.signal
        # Apply high-pass by removing DC component
        processed_data = processed_data - processed_data.mean(dim=1, keepdim=True)
        
        # Simple low-pass by smoothing (not ideal but works for demo)
        if bandpass_freq[1] < sampling_rate / 2:
            kernel_size = int(sampling_rate / bandpass_freq[1])
            if kernel_size > 1 and kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size > 1:
                processed_data = F.avg_pool1d(
                    processed_data.transpose(1, 2), 
                    kernel_size=kernel_size, 
                    stride=1, 
                    padding=kernel_size//2
                ).transpose(1, 2)
    
    # Normalization
    if normalize:
        # Z-score normalization per channel
        mean_vals = processed_data.mean(dim=1, keepdim=True)
        std_vals = processed_data.std(dim=1, keepdim=True)
        processed_data = (processed_data - mean_vals) / (std_vals + 1e-8)
    
    return processed_data


def create_eeg_10_20_graph(selected_channels: Optional[List[str]] = None) -> torch.Tensor:
    """
    Create electrode connectivity graph based on 10-20 system.
    
    Args:
        selected_channels (Optional[List[str]]): List of channel names to include
        
    Returns:
        torch.Tensor: Edge index tensor for 10-20 electrode connectivity
    """
    # Standard 10-20 system electrode positions (simplified)
    electrode_positions = {
        'Fp1': (0, 0), 'Fp2': (0, 1), 'F7': (1, -1), 'F3': (1, 0),
        'Fz': (1, 1), 'F4': (1, 2), 'F8': (1, 3), 'T3': (2, -1),
        'C3': (2, 0), 'Cz': (2, 1), 'C4': (2, 2), 'T4': (2, 3),
        'T5': (3, -1), 'P3': (3, 0), 'Pz': (3, 1), 'P4': (3, 2),
        'T6': (3, 3), 'O1': (4, 0), 'O2': (4, 1)
    }
    
    if selected_channels is None:
        selected_channels = list(electrode_positions.keys())
    
    # Filter positions for selected channels
    positions = {ch: electrode_positions[ch] for ch in selected_channels 
                if ch in electrode_positions}
    
    if not positions:
        raise ValueError("No valid channels found in 10-20 system")
    
    # Create channel to index mapping
    ch_to_idx = {ch: idx for idx, ch in enumerate(positions.keys())}
    
    # Create edges based on spatial proximity
    edges = []
    for ch1, pos1 in positions.items():
        for ch2, pos2 in positions.items():
            if ch1 != ch2:
                # Calculate Manhattan distance
                dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                if dist <= 1.5:  # Connect adjacent electrodes
                    edges.append([ch_to_idx[ch1], ch_to_idx[ch2]])
    
    if not edges:
        # Fallback: create linear chain
        edges = [[i, i+1] for i in range(len(positions)-1)]
        edges.extend([[i+1, i] for i in range(len(positions)-1)])  # Bidirectional
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index


def analyze_reservoir_dynamics(
    model,
    data: torch.Tensor,
    edge_index: torch.Tensor,
    layer_idx: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Analyze reservoir dynamics for interpretability.
    
    Args:
        model: Trained MP-GESN model
        data (torch.Tensor): Input data of shape (batch, time, features)
        edge_index (torch.Tensor): Graph edge indices
        layer_idx (int): Which reservoir layer to analyze
        
    Returns:
        Dict[str, torch.Tensor]: Analysis results containing reservoir states,
                                eigenvalues, and activity statistics
    """
    model.eval()
    
    with torch.no_grad():
        # Get reservoir states
        reservoir_states = model.get_reservoir_states(data, edge_index)
        
        # Get specific layer weights
        reservoir_layer = model.encoder.reservoir_layers[layer_idx]
        reservoir_weights = reservoir_layer.W_reservoir.data
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(reservoir_weights)
        spectral_radius = torch.max(torch.abs(eigenvalues)).real
        
        # Analyze activity patterns
        activity_mean = reservoir_states.mean(dim=(0, 1))
        activity_std = reservoir_states.std(dim=(0, 1))
        activity_max = reservoir_states.max(dim=1)[0].max(dim=0)[0]
        activity_min = reservoir_states.min(dim=1)[0].min(dim=0)[0]
        
        # Population-specific analysis
        population_sizes = reservoir_layer.population_sizes
        population_activities = []
        start_idx = 0
        
        for pop_size in population_sizes:
            end_idx = start_idx + pop_size
            pop_activity = reservoir_states[:, :, start_idx:end_idx]
            population_activities.append({
                'mean_activity': pop_activity.mean(),
                'std_activity': pop_activity.std(),
                'max_activity': pop_activity.max(),
                'min_activity': pop_activity.min()
            })
            start_idx = end_idx
    
    return {
        'reservoir_states': reservoir_states,
        'eigenvalues': eigenvalues,
        'spectral_radius': spectral_radius,
        'activity_statistics': {
            'mean': activity_mean,
            'std': activity_std,
            'max': activity_max,
            'min': activity_min
        },
        'population_activities': population_activities
    }


def create_sliding_windows(
    data: torch.Tensor,
    window_size: int,
    stride: int = 1,
    target_delay: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sliding windows for time series prediction.
    
    Args:
        data (torch.Tensor): Input data of shape (batch, time, features)
        window_size (int): Size of input windows
        stride (int): Stride between windows
        target_delay (int): Number of steps ahead to predict
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input windows and targets
    """
    batch_size, seq_len, n_features = data.shape
    
    # Calculate number of windows
    n_windows = (seq_len - window_size - target_delay) // stride + 1
    
    if n_windows <= 0:
        raise ValueError(f"Sequence too short for window_size={window_size} and target_delay={target_delay}")
    
    # Create input windows
    inputs = torch.zeros(batch_size * n_windows, window_size, n_features)
    targets = torch.zeros(batch_size * n_windows, n_features)
    
    idx = 0
    for b in range(batch_size):
        for i in range(0, seq_len - window_size - target_delay + 1, stride):
            if idx >= batch_size * n_windows:
                break
            inputs[idx] = data[b, i:i+window_size]
            targets[idx] = data[b, i+window_size+target_delay-1]
            idx += 1
    
    return inputs[:idx], targets[:idx]


def compute_frequency_features(
    data: torch.Tensor,
    sampling_rate: float = 250.0,
    freq_bands: Optional[List[Tuple[float, float]]] = None
) -> torch.Tensor:
    """
    Compute frequency-domain features for each channel.
    
    Args:
        data (torch.Tensor): Input data of shape (batch, time, channels)
        sampling_rate (float): Sampling rate in Hz
        freq_bands (Optional[List]): Frequency bands to analyze
        
    Returns:
        torch.Tensor: Frequency features of shape (batch, channels, n_bands)
    """
    if freq_bands is None:
        freq_bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]  # Delta, Theta, Alpha, Beta, Gamma
    
    batch_size, seq_len, n_channels = data.shape
    n_bands = len(freq_bands)
    
    # Compute FFT
    fft_data = torch.fft.fft(data, dim=1)
    power_spectrum = torch.abs(fft_data) ** 2
    
    # Frequency resolution
    freqs = torch.fft.fftfreq(seq_len, 1/sampling_rate)
    
    # Extract power in each frequency band
    band_powers = torch.zeros(batch_size, n_channels, n_bands)
    
    for band_idx, (low_freq, high_freq) in enumerate(freq_bands):
        # Find frequency indices
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
        
        if freq_mask.any():
            # Sum power in frequency band
            band_powers[:, :, band_idx] = power_spectrum[:, freq_mask, :].sum(dim=1)
        else:
            warnings.warn(f"No frequencies found in band {low_freq}-{high_freq} Hz")
    
    return band_powers


def validate_graph_structure(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """
    Validate graph structure for MP-GESN compatibility.
    
    Args:
        edge_index (torch.Tensor): Edge indices
        num_nodes (int): Expected number of nodes
        
    Returns:
        bool: True if graph structure is valid
    """
    if edge_index.numel() == 0:
        return True  # Empty graph is valid
    
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        return False
    
    if edge_index.max().item() >= num_nodes:
        return False
    
    if edge_index.min().item() < 0:
        return False
    
    return True


def estimate_optimal_parameters(
    data: torch.Tensor,
    target_spectral_radius: float = 0.9
) -> Dict[str, Union[int, float]]:
    """
    Estimate optimal MP-GESN parameters based on data characteristics.
    
    Args:
        data (torch.Tensor): Sample data of shape (batch, time, features)
        target_spectral_radius (float): Target spectral radius for reservoir
        
    Returns:
        Dict: Recommended parameters
    """
    batch_size, seq_len, n_features = data.shape
    
    # Estimate temporal complexity
    autocorr = torch.zeros(min(seq_len//4, 50))
    for lag in range(len(autocorr)):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            x1 = data[:, :-lag, :].flatten()
            x2 = data[:, lag:, :].flatten()
            autocorr[lag] = torch.corrcoef(torch.stack([x1, x2]))[0, 1]
    
    # Find autocorrelation decay
    decay_point = 1
    for i in range(1, len(autocorr)):
        if autocorr[i] < 0.5:
            decay_point = i
            break
    
    # Estimate data complexity
    data_std = data.std().item()
    data_range = (data.max() - data.min()).item()
    
    # Recommend parameters
    hidden_size = min(max(n_features * 2, 64), 512)
    num_layers = 2 if decay_point > 10 else 1
    leaking_rate = 0.3 if decay_point < 5 else 0.1
    
    return {
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_populations': 4,
        'spectral_radius': target_spectral_radius,
        'leaking_rate': leaking_rate,
        'input_scaling': 1.0 / data_std if data_std > 0 else 1.0,
        'bias_scaling': 0.1,
        'autocorr_decay': decay_point,
        'data_complexity': data_range / data_std if data_std > 0 else 1.0
    }