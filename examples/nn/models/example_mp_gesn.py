"""
Example script demonstrating the Multi-Perspective Graph ESN for neural signal processing.

This example shows how to use MP-GESN for EEG signal prediction and classification tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time
import argparse
from typing import Dict, Tuple, List

# Import MP-GESN components
from torch_geometric.nn.models.mp_gesn import (
    MultiPerspectiveGraphESN,
    MPGESNLoss,
    create_electrode_graph,
    create_default_config
)


def generate_synthetic_eeg_data(
    num_samples: int = 1000,
    num_channels: int = 64,
    seq_length: int = 200,
    sampling_rate: int = 250,
    noise_level: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic EEG-like data for testing.
    
    Args:
        num_samples (int): Number of samples to generate
        num_channels (int): Number of EEG channels
        seq_length (int): Length of each sequence
        sampling_rate (int): Sampling rate in Hz
        noise_level (float): Noise level
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input data and targets
    """
    # Time vector
    t = torch.linspace(0, seq_length / sampling_rate, seq_length)
    
    # Generate frequency components
    frequencies = [1, 4, 8, 12, 20, 30]  # Different brain rhythms
    
    data = []
    targets = []
    
    for sample in range(num_samples):
        # Create multi-channel signal
        signal = torch.zeros(seq_length, num_channels)
        
        for ch in range(num_channels):
            # Mix different frequency components
            channel_signal = torch.zeros(seq_length)
            
            for freq in frequencies:
                # Random amplitude and phase for each frequency
                amplitude = torch.rand(1) * 0.5 + 0.1
                phase = torch.rand(1) * 2 * np.pi
                
                # Add frequency component
                channel_signal += amplitude * torch.sin(2 * np.pi * freq * t + phase)
            
            # Add spatial correlation between nearby channels
            if ch > 0:
                channel_signal += 0.3 * signal[:, ch - 1]
            
            # Add noise
            channel_signal += noise_level * torch.randn(seq_length)
            
            signal[:, ch] = channel_signal
        
        # Create target as prediction of next few time steps
        target = signal[10:, :]  # Predict 10 steps ahead
        input_data = signal[:-10, :]
        
        data.append(input_data.unsqueeze(0))
        targets.append(target.unsqueeze(0))
    
    return torch.cat(data, dim=0), torch.cat(targets, dim=0)


def create_train_test_split(
    data: torch.Tensor,
    targets: torch.Tensor,
    train_ratio: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data into training and testing sets.
    
    Args:
        data (torch.Tensor): Input data
        targets (torch.Tensor): Target data
        train_ratio (float): Ratio of training data
        
    Returns:
        Tuple: Train and test data splits
    """
    num_samples = data.size(0)
    num_train = int(num_samples * train_ratio)
    
    # Random shuffle
    indices = torch.randperm(num_samples)
    
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    train_data = data[train_indices]
    train_targets = targets[train_indices]
    test_data = data[test_indices]
    test_targets = targets[test_indices]
    
    return train_data, train_targets, test_data, test_targets


def train_model(
    model: MultiPerspectiveGraphESN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index: torch.Tensor,
    config: Dict,
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Train the MP-GESN model.
    
    Args:
        model: The MP-GESN model
        train_loader: Training data loader
        val_loader: Validation data loader
        edge_index: Graph edge indices
        config: Training configuration
        device: Device to train on
        
    Returns:
        Dict: Training history
    """
    # Loss function and optimizer
    criterion = MPGESNLoss(**config['loss'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': []
    }
    
    # Move edge_index to device
    edge_index = edge_index.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        
        for batch_data, batch_targets in train_loader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_data, edge_index)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += nn.functional.mse_loss(outputs, batch_targets).item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for batch_data, batch_targets in val_loader:
                batch_data = batch_data.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_data, edge_index)
                loss = criterion(outputs, batch_targets)
                
                val_loss += loss.item()
                val_mse += nn.functional.mse_loss(outputs, batch_targets).item()
        
        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_mse /= len(train_loader)
        val_mse /= len(val_loader)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Train MSE: {train_mse:.4f}, "
                  f"Val MSE: {val_mse:.4f}")
    
    return history


def evaluate_model(
    model: MultiPerspectiveGraphESN,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        edge_index: Graph edge indices
        device: Device to evaluate on
        
    Returns:
        Dict: Evaluation metrics
    """
    model.eval()
    edge_index = edge_index.to(device)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data, batch_targets in test_loader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_data, edge_index)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(batch_targets.cpu())
    
    # Concatenate all predictions and targets
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    mse = mean_squared_error(targets.numpy().flatten(), predictions.numpy().flatten())
    r2 = r2_score(targets.numpy().flatten(), predictions.numpy().flatten())
    
    # Calculate correlation
    pred_flat = predictions.numpy().flatten()
    target_flat = targets.numpy().flatten()
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    return {
        'mse': mse,
        'r2': r2,
        'correlation': correlation,
        'rmse': np.sqrt(mse)
    }


def plot_results(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training results.
    
    Args:
        history: Training history
        save_path: Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # MSE curves
    ax2.plot(history['train_mse'], label='Train MSE', color='blue')
    ax2.plot(history['val_mse'], label='Validation MSE', color='red')
    ax2.set_title('Training and Validation MSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.legend()
    ax2.grid(True)
    
    # Combined plot
    ax3.plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.7)
    ax3.plot(history['val_loss'], label='Val Loss', color='red', alpha=0.7)
    ax3.set_title('Training Progress')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)
    
    # Final metrics summary
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_mse = history['train_mse'][-1]
    final_val_mse = history['val_mse'][-1]
    
    ax4.text(0.1, 0.8, f'Final Train Loss: {final_train_loss:.4f}', transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f'Final Val Loss: {final_val_loss:.4f}', transform=ax4.transAxes)
    ax4.text(0.1, 0.4, f'Final Train MSE: {final_train_mse:.4f}', transform=ax4.transAxes)
    ax4.text(0.1, 0.2, f'Final Val MSE: {final_val_mse:.4f}', transform=ax4.transAxes)
    ax4.set_title('Final Metrics')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """
    Main function to run the MP-GESN example.
    """
    parser = argparse.ArgumentParser(description='MP-GESN Example')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("Generating synthetic EEG data...")
    data, targets = generate_synthetic_eeg_data(
        num_samples=800,
        num_channels=32,
        seq_length=128,
        sampling_rate=250
    )
    
    print(f"Data shape: {data.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Split data
    train_data, train_targets, test_data, test_targets = create_train_test_split(
        data, targets, train_ratio=0.8
    )
    
    # Create data loaders
    train_dataset = TensorDataset(train_data, train_targets)
    test_dataset = TensorDataset(test_data, test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create validation loader from training data
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    
    # Create electrode graph - use number of nodes that matches expected usage
    num_graph_nodes = 8  # Use a reasonable number of graph nodes
    edge_index = create_electrode_graph(num_electrodes=num_graph_nodes)
    
    # Load configuration
    config = create_default_config()
    config['model']['input_size'] = data.shape[2]
    config['model']['output_size'] = targets.shape[2]
    config['model']['num_channels'] = data.shape[2]
    config['training']['batch_size'] = args.batch_size
    config['training']['max_epochs'] = args.num_epochs
    config['training']['learning_rate'] = args.learning_rate
    
    # Create model
    print("Creating MP-GESN model...")
    model = MultiPerspectiveGraphESN(**config['model'])
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("Training model...")
    start_time = time.time()
    
    history = train_model(
        model=model,
        train_loader=DataLoader(train_subset, batch_size=args.batch_size, shuffle=True),
        val_loader=val_loader,
        edge_index=edge_index,
        config=config,
        device=device
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = evaluate_model(model, test_loader, edge_index, device)
    
    print("Evaluation Results:")
    print(f"MSE: {eval_results['mse']:.6f}")
    print(f"RMSE: {eval_results['rmse']:.6f}")
    print(f"R²: {eval_results['r2']:.6f}")
    print(f"Correlation: {eval_results['correlation']:.6f}")
    
    # Plot results
    if args.save_plots:
        plot_results(history, save_path='mp_gesn_training_results.png')
    else:
        plot_results(history)
    
    # Save model
    torch.save(model.state_dict(), 'mp_gesn_model.pth')
    print("Model saved to mp_gesn_model.pth")


if __name__ == "__main__":
    main()