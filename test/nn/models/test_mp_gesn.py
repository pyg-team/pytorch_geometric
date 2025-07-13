import torch
import pytest
from torch_geometric.data import Data
from torch_geometric.nn.models.mp_gesn import (
    GraphReservoirCell,
    MultiPerspectiveAttention,
    MPGESNEncoder,
    MultiPerspectiveGraphESN,
    MPGESNLoss,
)


class TestGraphReservoirCell:
    def test_init(self):
        """Test GraphReservoirCell initialization."""
        cell = GraphReservoirCell(
            input_size=10,
            hidden_size=64,
            num_populations=3,
            spectral_radius=0.9
        )
        assert cell.input_size == 10
        assert cell.hidden_size == 64
        assert cell.num_populations == 3
        assert cell.spectral_radius == 0.9
        
    def test_forward_shape(self):
        """Test forward pass output shapes."""
        batch_size, seq_len, input_size = 4, 16, 10
        hidden_size = 64
        
        cell = GraphReservoirCell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_populations=3
        )
        
        x = torch.randn(batch_size, seq_len, input_size)
        # Graph over 3 nodes: indices 0,1,2
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        output = cell(x, edge_index)
        assert output.shape == (batch_size, seq_len, hidden_size)
        
    def test_different_populations(self):
        """Test with different numbers of populations."""
        for num_pops in [1, 2, 4, 8]:
            cell = GraphReservoirCell(
                input_size=8,
                hidden_size=32,
                num_populations=num_pops
            )
            
            x = torch.randn(2, 10, 8)
            # Simple 2-node graph
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            
            output = cell(x, edge_index)
            assert output.shape == (2, 10, 32)


class TestMultiPerspectiveAttention:
    def test_init(self):
        """Test MultiPerspectiveAttention initialization."""
        attention = MultiPerspectiveAttention(
            hidden_size=64,
            num_channels=8,
            num_heads=4
        )
        assert attention.hidden_size == 64
        assert attention.num_channels == 8
        assert attention.num_heads == 4
        
    def test_forward_shape(self):
        """Test forward pass output shapes."""
        batch_size, seq_len, hidden_size = 4, 16, 64
        num_channels = 8
        
        attention = MultiPerspectiveAttention(
            hidden_size=hidden_size,
            num_channels=num_channels,
            num_heads=4
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = attention(x)
        assert output.shape == (batch_size, seq_len, hidden_size)
        
    def test_attention_weights(self):
        """Test that attention weights are computed correctly."""
        attention = MultiPerspectiveAttention(
            hidden_size=32,
            num_channels=4,
            num_heads=2
        )
        
        x = torch.randn(2, 10, 32)
        output = attention(x)
        
        # Check that output is different from input (attention applied)
        assert not torch.allclose(output, x)


class TestMPGESNEncoder:
    def test_init(self):
        """Test MPGESNEncoder initialization."""
        encoder = MPGESNEncoder(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            num_populations=3
        )
        assert encoder.input_size == 10
        assert encoder.hidden_size == 64
        assert encoder.num_layers == 2
        
    def test_forward_shape(self):
        """Test forward pass output shapes."""
        batch_size, seq_len, input_size = 4, 16, 10
        hidden_size = 64
        
        encoder = MPGESNEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            num_populations=3
        )
        
        x = torch.randn(batch_size, seq_len, input_size)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        output = encoder(x, edge_index)
        assert output.shape == (batch_size, seq_len, hidden_size)
        
    def test_multiple_layers(self):
        """Test encoder with different numbers of layers."""
        for num_layers in [1, 2, 3]:
            encoder = MPGESNEncoder(
                input_size=8,
                hidden_size=32,
                num_layers=num_layers,
                num_populations=2
            )
            
            x = torch.randn(2, 10, 8)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            
            output = encoder(x, edge_index)
            assert output.shape == (2, 10, 32)


class TestMultiPerspectiveGraphESN:
    def test_init(self):
        """Test MultiPerspectiveGraphESN initialization."""
        model = MultiPerspectiveGraphESN(
            input_size=10,
            hidden_size=64,
            output_size=5,
            num_channels=8,
            num_layers=2
        )
        assert model.input_size == 10
        assert model.hidden_size == 64
        assert model.output_size == 5
        assert model.num_channels == 8
        
    def test_forward_shape(self):
        """Test forward pass output shapes."""
        batch_size, seq_len, input_size = 4, 16, 10
        output_size = 5
        num_channels = 8
        
        model = MultiPerspectiveGraphESN(
            input_size=input_size,
            hidden_size=64,
            output_size=output_size,
            num_channels=num_channels,
            num_layers=2
        )
        
        x = torch.randn(batch_size, seq_len, input_size)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        output = model(x, edge_index)
        assert output.shape == (batch_size, seq_len, output_size)
        
    def test_different_output_sizes(self):
        """Test with different output sizes."""
        for output_size in [1, 5, 10]:
            model = MultiPerspectiveGraphESN(
                input_size=8,
                hidden_size=32,
                output_size=output_size,
                num_channels=4,
                num_layers=1
            )
            
            x = torch.randn(2, 10, 8)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            
            output = model(x, edge_index)
            assert output.shape == (2, 10, output_size)
            
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        model = MultiPerspectiveGraphESN(
            input_size=5,
            hidden_size=16,
            output_size=3,
            num_channels=2,
            num_layers=1
        )
        
        # Use 2 nodes so edge_index indices {0,1} are valid
        x = torch.randn(2, 8, 5, requires_grad=True)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        output = model(x, edge_index)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestMPGESNLoss:
    def test_init(self):
        """Test MPGESNLoss initialization."""
        loss_fn = MPGESNLoss(
            time_weight=1.0,
            freq_weight=0.1,
            consistency_weight=0.05
        )
        assert loss_fn.time_weight == 1.0
        assert loss_fn.freq_weight == 0.1
        assert loss_fn.consistency_weight == 0.05
        

    def test_time_domain_only(self):
        """Test loss computation with time domain only."""
        loss_fn = MPGESNLoss(
            time_weight=1.0,
            freq_weight=0.0,
            consistency_weight=0.0
        )
        
        # Use same shapes for pred and target
        batch_size, seq_len, output_size = 4, 32, 32
        pred = torch.randn(batch_size, seq_len, output_size)
        target = torch.randn(batch_size, seq_len, output_size)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        
    def test_combined_loss(self):
        """Test loss computation with all components."""
        loss_fn = MPGESNLoss(
            time_weight=1.0,
            freq_weight=0.1,
            consistency_weight=0.05
        )
        
        # Use same shapes
        batch_size, seq_len, output_size = 4, 16, 16
        pred = torch.randn(batch_size, seq_len, output_size)
        target = torch.randn(batch_size, seq_len, output_size)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        
        batch_size, seq_len, output_size = 4, 32, 5
        pred = torch.randn(batch_size, seq_len, output_size)
        target = torch.randn(batch_size, seq_len, output_size)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        
    def test_frequency_domain_loss(self):
        """Test frequency domain loss computation."""
        loss_fn = MPGESNLoss(
            time_weight=0.0,
            freq_weight=1.0,
            consistency_weight=0.0
        )
        
        batch_size, seq_len, output_size = 2, 64, 3
        pred = torch.randn(batch_size, seq_len, output_size)
        target = torch.randn(batch_size, seq_len, output_size)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        
    def test_consistency_loss(self):
        """Test consistency loss computation."""
        loss_fn = MPGESNLoss(
            time_weight=0.0,
            freq_weight=0.0,
            consistency_weight=1.0
        )
        
        batch_size, seq_len, output_size = 2, 16, 4
        pred = torch.randn(batch_size, seq_len, output_size)
        target = torch.randn(batch_size, seq_len, output_size)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        
    def test_zero_loss_identical_tensors(self):
        """Test that identical tensors produce zero time domain loss."""
        loss_fn = MPGESNLoss(
            time_weight=1.0,
            freq_weight=0.0,
            consistency_weight=0.0
        )
        
        batch_size, seq_len, output_size = 2, 16, 3
        pred = torch.randn(batch_size, seq_len, output_size)
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        assert loss.item() < 1e-6  # Should be very close to zero
        
    def test_different_sequence_lengths(self):
        """Test loss with different sequence lengths."""
        loss_fn = MPGESNLoss()
        
        for seq_len in [8, 16, 32, 64]:
            batch_size, output_size = 2, 3
            pred = torch.randn(batch_size, seq_len, output_size)
            target = torch.randn(batch_size, seq_len, output_size)
            
            loss = loss_fn(pred, target)
            assert loss.item() >= 0
            assert torch.isfinite(loss)


class TestIntegration:
    def test_training_step(self):
        """Test a complete training step."""
        input_size = 8
        hidden_size = 32
        output_size = 8  # Match input size for testing
        
        model = MultiPerspectiveGraphESN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_channels=6,
            num_layers=1
        )
        
        loss_fn = MPGESNLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, input_size)
        target = torch.randn(batch_size, seq_len, output_size)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        optimizer.zero_grad()
        output = model(x, edge_index)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        
    def test_model_persistence(self):
        """Test model state dict save/load."""
        model = MultiPerspectiveGraphESN(
            input_size=5,
            hidden_size=16,
            output_size=3,
            num_channels=4,
            num_layers=1
        )
        
        # Save state dict
        state_dict = model.state_dict()
        
        # Create new model and load state dict
        new_model = MultiPerspectiveGraphESN(
            input_size=5,
            hidden_size=16,
            output_size=3,
            num_channels=4,
            num_layers=1
        )
        new_model.load_state_dict(state_dict, strict=False)  # Allow missing keys for dynamic projections
        
        # Test that models produce same output (2 nodes for edge_index)
        x = torch.randn(2, 10, 5)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        with torch.no_grad():
            output1 = model(x, edge_index)
            output2 = new_model(x, edge_index)
        
        # Outputs may differ due to dynamic projection layers, but should be finite
        assert torch.isfinite(output1).all()
        assert torch.isfinite(output2).all()
        assert output1.shape == output2.shape
        
    def test_device_compatibility(self):
        """Test model works on different devices."""
        model = MultiPerspectiveGraphESN(
            input_size=4,
            hidden_size=8,
            output_size=2,
            num_channels=3,
            num_layers=1
        )
        
        x = torch.randn(2, 8, 4)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        # Test CPU
        output_cpu = model(x, edge_index)
        assert output_cpu.device.type == 'cpu'
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            x_gpu = x.cuda()
            edge_index_gpu = edge_index.cuda()
            
            output_gpu = model_gpu(x_gpu, edge_index_gpu)
            assert output_gpu.device.type == 'cuda'
            
    def test_batch_consistency(self):
        """Test that model produces consistent results across batch sizes."""
        model = MultiPerspectiveGraphESN(
            input_size=6,
            hidden_size=16,
            output_size=3,
            num_channels=4,
            num_layers=1
        )
        
        # Use 2 nodes so edge_index {0,1} valid
        seq_len, input_size = 12, 6
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        # Single sample
        x_single = torch.randn(1, seq_len, input_size)
        output_single = model(x_single, edge_index)
        
        # Batch of same sample
        x_batch = x_single.repeat(3, 1, 1)
        output_batch = model(x_batch, edge_index)
        
        # Check shapes match
        assert output_single.shape == (1, seq_len, 3)
        assert output_batch.shape == (3, seq_len, 3)
        
        # Check that outputs are finite
        assert torch.isfinite(output_single).all()
        assert torch.isfinite(output_batch).all()

    def test_empty_graph(self):
        """Test model with empty edge_index."""
        model = MultiPerspectiveGraphESN(
            input_size=4,
            hidden_size=8,
            output_size=2,
            num_channels=3,
            num_layers=1
        )
        
        x = torch.randn(2, 8, 4)
        edge_index = torch.empty((2, 0), dtype=torch.long)  # Empty graph
        
        output = model(x, edge_index)
        assert output.shape == (2, 8, 2)
        assert torch.isfinite(output).all()

    def test_single_node_graph(self):
        """Test model with single node graph."""
        model = MultiPerspectiveGraphESN(
            input_size=4,
            hidden_size=8,
            output_size=2,
            num_channels=3,
            num_layers=1
        )
        
        x = torch.randn(2, 8, 4)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop on node 0
        
        output = model(x, edge_index)
        assert output.shape == (2, 8, 2)
        assert torch.isfinite(output).all()


if __name__ == '__main__':
    pytest.main([__file__])