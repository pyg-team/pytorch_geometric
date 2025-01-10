import unittest

import torch

from torch_geometric.data import Data
from torch_geometric.nn import XENetConv


class TestXENetConv(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Define test dimensions
        self.num_nodes = 4
        self.in_node_channels = 3
        self.in_edge_channels = 2
        self.node_channels = 5
        self.edge_channels = 4
        self.stack_channels = [8, 16]

        # Create a simple graph for testing
        self.x = torch.randn(self.num_nodes, self.in_node_channels)
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        self.edge_attr = torch.randn(self.edge_index.size(1),
                                     self.in_edge_channels)

        # Create different variants of the layer for testing
        self.conv_attention = XENetConv(in_node_channels=self.in_node_channels,
                                        in_edge_channels=self.in_edge_channels,
                                        stack_channels=self.stack_channels,
                                        node_channels=self.node_channels,
                                        edge_channels=self.edge_channels,
                                        attention=True)

        self.conv_no_attention = XENetConv(
            in_node_channels=self.in_node_channels,
            in_edge_channels=self.in_edge_channels,
            stack_channels=self.stack_channels,
            node_channels=self.node_channels, edge_channels=self.edge_channels,
            attention=False)

    def test_basic_forward(self):
        """Test basic forward pass with attention."""
        out_x, out_edge_attr = self.conv_attention(self.x, self.edge_index,
                                                   self.edge_attr)

        # Check output shapes
        self.assertEqual(out_x.shape, (self.num_nodes, self.node_channels))
        self.assertEqual(out_edge_attr.shape,
                         (self.edge_index.size(1), self.edge_channels))

        # Check that outputs contain no NaN values
        self.assertFalse(torch.isnan(out_x).any())
        self.assertFalse(torch.isnan(out_edge_attr).any())

    def test_no_attention_forward(self):
        """Test forward pass without attention."""
        out_x, out_edge_attr = self.conv_no_attention(self.x, self.edge_index,
                                                      self.edge_attr)

        # Check output shapes
        self.assertEqual(out_x.shape, (self.num_nodes, self.node_channels))
        self.assertEqual(out_edge_attr.shape,
                         (self.edge_index.size(1), self.edge_channels))

        # Check that outputs contain no NaN values
        self.assertFalse(torch.isnan(out_x).any())
        self.assertFalse(torch.isnan(out_edge_attr).any())

    def test_custom_activation(self):
        """Test with custom activation functions."""
        conv = XENetConv(in_node_channels=self.in_node_channels,
                         in_edge_channels=self.in_edge_channels,
                         stack_channels=self.stack_channels,
                         node_channels=self.node_channels,
                         edge_channels=self.edge_channels, attention=True,
                         node_activation=torch.tanh,
                         edge_activation=torch.relu)

        out_x, out_edge_attr = conv(self.x, self.edge_index, self.edge_attr)

        # Check output ranges for activations
        self.assertTrue(torch.all(out_x >= -1)
                        and torch.all(out_x <= 1))  # tanh range
        self.assertTrue(torch.all(out_edge_attr >= 0))  # ReLU range

    def test_single_stack_channel(self):
        """Test with a single stack channel instead of a list."""
        conv = XENetConv(
            in_node_channels=self.in_node_channels,
            in_edge_channels=self.in_edge_channels,
            stack_channels=32,  # single integer
            node_channels=self.node_channels,
            edge_channels=self.edge_channels)

        out_x, out_edge_attr = conv(self.x, self.edge_index, self.edge_attr)

        # Check output shapes
        self.assertEqual(out_x.shape, (self.num_nodes, self.node_channels))
        self.assertEqual(out_edge_attr.shape,
                         (self.edge_index.size(1), self.edge_channels))

    def test_batch_processing(self):
        """Test processing of batched graphs."""
        # Create two graphs with different sizes
        x1 = torch.randn(3, self.in_node_channels)
        edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]],
                                   dtype=torch.long)
        edge_attr1 = torch.randn(edge_index1.size(1), self.in_edge_channels)

        x2 = torch.randn(4, self.in_node_channels)
        edge_index2 = torch.tensor([[0, 1, 2, 2, 3], [1, 2, 1, 3, 2]],
                                   dtype=torch.long)
        edge_attr2 = torch.randn(edge_index2.size(1), self.in_edge_channels)

        # Create PyG Data objects
        data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1)
        data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)

        # Process each graph separately
        out_x1, out_edge_attr1 = self.conv_attention(data1.x, data1.edge_index,
                                                     data1.edge_attr)
        out_x2, out_edge_attr2 = self.conv_attention(data2.x, data2.edge_index,
                                                     data2.edge_attr)

        # Check output shapes
        self.assertEqual(out_x1.shape, (3, self.node_channels))
        self.assertEqual(out_edge_attr1.shape, (4, self.edge_channels))
        self.assertEqual(out_x2.shape, (4, self.node_channels))
        self.assertEqual(out_edge_attr2.shape, (5, self.edge_channels))

    def test_isolated_nodes(self):
        """Test handling of isolated nodes."""
        # Create a graph with an isolated node
        x = torch.randn(4, self.in_node_channels)
        edge_index = torch.tensor([[0, 1], [1, 2]],
                                  dtype=torch.long)  # Node 3 is isolated
        edge_attr = torch.randn(edge_index.size(1), self.in_edge_channels)

        out_x, out_edge_attr = self.conv_attention(x, edge_index, edge_attr)

        # Check that isolated node features are updated
        self.assertFalse(torch.isnan(out_x[3]).any())
        self.assertEqual(out_x.shape, (4, self.node_channels))
        self.assertEqual(out_edge_attr.shape, (2, self.edge_channels))

    def test_gradients(self):
        """Test gradient computation."""
        self.x.requires_grad_()
        self.edge_attr.requires_grad_()

        out_x, out_edge_attr = self.conv_attention(self.x, self.edge_index,
                                                   self.edge_attr)

        # Compute gradients
        loss = out_x.sum() + out_edge_attr.sum()
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(self.x.grad)
        self.assertIsNotNone(self.edge_attr.grad)
        self.assertFalse(torch.isnan(self.x.grad).any())
        self.assertFalse(torch.isnan(self.edge_attr.grad).any())


if __name__ == '__main__':
    unittest.main()
