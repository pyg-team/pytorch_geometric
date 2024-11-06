import unittest
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models.tokengt import TokenGT

class TestTokenGT(unittest.TestCase):
    def test_tokengt_forward(self):
        # Create a simple graph with 5 nodes with 16-dimensional features
        x = torch.randn(5, 16)  #
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 4]
        ])  
        edge_attr = torch.randn(4, 8) 
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.batch = torch.zeros(5, dtype=torch.long) 

        batch = Batch.from_data_list([data, data])  

        model = TokenGT(
            node_feat_dim=16,
            edge_feat_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            num_classes=3,
            method='orf',
            d_p=8,
            d_e=4,
            use_graph_token=True
        )

        logits = model(batch)
        # Should have shape [batch_size, num_classes]
        self.assertEqual(logits.size(), (2, 3)) 

        self.assertTrue(torch.isfinite(logits).all())

    def test_tokengt_laplacian(self):
        x = torch.randn(5, 16)
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 4]
        ])
        edge_attr = torch.randn(4, 8)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.batch = torch.zeros(5, dtype=torch.long)

        batch = Batch.from_data_list([data])

        model = TokenGT(
            node_feat_dim=16,
            edge_feat_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            num_classes=3,
            method='laplacian',
            d_p=3,
            d_e=4,
            use_graph_token=True
        )

        logits = model(batch)
        self.assertEqual(logits.size(), (1, 3))

        self.assertTrue(torch.isfinite(logits).all())

if __name__ == '__main__':
    unittest.main()