import sys
import os

local_file_dir = os.path.abspath("../")
if local_file_dir not in sys.path:
    sys.path.insert(0, local_file_dir)

import unittest
import torch
from torch_geometric.data import Data, Batch
from tokenGT import TokenGT

def run_token_gt(input_feat_dim, method):
    # Create a simple graph with 5 nodes
    x = torch.randn(5, input_feat_dim)  #
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4]
    ])  
    edge_attr = torch.randn(4, input_feat_dim) 
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(5, dtype=torch.long) 
    batch = Batch.from_data_list([data])
    model = TokenGT(
        input_feat_dim=input_feat_dim,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_classes=3,
        method=method,
        d_p=9,
        d_e=4,
        use_graph_token=True
    )
    logits = model(batch)
    return logits

class TestTokenGTWrapper(unittest.TestCase):
    
    def test_token_gt_orf(self):
        for input_feat_dim in range(20):
            logits = run_token_gt(input_feat_dim=input_feat_dim, method="orf")
            self.assertEqual(logits.size(), (1, 3)) # Should have shape [batch_size, num_classes]
            self.assertTrue(torch.isfinite(logits).all())

    def test_token_gt_laplacian(self):
        for input_feat_dim in range(20):
            logits = run_token_gt(input_feat_dim=input_feat_dim, method="laplacian")
            self.assertEqual(logits.size(), (1, 3)) 
            self.assertTrue(torch.isfinite(logits).all())

if __name__ == '__main__':
    unittest.main()