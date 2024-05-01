import math

import numpy as np
import torch

from torch_geometric.nn.dense import DMoNPooling as DenseDMoNPooling
from torch_geometric.nn.pool.dmon import SparseDMoNPooling


class TestDmoNPooling:
    RTOL = 1e-6  # relative tolerance

    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    sparsity = 0.2

    def create_input(self):
        x = torch.randn((self.batch_size, self.num_nodes, self.channels))
        batched_adj = (
            # a matrix with expected sparsity = 0.2
            (torch.rand((self.batch_size, self.num_nodes, self.num_nodes))
             <= self.sparsity).type(torch.float).to_sparse())
        adj_list = [
            batched_adj[b].to_sparse_csr() for b in range(self.batch_size)
        ]
        mask = torch.randint(0, 2, (self.batch_size, self.num_nodes),
                             dtype=torch.bool)

        return x, batched_adj, adj_list, mask

    def test_output_shape_and_range(self):
        x, batched_adj, adj_list, mask = self.create_input()
        pool = SparseDMoNPooling([self.channels, self.channels],
                                 self.num_clusters)
        s, x, adj, spectral_loss, ortho_loss, cluster_loss = pool(
            x, adj_list, batched_adj, mask)
        assert s.size() == (self.batch_size, self.num_nodes, self.num_clusters)
        assert x.size() == (self.batch_size, self.num_clusters, self.channels)
        assert adj.size() == (2, self.num_clusters, self.num_clusters)
        assert -1 <= spectral_loss <= 0.5
        assert 0 <= ortho_loss <= math.sqrt(2)
        assert 0 <= cluster_loss <= math.sqrt(self.num_clusters) - 1

    def test_output_vs_dense_dmon(self):
        """Check the output matches that of the dense DMoNPooling."""
        x, adj, mask = self.create_input()

        dense_pool = DenseDMoNPooling([self.channels, self.channels],
                                      self.num_clusters)
        dense_output = dense_pool(x, adj.to_dense(), mask)

        sparse_pool = SparseDMoNPooling([self.channels, self.channels],
                                        self.num_clusters)
        # copy the model parameters
        sparse_pool.load_state_dict(dense_pool.state_dict())

        sparse_output = sparse_pool(x, adj, mask)

        # convert to numpy
        (
            expected_s,
            expected_x,
            expected_adj,
            expected_spectral_loss,
            expected_ortho_loss,
            expected_cluster_loss,
            actual_s,
            actual_x,
            actual_adj,
            actual_spectral_loss,
            actual_ortho_loss,
            actual_cluster_loss,
        ) = map(lambda v: v.detach().numpy(), dense_output + sparse_output)

        np.testing.assert_allclose(expected_s, actual_s, rtol=self.RTOL)
        np.testing.assert_allclose(expected_x, actual_x, rtol=self.RTOL)
        np.testing.assert_allclose(expected_adj, actual_adj, rtol=self.RTOL)
        np.testing.assert_allclose(expected_spectral_loss,
                                   actual_spectral_loss, rtol=self.RTOL)
        np.testing.assert_allclose(expected_ortho_loss, actual_ortho_loss,
                                   rtol=self.RTOL)
        np.testing.assert_allclose(expected_cluster_loss, actual_cluster_loss,
                                   rtol=self.RTOL)
