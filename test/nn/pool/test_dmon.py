import math

# import numpy as np
import torch

# from torch_geometric.nn import dense_dmon
from torch_geometric.nn.pool.dmon import SparseDMoNPooling

# from torch_geometric.testing import is_full_test


class TestDmoNPooling:
    RTOL = 1e-6  # relative tolerance

    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    sparsity = 0.2

    def create_input(self):
        x = torch.randn((self.batch_size, self.num_nodes, self.channels))
        adj = (
            # a matrix with expected sparsity = 0.2
            (torch.rand((self.batch_size, self.num_nodes, self.num_nodes))
             <= self.sparsity).type(torch.float).to_sparse())
        s = torch.randn((self.batch_size, self.num_nodes, self.num_clusters))
        mask = torch.randint(0, 2, (self.batch_size, self.num_nodes),
                             dtype=torch.bool)

        return x, adj, s, mask

    def test_output_shape_and_range(self):
        x, adj, s, mask = self.create_input()
        pool = SparseDMoNPooling([self.channels, self.channels],
                                 self.num_clusters)
        s, x, adj, spectral_loss, ortho_loss, cluster_loss = pool(x, adj, mask)
        assert s.size() == (2, 20, 10)
        assert x.size() == (2, 10, 16)
        assert adj.size() == (2, 10, 10)
        assert -1 <= spectral_loss <= 0.5
        assert 0 <= ortho_loss <= math.sqrt(2)
        assert 0 <= cluster_loss <= math.sqrt(self.num_clusters) - 1

        # if is_full_test():
        #     jit = torch.jit.script(mincut_pool)

        #     x_jit, adj_jit, mincut_loss, ortho_loss = jit(x, adj, s, mask)
        #     assert x_jit.size() == (self.batch_size, self.num_clusters,
        #                             self.channels)
        #     assert adj_jit.size() == (
        #         self.batch_size,
        #         self.num_clusters,
        #         self.num_clusters,
        #     )
        #     assert -1 <= mincut_loss <= 0
        #     assert 0 <= ortho_loss <= math.sqrt(2)

    # def test_output_vs_dense_mincut(self):
    #     """Check the output matches that of the dense_mincut_pool."""
    #     x, adj, s, mask = self.create_input()

    #     (
    #         expected_x_out,
    #         expected_adj_out,
    #         expected_mincut_loss,
    #         expected_ortho_loss,
    #     ) = dense_mincut_pool(x, adj.to_dense(), s, mask)

    #     actual_x_out, actual_adj_out, actual_mincut_loss,
    # actual_ortho_loss = (
    #         mincut_pool(x, adj, s, mask))

    #     np.testing.assert_allclose(expected_x_out, actual_x_out,
    #                                rtol=self.RTOL)
    #     np.testing.assert_allclose(expected_adj_out, actual_adj_out,
    #                                rtol=self.RTOL)
    #     np.testing.assert_allclose(expected_mincut_loss, actual_mincut_loss,
    #                                rtol=self.RTOL)
    #     np.testing.assert_allclose(expected_ortho_loss, actual_ortho_loss,
    #                                rtol=self.RTOL)
