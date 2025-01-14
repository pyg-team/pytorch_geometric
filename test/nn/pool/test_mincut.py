import math

import numpy as np
import pytest
import torch

from torch_geometric.nn import dense_mincut_pool
from torch_geometric.nn.pool.mincut import (
    _sparse_diag_2d,
    _sparse_rank3_diag,
    batched_block_diagonal_dense,
    batched_block_diagonal_sparse,
    block_diagonal_to_batched_3d,
    mincut_pool,
)
from torch_geometric.testing import is_full_test, withPackage


def test__sparse_diag_2d():
    diag = torch.Tensor([1, 2, 3, 4])
    actual = _sparse_diag_2d(diag)
    expected = torch.eye(diag.size(0)) * diag.unsqueeze(1)
    assert (actual.to_dense() == expected).all()


def test__sparse_rank3_diag():
    d_flat = torch.Tensor([[2, 3, 4], [1, 5, 6]])
    actual = _sparse_rank3_diag(d_flat)
    expected = torch.eye(3) * d_flat.unsqueeze(2)
    assert (actual.to_dense() == expected).all()


@withPackage("torch>=2.0.0")
class TestMinCut:
    RTOL = 1e-6  # relative tolerance

    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    sparsity = 0.2

    def create_input(self):
        # batched feature matrix
        x = torch.randn((self.batch_size, self.num_nodes, self.channels))
        adj = (
            # batched adjacency matrix, a 3D tensor with expected sparsity 0.2
            (torch.rand((self.batch_size, self.num_nodes, self.num_nodes))
             <= self.sparsity).type(torch.float).to_sparse())
        # batched node clustering tensor
        s = torch.randn((self.batch_size, self.num_nodes, self.num_clusters))
        # batched random masks
        mask = torch.randint(0, 2, (self.batch_size, self.num_nodes),
                             dtype=torch.bool)

        return x, adj, s, mask

    @classmethod
    def check_output_shape_and_range(self, x, adj, mincut_loss, ortho_loss):
        assert x.size() == (self.batch_size, self.num_clusters, self.channels)
        assert adj.size() == (self.batch_size, self.num_clusters,
                              self.num_clusters)
        assert -1 <= mincut_loss <= 0
        assert 0 <= ortho_loss <= math.sqrt(2)

    def test_output_shape_and_range(self):
        x, adj, s, mask = self.create_input()
        x_out, adj_out, mincut_loss, ortho_loss = mincut_pool(x, adj, s, mask)
        self.check_output_shape_and_range(x_out, adj_out, mincut_loss,
                                          ortho_loss)

        if is_full_test():
            jit = torch.jit.script(mincut_pool)

            x_jit, adj_jit, mincut_loss, ortho_loss = jit(x, adj, s, mask)
            self.check_output_shape_and_range(x_jit, adj_jit, mincut_loss,
                                              ortho_loss)

    def test_output_equivalence_to_dense_mincut(self):
        """Check the output matches that of the dense_mincut_pool."""
        x, adj, s, mask = self.create_input()

        (
            expected_x_out,
            expected_adj_out,
            expected_mincut_loss,
            expected_ortho_loss,
        ) = dense_mincut_pool(x, adj.to_dense(), s, mask)

        actual_x_out, actual_adj_out, actual_mincut_loss, actual_ortho_loss = (
            mincut_pool(x, adj, s, mask))

        np.testing.assert_allclose(expected_x_out, actual_x_out,
                                   rtol=self.RTOL)
        np.testing.assert_allclose(expected_adj_out, actual_adj_out,
                                   rtol=self.RTOL)
        np.testing.assert_allclose(expected_mincut_loss, actual_mincut_loss,
                                   rtol=self.RTOL)
        np.testing.assert_allclose(expected_ortho_loss, actual_ortho_loss,
                                   rtol=self.RTOL)


class TestBatchedBlockDiagonalConversion:
    @pytest.mark.parametrize("input_is_sparse", [True, False])
    @pytest.mark.parametrize(
        "batched_tensor, block_diagonal_tensor",
        [
            (
                torch.tensor([[[1]], [[2]], [[3]]]),
                torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]]).to_sparse(),
            ),
            (
                torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                torch.tensor([
                    [1, 2, 0, 0],
                    [3, 4, 0, 0],
                    [0, 0, 5, 6],
                    [0, 0, 7, 8],
                ]).to_sparse(),
            ),
            (
                # b=1
                torch.tensor([[[1, 2], [3, 4]]]),
                torch.tensor([
                    [1, 2],
                    [3, 4],
                ]).to_sparse(),
            ),
            (
                torch.tensor([[[1, 2]], [[3, 4]]]),
                torch.tensor([
                    [1, 2, 0, 0],
                    [0, 0, 3, 4],
                ]).to_sparse(),
            ),
            (
                torch.tensor([[[1], [2]], [[3], [4]]]),
                torch.tensor([
                    [1, 0],
                    [2, 0],
                    [0, 3],
                    [0, 4],
                ]).to_sparse(),
            ),
        ],
    )
    def test_conversion(self, input_is_sparse, batched_tensor,
                        block_diagonal_tensor):
        # from batched to block diagonal
        if input_is_sparse:
            actual = batched_block_diagonal_sparse(batched_tensor.to_sparse())
        else:
            actual = batched_block_diagonal_dense(batched_tensor)
        assert actual.is_sparse
        assert actual.layout == torch.sparse_coo
        torch.testing.assert_close(actual.to_dense(),
                                   block_diagonal_tensor.to_dense())

        # from block diagonal to batched
        if input_is_sparse:
            b, h, w = batched_tensor.size()
            actual = block_diagonal_to_batched_3d(block_diagonal_tensor, b, h,
                                                  w)
            assert not actual.is_sparse
            torch.testing.assert_close(actual.to_dense(),
                                       batched_tensor.to_dense())

    def test_invalid_input(self):
        dense_tensor = torch.tensor([[[1]], [[2]], [[3]]])
        sparse_tensor = dense_tensor.to_sparse()
        with pytest.raises(TypeError, match=".*dense.*"):
            batched_block_diagonal_dense(sparse_tensor)

        with pytest.raises(TypeError):
            batched_block_diagonal_sparse(dense_tensor)

        dense_tensor_1d = dense_tensor.flatten()
        sparse_tensor_1d = dense_tensor_1d.to_sparse()

        with pytest.raises(ValueError):
            batched_block_diagonal_dense(dense_tensor_1d)

        with pytest.raises(ValueError):
            batched_block_diagonal_sparse(sparse_tensor_1d)

        with pytest.raises(TypeError, match=".*dense.*"):
            block_diagonal_to_batched_3d(dense_tensor, 3, 1, 1)

        sparse_tensor_bd = batched_block_diagonal_sparse(sparse_tensor)
        with pytest.raises(TypeError, match=".*coo.*"):
            block_diagonal_to_batched_3d(sparse_tensor_bd.to_sparse_csr(), 3,
                                         1, 1)

        with pytest.raises(ValueError, match=".*ndim = 2.*"):
            block_diagonal_to_batched_3d(
                sparse_tensor_bd.to_dense().unsqueeze(0).to_sparse(), 3, 1, 1)
