import pytest
import torch

from torch_geometric.nn import L2KNNIndex, MIPSKNNIndex
from torch_geometric.testing import withCUDA, withPackage


@withCUDA
@withPackage('faiss')
@pytest.mark.parametrize('k', [2])
def test_L2_knn(device, k):
    lhs = torch.randn(10, 16, device=device)
    rhs = torch.randn(100, 16, device=device)

    index = L2KNNIndex(rhs)
    assert index.get_emb().device == device
    assert torch.equal(index.get_emb(), rhs)

    out = index.search(lhs, k=k)
    assert out.score.device == device
    assert out.index.device == device

    mat = torch.linalg.norm(lhs.unsqueeze(1) - rhs.unsqueeze(0), dim=-1).pow(2)
    score, index = mat.sort(dim=-1)

    assert torch.allclose(out.score, score[:, :k])
    assert torch.equal(out.index, index[:, :k])


@withCUDA
@withPackage('faiss')
@pytest.mark.parametrize('k', [2])
def test_MIPS_knn(device, k):
    lhs = torch.randn(10, 16, device=device)
    rhs = torch.randn(100, 16, device=device)

    index = MIPSKNNIndex(rhs)
    assert index.get_emb().device == device
    assert torch.equal(index.get_emb(), rhs)

    out = index.search(lhs, k=k)
    assert out.score.device == device
    assert out.index.device == device

    mat = lhs @ rhs.t()
    score, index = mat.sort(dim=-1, descending=True)

    assert torch.allclose(out.score, score[:, :k])
    assert torch.equal(out.index, index[:, :k])
