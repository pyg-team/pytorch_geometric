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

    out = index.search(lhs, k)
    assert out.score.device == device
    assert out.index.device == device
    assert out.score.size() == (10, k)
    assert out.index.size() == (10, k)

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

    out = index.search(lhs, k)
    assert out.score.device == device
    assert out.index.device == device
    assert out.score.size() == (10, k)
    assert out.index.size() == (10, k)

    mat = lhs @ rhs.t()
    score, index = mat.sort(dim=-1, descending=True)

    assert torch.allclose(out.score, score[:, :k])
    assert torch.equal(out.index, index[:, :k])


@withCUDA
@withPackage('faiss')
@pytest.mark.parametrize('k', [50])
def test_MIPS_exclude(device, k):
    lhs = torch.randn(10, 16, device=device)
    rhs = torch.randn(100, 16, device=device)

    exclude_lhs = torch.randint(0, 10, (500, ), device=device)
    exclude_rhs = torch.randint(0, 100, (500, ), device=device)
    exclude_links = torch.stack([exclude_lhs, exclude_rhs], dim=0)
    exclude_links = exclude_links.unique(dim=1)

    index = MIPSKNNIndex(rhs)

    out = index.search(lhs, k, exclude_links)
    assert out.score.device == device
    assert out.index.device == device
    assert out.score.size() == (10, k)
    assert out.index.size() == (10, k)

    # Ensure that excluded links are not present in `out.index`:
    batch = torch.arange(lhs.size(0), device=device).repeat_interleave(k)
    knn_links = torch.stack([batch, out.index.view(-1)], dim=0)
    knn_links = knn_links[:, knn_links[1] >= 0]

    unique_links = torch.cat([knn_links, exclude_links], dim=1).unique(dim=1)
    assert unique_links.size(1) == knn_links.size(1) + exclude_links.size(1)
