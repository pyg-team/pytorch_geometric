import pytest
import torch

from torch_geometric.nn import BatchNorm, HeteroBatchNorm, HeteroInstanceNorm, HeteroLayerNorm, InstanceNorm, LayerNorm


@pytest.mark.parametrize('conf', [True, False])
def test_heterobatch_norm(conf):
    x_dict = {'n'+str(i):torch.randn(100, 16) for i in range(10)}

    hetero_norm = HeteroBatchNorm(16, types=x.keys(), affine=conf, track_running_stats=conf)
    homo_norm = BatchNorm(16, affine=conf, track_running_stats=conf)
    from_homo_norm = HeteroBatchNorm.from_homogeneous(homo_norm)
    assert norm.__repr__() == 'HeteroBatchNorm(16)'
    assert from_homo_norm.__repr__() == 'HeteroBatchNorm(16)'

    out_dict1 = hetero_norm(x_dict)
    out_dict2 = from_homo_norm(x_dict)
    for x_type in x_dict.keys():
        assert out_dict1[x_type].size() == (100, 16)
        assert out_dict2[x_type].size() == (100, 16)


def test_batch_norm_single_element():
    x = torch.randn(1, 16)

    norm = BatchNorm(16)
    with pytest.raises(ValueError, match="Expected more than 1 value"):
        norm(x)

    with pytest.raises(ValueError, match="requires 'track_running_stats'"):
        norm = BatchNorm(16, track_running_stats=False,
                         allow_single_element=True)

    norm = BatchNorm(16, track_running_stats=True, allow_single_element=True)
    out = norm(x)
    assert torch.allclose(out, x)

from torch_geometric.nn import InstanceNorm
from torch_geometric.testing import is_full_test


@pytest.mark.parametrize('conf', [True, False])
def test_heteroinstance_norm(conf):
    batch = torch.zeros(100, dtype=torch.long)

    x1 = torch.randn(100, 16)
    x2 = torch.randn(100, 16)

    norm1 = InstanceNorm(16, affine=conf, track_running_stats=conf)
    norm2 = InstanceNorm(16, affine=conf, track_running_stats=conf)
    assert norm1.__repr__() == 'InstanceNorm(16)'

    if is_full_test():
        torch.jit.script(norm1)

    out1 = norm1(x1)
    out2 = norm2(x1, batch)
    assert out1.size() == (100, 16)
    assert torch.allclose(out1, out2, atol=1e-7)
    if conf:
        assert torch.allclose(norm1.running_mean, norm2.running_mean)
        assert torch.allclose(norm1.running_var, norm2.running_var)

    out1 = norm1(x2)
    out2 = norm2(x2, batch)
    assert torch.allclose(out1, out2, atol=1e-7)
    if conf:
        assert torch.allclose(norm1.running_mean, norm2.running_mean)
        assert torch.allclose(norm1.running_var, norm2.running_var)

    norm1.eval()
    norm2.eval()

    out1 = norm1(x1)
    out2 = norm2(x1, batch)
    assert torch.allclose(out1, out2, atol=1e-7)

    out1 = norm1(x2)
    out2 = norm2(x2, batch)
    assert torch.allclose(out1, out2, atol=1e-7)

    out1 = norm2(x1)
    out2 = norm2(x2)
    out3 = norm2(torch.cat([x1, x2], dim=0), torch.cat([batch, batch + 1]))
    assert torch.allclose(out1, out3[:100], atol=1e-7)
    assert torch.allclose(out2, out3[100:], atol=1e-7)

@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('mode', ['graph', 'node'])
def test_heterolayer_norm(affine, mode):
    x = torch.randn(100, 16)
    batch = torch.zeros(100, dtype=torch.long)

    norm = LayerNorm(16, affine=affine, mode=mode)
    assert norm.__repr__() == f'LayerNorm(16, affine={affine}, mode={mode})'

    if is_full_test():
        torch.jit.script(norm)

    out1 = norm(x)
    assert out1.size() == (100, 16)
    assert torch.allclose(norm(x, batch), out1, atol=1e-6)

    out2 = norm(torch.cat([x, x], dim=0), torch.cat([batch, batch + 1], dim=0))
    assert torch.allclose(out1, out2[:100], atol=1e-6)
    assert torch.allclose(out1, out2[100:], atol=1e-6)
