import pytest
import torch

from torch_geometric.nn import BatchNorm, HeteroBatchNorm, HeteroInstanceNorm, HeteroLayerNorm, InstanceNorm, LayerNorm


@pytest.mark.parametrize('conf', [True, False])
def test_heterobatch_norm(conf):
    x_dict = {'n'+str(i):torch.randn(100, 16) for i in range(10)}
    types = x_dict.keys()
    hetero_norm = HeteroBatchNorm(16, types=types, affine=conf, track_running_stats=conf)
    homo_norm = BatchNorm(16, affine=conf, track_running_stats=conf)
    from_homo_norm = HeteroBatchNorm.from_homogeneous(homo_norm, types)
    expected_str_repr = 'HeteroBatchNorm(16)'
    assert hetero_norm.__repr__() == expected_str_repr
    assert from_homo_norm.__repr__() == expected_str_repr

    out_dict1 = hetero_norm(x_dict)
    out_dict2 = from_homo_norm(x_dict)
    for x_type in x_dict.keys():
        assert out_dict1[x_type].size() == (100, 16)
        assert out_dict2[x_type].size() == (100, 16)


def test_batch_norm_single_element():
    x = torch.randn(1, 16)
    with pytest.raises(ValueError, match="requires 'track_running_stats'"):
        norm = BatchNorm(16, track_running_stats=False,
                         allow_single_element=True)

    norm = BatchNorm(16, track_running_stats=True, allow_single_element=True)
    out = norm(x)
    assert torch.allclose(out, x)


@pytest.mark.parametrize('affine', [True, False])
def test_heterolayer_norm(affine):
    x_dict = {'n'+str(i):torch.randn(100, 16) for i in range(10)}
    batch_dict = {'n'+str(i):torch.randn(100, 16) for i in range(10)}
    types = list(x_dict.keys())
    hetero_norm = HeteroLayerNorm(16, types=types, affine=affine)
    homo_norm = LayerNorm(16, affine=affine)
    from_homo_norm = HeteroLayerNorm.from_homogeneous(homo_norm, types)
    expected_str_repr = 'HeteroLayerNorm(16)'
    assert hetero_norm.__repr__() == expected_str_repr
    assert from_homo_norm.__repr__() == expected_str_repr

    out1_dict = hetero_norm(x_dict)
    out2_dict = from_homo_norm(x_dict)
    batch_out_dict = hetero_norm(x_dict, batch_dict=batch_dict)
    for x_type in x_dict.keys():
        assert out1_dict[x_type].size() == (100, 16)
        assert out2_dict[x_type].size() == (100, 16)
        assert torch.allclose(batch_out_dict[x_type], out1_dict[x_type], atol=1e-6)

    catted_x_dict = {x_type:torch.cat([x, x], dim=0) for x_type, x in x_dict}
    catted_batch_dict = {x_type:torch.cat([batch, batch + 1], dim=0) for x_type, batch in batch_dict}
    batch_cat_out_dict = hetero_norm(catted_x_dict, catted_batch_dict)
    for x_type in x_dict.keys():
        assert torch.allclose(out1_dict[x_type], batch_cat_out_dict[x_type][:100], atol=1e-6)
        assert torch.allclose(out1_dict[x_type], batch_cat_out_dict[x_type][100:], atol=1e-6)


@pytest.mark.parametrize('conf', [True, False])
def test_heteroinstance_norm(conf):
    x_dict = {'n'+str(i):torch.randn(100, 16) for i in range(10)}
    batch_dict = {'n'+str(i):torch.randn(100, 16) for i in range(10)}
    types = list(x_dict.keys())
    hetero_norm = HeteroInstanceNorm(16, types=types, affine=conf, track_running_stats=conf)
    homo_norm = InstanceNorm(16, affine=conf, track_running_stats=conf)
    from_homo_norm = HeteroInstanceNorm.from_homogeneous(homo_norm, types)
    expected_str_repr = 'HeteroInstanceNorm(16)'
    assert hetero_norm.__repr__() == expected_str_repr
    assert from_homo_norm.__repr__() == expected_str_repr

    out1_dict = hetero_norm(x_dict)
    out2_dict = from_homo_norm(x_dict)
    batch_out_dict = hetero_norm(x_dict, batch_dict=batch_dict)
    for x_type in x_dict.keys():
        assert out1_dict[x_type].size() == (100, 16)
        assert out2_dict[x_type].size() == (100, 16)
        assert torch.allclose(batch_out_dict[x_type], out1_dict[x_type], atol=1e-6)

    catted_x_dict = {x_type:torch.cat([x, x], dim=0) for x_type, x in x_dict}
    catted_batch_dict = {x_type:torch.cat([batch, batch + 1], dim=0) for x_type, batch in batch_dict}
    batch_cat_out_dict = hetero_norm(catted_x_dict, catted_batch_dict)
    for x_type in x_dict.keys():
        assert torch.allclose(out1_dict[x_type], batch_cat_out_dict[x_type][:100], atol=1e-6)
        assert torch.allclose(out1_dict[x_type], batch_cat_out_dict[x_type][100:], atol=1e-6)
