import warnings

import pytest
import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GeoScatConv
from torch_geometric.nn.conv.geo_scat_conv import (
    _initialize_legs_parameters,
    get_sparse_lrw_diffusion_operator,
    multiorder_scatter,
)


def test_geo_scat_conv_node_level():
    in_channels = 4
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(in_channels, pool=None)
    assert conv.num_wavelet_filters == 6
    assert conv.num_scattering_filters == 7
    assert str(conv).startswith('GeoScatConv(')

    out = conv(x, edge_index)
    assert out.size() == (num_nodes, in_channels, 7)

    out_weighted = conv(x, edge_index, edge_weight)
    assert out_weighted.size() == (num_nodes, in_channels, 7)


def test_geo_scat_conv_second_order():
    in_channels = 3
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(
        in_channels,
        scattering_orders=[0, 1, 2],
        pool=None,
    )
    assert conv.num_scattering_filters == 22

    out = conv(x, edge_index)
    assert out.size() == (num_nodes, in_channels, 22)


def test_geo_scat_conv_graph_level_pooling():
    in_channels = 2
    batch = torch.tensor([0, 0, 1, 1])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(
        in_channels,
        scattering_orders=[0, 1, 2],
        pool=['mean', 'max'],
    )
    assert conv.out_channels == 44

    out = conv(x, edge_index, batch=batch)
    assert out.size() == (2, in_channels, 44)


def test_geo_scat_conv_all_pooling_ops():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(
        in_channels,
        pool=['min', 'median', 'mean', 'max', 'var'],
    )
    assert conv.out_channels == 7 * 5

    out = conv(x, edge_index)
    assert out.size() == (1, in_channels, 35)


def test_geo_scat_conv_vector_features():
    vector_dim = 3
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, vector_dim))

    P_node = get_sparse_lrw_diffusion_operator(
        edge_index,
        None,
        num_nodes,
        'row',
        x.dtype,
        x.device,
    ).to_dense()
    P = torch.kron(
        P_node,
        torch.eye(vector_dim, device=x.device, dtype=x.dtype),
    )

    conv = GeoScatConv(
        vector_dim,
        is_vector_feature=True,
        pool=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        out = conv(
            x,
            diffusion_op=P,
            diffusion_op_key='diffusion_adj',
        )
    assert out.size() == (num_nodes, vector_dim, 7)


def test_geo_scat_conv_vector_features_require_diffusion_op():
    vector_dim = 3
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, vector_dim))

    conv = GeoScatConv(
        vector_dim,
        is_vector_feature=True,
        pool=None,
    )

    with pytest.raises(ValueError, match='require a precomputed diffusion_op'):
        conv(x, edge_index)


def test_geo_scat_conv_activation():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(
        in_channels,
        scattering_orders=[1],
        activation=torch.abs,
        pool=None,
    )
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, in_channels, 6)
    assert (out >= 0).all()


def test_geo_scat_conv_wavelet_only():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(
        in_channels,
        scattering_orders=[1],
        pool=None,
    )
    assert conv.num_scattering_filters == 6

    out = conv(x, edge_index)
    assert out.size() == (num_nodes, in_channels, 6)


def test_geo_scat_conv_batch_consistency():
    in_channels = 3
    x1 = torch.randn(4, in_channels)
    edge_index1 = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    data1 = Data(x=x1, edge_index=edge_index1)

    x2 = torch.randn(3, in_channels)
    edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data2 = Data(x=x2, edge_index=edge_index2)

    conv = GeoScatConv(in_channels, pool=None)

    out1 = conv(x1, edge_index1)
    out2 = conv(x2, edge_index2)

    batch = Batch.from_data_list([data1, data2])
    out = conv(batch.x, batch.edge_index, batch=batch.batch)

    assert out.size() == (7, in_channels, 7)
    assert torch.allclose(out1, out[:4], atol=1e-6)
    assert torch.allclose(out2, out[4:], atol=1e-6)


def test_geo_scat_conv_removes_self_loops():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_index_with_loops = torch.tensor([
        [0, 1, 1, 2, 0, 1, 2],
        [1, 0, 2, 1, 0, 1, 2],
    ])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0])

    conv = GeoScatConv(in_channels, pool=None)

    with torch.no_grad():
        out = conv(x, edge_index)
        out_with_loops = conv(x, edge_index_with_loops, edge_weight)

    assert torch.allclose(out, out_with_loops, atol=1e-6)


def test_geo_scat_conv_custom_scales():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(
        in_channels,
        diffusion_scales=[0, 1, 2],
        include_lowpass=False,
        scattering_orders=[1],
        pool=None,
    )
    assert conv.num_wavelet_filters == 2
    assert conv.num_scattering_filters == 2

    out = conv(x, edge_index)
    assert out.size() == (num_nodes, in_channels, 2)


def test_geo_scat_conv_precomputed_diffusion_op():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(in_channels, pool=None)

    P = get_sparse_lrw_diffusion_operator(
        edge_index,
        None,
        num_nodes,
        'row',
        x.dtype,
        x.device,
    )

    with torch.no_grad():
        out_edge = conv(x, edge_index)
        out_P = conv(
            x,
            diffusion_op=P,
            diffusion_op_key='diffusion_adj',
        )

    assert torch.allclose(out_edge, out_P, atol=1e-6)


def test_geo_scat_conv_diffusion_op_batching_warning():
    in_channels = 2
    num_nodes = 3
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(in_channels, pool=None)

    row = torch.tensor([0, 1, 2, 0, 1, 2])
    col = torch.tensor([0, 1, 2, 1, 0, 1])
    val = torch.tensor([0.5, 0.5, 0.5, 0.25, 0.25, 0.25])
    P_sparse = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        val,
        (num_nodes, num_nodes),
    ).coalesce()
    P_dense = P_sparse.to_dense()

    with pytest.warns(UserWarning, match='Dense diffusion_op'):
        conv(x, diffusion_op=P_dense)

    with pytest.warns(UserWarning, match="diffusion_op_key='P'"):
        conv(x, diffusion_op=P_sparse, diffusion_op_key='P')

    with pytest.warns(UserWarning, match='without diffusion_op_key'):
        conv(x, diffusion_op=P_sparse)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        conv(
            x,
            diffusion_op=P_sparse,
            diffusion_op_key='diffusion_adj',
        )


@pytest.mark.parametrize('normalization', ['row', 'column', 'symmetric'])
def test_geo_scat_conv_normalization(normalization):
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(in_channels, normalization=normalization, pool=None)
    out = conv(x, edge_index)

    assert out.size() == (num_nodes, in_channels, 7)
    assert torch.isfinite(out).all()


def test_geo_scat_conv_scattering_order_zero_only():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(in_channels, scattering_orders=[0], pool=None)
    assert conv.num_scattering_filters == 1

    out = conv(x, edge_index)
    assert out.size() == (num_nodes, in_channels, 1)
    assert torch.allclose(out.squeeze(-1), x)


def test_geo_scat_conv_second_order_activation():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(
        in_channels,
        scattering_orders=[0, 1, 2],
        activation=torch.abs,
        pool=None,
    )
    out = conv(x, edge_index)

    assert out.size() == (num_nodes, in_channels, 22)
    assert (out[:, :, 1:] >= 0).all()


def test_geo_scat_conv_multiorder_scatter_consistency():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(
        in_channels,
        scattering_orders=[0, 1, 2],
        pool=None,
    )
    P = _make_dense_lrw_diffusion_operator(edge_index, num_nodes)

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            out_conv = conv(x, diffusion_op=P)
        out_scatter = multiorder_scatter(
            x,
            P,
            diffusion_scales=torch.tensor(conv.diffusion_scales),
            include_lowpass=conv.include_lowpass,
            scattering_orders=conv.scattering_orders,
            is_vector_feature=conv.is_vector_feature,
        )

    assert torch.allclose(out_conv, out_scatter, atol=1e-6)


def test_geo_scat_conv_check_nonfinite():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    x[0, 0] = float('nan')

    conv = GeoScatConv(in_channels, check_nonfinite=True, pool=None)

    with pytest.raises(RuntimeError, match='Non-finite value detected'):
        conv(x, edge_index)


def test_geo_scat_conv_diffusion_op_shape_mismatch():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes + 1, in_channels))

    conv = GeoScatConv(in_channels, pool=None)
    P = get_sparse_lrw_diffusion_operator(
        edge_index,
        None,
        num_nodes,
        'row',
        x.dtype,
        x.device,
    )

    with pytest.raises(ValueError, match='does not match expected size'):
        conv(x, diffusion_op=P)


@pytest.mark.parametrize(
    'kwargs,match',
    [
        ({
            'scattering_orders': []
        }, 'must be a non-empty tuple'),
        ({
            'scattering_orders': [0, 0, 1]
        }, 'must not contain duplicates'),
        ({
            'scattering_orders': [0, 2]
        }, 'must also contain 1'),
        ({
            'scattering_orders': [1, 0]
        }, 'must be strictly increasing'),
        ({
            'diffusion_scales': [0]
        }, 'at least two entries'),
        ({
            'diffusion_scales': [0, 2, 1]
        }, 'strictly increasing'),
        ({
            'pool': []
        }, 'non-empty tuple'),
        ({
            'pool': ['invalid']
        }, 'Invalid pooling operations'),
        ({
            'normalization': 'invalid'
        }, "must be one of {'row', 'column'"),
    ],
)
def test_geo_scat_conv_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GeoScatConv(2, **kwargs)


def test_geo_scat_conv_forward_validation():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    conv = GeoScatConv(in_channels, pool=None)

    with pytest.raises(ValueError, match='Exactly one of edge_index'):
        conv(x)

    with pytest.raises(ValueError, match='Exactly one of edge_index'):
        conv(x, edge_index, diffusion_op=torch.eye(num_nodes))

    with pytest.raises(ValueError, match='Expected input with 2 channels'):
        conv(torch.randn(num_nodes, in_channels + 1), edge_index)


def _make_dense_lrw_diffusion_operator(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    adj = torch.zeros(num_nodes, num_nodes)
    for src, dst in edge_index.t():
        adj[src, dst] += 1.0
    degree = adj.sum(dim=1)
    adj_norm = adj / degree.unsqueeze(1).clamp(min=1.0)
    return 0.5 * (torch.eye(num_nodes) + adj_norm)


def test_geo_scat_conv_legs_construction():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GeoScatConv(in_channels, diffusion_scales='legs', pool=None)
    assert conv.use_legs is True
    assert conv.legs_kwargs == {'legs_J': 4}
    assert conv.F.shape == (5, 17)
    assert conv.num_wavelet_filters == 6
    assert conv.num_scattering_filters == 22
    assert "diffusion_scales='legs'" in str(conv)

    P = _make_dense_lrw_diffusion_operator(edge_index, num_nodes)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        out = conv(x, diffusion_op=P)
    assert out.size() == (num_nodes, in_channels, 22)


def test_geo_scat_conv_legs_init_rows():
    conv = GeoScatConv(2, diffusion_scales='legs', pool=None)
    F = conv.F
    assert F.shape == (5, 17)
    assert F[0, 0].item() == 1.0
    assert F[0, 1].item() == -1.0
    assert F[1, 1].item() == 1.0
    assert F[1, 2].item() == -1.0
    assert F[4, 8].item() == 1.0
    assert F[4, 16].item() == -1.0


def test_geo_scat_conv_legs_reset_parameters():
    conv = GeoScatConv(
        2,
        diffusion_scales='LEGS',
        legs_kwargs={'legs_J': 4},
        pool=None,
    )
    expected = _initialize_legs_parameters(4)
    conv.F.data.fill_(0.0)
    conv.reset_parameters()
    assert torch.allclose(conv.F, expected)


def test_geo_scat_conv_legs_gradient_flow():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    P = _make_dense_lrw_diffusion_operator(edge_index, num_nodes)

    conv = GeoScatConv(
        in_channels,
        diffusion_scales='legs',
        scattering_orders=[1],
        pool=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        out = conv(x, diffusion_op=P)
    out.sum().backward()
    assert conv.F.grad is not None
    assert torch.isfinite(conv.F.grad).all()


def test_geo_scat_conv_legs_lowpass_before_activation():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    P = _make_dense_lrw_diffusion_operator(edge_index, num_nodes)

    conv = GeoScatConv(
        in_channels,
        diffusion_scales='legs',
        scattering_orders=[1],
        activation=torch.abs,
        include_lowpass=True,
        pool=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        out = conv(x, diffusion_op=P)
    assert out.size() == (num_nodes, in_channels, 6)
    assert (out >= 0).all()


def test_geo_scat_conv_legs_partial_equivalence_at_init():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    P = _make_dense_lrw_diffusion_operator(edge_index, num_nodes)

    conv_legs = GeoScatConv(
        in_channels,
        diffusion_scales='legs',
        legs_kwargs={'legs_J': 4},
        include_lowpass=False,
        scattering_orders=[1],
        pool=None,
    )
    conv_fixed = GeoScatConv(
        in_channels,
        diffusion_scales=(0, 1, 2, 4, 8, 16),
        include_lowpass=False,
        scattering_orders=[1],
        pool=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with torch.no_grad():
            out_legs = conv_legs(x, diffusion_op=P)
            out_fixed = conv_fixed(x, diffusion_op=P)

    assert torch.allclose(out_legs[..., 0], out_fixed[..., 0], atol=1e-6)
    assert torch.allclose(out_legs[..., 1], out_fixed[..., 1], atol=1e-6)
    assert torch.allclose(out_legs[..., 2], out_fixed[..., 2], atol=1e-6)
    assert torch.allclose(out_legs[..., 3], out_fixed[..., 3], atol=1e-6)
    assert torch.allclose(out_legs[..., 4], out_fixed[..., 4], atol=1e-6)


def test_geo_scat_conv_legs_full_equivalence_at_init():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    P = _make_dense_lrw_diffusion_operator(edge_index, num_nodes)

    conv_legs = GeoScatConv(
        in_channels,
        diffusion_scales='legs',
        legs_kwargs={'legs_J': 4},
        scattering_orders=[0, 1, 2],
        pool=None,
    )
    conv_fixed = GeoScatConv(
        in_channels,
        diffusion_scales=(0, 1, 2, 4, 8, 16),
        scattering_orders=[0, 1, 2],
        pool=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with torch.no_grad():
            out_legs = conv_legs(x, diffusion_op=P)
            out_fixed = conv_fixed(x, diffusion_op=P)

    assert out_legs.shape == out_fixed.shape
    assert torch.allclose(out_legs, out_fixed, atol=1e-6)


def test_geo_scat_conv_legs_second_order():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    P = _make_dense_lrw_diffusion_operator(edge_index, num_nodes)

    conv = GeoScatConv(
        in_channels,
        diffusion_scales='legs',
        legs_kwargs={'legs_J': 4},
        scattering_orders=[0, 1, 2],
        pool=None,
    )
    assert conv.num_scattering_filters == 22

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        out = conv(x, diffusion_op=P)
    assert out.size() == (num_nodes, in_channels, 22)


def test_geo_scat_conv_legs_multiorder_scatter_consistency():
    in_channels = 2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))
    P = _make_dense_lrw_diffusion_operator(edge_index, num_nodes)

    conv = GeoScatConv(
        in_channels,
        diffusion_scales='legs',
        scattering_orders=[0, 1, 2],
        pool=None,
    )

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            out_conv = conv(x, diffusion_op=P)
        out_scatter = multiorder_scatter(
            x,
            P,
            F=conv.F,
            include_lowpass=conv.include_lowpass,
            scattering_orders=conv.scattering_orders,
            is_vector_feature=conv.is_vector_feature,
        )

    assert torch.allclose(out_conv, out_scatter, atol=1e-6)


@pytest.mark.parametrize(
    'kwargs,match',
    [
        ({
            'diffusion_scales': 'legs',
            'legs_kwargs': {
                'legs_J': 0
            }
        }, 'legs_J must be at least 1'),
        ({
            'diffusion_scales': 'legs',
            'legs_kwargs': {
                'legs_J': 1
            },
            'scattering_orders': [0, 1, 2],
        }, 'requires legs_J >= 2'),
        ({
            'diffusion_scales': 'legs',
            'legs_kwargs': {
                'unknown': 1
            }
        }, 'Unknown legs_kwargs keys'),
        ({
            'diffusion_scales': 'legs',
            'legs_kwargs': {}
        }, "must contain 'legs_J'"),
    ],
)
def test_geo_scat_conv_legs_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GeoScatConv(2, **kwargs)
