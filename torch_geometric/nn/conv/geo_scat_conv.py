import warnings
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

import torch_geometric.typing
from torch_geometric.nn.aggr.basic import MinAggregation, VarAggregation
from torch_geometric.nn.aggr.quantile import MedianAggregation
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import (
    contains_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    scatter,
    spmm,
    to_torch_coo_tensor,
)

DEFAULT_DIFFUSION_SCALES = (0, 1, 2, 4, 8, 16)

VALID_POOL_OPS = frozenset({'mean', 'max', 'min', 'median', 'var'})
VALID_SCATTERING_ORDERS = frozenset({0, 1, 2})

_BATCHING_DOC_URL = (
    'https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html'
)


def _raise_if_nonfinite(
    tensor: Tensor,
    *,
    name: str,
) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite value detected in {name}.")


def _apply_activation(
    x: Tensor,
    activation: Optional[Union[Module, Callable[[Tensor], Tensor]]],
) -> Tensor:
    if activation is None:
        return x
    return activation(x)


def _is_sparse_diffusion_op(diffusion_op: Union[Tensor, SparseTensor]) -> bool:
    if isinstance(diffusion_op, SparseTensor):
        return True
    if isinstance(diffusion_op, Tensor):
        return is_torch_sparse_tensor(diffusion_op)
    raise TypeError(f"Expected torch.Tensor or SparseTensor, got "
                    f"type {type(diffusion_op)}.")


def _diffusion_matmul(
    diffusion_op: Union[Tensor, SparseTensor],
    x: Tensor,
) -> Tensor:
    # reduce='sum' performs standard sparse-dense matrix multiplication.
    if _is_sparse_diffusion_op(diffusion_op):
        return spmm(_as_adj(diffusion_op), x, reduce='sum')
    return diffusion_op @ x


def _get_diffusion_op_size(diffusion_op: Union[Tensor, SparseTensor]) -> int:
    if isinstance(diffusion_op, SparseTensor):
        return int(diffusion_op.size(0))
    return int(diffusion_op.size(0))


def _validate_diffusion_op_shape(
    diffusion_op: Union[Tensor, SparseTensor],
    expected_size: int,
) -> None:
    size = _get_diffusion_op_size(diffusion_op)
    if isinstance(diffusion_op, SparseTensor):
        if diffusion_op.size(1) != size:
            raise ValueError(
                f"diffusion_op must be square (got shape "
                f"{tuple(diffusion_op.size())}).", )
    elif isinstance(diffusion_op, Tensor):
        if diffusion_op.dim() != 2 or diffusion_op.size(1) != size:
            raise ValueError(
                f"diffusion_op must be a square matrix (got shape "
                f"{tuple(diffusion_op.shape)}).", )
    if size != expected_size:
        raise ValueError(
            f"diffusion_op size {size} does not match expected size "
            f"{expected_size}.", )


def _warn_if_diffusion_op_batching_unsafe(
    diffusion_op: Union[Tensor, SparseTensor],
    key: Optional[str],
) -> None:
    is_sparse = _is_sparse_diffusion_op(diffusion_op)
    if is_sparse and key is not None and 'adj' in key:
        return

    if not is_sparse:
        warnings.warn(
            'Dense diffusion_op passed to GeoScatConv. For batched Data '
            'objects, override Data.__cat_dim__ to return (0, 1) for the '
            'operator attribute so collation block-diagonal-stacks P. '
            f'See {_BATCHING_DOC_URL}.',
            stacklevel=3,
        )
    elif key is None:
        warnings.warn(
            'Sparse diffusion_op passed without diffusion_op_key. Cannot '
            'verify that Data batching will block-diagonal-stack P. Pass '
            "diffusion_op_key with 'adj' in the name (e.g. 'diffusion_adj') "
            f'if P was stored on Data objects, or see {_BATCHING_DOC_URL}.',
            stacklevel=3,
        )
    else:
        warnings.warn(
            f"Sparse diffusion_op passed with diffusion_op_key='{key}' "
            "which does not contain 'adj'. PyG will not block-diagonal-stack "
            "this operator during batching by default. Rename the attribute "
            "to include 'adj' (e.g. 'diffusion_adj') or override "
            f'Data.__cat_dim__. See {_BATCHING_DOC_URL}.',
            stacklevel=3,
        )


def _validate_scattering_orders(scattering_orders: Tuple[int, ...], ) -> None:
    if not scattering_orders:
        raise ValueError('scattering_orders must be a non-empty tuple.')
    if len(scattering_orders) != len(set(scattering_orders)):
        raise ValueError('scattering_orders must not contain duplicates.')
    invalid = set(scattering_orders) - VALID_SCATTERING_ORDERS
    if invalid:
        raise ValueError(
            f"scattering_orders entries must be in {{0, 1, 2}} "
            f"(got invalid values {sorted(invalid)}).", )
    if any(scattering_orders[i] >= scattering_orders[i + 1]
           for i in range(len(scattering_orders) - 1)):
        raise ValueError('scattering_orders must be strictly increasing.')
    if 2 in scattering_orders and 1 not in scattering_orders:
        raise ValueError(
            'scattering_orders containing 2 must also contain 1.', )


def _as_adj(matrix: Union[Tensor, SparseTensor]) -> Adj:
    if isinstance(matrix, SparseTensor):
        return matrix
    if not isinstance(matrix, Tensor):
        raise TypeError(
            f"Expected torch.Tensor or SparseTensor, got type {type(matrix)}.",
        )
    if matrix.layout != torch.sparse_coo:
        matrix = matrix.to_sparse()
    matrix = matrix.coalesce()
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        row, col = matrix.indices()
        return SparseTensor(
            row=row,
            col=col,
            value=matrix.values(),
            sparse_sizes=matrix.size(),
        ).coalesce()
    return matrix


def _sparse_identity(
    size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    indices = torch.arange(size, device=device).unsqueeze(0).repeat(2, 1)
    values = torch.ones(size, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, (size, size))


def _get_sparse_normalized_A(
    edge_index: Tensor,
    edge_weight: OptTensor,
    num_nodes: int,
    normalization: Literal['row', 'column', 'symmetric'],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if normalization not in {'row', 'column', 'symmetric'}:
        raise ValueError(
            "normalization must be one of {'row', 'column', 'symmetric'}, "
            f"got '{normalization}'.", )

    edge_index = edge_index.to(device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=device)
    else:
        edge_weight = edge_weight.to(device=device, dtype=dtype)

    row = edge_index[0]
    col = edge_index[1]
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)

    if normalization == 'row':
        degree.index_add_(0, row, edge_weight)
        norm = torch.where(degree > 0, 1.0 / degree, torch.zeros_like(degree))
        normalized_weight = edge_weight * norm[row]
    elif normalization == 'column':
        degree.index_add_(0, col, edge_weight)
        norm = torch.where(degree > 0, 1.0 / degree, torch.zeros_like(degree))
        normalized_weight = edge_weight * norm[col]
    else:
        degree.index_add_(0, row, edge_weight)
        norm = torch.where(
            degree > 0,
            torch.rsqrt(degree),
            torch.zeros_like(degree),
        )
        normalized_weight = edge_weight * norm[row] * norm[col]

    return to_torch_coo_tensor(
        edge_index,
        normalized_weight,
        (num_nodes, num_nodes),
    )


def get_sparse_lrw_diffusion_operator(
    edge_index: Tensor,
    edge_weight: OptTensor,
    num_nodes: int,
    normalization: Literal['row', 'column', 'symmetric'],
    dtype: torch.dtype,
    device: torch.device,
) -> Adj:
    if contains_self_loops(edge_index):
        warnings.warn(
            'Self-loops detected in edge_index and will be removed before '
            'building the diffusion operator. Lazy self-transitions are '
            'already provided by the identity term in '
            'P = 0.5 * (I + A_norm).',
            stacklevel=3,
        )
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='Sparse CSR tensor support is in beta state.*',
            category=UserWarning,
        )
        sparse_norm_A = _get_sparse_normalized_A(
            edge_index,
            edge_weight,
            num_nodes,
            normalization,
            dtype,
            device,
        )
        identity = _sparse_identity(
            num_nodes,
            dtype=dtype,
            device=device,
        )
        P_sparse = 0.5 * (identity + sparse_norm_A)
        return _as_adj(P_sparse.coalesce())


def _subset_second_order_wavelets(
    W2raw: Tensor,
    *,
    feature_type: Literal['scalar', 'vector'],
) -> Tensor:
    if feature_type not in ('scalar', 'vector'):
        raise ValueError(
            f"feature_type must be 'scalar' or 'vector', got {feature_type}.",
        )

    n_prev = int(W2raw.shape[-2])
    n_next = int(W2raw.shape[-1])
    if n_prev < 2:
        raise ValueError(
            'Second-order scattering requires at least two first-order '
            'wavelet filters.', )

    mask = torch.triu(
        torch.ones(n_prev, n_next, dtype=torch.bool, device=W2raw.device),
        diagonal=1,
    )

    if feature_type == 'scalar':
        if W2raw.ndim != 4:
            raise ValueError(
                'Scalar 2nd-order tensors must have shape '
                '(N, C, W_prev, W_next).', )
        return W2raw[:, :, mask]

    if W2raw.ndim != 3:
        raise ValueError(
            'Vector 2nd-order tensors must have shape (N*d, W_prev, W_next).',
        )
    return W2raw[:, mask].unsqueeze(1)


def diffusion_wavelet_transform(
    x: Tensor,
    P: Union[Tensor, SparseTensor],
    diffusion_scales: Tensor,
    include_lowpass: bool = True,
    filter_stack_dim: int = -1,
    check_nonfinite: bool = False,
) -> Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(1)

    if isinstance(P, SparseTensor):
        if P.dtype() != x.dtype:
            P = P.to(dtype=x.dtype)
    elif isinstance(P, Tensor):
        if _is_sparse_diffusion_op(P):
            if P.dtype != x.dtype:
                P = P.to(dtype=x.dtype)
        elif P.dtype != x.dtype:
            P = P.to(dtype=x.dtype)

    if check_nonfinite:
        if isinstance(P, SparseTensor):
            _raise_if_nonfinite(
                P.storage.value(),
                name='GeoScatConv: P values',
            )
        elif isinstance(P, Tensor):
            if _is_sparse_diffusion_op(P):
                _raise_if_nonfinite(
                    P.values(),
                    name='GeoScatConv: P values',
                )
            else:
                _raise_if_nonfinite(P, name='GeoScatConv: P')
        _raise_if_nonfinite(x, name='GeoScatConv: x')

    if not isinstance(diffusion_scales, Tensor):
        diffusion_scales = torch.as_tensor(
            diffusion_scales,
            dtype=torch.long,
            device=x.device,
        )
    else:
        diffusion_scales = diffusion_scales.to(device=x.device,
                                               dtype=torch.long)

    if diffusion_scales.dim() != 1:
        raise ValueError(
            f"diffusion_scales must be one-dimensional "
            f"(got {diffusion_scales.dim()} dimensions).", )

    device = x.device
    powers_to_save = diffusion_scales
    range_upper_lim = int(diffusion_scales.numel())

    Ptxs = [x.to(device)]
    Ptx = x.to(device)
    max_power = int(powers_to_save[-1].item())

    for j in range(1, max_power + 1):
        Ptx = _diffusion_matmul(P, Ptx)
        if check_nonfinite:
            _raise_if_nonfinite(Ptx, name=f'GeoScatConv: P^t x (t={j})')
        if j in powers_to_save:
            j_ct = int((powers_to_save == j).sum().item())
            for _ in range(j_ct):
                Ptxs.append(Ptx.to(device))

    Wjxs = [Ptxs[j - 1] - Ptxs[j] for j in range(1, range_upper_lim)]
    if include_lowpass:
        Wjxs.append(Ptxs[-1])
    return torch.stack(Wjxs, dim=filter_stack_dim).to(device)


def multiorder_scatter(
    x: Tensor,
    diffusion_op: Union[Tensor, SparseTensor],
    *,
    diffusion_scales: Tensor,
    include_lowpass: bool,
    scattering_orders: Tuple[int, ...],
    is_vector_feature: bool = False,
    activation: Optional[Union[Module, Callable[[Tensor], Tensor]]] = None,
    check_nonfinite: bool = False,
) -> Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(1)

    if is_vector_feature:
        num_nodes, vector_dim = x.shape
        flat = x.reshape(num_nodes * vector_dim, 1)
        feature_type: Literal['scalar', 'vector'] = 'vector'
    else:
        num_nodes = x.shape[0]
        flat = x
        feature_type = 'scalar'

    scatter_kwargs = {
        'diffusion_scales': diffusion_scales,
        'include_lowpass': include_lowpass,
        'check_nonfinite': check_nonfinite,
    }

    need_first_order = 1 in scattering_orders or 2 in scattering_orders
    coeffs: List[Tensor] = []

    if 0 in scattering_orders:
        coeffs.append(flat.unsqueeze(-1))

    W1: Optional[Tensor] = None
    if need_first_order:
        W1 = diffusion_wavelet_transform(
            x=flat,
            P=diffusion_op,
            **scatter_kwargs,
        )
        W1 = _apply_activation(W1, activation)
        if 1 in scattering_orders:
            coeffs.append(W1)

    if 2 in scattering_orders and W1 is not None:
        num_wavelets = int(W1.shape[-1])
        if num_wavelets > 1:
            if is_vector_feature:
                x_second = W1.squeeze(1)
                W2raw = diffusion_wavelet_transform(
                    x=x_second,
                    P=diffusion_op,
                    **scatter_kwargs,
                )
                nd = x_second.shape[0]
                W2raw = W2raw.view(nd, num_wavelets, -1)
            else:
                num_channels = int(W1.shape[1])
                x_second = W1.reshape(num_nodes, num_channels * num_wavelets)
                W2raw = diffusion_wavelet_transform(
                    x=x_second,
                    P=diffusion_op,
                    **scatter_kwargs,
                )
                W2raw = W2raw.view(num_nodes, num_channels, num_wavelets, -1)
            W2 = _subset_second_order_wavelets(
                W2raw,
                feature_type=feature_type,
            )
            W2 = _apply_activation(W2, activation)
            coeffs.append(W2)

    if not coeffs:
        raise ValueError('scattering_orders produced no output coefficients.')

    W_tot = torch.cat(coeffs, dim=-1)
    if is_vector_feature:
        return W_tot.view(num_nodes, vector_dim, -1)
    return W_tot


def _compute_num_wavelet_filters(
    diffusion_scales: Tuple[int, ...],
    include_lowpass: bool,
) -> int:
    return (len(diffusion_scales) - 1) + int(include_lowpass)


def _compute_num_scattering_filters(
    num_wavelet_filters: int,
    scattering_orders: Tuple[int, ...],
) -> int:
    total = 0
    if 0 in scattering_orders:
        total += 1
    if 1 in scattering_orders:
        total += num_wavelet_filters
    if 2 in scattering_orders:
        W = num_wavelet_filters
        total += W * (W - 1) // 2
    return total


def _pool_scattering_coefficients(
    coeffs: Tensor,
    batch: OptTensor,
    pool: Tuple[str, ...],
) -> Tensor:
    num_nodes, num_features, num_filters = coeffs.shape
    x_flat = coeffs.reshape(num_nodes, num_features * num_filters)

    if batch is None:
        batch = coeffs.new_zeros(num_nodes, dtype=torch.long)

    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

    min_aggr = MinAggregation()
    median_aggr = MedianAggregation()
    var_aggr = VarAggregation()

    pooled: List[Tensor] = []
    for op in pool:
        if op == 'mean':
            out = scatter(x_flat, batch, dim=0, dim_size=num_graphs,
                          reduce='mean')
        elif op == 'max':
            out = scatter(x_flat, batch, dim=0, dim_size=num_graphs,
                          reduce='max')
        elif op == 'min':
            out = min_aggr(x_flat, batch, dim_size=num_graphs)
        elif op == 'median':
            out = median_aggr(x_flat, batch, dim_size=num_graphs)
        elif op == 'var':
            out = var_aggr(x_flat, batch, dim_size=num_graphs)
        else:
            raise ValueError(
                f"Unknown pooling operation '{op}'. "
                f"Valid options are {sorted(VALID_POOL_OPS)}.", )
        pooled.append(out.view(num_graphs, num_features, num_filters))

    return torch.cat(pooled, dim=-1)


class GeoScatConv(Module):
    r"""Implements the geometric scattering transform built from
    graph diffusion wavelets, introduced in the `"Geometric
    Scattering for Graph Data Analysis"
    <https://arxiv.org/abs/1810.03068>`_ and `"Diffusion
    Scattering Transforms on Graphs"
    <https://arxiv.org/abs/1806.08829>`_ papers. This technique
    has been utilized and extended further in, for example,
    `BLIS-Net <https://proceedings.mlr.press/v238/xu24c.html>`_,
    `HiPoNet <https://arxiv.org/abs/2502.07746>`_, and
    `VDW-GNNs <https://arxiv.org/abs/2510.01022>`_.

    In particular, this layer computes diffusion wavelet scattering
    coefficients on graph node features. A diffusion wavelet
    convolution of a graph signal :math:`\mathbf{x}` has the
    general form

    .. math::
        \mathbf{W}_j \, \mathbf{x} = (\mathbf{P}^{t_{j-1}} -
        \mathbf{P}^{t_{j}}) \mathbf{x}, \quad j = 1, \ldots, J,

    where :math:`\mathbf{P}` is a lazy random walk operator
    (row-normalized by default), and :math:`t_j` are the diffusion
    scales.
    A low-pass filter :math:`A_J = \mathbf{P}^{t_J} \mathbf{x}`
    is also included by default, such that the full filter bank is
    :math:`\{\mathbf{W}_j\}_{j=0}^{J} \cup \{A_J\}`. The low-pass
    filter can be excluded by setting :obj:`include_lowpass`
    to :obj:`False`.

    By default, the :math:`t_j` are set to be dyadic integers
    :math:`0, 1, 2, \ldots, 16`. However, one may also specify a
    custom set of scales using the :obj:`diffusion_scales`
    argument (pass a tuple of integers, or a tuple of tuples,
    for different diffusion scales for each feature channel).
    One option for generating custom scales is `InfoGain Wavelets
    <https://arxiv.org/abs/2504.08802>`_.

    This layer returns concatenated zeroeth and first-order scattering
    coefficients by default. The zeroeth-order scattering coefficient
    is the unfiltered input feature vector, and can be excluded by
    setting :obj:`scattering_orders` to :obj:`(1,)`. Second-order
    scattering coefficients can be included by including :obj:`2` in
    the :obj:`scattering_orders` tuple. These are defined as:

    .. math::
        \sigma(\mathbf{W}_{j^{\,\prime}} \, \sigma (\mathbf{W}_j \,
        \mathbf{x})), \quad j^{\, \prime} > j,

    where :math:`\sigma` is an optional activation (e.g. the modulus
    operator via :obj:`torch.abs`), disabled by default.

    Note there is not an :obj:`out_channels` property for this layer,
    as the number of output channels is deterministic via the
    parameterization of the scattering transform. Including 0th-order
    scattering coefficients increases the number of output channels by
    1; including first-order scattering coefficients increases the number
    of output channels by the number of wavelets :math:`w`; including
    second-order scattering coefficients increases the number of output
    channels by :math:`w \cdot (w - 1) / 2`.

    For graph-level tasks, scattering coefficients can be pooled across
    nodes within each graph and feature channel, changing the output shape
    from :math:`(|\mathcal{V}|, F_{\mathrm{in}}, W_{\mathrm{total}})` to
    :math:`(B, F_{\mathrm{in}}, W_{\mathrm{total}} \cdot |\mathrm{pool}|)`.
    Currently, the built-in options for pooling are :obj:`"mean"`,
    :obj:`"max"`, :obj:`"min"`, :obj:`"median"`, and :obj:`"var"`.
    Alternatively, leave :obj:`pool` as :obj:`None` to return node-level
    scattering coefficients, and apply other pooling operations downstream
    of this layer.

    Vector-valued graph signals (as explored in the `VDW-GNNs paper
    <https://arxiv.org/abs/2510.01022>`_) are supported by
    setting :obj:`is_vector_feature` to :obj:`True`. In this mode, node
    features :math:`\mathbf{x} \in \mathbb{R}^{n \times d}` are flattened
    internally and a user-supplied diffusion operator
    :math:`\mathbf{P} \in \mathbb{R}^{nd \times nd}` must be passed via
    :obj:`diffusion_op` (for example, stored on
    :class:`~torch_geometric.data.Data` under an attribute such as
    :obj:`diffusion_adj`). Lazy random walk operators built from
    :obj:`edge_index` are not supported for vector features.

    .. note::
        Self-loops in :obj:`edge_index` are automatically removed before
        normalization of the lazy random walk operator, which includes
        self-transitions due to the laziness of the random walk (i.e.,
        a nonzero probability of staying at the same node).

    .. note::
        A cached diffusion operator :obj:`P` may be passed via
        :obj:`diffusion_op` instead of recomputed from the :obj:`edge_index`
        in every forward pass. Store (sparse)
        :obj:`P` under an attribute name containing :obj:`'adj'` (e.g.
        :obj:`P_adj`) so that PyG block-diagonal-stacks operators
        during collation, and pass this name to :obj:`diffusion_op_key`
        accordingly. (Block-diagonal collation is required to ensure that,
        in multi-graph settings, :obj:`P` is correctly block-diagonal-stacked
        so it applies only to nodes within its own graph in a batched
        :class:`~torch_geometric.data.Data` object.) Forcing block-diagonal
        collation of operators keyed under arbitrary names requires overriding
        :meth:`~torch_geometric.data.Data.__cat_dim__`; see `Advanced
        Mini-Batching
        <https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html>`_.

    Args:
        in_channels (int): Size of each input sample. For vector features,
            this is the vector dimension :math:`d`.
        scattering_orders (Tuple[int, ...], optional): Scattering orders to
            include in the output. Each entry must be :obj:`0` (unfiltered
            input), :obj:`1` (first-order wavelets), or :obj:`2`
            (second-order). Pass :obj:`(1,)` for a diffusion wavelet
            transform only. (default: :obj:`(0, 1)`)
        diffusion_scales (Tuple[int, ...], optional): Monotonically increasing
            diffusion powers, i.e., :math:`t_j` in each `:math:`P^{t_j}`.
            Wavelet filters are consecutive differences between adjacent
            :math:`t_{j-1}` and :math:`t_j`.
            (default: :obj:`(0, 1, 2, 4, 8, 16)`)
        include_lowpass (bool, optional): If set to :obj:`True`, append the
            low-pass filter :math:`P^{t_J} \mathbf{x}`.
            (default: :obj:`True`)
        activation (torch.nn.Module or Callable, optional): Activation
            applied to first- and second-order scattering coefficients, e.g.
            :obj:`torch.abs` for the modulus. (default: :obj:`None`)
        normalization (str, optional): Normalization scheme for the adjacency
            matrix before building the lazy random walk operator
            :math:`P`. Ignored when :obj:`diffusion_op` is provided. Options:
            :obj:`"row"`, :obj:`"column"`, :obj:`"symmetric"`.
            (default: :obj:`"row"`)
        is_vector_feature (bool, optional): If set to :obj:`True`, treat each
            node feature as a vector signal of dimension :obj:`in_channels`
            and require a precomputed :obj:`diffusion_op` of shape
            :math:`(n \cdot d, n \cdot d)`.
            (default: :obj:`False`)
        pool (Tuple[str, ...], optional): Graph-level pooling operations to
            apply across nodes. Multiple pooling operations can be applied by
            passing a tuple of strings, the results of which are concatenated.
            Each entry must be one of :obj:`"mean"`, :obj:`"max"`,
            :obj:`"min"`, :obj:`"median"`, or  :obj:`"var"`. If :obj:`None`,
            node-level scattering coefficients are returned.
            (default: :obj:`None`)
        check_nonfinite (bool, optional): If set to :obj:`True`, raise an
            error when non-finite values are detected in inputs or
            intermediate results. (default: :obj:`False`). Useful for
            debugging.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`(|\mathcal{V}|, d)` for vector features,
          edge indices :math:`(2, |\mathcal{E}|)` *(optional if*
          :obj:`diffusion_op` *is set)*,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*,
          batch vector :math:`(|\mathcal{V}|)` *(optional, required for
          graph-level pooling on batched graphs)*
        - **output (node-level,** :obj:`pool=None` **):**
          scattering coefficients
          :math:`(|\mathcal{V}|, F_{in}, W_{\mathrm{total}})`
        - **output (graph-level,** :obj:`pool` **set):**
          pooled scattering coefficients
          :math:`(B, F_{in}, W_{\mathrm{total}} \cdot |\texttt{pool}|)`
    """
    def __init__(
        self,
        in_channels: int,
        scattering_orders: Optional[Tuple[int, ...]] = (0, 1),
        diffusion_scales: Optional[Tuple[int, ...]] = (0, 1, 2, 4, 8, 16),
        include_lowpass: bool = True,
        activation: Optional[Union[Module, Callable[[Tensor], Tensor]]] = None,
        normalization: Literal['row', 'column', 'symmetric'] = 'row',
        is_vector_feature: bool = False,
        pool: Optional[Tuple[str, ...]] = None,
        check_nonfinite: bool = False,
    ):
        super().__init__()

        if scattering_orders is None:
            scattering_orders = (0, 1)
        else:
            scattering_orders = tuple(scattering_orders)
        _validate_scattering_orders(scattering_orders)

        if normalization not in {'row', 'column', 'symmetric'}:
            raise ValueError(
                "normalization must be one of {'row', 'column', 'symmetric'}, "
                f"got '{normalization}'.", )

        if diffusion_scales is None:
            diffusion_scales = DEFAULT_DIFFUSION_SCALES
        else:
            diffusion_scales = tuple(diffusion_scales)
        if len(diffusion_scales) < 2:
            raise ValueError(
                'diffusion_scales must contain at least two entries.', )
        if any(diffusion_scales[i] >= diffusion_scales[i + 1]
               for i in range(len(diffusion_scales) - 1)):
            raise ValueError('diffusion_scales must be strictly increasing.')

        if pool is not None:
            pool = tuple(pool)
            if len(pool) == 0:
                raise ValueError('pool must be None or a non-empty tuple.')
            invalid = set(pool) - VALID_POOL_OPS
            if invalid:
                raise ValueError(
                    f"Invalid pooling operations {sorted(invalid)}. "
                    f"Valid options are {sorted(VALID_POOL_OPS)}.", )

        self.in_channels = in_channels
        self.scattering_orders = scattering_orders
        self.diffusion_scales = diffusion_scales
        self.include_lowpass = include_lowpass
        self.activation = activation
        self.normalization = normalization
        self.is_vector_feature = is_vector_feature
        self.pool = pool
        self.check_nonfinite = check_nonfinite

    @property
    def num_wavelet_filters(self) -> int:
        return _compute_num_wavelet_filters(
            self.diffusion_scales,
            self.include_lowpass,
        )

    @property
    def num_scattering_filters(self) -> int:
        return _compute_num_scattering_filters(
            self.num_wavelet_filters,
            self.scattering_orders,
        )

    @property
    def out_channels(self) -> int:
        num_filters = self.num_scattering_filters
        if self.pool is None:
            return num_filters
        return num_filters * len(self.pool)

    def reset_parameters(self) -> None:
        """This layer does not have any learnable parameters to reset.
        It is intended as a multiscale feature extractor layer for graph
        data, whose coefficients can be fed into learnable layers downstream.
        """

    def forward(
        self,
        x: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        diffusion_op: Optional[Union[Tensor, SparseTensor]] = None,
        diffusion_op_key: Optional[str] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x: Node features of shape :math:`(|\mathcal{V}|, F_{\mathrm{in}})`,
                or :math:`(|\mathcal{V}|, d)` for vector features.
            edge_index: Edge indices of shape :math:`(2, |\mathcal{E}|)`.
            edge_weight: Edge weights of shape :math:`(|\mathcal{E}|)`.
            batch: Batch vector of shape :math:`(|\mathcal{V}|)`.
            diffusion_op: Precomputed diffusion operator of shape
                :math:`(n, n)` for scalar node features, or
                :math:`(n \cdot d, n \cdot d)` for vector features.
            diffusion_op_key: Key of the precomputed diffusion operator in the
                Data object(s). Should contain the substring 'adj' for proper
                block-diagonal collation; otherwise, the :obj:`__cat_dim__`
                method of the :obj:`Data` object(s) will need to be overridden.

        Returns:
            Scattering coefficients of shape
            :math:`(|\mathcal{V}|, F_{\mathrm{in}}, W_{\mathrm{total}})`,
            or :math:`(B, F_{\mathrm{in}}, W_{\mathrm{total}}
            \cdot |\mathrm{pool}|)` if :obj:`pool` is set.
        """
        if x.size(-1) != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels "
                f"(got {x.size(-1)}).", )

        num_nodes = x.size(0)

        if self.is_vector_feature:
            if diffusion_op is None:
                raise ValueError(
                    'Vector features require a precomputed diffusion_op of '
                    'shape (num_nodes * in_channels, num_nodes * '
                    'in_channels). Store it on Data objects (e.g. as '
                    "'diffusion_adj') and pass it to forward.", )
            if edge_index is not None:
                raise ValueError(
                    'When is_vector_feature is True, pass diffusion_op only '
                    '(not edge_index).', )
            expected_size = num_nodes * self.in_channels
            _validate_diffusion_op_shape(diffusion_op, expected_size)
            _warn_if_diffusion_op_batching_unsafe(
                diffusion_op,
                diffusion_op_key,
            )
            op = diffusion_op
        elif diffusion_op is not None:
            if edge_index is not None:
                raise ValueError(
                    'Exactly one of edge_index or diffusion_op must be '
                    'provided.', )
            _validate_diffusion_op_shape(diffusion_op, num_nodes)
            _warn_if_diffusion_op_batching_unsafe(diffusion_op,
                                                  diffusion_op_key)
            op = diffusion_op
        else:
            if edge_index is None:
                raise ValueError(
                    'Exactly one of edge_index or diffusion_op must be '
                    'provided.', )
            op = get_sparse_lrw_diffusion_operator(
                edge_index,
                edge_weight,
                num_nodes,
                self.normalization,
                x.dtype,
                x.device,
            )

        diffusion_scales = torch.as_tensor(
            self.diffusion_scales,
            dtype=torch.long,
            device=x.device,
        )

        coeffs = multiorder_scatter(
            x,
            op,
            diffusion_scales=diffusion_scales,
            include_lowpass=self.include_lowpass,
            scattering_orders=self.scattering_orders,
            is_vector_feature=self.is_vector_feature,
            activation=self.activation,
            check_nonfinite=self.check_nonfinite,
        )

        if self.pool is None:
            return coeffs

        return _pool_scattering_coefficients(coeffs, batch, self.pool)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'scattering_orders={self.scattering_orders}, '
                f'num_scattering_filters={self.num_scattering_filters}, '
                f'normalization={self.normalization}, '
                f'pool={self.pool})')
