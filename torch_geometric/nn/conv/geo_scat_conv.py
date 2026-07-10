import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

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
VALID_LEGS_KWARGS = frozenset({'legs_J'})
DEFAULT_LEGS_KWARGS: Dict[str, Any] = {'legs_J': 4}

DiffusionScalesArg = Union[Tuple[int, ...], Literal['legs', 'LEGS']]

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


def _is_legs_diffusion_scales(diffusion_scales: DiffusionScalesArg, ) -> bool:
    return isinstance(diffusion_scales, str) \
        and diffusion_scales.lower() == 'legs'


def _validate_legs_kwargs(
    legs_kwargs: Optional[Dict[str, Any]],
    scattering_orders: Tuple[int, ...],
) -> Dict[str, Any]:
    if legs_kwargs is None:
        legs_kwargs = dict(DEFAULT_LEGS_KWARGS)
    else:
        legs_kwargs = dict(legs_kwargs)

    unknown = set(legs_kwargs) - VALID_LEGS_KWARGS
    if unknown:
        raise ValueError(
            f"Unknown legs_kwargs keys {sorted(unknown)}. "
            f"Valid options are {sorted(VALID_LEGS_KWARGS)}.", )

    if 'legs_J' not in legs_kwargs:
        raise ValueError("legs_kwargs must contain 'legs_J'.")

    legs_J = legs_kwargs['legs_J']
    if not isinstance(legs_J, int) or isinstance(legs_J, bool):
        raise ValueError(
            f"legs_J must be an integer (got {type(legs_J).__name__}).", )
    if legs_J < 1:
        raise ValueError('legs_J must be at least 1.')
    if 2 in scattering_orders and legs_J < 2:
        raise ValueError(
            'Second-order scattering with LEGS requires legs_J >= 2.', )

    return legs_kwargs


def _prepare_diffusion_op(
    P: Union[Tensor, SparseTensor],
    x: Tensor,
) -> Union[Tensor, SparseTensor]:
    if isinstance(P, SparseTensor):
        if P.dtype() != x.dtype:
            return P.to(dtype=x.dtype)
        return P
    if _is_sparse_diffusion_op(P):
        if P.dtype != x.dtype:
            return P.to(dtype=x.dtype)
        return P
    if P.dtype != x.dtype:
        return P.to(dtype=x.dtype)
    return P


def _check_diffusion_inputs_nonfinite(
    P: Union[Tensor, SparseTensor],
    x: Tensor,
    *,
    check_nonfinite: bool,
) -> None:
    if not check_nonfinite:
        return
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


def _compute_pt_xs(
    x: Tensor,
    P: Union[Tensor, SparseTensor],
    max_power: int,
    *,
    check_nonfinite: bool = False,
) -> Tensor:
    """Stack diffusion powers ``P^t x`` for ``t = 0, ..., max_power``.

    Returns a tensor of shape ``(max_power + 1, num_nodes, num_features)``,
    where row ``t`` holds ``P^t x``.
    """
    if x.ndim == 1:
        x = x.unsqueeze(1)

    P = _prepare_diffusion_op(P, x)
    _check_diffusion_inputs_nonfinite(P, x, check_nonfinite=check_nonfinite)

    device = x.device
    pt_xs = [x.to(device)]
    pt_x = x.to(device)

    for t in range(1, max_power + 1):
        pt_x = _diffusion_matmul(P, pt_x)
        if check_nonfinite:
            _raise_if_nonfinite(pt_x, name=f'GeoScatConv: P^t x (t={t})')
        pt_xs.append(pt_x.to(device))

    return torch.stack(pt_xs, dim=0)


def _subset_second_order_wavelets(
    W2raw: Tensor,
    *,
    feature_type: Literal['scalar', 'vector'],
) -> Tensor:
    """Keep only second-order terms where the outer filter is higher-pass.

    ``W2raw`` holds all pairs ``(W_prev, W_next)`` of first- and second-order
    wavelet indices. Scattering theory requires ``W_next`` to be applied to
    the output of a lower-pass ``W_prev``, i.e. ``j' > j``. The upper-
    triangular mask (excluding the diagonal) enforces that ordering.
    """
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
        diagonal=1,  # keep pairs with second index > first index
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
    """Fixed-scale wavelet transform via consecutive diffusion differences.

    For scales ``(t_0, t_1, ..., t_J)``, filter ``j`` is
    ``P^{t_{j-1}} x - P^{t_j} x``. Optionally appends the lowpass
    ``P^{t_J} x`` (non-learnable).
    """
    if not isinstance(diffusion_scales, Tensor):
        diffusion_scales = torch.as_tensor(
            diffusion_scales,
            dtype=torch.long,
            device=x.device,
        )
    else:
        diffusion_scales = diffusion_scales.to(
            device=x.device,
            dtype=torch.long,
        )

    if diffusion_scales.dim() != 1:
        raise ValueError(
            f"diffusion_scales must be one-dimensional "
            f"(got {diffusion_scales.dim()} dimensions).", )

    max_power = int(diffusion_scales[-1].item())
    pt_xs = _compute_pt_xs(
        x,
        P,
        max_power,
        check_nonfinite=check_nonfinite,
    )
    pt_at_scales = pt_xs[diffusion_scales]
    num_scales = int(diffusion_scales.numel())

    # Wavelet j approximates (P^{t_{j-1}} - P^{t_j}) x via stored powers.
    wjxs = [
        pt_at_scales[j - 1] - pt_at_scales[j] for j in range(1, num_scales)
    ]
    if include_lowpass:
        wjxs.append(pt_at_scales[-1])  # append P^{t_J} x before activation
    return torch.stack(wjxs, dim=filter_stack_dim).to(x.device)


def _legs_max_power(F: Tensor) -> int:
    """Return ``T = 2^J`` from a LEGS matrix with ``J + 1`` filter rows."""
    return 2**(int(F.size(0)) - 1)


def _initialize_legs_parameters(J: int = 4) -> Tensor:
    r"""Initialize the LEGS selector matrix :math:`F` with dyadic scales.

    For ``J = 4``, ``F`` has shape ``(5, 17)``: ``J + 1`` learnable wavelet
    rows matching ``(P^{t_{j-1}} - P^{t_j})`` for scales
    ``(0, 1, 2, 4, 8, 16)``, and ``2^J + 1`` diffusion-power columns.
    """
    num_rows = J + 1
    T = 2**J
    m = torch.zeros((num_rows, T + 1))
    for r in range(num_rows):
        lower = 0 if r == 0 else 2**(r - 1)
        upper = 2**r
        m[r, lower] = 1.0
        m[r, upper] = -1.0
    return m.to(torch.float)


def legs_wavelet_transform(
    x: Tensor,
    P: Union[Tensor, SparseTensor],
    F: Tensor,
    *,
    include_lowpass: bool,
    filter_stack_dim: int = -1,
    check_nonfinite: bool = False,
) -> Tensor:
    """LEGS first-order transform: learnable linear combos of diffusion powers.

    Row ``j`` of ``F`` selects which ``P^t x`` terms form wavelet ``j``.
    At initialization, each row is a dyadic difference; during training ``F``
    can learn a continuous approximation to discrete wavelet scales.
    """
    T = _legs_max_power(F)
    pt_xs = _compute_pt_xs(
        x,
        P,
        T,
        check_nonfinite=check_nonfinite,
    )
    # F[j, t] weights P^t x; sum over t gives the j-th first-order coefficient.
    w1 = torch.einsum('jt,tnf->nfj', F, pt_xs)
    if include_lowpass:
        # Lowpass P^T x is fixed (not a row of F); concat before activation.
        w1 = torch.cat([w1, pt_xs[T].unsqueeze(-1)], dim=-1)
    if filter_stack_dim != -1:
        w1 = w1.movedim(-1, filter_stack_dim)
    return w1


def _prepare_scatter_inputs(
    x: Tensor,
    is_vector_feature: bool,
) -> Tuple[Tensor, int, int, Literal['scalar', 'vector']]:
    if is_vector_feature:
        num_nodes, vector_dim = x.shape
        flat = x.reshape(num_nodes * vector_dim, 1)
        return flat, num_nodes, vector_dim, 'vector'

    num_nodes = x.shape[0]
    return x, num_nodes, x.size(-1), 'scalar'


def _compute_second_order_scattering(
    W1: Tensor,
    diffusion_op: Union[Tensor, SparseTensor],
    *,
    diffusion_scales: Optional[Tensor],
    F: Optional[Tensor],
    include_lowpass: bool,
    is_vector_feature: bool,
    num_nodes: int,
    feature_type: Literal['scalar', 'vector'],
    check_nonfinite: bool = False,
) -> Tensor:
    """Compute ``W_{j'}(W_j x)`` and retain only pairs with ``j' > j``.

    ``W1`` must already include any activation from first order. The raw
    second-order tensor ``W2raw[..., j, k]`` holds filter ``k`` applied to
    first-order coefficient ``j``; ``_subset_second_order_wavelets`` then
    drops invalid (lower-on-lower) pairs.
    """
    num_wavelets = int(W1.shape[-1])
    if num_wavelets <= 1:
        raise ValueError(
            'Second-order scattering requires at least two first-order '
            'wavelet filters.', )

    if F is not None:
        num_f_rows = int(F.size(0))
        T = _legs_max_power(F)
        w2_parts: List[Tensor] = []
        lowpass_parts: List[Tensor] = []
        for j in range(num_f_rows):
            # One power stack per first-order channel;
            # reuse for einsum and lowpass.
            pt_xs = _compute_pt_xs(
                W1[..., j],
                diffusion_op,
                T,
                check_nonfinite=check_nonfinite,
            )
            w2_parts.append(
                torch.einsum('kt,tnf->nfk', F, pt_xs).unsqueeze(-2))
            if include_lowpass:
                lowpass_parts.append(pt_xs[T])

        w2_learnable = torch.cat(w2_parts, dim=-2)

        if include_lowpass:
            num_wavelets = num_f_rows + 1
            # Lowpass second-order: P^T applied to W_j x (from pt_xs[T] above).
            lowpass_on_j = torch.stack(lowpass_parts, dim=-1)
            if is_vector_feature:
                nd = W1.shape[0]
                w2_full = w2_learnable.new_zeros(nd, num_wavelets,
                                                 num_wavelets)
                w2_full[:, :num_f_rows, :num_f_rows] = w2_learnable.view(
                    nd, num_f_rows, num_f_rows)
                w2_full[:, :num_f_rows, num_f_rows] = lowpass_on_j.squeeze(1)
                W2raw = w2_full
            else:
                num_channels = int(W1.shape[1])
                w2_full = w2_learnable.new_zeros(
                    num_nodes,
                    num_channels,
                    num_wavelets,
                    num_wavelets,
                )
                w2_full[:, :, :num_f_rows, :num_f_rows] = w2_learnable.view(
                    num_nodes, num_channels, num_f_rows, num_f_rows)
                w2_full[:, :, :num_f_rows, num_f_rows] = lowpass_on_j
                W2raw = w2_full
        elif is_vector_feature:
            W2raw = w2_learnable.view(W1.shape[0], num_f_rows, num_f_rows)
        else:
            num_channels = int(W1.shape[1])
            W2raw = w2_learnable.view(
                num_nodes,
                num_channels,
                num_f_rows,
                num_f_rows,
            )
    else:
        # Fixed scales: batch all first-order channels through the wavelet
        # transform again, producing (W_prev, W_next) pairs for every channel.
        scatter_kwargs = {
            'diffusion_scales': diffusion_scales,
            'include_lowpass': include_lowpass,
            'check_nonfinite': check_nonfinite,
        }
        if is_vector_feature:
            x_second = W1.squeeze(1)
            W2raw = diffusion_wavelet_transform(
                x=x_second,
                P=diffusion_op,
                **scatter_kwargs,
            )
            W2raw = W2raw.view(x_second.shape[0], num_wavelets, -1)
        else:
            num_channels = int(W1.shape[1])
            x_second = W1.reshape(num_nodes, num_channels * num_wavelets)
            W2raw = diffusion_wavelet_transform(
                x=x_second,
                P=diffusion_op,
                **scatter_kwargs,
            )
            W2raw = W2raw.view(num_nodes, num_channels, num_wavelets, -1)

    return _subset_second_order_wavelets(
        W2raw,
        feature_type=feature_type,
    )


def multiorder_scatter(
    x: Tensor,
    diffusion_op: Union[Tensor, SparseTensor],
    *,
    include_lowpass: bool,
    scattering_orders: Tuple[int, ...],
    diffusion_scales: Optional[Tensor] = None,
    F: Optional[Tensor] = None,
    is_vector_feature: bool = False,
    activation: Optional[Union[Module, Callable[[Tensor], Tensor]]] = None,
    check_nonfinite: bool = False,
) -> Tensor:
    """Build scattering coefficients from zeroth, first, and second order.

    Pass ``diffusion_scales`` for fixed dyadic/custom scales, or ``F`` for
    LEGS learnable scales (exactly one must be set). First-order lowpass is
    concatenated before activation; second order uses the activated ``W1``.
    """
    if (F is None) == (diffusion_scales is None):
        raise ValueError(
            'Exactly one of diffusion_scales or F must be provided.', )

    if x.dim() == 1:
        x = x.unsqueeze(1)

    flat, num_nodes, feature_dim, feature_type = _prepare_scatter_inputs(
        x,
        is_vector_feature,
    )

    need_first_order = 1 in scattering_orders or 2 in scattering_orders
    coeffs: List[Tensor] = []

    if 0 in scattering_orders:
        coeffs.append(flat.unsqueeze(-1))

    W1: Optional[Tensor] = None
    if need_first_order:
        if F is not None:
            W1 = legs_wavelet_transform(
                x=flat,
                P=diffusion_op,
                F=F,
                include_lowpass=include_lowpass,
                check_nonfinite=check_nonfinite,
            )
        else:
            W1 = diffusion_wavelet_transform(
                x=flat,
                P=diffusion_op,
                diffusion_scales=diffusion_scales,
                include_lowpass=include_lowpass,
                check_nonfinite=check_nonfinite,
            )
        W1 = _apply_activation(W1, activation)
        if 1 in scattering_orders:
            coeffs.append(W1)

    if 2 in scattering_orders and W1 is not None:
        if int(W1.shape[-1]) > 1:
            W2 = _compute_second_order_scattering(
                W1,
                diffusion_op,
                diffusion_scales=diffusion_scales,
                F=F,
                include_lowpass=include_lowpass,
                is_vector_feature=is_vector_feature,
                num_nodes=num_nodes,
                feature_type=feature_type,
                check_nonfinite=check_nonfinite,
            )
            W2 = _apply_activation(W2, activation)
            coeffs.append(W2)

    if not coeffs:
        raise ValueError('scattering_orders produced no output coefficients.')

    W_tot = torch.cat(coeffs, dim=-1)
    if is_vector_feature:
        return W_tot.view(num_nodes, feature_dim, -1)
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
    r"""Implements the geometric scattering transform
    introduced in the papers `"Geometric Scattering for Graph Data
    Analysis" <https://arxiv.org/abs/1810.03068>`_ and `"Diffusion
    Scattering Transforms on Graphs"
    <https://arxiv.org/abs/1806.08829>`_. It
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
    A low-pass filter :math:`\mathbf{A}_J = \mathbf{P}^{t_J} \mathbf{x}`
    is also included by default, such that the full filter bank is
    :math:`\{\mathbf{W}_j\}_{j=0}^{J} \cup \{\mathbf{A}_J\}`. The low-pass
    filter can be excluded by setting :obj:`include_lowpass`
    to :obj:`False`.

    By default, the :math:`t_j` are set to be dyadic integers
    :math:`0, 1, 2, \ldots, 16`. However, one may also specify a
    custom set of scales using the :obj:`diffusion_scales`
    argument (pass a tuple of integers, or a tuple of tuples,
    for different diffusion scales for each feature channel).
    One option for generating custom scales is `InfoGain Wavelets
    <https://arxiv.org/abs/2504.08802>`_.

    Alternatively, set :obj:`diffusion_scales='legs'` to use a learnable
    LEGS selector matrix :math:`F \in \mathbb{R}^{(J+1) \times (2^J + 1)}`
    initialized to dyadic scales matching the default filter bank. The
    diffusion depth :math:`J` is set via :obj:`legs_kwargs['legs_J']`
    (default: :obj:`4`, giving ``5`` learnable wavelet rows and ``T = 16``).
    Note that :math:`J` sets the max diffusion scale to :math:`2^J`;
    given its dyadic initialization, :math:`J=4` or :math:`J=5` is
    recommended.

    This layer returns concatenated zeroeth and first-order scattering
    coefficients by default. The zeroeth-order scattering coefficient
    is the unfiltered input feature vector, and can be excluded by
    excluding :obj:`0` from :obj:`scattering_orders`. Second-order
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
    of output channels by the number of filters :math:`w`; including
    second-order scattering coefficients increases the number of output
    channels by :math:`w \cdot (w - 1) / 2`.

    For graph-level tasks, scattering coefficients can be pooled across
    nodes within each graph and feature channel, changing the output shape
    from :math:`(|\mathcal{V}|, F_{\mathrm{in}}, F_{\mathrm{out}})` to
    :math:`(B, F_{\mathrm{in}}, F_{\mathrm{out}} \cdot |\mathrm{pool}|)`.
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
            transform only. (default: :obj:`(0, 1, 2)`)
        diffusion_scales (Tuple[int, ...] or str, optional): Monotonically
            increasing diffusion powers, i.e., :math:`t_j` in each
            :math:`\mathbf{P}^{t_j}`. Wavelet filters are consecutive
            differences between adjacent :math:`t_{j-1}` and :math:`t_j`.
            Alternatively, pass :obj:`'legs'` or :obj:`'LEGS'` to use a
            learnable LEGS-style selector matrix :math:`F` initialized to
            dyadic scales. (default: :obj:`(0, 1, 2, 4, 8, 16)`)
        legs_kwargs (Dict[str, Any], optional): Keyword arguments for LEGS
            mode. Currently supports :obj:`'legs_J'`, the number of learnable
            wavelet filters (default: :obj:`4`, yielding ``J + 1`` rows in
            :obj:`F` and maximum diffusion power ``2^J``).
        include_lowpass (bool, optional): If set to :obj:`True`, append the
            low-pass filter :math:`\mathbf{P}^{t_J} \mathbf{x}` before the
            optional activation at first order.
            (default: :obj:`True`)
        activation (torch.nn.Module or Callable, optional): Activation
            applied to first- and second-order scattering coefficients, e.g.
            :obj:`torch.abs` for the modulus. (default: :obj:`None`)
        normalization (str, optional): Normalization scheme for the adjacency
            matrix before building the lazy random walk operator
            :math:`\mathbf{P}`. Ignored when :obj:`diffusion_op` is provided.
            Options: :obj:`"row"`, :obj:`"column"`, :obj:`"symmetric"`.
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
          :math:`(|\mathcal{V}|, F_{in}, F_{\mathrm{out}})`
        - **output (graph-level,** :obj:`pool` **set):**
          pooled scattering coefficients
          :math:`(B, F_{in}, F_{\mathrm{out}} \cdot |\texttt{pool}|)`
    """
    def __init__(
        self,
        in_channels: int,
        scattering_orders: Optional[Tuple[int, ...]] = (0, 1, 2),
        diffusion_scales: Optional[DiffusionScalesArg] = (
            0,
            1,
            2,
            4,
            8,
            16,
        ),
        legs_kwargs: Optional[Dict[str, Any]] = None,
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

        use_legs = (diffusion_scales is not None
                    and _is_legs_diffusion_scales(diffusion_scales))

        if use_legs:
            if legs_kwargs is not None and not isinstance(legs_kwargs, dict):
                raise ValueError('legs_kwargs must be a dict or None.')
            self.legs_kwargs = _validate_legs_kwargs(
                legs_kwargs,
                scattering_orders,
            )
            legs_J = self.legs_kwargs['legs_J']
            self.use_legs = True
            self.diffusion_scales = 'legs'
            self.max_diffusion_power = 2**legs_J
            self.register_parameter(
                'F',
                Parameter(_initialize_legs_parameters(legs_J)),
            )
        else:
            if legs_kwargs is not None:
                warnings.warn(
                    'legs_kwargs was provided but diffusion_scales is not '
                    "'legs'; ignoring legs_kwargs.",
                    stacklevel=2,
                )
            self.use_legs = False
            self.legs_kwargs = None
            self.max_diffusion_power = None
            self.F = None

            if diffusion_scales is None:
                diffusion_scales = DEFAULT_DIFFUSION_SCALES
            elif isinstance(diffusion_scales, str):
                raise ValueError(
                    "diffusion_scales must be a tuple of integers or 'legs' "
                    f"(got '{diffusion_scales}').", )
            else:
                diffusion_scales = tuple(diffusion_scales)
            if len(diffusion_scales) < 2:
                raise ValueError(
                    'diffusion_scales must contain at least two entries.', )
            if any(diffusion_scales[i] >= diffusion_scales[i + 1]
                   for i in range(len(diffusion_scales) - 1)):
                raise ValueError(
                    'diffusion_scales must be strictly increasing.', )
            self.diffusion_scales = diffusion_scales

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
        self.include_lowpass = include_lowpass
        self.activation = activation
        self.normalization = normalization
        self.is_vector_feature = is_vector_feature
        self.pool = pool
        self.check_nonfinite = check_nonfinite

    @property
    def num_wavelet_filters(self) -> int:
        if self.use_legs:
            return (self.legs_kwargs['legs_J'] + 1) + int(self.include_lowpass)
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
        r"""Resets learnable parameters.

        In LEGS mode, reinitializes :obj:`F` to dyadic scales via
        :func:`_initialize_legs_parameters`.
        """
        if self.use_legs:
            with torch.no_grad():
                self.F.copy_(
                    _initialize_legs_parameters(self.legs_kwargs['legs_J']), )

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
            :math:`(|\mathcal{V}|, F_{\mathrm{in}}, F_{\mathrm{out}})`,
            or :math:`(B, F_{\mathrm{in}}, F_{\mathrm{out}}
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

        scatter_kwargs = {
            'include_lowpass': self.include_lowpass,
            'scattering_orders': self.scattering_orders,
            'is_vector_feature': self.is_vector_feature,
            'activation': self.activation,
            'check_nonfinite': self.check_nonfinite,
        }
        if self.use_legs:
            coeffs = multiorder_scatter(
                x,
                op,
                F=self.F,
                **scatter_kwargs,
            )
        else:
            coeffs = multiorder_scatter(
                x,
                op,
                diffusion_scales=torch.as_tensor(
                    self.diffusion_scales,
                    dtype=torch.long,
                    device=x.device,
                ),
                **scatter_kwargs,
            )

        if self.pool is None:
            return coeffs

        return _pool_scattering_coefficients(coeffs, batch, self.pool)

    def __repr__(self) -> str:
        if self.use_legs:
            return (f'{self.__class__.__name__}('
                    f'in_channels={self.in_channels}, '
                    f'scattering_orders={self.scattering_orders}, '
                    f"diffusion_scales='legs', "
                    f'legs_kwargs={self.legs_kwargs}, '
                    f'num_scattering_filters={self.num_scattering_filters}, '
                    f'normalization={self.normalization}, '
                    f'pool={self.pool})')
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'scattering_orders={self.scattering_orders}, '
                f'num_scattering_filters={self.num_scattering_filters}, '
                f'normalization={self.normalization}, '
                f'pool={self.pool})')
