import copy
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import WITH_TORCH_SPARSE, SparseTensor
from torch_geometric.utils import to_torch_coo_tensor, to_torch_csc_tensor

SPARSE_LAYOUTS: List[Union[str, torch.layout]] = [
    'torch_sparse', torch.sparse_csc, torch.sparse_coo
]


def assert_module(
    module: torch.nn.Module,
    x: Any,
    edge_index: Tensor,
    *,
    expected_size: Tuple[int, ...],
    test_edge_permutation: bool = True,
    test_node_permutation: bool = False,
    test_sparse_layouts: Optional[List[Union[str, torch.layout]]] = None,
    sparse_size: Optional[Tuple[int, int]] = None,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    equal_nan: bool = False,
    **kwargs: Any,
) -> Any:
    r"""Asserts that the output of a :obj:`module` is correct.

    Specifically, this method tests that:

    1. The module output has the correct shape.
    2. The module is invariant to the permutation of edges.
    3. The module is invariant to the permutation of nodes.
    4. The module is invariant to the layout of :obj:`edge_index`.

    Args:
        module (torch.nn.Module): The module to test.
        x (Any): The input features to the module.
        edge_index (torch.Tensor): The input edge indices.
        expected_size (Tuple[int, ...]): The expected output size.
        test_edge_permutation (bool, optional): If set to :obj:`False`, will
            not test the module for edge permutation invariance.
        test_node_permutation (bool, optional): If set to :obj:`False`, will
            not test the module for node permutation invariance.
        test_sparse_layouts (List[str or int], optional): The sparse layouts to
            test for module invariance. (default: :obj:`["torch_sparse",
            torch.sparse_csc, torch.sparse_coo]`)
        sparse_size (Tuple[int, int], optional): The size of the sparse
            adjacency matrix. If not given, will try to automatically infer it.
            (default: :obj:`None`)
        atol (float, optional): Absolute tolerance. (default: :obj:`1e-08`)
        rtol (float, optional): Relative tolerance. (default: :obj:`1e-05`)
        equal_nan (bool, optional): If set to :obj:`True`, then two :obj:`NaN`s
            will be considered equal. (default: :obj:`False`)
        **kwargs (optional): Additional arguments passed to
            :meth:`module.forward`.
    """
    if test_sparse_layouts is None:
        test_sparse_layouts = SPARSE_LAYOUTS

    if sparse_size is None:
        if 'size' in kwargs:
            sparse_size = kwargs['size']
        elif isinstance(x, Tensor):
            sparse_size = (x.size(0), x.size(0))
        elif (isinstance(x, (tuple, list)) and isinstance(x[0], Tensor)
              and isinstance(x[1], Tensor)):
            sparse_size = (x[0].size(0), x[1].size(0))

    if len(test_sparse_layouts) > 0 and sparse_size is None:
        raise ValueError(f"Got sparse layouts {test_sparse_layouts}, but no "
                         f"'sparse_size' were specified")

    expected = module(x, edge_index=edge_index, **kwargs)
    assert expected.size() == expected_size

    if test_edge_permutation:
        perm = torch.randperm(edge_index.size(1))
        perm_kwargs = copy.copy(kwargs)
        for key, value in kwargs.items():
            if isinstance(value, Tensor) and value.size(0) == perm.numel():
                perm_kwargs[key] = value[perm]
        out = module(x, edge_index[:, perm], **perm_kwargs)
        assert torch.allclose(out, expected, rtol, atol, equal_nan)

    if test_node_permutation:
        raise NotImplementedError

    for layout in (test_sparse_layouts or []):
        # TODO Add support for values.
        if layout == 'torch_sparse':
            if not WITH_TORCH_SPARSE:
                continue

            adj = SparseTensor.from_edge_index(
                edge_index,
                sparse_sizes=sparse_size,
            )
            adj_t = adj.t()

        elif layout == torch.sparse_csc:
            adj = to_torch_csc_tensor(edge_index, size=sparse_size)
            adj_t = adj.t()

        elif layout == torch.sparse_coo:
            warnings.filterwarnings('ignore', ".*to CSR format.*")
            adj = to_torch_coo_tensor(edge_index, size=sparse_size)
            adj_t = adj.t().coalesce()

        else:
            raise ValueError(f"Got invalid sparse layout '{layout}'")

        out = module(x, adj_t, **kwargs)
        assert torch.allclose(out, expected, rtol, atol, equal_nan)

    return expected
