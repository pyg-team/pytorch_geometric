import torch
from torch.utils.benchmark import Timer

import torch_geometric
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    scatter,
    softmax,
    to_torch_coo_tensor,
    to_torch_csc_tensor,
    to_torch_csr_tensor,
)
from torch_geometric.utils.map import map_index

WITH_TORCH_SCATTER = True
try:
    import torch_scatter
except ImportError:
    WITH_TORCH_SCATTER = False


def pytorch_scatter(x, index, dim_size, reduce):
    if reduce == "min" or reduce == "max":
        reduce = f"a{reduce}"  # `amin` or `amax`
    elif reduce == "mul":
        reduce = "prod"
    out = x.new_zeros((dim_size, x.size(-1)))
    include_self = reduce in ["sum", "mean"]
    index = index.view(-1, 1).expand(-1, x.size(-1))
    out.scatter_reduce_(0, index, x, reduce, include_self=include_self)
    return out


def own_scatter(x, index, dim_size, reduce):
    return torch_scatter.scatter(x, index, dim=0, dim_size=dim_size, reduce=reduce)


def optimized_scatter(x, index, dim_size, reduce):
    return scatter(x, index, dim=0, dim_size=dim_size, reduce=reduce)


def pytorch_index_add(x, index, dim_size, reduce):
    out = x.new_zeros(dim_size, x.size(-1))
    out.index_add_(0, index, x)
    return out


def grads_like(x):
    return torch.ones_like(x, requires_grad=True)


class Scatter:
    param_names = ["f", "reduce", "num_nodes, num_edges", "device"]
    params = [
        [pytorch_scatter, own_scatter, optimized_scatter, pytorch_index_add],
        ["sum", "mean", "min", "max", "mul"],
        [(4_000, 4_000 * 50), (16_000, 16_000 * 50), (64_000, 64_000 * 50)],
        ["cuda"],  # TODO: Enable "cpu"
    ]
    unit = "us"

    def setup(self, *params):
        f, reduce, (num_nodes, num_edges), device = params

        if f is own_scatter and not WITH_TORCH_SCATTER:
            raise NotImplementedError("torch-scatter not found", WITH_TORCH_SCATTER)

        if f is pytorch_index_add and reduce != "sum":
            raise NotImplementedError

        self.globals = {
            "x": torch.randn(num_edges, 64, device=device, requires_grad=True),
            "index": torch.randint(num_nodes, (num_edges,), device=device),
            "dim_size": num_nodes,
            "reduce": reduce,
        }

    def track_fwd(self, *params):
        f, *_ = params
        t = Timer(
            stmt=f"{f.__name__}(x, index, dim_size, reduce)",
            setup=f"from {__name__} import {f.__name__}",
            globals=self.globals,
            num_threads=4,
            label="scatter",
            sub_label=f.__name__,
            description=self.globals["reduce"],
        )
        m = t.blocked_autorange(min_run_time=1)
        return m.median * 1_000**2  # us

    def track_bwd(self, *params):
        f, *_ = params
        t = Timer(
            stmt="out.backward(out_grad, retain_graph=True)",
            setup=(
                f"from {__name__} import {f.__name__}, grads_like\n"
                f"out = {f.__name__}(x, index, dim_size, reduce)\n"
                f"out_grad = grads_like(out)"
            ),
            globals=self.globals,
            num_threads=4,
            label="scatter",
            sub_label=f.__name__,
            description=self.globals["reduce"],
        )
        m = t.blocked_autorange(min_run_time=1)
        return m.median * 1_000**2  # us


class Sparse:
    param_names = ["f", "num_nodes, num_edges", "device"]
    params = [
        [
            SparseTensor.from_edge_index,
            to_torch_coo_tensor,
            to_torch_csr_tensor,
            to_torch_csc_tensor,
        ],
        [(10_000, 200_000)],
        ["cuda"],  # TODO: Enable "cpu"
    ]
    unit = "us"

    def setup(self, *params):
        f, (num_nodes, num_edges), device = params

        self.globals = {
            "f": f,
            "edge_index": torch.randint(num_nodes, (2, num_edges), device=device),
            "size": num_nodes,
        }

    def track_fwd(self, *params):
        f, *_ = params
        t = Timer(
            stmt="f(edge_index, None, (size, size))",
            globals=self.globals,
            num_threads=4,
            label="sparse",
            sub_label=f.__name__,
            description=" ",
        )
        m = t.blocked_autorange(min_run_time=1)
        return m.median * 1_000**2  # us


class Spmm:
    param_names = ["layout", "reduce", "num_nodes, num_edges", "device"]
    params = [
        [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc],
        ["sum", "mean"],  # TODO: if not cuda, add ["min", "max"]
        [(10_000, 200_000)],
        ["cuda"],  # TODO: Enable "cpu"
    ]
    unit = "us"

    def setup(self, *params):
        layout, reduce, (num_nodes, num_edges), device = params
        x = torch.randn(num_nodes, 64, device=device, requires_grad=True)
        edge_index = torch.randint(num_nodes, (2, num_edges), device=device)
        adj = to_torch_coo_tensor(edge_index, size=num_nodes).to_sparse(layout=layout)
        self.globals = {
            "adj": adj,
            "x": x,
            "reduce": reduce,
        }

    def track_fwd(self, *params):
        layout, *_ = params
        t = Timer(
            stmt="spmm(adj, x, reduce)",
            setup=f"from torch_geometric.utils import spmm",
            globals=self.globals,
            num_threads=4,
            label="spmm",
            sub_label=layout,
            description=" ",
        )
        m = t.blocked_autorange(min_run_time=1)
        return m.median * 1_000**2  # us

    def track_bwd(self, *params):
        layout, *_ = params
        t = Timer(
            stmt="out.backward(out_grad, retain_graph=True)",
            setup=f"from torch_geometric.utils import spmm; from {__name__} import grads_like; out = spmm(adj, x, reduce); out_grad = grads_like(out)",
            globals=self.globals,
            num_threads=4,
            label="spmm_bwd",
            sub_label=layout,
            description=" ",
        )
        m = t.blocked_autorange(min_run_time=1)
        return m.median * 1_000**2  # us


def trivial_map(src, index, max_index, inclusive):
    if max_index is None:
        max_index = max(src.max(), index.max())

    if inclusive:
        assoc = src.new_empty(max_index + 1)
    else:
        assoc = src.new_full((max_index + 1,), -1)
    assoc[index] = torch.arange(index.numel(), device=index.device)
    out = assoc[src]

    if inclusive:
        return out, None
    else:
        mask = out != -1
        return out[mask], mask


class Map:
    param_names = ["f", "device"]
    params = [
        [trivial_map, map_index],
        ["cpu"],
    ]
    unit = "us"
    # TODO: Don't skip "cuda" if cudf is installed
    # TODO: Skip "cpu" if pandas not installed

    def setup(self, *params):
        f, device = params
        src = torch.randint(0, 100_000_000, (100_000,), device=device)
        index = src.unique()
        self.globals = {
            "f": f,
            "src": src,
            "index": index,
        }

    def track_inclusive(self, *_):
        t = Timer(
            stmt="f(src, index, None, True)",
            globals=self.globals,
            num_threads=4,
            label="map",
            sub_label=" ",
            description=" ",
        )
        m = t.blocked_autorange(min_run_time=1)
        return m.median * 1_000**2  # us

    def track_exclusive(self, *_):
        t = Timer(
            stmt="f(src, index[:50_000], None, False)",
            globals=self.globals,
            num_threads=4,
            label="map",
            sub_label=" ",
            description=" ",
        )
        m = t.blocked_autorange(min_run_time=1)
        return m.median * 1_000**2  # us


def dense_softmax(x, index):
    x = x.view(x.size(0), -1, x.size(-1))
    return x.softmax(dim=-1)


class Softmax:
    param_names = ["f", "compile", "num_nodes, num_edges", "device"]
    params = [
        [softmax, dense_softmax],
        [False, True],
        [(10_000, 200_000)],
        ["cuda"],  # TODO: Enable "cpu"
    ]
    unit = "us"

    def setup(self, *params):
        f, compile, (num_nodes, num_edges), device = params
        self.globals = {
            "f": torch_geometric.compile(f) if compile else f,
            "x": torch.randn(num_edges, 64, device=device),
            "index": torch.randint(num_nodes, (num_edges,), device=device),
        }

    def track_fwd(self, *_):
        t = Timer(
            stmt="f(x, index)",
            globals=self.globals.copy(),
            num_threads=4,
            label="softmax_fwd",
            sub_label=" ",
            description=" ",
        )
        m = t.blocked_autorange(min_run_time=1)
        return m.median * 1_000**2  # us

    def track_bwd(self, *_):
        t = Timer(
            stmt="out.backward(out_grad, retain_graph=True)",
            setup=f"from {__name__} import grads_like; out = f(x, index); out_grad = grads_like(out)",
            globals=self.globals,
            num_threads=1,
            label="softmax_bwd",
            sub_label=" ",
            description=" ",
        )
        m = t.blocked_autorange(min_run_time=1)
        return m.median * 1_000**2  # us
