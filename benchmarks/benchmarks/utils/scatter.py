import torch

from torch_geometric.utils import scatter
from benchmarks import Benchmark, devices
from benchmarks.helpers import benchmark_torch_function_in_milliseconds

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


# TODO: Isn't this the same as own_scatter if reduce != 'any'?
def optimized_scatter(x, index, dim_size, reduce):
    return scatter(x, index, dim=0, dim_size=dim_size, reduce=reduce)


def pytorch_index_add(x, index, dim_size, reduce):
    out = x.new_zeros(dim_size, x.size(-1))
    out.index_add_(0, index, x)
    return out


num_nodes_edges = [(4_000, 4_000 * 50)]
# TODO: Extend to more sizes
# num_nodes_edges = [(4_000, 4_000 * 50), (16_000, 16_000 * 50), (64_000, 64_000 * 50)]


class Sum(Benchmark):
    param_names = ["f", "num_nodes, num_edges", "device"]
    params = [
        [pytorch_scatter, own_scatter, optimized_scatter, pytorch_index_add],
        num_nodes_edges,
        devices,
    ]
    unit = "ms"

    def setup(self, *params):
        f, (num_nodes, num_edges), device = params

        self.inputs = {
            "x": torch.randn(num_edges, 64, device=device, requires_grad=True),
            "index": torch.randint(num_nodes, (num_edges,), device=device),
            "dim_size": num_nodes,
            "reduce": "sum",
        }

    def track_fwd(self, *params):
        f, *_ = params
        return benchmark_torch_function_in_milliseconds(
            f,
            **self.inputs,
        )

    def track_bwd(self, *params):
        f, *_ = params
        out = f(**self.inputs)
        return benchmark_torch_function_in_milliseconds(
            out.backward,
            torch.ones_like(out, requires_grad=True),
            retain_graph=True,  # TODO: check if this is necessary
        )


class Mean(Benchmark):
    param_names = ["f", "num_nodes, num_edges", "device"]
    params = [
        [pytorch_scatter, own_scatter, optimized_scatter],
        num_nodes_edges,
        devices,
    ]
    unit = "ms"

    def setup(self, *params):
        f, (num_nodes, num_edges), device = params
        self.inputs = {
            "x": torch.randn(num_edges, 64, device=device, requires_grad=True),
            "index": torch.randint(num_nodes, (num_edges,), device=device),
            "dim_size": num_nodes,
            "reduce": "mean",
        }

    def track_fwd(self, *params):
        f, (num_nodes, num_edges), device = params
        return benchmark_torch_function_in_milliseconds(
            f,
            **self.inputs,
        )

    def track_bwd(self, *params):
        f, (num_nodes, num_edges), device = params
        out = f(**self.inputs)
        return benchmark_torch_function_in_milliseconds(
            out.backward,
            torch.ones_like(out, requires_grad=True),
            retain_graph=True,  # TODO: check if this is necessary
        )



class Min(Benchmark):
    param_names = ["f", "num_nodes, num_edges", "device"]
    params = [
        [pytorch_scatter, own_scatter, optimized_scatter],
        num_nodes_edges,
        devices,
    ]
    unit = "ms"

    def setup(self, *params):
        f, (num_nodes, num_edges), device = params
        self.inputs = {
            "x": torch.randn(num_edges, 64, device=device, requires_grad=True),
            "index": torch.randint(num_nodes, (num_edges,), device=device),
            "dim_size": num_nodes,
            "reduce": "min",
        }

    def track_fwd(self, *params):
        f, (num_nodes, num_edges), device = params
        return benchmark_torch_function_in_milliseconds(
            f,
            **self.inputs,
        )

    def track_bwd(self, *params):
        f, (num_nodes, num_edges), device = params
        out = f(**self.inputs)
        return benchmark_torch_function_in_milliseconds(
            out.backward,
            torch.ones_like(out, requires_grad=True),
            retain_graph=True,  # TODO: check if this is necessary
        )


class Max(Benchmark):
    param_names = ["f", "num_nodes, num_edges", "device"]
    params = [
        [pytorch_scatter, own_scatter, optimized_scatter],
        num_nodes_edges,
        devices,
    ]
    unit = "ms"

    def setup(self, *params):
        f, (num_nodes, num_edges), device = params
        self.inputs = {
            "x": torch.randn(num_edges, 64, device=device, requires_grad=True),
            "index": torch.randint(num_nodes, (num_edges,), device=device),
            "dim_size": num_nodes,
            "reduce": "max",
        }

    def track_fwd(self, *params):
        f, (num_nodes, num_edges), device = params
        return benchmark_torch_function_in_milliseconds(
            f,
            **self.inputs,
        )

    def track_bwd(self, *params):
        f, (num_nodes, num_edges), device = params
        out = f(**self.inputs)
        return benchmark_torch_function_in_milliseconds(
            out.backward,
            torch.ones_like(out, requires_grad=True),
            retain_graph=True,  # TODO: check if this is necessary
        )



class Mul(Benchmark):
    param_names = ["f", "num_nodes, num_edges", "device"]
    params = [
        [pytorch_scatter, own_scatter, optimized_scatter],
        num_nodes_edges,
        devices,
    ]
    unit = "ms"

    def setup(self, *params):
        f, (num_nodes, num_edges), device = params
        self.inputs = {
            "x": torch.randn(num_edges, 64, device=device, requires_grad=True),
            "index": torch.randint(num_nodes, (num_edges,), device=device),
            "dim_size": num_nodes,
            "reduce": "mul",
        }

    def track_fwd(self, *params):
        f, (num_nodes, num_edges), device = params
        return benchmark_torch_function_in_milliseconds(
            f,
            **self.inputs,
        )

    def track_bwd(self, *params):
        f, (num_nodes, num_edges), device = params
        out = f(**self.inputs)
        return benchmark_torch_function_in_milliseconds(
            out.backward,
            torch.ones_like(out, requires_grad=True),
            retain_graph=True,  # TODO: check if this is necessary
        )
