import torch
from torch_geometric.utils import scatter

from benchmarks.utils.benchmark import benchmark

WITH_TORCH_SCATTER = True
try:
    import torch_scatter
except ImportError:
    WITH_TORCH_SCATTER = False


# Insights on GPU:
# ================
# * "sum": Prefer `scatter_add_` implementation
# * "mean": Prefer manual implementation via `scatter_add_` + `count`
# * "min"/"max":
#   * Prefer `scatter_reduce_` implementation without gradients
#   * Prefer `torch_sparse` implementation with gradients
# * "mul": Prefer `torch_sparse` implementation
#
# Insights on CPU:
# ================
# * "sum": Prefer `scatter_add_` implementation
# * "mean": Prefer manual implementation via `scatter_add_` + `count`
# * "min"/"max": Prefer `scatter_reduce_` implementation
# * "mul" (probably not worth branching for this):
#   * Prefer `scatter_reduce_` implementation without gradients
#   * Prefer `torch_sparse` implementation with gradients


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


class Scatter:
    def setup(self, n):
        if not WITH_TORCH_SCATTER:
            raise NotImplementedError

    def setup_cache(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_nodes, num_edges = 1_000, 50_000
        x = torch.randn(num_edges, 64, device=device)
        index = torch.randint(num_nodes, (num_edges,), device=device)
        ts = benchmark(
            funcs=[pytorch_scatter, own_scatter, optimized_scatter],
            func_names=["PyTorch", "torch_scatter", "Optimized"],
            args=(x, index, num_nodes, "sum"),
            num_steps=100 if device == "cpu" else 1000,
            num_warmups=50 if device == "cpu" else 500,
        )
        return ts

    def track_pytorch_scatter(self, ts):
        return ts[0][1]

    def track_own_scatter(self, ts):
        return ts[1][1]

    def track_optimized_scatter(self, ts):
        return ts[2][1]


if __name__ == "__main__":
    bench = Scatter()
    print(bench.setup_cache())
