import torch
from torch.utils.benchmark import Timer

import torch_geometric
from torch_geometric.utils import softmax

from benchmarks import devices, Benchmark
from benchmarks.helpers import benchmark_torch_function_in_milliseconds


def dense_softmax(src, index):
    src = src.view(src.size(0), -1, src.size(-1))
    return src.softmax(dim=-1)


class Softmax(Benchmark):
    param_names = ["f", "compile", "num_nodes, num_edges", "device"]
    params = [
        [softmax, dense_softmax],
        [False, True],
        [(10_000, 200_000)],
        devices,
    ]
    unit = "ms"

    def setup(self, *params):
        f, compile, (num_nodes, num_edges), device = params
        self.inputs = {
            "src": torch.randn(num_edges, 64, device=device),
            "index": torch.randint(num_nodes, (num_edges,), device=device),
        }

    def track_fwd(self, *params):
        f, compile, (num_nodes, num_edges), device = params
        return benchmark_torch_function_in_milliseconds(
            torch_geometric.compile(f) if compile else f,
            **self.inputs,
        )

    # TODO: Fix benchmarking compiled backward
    def track_bwd(self, *params):
        f, compile, (num_nodes, num_edges), device = params
        f = torch_geometric.compile(f) if compile else f
        out = f(**self.inputs)
        return benchmark_torch_function_in_milliseconds(
            out.backward,
            torch.ones_like(out, requires_grad=True),
            retain_graph=True,
        )
