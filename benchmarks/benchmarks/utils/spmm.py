import torch

from benchmarks import Benchmark, devices
from benchmarks.helpers import benchmark_torch_function_in_milliseconds
from torch_geometric.utils import spmm, to_torch_coo_tensor


class Spmm(Benchmark):
    param_names = ["layout", "reduce", "num_nodes, num_edges", "device"]
    params = [
        [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc],
        ["sum", "mean", "min", "max"],
        [(10_000, 200_000)],
        devices,
    ]
    unit = "ms"

    def setup(self, *params):
        layout, reduce, (num_nodes, num_edges), device = params

        if reduce in ["min", "max"] and device == "cuda":
            raise NotImplementedError("min/max reductions are not supported on cuda")

        x = torch.randn(num_nodes, 64, device=device, requires_grad=True)
        edge_index = torch.randint(num_nodes, (2, num_edges), device=device)
        adj = to_torch_coo_tensor(edge_index, size=num_nodes).to_sparse(layout=layout)
        self.inputs = {
            "src": adj,
            "other": x,
            "reduce": reduce,
        }

    def track_fwd(self, *params):
        return benchmark_torch_function_in_milliseconds(
            spmm,
            **self.inputs,
        )

    def track_bwd(self, *params):
        out = spmm(**self.inputs)
        return benchmark_torch_function_in_milliseconds(
            out.backward,
            torch.ones_like(out, requires_grad=True),
            retain_graph=True,  # TODO: check if this is necessary
        )
