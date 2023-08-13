import torch

from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    to_torch_coo_tensor,
    to_torch_csc_tensor,
    to_torch_csr_tensor,
)

from benchmarks import devices
from benchmarks.helpers import benchmark_torch_function_in_milliseconds


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
        devices,
    ]
    unit = "ms"

    def track_fwd(self, *params):
        f, (num_nodes, num_edges), device = params
        return benchmark_torch_function_in_milliseconds(
            f,
            torch.randint(num_nodes, (2, num_edges), device=device),
            None,
            (num_nodes, num_nodes),
        )
