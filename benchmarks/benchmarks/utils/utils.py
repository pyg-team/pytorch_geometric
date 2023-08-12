# import torch
# from torch.utils.benchmark import Timer

# import torch_geometric
# from torch_geometric.typing import SparseTensor
# from torch_geometric.utils import (
#     softmax,
#     to_torch_coo_tensor,
#     to_torch_csc_tensor,
#     to_torch_csr_tensor,
# )
# from torch_geometric.utils.map import map_index

# from benchmarks import DEFAULT_NUM_THREADS



# class Sparse:
#     param_names = ["f", "num_nodes, num_edges", "device"]
#     params = [
#         [
#             SparseTensor.from_edge_index,
#             to_torch_coo_tensor,
#             to_torch_csr_tensor,
#             to_torch_csc_tensor,
#         ],
#         [(10_000, 200_000)],
#         ["cuda"],  # TODO: Enable "cpu"
#     ]
#     unit = "us"

#     def setup(self, *params):
#         f, (num_nodes, num_edges), device = params

#         self.globals = {
#             "f": f,
#             "edge_index": torch.randint(num_nodes, (2, num_edges), device=device),
#             "size": num_nodes,
#         }

#     def track_fwd(self, *params):
#         f, *_ = params

#         t = Timer(
#             stmt="f(edge_index, None, (size, size))",
#             globals=self.globals,
#             num_threads=DEFAULT_NUM_THREADS,
#             label="sparse",
#             sub_label=f.__name__,
#             description=" ",
#         )
#         m = t.blocked_autorange(min_run_time=1)
#         return m.median * 1_000**2  # us


# class Spmm:
#     param_names = ["layout", "reduce", "num_nodes, num_edges", "device"]
#     params = [
#         [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc],
#         ["sum", "mean"],  # TODO: if not cuda, add ["min", "max"]
#         [(10_000, 200_000)],
#         ["cuda"],  # TODO: Enable "cpu"
#     ]
#     unit = "us"

#     def setup(self, *params):
#         layout, reduce, (num_nodes, num_edges), device = params
#         x = torch.randn(num_nodes, 64, device=device, requires_grad=True)
#         edge_index = torch.randint(num_nodes, (2, num_edges), device=device)
#         adj = to_torch_coo_tensor(edge_index, size=num_nodes).to_sparse(layout=layout)
#         self.globals = {
#             "adj": adj,
#             "x": x,
#             "reduce": reduce,
#         }

#     def track_fwd(self, *params):
#         layout, *_ = params
#         t = Timer(
#             stmt="spmm(adj, x, reduce)",
#             setup=f"from torch_geometric.utils import spmm",
#             globals=self.globals,
#             num_threads=DEFAULT_NUM_THREADS,
#             label="spmm",
#             sub_label=layout,
#             description=" ",
#         )
#         m = t.blocked_autorange(min_run_time=1)
#         return m.median * 1_000**2  # us

#     def track_bwd(self, *params):
#         layout, *_ = params
#         t = Timer(
#             stmt="out.backward(out_grad, retain_graph=True)",
#             setup=f"from torch_geometric.utils import spmm; from {__name__} import grads_like; out = spmm(adj, x, reduce); out_grad = grads_like(out)",
#             globals=self.globals,
#             num_threads=DEFAULT_NUM_THREADS,
#             label="spmm_bwd",
#             sub_label=layout,
#             description=" ",
#         )
#         m = t.blocked_autorange(min_run_time=1)
#         return m.median * 1_000**2  # us




# def dense_softmax(x, index):
#     x = x.view(x.size(0), -1, x.size(-1))
#     return x.softmax(dim=-1)


# class Softmax:
#     param_names = ["f", "compile", "num_nodes, num_edges", "device"]
#     params = [
#         [softmax, dense_softmax],
#         [False, True],
#         [(10_000, 200_000)],
#         ["cuda"],  # TODO: Enable "cpu"
#     ]
#     unit = "us"

#     def setup(self, *params):
#         f, compile, (num_nodes, num_edges), device = params
#         self.globals = {
#             "f": torch_geometric.compile(f) if compile else f,
#             "x": torch.randn(num_edges, 64, device=device),
#             "index": torch.randint(num_nodes, (num_edges,), device=device),
#         }

#     def track_fwd(self, *_):
#         t = Timer(
#             stmt="f(x, index)",
#             globals=self.globals.copy(),
#             num_threads=DEFAULT_NUM_THREADS,
#             label="softmax_fwd",
#             sub_label=" ",
#             description=" ",
#         )
#         m = t.blocked_autorange(min_run_time=1)
#         return m.median * 1_000**2  # us

#     def track_bwd(self, *_):
#         t = Timer(
#             stmt="out.backward(out_grad, retain_graph=True)",
#             setup=f"from {__name__} import grads_like; out = f(x, index); out_grad = grads_like(out)",
#             globals=self.globals,
#             num_threads=1,
#             label="softmax_bwd",
#             sub_label=" ",
#             description=" ",
#         )
#         m = t.blocked_autorange(min_run_time=1)
#         return m.median * 1_000**2  # us
