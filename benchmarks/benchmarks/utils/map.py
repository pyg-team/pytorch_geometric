import torch

from torch_geometric.utils.map import map_index

from benchmarks.helpers import benchmark_torch_function_in_milliseconds


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
        ["cpu", "cuda"],
    ]
    unit = "ms"

    def setup(self, *params):
        f, device = params

        # TODO: Install cudf to benchmark map_index on cuda device
        if "cuda" == device:
            try:
                import cudf
            except ModuleNotFoundError as e:
                raise NotImplementedError("cudf not found") from e

        src = torch.randint(0, 100_000_000, (100_000,), device=device)
        index = src.unique()
        self.inputs = {
            "src": src,
            "index": index,
        }

    def track_inclusive(self, *params):
        f, device = params
        return benchmark_torch_function_in_milliseconds(
            f,
            self.inputs["src"],
            self.inputs["index"],
            None,
            True,
        )

    def track_exclusive(self, *params):
        f, device = params
        return benchmark_torch_function_in_milliseconds(
            f,
            self.inputs["src"],
            self.inputs["index"][:50_000],
            None,
            False,
        )
