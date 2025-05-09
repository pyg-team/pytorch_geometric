# TODO: Benchmark across different configurations
# TODO: Make NUM_HOPS configurable
# TODO: Benchmark peak device memory usage
# TODO: Add torch.compile support to EdgeIndex
# TODO: Add torch.compile support to custom kernels
import os
from itertools import product

import torch
import torch._inductor.codecache
from torch.utils.benchmark import Compare, Measurement, Timer

from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GAT, GCN, GIN, EdgeCNN, GraphSAGE

torch.set_float32_matmul_precision("high")
NUM_HOPS = 6


def gen_batch() -> Data:
    num_nodes = 10_000
    x = torch.randn(num_nodes, 64)
    edge_index = torch.stack([
        torch.arange(num_nodes).repeat_interleave(num_nodes),
        torch.arange(num_nodes).repeat(num_nodes),
    ], dim=0)
    data = Data(x=x, edge_index=edge_index)
    loader = NeighborLoader(
        data,
        num_neighbors=[36] * NUM_HOPS,
        batch_size=1,
        shuffle=True,
    )
    return next(iter(loader))


def run_benchmark(
    *,
    model: torch.nn.Module,
    batch: Data,
    backward: bool,
    trim: bool,
    compile: bool,
    compile_dynamic: bool | None,
    subclass: bool,
    profile: bool = False,
) -> Measurement:
    edge_index = EdgeIndex(batch.edge_index) if subclass else batch.edge_index
    edge_index, indices = edge_index.sort_by('col')
    print(edge_index.is_sorted_by_row)
    if trim:
        args = (batch.x, edge_index)
        kwargs = dict(
            num_sampled_nodes_per_hop=batch.num_sampled_nodes,
            num_sampled_edges_per_hop=batch.num_sampled_edges,
        )
    else:
        args = (batch.x, edge_index)
        kwargs = dict()

    grad = None
    if backward:
        grad = torch.randn_like(model(*args, **kwargs))

    sublabel = f'{model.__class__.__name__}_fwd{"_bwd" if backward else ""}'
    description = (f'{"compile" if compile else "eager"}'
                   f'{"_dynamic" if compile_dynamic else ""}'
                   f'{"_trim" if trim else ""}'
                   f'{"_subclass" if subclass else ""}')

    if compile:
        torch.compiler.reset()
        model = torch.compile(model, dynamic=compile_dynamic)

    if profile:

        with torch.profiler.profile(
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(5):
                if backward:
                    model(*args, **kwargs).backward(grad)
                else:
                    model(*args, **kwargs)
                prof.step()

            if subclass:
                # reset for starting from scratch wihotu caching
                args = (batch.x, EdgeIndex(batch.edge_index))

            for _ in range(5):
                if backward:
                    model(*args, **kwargs).backward(grad)
                else:
                    model(*args, **kwargs)
                prof.step()

        filename = f"profiles/{sublabel}_{description}.json"
        os.makedirs("profiles", exist_ok=True)
        prof.export_chrome_trace(filename)
        print(f"Profile saved to {filename}")
        return

    return Timer(
        "model(*args, **kwargs).backward(grad)"
        if backward else "model(*args, **kwargs)",
        globals=dict(model=model, args=args, kwargs=kwargs, grad=grad),
        label='Benchmark',
        sub_label=sublabel,
        description=description,
    ).blocked_autorange(min_run_time=3)


def main():
    device = torch.device("cuda")
    global_results = []
    models = [
        lambda: GCN(64, 64, num_layers=NUM_HOPS, add_self_loops=False),
        # lambda: GAT(64, 64, num_layers=NUM_HOPS, add_self_loops=False),
        # lambda: GIN(64, 64, num_layers=NUM_HOPS),
        # lambda: EdgeCNN(64, 64, num_layers=NUM_HOPS),
        # lambda: GraphSAGE(64, 64, num_layers=NUM_HOPS),
    ]
    batch = gen_batch().to(device)
    for model_cls in models:
        results = []
        model = model_cls().to(device)
        # for backward, trim, compile, compile_dynamic, subclass in product(
        #     [False, True],  # backward
        #     [False, True],  # trim
        #     [False, True],  # compile
        #     [None, True],  # compile_dynamic
        #     [False, True],  # subclass
        # ):
        for backward, trim, compile, compile_dynamic, subclass in product(
            [False],  # backward
            [False],  # trim
            [False],  # compile
            [None],  # compile_dynamic
            [True],  # subclass
        ):
            if subclass and compile:
                # EdgeIndex subclass is not supported with torch.compile
                continue
            if not compile and compile_dynamic:
                # No need to benchmark eager with `torch.compile(dynamic=True)`
                continue
            m = run_benchmark(
                model=model,
                batch=batch,
                backward=backward,
                trim=trim,
                compile=compile,
                compile_dynamic=compile_dynamic,
                subclass=subclass,
            )
            results.append(m)

        if results[0] is not None:
            # Dump intermediate results
            compare = Compare(sorted(results, key=lambda x: x.title))
            compare.highlight_warnings()
            compare.colorize(rowwise=True)
            compare.print()
            global_results.extend(results)

    if global_results:
        compare = Compare(sorted(global_results, key=lambda x: x.title))
        compare.highlight_warnings()
        compare.colorize(rowwise=True)
        compare.print()


if __name__ == "__main__":
    for i in range(1):
        main()
