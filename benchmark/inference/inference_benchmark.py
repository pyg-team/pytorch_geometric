import argparse
import warnings
from collections import defaultdict
from contextlib import nullcontext

import torch

from benchmark.utils import (
    emit_itt,
    get_dataset_with_transformation,
    get_model,
    get_split_masks,
    save_benchmark_data,
    test,
    write_to_csv,
)
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv
from torch_geometric.profile import (
    rename_profile_file,
    timeit,
    torch_profile,
    xpu_profile,
)

supported_sets = {
    'ogbn-mag': ['rgat', 'rgcn'],
    'ogbn-products': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
    'Reddit': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
}


@torch.no_grad()
def full_batch_inference(model, data):
    model.eval()
    if hasattr(data, 'adj_t'):
        edge_index = data.adj_t
    else:
        edge_index = data.edge_index
    return model(data.x, edge_index)


def run(args: argparse.ArgumentParser):
    csv_data = defaultdict(list)

    if args.write_csv == 'prof' and not args.profile:
        warnings.warn("Cannot write profile data to CSV because profiling is "
                      "disabled")

    if args.device == 'xpu':
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            raise RuntimeError('XPU device requires IPEX to be installed')

    if ((args.device == 'cuda' and not torch.cuda.is_available())
            or (args.device == 'xpu' and not torch.xpu.is_available())):
        raise RuntimeError(f'{args.device.upper()} is not available')

    if args.device == 'cuda' and args.full_batch:
        raise RuntimeError('CUDA device is not suitable for full batch mode')

    device = torch.device(args.device)

    print('BENCHMARK STARTS')
    print(f'Running on {args.device.upper()}')
    for dataset_name in args.datasets:
        assert dataset_name in supported_sets.keys(
        ), f"Dataset {dataset_name} isn't supported."
        print(f'Dataset: {dataset_name}')
        load_time = timeit() if args.measure_load_time else nullcontext()
        with load_time:
            result = get_dataset_with_transformation(dataset_name, args.root,
                                                     args.use_sparse_tensor,
                                                     args.bf16)
            dataset, num_classes, transformation = result
        data = dataset.to(device)
        hetero = True if dataset_name == 'ogbn-mag' else False
        mask = ('paper', None) if dataset_name == 'ogbn-mag' else None
        _, _, test_mask = get_split_masks(data, dataset_name)
        degree = None

        if hetero and args.cached_loader:
            args.cached_loader = False
            print('Disabling CachedLoader, not supported in Hetero models')
        if args.num_layers != [1] and not hetero and args.num_steps != -1:
            raise ValueError("Layer-wise inference requires `steps=-1`")

        if args.device == 'cuda':
            amp = torch.amp.autocast('cuda', enabled=False)
        elif args.device == 'xpu':
            amp = torch.xpu.amp.autocast(enabled=False)
        else:
            amp = torch.cpu.amp.autocast(enabled=args.bf16)

        if args.device == 'xpu' and args.warmup < 1:
            print('XPU device requires warmup - setting warmup=1')
            args.warmup = 1

        inputs_channels = data[
            'paper'].num_features if dataset_name == 'ogbn-mag' \
            else dataset.num_features

        for model_name in args.models:
            if model_name not in supported_sets[dataset_name]:
                print(f'Configuration of {dataset_name} + {model_name} '
                      f'not supported. Skipping.')
                continue
            with_loader = not args.full_batch or (model_name == 'pna'
                                                  and degree is None)
            print(f'Evaluation bench for {model_name}:')

            for batch_size in args.eval_batch_sizes:
                num_nodes = data[
                    'paper'].num_nodes if hetero else data.num_nodes
                sampler = torch.utils.data.RandomSampler(
                    range(num_nodes), num_samples=args.num_steps * batch_size
                ) if args.num_steps != -1 and with_loader else None
                kwargs = {
                    'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': args.num_workers,
                }
                if not hetero:
                    subgraph_loader = NeighborLoader(
                        data,
                        num_neighbors=[-1],  # layer-wise inference
                        input_nodes=mask,
                        sampler=sampler,
                        **kwargs,
                    ) if with_loader else None
                    if args.evaluate and not args.full_batch:
                        test_loader = NeighborLoader(
                            data,
                            num_neighbors=[-1],  # layer-wise inference
                            input_nodes=test_mask,
                            sampler=None,
                            **kwargs,
                        )

                for layers in args.num_layers:
                    num_neighbors = [args.hetero_num_neighbors] * layers
                    if hetero:
                        # batch-wise inference
                        subgraph_loader = NeighborLoader(
                            data,
                            num_neighbors=num_neighbors,
                            input_nodes=mask,
                            sampler=sampler,
                            **kwargs,
                        ) if with_loader else None
                        if args.evaluate and not args.full_batch:
                            test_loader = NeighborLoader(
                                data,
                                num_neighbors=num_neighbors,
                                input_nodes=test_mask,
                                sampler=None,
                                **kwargs,
                            )

                    for hidden_channels in args.num_hidden_channels:
                        print('----------------------------------------------')
                        print(f'Batch size={batch_size}, '
                              f'Layers amount={layers}, '
                              f'Num_neighbors={num_neighbors}, '
                              f'Hidden features size={hidden_channels}, '
                              f'Sparse tensor={args.use_sparse_tensor}')
                        params = {
                            'inputs_channels': inputs_channels,
                            'hidden_channels': hidden_channels,
                            'output_channels': num_classes,
                            'num_heads': args.num_heads,
                            'num_layers': layers,
                        }

                        if model_name == 'pna':
                            if degree is None:
                                degree = PNAConv.get_degree_histogram(
                                    subgraph_loader)
                                print(f'Calculated degree for {dataset_name}.')
                            params['degree'] = degree

                        model = get_model(
                            model_name, params,
                            metadata=data.metadata() if hetero else None)
                        model = model.to(device)
                        # TODO: Migrate to ModelHubMixin.
                        if args.ckpt_path:
                            state_dict = torch.load(args.ckpt_path)
                            model.load_state_dict(state_dict)
                        model.eval()
                        if args.device == 'xpu':
                            model = ipex.optimize(model)

                        # Define context manager parameters:
                        if args.cpu_affinity and with_loader:
                            cpu_affinity = subgraph_loader.enable_cpu_affinity(
                                args.loader_cores)
                        else:
                            cpu_affinity = nullcontext()
                        if args.profile and args.device == 'xpu':
                            profile = xpu_profile(args.export_chrome_trace)
                        elif args.profile:
                            profile = torch_profile(args.export_chrome_trace,
                                                    csv_data, args.write_csv)
                        else:
                            profile = nullcontext()
                        itt = emit_itt(
                        ) if args.vtune_profile else nullcontext()

                        if args.full_batch and args.use_sparse_tensor:
                            data = transformation(data)

                        with cpu_affinity, amp, timeit() as time:
                            inference_kwargs = dict(cache=args.cached_loader)
                            if args.reuse_device_for_embeddings and not hetero:
                                inference_kwargs['embedding_device'] = device
                            for _ in range(args.warmup):
                                if args.full_batch:
                                    full_batch_inference(model, data)
                                else:
                                    model.inference(
                                        subgraph_loader,
                                        device,
                                        progress_bar=True,
                                        **inference_kwargs,
                                    )
                            if args.warmup > 0:
                                time.reset()
                            with itt, profile:
                                if args.full_batch:
                                    y = full_batch_inference(model, data)
                                    if args.evaluate:
                                        mask = data.test_mask
                                        pred = y[mask].argmax(1)
                                        test_acc = pred.eq(data.y[mask]).sum(
                                        ).item() / mask.sum().item()
                                        print(f'Full Batch Test Accuracy: \
                                            {test_acc:.4f}')
                                else:
                                    y = model.inference(
                                        subgraph_loader,
                                        device,
                                        progress_bar=True,
                                        **inference_kwargs,
                                    )
                                    if args.evaluate:
                                        test_acc = test(
                                            model,
                                            test_loader,
                                            device,
                                            hetero,
                                            progress_bar=True,
                                        )
                                        print(f'Mini Batch Test Accuracy: \
                                            {test_acc:.4f}')

                        if args.profile and args.export_chrome_trace:
                            rename_profile_file(model_name, dataset_name,
                                                str(batch_size), str(layers),
                                                str(hidden_channels),
                                                str(num_neighbors))
                        total_time = time.duration
                        if args.num_steps != -1:
                            total_num_samples = args.num_steps * batch_size
                        else:
                            total_num_samples = num_nodes
                        throughput = total_num_samples / total_time
                        latency = total_time / total_num_samples * 1000
                        print(f'Throughput: {throughput:.3f} samples/s')
                        print(f'Latency: {latency:.3f} ms')

                        num_records = 1
                        if args.write_csv == 'prof':
                            # For profiling with PyTorch, we save the top-5
                            # most time consuming operations. Therefore, the
                            # same data should be entered for each of them.
                            num_records = 5
                        for _ in range(num_records):
                            save_benchmark_data(
                                csv_data,
                                batch_size,
                                layers,
                                num_neighbors,
                                hidden_channels,
                                total_time,
                                model_name,
                                dataset_name,
                                args.use_sparse_tensor,
                            )
    if args.write_csv:
        write_to_csv(csv_data, args.write_csv)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN inference benchmark')
    add = argparser.add_argument

    add('--device', choices=['cpu', 'cuda', 'xpu'], default='cpu',
        help='Device to run benchmark on')
    add('--reuse-device-for-embeddings', action='store_true',
        help='Use the same device for embeddings as specified in "--device"')
    add('--datasets', nargs='+',
        default=['ogbn-mag', 'ogbn-products', 'Reddit'], type=str)
    add('--use-sparse-tensor', action='store_true',
        help='use torch_sparse.SparseTensor as graph storage format')
    add('--models', nargs='+',
        default=['edge_cnn', 'gat', 'gcn', 'pna', 'rgat', 'rgcn'], type=str)
    add('--root', default='../../data', type=str,
        help='relative path to look for the datasets')
    add('--eval-batch-sizes', nargs='+', default=[512, 1024, 2048, 4096, 8192],
        type=int)
    add('--num-layers', nargs='+', default=[2, 3], type=int)
    add('--num-hidden-channels', nargs='+', default=[64, 128, 256], type=int)
    add('--num-heads', default=2, type=int,
        help='number of hidden attention heads, applies only for gat and rgat')
    add('--hetero-num-neighbors', default=10, type=int,
        help='number of neighbors to sample per layer for hetero workloads')
    add('--num-workers', default=0, type=int)
    add('--num-steps', default=-1, type=int,
        help='number of steps, -1 means iterating through all the data')
    add('--warmup', default=1, type=int)
    add('--profile', action='store_true')
    add('--vtune-profile', action='store_true')
    add('--bf16', action='store_true')
    add('--cpu-affinity', action='store_true',
        help='Use DataLoader affinitzation.')
    add('--loader-cores', nargs='+', default=[], type=int,
        help="List of CPU core IDs to use for DataLoader workers")
    add('--measure-load-time', action='store_true')
    add('--full-batch', action='store_true', help='Use full batch mode')
    add('--evaluate', action='store_true')
    add('--ckpt_path', type=str, help='Checkpoint path for loading a model')
    add('--write-csv', choices=[None, 'bench', 'prof'], default=None,
        help='Write benchmark or PyTorch profile data to CSV')
    add('--export-chrome-trace', default=True, type=bool,
        help='Export chrome trace file. Works only with PyTorch profiler')
    add('--cached-loader', action='store_true', help='Use CachedLoader')
    run(argparser.parse_args())
