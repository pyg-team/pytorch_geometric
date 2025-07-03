import argparse
import os.path as osp
import time

import cupy
import psutil
import rmm
import torch
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.torch import rmm_torch_allocator

# Must change allocators immediately upon import
# or else other imports will cause memory to be
# allocated and prevent changing the allocator
# rmm.reinitialize() provides an easy way to initialize RMM
# with specific memory resource options across multiple devices.
# See help(rmm.reinitialize) for full details.
rmm.reinitialize(devices=[0], pool_allocator=True, managed_memory=True)
cupy.cuda.set_allocator(rmm_cupy_allocator)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

import cudf  # noqa
import cugraph_pyg  # noqa
import torch.nn.functional as F  # noqa
# Enable cudf spilling to save gpu memory
from cugraph_pyg.loader import NeighborLoader  # noqa
from ogb.nodeproppred import PygNodePropPredDataset  # noqa

import torch_geometric  # noqa

cudf.set_option("spill", True)


def arg_parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ogbn-arxiv',
        choices=['ogbn-papers100M', 'ogbn-products', 'ogbn-arxiv'],
        help='Dataset name.',
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='/workspace/data',
        help='Root directory of dataset.',
    )
    parser.add_argument(
        "--dataset_subdir",
        type=str,
        default="ogbn-arxiv",
        help="directory of dataset.",
    )
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('--fan_out', type=int, default=10)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument(
        '--use_directed_graph',
        action='store_true',
        help='Whether or not to use directed graph',
    )
    parser.add_argument(
        '--add_self_loop',
        action='store_true',
        help='Whether or not to add self loop',
    )
    parser.add_argument(
        "--model",
        type=str,
        default='SAGE',
        choices=[
            'SAGE',
            'GAT',
            'GCN',
            # TODO: Uncomment when we add support for disjoint sampling
            # 'SGFormer',
        ],
        help="Model used for training, default SAGE",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="If using GATConv or GT, number of attention heads to use",
    )
    parser.add_argument('--tempdir_root', type=str, default=None)
    args = parser.parse_args()
    return args


def create_loader(
    data,
    num_neighbors,
    input_nodes,
    replace,
    batch_size,
    stage_name,
    shuffle=False,
):
    print(f'Creating {stage_name} loader...')

    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        replace=replace,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train(model, train_loader):
    model.train()

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        batch = batch.cuda()
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        y = batch.y[:batch.batch_size].view(-1).to(torch.long)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss * y.size(0)
        total_correct += out.argmax(dim=-1).eq(y).sum()
        total_examples += y.size(0)

    return total_loss.item() / total_examples, total_correct.item(
    ) / total_examples


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = total_examples = 0
    for batch in loader:
        batch = batch.cuda()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        y = batch.y[:batch.batch_size].view(-1).to(torch.long)

        total_correct += out.argmax(dim=-1).eq(y).sum()
        total_examples += y.size(0)

    return total_correct.item() / total_examples


if __name__ == '__main__':
    args = arg_parse()
    torch_geometric.seed_everything(123)
    if "papers" in str(args.dataset) and (psutil.virtual_memory().total /
                                          (1024**3)) < 390:
        print("Warning: may not have enough RAM to use this many GPUs.")
        print("Consider upgrading RAM if an error occurs.")
        print("Estimated RAM Needed: ~390GB.")
    wall_clock_start = time.perf_counter()

    root = osp.join(args.dataset_dir, args.dataset_subdir)
    print('The root is: ', root)
    dataset = PygNodePropPredDataset(name=args.dataset, root=root)
    split_idx = dataset.get_idx_split()

    data = dataset[0]
    if not args.use_directed_graph:
        data.edge_index = torch_geometric.utils.to_undirected(
            data.edge_index, reduce="mean")
    if args.add_self_loop:
        data.edge_index, _ = torch_geometric.utils.remove_self_loops(
            data.edge_index)
        data.edge_index, _ = torch_geometric.utils.add_self_loops(
            data.edge_index, num_nodes=data.num_nodes)

    graph_store = cugraph_pyg.data.GraphStore()
    graph_store[dict(
        edge_type=('node', 'rel', 'node'),
        layout='coo',
        is_sorted=False,
        size=(data.num_nodes, data.num_nodes),
    )] = data.edge_index

    feature_store = cugraph_pyg.data.TensorDictFeatureStore()
    feature_store['node', 'x', None] = data.x
    feature_store['node', 'y', None] = data.y

    data = (feature_store, graph_store)

    print(f"Training {args.dataset} with {args.model} model.")
    if args.model == "GAT":
        model = torch_geometric.nn.models.GAT(dataset.num_features,
                                              args.hidden_channels,
                                              args.num_layers,
                                              dataset.num_classes,
                                              heads=args.num_heads).cuda()
    elif args.model == "GCN":
        model = torch_geometric.nn.models.GCN(
            dataset.num_features,
            args.hidden_channels,
            args.num_layers,
            dataset.num_classes,
        ).cuda()
    elif args.model == "SAGE":
        model = torch_geometric.nn.models.GraphSAGE(
            dataset.num_features,
            args.hidden_channels,
            args.num_layers,
            dataset.num_classes,
        ).cuda()
    elif args.model == 'SGFormer':
        # TODO add support for this with disjoint sampling
        model = torch_geometric.nn.models.SGFormer(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            trans_num_heads=args.num_heads,
            trans_dropout=args.dropout,
            gnn_num_layers=args.num_layers,
            gnn_dropout=args.dropout,
        ).cuda()
    else:
        raise ValueError('Unsupported model type: {args.model}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.wd)

    loader_kwargs = dict(
        data=data,
        num_neighbors=[args.fan_out] * args.num_layers,
        replace=False,
        batch_size=args.batch_size,
    )

    train_loader = create_loader(
        input_nodes=split_idx['train'],
        stage_name='train',
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = create_loader(
        input_nodes=split_idx['valid'],
        stage_name='val',
        **loader_kwargs,
    )

    test_loader = create_loader(
        input_nodes=split_idx['test'],
        stage_name='test',
        **loader_kwargs,
    )
    prep_time = round(time.perf_counter() - wall_clock_start, 2)
    print("Total time before training begins (prep_time) =", prep_time,
          "seconds")
    print("Beginning training...")
    val_accs = []
    times = []
    train_times = []
    inference_times = []
    best_val = 0.
    start = time.perf_counter()
    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        train_start = time.perf_counter()
        loss, train_acc = train(model, train_loader)
        train_end = time.perf_counter()
        train_times.append(train_end - train_start)
        inference_start = time.perf_counter()
        train_acc = test(model, train_loader)
        val_acc = test(model, val_loader)

        inference_times.append(time.perf_counter() - inference_start)
        val_accs.append(val_acc)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train:'
              f' {train_acc:.4f} Time: {train_end - train_start:.4f}s')
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, ')

        times.append(time.perf_counter() - train_start)
        if val_acc > best_val:
            best_val = val_acc

    print(f"Total time used: is {time.perf_counter()-start:.4f}")
    val_acc = torch.tensor(val_accs)
    print('============================')
    print("Average Epoch Time on training: {:.4f}".format(
        torch.tensor(train_times).mean()))
    print("Average Epoch Time on inference: {:.4f}".format(
        torch.tensor(inference_times).mean()))
    print(f"Average Epoch Time: {torch.tensor(times).mean():.4f}")
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
    print(f'Final Validation: {val_acc.mean():.4f} ± {val_acc.std():.4f}')
    print(f"Best validation accuracy: {best_val:.4f}")

    print("Testing...")
    final_test_acc = test(model, test_loader)
    print(f'Test Accuracy: {final_test_acc:.4f}')

    total_time = round(time.perf_counter() - wall_clock_start, 2)
    print("Total Program Runtime (total_time) =", total_time, "seconds")
