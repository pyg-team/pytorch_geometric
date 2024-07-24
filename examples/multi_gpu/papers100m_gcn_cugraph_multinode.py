# Copyright (c) 2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Multi-node, multi-GPU example with WholeGraph feature storage.
# Can be run with srun
# i.e. srun

import argparse
import os
import tempfile
import time
import json

from datetime import timedelta


import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel

import torch_geometric

from cugraph.gnn import (
    cugraph_comms_init,
    cugraph_comms_shutdown,
    cugraph_comms_create_unique_id,
)

from pylibwholegraph.torch.initialize import (
    init as wm_init,
    finalize as wm_finalize,
)

# Allow computation on objects that are larger than GPU memory
# https://docs.rapids.ai/api/cudf/stable/developer_guide/library_design/#spilling-to-host-memory
os.environ["CUDF_SPILL"] = "1"

# Ensures that a CUDA context is not created on import of rapids.
# Allows pytorch to create the context instead
os.environ["RAPIDS_NO_INITIALIZE"] = "1"


def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=True,
        pool_allocator=True,
    )

    import cupy

    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from cugraph.testing.mg_utils import enable_spilling

    enable_spilling()

    torch.cuda.set_device(local_rank)

    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())


def partition_data(dataset, split_idx, edge_path, feature_path, label_path, meta_path):
    data = dataset[0]

    # Split and save edge index
    os.makedirs(
        edge_path,
        exist_ok=True,
    )
    for (r, e) in enumerate(torch.tensor_split(data.edge_index, world_size, dim=1)):
        rank_path = os.path.join(edge_path, f"rank={r}.pt")
        torch.save(
            e.clone(),
            rank_path,
        )

    # Split and save features
    os.makedirs(
        feature_path,
        exist_ok=True,
    )

    for (r, f) in enumerate(torch.tensor_split(data.x, world_size)):
        rank_path = os.path.join(feature_path, f"rank={r}_x.pt")
        torch.save(
            f.clone(),
            rank_path,
        )
    for (r, f) in enumerate(torch.tensor_split(data.y, world_size)):
        rank_path = os.path.join(feature_path, f"rank={r}_y.pt")
        torch.save(
            f.clone(),
            rank_path,
        )

    # Split and save labels
    os.makedirs(
        label_path,
        exist_ok=True,
    )
    for (d, i) in split_idx.items():
        i_parts = torch.tensor_split(i, world_size)
        for r, i_part in enumerate(i_parts):
            rank_path = os.path.join(label_path, f"rank={r}")
            os.makedirs(rank_path, exist_ok=True)
            torch.save(i_part, os.path.join(rank_path, f"{d}.pt"))

    # Save metadata
    meta = {
        "num_classes": int(dataset.num_classes),
        "num_features": int(dataset.num_features),
        "num_nodes": int(data.num_nodes),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def load_partitioned_data(
    rank, edge_path, feature_path, label_path, meta_path, wg_mem_type
):
    from cugraph_pyg.data import GraphStore, WholeFeatureStore

    graph_store = GraphStore(is_multi_gpu=True)
    feature_store = WholeFeatureStore(memory_type=wg_mem_type)

    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load labels
    split_idx = {}
    for split in ["train", "test", "valid"]:
        split_idx[split] = torch.load(
            os.path.join(label_path, f"rank={rank}", f"{split}.pt")
        )

    # Load features
    feature_store["node", "x"] = torch.load(
        os.path.join(feature_path, f"rank={rank}_x.pt")
    )
    feature_store["node", "y"] = torch.load(
        os.path.join(feature_path, f"rank={rank}_y.pt")
    )

    # Load edge index
    eix = torch.load(os.path.join(edge_path, f"rank={rank}.pt"))
    graph_store[
        ("node", "rel", "node"), "coo", False, (meta["num_nodes"], meta["num_nodes"])
    ] = eix

    return (feature_store, graph_store), split_idx, meta


def run_train(
    global_rank,
    data,
    split_idx,
    world_size,
    device,
    model,
    epochs,
    batch_size,
    fan_out,
    num_classes,
    wall_clock_start,
    tempdir=None,
    num_layers=3,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    kwargs = dict(
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
    )
    # Set Up Neighbor Loading
    from cugraph_pyg.loader import NeighborLoader

    ix_train = split_idx["train"].cuda()
    train_path = os.path.join(tempdir, f"train_{global_rank}")
    os.mkdir(train_path)
    train_loader = NeighborLoader(
        data,
        input_nodes=ix_train,
        directory=train_path,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )

    ix_test = split_idx["test"].cuda()
    test_path = os.path.join(tempdir, f"test_{global_rank}")
    os.mkdir(test_path)
    test_loader = NeighborLoader(
        data,
        input_nodes=ix_test,
        directory=test_path,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=80000,
        **kwargs,
    )

    ix_valid = split_idx["valid"].cuda()
    valid_path = os.path.join(tempdir, f"valid_{global_rank}")
    os.mkdir(valid_path)
    valid_loader = NeighborLoader(
        data,
        input_nodes=ix_valid,
        directory=valid_path,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )

    dist.barrier()

    eval_steps = 1000
    warmup_steps = 20
    dist.barrier()
    torch.cuda.synchronize()

    if global_rank == 0:
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total time before training begins (prep_time) =", prep_time, "seconds")
        print("Beginning training...")

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if i == warmup_steps:
                torch.cuda.synchronize()
                start = time.time()

            batch = batch.to(device)
            batch_size = batch.batch_size

            batch.y = batch.y.view(-1).to(torch.long)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
            loss.backward()
            optimizer.step()
            if global_rank == 0 and i % 10 == 0:
                print(
                    "Epoch: "
                    + str(epoch)
                    + ", Iteration: "
                    + str(i)
                    + ", Loss: "
                    + str(loss)
                )
        nb = i + 1.0

        if global_rank == 0:
            print(
                "Average Training Iteration Time:",
                (time.time() - start) / (nb - warmup_steps),
                "s/iter",
            )

        with torch.no_grad():
            total_correct = total_examples = 0
            for i, batch in enumerate(valid_loader):
                if i >= eval_steps:
                    break

                batch = batch.to(device)
                batch_size = batch.batch_size

                batch.y = batch.y.to(torch.long)
                out = model(batch.x, batch.edge_index)[:batch_size]

                pred = out.argmax(dim=-1)
                y = batch.y[:batch_size].view(-1).to(torch.long)

                total_correct += int((pred == y).sum())
                total_examples += y.size(0)

            acc_val = total_correct / total_examples
            if global_rank == 0:
                print(
                    f"Validation Accuracy: {acc_val * 100.0:.4f}%",
                )

        torch.cuda.synchronize()

    with torch.no_grad():
        total_correct = total_examples = 0
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            batch_size = batch.batch_size

            batch.y = batch.y.to(torch.long)
            out = model(batch.x, batch.edge_index)[:batch_size]

            pred = out.argmax(dim=-1)
            y = batch.y[:batch_size].view(-1).to(torch.long)

            total_correct += int((pred == y).sum())
            total_examples += y.size(0)

        acc_test = total_correct / total_examples
        if global_rank == 0:
            print(
                f"Test Accuracy: {acc_test * 100.0:.4f}%",
            )

    if global_rank == 0:
        total_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total Program Runtime (total_time) =", total_time, "seconds")
        print("total_time - prep_time =", total_time - prep_time, "seconds")

    wm_finalize()
    cugraph_comms_shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--fan_out", type=int, default=30)
    parser.add_argument("--tempdir_root", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--skip_partition", action="store_true")
    parser.add_argument("--wg_mem_type", type=str, default="distributed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wall_clock_start = time.perf_counter()

    # Set a very high timeout so that PyTorch does not crash while
    # partitioning the data.
    dist.init_process_group("nccl", timeout=timedelta(minutes=60), rank=os.environ['SLURM_PROCID'])
    world_size = dist.get_world_size()
    assert dist.is_initialized(), "Distributed cluster not initialized"

    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)

    # Create the uid needed for cuGraph comms
    if global_rank == 0:
        cugraph_id = [cugraph_comms_create_unique_id()]
    else:
        cugraph_id = [None]
    dist.broadcast_object_list(cugraph_id, src=0, device=device)
    cugraph_id = cugraph_id[0]

    init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)

    # Split the data
    edge_path = os.path.join(args.dataset_root, args.dataset + "_eix_part")
    feature_path = os.path.join(args.dataset_root, args.dataset + "_fea_part")
    label_path = os.path.join(args.dataset_root, args.dataset + "_label_part")
    meta_path = os.path.join(args.dataset_root, args.dataset + "_meta.json")

    # We partition the data to avoid loading it in every worker, which will
    # waste memory and can lead to an out of memory exception.
    # cugraph_pyg.GraphStore and cugraph_pyg.WholeFeatureStore are always
    # constructed from partitions of the edge index and features, respectively,
    # so this works well.
    if not args.skip_partition and global_rank == 0:
        print("Partitioning the data into equal size parts per worker")
        dataset = PygNodePropPredDataset(name=args.dataset, root=args.dataset_root)
        split_idx = dataset.get_idx_split()

        partition_data(
            dataset,
            split_idx,
            meta_path=meta_path,
            label_path=label_path,
            feature_path=feature_path,
            edge_path=edge_path,
        )

    dist.barrier()
    print("Loading partitioned data")
    data, split_idx, meta = load_partitioned_data(
        rank=global_rank,
        edge_path=edge_path,
        feature_path=feature_path,
        label_path=label_path,
        meta_path=meta_path,
        wg_mem_type=args.wg_mem_type,
    )
    dist.barrier()

    model = torch_geometric.nn.models.GCN(
        meta["num_features"],
        args.hidden_channels,
        args.num_layers,
        meta["num_classes"],
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    print("Beginning training")
    with tempfile.TemporaryDirectory(dir=args.tempdir_root) as tempdir:
        run_train(
            global_rank,
            data,
            split_idx,
            world_size,
            device,
            model,
            args.epochs,
            args.batch_size,
            args.fan_out,
            meta["num_classes"],
            wall_clock_start,
            tempdir,
            args.num_layers,
        )
