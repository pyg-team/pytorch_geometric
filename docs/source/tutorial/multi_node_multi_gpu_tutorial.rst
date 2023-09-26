Multi-Node-Multi-GPU GNN Training
==================================

Before doing this tutorial we recommend going through <insert single-node-multi-gpu tutorial> as a warm up.
Our first step is to understand the basic structure of a multi-node-multi-gpu example.


.. code-block:: python

    import argparse
    import time
    import warnings

    import torch
    import torch.distributed as dist
    from torch_geometric.datasets import FakeDataset

    from torch_geometric.nn.models import GCN

    warnings.filterwarnings("ignore")


    _LOCAL_PROCESS_GROUP = None


    def create_local_process_group(num_workers_per_node):
        ...


    def get_local_process_group():
        ...


    def run(device, data, world_size, model, epochs, batch_size, fan_out,
                  split_idx, num_classes):
        ...


    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--hidden_channels', type=int, default=64)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--epochs', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--fan_out', type=int, default=50)
        parser.add_argument(
            "--ngpu_per_node",
            type=int,
            default="1",
            help="number of GPU(s) for each node for multi-gpu training,",
        )
        args = parser.parse_args()
        # setup multi node
        torch.distributed.init_process_group("nccl")
        nprocs = dist.get_world_size()
        create_local_process_group(args.ngpu_per_node)
        local_group = get_local_process_group()
        device_id = dist.get_rank(
            group=local_group) if dist.is_initialized() else 0
        torch.cuda.set_device(device_id)
        device = torch.device(device_id)

        dataset = FakeDataset(avg_num_nodes=100000)
        data = dataset.data
        num_nodes = data.num_nodes
        rand_id = torch.randperm(num_nodes)

        # 60/20/20 split
        split_idx = {
            'train':rand_id[:int(.6 * num_nodes)],
            'valid':rand_id[int(.6 * num_nodes):int(.8 * num_nodes)],
            'test':rand_id[:int(.8 * num_nodes):],
        }

        model = GCN(dataset.num_features, args.hidden_channels, 2,
                    dataset.num_classes)
        run(device, data, nprocs, model, args.epochs, args.batch_size,
                  args.fan_out, split_idx, dataset.num_classes)


Similarly to the warm up example, we define a :meth:`run` function. However, in this case we are using torch distributed with NVIDIA NCCL backend, instead of relying on :class:`~torch.multiprocessing`. Because we are running on multiple nodes, we want to set up a local process group for each node, and use :obj:`args.ngpu_per_node` GPUs per node. We then select the the CUDA device that will be used by each process within each process group. The next steps are fairly basic :pyg:`PyG` and :pytorch:`PyTorch` usage. We load our (synthetic) dataset and then set up our 60/20/20 train/val/test split. Next, we define our :class:`~torch_geometric.nn.models.GCN` model and finally call our :meth:`run` function.

Before we look into how our run function should be defined, we need to understand how we create and get our local process groups.


.. code-block:: python

    def create_local_process_group(num_workers_per_node):
        global _LOCAL_PROCESS_GROUP
        assert _LOCAL_PROCESS_GROUP is None
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        assert world_size % num_workers_per_node == 0

        num_nodes = world_size // num_workers_per_node
        node_rank = rank // num_workers_per_node
        for i in range(num_nodes):
            ranks_on_i = list(
                range(i * num_workers_per_node, (i + 1) * num_workers_per_node))
            pg = dist.new_group(ranks_on_i)
            if i == node_rank:
                _LOCAL_PROCESS_GROUP = pg


    def get_local_process_group():
        assert _LOCAL_PROCESS_GROUP is not None
        return _LOCAL_PROCESS_GROUP

To create our local process groups we create a :class:`~torch.distributed.new_group` from the sequential ranks split into groups of :obj:`num_workers_per_node`. We then store this value in a global variable for each node which we access via :meth:`get_local_process_group`.

The final step of coding is to define our :meth:`run` function:

.. code-block:: python

    from torch.nn.parallel import DistributedDataParallel
    from torchmetrics import Accuracy
    import torch.nn.functional as F
    from torch_geometric.loader import NeighborLoader

    def run(device, data, world_size, model, epochs, batch_size, fan_out,
                  split_idx, num_classes):
        local_group = get_local_process_group()
        loc_id = dist.get_rank(group=local_group)
        rank = torch.distributed.get_rank()
        if rank == 0:
            print("Data =", data)
            print('Using', nprocs, 'GPUs...')
        split_idx['train'] = split_idx['train'].split(
            split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[loc_id])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                     weight_decay=0.0005)
        acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

        train_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                      input_nodes=split_idx['train'],
                                      batch_size=batch_size)
        if rank == 0:
            eval_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                         input_nodes=split_idx['valid'],
                                         batch_size=batch_size)
            test_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                         input_nodes=split_idx['test'],
                                         batch_size=batch_size)
        eval_steps = 100
        acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        if rank == 0:
            print("Beginning training...")
        for epoch in range(epochs):
            for i, batch in enumerate(train_loader):
                if i >= 10:
                    start = time.time()
                batch = batch.to(device)
                batch.y = batch.y.to(torch.long)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
                loss.backward()
                optimizer.step()
                if rank == 0 and i % 10 == 0:
                    print("Epoch: " + str(epoch) + ", Iteration: " + str(i) +
                          ", Loss: " + str(loss))
            if rank == 0:
                print("Average Training Iteration Time:",
                      (time.time() - start) / (i - 10), "s/iter")
                acc_sum = 0.0
                with torch.no_grad():
                    for i, batch in enumerate(eval_loader):
                        if i >= eval_steps:
                            break
                        if i >= 10:
                            start = time.time()
                        batch = batch.to(device)
                        batch.y = batch.y.to(torch.long)
                        out = model(batch.x, batch.edge_index)
                        acc_sum += acc(out[:batch_size].softmax(dim=-1),
                                       batch.y[:batch_size])
                # We should expect poor Val/Test accuracy's since data is random
                print(f"Validation Accuracy: {acc_sum/(i) * 100.0:.4f}%", )
                print("Average Inference Iteration Time:",
                      (time.time() - start) / (i - 10), "s/iter")
        if rank == 0:
            acc_sum = 0.0
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    batch = batch.to(device)
                    batch.y = batch.y.to(torch.long)
                    out = model(batch.x, batch.edge_index)
                    acc_sum += acc(out[:batch_size].softmax(dim=-1),
                                   batch.y[:batch_size])
                print(f"Test Accuracy: {acc_sum/(i) * 100.0:.4f}%", )

Our :meth:`run` function is very similar to that of our warm up example except for the beginning. In this tutorial our distributed groups have already been initialized so we only need to assign our :obj:`loc_id` for the local GPU id for each device on each node. We also need to assign our global :obj:`rank`. As an example to understand this better, consider a scendario where we use use 3 nodes with 8 GPUs each. The 7th GPU on the 3rd node, or the 23rd GPU in our system, that GPUs process would be rank :obj:`22`. However the value of :obj:`loc_id` for that GPU would be :obj:`6`.

After that its very similar to our warm up:
    1. We put :class:`~torch_geometric.nn.GCN` model on :obj:`device` and wrap it inside :class:`~torch.nn.parallel.DistributedDataParallel`, passing the :obj:`loc_id` for :obj:`device_id` parameter.
    2. We then set up our optimizer and accuracy objective for evalution and testing.
    3. We split training indices into :obj:`world_size` many chunks for each GPU, and initialize the :class:`~torch_geometric.loader.NeighborLoader` class to only operate on its specific chunk of the training set.
    4. We create a :class:`~torch_geometric.loader.NeighborLoader` instance for evaluation. Again, for simplicity, we only do this on rank :obj:`0`
    5. Finally we follow a similar training and evaluation loop as our warmup example.

And that's all the coding.

Putting it all together gives a working multi-node-multi-GPU example that follows a training flow that is similar to single GPU training.
You can run the shown tutorial by yourself by looking at `examples/multi_gpu/multi_node_multi_gpu_synthetic.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/multi_node_multi_gpu_synthetic.py>`_.

However, to run the example you need to use slurm on a cluster with pyxis enabled. Here's how:

Step 1:

In your slurm login terminal:

.. code-block:: bash

    srun --overlap -A <slurm_access_group> -p interactive -J <experiment-name> -N 2 -t 02:00:00 --pty bash


Then open another slurm login terminal for step 2:

.. code-block:: bash

    squeue -u <slurm-unix-account-id>
    export jobid=<JOBID from SQUEUE>


Step 3:

Now we are going to pull a container with a functional PyG and CUDA environment onto each node.

.. code-block:: bash

    srun -l -N<num_nodes> --ntasks-per-node=1 --overlap --jobid=$jobid \
    --container-image=<image_url> --container-name=cont \
    --container-mounts=<data-directory>/ogb-papers100m/:/workspace/dataset true

NVIDIA recommends using our NVIDIA PyG container updated each month with the latest from NVIDIA and PyG. Sign up for early access at `developer.nvidia.com/pyg-container-early-access <https://developer.nvidia.com/pyg-container-early-access>`_. General availability on `NVIDIA NGC <https://www.ngc.nvidia.com/>`_ is set for the end of 2023. Alternatively, see `docker.com <https://www.docker.com/>`_ for information on creating your own container.

Once you have your container loaded, simply run:
Step 4:

    srun -l -N<num_nodes> --ntasks-per-node=<ngpu_per_node> --overlap --jobid=$jobid \
    --container-name=cont \
    python3 pyg_multinode_tutorial.py --ngpu_per_node <>

Give it a try!
