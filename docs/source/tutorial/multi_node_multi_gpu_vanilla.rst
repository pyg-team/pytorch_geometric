Multi-Node-Multi-GPU Training in Pure PyTorch
=============================================

This tutorial introduces a skeleton on how to perform distributed training on multiple GPUs over multiple nodes.
Before going through this tutorial, we recommend reading `our tutorial on single-node multi-GPU training <multi_gpu_vanilla.html>`_ as a warm up.

.. note::
    A runnable example of this tutorial can be found at `examples/multi_gpu/multinode_multigpu_papers100m_gcn.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/multinode_multigpu_papers100m_gcn.py>`_.

Our first step is to understand the basic structure of a multi-node-multi-GPU example:

.. code-block:: python

    import argparse

    import torch
    import torch.distributed as dist
    from torch_geometric.datasets import FakeDataset
    from torch_geometric.nn.models import GCN
    import os

    def run(device, world_size, data, model):
        pass

    if __name__ == '__main__':
        torch.distributed.init_process_group("nccl")
        nprocs = dist.get_world_size()
        if dist.is_initialized():
            device_id = dist.get_rank()
        else:
            device_id = 0
        torch.cuda.set_device(device_id)
        device = torch.device(device_id)

        dataset = FakeDataset(avg_num_nodes=100_000)
        model = GCN(dataset.num_features, 64, dataset.num_classes, num_layers=2)

        run(device, nprocs, dataset[0], model)

Similarly to the single-node multi-GPU example, we define a :meth:`run` function. However, in this case we are using :obj:`torch.distributed` with NVIDIA NCCL backend instead of relying on :class:`~torch.multiprocessing`.
Check out this :pytorch:`null` `PyTorch tutorial on multi-node multi-GPU training <https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html>`_ for more details.
We then select the CUDA device that will be used by each process within each process group.
The next steps are fairly basic :pyg:`PyG` and :pytorch:`PyTorch` usage.
We load our (synthetic) dataset and define a :class:`~torch_geometric.nn.models.GCN` model and pass these to our :meth:`run` function.

The final step of coding is to define our :meth:`run` function:

.. code-block:: python

    def run(device, world_size, data, model):
        local_id = int(os.environ['LOCAL_RANK'])
        rank = torch.distributed.get_rank()

        ...

As such, we only need to assign our :obj:`local_id` and our global :obj:`rank`.
To understand this better, consider a scenario where we use three nodes with 8 GPUs each.
The 7th GPU on the 3rd node, or the 23rd GPU in our system, has the global process rank :obj:`22`, however, its local rank :obj:`local_id` is :obj:`6`.

After that, model training is very similar to `our single-node multi-GPU tutorial <multi_gpu_vanilla.html>`_:

.. code-block:: python

    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel
    from torch_geometric.loader import NeighborLoader

    def run(device, world_size, data, model):
        ...

        model = DistributedDataParallel(model.to(device), device_ids=[local_id])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        input_nodes = torch.arange(data.num_nodes).split(
            data.num_nodes // world_size,
        )[rank].clone()

        loader = NeighborLoader(
            dataset,
            input_nodes=input_nodes,
            num_neighbors=[10, 10],
            batch_size=128,
            shuffle=True,
        )

        for epoch in range(1, 10):
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
                y = batch.y[:batch.batch_size]
                loss = F.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()

1. We put our :class:`~torch_geometric.nn.models.GCN` model on its respective :obj:`device` and wrap it inside :class:`~torch.nn.parallel.DistributedDataParallel` while we pass :obj:`local_id` to its :obj:`device_id` parameter.
2. We then set up our optimizer for training.
3. We then split our input/seed nodes into :obj:`world_size` many chunks for each GPU, and initialize the :class:`~torch_geometric.loader.NeighborLoader` class to only operate on its specific subset of nodes.
4. Finally, we iterate over epochs and batches to train our GNN as usual.

And that's all the coding.
Putting it all together gives a working multi-node-multi-GPU example that follows a training flow that is similar to single GPU training or single-node multi-GPU training.

However, to run the example you need to use Slurm on a cluster with :obj:`pyxis` for container management enabled.
Speak with your cluster management team for more information on usage for your specific cluster, especially for alternative set ups.

.. code-block:: bash

    srun -l -N<num_nodes> --ntasks-per-node=<ngpu_per_node> --container-name=cont --container-image=<image_url> --container-mounts=/ogb-papers100m/:/workspace/dataset python3 path_to_script.py

NVIDIA provides a ready-to-use :pyg:`PyG` container that is updated each month with the latest from NVIDIA and :pyg:`PyG`.
You can sign up for early access `here <https://developer.nvidia.com/pyg-container-early-access>`_.
General availability on `NVIDIA NGC <https://www.ngc.nvidia.com/>`_ is set for the end of 2023.
Alternatively, see `docker.com <https://www.docker.com/>`_ for information on how to create your own container.