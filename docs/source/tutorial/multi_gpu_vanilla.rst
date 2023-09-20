Multi-GPU Training in Pure PyTorch
==================================

For many large scale, real-world datasets, it may be necessary to scale-up training across multiple GPUs.
This tutorial goes over how to set up a multi-GPU training and inference pipeline in :pyg:`PyG` with pure :pytorch:`PyTorch` via :class:`torch.nn.parallel.DistributedDataParallel`, without the need for any other third-party libraries (such as :lightning:`PyTorch Lightning`).

In particular, this tutorials shows how to train a :class:`~torch_geometric.nn.models.GraphSAGE` GNN model on the :class:`~torch_geometric.datasets.Reddit` dataset.
For this, we utilize the :class:`~torch_geometric.loader.NeighborLoader` together with :class:`torch.nn.parallel.DistributedDataParallel` to scale-up training across all available GPUs.

.. note::
    A runnable example of this tutorial can be found at `examples/multi_gpu/distributed_sampling.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py>`_.

Defining a Spawnable Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`DistributedDataParallel (DDP) <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_ implements data parallelism at the module level which can run across multiple machines.
Applications using DDP spawn multiple processes and create a single DDP instance per process.
DDP processes can be placed on the same machine or across machines.

To create a DDP module, we first need to set up process groups properly and define a spawnable runner function.
Here, the :obj:`world_size` corresponds to the number of GPUs we will be using at once.
For each GPU, the process is labeled with a process ID which we call :obj:`rank`.
:meth:`torch.multiprocessing.spawn` will take care of spawning :obj:`world_size` many processes:

.. code-block:: python

    from torch_geometric.datasets import Reddit
    import torch.multiprocessing as mp

    def run(rank: int, world_size: int, dataset: Reddit):
        pass

    if __name__ == '__main__':
        dataset = Reddit('./data/Reddit')
        world_size = torch.cuda.device_count()
        mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)

Note that we initialize the dataset *before* spawning any processes.
With this, we only initialize the dataset once, and any data inside it will be automatically moved to shared memory via :obj:`torch.multiprocessing` such that processes do not need to work on replicas of the data.

With this, we can start to implement our spawnable runner function:

.. code-block:: python

    import os
    import torch.distributed as dist
    import torch

    def run(rank: int, world_size: int, dataset: Reddit):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        data = dataset[0]

The first step above is initializing :obj:`torch.distributed`.
More details can be found in `Writing Distributed Applications with PyTorch <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`_.

Next, we split training indices into :obj:`world_size` many chunks for each GPU, and initialize the :class:`~torch_geometric.loader.NeighborLoader` class to only operate on its specific chunk of the training set:

.. code-block:: python

    from torch_geometric.loader import NeighborLoader

    def run(rank: int, world_size: int, dataset: Reddit):
        ...

        train_index = data.train_mask.nonzero().view(-1)
        train_index = train_index.split(train_index.size(0) // world_size)[rank]

        train_loader = NeighborLoader(
            data,
            input_nodes=train_index,
            num_neighbors=[25, 10],
            batch_size=1024,
            num_workers=4,
            shuffle=True,
        )

Note that our :meth:`run` function is called on each rank, which means that each rank holds a separate :class:`~torch_geometric.loader.NeighborLoader` instance.

Similarly, we create a :class:`~torch_geometric.loader.NeighborLoader` instance for evaluation.
For simplicity, we only do this on rank :obj:`0` such that computation of metrics do not need to communicate across different processes.
We recommend to take a look at the `torchmetrics <https://torchmetrics.readthedocs.io/en/stable/>`_ package for distributed computation of metrics.

.. code-block:: python

    def run(rank: int, world_size: int, dataset: Reddit):
        ...

        if rank == 0:
            val_index = data.val_mask.nonzero().view(-1)
            val_loader = NeighborLoader(
                data,
                input_nodes=val_index,
                num_neighbors=[25, 10],
                batch_size=1024,
                num_workers=4,
                shuffle=False,
            )

Now that we have our data loaders defined, we initialize our :class:`~torch_geometric.nn.GraphSAGE` model and wrap it inside :pytorch:`PyTorch`'s :class:`~torch.nn.parallel.DistributedDataParallel`.
This wrapper on our model manages communication between each rank and reduces loss gradients from each process before updating the models parameters across all ranks:

.. code-block:: python

    from torch.nn.parallel import DistributedDataParallel
    from torch_geometric.nn import GraphSAGE

    def run(rank: int, world_size: int, dataset: Reddit):
        ...

        torch.manual_seed(12345)
        model = GraphSAGE(
            in_channels=dataset.num_features,
            hidden_channels=256,
            num_layers=2,
            out_channels=dataset.num_classes,
        ).to(rank)
        model = DistributedDataParallel(model, device_ids=[rank])

Finally, we can set up our optimizer and define our training loop, which follows a similar flow as usual single GPU training loops  - the actual magic of gradient and model weight synchronization across different processes will happen behind the scenes within :class:`~torch.nn.parallel.DistributedDataParallel`:

.. code-block:: python

    import torch.nn.functional as F

    def run(rank: int, world_size: int, dataset: Reddit):
        ...

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, 11):
            model.train()
            for batch in train_loader:
                batch = batch.to_rank
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
                loss = F.cross_entropy(out, batch.y[:batch.batch_size])
                loss.backward()
                optimizer.step()

After each training epoch, we evaluate and report validation metrics.
As previously mentioned, we do this on a single GPU only.
To synchronize all processes and to ensure that model weights have been updated, we need to call :meth:`torch.distributed.barrier`:

.. code-block:: python

            dist.barrier()

            if rank == 0:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

            if rank == 0:
                model.eval()
                count = correct = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(rank)
                        out = model(batch.x, batch.edge_index)[:batch.batch_size]
                        pred = out.argmax(dim=-1)
                        correct += (pred == batch.y[:batch.batch_size]).sum()
                        count += batch.batch_size
                print(f'Validation Accuracy: {correct/count:.4f}')

            dist.barrier()

After finishing training, we can clean up processes and destroy the process group via:

.. code-block:: python

        dist.destroy_process_group()

And that's it.
Putting it all together gives a working multi-GPU example that follows a similar training flow than single GPU training.
You can run the shown tutorial by yourself by looking at `examples/multi_gpu/distributed_sampling.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py>`_.
