Multi-GPU Training in Pure PyTorch
==================================

For many large scale, real-world datasets, it may be necessary to scale-up training across multiple GPUs.
This tutorial goes over how to set up a multi-GPU training  pipeline in :pyg:`PyG` with :pytorch:`PyTorch` via :class:`torch.nn.parallel.DistributedDataParallel`, without the need for any other third-party libraries (such as :lightning:`PyTorch Lightning`).
Note that this approach is based on data-parallelism.
This means that each GPU runs an identical copy of the model; you might want to look into `PyTorch FSDP <https://arxiv.org/abs/2304.11277>`_ if you want to scale your model across devices.
Data-parallelism allows you to increase the batch size of your model by aggregating gradients across GPUs and then sharing the same optimizer step within every model replica.
This `DDP+MNIST-tutorial <https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp#overall-idea-of-distributed-data-parallel>`_  by the Princeton University has some nice illustrations of the process.

Specifically this tutorial shows how to train a :class:`~torch_geometric.nn.models.GraphSAGE` GNN model on the :class:`~torch_geometric.datasets.Reddit` dataset.
For this, we will use :class:`torch.nn.parallel.DistributedDataParallel` to scale-up training across all available GPUs.
We will do this by spawning multiple processes from our :python:`Python` code which will all execute the same function.
Per process, we set up our model instance and feed data through it by utilizing the :class:`~torch_geometric.loader.NeighborLoader`.
Gradients are synchronized by wrapping the model in :class:`torch.nn.parallel.DistributedDataParallel` (as described in its `official tutorial <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_), which in turn relies on :obj:`torch.distributed`-IPC-facilities.

.. note::
    The complete script of this tutorial can be found at `examples/multi_gpu/distributed_sampling.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py>`_.

Defining a Spawnable Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create our training script, we use the :pytorch:`PyTorch`-provided wrapper of the vanilla :python:`Python` :class:`multiprocessing` module.
Here, the :obj:`world_size` corresponds to the number of GPUs we will be using at once.
:meth:`torch.multiprocessing.spawn` will take care of spawning :obj:`world_size` processes.
Each process will load the same script as a module and subsequently execute the :meth:`run`-function:

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
With this, we only initialize the dataset once, and any data inside it will be automatically moved to shared memory via :obj:`torch.multiprocessing` such that processes do not need to create their own replica of the data.
In addition, note how the :meth:`run` function accepts :obj:`rank` as its first argument.
This argument is not explicitly provided by us.
It corresponds to the process ID (starting at :obj:`0`) injected by :pytorch:`PyTorch`.
Later we will use this to select a unique GPU for every :obj:`rank`.

With this, we can start to implement our spawnable runner function.
The first step is to initialize a process group with :obj:`torch.distributed`.
To this point, processes are not aware of each other and we set a hardcoded server-address for rendezvous using the :obj:`nccl` protocol.
More details can be found in the `"Writing Distributed Applications with PyTorch" <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`_ tutorial:

.. code-block:: python

    import os
    import torch.distributed as dist
    import torch

    def run(rank: int, world_size: int, dataset: Reddit):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

Next, we split training indices into :obj:`world_size` many chunks for each GPU, and initialize the :class:`~torch_geometric.loader.NeighborLoader` class to only operate on its specific chunk of the training set:

.. code-block:: python

    from torch_geometric.loader import NeighborLoader

    def run(rank: int, world_size: int, dataset: Reddit):
        ...

        data = dataset[0]

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

Note that our :meth:`run` function is called for each rank, which means that each rank holds a separate :class:`~torch_geometric.loader.NeighborLoader` instance.

Similarly, we create a :class:`~torch_geometric.loader.NeighborLoader` instance for evaluation.
For simplicity, we only do this on rank :obj:`0` such that computation of metrics does not need to communicate across different processes.
We recommend taking a look at the `torchmetrics <https://torchmetrics.readthedocs.io/en/stable/>`_ package for distributed computation of metrics.

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

Now that we have our data loaders defined, we initialize our :class:`~torch_geometric.nn.GraphSAGE` model and wrap it inside :class:`torch.nn.parallel.DistributedDataParallel`.
We also move the model to its exclusive GPU using the :obj:`rank` as a shortcut for the full device identifier.
The wrapper on our model manages communication between each rank and synchronizes gradients across all ranks before updating the model parameters across all ranks:

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

Finally, we can set up our optimizer and define our training loop, which follows a similar flow as usual single GPU training loops - the actual magic of gradient and model weight synchronization across different processes will happen behind the scenes within :class:`~torch.nn.parallel.DistributedDataParallel`:

.. code-block:: python

    import torch.nn.functional as F

    def run(rank: int, world_size: int, dataset: Reddit):
        ...

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, 11):
            model.train()
            for batch in train_loader:
                batch = batch.to(rank)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
                loss = F.cross_entropy(out, batch.y[:batch.batch_size])
                loss.backward()
                optimizer.step()

After each training epoch, we evaluate and report validation metrics.
As previously mentioned, we do this on a single GPU only.
To synchronize all processes and to ensure that the model weights have been updated, we need to call :meth:`torch.distributed.barrier`:

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
Putting it all together gives a working multi-GPU example that follows a training flow that is similar to single GPU training.
You can run the shown tutorial by yourself by looking at `examples/multi_gpu/distributed_sampling.py <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py>`_.
