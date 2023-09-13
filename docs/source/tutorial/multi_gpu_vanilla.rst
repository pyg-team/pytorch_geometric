Multi-GPU Training in Pure PyTorch
==================================

For many large scale, real-world datasets, it may be necessary to scale-up training across multiple GPUs.
This tutorial goes over mutli-GPU training
In particular, this tutorial introduces how to utilize :pyg:`PyG` with pure :pytorch:`PyTorch` for multi-GPU training via :class:`torch.nn.parallel.DistributedDataParallel`, without the need for any other third-party libraries (such as :lightning:`PyTorch Lightning`).

To start, we can take a look at the `distributed sampling <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py>`__ example from :pyg:`PyG`.
This example shows how to use train a :class:`~torch_geometric.nn.models.GraphSAGE` GNN model on the :class:`~torch_geometric.datasets.Reddit` dataset.
This example uses the :class:`~torch_geometric.loader.NeighborLoader` with :class:`torch.nn.parallel.DistributedDataParallel` to scale-up training across all available GPU's on your machine.


Defining our Spawnable Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To begin, we want to define a spawnable runner function.
In our function, the world_size is the number of GPU's we will be using at once.
For each gpu, the process is labeled with a process ID we call rank.
To run multi-gpu we spawn a runner for each GPU in main.
Note that we initialize the dataset in main before spawning our runners to keep the dataset in shared memory.

.. code-block:: python

    def run(rank, world_size, dataset):
        ...

    if __name__ == '__main__':
        from torch_geometric.datasets import Reddit
        import torch.multiprocessing as mp
        dataset = Reddit('../../data/Reddit')
        world_size = torch.cuda.device_count()
        print('Let\'s use', world_size, 'GPUs!')
        mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)

Now we start to define our spawnable runner function.

.. code-block:: python

    import os
    import torch.distributed as dist
    import torch
    def run(rank, world_size, dataset):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        data = dataset[0]
        data = data.to(rank, 'x', 'y')

The first step above is initializing torch distributed.
Notice also that we move to features and labels to GPU for faster feature fetching.
Next we split training indices into :obj:`world_size` many chunks for each GPU:

.. code-block:: python

        from torch_geometric.loader import NeighborLoader
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

        kwargs = dict(batch_size=1024, num_workers=4, persistent_workers=True)
        train_loader = NeighborLoader(data, input_nodes=train_idx,
                                     num_neighbors=[25, 10], shuffle=True,
                                     drop_last=True, **kwargs)

Note that our run function is called on each rank, which means each rank's NeighborLoader is sampling from a reduced set of the training indices.

We also create a single-hop evaluation neighbor loader. Note that we only do this on rank 0 since only one process needs to evaluate.

.. code-block:: python

        if rank == 0:
            val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
            val_loader = NeighborLoader(data, num_neighbors=[25, 10], input_nodes=val_idx, shuffle=False, **kwargs)

Now that we have our data loaders defined, we initialize our model and wrap it in PyTorch's DistributedDataParallel.
This wrapper on our model manages communication between each rank and reduces loss gradients from each process before updating the models parameters across all ranks.

.. code-block:: python

        from torch.nn.parallel import DistributedDataParallel
        from torch_geometric.nn.models import GraphSAGE
        torch.manual_seed(12345)
        model = GraphSAGE(in_channels=dataset.num_features,
                hidden_channels=256,
                num_layers=2,
                out_channels=dataset.num_classes).to(rank)
        model = DistributedDataParallel(model, device_ids=[rank])

Now we set up our optimizer and define our training loop. Notice that we move the edge indices of each mini-batch to GPU while the features and labels are already on GPU.

.. code-block:: python

        import torch.nn.functional as F
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, 21):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index.to(rank))[:batch.batch_size]
                loss = F.cross_entropy(out, batch.y[:batch.batch_size])
                loss.backward()
                optimizer.step()

After each training epoch, we evaluate and report accuracies:

.. code-block:: python

        dist.barrier()

        if rank == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            count = 0.0
            correct = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch.x, batch.edge_index.to(rank))[:batch.batch_size]
                    correct += (out.argmax(dim=-1) == batch.y[:batch.batch_size]).sum()
                    count += batch.batch_size
            print(f'Val Accuracy: {correct/count:.4f}')

        dist.barrier()

      dist.destroy_process_group()

Putting it all together gives a working multi-gpu example!
