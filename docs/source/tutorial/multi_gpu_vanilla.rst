Multi-GPU Training in Pure PyTorch
==================================

For many large scale, real-world datasets, it may be necessary to scale-up training across multiple GPUs.
This tutorial goes over how to do this on a small scale dataset and then shows how the same technique can be applied for large-scale recommendation systems.
In particular, this tutorial introduces how to utilize :pyg:`PyG` with pure :pytorch:`PyTorch` for multi-GPU training via :class:`torch.nn.parallel.DistributedDataParallel`, without the need for any other third-party libraries (such as :lightning:`PyTorch Lightning`).

Starting off small
------------------

To start, we can take a look at the `distributed sampling <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py>`__ example from :pyg:`PyG`.
This example shows how to use train a :class:`~torch_geometric.nn.models.GraphSAGE` GNN model on the :class:`~torch_geometric.datasets.Reddit` dataset.
This example uses the :class:`~torch_geometric.loader.NeighborLoader` with :class:`torch.nn.parallel.DistributedDataParallel` to scale-up training across all available GPU's on your machine.

Defining our GNN
~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch_geometric.nn import SAGEConv
   import tqdm

   class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all: Tensor, device: torch.device,
                  subgraph_loader: NeighborLoader) -> Tensor:

        pbar = tqdm(total=len(subgraph_loader) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.node_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                x = x[:batch.batch_size]
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
                pbar.update(1)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all

Defining our Spawnable Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we want to define a spawnable runner function. In our function, the world_size is the number of GPU's we will be using at once.
For each gpu, the process is labeled with a process ID we call rank.

.. code-block:: python

   import os
   import torch.distributed as dist
   def run(rank, world_size, dataset):
       os.environ['MASTER_ADDR'] = 'localhost'
       os.environ['MASTER_PORT'] = '12355'
       dist.init_process_group('nccl', rank=rank, world_size=world_size)

       data = dataset[0]
       data = data.to(rank, 'x', 'y')

Notice we move to features and labels to GPU for faster feature fetching.

Now we split training indices into :obj:`world_size` many chunks for each GPU:

.. code-block:: python

       train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
       train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

       kwargs = dict(batch_size=1024, num_workers=4, persistent_workers=True)
       train_loader = NeighborLoader(data, input_nodes=train_idx,
                                     num_neighbors=[25, 10], shuffle=True,
                                     drop_last=True, **kwargs)

Note that since our run function is called on each rank, which means each NeighborLoader is sampling from a reduced set of the training indices.

We also create a single-hop evaluation neighbor loader. Note that we only do this on rank 0 since only one process needs to evaluate.

.. code-block:: python

       if rank == 0:
           subgraph_loader = NeighborLoader(copy.copy(data), num_neighbors=[-1],
                                            shuffle=False, **kwargs)
           # No need to maintain these features during evaluation:
           del subgraph_loader.data.x, subgraph_loader.data.y
           # Add global node index information:
           subgraph_loader.data.node_id = torch.arange(data.num_nodes)

Now that we have our data loaders defined initialize our model and wrap it in PyTorch's DistributedDataParallel.
This wrapper on our model manages communication between each rank and reduces loss gradients from each process before updating the models parameters across all ranks.

.. code-block:: python

      from torch.nn.parallel import DistributedDataParallel
      torch.manual_seed(12345)
      model = SAGE(dataset.num_features, 256, dataset.num_classes).to(rank)
      model = DistributedDataParallel(model, device_ids=[rank])

Now we set up our optimizer and define our training loop. Notice that we move the edge indices of each mini-batch to GPU while the features and labels are already on GPU.

.. code-block:: python

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
            with torch.no_grad():
                out = model.module.inference(data.x, rank, subgraph_loader)
            res = out.argmax(dim=-1) == data.y.to(out.device)
            acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
            acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
            acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

      dist.destroy_process_group()


Finally we put it all together by spawning our runners for each GPU. Note that we initialize the dataset here so it is in shared memory.

.. code-block:: python

   if __name__ == '__main__':
       dataset = Reddit('../../data/Reddit')

       world_size = torch.cuda.device_count()
       print('Let\'s use', world_size, 'GPUs!')
       mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
