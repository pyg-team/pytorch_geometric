Multi-GPU Training in Vanilla PyTorch
=====================================
For many large scale, real-world datasets, it may be necessary to scale up training across multiple GPUs. This tutorial goes over how to do this on a small scale dataset and then
shows how the same technique can be applied to a large scale recommendation graph.
A large set of real-world datasets are stored as heterogeneous graphs, motivating the introduction of specialized functionality for them in :pyg:`PyG`.

Starting off small
------------------

To start, we can take a look at the `distributed_sampling <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py>`__ example from PyG:
This example shows how to use train a GraphSage GNN Model on the `Reddit dataset <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Reddit.html>`__. This example uses NeighborLoader with `torch.data.DistributedDataParallel <https://pytorch.org/docs/stable/notes/ddp.html>`__ to scale up training across all available GPU's on your machine.




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

Now we split training indices into `world_size` many chunks for each GPU:

.. code-block:: python

       train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
       train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

       kwargs = dict(batch_size=1024, num_workers=4, persistent_workers=True)
       train_loader = NeighborLoader(data, input_nodes=train_idx,
                                     num_neighbors=[25, 10], shuffle=True,
                                     drop_last=True, **kwargs)

We also create a single-hop evaluation neighbor loader:

.. code-block:: python

       if rank == 0:
           subgraph_loader = NeighborLoader(copy.copy(data), num_neighbors=[-1],
                                            shuffle=False, **kwargs)
           # No need to maintain these features during evaluation:
           del subgraph_loader.data.x, subgraph_loader.data.y
           # Add global node index information:
           subgraph_loader.data.node_id = torch.arange(data.num_nodes)

Now that we have our data loaders defined initialize our model and wrap it in DistributedDataParallel

.. code-block:: python

      from torch.nn.parallel import DistributedDataParallel
      torch.manual_seed(12345)
      model = SAGE(dataset.num_features, 256, dataset.num_classes).to(rank)
      model = DistributedDataParallel(model, device_ids=[rank])

Now we set up our optimizer and define our training loop. Notice that we move the edge indices of each mini batch to GPU while the features and labels are already on GPU.

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


Putting it all together, we spawn our runners for each GPU:

.. code-block:: python

   if __name__ == '__main__':
       dataset = Reddit('../../data/Reddit')

       world_size = torch.cuda.device_count()
       print('Let\'s use', world_size, 'GPUs!')
       mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)

Large Scale
-----------
Now that we have a working small scale example, lets try to scale up to a larger dataset, papers100m.

The first step is defining the necessary imports:

.. code-block:: python

   import argparse
   import os
   import time

   import torch
   import torch.distributed as dist
   import torch.multiprocessing as mp
   import torch.nn.functional as F
   from ogb.nodeproppred import PygNodePropPredDataset
   from torch.nn.parallel import DistributedDataParallel
   from torchmetrics import Accuracy

   from torch_geometric.loader import NeighborLoader
   from torch_geometric.nn import GCNConv


Next we define our GCN model:

.. code-block:: python

   class GCN(torch.nn.Module):
       def __init__(self, in_channels, hidden_channels, out_channels):
           super().__init__()
           self.conv1 = GCNConv(in_channels, hidden_channels)
           self.conv2 = GCNConv(hidden_channels, out_channels)

       def forward(self, x, edge_index, edge_weight=None):
           x = F.dropout(x, p=0.5, training=self.training)
           x = self.conv1(x, edge_index, edge_weight).relu()
           x = F.dropout(x, p=0.5, training=self.training)
           x = self.conv2(x, edge_index, edge_weight)
           return x


Similarly to our last example we now set up our spawnable runner:

.. code-block:: python

   def pyg_num_work():
       num_work = None
       if hasattr(os, "sched_getaffinity"):
           try:
               num_work = len(os.sched_getaffinity(0)) / 2
   puririshi98 marked this conversation as resolved.
   puririshi98 marked this conversation as resolved.
           except Exception:
               pass
       if num_work is None:
           num_work = os.cpu_count() / 2
       return int(num_work)

   def run_train(rank, data, world_size, model, epochs, batch_size, fan_out,
                 split_idx, num_classes):
       os.environ['MASTER_ADDR'] = 'localhost'
       os.environ['MASTER_PORT'] = '12355'
       dist.init_process_group('nccl', rank=rank, world_size=world_size)
       split_idx['train'] = split_idx['train'].split(
           split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
       model = model.to(rank)
       model = DistributedDataParallel(model, device_ids=[rank])
       optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                    weight_decay=0.0005)
       train_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                     input_nodes=split_idx['train'],
                                     batch_size=batch_size,
                                     num_workers=pyg_num_work())
       if rank == 0:
           eval_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                        input_nodes=split_idx['valid'],
                                        batch_size=batch_size,
                                        num_workers=pyg_num_work())
           test_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                        input_nodes=split_idx['test'],
                                        batch_size=batch_size,
                                        num_workers=pyg_num_work())
       eval_steps = 100
       acc = Accuracy(task="multiclass", num_classes=num_classes).to(rank)
       if rank == 0:
           print("Beginning training...")
       for epoch in range(epochs):
           for i, batch in enumerate(train_loader):
               if i >= 10:
                   start = time.time()
               batch = batch.to(rank)
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
                       batch = batch.to(rank)
                       batch.y = batch.y.to(torch.long)
                       out = model(batch.x, batch.edge_index)
                       acc_sum += acc(out[:batch_size].softmax(dim=-1),
                                      batch.y[:batch_size])
               print(f"Validation Accuracy: {acc_sum/(i) * 100.0:.4f}%", )
               print("Average Inference Iteration Time:",
                     (time.time() - start) / (i - 10), "s/iter")
       if rank == 0:
           acc_sum = 0.0
           with torch.no_grad():
               for i, batch in enumerate(test_loader):
                   batch = batch.to(rank)
                   batch.y = batch.y.to(torch.long)
                   out = model(batch.x, batch.edge_index)
                   acc_sum += acc(out[:batch_size].softmax(dim=-1),
                                  batch.y[:batch_size])
               print(f"Test Accuracy: {acc_sum/(i) * 100.0:.4f}%", )


Again, like in our last example, we put it all together by spawning our runners for each GPU:

.. code-block:: python

   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       parser.add_argument('--hidden_channels', type=int, default=64)
       parser.add_argument('--lr', type=float, default=0.01)
       parser.add_argument('--epochs', type=int, default=3)
       parser.add_argument('--batch_size', type=int, default=128)
       parser.add_argument('--fan_out', type=int, default=50)

       args = parser.parse_args()

       dataset = PygNodePropPredDataset(name='ogbn-papers100M')
       split_idx = dataset.get_idx_split()
       data = dataset[0]
       data.y = data.y.reshape(-1)
       model = GCN(dataset.num_features, args.hidden_channels,
                   dataset.num_classes)
       print("Data =", data)
       world_size = torch.cuda.device_count()
       print('Let\'s use', world_size, 'GPUs!')
       mp.spawn(
           run_train, args=(data, world_size, model, args.epochs, args.batch_size,
                            args.fan_out, split_idx, dataset.num_classes),
           nprocs=world_size, join=True)
