Multi-GPU training in Vanilla PyTorch
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

Defining our Spawnable Trainer
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



Large Scale
-----------

