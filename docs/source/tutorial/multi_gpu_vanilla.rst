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

(TODO) break into chunks and describe

.. code-block:: python

   # An Multi GPU implementation of unsupervised bipartite GraphSAGE
   # using the Alibaba Taobao dataset.
   import os
   import os.path as osp

   import torch
   import torch.distributed as dist
   import torch.multiprocessing as mp
   import torch.nn.functional as F
   import tqdm
   from sklearn.metrics import roc_auc_score
   from torch.nn import Embedding, Linear
   from torch.nn.parallel import DistributedDataParallel

   import torch_geometric.transforms as T
   from torch_geometric.datasets import Taobao
   from torch_geometric.loader import LinkNeighborLoader
   from torch_geometric.nn import SAGEConv
   from torch_geometric.utils.convert import to_scipy_sparse_matrix


   class ItemGNNEncoder(torch.nn.Module):
       def __init__(self, hidden_channels, out_channels):
           super().__init__()
           self.conv1 = SAGEConv(-1, hidden_channels)
           self.conv2 = SAGEConv(hidden_channels, hidden_channels)
           self.lin = Linear(hidden_channels, out_channels)

       def forward(self, x, edge_index):
           x = self.conv1(x, edge_index).relu()
           x = self.conv2(x, edge_index).relu()
           return self.lin(x)


   class UserGNNEncoder(torch.nn.Module):
       def __init__(self, hidden_channels, out_channels):
           super().__init__()
           self.conv1 = SAGEConv((-1, -1), hidden_channels)
           self.conv2 = SAGEConv((-1, -1), hidden_channels)
           self.conv3 = SAGEConv((-1, -1), hidden_channels)
           self.lin = Linear(hidden_channels, out_channels)

       def forward(self, x_dict, edge_index_dict):
           item_x = self.conv1(
               x_dict['item'],
               edge_index_dict[('item', 'to', 'item')],
           ).relu()

           user_x = self.conv2(
               (x_dict['item'], x_dict['user']),
               edge_index_dict[('item', 'rev_to', 'user')],
           ).relu()

           user_x = self.conv3(
               (item_x, user_x),
               edge_index_dict[('item', 'rev_to', 'user')],
           ).relu()

           return self.lin(user_x)


   class EdgeDecoder(torch.nn.Module):
       def __init__(self, hidden_channels):
           super().__init__()
           self.lin1 = Linear(2 * hidden_channels, hidden_channels)
           self.lin2 = Linear(hidden_channels, 1)

       def forward(self, z_src, z_dst, edge_label_index):
           row, col = edge_label_index
           z = torch.cat([z_src[row], z_dst[col]], dim=-1)

           z = self.lin1(z).relu()
           z = self.lin2(z)
           return z.view(-1)


   class Model(torch.nn.Module):
       def __init__(self, num_users, num_items, hidden_channels, out_channels):
           super().__init__()
           self.user_emb = Embedding(num_users, hidden_channels)
           self.item_emb = Embedding(num_items, hidden_channels)
           self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
           self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
           self.decoder = EdgeDecoder(out_channels)

       def forward(self, x_dict, edge_index_dict, edge_label_index):
           z_dict = {}
           x_dict['user'] = self.user_emb(x_dict['user'])
           x_dict['item'] = self.item_emb(x_dict['item'])
           z_dict['item'] = self.item_encoder(
               x_dict['item'],
               edge_index_dict[('item', 'to', 'item')],
           )
           z_dict['user'] = self.user_encoder(x_dict, edge_index_dict)

           return self.decoder(z_dict['user'], z_dict['item'], edge_label_index)


   def run_train(rank, data, train_data, val_data, test_data, world_size):
       if rank == 0:
           print("Setting up Data Loaders...")
       train_edge_label_idx = train_data[('user', 'to', 'item')].edge_label_index
       train_edge_label_idx = train_edge_label_idx.split(
           train_edge_label_idx.size(1) // world_size, dim=1)[rank].clone()
       train_loader = LinkNeighborLoader(
           data=train_data,
           num_neighbors=[8, 4],
           edge_label_index=(('user', 'to', 'item'), train_edge_label_idx),
           neg_sampling='binary',
           batch_size=2048,
           shuffle=True,
           num_workers=16,
           drop_last=True,
       )

       val_loader = LinkNeighborLoader(
           data=val_data,
           num_neighbors=[8, 4],
           edge_label_index=(
               ('user', 'to', 'item'),
               val_data[('user', 'to', 'item')].edge_label_index,
           ),
           edge_label=val_data[('user', 'to', 'item')].edge_label,
           batch_size=2048,
           shuffle=False,
           num_workers=16,
       )

       test_loader = LinkNeighborLoader(
           data=test_data,
           num_neighbors=[8, 4],
           edge_label_index=(
               ('user', 'to', 'item'),
               test_data[('user', 'to', 'item')].edge_label_index,
           ),
           edge_label=test_data[('user', 'to', 'item')].edge_label,
           batch_size=2048,
           shuffle=False,
           num_workers=16,
       )

       def train():
           model.train()

           total_loss = total_examples = 0
           for batch in tqdm.tqdm(train_loader):
               batch = batch.to(rank)
               optimizer.zero_grad()

               pred = model(
                   batch.x_dict,
                   batch.edge_index_dict,
                   batch['user', 'item'].edge_label_index,
               )
               loss = F.binary_cross_entropy_with_logits(
                   pred, batch['user', 'item'].edge_label)

               loss.backward()
               optimizer.step()
               total_loss += float(loss)
               total_examples += pred.numel()

           return total_loss / total_examples

       @torch.no_grad()
       def test(loader):
           model.eval()

           preds, targets = [], []
           for batch in tqdm.tqdm(loader):
               batch = batch.to(rank)

               pred = model(
                   batch.x_dict,
                   batch.edge_index_dict,
                   batch['user', 'item'].edge_label_index,
               ).sigmoid().view(-1).cpu()
               target = batch['user', 'item'].edge_label.long().cpu()

               preds.append(pred)
               targets.append(target)

           pred = torch.cat(preds, dim=0).numpy()
           target = torch.cat(targets, dim=0).numpy()

           return roc_auc_score(target, pred)

       os.environ['MASTER_ADDR'] = 'localhost'
       os.environ['MASTER_PORT'] = '12355'
       dist.init_process_group('nccl', rank=rank, world_size=world_size)
       model = Model(
           num_users=data['user'].num_nodes,
           num_items=data['item'].num_nodes,
           hidden_channels=64,
           out_channels=64,
       ).to(rank)
       # Initialize lazy modules
       for batch in train_loader:
           batch = batch.to(rank)
           _ = model(
               batch.x_dict,
               batch.edge_index_dict,
               batch['user', 'item'].edge_label_index,
           )
           break
       model = DistributedDataParallel(model, device_ids=[rank])
       optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
       for epoch in range(1, 21):
           loss = train()
           if rank == 0:
               val_auc = test(val_loader)
               test_auc = test(test_loader)
           if rank == 0:
               print(f'Epoch: {epoch:02d}, Loss: {loss:4f}, Val: {val_auc:.4f}, '
                     f'Test: {test_auc:.4f}')


   if __name__ == '__main__':
       path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/Taobao')

       dataset = Taobao(path)
       data = dataset[0]

       data['user'].x = torch.arange(0, data['user'].num_nodes)
       data['item'].x = torch.arange(0, data['item'].num_nodes)

       # Only consider user<>item relationships for simplicity:
       del data['category']
       del data['item', 'category']
       del data['user', 'item'].time
       del data['user', 'item'].behavior

       # Add a reverse ('item', 'rev_to', 'user') relation for message passing:
       data = T.ToUndirected()(data)

       # Perform a link-level split into training, validation, and test edges:
       print('Computing data splits...')
       train_data, val_data, test_data = T.RandomLinkSplit(
           num_val=0.1,
           num_test=0.1,
           neg_sampling_ratio=1.0,
           add_negative_train_samples=False,
           edge_types=[('user', 'to', 'item')],
           rev_edge_types=[('item', 'rev_to', 'user')],
       )(data)
       print('Done!')

       # Compute sparsified item<>item relationships through users:
       print('Computing item<>item relationships...')
       mat = to_scipy_sparse_matrix(data['user', 'item'].edge_index).tocsr()
       mat = mat[:data['user'].num_nodes, :data['item'].num_nodes]
       comat = mat.T @ mat
       comat.setdiag(0)
       comat = comat >= 3.
       comat = comat.tocoo()
       row = torch.from_numpy(comat.row).to(torch.long)
       col = torch.from_numpy(comat.col).to(torch.long)
       item_to_item_edge_index = torch.stack([row, col], dim=0)

       # Add the generated item<>item relationships for high-order information:
       train_data['item', 'item'].edge_index = item_to_item_edge_index
       val_data['item', 'item'].edge_index = item_to_item_edge_index
       test_data['item', 'item'].edge_index = item_to_item_edge_index
       print('Done!')

       world_size = torch.cuda.device_count()
       print('Let\'s use', world_size, 'GPUs!')
       mp.spawn(run_train,
                args=(data, train_data, val_data, test_data, world_size),
                nprocs=world_size, join=True)
