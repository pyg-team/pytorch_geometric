import os
import torch.distributed as dist
import tqdm
import torch
def run(rank, world_size, dataset):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    data = dataset[0]
    data = data.to(rank, 'x', 'y')
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]


    kwargs = dict(batch_size=1024, num_workers=4, persistent_workers=True)
    from torch_geometric.loader import NeighborLoader
    train_loader = NeighborLoader(data, input_nodes=train_idx,
                                 num_neighbors=[25, 10], shuffle=True,
                                 drop_last=True, **kwargs)
    if rank == 0:
        val_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        val_loader = NeighborLoader(data, num_neighbors=[25, 10], input_nodes=val_idx, shuffle=False)
    from torch.nn.parallel import DistributedDataParallel
    torch.manual_seed(12345)
    from torch_geometric.nn.models import GraphSAGE
    model = GraphSAGE(in_channels=dataset.num_features,
            hidden_channels=256,
            num_layers=2,
            out_channels=dataset.num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    import torch.nn.functional as F
    for epoch in range(1, 21):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index.to(rank))[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
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
                    correct += (out == batch.y[:batch.batch_size].to(out.device)).sum()
                    count += batch.batch_size
            print(f'Val Accuracy: {correct/count:.4f}')

        dist.barrier()

    dist.destroy_process_group()
if __name__ == '__main__':
    from torch_geometric.datasets import Reddit
    import torch.multiprocessing as mp
    dataset = Reddit('../../data/Reddit')

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
