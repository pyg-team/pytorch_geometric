# This script shows how to use Quiver in an existing PyG example:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py
import os

import quiver
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def run(rank, world_size, dataset, quiver_feature, quiver_sampler):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024,
                                               shuffle=True, num_workers=0)

    if rank == 0:
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None,
                                          sizes=[-1], batch_size=2048,
                                          shuffle=False, num_workers=6)

    torch.manual_seed(12345)
    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    y = data.y.to(rank)

    for epoch in range(1, 21):
        model.train()

        for seeds in train_loader:
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            adjs = [adj.to(rank) for adj in adjs]

            optimizer.zero_grad()
            out = model(quiver_feature[n_id].to(rank), adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

        dist.barrier()

        if rank == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                out = model.module.inference(quiver_feature, rank,
                                             subgraph_loader)
            res = out.argmax(dim=-1) == data.y
            acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
            acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
            acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    dataset = Reddit('../../data/Reddit')
    data = dataset[0]
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')

    ########################################################################
    # The below code enable Quiver for PyG.
    # Please refer to: https://torch-quiver.readthedocs.io/en/latest/api/ for
    # how to configure the CSRTopo, Sampler and Feature of Quiver.
    ########################################################################
    csr_topo = quiver.CSRTopo(data.edge_index)  # Quiver
    quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [25, 10], 0,
                                                 mode='GPU')  # Quiver
    quiver_feature = quiver.Feature(rank=0,
                                    device_list=list(range(world_size)),
                                    device_cache_size="2G",
                                    cache_policy="device_replicate",
                                    csr_topo=csr_topo)  # Quiver
    quiver_feature.from_cpu_tensor(data.x)  # Quiver

    mp.spawn(run, args=(world_size, dataset, quiver_feature, quiver_sampler),
             nprocs=world_size, join=True)
