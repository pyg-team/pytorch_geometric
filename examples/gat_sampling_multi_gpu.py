import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from torch.nn import Linear as Lin
from tqdm import tqdm
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Reddit
# from utils import thread_wrapped_func


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels,heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        # residual
        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(in_channels, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)

                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


@torch.no_grad()
def test(model, x, y, data, device, subgraph_loader):
    model.eval()

    out = model.inference(x, device, subgraph_loader)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


def run(proc_id, n_gpus, devices, dataset, args):
    data = dataset[0]

    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12346')
        world_size = n_gpus
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=proc_id)
    torch.cuda.set_device(dev_id)

    train_nid = torch.LongTensor(np.nonzero(data.train_mask))
    # Split trian_id
    train_nid = torch.split(train_nid, len(train_nid) // n_gpus)[proc_id]

    train_loader = NeighborSampler(data.edge_index, node_idx=train_nid.view(-1),
                                   sizes=list(map(int, args.sample_size.split(','))),
                                   batch_size=args.train_batch_size,
                                   shuffle=True, num_workers=args.num_workers)

    model = GAT(dataset.num_features, args.num_hidden, dataset.num_classes, num_layers=args.num_layers,
                heads=args.num_heads)
    model = model.to(dev_id)
    model.reset_parameters()

    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    x = data.x.to(dev_id)
    y = data.y.squeeze().to(dev_id)

    for epoch in range(1, args.num_epochs):
        tic = time.time()
        device = dev_id
        model.train()

        step = 0
        for batch_size, n_id, adjs in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]

            optimizer.zero_grad()
            out = model(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            if n_gpus > 1:
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        torch.distributed.all_reduce(param.grad.data,
                                                     op=torch.distributed.ReduceOp.SUM)
                        param.grad.data /= n_gpus
            optimizer.step()

            if step % args.log_every == 0 and proc_id == 0:
                pred_correct = int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
                cur_acc = float(pred_correct) / batch_size
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print(
                    'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MiB'.format(
                        epoch, step, loss.item(), cur_acc, gpu_mem_alloc))

            step = step + 1

        if n_gpus > 1:
            torch.distributed.barrier()

        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))

        # eval only use proc0
        if epoch % args.eval_every == 0 and epoch != 0 and proc_id == 0:
            subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                              batch_size=args.eval_batch_size, shuffle=False,
                                              num_workers=args.num_workers)
            if n_gpus == 1:
                train_acc, val_acc, test_acc = test(model, x, y, data, devices[0], subgraph_loader)
            else:
                train_acc, val_acc, test_acc = test(model.module, x, y, data, devices[0], subgraph_loader)
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

        if n_gpus > 1:
            torch.distributed.barrier()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0,1,2',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--num-epochs', type=int, default=31)
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--sample-size', type=str, default='10,25')
    argparser.add_argument('--num-heads', type=int, default=4)
    argparser.add_argument('--train-batch-size', type=int, default=1024)
    argparser.add_argument('--eval-batch-size', type=int, default=1024)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=int, default=0.001)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--num-workers', type=int, default=2,
                           help="Number of sampling processes. Use 0 for no extra process.")

    args = argparser.parse_args()

    process_start_time = time.time()

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    # change for local data
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    load_data_start_time = time.time()
    dataset = Reddit(path)
    load_data_end_time = time.time()
    print("pyG load reddit data time: {:.4f} s".format(load_data_end_time - load_data_start_time))

    if n_gpus == 1:
        run(0, n_gpus, devices, dataset, args)
    else:
        mp.spawn(run, args=(n_gpus, devices, dataset, args), nprocs=n_gpus, join=True)