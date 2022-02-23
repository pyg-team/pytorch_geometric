# This script shows how to use Quiver in an existing PyG example:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
import os.path as osp

import quiver
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]

train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

################################
# Step 1: Using Quiver's sampler
################################

train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024,
                                           shuffle=True,
                                           drop_last=True)  # Quiver
########################################################################
# The below code enable Quiver for PyG.
# Please refer to: https://torch-quiver.readthedocs.io/en/latest/api/ for
# how to configure the CSRTopo, Sampler and Feature of Quiver.
########################################################################
csr_topo = quiver.CSRTopo(data.edge_index)  # Quiver
quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[25, 10],
                                             device=0, mode='GPU')  # Quiver

subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=12)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

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
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, 256, dataset.num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

################################
# Step 2: Using Quiver's Feature
################################
x = quiver.Feature(rank=0, device_list=[0], device_cache_size="4G",
                   cache_policy="device_replicate",
                   csr_topo=csr_topo)  # Quiver
x.from_cpu_tensor(data.x)  # Quiver

y = data.y.squeeze().to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    ############################################
    # Step 3: Training the PyG Model with Quiver
    ############################################
    # for batch_size, n_id, adjs in train_loader: # Original PyG Code
    for seeds in train_loader:  # Quiver
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)  # Quiver
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


for epoch in range(1, 11):
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    train_acc, val_acc, test_acc = test()
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
