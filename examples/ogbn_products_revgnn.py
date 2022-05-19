import os.path as osp

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch import nn
from torch.nn import LayerNorm
from torch_sparse import SparseTensor
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.loader import RandomNodeSampler
from torch_geometric.nn import GroupAddRev, SAGEConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.profile.utils import count_parameters
from torch_geometric.utils import index_to_mask


class SharedDropout(nn.Module):
    """Dropout with interface to set mask."""
    def __init__(self):
        super().__init__()
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        if self.training:
            assert self.mask is not None
            out = x * self.mask
            return out
        return x


class BasicBlock(nn.Module):
    """Pre-activation GNN block proposed in DeeperGCN."""
    def __init__(self, in_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)
        self.dropout = SharedDropout()
        self.gnn = None

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x, edge_index, dropout_mask=None):
        out = self.norm(x)
        out = F.relu(out)

        if dropout_mask is not None:
            self.dropout.set_mask(dropout_mask)
            out = self.dropout(out)

        out = self.gnn(out, edge_index)

        return out


class GNNBlock(BasicBlock):
    def __init__(self, in_channels, out_channels, aggr="mean"):
        super().__init__(in_channels)
        self.gnn = SAGEConv(in_channels, out_channels, aggr=aggr)


class RevGNN(torch.nn.Module):
    """Group Reversible GNN proposed in GNN1000:
    https://arxiv.org/abs/2106.07476."""
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        num_groups=2,
        aggr="mean",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.gnns = torch.nn.ModuleList()
        self.last_norm = LayerNorm(hidden_channels, elementwise_affine=True)

        self.node_features_encoder = Linear(in_channels, hidden_channels,
                                            bias=True,
                                            weight_initializer="glorot")
        self.node_pred_linear = Linear(
            hidden_channels,
            out_channels,
            bias=True,
            weight_initializer="glorot",
        )

        assert (hidden_channels // num_groups) * num_groups == hidden_channels
        for _ in range(self.num_layers):
            seed_gnn = GNNBlock(
                hidden_channels // num_groups,
                hidden_channels // num_groups,
                aggr,
            )
            gnn = GroupAddRev(seed_gnn, num_groups=num_groups)

            self.gnns.append(gnn)

    def reset_parameters(self):
        for gnn in self.gnns:
            gnn.reset_parameters()
        self.last_norm.reset_parameters()
        self.node_features_encoder.reset_parameters()
        self.node_pred_linear.reset_parameters()

    def forward(self, x, edge_index):
        h = self.node_features_encoder(x)

        # Generate a dropout mask which will be shared across GNN blocks
        if self.dropout != 0.:
            m = torch.zeros_like(h).bernoulli_(1 - self.dropout)
            mask = m.requires_grad_(False) / (1 - self.dropout)
        else:
            mask = None

        for layer in range(self.num_layers):
            h = self.gnns[layer](h, edge_index, mask)

        h = F.relu(self.last_norm(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.node_pred_linear(h)

        return torch.log_softmax(h, dim=-1)


root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "products")
dataset = PygNodePropPredDataset("ogbn-products", root,
                                 transform=T.AddSelfLoops())
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name="ogbn-products")
data = dataset[0]
data.y.squeeze_()

# Set split indices to masks.
for split in ["train", "valid", "test"]:
    data[f"{split}_mask"] = index_to_mask(split_idx[split],
                                          size=data.y.shape[0])

train_loader = RandomNodeSampler(data, num_parts=10, shuffle=True,
                                 num_workers=5)
# Increase the num_parts of the test loader if you cannot have fix
# the full batch graph into your GPU.
test_loader = RandomNodeSampler(data, num_parts=1, num_workers=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 160
n_layers = 7  # You can try 1000 layers for fun
model = RevGNN(
    dataset.num_features,
    hidden_size,
    dataset.num_classes,
    n_layers,
    0.5,
    num_groups=2,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
criterion = F.nll_loss
evaluator = Evaluator("ogbn-products")


def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f"Training epoch: {epoch:04d}")

    total_loss = total_examples = total_correct = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        # Memory-efficient aggregations
        adj_t = SparseTensor(row=data.edge_index[0],
                             col=data.edge_index[1]).t()
        out = model(data.x, adj_t)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())
        total_correct += int(out[data.train_mask].argmax(dim=-1).eq(
            data.y[data.train_mask]).sum())
        pbar.update(1)

    pbar.close()

    loss = total_loss / total_examples
    approx_acc = total_correct / total_examples
    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    y_true = {"train": [], "valid": [], "test": []}
    y_pred = {"train": [], "valid": [], "test": []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    for data in test_loader:
        data = data.to(device)
        # Memory-efficient aggregations
        adj_t = SparseTensor(row=data.edge_index[0],
                             col=data.edge_index[1]).t()
        out = model(data.x, adj_t)
        out = out.argmax(dim=-1, keepdim=True)

        for split in y_true.keys():
            mask = data[f"{split}_mask"]
            y_true[split].append(data.y[mask].unsqueeze(-1).cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_acc = evaluator.eval({
        "y_true": torch.cat(y_true["train"], dim=0),
        "y_pred": torch.cat(y_pred["train"], dim=0),
    })["acc"]

    valid_acc = evaluator.eval({
        "y_true": torch.cat(y_true["valid"], dim=0),
        "y_pred": torch.cat(y_pred["valid"], dim=0),
    })["acc"]

    test_acc = evaluator.eval({
        "y_true": torch.cat(y_true["test"], dim=0),
        "y_pred": torch.cat(y_pred["test"], dim=0),
    })["acc"]

    return train_acc, valid_acc, test_acc


def gpu_mem_usage():
    """Compute the GPU memory usage for the current device (GB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024**3


model.reset_parameters()
print(f"Model Paramters: {count_parameters(model)}")
highest_val = 0.0
final_test = 0.0
for epoch in range(1, 501):
    loss, approx_acc = train(epoch)
    if epoch == 1:
        print(f"Peak GPU memory usage: {gpu_mem_usage():.2f} G")
    train_acc, valid_acc, test_acc = test()
    torch.cuda.empty_cache()
    if valid_acc > highest_val:
        highest_val = valid_acc
        final_train = train_acc
        final_test = test_acc
    print(
        f"Loss: {loss:.4f}, Approx_acc: {approx_acc:.4f}",
        f"Train: {train_acc:.4f}, Val: {valid_acc:.4f}, Test: {test_acc:.4f}",
        f"Final Train: {final_train:.4f}, Highest Val: {highest_val:.4f}, \
            Final Test: {final_test:.4f}",
    )

# Model Paramters: 206607
# Peak GPU memory usage: 1.57 G
# RevGNN with 7 layers and 160 channels reaches around 0.8200 test accuracy.
# Final Train: 0.9373, Highest Val: 0.9230, Final Test: 0.8200.
# Training longer should produces better results.
