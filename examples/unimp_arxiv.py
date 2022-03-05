import os.path as osp
import time

import torch
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

import torch_geometric.transforms as T
from torch_geometric.nn import TransformerConv

root = osp.join("..", "data", "OGB")
dataset = PygNodePropPredDataset(
    "ogbn-arxiv",
    root,
    transform=T.Compose([
        T.ToUndirected(),
        T.ToSparseTensor(),
    ]),
)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name="ogbn-arxiv")
data = dataset[0]


class UnimpNet(torch.nn.Module):
    def __init__(self, feature_size: int, label_vocab: int,
                 transformer_dim: int, heads: int):
        super().__init__()

        self.label_embedding = torch.nn.Embedding(label_vocab, feature_size)
        self.conv = TransformerConv(feature_size, transformer_dim // heads,
                                    heads=heads)
        self.linear = torch.nn.Linear(transformer_dim, label_vocab)

    def forward(self, x, y: torch.Tensor, edge_index):
        y_embedding = self.label_embedding(y.squeeze())
        x_y = x + y_embedding
        return self.linear(self.conv(x_y, edge_index))


row, col, edge_attr = data.adj_t.t().coo()
edge_index = torch.stack([row, col], dim=0)

model = UnimpNet(128, 40, 16, 4)
optim = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98),
                         eps=1e-9)


def train_model(epochs, label_rate=0.5):

    model.train()

    start = time.time()

    total_loss = 0
    y = torch.zeros(data.x.shape[0]).type(torch.long)

    for epoch in range(epochs):

        mask = torch.rand(data.x.shape[0]) < label_rate

        y[mask] = data.y.squeeze()[mask]
        y[~mask] = 0

        out = model(data.x, data.y, edge_index)
        optim.zero_grad()
        loss = torch.nn.functional.cross_entropy(out[mask], y[mask])
        loss.backward()
        optim.step()

        total_loss += loss
        print(f"time = {(time.time() - start) // 60}")
        print(f"epoch = {epoch + 1}")
        print(f"loss = {loss}")

    with torch.no_grad():
        out = model(data.x, data.y, edge_index)
        optim.zero_grad()
        loss = torch.nn.functional.cross_entropy(out[mask], data.y)


train_model(20)
