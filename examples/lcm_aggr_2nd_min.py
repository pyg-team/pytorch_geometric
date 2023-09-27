# Final validation accuracy: ~95%
import argparse

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import LCMAggregation
from torch_geometric.transforms import BaseTransform

parser = argparse.ArgumentParser()
parser.add_argument('--num_bits', type=int, default=8)
args = parser.parse_args()


class RandomPermutation(BaseTransform):
    def forward(self, data: Data) -> Data:
        data.x = torch.x[torch.randperm(data.x.size(0))]
        return data


class Random2ndMinimumDataset(InMemoryDataset):
    r""""A labeled dataset, where each sample is a multiset of integers
    encoded as bit-vectors, and the label is the second smallest integer
    in the multiset."""
    def __init__(
        self,
        num_examples: int,
        num_bits: int,
        min_num_elems: int,
        max_num_elems: int,
    ):
        super().__init__(transform=RandomPermutation())

        self.data, self.slices = self.collate([
            self.get_data(num_bits, min_num_elems, max_num_elems)
            for _ in range(num_examples)
        ])

    def get_data(
        self,
        num_bits: int,
        min_num_elems: int,
        max_num_elems: int,
    ) -> Data:

        num_elems = int(torch.randint(min_num_elems, max_num_elems + 1, (1, )))

        x = torch.randint(0, 2, (num_elems, num_bits))

        power = torch.pow(2, torch.arange(num_bits)).flip([0])
        ints = (x * power.view(1, -1)).sum(dim=-1)
        y = x[ints.topk(k=2, largest=False).indices[-1:]].to(torch.float)

        return Data(x=x, y=y)


train_dataset = Random2ndMinimumDataset(
    num_examples=2**16,  # 65,536
    num_bits=args.num_bits,
    min_num_elems=2,
    max_num_elems=16,
)
# Validate on multi sets of size 32, larger than observed during training:
val_dataset = Random2ndMinimumDataset(
    num_examples=2**10,  # 1024
    num_bits=args.num_bits,
    min_num_elems=32,
    max_num_elems=32,
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)


class BitwiseEmbedding(torch.nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.embs = torch.nn.ModuleList(
            [torch.nn.Embedding(2, emb_dim) for _ in range(args.num_bits)])

    def forward(self, x: Tensor) -> Tensor:
        xs = [emb(b) for emb, b in zip(self.embs, x.t())]
        return torch.stack(xs, dim=0).sum(0)


class LCM(torch.nn.Module):
    def __init__(self, emb_dim: int, dropout: float = 0.25):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            BitwiseEmbedding(emb_dim),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.Dropout(),
            torch.nn.GELU(),
        )

        self.aggr = LCMAggregation(emb_dim, emb_dim, project=False)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.Dropout(dropout),
            torch.nn.GELU(),
            torch.nn.Linear(emb_dim, args.num_bits),
        )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.aggr(x, batch)
        x = self.decoder(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LCM(emb_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train():
    total_loss = total_examples = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += batch.num_graphs * float(loss)
        total_examples += batch.num_graphs
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    total_correct = total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.batch).sigmoid().round()
        num_mistakes = (pred != batch.y).sum(dim=-1)
        total_correct += int((num_mistakes == 0).sum())
        total_examples += batch.num_graphs
    return total_correct / total_examples


for epoch in range(1, 1001):
    loss = train()
    val_acc = test(val_loader)
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
