import os
import torch
import logging

import torch_geometric.transforms as T
from torch_geometric.nn import TransformerConv
from ogb.nodeproppred import PygNodePropPredDataset

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

# Get dataset
root = os.path.join("..", "data", "OGB")
dataset = PygNodePropPredDataset(
    "ogbn-arxiv",
    root,
    transform=T.Compose([T.ToUndirected()]),
)
data = dataset[0]
num_classes = 40

# Form masks for labels: mask gives labels being predicted
split_idx = dataset.get_idx_split()
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[split_idx["train"]] = True

valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
valid_mask[split_idx["valid"]] = True

test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask[split_idx["test"]] = True

# Model parameters
inner_dim = 16
heads = 2


def ratio_mask(mask: torch.Tensor, ratio: float, shuffle: bool = False):
    r"""Modifies the existing :obj:`mask` by additioning setting :obj:`ratio` of
    the :obj:`True` entries to :obj:`False`. Does not operate inplace.

    If shuffle is required the masking proportion is not exact.

    Args:
        mask (torch.Tensor): The mask to re-mask.
        ratio (float): The ratio of entries to remove.
        shuffle (bool): Whether or not the mask is pre-shuffled, if so there is
            no need to randomize which entires are set to :obj:`False`.
    """
    n = int(mask.count_nonzero().item())
    new_mask = torch.ones(len(mask), dtype=torch.bool)
    if not shuffle:
        new_mask[mask] = (torch.rand(n) < ratio).type(torch.bool)
    else:
        new_mask[mask] = (torch.arange(0, n) < int(ratio * n)).type(torch.bool)
    return mask & new_mask


class MaskLabel(torch.nn.Module):
    r"""A label embedding layer that replicates the label masking from `"Masked
    Label Prediction: Unified Message Passing Model for Semi-Supervised
    Classification" <https://arxiv.org/abs/2009.03509>`_ paper.

    In the forward pass both the labels, and a mask corresponding to which
    labels should be kept is provided. All entires that are not true in the mask
    are returned as zero.

    Args:
        num_classes (int): Size of the number of classes for the labels
        out_channels (int): Size of each output sample.
    """

    def __init__(self, num_classes, out_channels):
        super().__init__()

        self.emb = torch.nn.Embedding(num_classes, out_channels)
        self.out_channels = out_channels

    def forward(self, y: torch.Tensor, mask: torch.Tensor):
        out = torch.zeros(y.shape[0], self.out_channels, dtype=torch.float)
        out[mask] = self.emb(y[mask])
        return out


class UnimpNet(torch.nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        inner_dim: int,
        heads: int
    ):
        super().__init__()

        self.label_embedding = MaskLabel(num_classes, feature_size)
        self.conv = TransformerConv(
            feature_size,
            inner_dim // heads,
            heads=heads)
        self.linear = torch.nn.Linear(inner_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        label_mask: torch.Tensor,
    ):
        x = x + self.label_embedding(y, label_mask)
        x = self.conv(x, edge_index)
        out = self.linear(x)
        return out

    def predictions(self, out: torch.Tensor):
        probs = torch.nn.functional.softmax(out, dim=1)
        return probs.argmax(dim=1)

    def loss(self, out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        return torch.nn.functional.cross_entropy(out[mask], labels[mask])

    def accuracy(
        self,
        out:
        torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ):
        return (
            (self.predictions(out[mask]) == labels[mask]).sum().float()
            / float(labels[mask].size(0))
        )


def train(model, optim, epochs=20, label_rate=0.9):

    y = data.y.squeeze()
    for epoch in range(epochs):
        model.train()

        # create epoc training mask that chooses subset of train to remove
        epoc_mask = torch.rand(data.x.shape[0]) > label_rate

        # label mask is a mask to give what labels are allowed
        label_mask = ~(epoc_mask | test_mask | valid_mask)

        # forward pass
        out_train = model(data.x, data.y.squeeze(), data.edge_index, label_mask)

        optim.zero_grad()

        # get loss and accuracy
        loss_train = model.loss(out_train, y, ~epoc_mask)

        # apply gradients
        loss_train.backward()
        optim.step()

        # no grad ops
        with torch.no_grad():
            model.eval()
            label_mask_valid = ~(test_mask | valid_mask)
            label_mask_test = ~test_mask

            out_test = model(data.x, y, data.edge_index, label_mask_test)
            out_valid = model(data.x, y, data.edge_index, label_mask_valid)

            loss_valid = model.loss(out_valid, y, valid_mask)
            loss_test = model.loss(out_test, y, test_mask)

            accuracy_train = model.accuracy(out_train, y, epoc_mask)
            accuracy_valid = model.accuracy(out_valid, y, valid_mask)
            accuracy_test = model.accuracy(out_test, y, test_mask)

        # log info
        logger.info(
            f"""
            epoch = {epoch}:
            train loss = {loss_train}, train acc = {100*accuracy_train:.2f}%
            valid loss = {loss_valid}, valid acc = {100*accuracy_valid:.2f}%
            test loss = {loss_test}, test acc = {100*accuracy_test:.2f}%
            """
        )


model = UnimpNet(data.x.shape[1], num_classes, inner_dim, heads)
optim = torch.optim.Adam(model.parameters(), lr=0.01)

train(model, optim)
