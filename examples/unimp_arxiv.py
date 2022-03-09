import logging
import os

import torch
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
import torch_geometric.utils.mask as mask_util
from torch_geometric.nn import MaskLabel, TransformerConv

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
train_mask = mask_util.index_to_mask(split_idx["train"], size=data.num_nodes)
valid_mask = mask_util.index_to_mask(split_idx["valid"], size=data.num_nodes)
test_mask = mask_util.index_to_mask(split_idx["test"], size=data.num_nodes)

# Model parameters
inner_dim = 16
heads = 2


class UnimpNet(torch.nn.Module):
    def __init__(self, feature_size: int, num_classes: int, inner_dim: int,
                 heads: int):
        super().__init__()

        self.label_embedding = MaskLabel(num_classes, feature_size)
        self.conv = TransformerConv(feature_size, inner_dim // heads,
                                    heads=heads)
        self.linear = torch.nn.Linear(inner_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        label_mask: torch.Tensor,
    ):
        x = self.label_embedding(x, y, label_mask)
        x = self.conv(x, edge_index)
        out = self.linear(x)
        return out

    def predictions(self, out: torch.Tensor):
        probs = torch.nn.functional.softmax(out, dim=1)
        return probs.argmax(dim=1)

    def loss(self, out: torch.Tensor, labels: torch.Tensor,
             mask: torch.Tensor):
        return torch.nn.functional.cross_entropy(out[mask], labels[mask])

    def accuracy(self, out: torch.Tensor, labels: torch.Tensor,
                 mask: torch.Tensor):
        return ((self.predictions(out[mask]) == labels[mask]).sum().float() /
                float(labels[mask].size(0)))


def train(model, optim, epochs=50, label_rate=0.9):

    y = data.y.squeeze()
    for epoch in range(epochs):
        model.train()

        # create epoch training mask that chooses subset of train to remove
        epoch_mask = MaskLabel.ratio_mask(train_mask, 1 - label_rate)

        # label mask is a mask to give what labels are allowed
        label_mask = ~(epoch_mask | test_mask | valid_mask)

        # forward pass
        out_train = model(data.x, data.y.squeeze(), data.edge_index,
                          label_mask)

        optim.zero_grad()

        # get loss and accuracy
        loss_train = model.loss(out_train, y, ~epoch_mask)

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

            accuracy_train = model.accuracy(out_train, y, epoch_mask)
            accuracy_valid = model.accuracy(out_valid, y, valid_mask)
            accuracy_test = model.accuracy(out_test, y, test_mask)

        # log info
        logger.info(f"""
            epoch = {epoch}:
            train loss = {loss_train}, train acc = {100*accuracy_train:.2f}%
            valid loss = {loss_valid}, valid acc = {100*accuracy_valid:.2f}%
            test loss = {loss_test}, test acc = {100*accuracy_test:.2f}%
            """)


model = UnimpNet(data.x.shape[1], num_classes, inner_dim, heads)
optim = torch.optim.Adam(model.parameters(), lr=0.01)

train(model, optim)
