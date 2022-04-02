import logging
import os

import torch
from ogb.nodeproppred import PygNodePropPredDataset

import torch_geometric.transforms as T
import torch_geometric.utils.mask as mask_util
from torch_geometric.nn import MaskLabel, TransformerConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

logger.info(f"using device: {device}")

# Get dataset
root = os.path.join("..", "data", "OGB")
dataset = PygNodePropPredDataset(
    "ogbn-arxiv",
    root,
    transform=T.Compose([T.ToUndirected()]),
)
data = dataset[0]
data.to(device)
num_classes = 40

# Form masks for labels: mask gives labels being predicted
split_idx = dataset.get_idx_split()
train_mask = mask_util.index_to_mask(split_idx["train"], size=data.num_nodes)
valid_mask = mask_util.index_to_mask(split_idx["valid"], size=data.num_nodes)
test_mask = mask_util.index_to_mask(split_idx["test"], size=data.num_nodes)

# Model parameters
inner_dim = 64
heads = 2


class UnimpNet(torch.nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        inner_dim: int,
        heads: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.label_embedding = MaskLabel(num_classes, feature_size)

        self.conv1 = TransformerConv(feature_size, inner_dim // heads,
                                     heads=heads, dropout=dropout, beta=True,
                                     concat=True)
        self.norm1 = torch.nn.LayerNorm(inner_dim)

        self.conv2 = TransformerConv(inner_dim, inner_dim // heads,
                                     heads=heads, dropout=dropout, beta=True,
                                     concat=True)
        self.norm2 = torch.nn.LayerNorm(inner_dim)

        self.conv3 = TransformerConv(inner_dim, num_classes, heads=heads,
                                     dropout=dropout, beta=True, concat=False)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        label_mask: torch.Tensor,
    ):
        x = self.label_embedding(x, y, label_mask)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x, edge_index)
        return x

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


def train(model, optim, epochs=500, label_rate=0.65):

    y = data.y.squeeze()
    for epoch in range(epochs):
        model.train()

        # create epoch training mask that chooses subset of train to remove
        epoch_mask = MaskLabel.ratio_mask(train_mask, 1 - label_rate, True)

        # label mask is a mask to give what labels are allowed
        label_mask = train_mask ^ epoch_mask

        # forward pass
        out_train = model(data.x, data.y.squeeze(), data.edge_index,
                          epoch_mask)

        optim.zero_grad()

        # get loss and accuracy
        loss_train = model.loss(out_train, y, label_mask)

        # apply gradients
        loss_train.backward()
        optim.step()

        # no grad ops
        with torch.no_grad():
            model.eval()
            epoc_mask_valid = train_mask
            epoch_mask_test = train_mask | valid_mask

            out_test = model(data.x, y, data.edge_index, epoch_mask_test)
            out_valid = model(data.x, y, data.edge_index, epoc_mask_valid)

            loss_valid = model.loss(out_valid, y, valid_mask)
            loss_test = model.loss(out_test, y, test_mask)

            accuracy_train = model.accuracy(out_train, y, label_mask)
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
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

train(model, optim)
