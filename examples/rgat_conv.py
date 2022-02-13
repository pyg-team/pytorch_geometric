import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear
from torch.nn.utils import clip_grad_value_

from torch_geometric.datasets import Entities
from torch_geometric.nn import RGATConv


class RGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nclass=4):
        super().__init__()

        self.nclass = nclass
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.convs = nn.ModuleList([
            RGATConv(self.hidden_channels, 20, num_relations=90, num_bases=35,
                     mod="additive", attention_mechanism="within-relation",
                     attention_mode="multiplicative-self-attention", heads=2,
                     d=2),
            # The "in_channels" for every single layer after the first layer
            # must be equal to the product of three arguments:
            # "heads * d * out_channels" being used in the previous layer
            # because each RGATConv layer outputs "heads * d * out_channels"
            # features for each node
            RGATConv(80, self.out_channels, num_relations=90, num_blocks=2,
                     mod=None, attention_mechanism="across-relation",
                     attention_mode="additive-self-attention", heads=2, d=1,
                     dropout=0.6, edge_dim=16, bias=False),
        ])

        self.lin1 = Linear(self.in_channels, self.hidden_channels, bias=False)

        # The following layer is being used so that the final returned output
        # will consist of "nclass" features for each node
        self.lin2 = Linear(2 * self.out_channels, self.nclass)

    def forward(self, x, edge_index, edge_type, edge_attr):

        hid_x = self.lin1(x)

        # "edge_attr" is being put to some use only for the second layer
        # just to check if arbitrary ordering of "edge_attr" among layers
        # of a GNN module can be achieved. Nevertheless, "edge_attr" can
        # be passed to any "RGAT" layer.
        for yt, conv in enumerate(self.convs):
            if yt == 0:
                hid_x = conv(hid_x, edge_index, edge_type, edge_attr=None)
                hid_x = F.relu(hid_x)
            else:
                hid_x = conv(hid_x, edge_index, edge_type, edge_attr)

        x = self.lin2(hid_x)
        return F.log_softmax(x, dim=-1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Entities(root='/tmp/AIFB', name='AIFB')
    data = dataset[0]
    data = data.to(device)

    edge_index, edge_type, train_labels, test_labels, train_mask, test_mask, \
        num_nodes = data.edge_index, data.edge_type, \
        data.train_y, data.test_y, data.train_idx, \
        data.test_idx, data.num_nodes

    x = torch.randn(num_nodes, 16)
    edge_attr = torch.randn(58086, 2, 3, 7, 16)

    model = RGAT(16, 16, 25).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3,
                           amsgrad=False)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(x, edge_index, edge_type, edge_attr)
        loss = F.nll_loss(output[train_mask], train_labels)
        loss.backward()
        clip_grad_value_(model.parameters(), 2.0)
        optimizer.step()

        model.eval()
        train_acc = torch.sum(output[train_mask].argmax(dim=1) == train_labels)
        train_acc = train_acc / len(train_mask)
        test_acc = torch.sum(output[test_mask].argmax(dim=1) == test_labels)
        test_acc = test_acc.item() / len(test_mask)

        print('Epoch {:03d}'.format(epoch + 1),
              'train_loss: {:.4f}'.format(loss),
              'train_acc: {:.4f}'.format(train_acc),
              'test_acc: {:.4f}'.format(test_acc))

    print("Optimization Finished!")
