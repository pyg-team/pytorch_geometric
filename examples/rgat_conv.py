import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rgat_conv import RGATConv
from torch.nn import Linear
from torch.nn.utils import clip_grad_value_

from torch_geometric.datasets import Entities


class RGAT(nn.Module):
    def __init__(self, d=2):
        super(Net, self).__init__()

        self.nclass = 4
        self.readout = "add"
        self.in_channels = 16
        self.out_channels = 25
        self.d = d

        self.conv_layers = nn.ModuleList([
            RGATConv(16, 20, num_relations=90, num_bases=35, mod="additive",
                     attention_mechanism="within-relation",
                     attention_mode="multiplicative-self-attention", heads=2,
                     d=2),
            RGATConv(80, 25, num_relations=90, num_blocks=2, mod=None,
                     attention_mechanism="across-relation",
                     attention_mode="additive-self-attention", heads=2, d=1,
                     dropout=0.6, edge_dim=16, bias=False),
        ])

        self.fc = Linear(16, self.in_channels, bias=False)

        self.linear = Linear(self.d * self.out_channels, self.nclass)

    def forward(self, x, edge_index, edge_type, edge_attr):
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        hid_x = self.fc(x)

        for yt, conv in enumerate(self.conv_layers):
            if yt == 0:
                hid_x = conv(hid_x, edge_index, edge_type, edge_attr=None)
            else:
                hid_x = conv(hid_x, edge_index, edge_type, edge_attr)

        x = self.linear(hid_x)
        return F.log_softmax(x, dim=-1)


if __name__ == '__main__':
    dataset = Entities(root='/tmp/AIFB', name='AIFB')

    edge_index, edge_type, train_labels, test_labels, train_mask, test_mask, \
        num_nodes = dataset[0].edge_index, dataset[0].edge_type, \
        dataset[0].train_y, dataset[0].test_y, dataset[0].train_idx, \
        dataset[0].test_idx, dataset[0].num_nodes

    x = torch.randn(num_nodes, 16)
    edge_attr = torch.randn(58086, 2, 3, 7, 16)

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3,
                           msgrad=False)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(x, edge_index, edge_type, edge_attr, torch.tensor([0]))
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
