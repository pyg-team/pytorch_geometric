from math import sqrt
from itertools import chain

import torch
from torch_geometric.nn import MessagePassing

EPS = 1e-8


class GNNExplainer(torch.nn.Module):
    def __init__(self, model, epochs=100, lr=0.01):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        node_mask = torch.nn.Parameter(torch.randn(F) * 0.1)
        self.node_masks = torch.nn.ParameterList([node_mask])

        self.edge_masks = torch.nn.ParameterList()
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                edge_mask = torch.nn.Parameter(torch.randn(E) * std)
                self.edge_masks.append(edge_mask)
                module.__explain__ = True
                module.__edge_mask__ = edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False

    def __loss__(self, node_idx, log_logits, pred_label):
        # TODO: Add Laplacian regularization.
        # TODO: Add size regularization.

        loss = -log_logits[node_idx, pred_label[node_idx]]

        for m in chain(self.node_masks, self.edge_masks):
            m = m.sigmoid()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + ent.mean()

        return loss

    def explain_node(self, node_idx, x, edge_index, **kwargs):
        self.model.eval()

        with torch.no_grad():
            log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
            pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam(
            list(self.node_masks) + list(self.edge_masks), lr=self.lr)

        for epoch in range(1, self.epochs):
            optimizer.zero_grad()
            h = x * self.node_masks[0].view(1, -1).sigmoid()
            log_logits = self.model(x=h, edge_index=edge_index, **kwargs)
            loss = self.__loss__(node_idx, log_logits, pred_label)
            loss.backward()
            optimizer.step()

        self.__clear_masks__()

    def visualize_subgraph(self, node_idx, edge_index, num_hops,
                           threshold=None):
        # Generates
        pass
