from math import sqrt
from itertools import chain

import torch
import networkx as nx
from torch_geometric.nn import MessagePassing

EPS = 1e-15


class GNNExplainer(torch.nn.Module):
    def __init__(self, model, epochs=100, lr=0.01):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)
        self.node_feat_masks = torch.nn.ParameterList([node_feat_mask])

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

        for m in chain(self.node_feat_masks, self.edge_masks):
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
            list(self.node_feat_masks) + list(self.edge_masks), lr=self.lr)

        for epoch in range(1, self.epochs):
            optimizer.zero_grad()
            h = x * self.node_feat_masks[0].view(1, -1).sigmoid()
            log_logits = self.model(x=h, edge_index=edge_index, **kwargs)
            loss = self.__loss__(node_idx, log_logits, pred_label)
            loss.backward()
            optimizer.step()

        node_feat_masks = self.node_feat_masks
        node_feat_masks = [mask.detach().sigmoid() for mask in node_feat_masks]
        edge_masks = [mask.detach().sigmoid() for mask in self.edge_masks]

        self.__clear_masks__()

        return node_feat_masks[0], edge_masks

    def visualize_subgraph(self, node_idx, edge_index, threshold=None):
        assert self.edge_masks is not None

        G = nx.DiGraph()
        for i, (source, target) in enumerate(edge_index.t().tolist()):
            G.add_edge(source, target, idx=i)
        G = G.reverse()

        G_sub = nx.DiGraph()
        cur = [node_idx]
        for edge_mask in self.edge_masks:
            edge_mask = edge_mask.sigmoid().tolist()
            new_cur = []
            for node_idx in cur:
                for v, u, data in G.edges(node_idx, data=True):
                    new_cur.append(u)
                    if not G_sub.has_edge(u, v):
                        G_sub.add_edge(u, v, weight=edge_mask[data['idx']])
                    else:
                        G_sub[u][v]['weight'] = max(edge_mask[data['idx']],
                                                    G_sub[u][v]['weight'])
            cur = list(set(new_cur))

        nx.draw(G_sub, pos=nx.spring_layout(G_sub), with_labels=True,
                font_size=12, connectionstyle='arc3, rad = 0.1')

        # plt.show()
