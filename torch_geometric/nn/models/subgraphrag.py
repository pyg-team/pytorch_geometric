import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing


class PEConv(MessagePassing):
    """Positional Encoding Convolution layer.

    Simple message passing layer that propagates node features
    through the graph without any transformation.
    """
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class DDE(nn.Module):
    """Directional Distance Encoding module.

    DEE computes positional encodings based on the directional distance
    of nodes from topic entities in the graph. It performs message passing in both
    forward and reverse directions to capture the structural information of the graph
    relative to the topic entities.


    Args:
        num_rounds (int): Number of forward propagation rounds.
        num_reverse_rounds (int): Number of reverse propagation rounds.
    """
    def __init__(self, num_rounds, num_reverse_rounds):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())

        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())

    def forward(self, topic_entity_one_hot, edge_index, reverse_edge_index):
        """Forward pass of the DDE module.

        Computes directional distance encodings by propagating the topic entity
        indicators through the graph in both forward and reverse directions.

        Args:
            topic_entity_one_hot (Tensor): One-hot encoding indicating topic entities
                with shape [num_nodes, num_classes].
            edge_index (Tensor): The graph connectivity in COO format with shape [2, num_edges].
            reverse_edge_index (Tensor): The reversed graph connectivity in COO format
                with shape [2, num_edges].

        Returns:
            List[Tensor]: A list of node encodings from each layer of propagation,
                capturing the directional distance information at different hops.
        """
        result_list = []

        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            result_list.append(h_pe)

        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            result_list.append(h_pe_rev)

        return result_list


class SubgraphRAGRetriever(nn.Module):
    r"""The SubgraphRAG retriever model from the `"Simple Is Effective: The
    Roles of Graphs and Large Language Models in Knowledge-Graph-Based
    Retrieval-Augmented Generation"
    <https://arxiv.org/abs/2410.20724>`_ paper.

    Args:
        emb_size (int): The dimension of the embeddings.
        topic_pe (bool): Whether to include the topic_pe in the embeddings.
        dde_rounds (int, optional): The number of DDE passes to apply.
        rev_dde_rounds (int, optional): The number of reverse DDE passes to apply.
    """
    def __init__(self, emb_size, topic_pe, dde_rounds, rev_dde_rounds):
        super().__init__()

        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(dde_rounds, rev_dde_rounds)

        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (dde_rounds + rev_dde_rounds)

        self.pred = nn.Sequential(nn.Linear(pred_in_size, emb_size), nn.ReLU(),
                                  nn.Linear(emb_size, 1))

    def forward(self, edge_index, q_emb, entity_embs, relation_embs,
                topic_entity_one_hot):
        h_e = torch.cat([
            entity_embs,
        ], dim=0)
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        dde_list = self.dde(topic_entity_one_hot, edge_index,
                            reverse_edge_index)
        h_e_list.extend(dde_list)
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb

        h_triple = torch.cat([
            h_q.expand(len(relation_embs), -1),
            h_e[edge_index[0]],
            relation_embs,
            h_e[edge_index[1]],
        ], dim=1)

        return self.pred(h_triple)
