import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing

class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

class DDE(nn.Module):
    def __init__(
        self,
        num_rounds,
        num_reverse_rounds
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())
        
        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())
    
    def forward(
        self,
        topic_entity_one_hot,
        edge_index,
        reverse_edge_index
    ):
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

class Retriever(nn.Module):
    r"""The SubgraphRAG retriever model from the `"Simple Is Effective: The 
    Roles of Graphs and Large Language Models in Knowledge-Graph-Based 
    Retrieval-Augmented Generation"
    <https://arxiv.org/abs/2410.20724>`_ paper.

    Args:
        emb_size (int): The dimension of the embeddings.
        gnn (torch.nn.Module): The GNN to use.
        mlp_out_channels (dict, optional): 
    """
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs
    ):
        super().__init__()
        
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)
        
        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def forward(
        self,
        edge_index,
        q_emb,
        entity_embs,
        relation_embs,
        num_non_text_entities,
        topic_entity_one_hot
    ):
        device = entity_embs.device
        
        h_e = torch.cat(
            [
                entity_embs,
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
            ]
        , dim=0)
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        reverse_edge_index = torch.stack([
            edge_index[1],
            edge_index[0]
        ], dim=0)

        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb

        h_triple = torch.cat([
            h_q.expand(len(relation_embs), -1),
            h_e[edge_index[0]],
            relation_embs,
            h_e[edge_index[1]]
        ], dim=1)
        
        return self.pred(h_triple)
