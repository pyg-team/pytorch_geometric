import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeLayer(nn.Module):
    """TypeLayer performs entity embedding transformation based on relation features and graph structure.

    It computes fact embeddings from relation features and aggregates them to create enhanced entity embeddings.
    Used in ReaRev to encode entities with graph-based context.
    """
    def __init__(self, in_features, out_features, linear_drop, device,
                 norm_rel):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_drop = linear_drop
        self.kb_self_linear = nn.Linear(in_features, out_features)
        self.device = device
        self.norm_rel = norm_rel

    def forward(self, local_entity, edge_list, rel_features):
        """Compute enhanced entity embeddings using relation features and edge connections.
        Takes in Tensor of entity indices, Tuple containing graph edge information, Tensor of relation embeddings.
        Returns Enhanced entity embeddings.
        """
        (batch_heads, batch_rels, batch_tails, batch_ids, fact_ids,
         weight_list, weight_rel_list) = edge_list

        fact2head, fact2tail, batch_rels, batch_ids, val_one = self._prepare_sparse_indices(
            batch_heads, batch_rels, batch_tails, batch_ids, fact_ids,
            weight_rel_list, weight_list)
        fact_val = self._compute_fact_val(rel_features, batch_rels)
        f2e_emb = self._aggregate_facts(fact2head, fact2tail, val_one,
                                        fact_val, local_entity, len(fact_ids))
        return f2e_emb

    def _build_sparse_tensor(self, indices, values, size):
        """Creates a sparse tensor for efficient graph computation.
        """
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)

    def _prepare_sparse_indices(self, batch_heads, batch_rels, batch_tails,
                                batch_ids, fact_ids, weight_rel_list,
                                weight_list):
        """Prepares indices and values for sparse tensor creation.
        Takes in Graph edge information, Fact indexing information, Weights for relations and edges.

        Returns Sparse indices and values.
        """
        fact2head = torch.LongTensor([batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([batch_tails, fact_ids]).to(self.device)
        batch_rels = torch.LongTensor(batch_rels).to(self.device)
        batch_ids = torch.LongTensor(batch_ids).to(self.device)
        if self.norm_rel:
            val_one = torch.FloatTensor(weight_rel_list).to(self.device)
        else:
            val_one = torch.ones_like(batch_ids).float().to(self.device)
        return fact2head, fact2tail, batch_rels, batch_ids, val_one

    def _compute_fact_val(self, rel_features, batch_rels):
        """Computes fact embeddings from relation features. Takes in Relation feature embeddings and a batch of relation indices.
        Returns transformed fact embeddings.
        """
        fact_rel = torch.index_select(rel_features, dim=0, index=batch_rels)
        fact_val = self.kb_self_linear(fact_rel)
        return fact_val

    def _aggregate_facts(self, fact2head, fact2tail, val_one, fact_val,
                         local_entity, num_fact):
        """Aggregates fact embeddings into entity embeddings.

        Takes in:
        - fact2head, fact2tail: Sparse tensor indices for head and tail aggregation.
        - val_one: Edge weights.
        - fact_val: Fact embeddings.
        - local_entity: Entity indices.
        - num_fact: Number of facts.

        Returns:
        - Enhanced entity embeddings.
        """
        batch_size, max_local_entity = local_entity.size()
        hidden_size = self.in_features

        fact2tail_mat = self._build_sparse_tensor(
            fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        fact2head_mat = self._build_sparse_tensor(
            fact2head, val_one, (batch_size * max_local_entity, num_fact))

        f2e_emb = F.relu(
            torch.sparse.mm(fact2tail_mat, fact_val) +
            torch.sparse.mm(fact2head_mat, fact_val))
        assert not torch.isnan(f2e_emb).any()

        f2e_emb = f2e_emb.view(batch_size, max_local_entity, hidden_size)
        return f2e_emb


class Fusion(nn.Module):
    """Combines two input vectors with gating mechanisms. Implementation described in source paper (https://arxiv.org/abs/2210.13650).

    Used in ReaRev to integrate query embeddings and context information effectively.
    """
    def __init__(self, d_hid):
        super().__init__()
        self.r = nn.Linear(d_hid * 3, d_hid, bias=False)
        self.g = nn.Linear(d_hid * 3, d_hid, bias=False)

    def forward(self, x, y):
        """Compute fused output from two input tensors.
        """
        concat_input = torch.cat([x, y, x - y], dim=-1)
        r_ = self.r(concat_input)
        g_ = torch.sigmoid(self.g(concat_input))
        return g_ * r_ + (1 - g_) * x


class QueryReform(nn.Module):
    """QueryReform refines query node embeddings using context from entity embeddings and seed information.

    Used in ReaRev to iteratively adjust query representations during reasoning steps.
    """
    def __init__(self, h_dim):
        super().__init__()
        self.fusion = Fusion(h_dim)
        self.q_ent_attn = nn.Linear(h_dim, h_dim)

    def forward(self, q_node, ent_emb, seed_info, ent_mask):
        """Refine query node embeddings with attention and fusion mechanisms.
        """
        q_node_lin = self.q_ent_attn(q_node).unsqueeze(1)
        q_ent_score = (q_node_lin * ent_emb).sum(dim=2, keepdim=True)
        masked_scores = q_ent_score - (1 - ent_mask.unsqueeze(2)) * 1e8
        q_ent_attn = F.softmax(masked_scores, dim=1)
        (q_ent_attn * ent_emb).sum(1)
        seed_info_expanded = seed_info.unsqueeze(1)
        seed_retrieve = torch.bmm(seed_info_expanded, ent_emb).squeeze(1)
        fused_output = self.fusion(q_node, seed_retrieve)
        return fused_output
