import torch
from torch import nn

#referece from https://github.com/twitter-research/tgn
class TemporalAttentionLayer(torch.nn.Module):
    def __init__(self, node_feat_dim, neighbor_feat_dim, edge_feat_dim, time_dim,
                 output_dim, num_heads=2, dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()

        self.num_heads = num_heads

        self.node_feat_dim = node_feat_dim
        self.time_dim = time_dim

        self.query_dim = node_feat_dim + time_dim
        self.key_dim = neighbor_feat_dim + time_dim + edge_feat_dim

        self.merger = MergeLayer(self.query_dim, node_feat_dim, node_feat_dim, output_dim)

        self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                       kdim=self.key_dim,
                                                       vdim=self.key_dim,
                                                       num_heads=num_heads,
                                                       dropout=dropout)

    def forward(self, src_node_feats, src_time_feats, neighbors_feats,
                neighbors_time_feats, edge_feats, neighbors_padding_mask):
        src_node_feats_unrolled = torch.unsqueeze(src_node_feats, dim=1)

        query = torch.cat([src_node_feats_unrolled, src_time_feats], dim=2)
        key = torch.cat([neighbors_feats, edge_feats, neighbors_time_feats], dim=2)

        query = query.permute([1, 0, 2])
        key = key.permute([1, 0, 2])

        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
        neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False

        attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,
                                                                  key_padding_mask=neighbors_padding_mask)

        attn_output = attn_output.squeeze()
        attn_output_weights = attn_output_weights.squeeze()

        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

        attn_output = self.merger(attn_output, src_node_feats)

        return attn_output, attn_output_weights
