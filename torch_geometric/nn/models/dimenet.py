import torch
from torch_sparse import SparseTensor


def swish(x):
    return x * x.sigmoid(x)


class DimeNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_blocks, num_bilinear,
                 num_spherical, num_radial, cutoff=5.0, envelope_exponent=5,
                 num_before_skip=1, num_after_skip=2, num_dense_output=3,
                 num_targets=1, activation=swish):
        super(DimeNet, self).__init__()

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index  # i->j

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, x, pos, edge_index):
        row, col = edge_index
        idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=x.size(0))
