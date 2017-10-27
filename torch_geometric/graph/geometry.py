import torch


def edges_from_faces(faces):
    """Get undirected edges from triangular faces."""

    # Get directed edges.
    edges = torch.cat((faces[:, 0:2], faces[:, 1:3], faces[:, 0:3:2]))
    n = faces.max() + 1

    # Build directed adjacency matrix.
    adj = torch.sparse.FloatTensor(edges.t(),
                                   torch.ones(edges.size(0)),
                                   torch.Size([n, n]))

    # Convert to undirected adjacency matrix.
    adj = adj + adj.t()

    # Remove duplicate indices.
    # NOTE: `coalesce()` doesn't work if `transpose(...)` is removed.
    adj = adj.transpose(0, 1).coalesce()
    return adj._indices()
