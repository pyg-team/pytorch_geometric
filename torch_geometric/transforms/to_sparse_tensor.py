from torch_sparse import SparseTensor


class ToSparseTensor(object):
    r"""Converts the :obj:`edge_index` format into the sparse tensor format
    of :class:`torch_sparse.SparseTensor`.
    """
    def __call__(self, data):
        assert data.edge_index is not None

        (row, col), N, E = data.edge_index, data.num_nodes, data.num_edges
        data.edge_index = None
        perm = (col * N + row).argsort()
        row, col = row[perm], col[perm]

        value = None
        for key in ['edge_weight', 'edge_attr', 'edge_type']:
            if data[key] is not None:
                value = data[key][perm]
                data[key] = None
                break

        for key, item in data:
            if item.size(0) == E:
                data[key] = item[perm]

        data.adj_t = SparseTensor(row=col, col=row, value=value,
                                  sparse_sizes=(N, N), is_sorted=True)
        data.adj_t.storage.rowptr()
        data.adj_t.storage.csr2csc()

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'
