from torch_sparse import SparseTensor


class ToSparseTensor(object):
    r"""Converts the :obj:`edge_index` attribute of a data object into a
    (transposed) :class:`torch_sparse.SparseTensor` type with key
    :obj:`adj_.t`.

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the
            :obj:`edge_index` tensor will not be removed.
    """
    def __init__(self, remove_edge_index: bool = True):
        self.remove_edge_index = remove_edge_index

    def __call__(self, data):
        assert data.edge_index is not None

        (row, col), N, E = data.edge_index, data.num_nodes, data.num_edges
        perm = (col * N + row).argsort()
        row, col = row[perm], col[perm]

        if self.remove_edge_index:
            data.edge_index = None

        value = None
        for key in ['edge_weight', 'edge_attr', 'edge_type']:
            if data[key] is not None:
                value = data[key][perm]
                if self.remove_edge_index:
                    data[key] = None
                break

        for key, item in data:
            if item.size(0) == E:
                data[key] = item[perm]

        data.adj_t = SparseTensor(row=col, col=row, value=value,
                                  sparse_sizes=(N, N), is_sorted=True)

        # Pre-process some important attributes.
        data.adj_t.storage.rowptr()
        data.adj_t.storage.csr2csc()

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'
