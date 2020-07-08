from torch_scatter import scatter
from torch_geometric.utils import add_self_loops


def neighbor_max_pool(data, flow='source_to_target'):
    r"""Max pools neighborhood features

    Replace each feature value with the maximum value from this
    node and its neighbors.

    See: https://github.com/deepchem/deepchem/blob/2.3.0/deepchem/models/layers.py#L133  # noqa

    """
    x, edge_index = data.x, data.edge_index
    if flow == 'source_to_target':
        edge_index = add_self_loops(edge_index)[0]
        x = scatter(x[edge_index[1]], edge_index[0], reduce='max', dim=0)
    elif flow == 'target_to_source':
        edge_index = add_self_loops(edge_index)[1]
        x = scatter(x[edge_index[0]], edge_index[1], reduce='max', dim=0)

    data.x = x
    return data
