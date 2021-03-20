import numpy as np


def get_edge_points(edges, edge_indexes, edges_count):
    """ returns: edge_points (#E x 4) tensor, with four vertex ids per edge
        for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are
        the two vertices which define edge_id each adjacent face to edge_id has
        another vertex, which is edge_points[edge_id, 2] or
        edge_points[edge_id, 3]
    """
    edge_points = np.zeros([edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(edges):
        edge_points[edge_id] = get_side_points(edges, edge_indexes, edge_id)
    return edge_points


def get_side_points(mesh_edges, mesh_edge_indexes, edge_id):
    edge_a = mesh_edges[edge_id]
    edge_id_pyg = edge_id * 5

    if mesh_edge_indexes[0, edge_id_pyg + 1] == -1:
        edge_b = mesh_edges[mesh_edge_indexes[0, edge_id_pyg + 3]]
        edge_c = mesh_edges[mesh_edge_indexes[0, edge_id_pyg + 4]]
    else:
        edge_b = mesh_edges[mesh_edge_indexes[0, edge_id_pyg + 1]]
        edge_c = mesh_edges[mesh_edge_indexes[0, edge_id_pyg + 2]]
    if mesh_edge_indexes[0, edge_id_pyg + 3] == -1:
        edge_d = mesh_edges[mesh_edge_indexes[0, edge_id_pyg + 1]]
        edge_e = mesh_edges[mesh_edge_indexes[0, edge_id_pyg + 2]]
    else:
        edge_d = mesh_edges[mesh_edge_indexes[0, edge_id_pyg + 3]]
        edge_e = mesh_edges[mesh_edge_indexes[0, edge_id_pyg + 4]]

    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex],
            edge_b[second_vertex], edge_d[third_vertex]]


def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div
