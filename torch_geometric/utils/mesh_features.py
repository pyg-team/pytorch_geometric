import numpy as np
from .mesh_utils import get_edge_points, fixed_division


def mesh_extract_features(mesh):
    features = []
    edge_points = get_edge_points(mesh.edges, mesh.edge_index,
                                  mesh.edges_count)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide='raise'):
        try:
            feature = dihedral_angle(mesh.pos, edge_points)
            features.append(feature)
            feature = symmetric_opposite_angles(mesh, edge_points)
            features.append(feature)
            feature = symmetric_ratios(mesh, edge_points)
            features.append(feature)
            return np.concatenate(features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad features')


def set_edge_lengths(mesh, edge_points=None):
    if edge_points is not None:
        edge_points = get_edge_points(mesh.edges, mesh.edge_index,
                                      mesh.edges_count)
    edge_lengths = np.linalg.norm(
        mesh.pos[edge_points[:, 0]] - mesh.pos[edge_points[:, 1]], ord=2,
        axis=1)
    mesh.edge_lengths = edge_lengths


def dihedral_angle(pos, edge_points):
    normals_a = get_normals(pos, edge_points, 0)
    normals_b = get_normals(pos, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles


def symmetric_opposite_angles(mesh, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = np.concatenate(
        (np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
    angles = np.sort(angles, axis=0)
    return angles


def get_normals(pos, edge_points, side):
    edge_a = pos[edge_points[:, side // 2 + 2]] - pos[
        edge_points[:, side // 2]]
    edge_b = pos[edge_points[:, 1 - side // 2]] - pos[
        edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals


def get_opposite_angles(mesh, edge_points, side):
    edges_a = np.array(mesh.pos[edge_points[:, side // 2]] - mesh.pos[
        edge_points[:, side // 2 + 2]])
    edges_b = np.array(mesh.pos[edge_points[:, 1 - side // 2]] - mesh.pos[
        edge_points[:, side // 2 + 2]])

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1),
                              epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1),
                              epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def symmetric_ratios(mesh, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = np.concatenate(
        (np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
    return np.sort(ratios, axis=0)


def get_ratios(mesh, edge_points, side):
    edges_lengths = np.linalg.norm(
        mesh.pos[edge_points[:, side // 2]] - mesh.pos[
            edge_points[:, 1 - side // 2]],
        ord=2, axis=1)
    point_o = np.array(mesh.pos[edge_points[:, side // 2 + 2]])
    point_a = np.array(mesh.pos[edge_points[:, side // 2]])
    point_b = np.array(mesh.pos[edge_points[:, 1 - side // 2]])
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a),
                               axis=1) / fixed_division(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    closest_point = \
        point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths
