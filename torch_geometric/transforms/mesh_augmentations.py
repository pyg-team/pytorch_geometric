import numpy as np
from ..utils.mesh_utils import fixed_division, get_edge_points
from ..utils.mesh_features import dihedral_angle


#########################
# Mesh Augmentations
#########################
def scale_verts(vs, mean=1, var=0.1):
    for i in range(vs.shape[1]):
        vs[:, i] = vs[:, i] * np.random.normal(mean, var)


def flip_edges(vs, faces, prct):
    edge_count, edge_faces, edges_dict = get_edge_faces(faces)
    dihedral = angles_from_faces(vs, edge_faces[:, 2:], faces)
    edges2flip = np.random.permutation(edge_count)
    target = int(prct * edge_count)
    flipped = 0
    for edge_key in edges2flip:
        if flipped == target:
            break
        if dihedral[edge_key] > 2.7:
            edge_info = edge_faces[edge_key]
            if edge_info[3] == -1:
                continue
            new_edge = tuple(sorted(list(set(faces[edge_info[2]]) ^ set(faces[edge_info[3]]))))
            if new_edge in edges_dict:
                continue
            new_faces = np.array(
                [[edge_info[1], new_edge[0], new_edge[1]], [edge_info[0], new_edge[0], new_edge[1]]])
            if check_area(vs, new_faces):
                del edges_dict[(edge_info[0], edge_info[1])]
                edge_info[:2] = [new_edge[0], new_edge[1]]
                edges_dict[new_edge] = edge_key
                rebuild_face(faces[edge_info[2]], new_faces[0])
                rebuild_face(faces[edge_info[3]], new_faces[1])
                for i, face_id in enumerate([edge_info[2], edge_info[3]]):
                    cur_face = faces[face_id]
                    for j in range(3):
                        cur_edge = tuple(sorted((cur_face[j], cur_face[(j + 1) % 3])))
                        if cur_edge != new_edge:
                            cur_edge_key = edges_dict[cur_edge]
                            for idx, face_nb in enumerate(
                                    [edge_faces[cur_edge_key, 2], edge_faces[cur_edge_key, 3]]):
                                if face_nb == edge_info[2 + (i + 1) % 2]:
                                    edge_faces[cur_edge_key, 2 + idx] = face_id
                flipped += 1
    return faces


def get_edge_faces(faces):
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, np.array(edge_faces), edge2keys


def angles_from_faces(vs, edge_faces, faces):
    normals = [None, None]
    for i in range(2):
        edge_a = vs[faces[edge_faces[:, i], 2]] - vs[faces[edge_faces[:, i], 1]]
        edge_b = vs[faces[edge_faces[:, i], 1]] - vs[faces[edge_faces[:, i], 0]]
        normals[i] = np.cross(edge_a, edge_b)
        div = fixed_division(np.linalg.norm(normals[i], ord=2, axis=1), epsilon=0)
        normals[i] /= div[:, np.newaxis]
    dot = np.sum(normals[0] * normals[1], axis=1).clip(-1, 1)
    angles = np.pi - np.arccos(dot)
    return angles


def rebuild_face(face, new_face):
    new_point = list(set(new_face) - set(face))[0]
    for i in range(3):
        if face[i] not in new_face:
            face[i] = new_point
            break
    return face


def check_area(vs, faces):
    face_normals = np.cross(vs[faces[:, 1]] - vs[faces[:, 0]],
                            vs[faces[:, 2]] - vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_areas *= 0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def slide_verts(mesh_edges, edge_indexes, edges_count, ve, vs, prct):
    edge_points = get_edge_points(mesh_edges, edge_indexes, edges_count)
    dihedral = dihedral_angle(vs, edge_points).squeeze()
    vids = np.random.permutation(len(ve))
    target = int(prct * len(vids))
    shifted = 0
    for vi in vids:
        if shifted < target:
            edges = ve[vi]
            if min(dihedral[edges]) > 2.65:
                edge = mesh_edges[np.random.choice(edges)]
                vi_t = edge[1] if vi == edge[0] else edge[0]
                nv = vs[vi] + np.random.uniform(0.2, 0.5) * (vs[vi_t] - vs[vi])
                vs[vi] = nv
                shifted += 1
        else:
            break
    shifted = shifted / len(ve)
    return shifted
