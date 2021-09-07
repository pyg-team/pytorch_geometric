import numpy as np
import torch
import os
import ntpath
from .mesh_augmentations import (scale_verts, flip_edges, slide_verts)
from ..utils.mesh_features import mesh_extract_features
from ..transforms.meshcnnhandler import MeshCNNData
from torch_geometric.io import read_ply


class MeshCNNPrepare:
    r"""This class is a utility class to read and create MeshCNN structure for
    the MeshCNN paper: `"MeshCNN: A Network with an Edge"
    <https://arxiv.org/abs/1809.05910>`_ paper.
    Args:
        aug_slide_verts (float, optional): values between 0.0 to 1.0 - if set
                                           above 0 will apply shift along mesh
                                           surface with percentage of
                                           aug_slide_verts value.
        aug_scale_verts (bool, optional): If set to `False` - do not apply
                                          scale vertex augmentation.
                                          If set to `True` - apply non-
                                          uniformly scale on the mesh.
        aug_flip_edges (float, optional): values between 0.0 to 1.0 - if set
                                          above 0 will apply random flip of
                                          edge with percentage of
                                          aug_flip_edges value.
    """

    def __init__(self, aug_slide_verts: float = 0.0,
                 aug_scale_verts: bool = False,
                 aug_flip_edges: float = 0.0,
                 aug_file: str = ''):

        self.aug_slide_verts = aug_slide_verts
        self.aug_scale_verts = aug_scale_verts
        self.aug_flip_edges = aug_flip_edges
        self.aug_file = aug_file
        self.mesh_data = MeshCNNData()

    def __call__(self, data):
        """
        This function will fill MeshCNNData with data input.
        Args:
            data (torch_geometric.data.Data) - if exists must have keys of
            'pos' and 'faces' or 'pos' and 'face'. Holds the mesh data.

        Returns:
            mesh_data (torch_geometric.transforms.mesh.MeshCNNData) -
            MeshCNNData structure.

        """
        load_path = self.aug_file
        if os.path.exists(load_path):
            loaded_data = np.load(load_path, encoding='latin1',
                                  allow_pickle=True)
            self.mesh_data.pos = torch.tensor(loaded_data['pos'])
            self.mesh_data.edges = torch.tensor(loaded_data['edges'])
            self.mesh_data.edges_count = int(loaded_data['edges_count'])
            self.mesh_data.ve = loaded_data['ve'].tolist()
            self.mesh_data.v_mask = torch.tensor(loaded_data['v_mask'])
            self.mesh_data.filename = str(loaded_data['filename'])
            self.mesh_data.edge_lengths = loaded_data['edge_lengths']
            self.mesh_data.edge_areas = torch.tensor(loaded_data['edge_areas'])
            self.mesh_data.edge_attr = torch.tensor(loaded_data['edge_attr'])
            self.mesh_data.sides = torch.tensor(loaded_data['sides'])
            self.mesh_data.edge_index = torch.tensor(loaded_data['edge_index'],
                                                     dtype=torch.long)
        else:
            self.mesh_data = self.from_scratch(data)
            if load_path is not '':
                np.savez_compressed(load_path, pos=self.mesh_data.pos,
                                    edges=self.mesh_data.edges,
                                    edges_count=int(self.mesh_data.edges_count),
                                    ve=self.mesh_data.ve,
                                    v_mask=self.mesh_data.v_mask,
                                    filename=str(self.mesh_data.filename),
                                    sides=self.mesh_data.sides,
                                    edge_lengths=self.mesh_data.edge_lengths,
                                    edge_areas=self.mesh_data.edge_areas,
                                    edge_attr=self.mesh_data.edge_attr,
                                    edge_index=self.mesh_data.edge_index)

        return self.mesh_data

    def from_scratch(self, data):
        keys = data.keys
        assert (('pos' in keys and 'face' in keys)
                or
                ('pos' in keys and 'faces' in keys))

        mesh_data = MeshCNNData()
        if 'face' in keys:
            mesh_data.pos = data.pos
            faces = np.array(data.face.transpose(0, 1))
        elif 'faces':
            mesh_data.pos = torch.tensor(data.pos)
            faces = data.faces
        if 'y' in keys:
            mesh_data.y = data.y

        mesh_data.v_mask = torch.ones(len(mesh_data.pos), dtype=bool)

        faces, face_areas = self.remove_non_manifolds(mesh_data, faces)
        faces = self.mesh_augmentations(mesh_data, faces)
        self.build_edge_indexes(mesh_data, faces, face_areas)
        self.mesh_post_augmentations(mesh_data)
        mesh_data.edge_attr = torch.tensor(mesh_extract_features(mesh_data))
        return mesh_data

    def mesh_augmentations(self, mesh, faces):
        if self.aug_scale_verts:
            mesh.pos = scale_verts(mesh.pos)
        if self.aug_flip_edges > 0.0:
            faces = flip_edges(mesh.pos, faces, self.aug_flip_edges)
        return faces

    def mesh_post_augmentations(self, mesh):
        if self.aug_slide_verts > 0.0:
            mesh.shifted = slide_verts(mesh.edges, mesh.edge_index,
                                       mesh.edges_count,
                                       mesh.ve, mesh.pos, self.aug_slide_verts)

    def fill_from_file(self, mesh):
        mesh.filename = ntpath.split(self.file)[1]
        mesh.fullfilename = self.file
        _, suffix = os.path.splitext(self.file)

        if suffix == '.ply':
            data = read_ply(self.file)
            pos = np.array(data.pos, dtype=float)
            faces = np.array(data.face.transpose(0, 1), dtype=int)
        elif suffix == '.obj':
            pos, faces = [], []
            f = open(self.file)
            for line in f:
                line = line.strip()
                splitted_line = line.split()
                if not splitted_line:
                    continue
                elif splitted_line[0] == 'v':
                    pos.append([float(v) for v in splitted_line[1:4]])
                elif splitted_line[0] == 'f':
                    face_vertex_ids = [int(c.split('/')[0]) for c in
                                       splitted_line[1:]]
                    assert len(face_vertex_ids) == 3
                    face_vertex_ids = [
                        (ind - 1) if (ind >= 0) else (len(pos) + ind)
                        for ind in face_vertex_ids]
                    faces.append(face_vertex_ids)
            f.close()
            pos = np.array(pos, dtype=float)
            faces = np.array(faces, dtype=int)
        assert np.logical_and(faces >= 0, faces < len(pos)).all()
        return pos, faces

    def remove_non_manifolds(self, mesh, faces):
        mesh.ve = [[] for _ in mesh.pos]
        edges_set = set()
        mask = torch.ones(len(faces), dtype=bool)
        _, face_areas = self.compute_face_normals_and_areas(mesh=mesh,
                                                            faces=faces)
        for face_id, face in enumerate(faces):
            if face_areas[face_id] == 0:
                mask[face_id] = False
                continue
            faces_edges = []
            is_manifold = False
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                if cur_edge in edges_set:
                    is_manifold = True
                    break
                else:
                    faces_edges.append(cur_edge)
            if is_manifold:
                mask[face_id] = False
            else:
                for idx, edge in enumerate(faces_edges):
                    edges_set.add(edge)
        return faces[mask], face_areas[mask]

    @staticmethod
    def build_edge_pairs(edge_nb, edges_count):
        num_edge_pairs = edge_nb.shape[0] * (edge_nb.shape[1] + 1)
        edge_index = np.zeros([2, num_edge_pairs], dtype=np.int64)
        edge_index[0, :] = np.concatenate(
            (np.expand_dims(np.arange(edges_count), axis=1),
             edge_nb), axis=1).flatten()
        edge_index[1, :] = (
                np.linspace(0, num_edge_pairs - 1, num_edge_pairs) // (
                edge_nb.shape[1] + 1))
        return edge_index

    def build_edge_indexes(self, mesh, faces, face_areas):
        """
        Builds gemm_edges and converts it to edge indexes pairs.
        gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
        sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an
              edge is in the gemm_edge entry of the 4 neighboring edges
        for example edge i -> gemm_edges[gemm_edges[i], sides[i]] ==
                                                                [i, i, i, i]
        """
        mesh.ve = [[] for _ in mesh.pos]
        edge_nb = []
        sides = []
        edge2key = dict()
        edges = []
        edges_count = 0
        nb_count = []
        for face_id, face in enumerate(faces):
            faces_edges = []
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                faces_edges.append(cur_edge)
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                faces_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edge_nb.append([-1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1])
                    mesh.ve[edge[0]].append(edges_count)
                    mesh.ve[edge[1]].append(edges_count)
                    mesh.edge_areas.append(0)
                    nb_count.append(0)
                    edges_count += 1
                mesh.edge_areas[edge2key[edge]] += face_areas[face_id] / 3
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[
                    faces_edges[(idx + 1) % 3]]
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[
                    faces_edges[(idx + 2) % 3]]
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[
                    faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[
                    faces_edges[(idx + 2) % 3]]] - 2

        mesh.edges = torch.tensor(edges, dtype=torch.int32)
        edge_nb = torch.tensor(edge_nb, dtype=torch.int64)
        mesh.edge_index = torch.tensor(
            self.build_edge_pairs(edge_nb, edges_count), dtype=torch.long)
        mesh.sides = torch.tensor(sides, dtype=torch.int64)
        mesh.edges_count = edges_count
        mesh.edge_areas = torch.tensor(mesh.edge_areas,
                                       dtype=torch.float32) / np.sum(
            face_areas)

    @staticmethod
    def compute_face_normals_and_areas(mesh, faces):
        face_normals = np.cross(mesh.pos[faces[:, 1]] - mesh.pos[faces[:, 0]],
                                mesh.pos[faces[:, 2]] - mesh.pos[faces[:, 1]])
        face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
        face_normals /= face_areas[:, np.newaxis]
        assert (not np.any(face_areas[:,
                           np.newaxis] == 0)), 'has zero area face: %s' \
                                               % mesh.filename
        face_areas *= 0.5
        return face_normals, face_areas
