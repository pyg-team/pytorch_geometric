import os
import ntpath
from .mesh_augmentations import *
from ..utils.mesh_features import mesh_extract_features
from ..transforms.mesh import MeshData


class MeshPrepare:
    r"""This class is a utility class to read and create Mesh structure for the MeshCNN: `"MeshCNN: A Network with an
    Edge" <https://arxiv.org/abs/1809.05910>`_ paper
    Args:
        file (str): Number of input channels of edge features.
        num_aug (int): Number of augmentations to apply on mesh.
        aug_slide_verts (float, optional): values between 0.0 to 1.0 - if set above 0 will apply shift along mesh
                                           surface with percentage of aug_slide_verts value.
        aug_scale_verts (bool, optional): If set to `False` - do not apply scale vertex augmentation
                                          If set to `True` - apply non-uniformly scale on the mesh
        aug_flip_edges (float, optional): values between 0.0 to 1.0 - if set above 0 will apply random flip of edges
                                          with percentage of aug_flip_edges value.
    """

    def __init__(self, raw_file: str, num_aug: int = 0, aug_slide_verts: float = 0.0,
                 aug_scale_verts: bool = False, aug_flip_edges: float = 0.0, aug_file: str = ''):
        self.file = raw_file
        self.num_aug = num_aug
        self.aug_slide_verts = aug_slide_verts
        self.aug_scale_verts = aug_scale_verts
        self.aug_flip_edges = aug_flip_edges
        self.mesh_data = MeshData()

        load_path = aug_file
        if load_path == '':
            load_path = self.get_mesh_path(raw_file)
        if os.path.exists(load_path):
            data = np.load(load_path, encoding='latin1', allow_pickle=True)
            self.mesh_data.vs = data.__getitem__('vs')
            self.mesh_data.edges = data.__getitem__('edges')
            self.mesh_data.edges_count = int(data.__getitem__('edges_count'))
            self.mesh_data.ve = data.__getitem__('ve')
            self.mesh_data.v_mask = data.__getitem__('v_mask')
            self.mesh_data.filename = str(data.__getitem__('filename'))
            self.mesh_data.edge_lengths = data.__getitem__('edge_lengths')
            self.mesh_data.edge_areas = data.__getitem__('edge_areas')
            self.mesh_data.features = data.__getitem__('features')
            self.mesh_data.sides = data.__getitem__('sides')
            self.mesh_data.edge_index = data.__getitem__('edge_index')
        else:
            self.mesh_data = self.from_scratch()
            np.savez_compressed(load_path, vs=self.mesh_data.vs, edges=self.mesh_data.edges,
                                edges_count=int(self.mesh_data.edges_count), ve=self.mesh_data.ve,
                                v_mask=self.mesh_data.v_mask, filename=str(self.mesh_data.filename),
                                sides=self.mesh_data.sides, edge_lengths=self.mesh_data.edge_lengths,
                                edge_areas=self.mesh_data.edge_areas, features=self.mesh_data.features,
                                edge_index=self.mesh_data.edge_index)

    def get_mesh_path(self, file):
        filename, _ = os.path.splitext(file)
        dir_name = os.path.dirname(filename)
        prefix = os.path.basename(filename)
        load_dir = os.path.join(dir_name, 'cache')
        load_file = os.path.join(load_dir, '%s_%03d.npz' % (prefix, np.random.randint(0, self.num_aug)))
        if not os.path.isdir(load_dir):
            os.makedirs(load_dir, exist_ok=True)
        return load_file

    def from_scratch(self):
        mesh_data = MeshData()
        mesh_data.vs, faces = self.fill_from_file(mesh_data)
        mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)

        faces, face_areas = self.remove_non_manifolds(mesh_data, faces)
        if self.num_aug > 1:
            faces = self.mesh_augmentations(mesh_data, faces)
        self.build_edge_indexes(mesh_data, faces, face_areas)
        if self.num_aug > 1:
            self.mesh_post_augmentations(mesh_data)
        mesh_data.features = mesh_extract_features(mesh_data)
        return mesh_data

    def mesh_augmentations(self, mesh, faces):
        if self.aug_scale_verts:
            scale_verts(mesh.vs)
        if self.aug_flip_edges > 0.0:
            faces = flip_edges(mesh.vs, faces, self.aug_flip_edges)
        return faces

    def mesh_post_augmentations(self, mesh):
        if self.aug_slide_verts > 0.0:
            mesh.shifted = slide_verts(mesh.edges, mesh.edge_index, mesh.edges_count,
                                       mesh.ve, mesh.vs, self.aug_slide_verts)

    def fill_from_file(self, mesh):
        mesh.filename = ntpath.split(self.file)[1]
        mesh.fullfilename = self.file
        vs, faces = [], []
        f = open(self.file)
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                                   for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
        f.close()
        vs = np.asarray(vs)
        faces = np.asarray(faces, dtype=int)
        assert np.logical_and(faces >= 0, faces < len(vs)).all()
        return vs, faces

    def remove_non_manifolds(self, mesh, faces):
        mesh.ve = [[] for _ in mesh.vs]
        edges_set = set()
        mask = np.ones(len(faces), dtype=bool)
        _, face_areas = self.compute_face_normals_and_areas(mesh=mesh, faces=faces)
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
        edge_index[0, :] = np.concatenate((np.expand_dims(np.arange(edges_count), axis=1),
                                           edge_nb), axis=1).flatten()
        edge_index[1, :] = (np.linspace(0, num_edge_pairs - 1, num_edge_pairs) // (edge_nb.shape[1] + 1))
        return edge_index

    def build_edge_indexes(self, mesh, faces, face_areas):
        """
        Builds gemm_edges and converts it to edge indexes pairs.
        gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
        sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
        for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
        """
        mesh.ve = [[] for _ in mesh.vs]
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
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2

        mesh.edges = np.array(edges, dtype=np.int32)
        edge_nb = np.array(edge_nb, dtype=np.int64)
        mesh.edge_index = self.build_edge_pairs(edge_nb, edges_count)
        mesh.sides = np.array(sides, dtype=np.int64)
        mesh.edges_count = edges_count
        mesh.edge_areas = np.array(mesh.edge_areas, dtype=np.float32) / np.sum(face_areas)

    @staticmethod
    def compute_face_normals_and_areas(mesh, faces):
        face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                                mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
        face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
        face_normals /= face_areas[:, np.newaxis]
        assert (not np.any(face_areas[:, np.newaxis] == 0)), 'has zero area face: %s' % mesh.filename
        face_areas *= 0.5
        return face_normals, face_areas
