from tempfile import mkstemp
from shutil import move
import torch
import numpy as np
import os
from ..nn.mesh.mesh_union import MeshUnion
from torch_geometric.data import Data


class MeshCNNData(Data):
    r"""MeshCNNData class which holds a mesh data for implementing `"MeshCNN: A
    Network with an Edge" <https://arxiv.org/abs/1809.05910>`_ paper
    This class is a torch geometric Data extension.

    Args:
        filename (str): obj file name
        pos (Tensor [Num_vertices, num_dimensions=3], dtype=float64): vertices
                                                                      positions
        edges (Tensor [Num_edges, 2], dtype=int32): pairs of vertices indexes
                                                    which construct an edge
        sides (Tensor [Num_edges, 4], dtype=int32): values of: 0,1,2,3 -
                                                    sides[E][N] indicates the
                                                    index of edge E for the N
                                                    neighbour.
        edges_count (int): number of edges in the mesh structure.
        ve (Tensor [Num_vertices,
                    Num_connections],
                    dtype=int32 ): array of all edge id connections of
                                   vertex i e.g., ve[i,:] = [edge_ind_j,
                                   edge_ind_k] means vertex i have 2
                                   connections with vertices according to
                                   edges[edge_ind_j, :] and
                                   edges[edge_ind_k, :]
        v_mask (Tensor [Num_vetices], dtype=bool): indicates if a vertex
                                                   still in the mesh obj
                                                   (True - exists,
                                                   False - removed)
        edge_lengths (Tensor [Num_edges], dtype=float64): edges L2 norm
        edge_areas (Tensor [Num_edges],
                   dtype=float64): for each edge it is the sum of 1/3
                                   face(s) area the edge constructs
                                   (maximum 2 faces), normalized by the total
                                   area of all the faces. Used for
                                   segmentation accuracy.
        edge_index (Tensor [2, Num_edges x 5],
                   dtype=int64)): pairs of edge indexes which are neighbours.
        edge_attr (Tensor [Feature_dim, Num_edges],
                 dtype=float64)): edge features. Initial features dimension is
                                  5 and consists of geometric features as state
                                  in the paper.
        label (int): mesh label number.
        """

    def __init__(self, filename='unknown', pos=None, edges=None, sides=None,
                 edges_count=0, ve=None, v_mask=None, edge_lengths=None,
                 edge_areas=[], edge_index=None, edge_attr=None, label=None):
        super(MeshCNNData, self).__init__()
        self.filename = filename
        self.pos = pos
        self.edges = edges
        self.sides = sides
        self.edges_count = edges_count
        self.ve = ve
        self.v_mask = v_mask
        self.edge_lengths = edge_lengths
        self.edge_areas = edge_areas
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.label = label


class MeshCNNHandler:
    r"""The Mesh class structure from the `"MeshCNN: A Network with an Edge"
    <https://arxiv.org/abs/1809.05910>`_ paper
    This class implements holds MeshCNNData structure along with operations on
    the Mesh structure.

    Args:
        mesh_data (MeshCNNData): MeshCNNData structure with input data.
        hold_history (bool): if True - mesh class will hold mesh history on
                             different resolutions (used in un-pooling
                             operator). Default - False.
        export_folder (str): path to data export. Default - an empty string,
                             means no export.
    """

    def __init__(self, mesh_data: MeshCNNData, hold_history: bool = False,
                 export_folder: str = ''):

        self.mesh_data = mesh_data
        self.pool_count = 0
        self.export_folder = export_folder
        self.history_data = None
        if hold_history:
            self.init_history()
        self.export()

    def extract_features(self):
        return self.mesh_data.edge_attr

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.mesh_data.edges[edge_id]
        v_a = self.mesh_data.pos[edge[0]]
        v_b = self.mesh_data.pos[edge[1]]
        # update pA
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.mesh_data.v_mask[edge[1]] = False
        mask = self.mesh_data.edges == edge[1]
        self.mesh_data.ve[edge[0]].extend(self.mesh_data.ve[edge[1]])
        mask_ij = torch.where(mask == True)
        mask_ij = [(j, mask_ij[1][i]) for i, j in enumerate(mask_ij[0])]
        for ij in mask_ij:
            self.mesh_data.edges[ij[0], ij[1]] = edge[0]

    def remove_vertex(self, v):
        self.mesh_data.v_mask[v] = False

    def remove_edge(self, edge_id):
        vs = self.mesh_data.edges[edge_id]
        for v in vs:
            if edge_id not in self.mesh_data.ve[v]:
                print(self.mesh_data.ve[v])
                print(self.mesh_data.filename)
            self.mesh_data.ve[v].remove(edge_id)

    def clean(self, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())

        index_mask = np.array(np.where(edges_mask == True)).flatten()
        index_pyg_mask = [5 * ind_mask + i for ind_mask in index_mask for i in
                          range(5)]
        self.mesh_data.edge_index = self.mesh_data.edge_index[:,
                                    index_pyg_mask]

        self.mesh_data.edges = self.mesh_data.edges[edges_mask]
        self.mesh_data.sides = self.mesh_data.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, index_mask.shape[0])
        new_indices = torch.from_numpy(new_indices)
        self.mesh_data.edge_index[0, :] = new_indices[
            self.mesh_data.edge_index[0, :]]
        self.mesh_data.edge_index[1, :] = torch.tensor(
            [i for i in range(len(index_mask)) for j in range(5)])

        for v_index, ve in enumerate(self.mesh_data.ve):
            update_ve = []
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.mesh_data.ve = new_ve
        self.__clean_history(groups, torch_mask)
        self.pool_count += 1
        self.export()

    def export(self, file=None, vcolor=None):
        if file is None:
            if self.export_folder:
                filename, file_extension = os.path.splitext(
                    self.mesh_data.filename)
                file = '%s/%s_%d%s' % (
                    self.export_folder, filename, self.pool_count,
                    file_extension)
            else:
                return
        faces = []
        vs = self.mesh_data.pos[self.mesh_data.v_mask]
        edge_indexes = np.array(self.mesh_data.edge_index)
        new_indices = np.zeros(self.mesh_data.v_mask.shape[0], dtype=np.int32)
        new_indices[self.mesh_data.v_mask] = np.arange(0, np.ma.where(
            self.mesh_data.v_mask)[0].shape[0])
        for edge_index in range(self.mesh_data.edges_count):
            cycles = self.__get_cycle(edge_indexes, edge_index)
            for cycle in cycles:
                faces.append(self.__cycle_to_face(cycle, new_indices))
        with open(file, 'w+') as f:
            for vi, v in enumerate(vs):
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[
                    vi, 2]) if vcolor is not None else ''
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(faces) - 1):
                f.write("f %d %d %d\n" % (
                    faces[face_id][0] + 1, faces[face_id][1] + 1,
                    faces[face_id][2] + 1))
            f.write("f %d %d %d" % (
                faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            for edge in self.mesh_data.edges:
                f.write("\ne %d %d" % (
                    new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))

    def export_segments(self, segments):
        if not self.export_folder:
            return
        cur_segments = segments
        for i in range(self.pool_count + 1):
            filename, file_extension = os.path.splitext(
                self.mesh_data.filename)
            file = '%s/%s_%d%s' % (
                self.export_folder, filename, i, file_extension)
            fh, abs_path = mkstemp()
            edge_key = 0
            with os.fdopen(fh, 'w') as new_file:
                with open(file) as old_file:
                    for line in old_file:
                        if line[0] == 'e':
                            new_file.write('%s %d' % (
                                line.strip(), cur_segments[edge_key]))
                            if edge_key < len(cur_segments):
                                edge_key += 1
                                new_file.write('\n')
                        else:
                            new_file.write(line)
            os.remove(file)
            move(abs_path, file)
            if i < len(self.history_data['edges_mask']):
                cur_segments = segments[
                               :len(self.history_data['edges_mask'][i])]
                cur_segments = cur_segments[self.history_data['edges_mask'][i]]

    def __get_cycle(self, edge_indexes, edge_id):
        cycles = []
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            if edge_indexes[0, edge_id * 5 + 1 + start_point] == -1:
                continue
            cycles.append([])
            for i in range(3):
                tmp_next_key = edge_indexes[0, next_key * 5 + 1 + next_side]
                tmp_next_side = self.mesh_data.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                edge_indexes[0, 5 * next_key + 1 + next_side] = -1
                edge_indexes[0, 5 * next_key + 1 + next_side + 1 - 2 * (
                        next_side % 2)] = -1
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles

    def __cycle_to_face(self, cycle, v_indices):
        face = []
        for i in range(3):
            v = list(set(self.mesh_data.edges[cycle[i]]) & set(
                self.mesh_data.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face

    def init_history(self):
        self.history_data = {
            'groups': [],
            'edge_index': [self.mesh_data.edge_index.clone()],
            'occurrences': [],
            'old2current': np.arange(self.mesh_data.edges_count,
                                     dtype=np.int32),
            'current2old': np.arange(self.mesh_data.edges_count,
                                     dtype=np.int32),
            'edges_mask': [
                torch.ones(self.mesh_data.edges_count, dtype=torch.bool)],
            'edges_count': [self.mesh_data.edges_count],
        }
        if self.export_folder:
            self.history_data['collapses'] = MeshUnion(
                self.mesh_data.edges_count)

    def union_groups(self, source, target):
        if self.export_folder and self.history_data:
            self.history_data['collapses'].union(
                self.history_data['current2old'][source],
                self.history_data['current2old'][target])
        return

    def remove_group(self, index):
        if self.history_data is not None:
            self.history_data['edges_mask'][-1][
                self.history_data['current2old'][index]] = 0
            self.history_data['old2current'][
                self.history_data['current2old'][index]] = -1
            if self.export_folder:
                self.history_data['collapses'].remove_group(
                    self.history_data['current2old'][index])

    def get_groups(self):
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        return self.history_data['occurrences'].pop()

    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            mask = self.history_data['old2current'] != -1
            self.history_data['old2current'][mask] = np.arange(
                self.mesh_data.edges_count, dtype=np.int32)
            self.history_data['current2old'][0: self.mesh_data.edges_count] = \
                np.ma.where(mask)[0]
            if self.export_folder != '':
                self.history_data['edges_mask'].append(
                    self.history_data['edges_mask'][-1].clone())
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            self.history_data['edge_index'].append(
                self.mesh_data.edge_index.clone())
            self.history_data['edges_count'].append(self.mesh_data.edges_count)

    def unroll_edge_indexes(self):
        self.history_data['edge_index'].pop()
        self.mesh_data.edge_index = self.history_data['edge_index'][-1]

        self.history_data['edges_count'].pop()
        self.mesh_data.edges_count = self.history_data['edges_count'][-1]

    def get_edge_areas(self):
        return self.mesh_data.edge_areas
