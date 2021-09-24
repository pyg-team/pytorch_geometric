import torch
import torch.nn as nn
from threading import Thread
from ..mesh.mesh_union import MeshUnion
from ..mesh.mesh_edge_padding import get_meshes_edge_index
import numpy as np
from heapq import heappop, heapify


class MeshPool(nn.Module):
    r"""The Mesh Pooling operator from the `"MeshCNN: A Network with an Edge"
     <https://arxiv.org/abs/1809.05910>`_ paper
    This class implements the pooling operation on a Mesh structure.
    We define mesh pooling as a series of edge collapse operations, where each
    edge collapse converts five edges into two.
    Edge collapse: For an edge :math:`\mathbf{e}` with edge neighbors
    :math:`\left(a, b \right)` and :math:`\left(c,d \right)`, which define two
    triangles, respectively, we create two new edge features according to:
    .. math::
    `\mathbf{MeshConv(p)_i} = \mathbf{avg}\left ( \mathbf(a)_i, \mathbf(b)_i,
                                                  \mathbf(e)_i \right )`
    `\mathbf{MeshConv(q)_i} = \mathbf{avg}\left ( \mathbf(c)_i, \mathbf(d)_i,
                                                  \mathbf(e)_i \right )`
    where :math:`\mathbf{p}` and :math:`\mathbf{q}` are the resulting two new
    feature vectors, and :math:`\mathbf{i}` represents the feature channel
    index.

    The edge collapse order is according to L2-norm of the edge features, where
    the smallest norms collapse first. Edge collapse stops according to the
    target pooling resolution.

    Args:
        target_resolution (int): number of edges after pooling operation
        multi_thread (bool, optional): use multi-thread for the pooling process
                                       (default - False)
    """

    def __init__(self, target_resolution: int, multi_thread: bool = False):
        super(MeshPool, self).__init__()
        self.__out_target = target_resolution
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_edges = [-1, -1]

    def forward(self, fe, meshes):
        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(
                    Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1,
                                                         self.__out_target)
        out_edge_index = get_meshes_edge_index(meshes)
        return out_features, out_edge_index

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        queue = MeshPool.__build_queue(
            self.__fe[mesh_index, :, :mesh.mesh_data.edges_count],
            mesh.mesh_data.edges_count)
        mask = np.ones(mesh.mesh_data.edges_count, dtype=np.bool)
        edge_groups = MeshUnion(mesh.mesh_data.edges_count, self.__fe.device)
        while mesh.mesh_data.edges_count > self.__out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            if mask[edge_id]:
                self.__pool_edge(mesh, edge_id, mask, edge_groups)
        mesh.clean(mask, edge_groups)
        fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask,
                                          self.__out_target)
        self.__updated_fe[mesh_index] = fe

    def __pool_edge(self, mesh, edge_id, mask, edge_groups):
        if self.has_boundaries(mesh.mesh_data, edge_id):
            return False
        elif self.__clean_side(mesh, edge_id, mask, edge_groups, 0) \
                and self.__clean_side(mesh, edge_id, mask, edge_groups, 2) \
                and self.__is_one_ring_valid(mesh.mesh_data, edge_id):
            self.__merge_edges[0] = self.__pool_side(mesh, edge_id, mask,
                                                     edge_groups, 0)
            self.__merge_edges[1] = self.__pool_side(mesh, edge_id, mask,
                                                     edge_groups, 2)
            mesh.merge_vertices(edge_id)
            mask[edge_id] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_id)
            mesh.mesh_data.edges_count -= 1
            return True
        else:
            return False

    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        if mesh.mesh_data.edges_count <= self.__out_target:
            return False
        invalid_edges = MeshPool.__get_invalids(mesh, edge_id, edge_groups,
                                                side)
        while len(
                invalid_edges) != 0 and mesh.mesh_data.edges_count > \
                self.__out_target:
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges)
            if mesh.mesh_data.edges_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh.mesh_data, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups,
                                                side)
        return True

    @staticmethod
    def has_boundaries(mesh_data, edge_id):
        edge_id_pyg = edge_id * 5
        for edge in mesh_data.edge_index[0, edge_id_pyg + 1:edge_id_pyg + 5]:
            if edge == -1 or -1 \
                    in mesh_data.edge_index[0, 5 * edge + 1:5 * edge + 5]:
                return True
        return False

    @staticmethod
    def __is_one_ring_valid(mesh_data, edge_id):
        v_a = set(np.array(mesh_data.edges[np.array(
            mesh_data.ve[np.array(mesh_data.edges[edge_id, 0])])].reshape(-1)))
        v_b = set(np.array(mesh_data.edges[np.array(
            mesh_data.ve[np.array(mesh_data.edges[edge_id, 1])])].reshape(-1)))
        shared = v_a & v_b - set(np.array(mesh_data.edges[edge_id]))
        return len(shared) == 2

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        info = MeshPool.__get_face_info(mesh.mesh_data, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.__redirect_edges(mesh.mesh_data, key_a, side_a - side_a % 2,
                              other_keys_b[0],
                              mesh.mesh_data.sides[key_b, other_side_b])
        self.__redirect_edges(mesh.mesh_data, key_a, side_a - side_a % 2 + 1,
                              other_keys_b[1],
                              mesh.mesh_data.sides[key_b, other_side_b + 1])
        MeshPool.__union_groups(mesh, edge_groups, key_b, key_a)
        MeshPool.__union_groups(mesh, edge_groups, edge_id, key_a)
        mask[key_b] = False
        MeshPool.__remove_group(mesh, edge_groups, key_b)
        mesh.remove_edge(key_b)
        mesh.mesh_data.edges_count -= 1
        return key_a

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        info = MeshPool.__get_face_info(mesh.mesh_data, edge_id, side)
        key_a, key_b, side_a, side_b, other_side_a, other_side_b, \
        other_keys_a, other_keys_b = info
        shared_items = MeshPool.__get_shared_items(other_keys_a, other_keys_b)
        if len(shared_items) == 0:
            return []
        else:
            assert (len(shared_items) == 2)
            middle_edge = other_keys_a[shared_items[0]]
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.mesh_data.sides[
                key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.mesh_data.sides[
                key_b, other_side_b + 1 - shared_items[1]]
            MeshPool.__redirect_edges(mesh.mesh_data, edge_id, side,
                                      update_key_a, update_side_a)
            MeshPool.__redirect_edges(mesh.mesh_data, edge_id, side + 1,
                                      update_key_b, update_side_b)
            MeshPool.__redirect_edges(mesh.mesh_data, update_key_a,
                                      MeshPool.__get_other_side(update_side_a),
                                      update_key_b,
                                      MeshPool.__get_other_side(update_side_b))
            MeshPool.__union_groups(mesh, edge_groups, key_a, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_b, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_a, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge,
                                    update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, key_b, update_key_b)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge,
                                    update_key_b)
            return [key_a, key_b, middle_edge]

    @staticmethod
    def __redirect_edges(mesh_data, edge_a_key, side_a, edge_b_key, side_b):
        mesh_data.edge_index[0, 5 * edge_a_key + 1 + side_a] = edge_b_key
        mesh_data.edge_index[0, 5 * edge_b_key + 1 + side_b] = edge_a_key
        mesh_data.sides[edge_a_key, side_a] = side_b
        mesh_data.sides[edge_b_key, side_b] = side_a

    @staticmethod
    def __get_shared_items(list_a, list_b):
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items

    @staticmethod
    def __get_other_side(side):
        return side + 1 - 2 * (side % 2)

    @staticmethod
    def __get_face_info(mesh_data, edge_id, side):
        edge_id_pyg = 5 * edge_id
        key_a = mesh_data.edge_index[0, edge_id_pyg + 1 + side].item()
        key_b = mesh_data.edge_index[0, edge_id_pyg + 1 + side + 1].item()
        side_a = mesh_data.sides[edge_id, side]
        side_b = mesh_data.sides[edge_id, side + 1]
        other_side_a = (side_a - (side_a % 2) + 2) % 4
        other_side_b = (side_b - (side_b % 2) + 2) % 4
        key_a_pyg = key_a * 5
        other_keys_a = [
            mesh_data.edge_index[0, key_a_pyg + 1 + other_side_a].item(),
            mesh_data.edge_index[0, key_a_pyg + 1 + other_side_a + 1].item()]
        key_b_pyg = key_b * 5
        other_keys_b = [
            mesh_data.edge_index[0, key_b_pyg + 1 + other_side_b].item(),
            mesh_data.edge_index[0, key_b_pyg + 1 + other_side_b + 1].item()]

        return key_a, key_b, side_a, side_b, other_side_a, other_side_b, \
               other_keys_a, other_keys_b

    @staticmethod
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        vertex = set(np.array(mesh.mesh_data.edges[invalid_edges[0]]))
        for edge_key in invalid_edges:
            vertex &= set(np.array(mesh.mesh_data.edges[edge_key]))
            mask[edge_key] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_key)
        mesh.mesh_data.edges_count -= 3
        vertex = list(vertex)
        assert (len(vertex) == 1)
        mesh.remove_vertex(vertex[0])

    @staticmethod
    def __build_queue(features, edges_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device,
                                dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap

    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        edge_groups.union(source, target)
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        edge_groups.remove_group(index)
        mesh.remove_group(index)
