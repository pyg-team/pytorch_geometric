import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('mesh_to_dual_graph')
class MeshToDualGraph(BaseTransform):
    r"""Constructs the dual graph  :math:`G^* = (V^*, E^*)` of the given
    _triangular_ mesh :math:`G=(V,F)`, where a unique vertex :math:`v^*`
    in :math:`V^*` corresponds to a unique face :math:`f` in :math:`F`,
    and an edge :math:`e = (v^*, v^{*'})` is in :math:`E^*` if and only if
    the two corresponding faces :math:`f, f' \in F` in are adjacent to
    each other in the orginal mesh :math:`G`.

    The triangular mesh should be a :obj:`torch_geometric.data.Data` object
    which has attributes :obj:`data.pos` of shape
    :obj:`[number_of_vertices, 3]` (vertex positions)
    and by :obj:`data.face` of shape :obj:`[number_of_faces, 3]`
    (three vertices in data.post).

    In the dual graph, each face in the original mesh is represented by a node,
    and an edge exists between two nodes if their corresponding
    faces share an edge in the original graph. Explicitly, let
    :math:`\phi: V^* \mapsto F` assign each node in the dual graph to a unique
    face in :math:`G`. An edge :math:`e^* = (v^*, v'^*)` is in the dual graph
    when the corresponding faces :math:`\phi(v^*), \phi(v'^*)` in :math:`G`,
    are adjacent to each other (i.e) they share an edge.

    Args:
        data (torch_geometric.data.Data): The input mesh data object where
            :obj:`data.pos` are the vertex positions of shape
            :obj:`[num_vertices, 3]`, and :obj:`data.face` has
            shape :obj:`[3, num_faces]`.
        sorted (bool, optional): If set to :obj:`True`, will sort
            neighbor indices for each node in ascending order.
            (default: :obj:`False`)

    Returns:
        torch_geometric.data.Data: A new :obj:`Data` object where
        :obj:`edge_index`
        is of shape :math:`[2,|V^*|]`, and defines the adjacency of
        vertices in the dual graph and :obj:`num_nodes`
        equals the number of faces in the original mesh.

    Example:
        >>> import torch
        >>> from torch_geometric.data import Data
        >>> from torch_geometric.transforms import mesh_to_dual_graph
        >>>
        >>> # Define a simple square mesh made of two triangles:
        >>> pos = torch.tensor([
        ...     [0, 0, 0],
        ...     [1, 0, 0],
        ...     [1, 1, 0],
        ...     [0, 1, 0]
        ... ], dtype=torch.float)
        >>> face = torch.tensor([
        ...     [0, 1, 2],
        ...     [0, 2, 3]
        ... ], dtype=torch.long)
        >>>
        >>> # Wrap in a PyG Data object, pos must have shape [num_nodes, 3]
        >>> # and face must have shape [3, num_faces]
        >>> data = Data(pos=pos, face=face.t())
        >>>
        >>> # Convert the triangular mesh to the dual graph
        >>> dual_graph = mesh_to_dual_graph(data)
        >>>
        >>> # NOTE: While this transform takes in a tensor in triangular "mesh"
        >>> # format, that is, a tensor with attributes pos and face
        >>> # of shapes [num_vertices, 3]
        >>> # [3, num_faces], respectively, it returns a tensor in that
        >>> # describes
        >>> # a graph in "pytorch_geometric" format, that is a tensor which
        >>> # attributes
        >>> # attributes pos and edge_index of shapes
        >>> # [num_vertices, 3] and [2, num_edges], respectively.
        >>> # dual_data.edge_index holds the connectivity between the two faces
        >>> print(dual_data)
        Data(edge_index=[2, 1], pos=[2,3])

    .. note::

        - Although this transform takes in a triangle mesh as input,
          it outputs a pytorch geometric tensor in coo format. That is
          because the dual of a triangle mesh may have faces of varying sides.
        - Technically, dual graphs are often defined for planar graphs,
          but the concept is commonly applied to meshes as well.
        - This function assumes that all faces in :obj:`data.face` are valid
          triangles.
        - The dual graph is only fully planar if the original mesh is planar,
          but
          constructing duals of 3D meshes is still commonly used for mesh
          but
          constructing duals of 3D meshes is still commonly used for mesh
          analysis.
          analysis.

    .. seealso::

        - `Dual Graph (Wikipedia) <https://en.wikipedia.org/wiki/Dual_graph>`_
        - `Gross, Jonathan L.; Yellen, Jay, eds. (2003),
          Handbook of Graph Theory,
          CRC Press, p. 724, ISBN 978-1-58488-090-5.
          <https://books.google.com/books?id=mKkIGIea_BkC&lpg=PA724>`_
    """
    def forward(self, data: Data) -> Data:
        """Compute the dual of a mesh
        Args:
            data (torch_geometric.data.Data): The input mesh data object where
                :obj:`data.pos` are the vertex positions of shape
                :obj:`[num_vertices, 3]`, and :obj:`data.face` has
                shape :obj:`[3, num_faces]`.
            sorted (bool, optional): If set to :obj:`True`, will sort
                neighbor indices for each node in ascending order.
                (default: :obj:`False`).

        Returns:
            torch_geometric.data.Data: A new :obj:`Data` object where
            :obj:`edge_index`
            is of shape :math:`[2,|V^*|]`, and defines the adjacency of
            vertices in the dual graph and :obj:`num_nodes`
            equals the number of faces in the original mesh.
        """
        assert data.pos is not None, ('Argument `data` must have attribute \
        `pos`')

        assert data.face is not None, ('Argument `data` must have attribute \
        `face`')
        if data.pos.size(1) != 3:
            raise RuntimeError(
                f'data.pos is shape ({data.face.shape}) which is not equal to \
                shape (num_vertices, 3)')
        if data.face.size(0) not in [3, 4]:
            raise RuntimeError(
                f'data.face is shape ({data.face.shape}) which is not equal to \
                shape {[f"({x}, num_faces)" for x in [3, 4]]}')

        centroids = torch.mean(data.pos[data.face], dim=0)
        # Build out adjacency.
        # TODO: Is there a faster way?
        # TODO: Should I use int32
        # A key value pair, where a key is a tuple of vertices
        # and the value is the corresponding face index in data.face
        visited_edges = {}
        dual_edges = torch.empty((2, data.face.size(1) * data.face.size(0)),
                                 dtype=torch.int64)
        _dual_face_counter = 0
        for face_idx, face in enumerate(data.face.t()):
            for i in range(face.size(0)):
                v1 = int(face[i].item())
                v2 = int(face[(i + 1) % face.size(0)].item())
                # Represent the edge as a sorted tuple.
                edge = (min(v1, v2), max(v1, v2))
                if edge not in visited_edges:
                    visited_edges[edge] = face_idx
                else:
                    adj_face_idx = visited_edges[edge]
                    # This implies the edge has been visited twice already
                    if adj_face_idx == -1:
                        raise ValueError(f'Edge {edge} can be incidient \
                        to at most two faces.')
                    else:
                        dual_edges[:, _dual_face_counter] = torch.tensor(
                            [face_idx, adj_face_idx])
                        _dual_face_counter = _dual_face_counter + 1
                        dual_edges[:, _dual_face_counter] = torch.tensor(
                            [adj_face_idx, face_idx])
                        _dual_face_counter = _dual_face_counter + 1
                        # If edge has been observed in two faces,
                        # we mark it as complete.
                        visited_edges[edge] = -1
        dual = Data(pos=centroids, edge_index=dual_edges)
        return dual
