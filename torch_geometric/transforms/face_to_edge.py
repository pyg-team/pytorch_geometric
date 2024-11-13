import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected


@functional_transform('face_to_edge')
class FaceToEdge(BaseTransform):
    r"""Converts mesh faces of shape :obj:`[3, num_faces]` or
    :obj:`[4, num_faces]` to edge indices of shape :obj:`[2, num_edges]`
    (functional name: :obj:`face_to_edge`).

    This transform supports both 2D triangular faces, represented by a
    tensor of shape :obj:`[3, num_faces]`, and 3D tetrahedral mesh faces,
    represented by a tensor of shape :obj:`[4, num_faces]`. It will convert
    these faces into edge indices, where each edge is defined by the indices
    of its two endpoints.

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed.
    """
    def __init__(self, remove_faces: bool = True) -> None:
        self.remove_faces = remove_faces

    def forward(self, data: Data) -> Data:
        if hasattr(data, 'face'):
            assert data.face is not None
            face = data.face

            if face.size(0) not in [3, 4]:
                raise RuntimeError(f"Expected 'face' tensor with shape "
                                   f"[3, num_faces] or [4, num_faces] "
                                   f"(got {list(face.size())})")

            if face.size()[0] == 3:
                edge_index = torch.cat([
                    face[:2],
                    face[1:],
                    face[::2],
                ], dim=1)
            else:
                assert face.size()[0] == 4
                edge_index = torch.cat([
                    face[:2],
                    face[1:3],
                    face[2:4],
                    face[::2],
                    face[1::2],
                    face[::3],
                ], dim=1)

            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

            data.edge_index = edge_index
            if self.remove_faces:
                data.face = None

        return data
