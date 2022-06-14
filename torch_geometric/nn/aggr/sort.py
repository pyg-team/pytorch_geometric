from typing import Optional

from torch import Tensor

from torch_geometric.nn.aggr import Aggregation


class SortPooling(Aggregation):
    r"""The pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
    where node features are sorted in descending order based on their last
    feature channel. The first :math:`k` elements form the output of the layer.

    Args:
        k (int): The number of elements to maintain.
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: Tensor, index: Optional[Tensor] = None, *,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        fill_value = x.min().item() - 1
        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim, fill_value)

        _, perm = x[:, :, -1].sort(dim=-1, descending=True)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'
