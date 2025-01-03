import warnings
from typing import List, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data.dist_uva_tensor import DistEmbedding, DistTensor
from torch_geometric.transforms import BaseTransform


@functional_transform('transform_features_to_wholememory')
class ToWholeGraphFeatures(BaseTransform):
    r"""A transform that moves graph features into wholememory managed distributed uva or cuda tensors.

    .. note::

        In case of composing multiple transforms, it is best to convert the
        :obj:`data` object via :class:`ToWholeGraphFeatures` as late as possible.

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
        device (str): The device to store the distributed tensors.
            (default: :obj:`"cpu"`)
        cache_ratio (float): Expected cache ratio. A higher `cache_ratio` allocates
            more cache space, optimizing for increased cache hits.
            `0` means no caching, and it must be strictly less than `1`.
            (default: :obj:`0.5`)
    """
    def __init__(self, attrs: List[str] = ["x"], device: str = "cpu",
                 cache_ratio: float = 0.5):
        self.attrs = attrs
        self.device = device
        self.cache_ratio = cache_ratio
        warnings.warn(
            "'transform_features_to_wholememory' is a beta feature. "
            "It may not support all tensor-like operators and could exhibit limited functionality."
        )

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if not isinstance(value, DistTensor) and value.numel() > 0:
                    if value.dim() == 1:
                        # No need to unsqueeze if WholeGraph fix this https://github.com/rapidsai/wholegraph/pull/229
                        dist_tensor = DistTensor(value.unsqueeze(1),
                                                 device=self.device)
                    else:
                        dist_tensor = DistEmbedding(
                            value, device=self.device,
                            cache_ratio=self.cache_ratio)
                    store[key] = dist_tensor
        return data
