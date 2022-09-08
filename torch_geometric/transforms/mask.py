from typing import List, Optional, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data.storage import BaseStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import index_to_mask, mask_to_index

AnyData = Union[Data, HeteroData]


def transform_attrs(attrs: Optional[List[str]], store: BaseStorage,
                    suffix: str):
    if attrs is not None:
        return attrs
    return tuple(filter(lambda s: s.endswith(suffix), store.keys()))


def update_suffix(name: str, old_suffix: str, new_suffix: str):
    if name.endswith(old_suffix):
        name = name[:-len(old_suffix)]

    return name + new_suffix


@functional_transform('index_to_mask')
class IndexToMask(BaseTransform):
    r"""Converts indices to a mask representation

    Args:
        attrs (str, [str], optional): If given, will only perform index to mask
            conversion for the given attributes. If omitted, will infer the
            attributes from the suffix `_idx`. (default: :obj:`None`)
        sizes (int, [int], optional). The size of the mask. If set to
            :obj:`None`, an automatically sized tensor is returned. The number
            of nodes will be used by default, except for edge attributes which
            will use the number of edges as the mask size.
            (default: :obj:`None`)
        replace (bool, optional): if set to :obj:`True` replaces the index
            attributes with mask tensors. (default: :obj:`False`)
    """
    def __init__(self, attrs: Optional[Union[str, List[str]]] = None,
                 sizes: Optional[Union[int, List[int]]] = None,
                 replace: bool = False):
        self.attrs = (attrs, ) if isinstance(attrs, str) else attrs
        self.sizes = sizes
        self.replace = replace

    def __call__(self, data: AnyData) -> AnyData:
        suffix = "_idx"

        for store in data.stores:
            attrs = transform_attrs(self.attrs, store, suffix)
            sizes = self.sizes if self.sizes else (None, ) * len(attrs)

            if isinstance(sizes, int):
                sizes = (sizes, ) * len(attrs)

            if len(attrs) != len(sizes):
                raise ValueError(
                    f"The number of attributes {len(attrs)} must match "
                    f"the number of sizes provided {len(sizes)}.")

            for index_attr, size in zip(attrs, sizes):
                size = self.auto_size(store, index_attr, size)
                mask = index_to_mask(store[index_attr], size=size)

                if not self.replace:
                    index_attr = update_suffix(index_attr, suffix, "_mask")

                store[index_attr] = mask

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(attrs={self.attrs}, '
                f'size={self.sizes}, replace={self.replace})')

    @staticmethod
    def auto_size(data: AnyData, key: str, size: Optional[int]):
        if size is not None:
            return size

        return data.num_edges if data.is_edge_attr(key) else data.num_nodes


@functional_transform('mask_to_index')
class MaskToIndex(BaseTransform):
    r"""Converts a mask to an index representation.

    Args:
        attrs (str, [str], optional): If given, will only perform mask to index
            conversion for the given attributes.  If omitted, will infer the
            attributes from the suffix `_mask` (default: :obj:`None`)
        replace (bool, optional): if set to :obj:`True` replaces the mask
            attributes with index tensors. (default: :obj:`False`)
    """
    def __init__(self, attrs: Optional[Union[str, List[str]]] = None,
                 replace: bool = False):
        self.attrs = (attrs, ) if isinstance(attrs, str) else attrs
        self.replace = replace

    def __call__(self, data: AnyData) -> AnyData:
        suffix = "_mask"

        for store in data.stores:
            attrs = transform_attrs(self.attrs, store, suffix)

            for mask_attr in attrs:
                index = mask_to_index(store[mask_attr])

                if not self.replace:
                    mask_attr = update_suffix(mask_attr, suffix, "_idx")

                store[mask_attr] = index

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(attrs={self.attrs}, '
                f'replace={self.replace})')
