from typing import List, Optional, Union

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import index_to_mask, mask_to_index


def attrs_ends_with(data: Data, suffix: str):
    return tuple(filter(lambda s: s.endswith(suffix), data.keys))


def update_suffix(name, old_suffix, new_suffix):
    if name.endswith(old_suffix):
        name = name[:-len(old_suffix)]

    return name + new_suffix


@functional_transform('index_to_mask')
class IndexToMask(BaseTransform):
    r"""Converts indices to a mask representation

    Args:
        attrs (str, [str], optional): If given, will only perform index to mask
            conversion for the given attributes. If omitted, will infer the
            attributes from the suffix `_indices`  (default: :obj:`None`)
        sizes (int, [int], optional). The size of the mask. If set to
            :obj:`None`, a minimal sized mask is created.
            (default: :obj:`None`)
    """
    def __init__(self, attrs: Optional[Union[str, List[str]]] = None,
                 sizes: Optional[Union[int, List[int]]] = None):
        self.attrs = (attrs, ) if isinstance(attrs, str) else attrs
        self.sizes = sizes

    def __call__(self, data: Data) -> Data:
        suffix = "_indices"
        attrs = self.attrs if self.attrs else attrs_ends_with(data, suffix)
        sizes = self.sizes if self.sizes else (None, ) * len(attrs)

        if isinstance(sizes, int):
            sizes = (sizes, ) * len(attrs)

        if len(attrs) != len(sizes):
            raise ValueError(
                f"The number of attributes {len(attrs)} must match "
                f"the number of sizes provided {len(sizes)}.")

        for index_attr, size in zip(attrs, sizes):
            index = getattr(data, index_attr)
            mask = index_to_mask(index, size=size)
            new_attr = update_suffix(index_attr, suffix, "_mask")
            setattr(data, new_attr, mask)

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(attrs={self.attrs}, '
                f'size={self.sizes})')


@functional_transform('mask_to_index')
class MaskToIndex(BaseTransform):
    r"""Converts a mask to an index representation.

    Args:
        attrs (str, [str], optional): If given, will only perform mask to index
            conversion for the given attributes.  If omitted, will infer the
            attributes from the suffix `_mask` (default: :obj:`None`)
    """
    def __init__(self, attrs: Optional[Union[str, List[str]]] = None):
        self.attrs = [attrs] if isinstance(attrs, str) else attrs

    def __call__(self, data: Data) -> Data:
        suffix = "_mask"
        attrs = self.attrs if self.attrs else attrs_ends_with(data, suffix)

        for mask_attr in attrs:
            mask = getattr(data, mask_attr)
            index = mask_to_index(mask)
            new_attr = update_suffix(mask_attr, suffix, "_indices")
            setattr(data, new_attr, index)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(attrs={self.attrs})'
