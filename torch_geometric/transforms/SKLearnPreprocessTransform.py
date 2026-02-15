from typing import List, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('sklearn_preprocess')
class SKLearnPreprocessTransform(BaseTransform):
    r"""Column-normalizes the attributes given in :obj:`attrs` to sum-up to standardize or scale them.

    Mainly useful for edges, if the scalar for the collective data edges is stored
    as data[(node1,to,node2)].edge_scaler = scaler

    where the scaler can be StandardScaler or MinMaxScaler from scikit-learn that
    has been fit to the total dataset

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["edge_attr"]`)
    """
    def __init__(self, attrs: List[str] = ["edge_attr"]):
        self.attrs = attrs

    def __call__(self, data: Union[Data, HeteroData]):
        for edge, store in list(zip(data.edge_types, data.edge_stores)):
            for key, value in store.items(*self.attrs):
                if "edge_scaler" in store.keys():
                    scaler = store["edge_scaler"]
                    # needs to be a float tensor since this is the edge_attr
                    value = torch.tensor(scaler.transform(value),
                                         dtype=torch.float)
                    store[key] = value
                    del store["edge_scaler"]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
