from typing import Any, Dict, List

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx


@functional_transform('node_property_split')
class NodePropertySplit(BaseTransform):
    r"""Creates a node-level split with distributional shift based on a given
    node property, as proposed in the `"Evaluating Robustness and Uncertainty
    of Graph Models Under Structural Distributional Shifts"
    <https://arxiv.org/abs/2302.13875v1>`__ paper
    (functional name: :obj:`node_property_split`).

    It splits the nodes in a given graph into five non-intersecting parts
    based on their structural properties.
    This can be used for transductive node prediction tasks with distributional
    shifts.
    It considers the in-distribution (ID) and out-of-distribution (OOD) subsets
    of nodes.
    The ID subset includes training, validation and testing parts, while
    the OOD subset includes validation and testing parts.
    As a result, it creates five associated node mask vectors for each graph,
    three which are for the ID nodes (:obj:`id_train_mask`,
    :obj:`id_val_mask`, :obj:`id_test_mask`), and two which are for the OOD
    nodes (:obj:`ood_val_mask`, :obj:`ood_test_mask`).

    This class implements three particular strategies for inducing
    distributional shifts in a graph — based on **popularity**, **locality**
    or **density**.

    Args:
        property_name (str): The name of the node property to be used
            (:obj:`"popularity"`, :obj:`"locality"`, :obj:`"density"`).
        ratios ([float]): A list of five ratio values for ID training,
            ID validation, ID test, OOD validation and OOD test parts.
            The values must sum to :obj:`1.0`.
        ascending (bool, optional): Whether to sort nodes in ascending order
            of the node property, so that nodes with greater values of the
            property are considered to be OOD (default: :obj:`True`)

    Example:

    .. code-block:: python

        from torch_geometric.transforms import NodePropertySplit
        from torch_geometric.datasets.graph_generator import ERGraph

        data = ERGraph(num_nodes=1000, edge_prob=0.4)()

        property_name = 'popularity'
        ratios = [0.3, 0.1, 0.1, 0.3, 0.2]
        tranaform = NodePropertySplit(property_name, ratios)

        data = transform(data)
    """
    def __init__(
        self,
        property_name: str,
        ratios: List[float],
        ascending: bool = True,
    ):
        if property_name not in {'popularity', 'locality', 'density'}:
            raise ValueError(f"Unexpected 'property_name' "
                             f"(got '{property_name}')")

        if len(ratios) != 5:
            raise ValueError(f"'ratios' must contain 5 values "
                             f"(got {len(ratios)})")

        if sum(ratios) != 1.0:
            raise ValueError(f"'ratios' must sum to 1.0 (got {sum(ratios)})")

        self.property_name = property_name
        self.compute_fn = _property_name_to_compute_fn[property_name]
        self.ratios = ratios
        self.ascending = ascending

    def forward(self, data: Data) -> Data:
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)
        property_values = self.compute_fn(G, self.ascending)
        mask_dict = self._mask_nodes_by_property(property_values, self.ratios)

        for key, mask in mask_dict.items():
            data[key] = mask

        return data

    @staticmethod
    def _compute_popularity_property(G: Any, ascending: bool = True) -> Tensor:
        import networkx.algorithms as A

        property_values = torch.tensor(list(A.pagerank(G).values()))
        property_values *= -1 if ascending else 1
        return property_values

    @staticmethod
    def _compute_locality_property(G: Any, ascending: bool = True) -> Tensor:
        import networkx.algorithms as A

        pagerank_values = torch.tensor(list(A.pagerank(G).values()))

        num_nodes = G.number_of_nodes()
        personalization = dict(zip(range(num_nodes), [0.0] * num_nodes))
        personalization[int(pagerank_values.argmax())] = 1.0

        property_values = torch.tensor(
            list(A.pagerank(G, personalization=personalization).values()))
        property_values *= -1 if ascending else 1
        return property_values

    @staticmethod
    def _compute_density_property(G: Any, ascending: bool = True) -> Tensor:
        import networkx.algorithms as A

        property_values = torch.tensor(list(A.clustering(G).values()))
        property_values *= -1 if ascending else 1
        return property_values

    @staticmethod
    def _mask_nodes_by_property(
        property_values: Tensor,
        ratios: List[float],
    ) -> Dict[str, Tensor]:

        num_nodes = property_values.size(0)
        sizes = (num_nodes * torch.tensor(ratios)).round().long()
        sizes[-1] -= sizes.sum() - num_nodes

        perm = torch.randperm(num_nodes)
        id_size = int(sizes[:3].sum())
        perm = perm[property_values[perm].argsort()]
        perm[:id_size] = perm[:id_size][torch.randperm(id_size)]

        node_splits = perm.split(sizes.tolist())
        names = [
            'id_train_mask',
            'id_val_mask',
            'id_test_mask',
            'ood_val_mask',
            'ood_test_mask',
        ]

        split_masks = {}
        for name, node_split in zip(names, node_splits):
            split_mask = torch.zeros(num_nodes, dtype=torch.bool)
            split_mask[node_split] = True
            split_masks[name] = split_mask
        return split_masks

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.property_name})'


_property_name_to_compute_fn = {
    'popularity': NodePropertySplit._compute_popularity_property,
    'locality': NodePropertySplit._compute_locality_property,
    'density': NodePropertySplit._compute_density_property,
}
