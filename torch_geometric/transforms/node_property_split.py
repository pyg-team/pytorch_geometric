from typing import Any, Dict, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx


@functional_transform("node_property_split")
class NodePropertySplit(BaseTransform):
    r"""Creates a node-level split with distributional shift based on a given
    node property by adding :obj:`split_masks` to the
    :class:`~torch_geometric.data.Data` object, as proposed in the
    `Evaluating Robustness and Uncertainty of Graph Models Under Structural
    Distributional Shifts <https://arxiv.org/abs/2302.13875v1>`__ paper
    (functional name: :obj:`node_property_split`)

    It splits the nodes in a given graph into 5 non-intersecting parts
    based on their structural properties. This can be used for transductive
    node prediction task with distributional shifts. It considers the
    in-distribution (ID) and out-of-distribution (OOD) subsets of nodes.
    The ID subset includes training, validation and testing parts, while
    the OOD subset includes validation and testing parts. As a result, it
    creates 5 associated node mask arrays for each graph, 3 of which are
    for the ID nodes: :obj:`"in_train_mask"`, :obj:`"in_valid_mask"`,
    :obj:`"in_test_mask"`, and the remaining 2 — for the OOD nodes:
    :obj:`"out_valid_mask"`, :obj:`"out_test_mask"`.

    This class implements 3 particular strategies for inducing
    distributional shifts in graph — based on
    **popularity**, **locality** or **density**.

    Args:
        property_name (str): The name of the node property to be used,
            which must be :obj:`"popularity"`, :obj:`"locality"`
            or :obj:`"density"`.
        part_ratios (list): A list of 5 ratio values for training,
            ID validation, ID test, OOD validation and OOD test parts.
            The values must sum to 1.0.
        ascending (bool, optional): Whether to sort nodes in the ascending
            order of the node property, so that nodes with greater values
            of the property are considered to be OOD (default: :obj:`True`)
        random_seed (int, optional): Random seed to fix for the initial
            permutation of nodes. It is used to create a random order for
            the nodes that have the same property values or belong to the
            ID subset. (default: :obj:`None`)

    Example:

    .. code-block::

        from torch_geometric.transforms import NodePropertySplit
        from torch_geometric.datasets.graph_generator import ERGraph

        data = ERGraph(num_nodes=1000, edge_prob=0.4)()

        property_name = 'popularity'
        part_ratios = [0.3, 0.1, 0.1, 0.3, 0.2]
        tranaform = NodePropertySplit(property_name, part_ratios)

        data = transform(data)
    """
    def __init__(
        self,
        property_name: str,
        part_ratios: List[float],
        ascending: bool = True,
        random_seed: Optional[int] = None,
    ):
        assert property_name in [
            "popularity",
            "locality",
            "density",
        ], "`property_name` has to be 'popularity', 'locality', or 'density'"

        assert len(part_ratios) == 5, "`part_ratios` must contain 5 values"
        assert sum(part_ratios) == 1.0, "`part_ratios` must sum to 1.0"

        self.property_name = property_name
        self.part_ratios = part_ratios
        self.ascending = ascending
        self.random_seed = random_seed

    def __call__(
        self,
        data: Data,
    ) -> Data:
        graph_nx = to_networkx(data, to_undirected=True,
                               remove_self_loops=True)
        compute_fn = _property_name_to_compute_fn[self.property_name]
        property_values = compute_fn(graph_nx, self.ascending)
        split_masks = self.mask_nodes_by_property(property_values,
                                                  self.part_ratios,
                                                  self.random_seed)
        for key, value in split_masks.items():
            data[key] = value

        return data

    @staticmethod
    def _compute_popularity_property(
            graph_nx: Any, ascending: Optional[bool] = True) -> np.ndarray:
        import networkx.algorithms as A

        direction = -1 if ascending else 1
        property_values = direction * np.array(
            list(A.pagerank(graph_nx).values()))
        return property_values

    @staticmethod
    def _compute_locality_property(
            graph_nx: Any, ascending: Optional[bool] = True) -> np.ndarray:
        import networkx.algorithms as A

        num_nodes = graph_nx.number_of_nodes()
        pagerank_values = np.array(list(A.pagerank(graph_nx).values()))

        personalization = dict(zip(range(num_nodes), [0.0] * num_nodes))
        personalization[np.argmax(pagerank_values)] = 1.0

        direction = -1 if ascending else 1
        property_values = direction * np.array(
            list(
                A.pagerank(graph_nx,
                           personalization=personalization).values()))
        return property_values

    @staticmethod
    def _compute_density_property(
            graph_nx: Any, ascending: Optional[bool] = True) -> np.ndarray:
        import networkx.algorithms as A

        direction = -1 if ascending else 1
        property_values = direction * np.array(
            list(A.clustering(graph_nx).values()))
        return property_values

    @staticmethod
    def mask_nodes_by_property(
        property_values: np.ndarray,
        part_ratios: List[float],
        random_seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Provides the split masks for a node split with distributional
        shift based on a given node property, as proposed in `Evaluating
        Robustness and Uncertainty of Graph Models Under Structural
        Distributional Shifts <https://arxiv.org/abs/2302.13875v1>`__

        It considers the in-distribution (ID) and out-of-distribution (OOD)
        subsets of nodes. The ID subset includes training, validation and
        testing parts, while the OOD subset includes validation and testing
        parts. It sorts the nodes in the ascending order of their property
        values, splits them into 5 non-intersecting parts, and creates 5
        associated node mask arrays, 3 of which are for the ID nodes:
        :obj:`"in_train_mask"`, :obj:`"in_val_mask"`,
        :obj:`"in_test_mask"`, and the remaining 2 — for the OOD nodes:
        :obj:`"out_val_mask"`, :obj:`"out_test_mask"`.

        It returns a python dict storing the mask names as keys
        and the corresponding node mask arrays as values.

        Args:
            property_values (np.ndarray): The node property (float) values
                by which the dataset will be split. The length of the array
                must be equal to the number of nodes in graph.
            part_ratios (list): A list of 5 ratios for training, ID validation,
                ID test, OOD validation, OOD testing parts. The values in the
                list must sum to one.
            random_seed (int, optional): Random seed to fix for the initial
                permutation of nodes. It is used to create a random order
                for the nodes that have the same property values or belong
                to the ID subset. (default: :obj:`None`)

        Example:

        .. code-block::

            from torch_geometric.transforms import NodePropertySplit

            num_nodes = 1000
            property_values = np.random.uniform(size=num_nodes)
            part_ratios = [0.3, 0.1, 0.1, 0.3, 0.2]
            split_masks = NodePropertySplit.mask_nodes_by_property(
                property_values, part_ratios
            )
        """
        assert len(part_ratios) == 5, "`part_ratios` must contain 5 values"
        assert sum(part_ratios) == 1.0, "`part_ratios` must sum to 1.0"

        num_nodes = len(property_values)
        part_sizes = np.round(num_nodes * np.array(part_ratios)).astype(int)
        part_sizes[-1] -= np.sum(part_sizes) - num_nodes

        generator = np.random.RandomState(random_seed)
        permutation = generator.permutation(num_nodes)

        node_indices = np.arange(num_nodes)[permutation]
        property_values = property_values[permutation]
        in_distribution_size = np.sum(part_sizes[:3])

        node_indices_ordered = node_indices[np.argsort(property_values)]
        node_indices_ordered[:in_distribution_size] = generator.permutation(
            node_indices_ordered[:in_distribution_size])

        sections = np.cumsum(part_sizes)
        node_split = np.split(node_indices_ordered, sections)[:-1]
        mask_names = [
            "in_train_mask",
            "in_val_mask",
            "in_test_mask",
            "out_val_mask",
            "out_test_mask",
        ]
        split_masks = {}

        for mask_name, node_indices in zip(mask_names, node_split):
            split_mask = np.zeros(num_nodes)
            split_mask[node_indices] = 1
            split_masks[mask_name] = torch.tensor(split_mask, dtype=bool)

        return split_masks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.property_name})"


_property_name_to_compute_fn = {
    "popularity": NodePropertySplit._compute_popularity_property,
    "locality": NodePropertySplit._compute_locality_property,
    "density": NodePropertySplit._compute_density_property,
}
