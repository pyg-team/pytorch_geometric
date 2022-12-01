from typing import Callable, Optional

from torch_geometric.data import InMemoryDataset


class ExplainerDataset(InMemoryDataset):
    r"""Generate custom synthetic dataset from Graph Generators

    Args:
        generator (Generator): The graph generator to be used:
        :obj:`torch.geometric.datasets.generators.ba_graph`,
        :obj:`torch.geometric.datasets.generators.er_graph`.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        generator: Callable,
        transform: Optional[Callable] = None,
    ):
        super().__init__('.', transform)
        data = generator.generate_base_graph()
        self.data, self.slices = self.collate([data])
