from typing import Callable, Optional

from torch_geometric.data import InMemoryDataset


class ExplainerDataset(InMemoryDataset):
    r"""Generate a custom synthetic dataset from Graph Generators.

    Args:
        generator (GraphGenerator): The graph generator instance object
            to be used: :obj:`torch.geometric.datasets.generators.BAGraph`,
            :obj:`torch.geometric.datasets.generators.ERGraph`.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        generator,
        transform: Optional[Callable] = None,
    ):
        super().__init__('.', transform)
        generator.generate_graph()
        self.data, self.slices = self.collate([generator.data])
