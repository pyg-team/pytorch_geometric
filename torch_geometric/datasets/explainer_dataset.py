from typing import Callable, Optional

from torch_geometric.data import InMemoryDataset


class ExplainerDataset(InMemoryDataset):
    def __init__(self, generator: Callable,
                 transform: Optional[Callable] = None):
        super().__init__('.', transform)
        data = generator.generate_base_graph()
        self.data, self.slices = self.collate([data])
