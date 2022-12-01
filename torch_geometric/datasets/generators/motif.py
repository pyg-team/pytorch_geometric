import torch


class Motif:
    def __init__(self, structure: str):
        self.edge_index = None
        self.label = None
        self.num_nodes = None
        self.build_motif(structure)

    def build_motif(self, structure):
        if structure == 'house':
            self.house()

    def house(self):
        self.num_nodes = 5
        self.edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                                        [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1]])
        self.label = torch.tensor([1, 1, 2, 2, 3])
