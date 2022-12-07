import torch


class Motif:
    r"""Generate a motif based on a given structure.

    Args:
        structure (str): The structure of the house to generate.
            Currently supports 'house' shape.
    """
    def __init__(self, structure: str):
        self.edge_index = None
        self.label = None
        self.num_nodes = None
        self.__build_motif(structure)

    def __build_motif(self, structure):
        if structure == 'house':
            self.__build_house()

    def __build_house(self):
        self.num_nodes = 5
        self.edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                                        [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1]])
        self.label = torch.tensor([1, 1, 2, 2, 3])
