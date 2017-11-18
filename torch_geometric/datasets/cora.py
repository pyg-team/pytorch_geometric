from .planetoid import Planetoid


class Cora(Planetoid):
    def __init__(self, root, transform=None):
        super(Cora, self).__init__(root, 'cora', transform)
