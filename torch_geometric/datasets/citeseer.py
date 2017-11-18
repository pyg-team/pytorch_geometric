from .planetoid import Planetoid


class CiteSeer(Planetoid):
    def __init__(self, root, transform=None):
        super(CiteSeer, self).__init__(root, 'citeseer', transform)
