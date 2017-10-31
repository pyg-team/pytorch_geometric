from .planetoid import Planetoid


class CiteSeer(Planetoid):
    def __init__(self, root, transform=None, target_transform=None):
        super(CiteSeer, self).__init__(root, 'citeseer', transform,
                                       target_transform)
