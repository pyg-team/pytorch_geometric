from .planetoid import Planetoid


class CiteSeer(Planetoid):
    """`CiteSeer <https://linqs.soe.ucsc.edu/node/236>`_ Dataset.

    Args:
        root (string): Root directory of dataset. Downloads and processes data
            automatically if dataset doesn't exist.
        transform (callable, optional): A function/transform that takes in a
            :class:`Data` object and returns a transformed version.
            (default: :obj:`None`)
    """

    def __init__(self, root, transform=None):
        super(CiteSeer, self).__init__(root, 'citeseer', transform)
