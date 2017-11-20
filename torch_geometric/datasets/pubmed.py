from .planetoid import Planetoid


class PubMed(Planetoid):
    """`PubMed <https://linqs.soe.ucsc.edu/node/236>`_ Dataset.

    Args:
        root (string): Root directory of dataset. Download and process data
            automatically if dataset doesn't exist.
        transform (callable, optional): A function/transform that takes in a
            ``Data`` object and returns a transformed version.
            (default: ``None``)
    """

    def __init__(self, root, transform=None):
        super(PubMed, self).__init__(root, 'pubmed', transform)
