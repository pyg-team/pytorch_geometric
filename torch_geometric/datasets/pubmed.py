from .planetoid import Planetoid


class PubMed(Planetoid):
    def __init__(self, root, transform=None):
        super(PubMed, self).__init__(root, 'pubmed', transform)
