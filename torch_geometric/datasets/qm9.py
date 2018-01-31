from torch.utils.data import Dataset


class QM9(Dataset):
    """`QM9 <http://quantum-machine.org/datasets>`_ Molecule Dataset from the
    `"Quantum Chemistry Structures and Properties of 134 Kilo Molecules"
    <https://www.nature.com/articles/sdata201422>`_ paper. QM9 consists of
    130k molecules with 13 properties for each molecule.

    Args:
        root (string): Root directory of dataset. Downloads data automatically
            if dataset doesn't exist.
        train (bool, optional): If :obj:`True`, returns training molecules,
        otherwise test molecules. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in a
            ``Data`` object and returns a transformed version.
            (default: ``None``)
    """

    url = 'http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/' \
          'mnist_superpixels.tar.gz'

    def __init__(self, root, train=True, transform=None):
        super(QM9, self).__init__()
