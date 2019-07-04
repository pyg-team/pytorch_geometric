from ..datasets import TUDataset


class SyntheticDataset(TUDataset):
    r""" Synthetic COLORS and TRIANGLES datasets from the
    `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ paper

    The datasets have the same format as :class:`TUDataset`,
    but have additional node attention data.

    This class has the same arguments as :class:`TUDataset`.
    """

    url = 'https://github.com/bknyaz/graph_attention_pool/raw/master/data'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False):
        self.name = name
        super(SyntheticDataset, self).__init__(root, name, transform, pre_transform,
                                               pre_filter, use_node_attr=use_node_attr)
