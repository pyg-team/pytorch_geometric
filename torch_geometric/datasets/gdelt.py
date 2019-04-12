from .icews import TemporalDataset


class GDELT(TemporalDataset):
    r"""The Global Database of Events, Language, and Tone (GDELT) dataset used
    in the, *e.g.*, `"Recurrent Event Network for Reasoning over Temporal
    Knowledge Graphs" <https://arxiv.org/abs/1904.05530>`_ paper, consisting of
    events collected from 4/1/2015 to 3/31/2016 (15 minutes time granularity).

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(GDELT, self).__init__(root, 'GDELT', transform, pre_transform)
