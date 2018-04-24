class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of :obj:`transform` objects): List of transforms to
            compose.

    .. testsetup::

        import torch
        from torch_geometric.data import Data

    .. testcode::

        import torch_geometric.transforms as T
        transform = T.Compose([T.Cartesian(), T.Cartesian()])
        pos = torch.Tensor([[0, 0], [1, 1]])
        edge_index = torch.LongTensor([[0, 1], [1, 0]])
        data = Data(edge_index=edge_index, pos=pos)
        data = transform(data)
        print(data.edge_attr)

    .. testoutput::

        -0.5  0.5 -0.5  0.5
        -0.5  0.5 -0.5  0.5
        [torch.FloatTensor of size 2x4]
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))
