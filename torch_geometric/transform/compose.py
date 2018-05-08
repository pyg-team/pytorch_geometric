class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of :obj:`transform` objects): List of transforms to
            compose.

    .. testsetup::

        import torch
        from torch_geometric.datasets.dataset import Data

    .. testcode::

        import torch_geometric.transform as T

        pos = torch.tensor([[-1, 0], [0, 0], [2, 0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        data = Data(None, pos, edge_index, None, None)

        transform = T.Compose([T.Cartesian(), T.TargetIndegree()])
        data = transform(data)

        print(data.weight)

    .. testoutput::

        tensor([[ 0.7500,  0.5000,  1.0000],
                [ 1.0000,  0.5000,  1.0000]])
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
