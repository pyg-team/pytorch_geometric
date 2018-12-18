class Center(object):
    r"""Centers node positions around the origin.

    .. testsetup::

        import torch
        from torch_geometric.data import Data

    .. testcode::

        from torch_geometric.transforms import Center

        pos = torch.Tensor([[0, 0], [2, 0], [4, 0]])
        data = Data(pos=pos)

        data = Center()(data)
    """

    def __call__(self, data):
        data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
