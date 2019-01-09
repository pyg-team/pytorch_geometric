from torch_geometric.transforms import Center


class NormalizeScale(object):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`.
    """

    def __init__(self):
        self.center = Center()

    def __call__(self, data):
        data = self.center(data)

        scale = (1 / data.pos.abs().max()) * 0.999999
        data.pos = data.pos * scale

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
