class Normalize(object):
    def __init__(self, dim=-1):
        self.dim = dim

    def __call__(self, data):
        data.x = data.x / data.x.sum(self.dim, keepdim=True)
        return data

    def __repr__(self):
        return '{}(dim={})'.format(self.__class__.__name__, self.dim)
