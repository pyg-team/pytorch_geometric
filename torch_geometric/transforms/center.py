class Center(object):
    def __call__(self, data):
        pos = data.pos

        mean = data.pos.mean(dim=0).view(1, -1)
        data.pos = pos - mean

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
