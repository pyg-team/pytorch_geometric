class NormalizeFeatures(object):
    def __call__(self, data):
        data.x = data.x / data.x.sum(1, keepdim=True).clamp(min=1)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
