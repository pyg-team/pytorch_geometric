class BaseTransform(object):
    def __call__(self, data):
        if isinstance(data, (tuple, list)):
            return [self._call(d) for d in data]
        else:
            return self._call(data)

    def _call(self, data):
        return data
