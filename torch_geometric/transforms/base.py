class BaseAdj(object):
    def __call__(self, data):
        adjs = data['adjs']
        positions = data['positions']

        if adjs is not None:
            for i in range(len(adjs)):
                adjs[i] = self._call(adjs[i], positions[i])
        else:
            data.adj = self._call(data.adj, data.position)

        return data

    def _call(self, adj, position):
        return adj
