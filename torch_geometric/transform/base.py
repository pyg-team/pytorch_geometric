class BaseAdj(object):
    def __call__(self, data):
        input, adj, position = data

        if not isinstance(adj, (tuple, list)):
            adj_out, position_out = self._call(adj, position)
        else:
            adj_out = []
            position_out = []
            for i in range(len(adj)):
                a, p = self._call(adj[i], position[i])
                adj_out.append(a)
                position_out.append(p)

        return input, adj_out, position_out

    def _call(self, adj, position):
        return adj, position
