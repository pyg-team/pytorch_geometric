class Graclus(object):
    def __init__(self, level):
        self.level = level

    def __call__(self, data):
        input, adj, position = data
        # Compute adjacencies
        # Permute inputs and fill with zeros.
        # permute positions and take mean.
        return (input, (adj, adj), (position, position))
