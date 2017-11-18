class Data(object):
    def __init__(self, input, adj, position, target):

        self.input = input
        self.adj = adj
        self.position = position
        self.target = target
        self.params = {}

    def add(self, key, value):
        self.params[key] = value

    def __getitem__(self, key):
        return self.params[key] if key in self.params else None
