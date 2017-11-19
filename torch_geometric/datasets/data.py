class Data(object):
    def __init__(self, input, adj, position, target):
        self.props = {}

        self.input = input
        self.adj = adj
        self.position = position
        self.target = target

    def add(self, key, value):
        if value is not None:
            self.props[key] = value
        else:
            if key in self.props:
                del self.props[key]

    def __getitem__(self, key):
        return self.props[key] if key in self.props else None

    def all(self):
        return self.props

    @property
    def input(self):
        return self['input']

    @input.setter
    def input(self, input):
        self.add('input', input)

    @property
    def adj(self):
        return self['adj']

    @adj.setter
    def adj(self, adj):
        self.add('adj', adj)

    @property
    def position(self):
        return self['position']

    @position.setter
    def position(self, position):
        self.add('position', position)

    @property
    def target(self):
        return self['target']

    @target.setter
    def target(self, target):
        self.add('target', target)
