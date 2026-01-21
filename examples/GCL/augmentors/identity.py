from GCL.augmentors.augmentor import Augmentor, Graph


class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, g: Graph) -> Graph:
        return g
