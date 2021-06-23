from typing import List, Callable


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Callable]): List of transforms to compose.
    """
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self):
        args = ['    {},'.format(transform) for transform in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))
