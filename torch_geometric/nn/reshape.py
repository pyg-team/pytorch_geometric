import torch


class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        """"""
        x = x.view(*self.shape)
        return x

    def __repr__(self):
        shape = ', '.join([str(dim) for dim in self.shape])
        return f'{self.__class__.__name__}({shape})'
