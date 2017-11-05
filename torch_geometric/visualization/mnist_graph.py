import numpy as np
import skimage
import skimage.io as io
from skimage.draw import draw
from torchvision.datasets import MNIST
import torch

import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from torch_geometric.datasets import MNISTSuperpixel75  # noqa
from torch_geometric.graph.grid import grid, grid_position  # noqa


def image_graph(image, input, graph, position, scale, offset):
    image = image.copy()
    image = np.repeat(image, scale, axis=0)
    image = np.repeat(image, scale, axis=1)
    image = skimage.img_as_float(image)
    image = skimage.color.gray2rgb(image)
    shape = image.shape

    position *= scale
    position += scale * offset

    # Draw edges
    index = graph._indices().t()
    for i in range(index.size(0)):
        start, end = index[i]
        start_x, start_y = position[start]
        end_x, end_y = position[end]
        start_x, start_y = int(start_x), int(start_y)
        end_x, end_y = int(end_x), int(end_y)

        rr, cc = draw.line(start_y, start_x, end_y, end_x)
        image[rr, cc] = [1, 0, 0]

    # Draw nodes
    for i in range(position.size(0)):
        x, y = position[i]
        rr, cc = draw.circle(y, x, 10, shape=shape)
        image[rr, cc] = [1, 0, 0]
        rr, cc = draw.circle(y, x, 9, shape=shape)
        image[rr, cc] = [input[i], input[i], input[i]]

    return image


image_dataset = MNIST('/Users/rusty1s/MNIST', train=True, download=True)
graph_dataset = MNISTSuperpixel75('~/MNISTSuperpixel75', train=True)

scale = 32
offset = torch.FloatTensor([1, 1])
example = 2

image, _ = image_dataset[example]
image = np.array(image)
(input, adj, position), _ = graph_dataset[example]

grid_adj = grid(torch.Size([28, 28]), connectivity=8)
grid_position = grid_position(torch.Size([28, 28]))
grid_input = image.flatten() / 255.0
grid_offset = torch.FloatTensor([0.5, 0.5])

# image = image_graph(image, input, adj, position, scale, offset)
image = image_graph(image, grid_input, grid_adj, grid_position, scale,
                    grid_offset)

io.imshow(image)
io.show()
