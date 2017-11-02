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

image_dataset = MNIST('/Users/rusty1s/MNIST', train=True, download=True)
graph_dataset = MNISTSuperpixel75('~/MNISTSuperpixel75', train=True)

scale = 32
offset_x = 1
offset_y = 1
example = 2

image, _ = image_dataset[example]
image = np.array(image)
image = np.repeat(image, scale, axis=0)
image = np.repeat(image, scale, axis=1)
image = skimage.img_as_float(image)
image = skimage.color.gray2rgb(image)
shape = image.shape

(input, adj, position), _ = graph_dataset[example]
position *= scale
position += torch.FloatTensor([scale * offset_x, scale * offset_y])

index = adj._indices().t()
for i in range(index.size(0)):
    start, end = index[i]
    start_x, start_y = position[start]
    end_x, end_y = position[end]
    start_x, start_y = int(start_x), int(start_y)
    end_x, end_y = int(end_x), int(end_y)

    rr, cc = draw.line(start_y, start_x, end_y, end_x)
    image[rr, cc] = [1, 0, 0]

for i in range(75):
    x, y = position[i]
    rr, cc = draw.circle(y, x, 18, shape=shape)
    image[rr, cc] = [1, 0, 0]
    rr, cc = draw.circle(y, x, 12, shape=shape)
    image[rr, cc] = [input[i], input[i], input[i]]

io.imshow(image)
io.show()
