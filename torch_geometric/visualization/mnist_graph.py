import numpy as np
import skimage
import skimage.io as io
from skimage.draw import draw
from torchvision.datasets import MNIST

import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from torch_geometric.datasets import MNISTSuperpixel75  # noqa

image_dataset = MNIST('/Users/rusty1s/MNIST', train=True, download=True)
graph_dataset = MNISTSuperpixel75('~/MNISTSuperpixel75', train=True)

scale = 16
offset_x = 1
offset_y = 1
example = 2

image, _ = image_dataset[example]
image = np.array(image)
image = np.repeat(image, scale, axis=0)
image = np.repeat(image, scale, axis=1)
(input, adj, position), _ = graph_dataset[example]

image = skimage.img_as_float(image)
image = skimage.color.gray2rgb(image)
x_max = position[:, 1].max()
y_max = position[:, 0].max()
shape = image.shape

for i in range(75):
    y, x = position[i]
    y = y_max - y
    x = x_max - x
    y, x = scale * y, scale * x
    y, x = scale * offset_y + y, scale * offset_x + x
    rr, cc = draw.circle(y, x, 8, shape=shape)
    image[rr, cc] = [0.5, 0, input[i]]

index = adj._indices().t()
for i in range(index.size(0)):
    start, end = index[i]
    start_y, start_x = position[start]
    end_y, end_x = position[end]

    start_y = y_max - start_y
    end_y = y_max - end_y
    start_x = x_max - start_x
    end_x = x_max - end_x

    start_y, start_x = scale * start_y, scale * start_x
    end_y, end_x = scale * end_y, scale * end_x

    start_y, start_x = scale * offset_y + start_y, scale * offset_x + start_x
    end_y, end_x = scale * offset_y + end_y, scale * offset_x + end_x

    start_y, start_x = int(start_y), int(start_x)
    end_y, end_x = int(end_y), int(end_x)

    rr, cc = draw.line(start_y, start_x, end_y, end_x)
    image[rr, cc] = [0.2, 0, 0]

io.imshow(image)
io.show()
