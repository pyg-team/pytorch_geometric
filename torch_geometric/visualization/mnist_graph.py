from PIL import Image, ImageDraw
import numpy as np
from torchvision.datasets import MNIST
from skimage.color import gray2rgb
# import skimage.io as io
# from skimage.draw import draw
import torch

import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from torch_geometric.datasets import MNISTSuperpixels  # noqa
from torch_geometric.graph.grid import grid_5x5, grid_position  # noqa
from torch_geometric.transforms.graclus import graclus, perm_input  # noqa

# def image_graph(image, input, graph, position, scale, offset):
#     image = image.copy()
#     image = np.repeat(image, scale, axis=0)
#     image = np.repeat(image, scale, axis=1)
#     image = skimage.img_as_float(image)
#     image = skimage.color.gray2rgb(image)
#     shape = image.shape

#     position *= scale
#     position += scale * offset

#     # Draw edges
#     index = graph._indices().t()
#     for i in range(index.size(0)):
#         start, end = index[i]
#         start_x, start_y = position[start]
#         end_x, end_y = position[end]
#         start_x, start_y = int(start_x), int(start_y)
#         end_x, end_y = int(end_x), int(end_y)

#         rr, cc = draw.line(start_y, start_x, end_y, end_x)
#         image[rr, cc] = [1, 0, 0]

#     # Draw nodes
#     for i in range(position.size(0)):
#         x, y = position[i]
#         rr, cc = draw.circle(y, x, 10, shape=shape)
#         image[rr, cc] = [1, 0, 0]
#         rr, cc = draw.circle(y, x, 9, shape=shape)
#         image[rr, cc] = [input[i], input[i], input[i]]

#     return image

image_dataset = MNIST('/tmp/MNIST', train=True, download=True)
graph_dataset = MNISTSuperpixels('~/MNISTSuperpixels', train=True)

scale = 32
rescale = 4
offset = torch.FloatTensor([1, 1])
example = 35

image, _ = image_dataset[example]
image = np.array(image)
data = graph_dataset[example]
input, adj, position = data['input'], data['adj'], data['position']
print(data['target'])

# adj = grid(torch.Size([28, 28]), connectivity=8)
# position = grid_position(torch.Size([28, 28]))
# input = image.flatten() / 255.0

# image = image_graph(image, input, adj, position, scale, offset)
# image = image_graph(image, grid_input, grid_adj, grid_position, scale,
#                     grid_offset)

# io.imshow(image)
# io.show()

# image = skimage.img_as_float(image)
image = np.ones((28, 28), np.uint8)
image = np.repeat(image, scale * rescale, axis=0)
image = np.repeat(image, scale * rescale, axis=1)
image = gray2rgb(image, alpha=True)
# image *= np.array([228, 246, 232], np.uint8)
image *= np.array([255, 255, 255, 1], np.uint8)
# image *= np.array([0, 0, 0, 0], np.uint8)
image = Image.fromarray(image)
draw = ImageDraw.Draw(image)

offset = torch.FloatTensor([(28 - position[:, 0].max()) / 2,
                            (28 - position[:, 1].max()) / 2])

# row, col = adj._indices()
# direction = position[row] - position[col]
# direction = (direction * direction).sum(dim=1)
# mask = direction < 30
# row = row[mask]
# col = col[mask]
# index = torch.stack([row, col], dim=0)
# weight = torch.FloatTensor(row.size(0)).fill_(1)
# n = adj.size(0)
# adj = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))

position *= scale * rescale
position += scale * offset * rescale

adjs, positions, perm = graclus(adj, position, level=2)
adj, position = adjs[2], positions[2]
input = perm_input(input, perm)
input = input.view(-1, 4).sum(dim=1)

index = adj._indices().t()
for i in range(index.size(0)):
    start, end = index[i]
    start_x, start_y = position[start]
    end_x, end_y = position[end]
    start_x, start_y = int(start_x), int(start_y)
    end_x, end_y = int(end_x), int(end_y)

    draw.line(
        (start_x, start_y, end_x, end_y),
        fill=(0, 0, 0, 255),
        width=rescale * 2)

for i in range(position.size(0)):
    x, y = position[i]
    r = 16 * rescale
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0, 255))
    r = 14 * rescale
    draw.ellipse(
        (x - r, y - r, x + r, y + r), fill=(49, 130, 219, int(255 * input[i])))
    # if v > 0:
    #     print(v)
    # c1 = [49 * v, 130 * v, 219 * v]
    # c2 = [228 * (1 - v), 246 * (1 - v), 232 * (1 - v)]
    # c = (int(c1[0] + c2[0]), 130, 219)

    # draw.ellipse((x - r, y - r, x + r, y + r), fill=c)
    # if input[i] > 0.3:
    #     draw.ellipse((x - r, y - r, x + r, y + r), fill=(49, 130, 219))
    # else:
    #     draw.ellipse((x - r, y - r, x + r, y + r), fill=(228, 246, 232))

image = image.resize((28 * scale, 28 * scale), Image.ANTIALIAS)

image.show()
