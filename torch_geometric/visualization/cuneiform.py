from PIL import Image, ImageDraw
import numpy as np
from skimage.color import gray2rgb
import torch
from torchvision.transforms import Compose

import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from torch_geometric.datasets import Cuneiform  # noqa
from torch_geometric.sparse import stack  # noqa
from torch_geometric.transforms import (RandomRotate, RandomScale,
                                        RandomTranslate)  # noqa

dataset = Cuneiform('/tmp/cuneiform', train=True)
transform = Compose([
    RandomRotate(0.2),
    RandomScale(1.3),
    RandomTranslate(0.2),
])
dataset = Cuneiform('/tmp/cuneiform', train=True, transform=transform)

num_signs = 30
positions, adjs = [], []
for i in range(0, num_signs):
    data = dataset[i]
    adjs.append(data['adj'])
    positions.append(data['position'][:, :2])
adj, _ = stack(adjs)
position = torch.cat(positions, dim=0)
position -= position.min(dim=0)[0]
position += 2
h, w = position.max(dim=0)[0]
s = int(max(h, w) + 2)

scale = 10
rescale = 2
position *= scale * rescale

image = np.ones((s, s), np.uint8)
image = np.repeat(image, scale * rescale, axis=0)
image = np.repeat(image, scale * rescale, axis=1)
image = gray2rgb(image, alpha=True)
image *= np.array([255, 225, 255, 1], np.uint8)

image = Image.fromarray(image)
draw = ImageDraw.Draw(image)

index = adj._indices().t()
for i in range(index.size(0)):
    start, end = index[i]
    start_x, start_y = position[start]
    end_x, end_y = position[end]
    start_x, start_y = int(start_x), int(start_y)
    end_x, end_y = int(end_x), int(end_y)

    draw.line(
        (start_y, start_x, end_y, end_x),
        fill=(0, 0, 0, 255),
        width=rescale * 1)

for i in range(position.size(0)):
    x, y = position[i]
    r = 2 * rescale
    draw.ellipse((y - r, x - r, y + r, x + r), fill=(0, 0, 0, 255))

image = image.resize((s * scale, s * scale), Image.ANTIALIAS)
image.show()
