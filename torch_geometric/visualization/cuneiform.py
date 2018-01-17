from PIL import Image, ImageDraw
import numpy as np
from skimage.color import gray2rgb
# import torch
# from torchvision.transforms import Compose

import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from torch_geometric.datasets import Cuneiform  # noqa
from torch_geometric.sparse import stack  # noqa
# from torch_geometric.transforms import (RandomRotate, RandomScale,
#                                         RandomTranslate)  # noqa

dataset1 = Cuneiform('/tmp/cuneiform_1', mode=1)
print(dataset1.index.size())
dataset2 = Cuneiform('/tmp/cuneiform_2', mode=2)
print(dataset2.index.size())
dataset = dataset1
# transform = Compose([
#     RandomRotate(0.2),
#     RandomScale(1.3),
#     RandomTranslate(0.2),
# ])
# dataset = Cuneiform('/tmp/cuneiform', train=True, transform=transform)

data = dataset1[60]
data = dataset2[60]
index, position = data['adj']._indices().t(), data['position']

print(dataset1.position[1300])
print(dataset2.position[1300])

raise NotImplementedError



position -= position.min(dim=0)[0] - 2

s = int(position.max()) + 2

scale = 32
rescale = 4
position *= scale * rescale

image = np.ones((s, s), np.uint8)
image = np.repeat(image, scale * rescale, axis=0)
image = np.repeat(image, scale * rescale, axis=1)
image = gray2rgb(image, alpha=True)
image *= np.array([255, 225, 255, 1], np.uint8)

image = Image.fromarray(image)
draw = ImageDraw.Draw(image)

for i in range(index.size(0)):
    start, end = index[i]
    start_x, start_y = position[start]
    end_x, end_y = position[end]
    start_x, start_y = int(start_x), int(start_y)
    end_x, end_y = int(end_x), int(end_y)

    draw.line(
        (start_y, start_x, end_y, end_x),
        fill=(0, 0, 0, 255),
        width=rescale * 2)

for i in range(position.size(0)):
    x, y = position[i]
    r = 4 * rescale
    draw.ellipse((y - r, x - r, y + r, x + r), fill=(0, 0, 0, 255))

image = image.resize((s * scale, s * scale), Image.ANTIALIAS)
image.show()
