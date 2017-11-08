import os

import torch
from scipy.interpolate import interp2d
from matplotlib.pyplot import figure, savefig
import numpy as np
import matplotlib.pyplot as plt

home = os.path.expanduser('~')
kernel = torch.load(os.path.join(home, 'Desktop', 'conv_1_weight.pt'))
kernel = kernel[:-1]  # Delete root node
kernel = kernel.squeeze().t().contiguous()  # 32 x 25

size = 200
h, w = size, size
partition_size = size // 4

image = np.zeros((size, size))

x = np.arange(0, size)
grid_x, grid_y = np.meshgrid(x, x)

x = np.arange(0, size + partition_size, partition_size)
y = np.arange(0, size + partition_size, partition_size)

fig = figure()
ax = fig.add_subplot(111)

for i in range(4):
    for j in range(8):
        image = np.zeros((size, size))
        print(i * 8 + j)
        k = kernel[i * 8 + j].numpy()
        f = interp2d(x, y, k)

        for yy in range(h):
            for xx in range(w):
                v = f(yy, xx)
                image[yy, xx] = v

        ax.pcolormesh(
            grid_x + (size * 1.1) * j,
            grid_y + (size * 1.1) * i,
            image,
            shading='gouraud')

plt.axes().set_aspect('equal')
plt.axis('off')
savefig(
    os.path.expanduser('~/Desktop/kernel.png'),
    bbox_inches='tight',
    transparent=True,
    pad_inches=0,
    dpi=300)
