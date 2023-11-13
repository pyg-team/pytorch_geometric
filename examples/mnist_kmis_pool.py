import os.path as osp

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import PILToTensor

from torch_geometric.data import Data
from torch_geometric.nn import KMISPooling
from torch_geometric.utils import grid, to_networkx

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
mnist = MNIST(path, download=True)

img = PILToTensor()(mnist[0][0])[0] / 255.
edge_index, pos = grid(28, 28)
d1, d2 = pos.T.long()
color = img[(d2, d1)]
pos[:, 1] *= -1  # Flip y axis

x = torch.cat([color.view(-1, 1), pos], dim=-1)

num_ks = 5
height = 3
fig, axes = plt.subplots(4, num_ks, figsize=(num_ks * height, 4 * height))


def lexical_scorer(x, *args, **kwargs):  # Custom scoring function
    return -x[:, 1] * 28 + x[:, 2]  # Lower indices have higher importance


def plot_mnist(ax, k, *args, **kwargs):
    pool = KMISPooling(k=k, *args, **kwargs)
    x_red, idx_red, _, _, _, _ = pool(x=x, edge_index=edge_index)
    col_red, pos_red = x_red[:, 0], x_red[:, 1:]

    G = to_networkx(Data(x=x_red, edge_index=idx_red), to_undirected=True)
    nx.draw_networkx(G, pos=pos_red.numpy(), node_color=col_red.numpy(),
                     node_size=height * 15 * (2 * k + 1), node_shape='s',
                     with_labels=False, vmin=0, vmax=1, ax=ax)


for k in range(num_ks):
    for i in range(4):
        axes[i, k].set_axis_off()

    # 1st Row: Average Pooling
    img_red = F.avg_pool2d(img.unsqueeze(0), k + 1, count_include_pad=False,
                           ceil_mode=True)
    axes[0, k].imshow(img_red[0].numpy(), vmin=0, vmax=1)

    # 2nd Row: KMISPooling + Lexical ordering + Mean Reduction
    plot_mnist(axes[1, k], k=k, scorer=lexical_scorer, score_heuristic=None,
               score_passthrough=None, aggr_x='mean')

    # 3rd Row: KMISPooling + Color Intensity ordering
    #                      + Mean Reduction + Greedy Heuristic
    plot_mnist(
        axes[2, k],
        k=k,
        scorer='first',  # color/intensity
        score_heuristic='greedy',
        score_passthrough=None,
        aggr_x='mean')

    # 4th Row: KMISPooling + Color Intensity ordering
    #                      + Strided/No Reduction + Greedy Heuristic
    plot_mnist(
        axes[3, k],
        k=k,
        scorer='first',  # color/intensity
        score_heuristic='greedy',
        score_passthrough=None,
        aggr_x=None)

plt.tight_layout()
plt.show()
