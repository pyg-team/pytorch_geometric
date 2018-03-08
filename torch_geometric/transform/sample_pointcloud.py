from __future__ import division
import torch
import numpy as np

from ..datasets.utils.nn_graph import nn_graph, radius_graph


class SamplePointcloud(object):
    def __init__(self, num_points=1024):
        self.num_points=num_points

    def __call__(self, data):
        triangles = data.faces
        probs = np.sqrt((np.cross(triangles[:, 1, :] - triangles[:, 0, :],
                                  triangles[:, 2, :] - triangles[:, 0,
                                                       :]) ** 2).sum(
            axis=1)) / 2.
        probs /= probs.sum()

        sample = torch.LongTensor(np.random.choice(np.arange(len(triangles)),
                                                   size=self.num_points,
                                                   p=probs))

        samples_0 = np.random.random((self.num_points, 1))
        samples_1 = np.random.random((self.num_points, 1))
        cond = (samples_0[:, 0] + samples_1[:, 0]) > 1.
        samples_0[cond, 0] = 1. - samples_0[cond, 0]
        samples_1[cond, 0] = 1. - samples_1[cond, 0]

        samples_0 = torch.FloatTensor(samples_0)
        samples_1 = torch.FloatTensor(samples_1)

        tri_points0 = triangles[sample][:, 0, :]
        tri_points1 = triangles[sample][:, 1, :]
        tri_points2 = triangles[sample][:, 2, :]

        data.pos = tri_points0 + samples_0 * (tri_points1 - tri_points0) + \
              samples_1 * (tri_points2 - tri_points0)

        data.input = torch.ones(data.pos.size(0))
        return data



class PCToGraphRad(object):
    def __init__(self, r=0.1):
        self.r = r

    def __call__(self, data):
        data.index = radius_graph(data.pos, self.r)


class PCToGraphNN(object):
    def __init__(self, nn=6):
        self.nn = nn

    def __call__(self, data):
        data.index = nn_graph(data.pos, k=self.nn)
