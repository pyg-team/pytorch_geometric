# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import torch

import torch_geometric.transforms as T
from torch_geometric.contrib.nn import SE3Transformer
from torch_geometric.contrib.utils import Fiber
from torch_geometric.datasets import QM9
from torch_geometric.datasets.graph_generator import BAGraph


def get_random_graph_generator(N, num_edges_factor=18):
    graph = BAGraph(N, num_edges_factor)
    return graph


def assign_relative_pos(graph, coords):
    graph.pos = coords
    graph = T.Cartesian()(graph)
    return graph


def get_max_diff(a, b):
    return (a - b).abs().max().item()


def rot_z(gamma):
    return torch.tensor(
        [[torch.cos(gamma), -torch.sin(gamma), 0],
         [torch.sin(gamma), torch.cos(gamma), 0], [0, 0, 1]],
        dtype=gamma.dtype)


def rot_y(beta):
    return torch.tensor(
        [[torch.cos(beta), 0, torch.sin(beta)], [0, 1, 0],
         [-torch.sin(beta), 0, torch.cos(beta)]], dtype=beta.dtype)


def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


# Tolerances for equivariance error abs( f(x) @ R  -  f(x @ R) )
TOL = 1e-3
CHANNELS, NODES = 32, 512


def _get_outputs(model, R):
    feats0 = torch.randn(NODES, CHANNELS, 1)
    feats1 = torch.randn(NODES, CHANNELS, 3)

    coords = torch.randn(NODES, 3)
    graph_generator = get_random_graph_generator(NODES)
    graph = graph_generator()
    if torch.cuda.is_available():
        feats0 = feats0.cuda()
        feats1 = feats1.cuda()
        R = R.cuda()
        coords = coords.cuda()
        graph = graph.to('cuda')
        model.cuda()

    graph1 = assign_relative_pos(graph.clone(), coords)
    graph1.node_feats = {'0': feats0, '1': feats1}
    # graph1.node_feats = {'0': feats0}
    graph1.edge_feats = {'0': graph1.edge_attr[:, 0, None, None]}
    # graph1.edge_feats = {}
    graph2 = assign_relative_pos(graph.clone(), coords @ R)
    graph2.node_feats = {'0': feats0, '1': feats1 @ R}
    # graph2.node_feats = {'0': feats0}
    graph2.edge_feats = {'0': graph2.edge_attr[:, 0, None, None]}
    # graph2.edge_feats = {}

    out1 = model(graph1)
    out2 = model(graph2)

    return out1, out2


def _get_model(**kwargs):
    # return SE3Transformer(
    #     num_layers=0,
    #     fiber_in=Fiber.create(2, CHANNELS),
    #     fiber_hidden=Fiber.create(3, CHANNELS),
    #     fiber_out=Fiber.create(2, CHANNELS),
    #     fiber_edge=Fiber({}),
    #     num_heads=8,
    #     channels_div=2,
    #     **kwargs
    # )
    return SE3Transformer(
        num_layers=0,  # TODO: Test with more layers
        fiber_in=Fiber.create(2, CHANNELS),
        fiber_hidden=Fiber.create(3, CHANNELS),
        fiber_out=Fiber.create(2, CHANNELS),
        fiber_edge=Fiber({0: 1}),
        num_heads=8,
        channels_div=2,
        **kwargs)


def test_equivariance():
    model = _get_model()
    R = rot(*torch.rand(3))
    if torch.cuda.is_available():
        R = R.cuda()
    out1, out2 = _get_outputs(model, R)

    assert torch.allclose(out2['0'], out1['0'], atol=TOL), \
        f'type-0 features should be invariant {get_max_diff(out1["0"], out2["0"])}'
    assert torch.allclose(out2['1'], (out1['1'] @ R), atol=TOL), \
        f'type-1 features should be equivariant {get_max_diff(out1["1"] @ R, out2["1"])}'


def test_equivariance_pooled():
    model = _get_model(pooling='avg', return_type=1)
    R = rot(*torch.rand(3))
    if torch.cuda.is_available():
        R = R.cuda()
    out1, out2 = _get_outputs(model, R)

    assert torch.allclose(out2, (out1 @ R), atol=TOL), \
        f'type-1 features should be equivariant {get_max_diff(out1 @ R, out2)}'


def test_invariance_pooled():
    model = _get_model(pooling='avg', return_type=0)
    R = rot(*torch.rand(3))
    if torch.cuda.is_available():
        R = R.cuda()
    out1, out2 = _get_outputs(model, R)

    assert torch.allclose(out2, out1, atol=TOL), \
        f'type-0 features should be invariant {get_max_diff(out1, out2)}'


if __name__ == "__main__":
    test_equivariance()
    test_equivariance_pooled()
    test_invariance_pooled()
