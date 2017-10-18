import torch

from torch_geometric.datasets.faust import FAUST
# from torch_geometric.graph.geometry import mesh_adj
from torch_geometric.utils.dataloader import DataLoader

path = '~/Downloads/MPI-FAUST'
train_dataset = FAUST(path, train=True)
test_dataset = FAUST(path, train=False)

train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)

for batch_idx, ((adj, slices), target) in enumerate(train_loader):
    # edges = edges.squeeze()
    print(adj.size())
    print(slices)
    # adjs = [mesh_adj(vertices[i], edges[i]) for i in range(0, edges.size(0))]
    # print(adjs[0].size())
    # print(vertices.size())
    # print(edges.size())
    print(target.size())

# # TODO:  COmpute 544-dimensional SHOT descriptors (local histogram of normal
# # vectors)
# # Architecture:
# # * Lin16
# # * ReLU
# # * CNN32
# # * AMP
# # * ReLU
# # * CNN64
# # * AMP
# # * ReLU
# # * CNN128
# # * AMP
# # * ReLU
# # * Lin256
# # * Lin6890
# #
# # output representing the soft correspondence as an 6890-dimensional vector.
# #
# # shape correspondence (labelling problem) where one tries to label each vertex
# # to the index of a corresponding point.
# # X = Indices
# # y_j denote the vertex corresponding to x_i
# # output representing the probability distribution on Y
# # loss: multinominal regression loss
# # ACN: adam optimizer 10^-3, beta_1 = 0.9, beta_2 = 0.999

# # SHOT descriptor: Salti, Tombari, Stefano. SHOT, 2014
# # Signature of Histograms of OrienTations (SHOT)
