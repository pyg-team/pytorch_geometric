from torch_geometric.datasets.faust import FAUST

dataset = FAUST('~/Downloads/MPI-FAUST')

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
