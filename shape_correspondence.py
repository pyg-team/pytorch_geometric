import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch_geometric.datasets.faust import FAUST
from torch_geometric.graph.geometry import MeshAdj
from torch_geometric.utils.dataloader import DataLoader
from torch_geometric.nn.modules.gcn import GCN
from torch_geometric.nn.modules.gcn import Lin

path = '~/MPI-FAUST'
train_dataset = FAUST(
    path, train=True, correspondence=True, transform=MeshAdj())
test_dataset = FAUST(
    path, train=False, correspondence=True, transform=MeshAdj())

if torch.cuda.is_available():
    # TODO: Doesn't work yet.
    # kwargs = {'num_workers': 1, 'pin_memory': True}
    kwargs = {}
else:
    kwargs = {}

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCN(1, 32)
        self.conv2 = GCN(32, 64)
        self.conv3 = GCN(64, 128)
        self.lin1 = Lin(128, 256)
        self.lin2 = Lin(256, 6890)

    def forward(self, adj, x):
        x = F.relu(self.conv1(adj, x))
        x = F.relu(self.conv2(adj, x))
        x = F.relu(self.conv3(adj, x))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x


model = Net()
if torch.cuda.is_available():
    model.cuda()


def train(epoch):
    model.train()
    print('Epoch:', epoch)
    pass
    # model.train()

    for batch_idx, ((_, (adj, _)), target) in enumerate(train_loader):
        # Only needed for Kipf.
        n = adj.size(0)
        adj = torch.sparse.FloatTensor(adj._indices(),
                                       torch.ones(adj._indices().size(1)),
                                       torch.Size([n, n]))

        x = torch.ones(n).view(-1, 1)
        if torch.cuda.is_available():
            x, adj, target = x.cuda(), adj.cuda(), target.cuda()

        x, adj, target = Variable(x), Variable(adj), Variable(target)

        print(x.size())
        print(adj.size())

        output = model(adj, x)
        print(output.size())

        # print(adjs.size())

    #     pass


for epoch in range(1, 3):
    train(epoch)

    # print(vertices.size())
    # print(adjs.size())
    # print(slices.size())
    # # print(adjs.size())
    # print(target.size())
    # Generate correspondence targets.
    # identity(torch.Size([6890, 6890]))

    # edges = edges.squeeze()
    # adjs = [mesh_adj(vertices[i], edges[i]) for i in range(0, edges.size(0))]
    # print(adjs[0].size())
    # print(vertices.size())
    # print(edges.size())
    # print(target.size())

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
