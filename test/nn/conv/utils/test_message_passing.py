import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm_sum
from torch_geometric.nn.conv.utils.message_passing import MessagePassing
from torch_geometric.datasets import Planetoid


class MyConv(MessagePassing):
    def __init__(self):
        super(MyConv, self).__init__(aggr='add')

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:

        return self.propagate(edge_index, x=x, y=x)

    def sparse_message_and_aggregate(self, adj_t, x_j):
        return adj_t.matmul(x_j, reduce=self.aggr)

    def dense_message_and_aggregate(self, adj_t, x_j):
        return adj_t.matmul(x_j)

    def message(self, x_j):
        return x_j


def test_message_passing():
    row = torch.tensor([0, 1, 1, 2, 2])
    col = torch.tensor([1, 0, 2, 0, 1])
    x = torch.Tensor([1, 2, 3]).view(-1, 1)

    sparse_adj_t = SparseTensor(row=row, col=col).t()
    dense_adj_t = sparse_adj_t.to_dense()
    edge_index = torch.stack([row, col], dim=0)

    conv = MyConv()

    out1 = conv.propagate(edge_index, x=x)
    out2 = conv.propagate(sparse_adj_t, x=x)
    out3 = conv.propagate(dense_adj_t, x=x)
    assert out1.tolist() == out2.tolist()
    assert out1.tolist() == out3.tolist()

    assert conv.check_consistency(sparse_adj_t, x=x)

    # Static graph.
    x = torch.Tensor([[1, 2, 3], [1, 2, 3]]).view(2, -1, 1)
    out1 = conv.propagate(edge_index, x=x)
    out2 = conv.propagate(sparse_adj_t, x=x)
    out3 = conv.propagate(dense_adj_t.unsqueeze(0), x=x)
    assert out1.tolist() == out2.tolist()
    assert out2.tolist() == out3.tolist()

    assert conv.check_consistency(sparse_adj_t, static_graph=True, x=x)


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, adj_type):
        x = self.lin(x)
        return self.propagate(adj_type, x=x)

    def sparse_message_and_aggregate(self, adj_t, x_j):
        return adj_t.matmul(x_j)

    def dense_message_and_aggregate(self, adj_t, x_j):
        return adj_t.matmul(x_j)

    def message(self, x_j):
        return x_j


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, adj_type):
        x = self.conv1(x, adj_type)
        x = F.relu(x)
        x = self.conv2(x, adj_type)
        return x.log_softmax(dim=-1)


class GCN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN2, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: SparseTensor) -> torch.Tensor:
        x = self.lin1(x)
        x = spmm_sum(adj, x)
        x = F.relu(x)
        x = self.lin2(x)
        x = spmm_sum(adj, x)
        return x.log_softmax(dim=-1)


def test_performance():
    device = torch.device('cuda')
    device = torch.device('cpu')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    d = 32

    for name in ['Cora', 'CiteSeer', 'PubMed']:
        print(name)
        dataset = Planetoid('/tmp/Planetoid', name)
        data = dataset[0].to(device)
        model = GCN(dataset.num_features, d, dataset.num_classes).to(device)
        model2 = GCN2(dataset.num_features, d, dataset.num_classes).to(device)
        model3 = torch.jit.script(model2)

        mask = data.train_mask.nonzero().flatten()
        y = data.y[mask]

        x = data.x
        edge_index = data.edge_index
        sparse_adj_t = SparseTensor(row=edge_index[0], col=edge_index[1]).t()
        dense_adj_t = sparse_adj_t.to_dense()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.01)

        for i in range(1000 + 100):
            if i == 10:
                start.record()
            optimizer.zero_grad()
            out = model(x, edge_index)[mask]
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
        end.record()
        torch.cuda.synchronize()
        print('edge_index', start.elapsed_time(end))

        for i in range(1000 + 100):
            if i == 100:
                start.record()
            # optimizer.zero_grad()
            out = model(x, sparse_adj_t)[mask]
            # loss = F.nll_loss(out, y)
            # loss.backward()
            # optimizer.step()
        end.record()
        torch.cuda.synchronize()
        print('sparse_adj', start.elapsed_time(end))

        for i in range(1000 + 100):
            if i == 100:
                start.record()
            # optimizer2.zero_grad()
            out = model2(x, sparse_adj_t)[mask]
            # loss = F.nll_loss(out, y)
            # loss.backward()
            # optimizer2.step()
        end.record()
        torch.cuda.synchronize()
        print('sparse_adj2', start.elapsed_time(end))

        for i in range(1000 + 100):
            if i == 100:
                start.record()
            # optimizer3.zero_grad()
            out = model3(x, sparse_adj_t)[mask]
            # loss = F.nll_loss(out, y)
            # loss.backward()
            # optimizer3.step()
        end.record()
        torch.cuda.synchronize()
        print('sparse_adj3', start.elapsed_time(end))

        for i in range(200 + 10):
            if i == 10:
                start.record()
            optimizer.zero_grad()
            out = model(x, dense_adj_t)[mask]
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
        end.record()
        torch.cuda.synchronize()
        print('dense_adj', 5 * start.elapsed_time(end))
