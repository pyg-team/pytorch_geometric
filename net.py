import torch
from torch.autograd import Variable
import torch.nn.functional as F

from torch_geometric.nn.modules import GCN

i = torch.LongTensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
w = torch.FloatTensor([1, 1, 1, 1, 1, 1])
a = torch.sparse.FloatTensor(i, w, torch.Size([3, 3]))
A = Variable(a)

f = torch.FloatTensor([[2], [4], [8]])
FF = Variable(f)
y = torch.FloatTensor([4, 16, 64])
Y = Variable(y)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = GCN(1, 10, bias=False)
        self.fc = torch.nn.Linear(30, 3)

    def forward(self, adj, features):
        output = self.conv(adj, features)
        output = F.relu(output)
        output = output.view(-1)
        output = self.fc(output)
        return output


model = Net()
loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

for epoch in range(1, 20):
    y_pred = model(A, FF)
    loss = loss_fn(y_pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
