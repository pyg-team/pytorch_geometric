import torch
from torch.autograd import Variable


class mm(torch.autograd.Function):
    def forward(self, W, x):
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W, grad_input)
        return grad_input_dL_dW, grad_input_dL_dx


A = Variable(
    torch.FloatTensor([[0, 0, 3], [4, 0, 0], [0, 0, 0]]), requires_grad=True)
F = Variable(torch.FloatTensor([[1, 2], [3, 4], [5, 6]]), requires_grad=True)
out = torch.mm(A, F)
out = out.mean()
out.backward()
print('A', A.grad)
print('F', F.grad)

i = torch.LongTensor([[0, 1], [2, 0]])
v = torch.FloatTensor([3, 4])
a = torch.sparse.FloatTensor(i, v, torch.Size([3, 3]))
f = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
A = Variable(a, requires_grad=True)
F = Variable(f, requires_grad=True)

out = mm()(A, F)
out = out.mean()
out.backward()
print('A', A.grad)
print('F', F.grad)
