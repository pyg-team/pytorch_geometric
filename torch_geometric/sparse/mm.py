import torch
from torch.autograd import Function


class _Mm(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return torch.mm(a, b)

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if self.needs_input_grad[0]:
            grad_a = torch.mm(grad_output, b.t())

        if self.needs_input_grad[1]:
            grad_b = torch.mm(a.t(), grad_output)

        return grad_a, grad_b


def mm(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.mm(a, b)
    else:
        return _Mm()(a, b)
