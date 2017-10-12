import torch
from torch.autograd import Function


class Mm(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return torch.mm(a, b)

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_da = torch.mm(grad_input, b.t())
        grad_input_dL_db = torch.mm(a.t(), grad_input)
        return grad_input_dL_da, grad_input_dL_db


autograd_mm = Mm()


def mm(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.mm(a, b)
    else:
        return autograd_mm(a, b)
