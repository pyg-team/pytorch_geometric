import torch
from torch.autograd import Function


class EdgewiseSplineWeightingCPU(Function):
    def __init__(self, amount, index):
        super(EdgewiseSplineWeightingCPU, self).__init__()
        self.amount = amount
        self.index = index

    def forward(self, input, weight):
        self.save_for_backward(input, weight)

        _, M_in, M_out = weight.size()
        k_max = self.amount.size(1)

        output = input.new(input.size(0), M_out).fill_(0)

        for k in range(k_max):
            b = self.amount[:, k]  # [|E|]
            c = self.index[:, k]  # [|E|]

            for i in range(M_in):
                w = weight[:, i]  # [K x M_out]
                w = w[c]  # [|E| x M_out]
                f = input[:, i]  # [|E|]

                # Need to transpose twice, so we can make use of broadcasting.
                output += (f * b * w.t()).t()  # [|E| x M_out]

        return output

    def backward(self, grad_output):
        input, weight = self.saved_tensors

        K, M_in, M_out = weight.size()
        k_max = self.amount.size(1)
        num_edges = input.size(0)

        grad_input = grad_output.new(num_edges, M_in).fill_(0)
        grad_weight = grad_output.new(K, M_in, M_out).fill_(0)

        for k in range(k_max):
            b = self.amount[:, k]  # [|E|]
            c = self.index[:, k]  # [|E|]
            c_expand = c.contiguous().view(-1, 1).expand(c.size(0), M_out)

            for i in range(M_in):
                w = weight[:, i]  # [K x M_out]
                w = w[c]  # [|E| x M_out]

                f = b * torch.sum(grad_output * w, dim=1)  # [|E|]
                grad_input[:, i] += f

                f = input[:, i]  # [|E|]
                w_grad = (f * b * grad_output.t()).t()  # [|E|, M_out]
                grad_weight[:, i, :].scatter_add_(0, c_expand, w_grad)

        return grad_input, grad_weight
