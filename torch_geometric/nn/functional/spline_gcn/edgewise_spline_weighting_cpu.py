import torch
from torch.autograd import Variable, Function


class _EdgewiseSplineGcn(Function):
    def __init__(self, values, kernel_size, max_radius, degree=1):
        self.dim = len(kernel_size)
        self.m = degree + 1
        amount, index = spline_weights(values, kernel_size, max_radius, degree)
        self.amount = amount
        self.index = index

    def forward(self, features, weight):
        self.save_for_backward(features, weight)
        K, M_in, M_out = weight.size()

        features_out = torch.zeros(features.size(0), M_out).type_as(features)

        for k in range(self.m**self.dim):
            b = self.amount[:, k]  # [|E|]
            c = self.index[:, k]  # [|E|]

            for i in range(M_in):
                w = weight[:, i]  # [K x M_out]
                w = w[c]  # [|E| x M_out]
                f = features[:, i]  # [|E|]

                # Need to transpose twice, so we can make use of broadcasting.
                features_out += (f * b * w.t()).t()  # [|E| x M_out]

        return features_out

    def backward(self, features_grad_out):
        # features_grad_out: [|E| x M_out]
        # features_in: [|E|] x M_in]
        # weight: [K x M_in x M_out]
        features_in, weight = self.saved_tensors
        K, M_in, M_out = weight.size()
        n = features_in.size(0)

        features_grad_in = torch.zeros(n, M_in).type_as(features_in)
        weight_grad_in = torch.zeros(weight.size()).type_as(weight)

        for k in range(self.m**self.dim):
            b = self.amount[:, k]  # [|E|]
            c = self.index[:, k]  # [|E|]
            c_expand = c.contiguous().view(-1, 1).expand(c.size(0), M_out)

            for i in range(M_in):
                w = weight[:, i]  # [K x M_out]
                w = w[c]  # [|E| x M_out]

                f = b * torch.sum(features_grad_out * w, dim=1)  # [|E|]
                features_grad_in[:, i] += f

                f = features_in[:, i]  # [|E|]
                w_grad = (f * b * features_grad_out.t()).t()  # [|E|, M_out]
                weight_grad_in[:, i, :].scatter_add_(0, c_expand, w_grad)

        return features_grad_in, weight_grad_in


def edgewise_spline_gcn(values,
                        features,
                        weight,
                        kernel_size,
                        max_radius,
                        degree=1):
    op = _EdgewiseSplineGcn(values, kernel_size, max_radius, degree)
    return op(features, weight)
