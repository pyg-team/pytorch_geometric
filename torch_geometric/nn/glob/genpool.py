import torch
from torch_scatter import scatter_add, scatter_softmax


class GENPool(torch.nn.Module):
    def __init__(self, family = "softmax", p = 1.0, beta = 1.0, 
                 trainable_p = True, trainable_beta = True):
        r""" Global Generalized Mean-Max-Sum pooling function from the
        `"Improving Graph Property Prediction with Generalized Readout Functions" <https://arxiv.org/abs/1511.05493>`_ paper.

        Performs batch-wise graph-level-outputs by transforming node
        features based on a Generalized Mean-Max-Sum function, so that
        for a single graph :math:`\mathcal{G}_i` its output is computed
        deppending on the family of transformations by:

        .. math::
            \mathbf{r}_i = \frac{N_{i}}{1+\beta*\left(N_{i}-1\right)} \sum_{n=1}^{N_i} \mathbf{softmax} \left( \mathbf{x}_n * p \right) * \mathbf{x}_n
        for softmax aggregation or
        .. math::
            \mathbf{r}_i = \left( \frac{1}{1+\beta*\left(N_{i}-1\right)} \sum_{n=1}^{N_i} \mathbf{x}_n^{p} \right)^{1/p}
        for power mean aggregation.

        Args:
            family (str): family of generalized mean-max functions to use. 
                Either "softmax" or "power" for eq. 1 or eq. 2 respectively.
            p (float): parameter for the generalized mean-max-sum function.
                Regulates the importance of the different embedding properties.
            beta (float): parameter for the generalized mean-max-sum function.
                Regulates the importance of the graph size. 
            trainable_p (bool): whether the value of p is learnable during training.
            trainable_beta (bool): whether the value of beta is learnable during training.
        """
        super(GENPool, self).__init__()
        
        self.family         = family
        self.base_p         = p
        self.base_beta      = beta
        self.trainable_p    = trainable_p
        self.trainable_beta = trainable_beta
        # define params
        self.p = torch.nn.Parameter(torch.tensor([p], device=device),
                                    requires_grad=trainable_p)# .to(device)
        self.beta = torch.nn.Parameter(torch.tensor([beta], device=device),
                                       requires_grad=trainable_beta)# .to(device)


    def forward(self, x, batch, bsize=None):
        r"""Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
            batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
                B-1\}}^N`, which assigns each node to a specific example.
            size (int, optional): Batch-size :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
        :rtype: :class:`Tensor`
        """
        bsize = int(batch.max().item() + 1) if bsize is None else bsize
        n_nodes = scatter_add(torch.ones_like(x), batch, dim=0, dim_size=bsize)
        if self.family == "softmax":
            out = scatter_softmax(self.p * x.detach(), batch, dim=0)
            return scatter_add(x * out,
                                batch, dim=0, dim_size=bsize)*n_nodes / (1+self.beta*(n_nodes-1))

        elif self.family == "power":
            # numerical stability - avoid powers of large numbers or negative ones
            min_x, max_x = 1e-7, 1e+3
            torch.clamp_(x, min_x, max_x)
            out = scatter_add(torch.pow(x, self.p),
                               batch, dim=0, dim_size=bsize) / (1+self.beta*(n_nodes-1))
            torch.clamp_(out, min_x, max_x)
            return torch.pow(out, 1 / self.p)


    def reset_parameters(self):
        if self.p and torch.is_tensor(self.p):
            self.p.data.fill_(self.base_p)
        if self.beta and torch.is_tensor(self.beta):
            self.beta.data.fill_(self.base_beta)


    def __repr__(self):
        return "Generalized Mean-Max-Sum global pooling layer with params:" + \
               str({"family": self.family,
                    "base_p": self.base_p,
                    "base_beta"     : self.base_beta,
                    "trainable_p"   : self.trainable_p,
                    "trainable_beta": self.trainable_beta})

