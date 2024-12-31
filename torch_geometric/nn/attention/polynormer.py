import torch
import torch.nn.functional as F


class PolynormerAttention(torch.nn.Module):
    r"""The linear scaled attention mechanism from the
    `"Rethinking Attention with Performers"
    <https://arxiv.org/abs/2009.14794>`_ paper.

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of parallel attention heads.
        head_channels (int, optional): Size of each attention head.
            (default: :obj:`64.`)
        kernel (Callable, optional): Kernels for generalized attention.
            If not specified, `ReLU` kernel will be used.
            (default: :obj:`torch.nn.ReLU()`)
        qkv_bias (bool, optional): If specified, add bias to query, key
            and value in the self attention. (default: :obj:`False`)
        attn_out_bias (bool, optional): If specified, add bias to the
            attention output. (default: :obj:`True`)
        dropout (float, optional): Dropout probability of the final
            attention output. (default: :obj:`0.0`)

    """
    def __init__(self, hidden_channels, heads, num_layers, beta, dropout,
                 qk_shared=True):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout
        self.qk_shared = qk_shared

        self.h_lins = torch.nn.ModuleList()
        if not self.qk_shared:
            self.q_lins = torch.nn.ModuleList()
        self.k_lins = torch.nn.ModuleList()
        self.v_lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for i in range(num_layers):
            self.h_lins.append(
                torch.nn.Linear(heads * hidden_channels,
                                heads * hidden_channels))
            if not self.qk_shared:
                self.q_lins.append(
                    torch.nn.Linear(heads * hidden_channels,
                                    heads * hidden_channels))
            self.k_lins.append(
                torch.nn.Linear(heads * hidden_channels,
                                heads * hidden_channels))
            self.v_lins.append(
                torch.nn.Linear(heads * hidden_channels,
                                heads * hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads * hidden_channels))
        self.lin_out = torch.nn.Linear(heads * hidden_channels,
                                       heads * hidden_channels)

    def reset_parameters(self):
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        if not self.qk_shared:
            for q_lin in self.q_lins:
                q_lin.reset_parameters()
        for k_lin in self.k_lins:
            k_lin.reset_parameters()
        for v_lin in self.v_lins:
            v_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, x):
        seq_len, _ = x.size()
        for i in range(self.num_layers):
            h = self.h_lins[i](x)
            k = F.sigmoid(self.k_lins[i](x)).view(seq_len,
                                                  self.hidden_channels,
                                                  self.heads)
            if self.qk_shared:
                q = k
            else:
                q = F.sigmoid(self.q_lins[i](x)).view(seq_len,
                                                      self.hidden_channels,
                                                      self.heads)
            v = self.v_lins[i](x).view(seq_len, self.hidden_channels,
                                       self.heads)

            # numerator
            kv = torch.einsum('ndh, nmh -> dmh', k, v)
            num = torch.einsum('ndh, dmh -> nmh', q, kv)

            # denominator
            k_sum = torch.einsum('ndh -> dh', k)
            den = torch.einsum('ndh, dh -> nh', q, k_sum).unsqueeze(1)

            # linear global attention based on kernel trick
            beta = self.beta
            x = (num / den).reshape(seq_len, -1)
            x = self.lns[i](x) * (h + beta)
            x = F.relu(self.lin_out(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'heads={self.heads}, '
                f'head_channels={self.head_channels} '
                f'kernel={self.kernel})')
