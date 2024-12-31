import torch
import torch.nn.functional as F


class PolynormerAttention(torch.nn.Module):
    r"""The polynomial-expressive attention mechanism from the
    `"Polynormer: Polynomial-Expressive Graph Transformer in Linear Time"
    <https://arxiv.org/abs/2403.01232>`_ paper.

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of parallel attention heads.
        head_channels (int, optional): Size of each attention head.
            (default: :obj:`64.`)
        beta (float, optional): Polynormer beta initialization.
            (default: :obj:`0.9`)
        qkv_bias (bool, optional): If specified, add bias to query, key
            and value in the self attention. (default: :obj:`False`)
        qk_shared (bool optional): Whether weight of query and key are shared.
            (default: :obj:`True`)
        dropout (float, optional): Dropout probability of the final
            attention output. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        channels: int,
        heads: int,
        head_channels: int = 64,
        beta: float = 0.9,
        qkv_bias: bool = False,
        qk_shared: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.head_channels = head_channels
        self.heads = heads
        self.beta = beta
        self.qk_shared = qk_shared

        inner_channels = heads * head_channels
        self.h_lins = torch.nn.Linear(channels, inner_channels)
        if not self.qk_shared:
            self.q = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.lns = torch.nn.LayerNorm(inner_channels)
        self.lin_out = torch.nn.Linear(inner_channels, inner_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        N = x.size(0)
        h = self.h_lins(x)
        k = F.sigmoid(self.k(x)).view(N, self.head_channels, self.heads)
        if self.qk_shared:
            q = k
        else:
            q = F.sigmoid(self.q(x)).view(N, self.head_channels, self.heads)
        v = self.v(x).view(N, self.head_channels, self.heads)

        # numerator
        kv = torch.einsum('ndh, nmh -> dmh', k, v)
        num = torch.einsum('ndh, dmh -> nmh', q, kv)

        # denominator
        k_sum = torch.einsum('ndh -> dh', k)
        den = torch.einsum('ndh, dh -> nh', q, k_sum).unsqueeze(1)

        # linear global attention based on kernel trick
        x = (num / den).reshape(N, -1)
        x = self.lns(x) * (h + self.beta)
        x = F.relu(self.lin_out(x))
        x = self.dropout(x)

        return x

    def reset_parameters(self):
        self.h_lins.reset_parameters()
        if not self.qk_shared:
            self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()
        self.lns.reset_parameters()
        self.lin_out.reset_parameters()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'heads={self.heads}, '
                f'head_channels={self.head_channels})')
