from typing import Union

import torch


def bro(
    x: torch.Tensor,
    batch: torch.Tensor,
    p: Union[int, str] = 2,
) -> torch.Tensor:
    r"""The Batch Representation Orthogonality penalty from the `"Improving
    Molecular Graph Neural Network Explainability with Orthonormalization
    and Induced Sparsity" <https://arxiv.org/abs/2105.04854>`_ paper.

    Computes a regularization for each graph representation in a mini-batch
    according to

    .. math::
        \mathcal{L}_{\textrm{BRO}}^\mathrm{graph} =
          || \mathbf{HH}^T - \mathbf{I}||_p

    and returns an average over all graphs in the batch.

    Args:
        x (torch.Tensor): The node feature matrix.
        batch (torch.Tensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        p (int or str, optional): The norm order to use. (default: :obj:`2`)
    """
    _, counts = torch.unique(batch, return_counts=True)
    diags = torch.stack([
        torch.diag(x) for x in torch.nn.utils.rnn.pad_sequence(
            sequences=torch.ones_like(batch).split_with_sizes(counts.tolist()),
            padding_value=0.,
            batch_first=True,
        )
    ])
    x = x.split_with_sizes(split_sizes=counts.tolist())
    x = torch.nn.utils.rnn.pad_sequence(
        sequences=x,
        padding_value=0.,
        batch_first=True,
    )
    return torch.sum(torch.norm(x @ x.transpose(1, 2) - diags, p=p,
                                dim=(1, 2))) / counts.shape[0]
