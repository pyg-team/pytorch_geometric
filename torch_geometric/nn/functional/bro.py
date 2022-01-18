from typing import Union

import torch


def bro(
    x: torch.Tensor,
    batch: torch.Tensor,
    p: Union[int, str] = 2,
) -> torch.Tensor:
    r"""The Batch Representation Orthogonality penalty from the `"Improving
    Molecular Graph Neural Network Explainability with Orthonormalization
    and Induced Sparsity" <https://arxiv.org/abs/2105.04854>`_ paper

    Computes a regularization for each graph representation in a minibatch
    according to:

    .. math::
        \mathcal{L}_{\textrm{BRO}}^\mathrm{graph} =
          || \mathbf{HH}^T - \mathbf{I}||_p

    And returns an average over all graphs in the batch.

    Args:
        x (torch.Tensor): node-wise feature tensor
        batch (torch.Tensor): one-dimensional tensor indicating node membership
         within batch
        p (str or int): order of the norm. See `torch.norm documentation
         <https://pytorch.org/docs/stable/generated/torch.norm.html>`_

    Returns:
        average BRO penalty in the minibatch

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
