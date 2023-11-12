from typing import Optional

import torch

from torch_geometric.utils import to_dense_adj


class NormalizedDCG():
    """A class for computing the Normalized Discounted Cumulative Gain
    (NDCG) metric.

    It may be used to evaluate link-prediction tasks. In this context,
    a given head node represents a query for which we should or should
    not retrieve such or such tail node to create a link.

    .. code-block::

        ndcgk = NormalizedDCG(k=2)

       # Prediction matrix holding link probability for each pair of nodes.
       pred_mat = torch.Tensor(
            [
                [1,0,0],
                [1,1,0],
                [0,1,1]
            ]
       )

       # Tensor holding valid links to consider.
       edge_label_index = torch.Tensor(
            [
                [0,0,2,2], # Head nodes (queries)
                [0,1,2,0]  # Valid tail nodes
            ]
       )

       ndcgk.update(pred_mat, edge_label_index)

       result = ndcgk.compute() # Output shape: [1]

    The update step can be repeated multiple times to accumulate the results
    of successive predictions. For instance, it can be used inside a mini-batch
    training loop.

    Args:
        k (int): Consider only the top k elements for each query.
    """
    def __init__(self, k: Optional[int] = None):
        self.k = k
        self.dcg = torch.Tensor([])
        self.idcg = torch.Tensor([])

    def _dcg(self, pred_mat: torch.Tensor,
             label_mat: torch.Tensor) -> torch.Tensor:
        """Computes the Discounted Cumulative Gain (DCG)
        for the given link-predictions and valid links.

        Args:
            pred_mat (torch.Tensor): The link-predictions matrix.
            label_mat (torch.Tensor): The valid link tensor.

        Returns:
            torch.Tensor: The DCG value per head node.
        """
        k = pred_mat.size(1) if self.k is None else self.k

        top_k_pred_mat = pred_mat.argsort(dim=1, descending=True)[:, :k]
        pred_mat_heads = torch.arange(top_k_pred_mat.size(0)).repeat(
            k, 1).T.reshape(-1)
        top_k_pred_labels = label_mat[(
            pred_mat_heads,
            top_k_pred_mat.reshape(-1))].reshape_as(top_k_pred_mat)
        log_ranks = torch.log2(torch.arange(k) + 2)
        return (top_k_pred_labels / log_ranks).sum(dim=1)

    def update(self, pred_mat: torch.Tensor,
               pos_edge_label_index: torch.Tensor) -> None:
        """Updates state with predictions and targets.

        Args:
            pred_mat (torch.Tensor): The matrix containing link-predictions
                for each pair of nodes.
            pos_edge_label_index (torch.Tensor): The edge indices of
                valid links.
        """
        if self.k is not None:
            assert pred_mat.size(
                1) >= self.k, f"At least {self.k} entries for each query."

        edge_label_matrix = to_dense_adj(pos_edge_label_index,
                                         max_num_nodes=pred_mat.size(0)).view(
                                             pred_mat.size(0), -1)

        row_dcg = self._dcg(pred_mat, edge_label_matrix)
        row_idcg = self._dcg(edge_label_matrix, edge_label_matrix)

        self.dcg = torch.concat((self.dcg, row_dcg))
        self.idcg = torch.concat((self.idcg, row_idcg))

    def compute(self) -> torch.Tensor:
        """Computes NDCG based on previous updates.

        Returns:
             torch.Tensor: The average NDCG value.
        """
        non_zeros = self.idcg.nonzero()
        return (self.dcg[non_zeros] / self.idcg[non_zeros]).mean()
