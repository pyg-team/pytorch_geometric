import torch
import torch.nn.functional as F

from torch_geometric.typing import Adj, OptTensor


class LabelUsage(torch.nn.Module):
    r"""The label usage operator for semi-supervised node classification,
    as introduced in `"Bag of Tricks for Node Classification"
    <https://arxiv.org/abs/2103.13355>`_ paper.

    .. note::

        When using the :class:`LabelUsage`, adjust the model's input dimension
        accordingly to include both features and classes.

    Args:
        base_model: An instance of the model that will do the
            inner forward pass.
        num_classes (int): Number of classes in dataset
        split_ratio (float): Proportion of true labels to use as features
            during training (default: :obj:'0.5')
        num_recycling_iterations (int): Number of iterations for the
            label reuse procedure to cycle predicted soft labels
            (default: :obj:'100')
        return_tuple (bool): If true, returns (output, train_label_idx,
            train_pred_idx) otherwise returns prediction output
            (default :obj:'False')
    """
    def __init__(
        self,
        base_model: torch.nn.Module,
        num_classes: int,
        split_ratio: float = 0.5,
        num_recycling_iterations: int = 100,
        return_tuple: bool = False,
    ):

        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.split_ratio = split_ratio
        self.num_recycling_iterations = num_recycling_iterations
        self.return_tuple = return_tuple

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        y: Tensor,
        train_idx: Tensor,
        batch: OptTensor = None,
    ):
        r"""Forward pass using label usage algorithm.

        Args:
            x (torch.Tensor): Node feature tensor
            edge_index (torch.Tensor or SparseTensor): The edge indices
            y (torch.Tensor): Node label tensor
            train_idx (torch.Tensor): Global indices of all
                nodes in training set
            batch (torch.Tensor, optional): Optional mini-batch specification
                indicating global node indices in batch (default: :obj:'None')
        """
        # re-assign train_idx to be nodes in mini-batch
        if batch is not None:
            train_idx = batch

        # random masking to split train_idx based on split ratio
        mask = torch.rand(train_idx.shape) < self.split_ratio
        train_labels_idx = train_idx[
            mask]  # D_L: nodes with features and labels
        train_pred_idx = train_idx[~mask]  # D_U: nodes to predict labels

        # add labels to features for train_labels_idx nodes
        # zero value nodes in train_pred_idx
        onehot = torch.zeros([x.shape[0], self.num_classes]).to(x.device)
        onehot[train_labels_idx,
               y[train_labels_idx]] = 1  # create a one-hot encoding
        feat = torch.cat([x, onehot], dim=-1)

        # label reuse procedure
        for _ in range(max(1, self.num_recycling_iterations)):
            output = self.base_model(feat, edge_index)
            pred_labels = F.softmax(output, dim=1)
            feat[train_pred_idx] = torch.cat(
                [x[train_pred_idx], pred_labels[train_pred_idx]], dim=1)

        # return tuples if specified
        if self.return_tuple:
            return output, train_labels_idx, train_pred_idx
        return output
