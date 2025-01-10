import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj


class LabelUsage(torch.nn.Module):
    r"""The label usage operator for semi-supervised node classification,
    as introduced in `"Bag of Tricks for Node Classification"
    <https://arxiv.org/abs/2103.13355>`_ paper.

    Label usage splits training nodes into labeled and unlabeled subsets. The
    labeled subset incorporates labels as features while the unlabeled subset
    labels are zeroed and used for prediction. During inference, previously
    predicted soft labels for unlabeled nodes are recycled as inputs for the
    model, refining predictions iteratively.

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
            (default: :obj:'0')
        return_tuple (bool): If true, returns (pred, train_label,
            train_pred) during training otherwise returns
            prediction output (default :obj:'False')
    """
    def __init__(
        self,
        base_model: torch.nn.Module,
        num_classes: int,
        split_ratio: float = 0.5,
        num_recycling_iterations: int = 0,
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
        feat: Tensor,
        edge_index: Adj,
        y: Tensor,
        mask: Tensor,
    ):
        r"""Forward pass using label usage algorithm.

        Args:
            feat (torch.Tensor): Node feature tensor of dimension (N,F)
                where N is the number of nodes and F is the number
                of features per node
            edge_index (torch.Tensor or SparseTensor): The edge connectivity
                to be passed to base_model
            y (torch.Tensor): Node ground-truth labels tensor of dimension
                of (N,) for 1D tensor or (N,1) for 2D tensor
            mask (torch.Tensor): A mask or index tensor denoting which nodes
                are used during training
        """
        assert feat.dim() == 2, f"feat must be 2D but got shape {feat.shape}"
        assert y.dim() == 1 or (y.dim() == 2 and y.size(1) == 1), \
            f"Expected y to be either (N,) or (N, 1), but got shape {y.shape}"

        # set unlabeled mask for unlabeled indices
        unlabeled_mask = torch.ones(feat.size(0),
                                    dtype=torch.bool).to(feat.device)

        # add labels to features for train_labels nodes if in training
        # else fill true labels for all nodes in mask
        # zero value nodes in train_pred
        onehot = torch.zeros([feat.shape[0], self.num_classes]).to(feat.device)
        if self.training:
            # random split mask based on split ratio
            if mask.dtype == torch.bool:
                mask = mask.nonzero(as_tuple=False).view(-1)
            split_mask = torch.rand(mask.shape) < self.split_ratio
            train_labels = mask[split_mask]  # D_L: nodes with labels
            train_pred = mask[~split_mask]  # D_U: nodes to predict labels

            unlabeled_mask[train_labels] = False

            # create a one-hot encoding according to tensor dim
            if y.dim() == 2:
                onehot[train_labels, y[train_labels, 0]] = 1
            else:
                onehot[train_labels, y[train_labels]] = 1
        else:
            unlabeled_mask[mask] = False

            # create a one-hot encoding according to tensor dim
            if y.dim() == 2:
                onehot[mask, y[mask, 0]] = 1
            else:
                onehot[mask, y[mask]] = 1

        feat = torch.cat([feat, onehot], dim=-1)

        pred = self.base_model(feat, edge_index)

        # label reuse procedure
        for _ in range(self.num_recycling_iterations):
            pred = pred.detach()
            feat[unlabeled_mask,
                 -self.num_classes:] = F.softmax(pred[unlabeled_mask], dim=-1)
            pred = self.base_model(feat, edge_index)

        # return tuples if specified
        if self.return_tuple and self.training:
            return pred, train_labels, train_pred
        return pred
