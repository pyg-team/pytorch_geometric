import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.typing import Adj, SparseTensor

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
            (default: :obj:'0')
        return_tuple (bool): If true, returns (output, train_label_idx, 
            train_pred_idx) otherwise returns prediction output 
            (default :obj:'False')
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        num_classes: int,
        split_ratio: float = 0.5,
        num_recycling_iterations: int = 0,
        return_tuple: bool = False,
    ):

        super(LabelUsage, self).__init__()
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
        val_idx: Tensor,
        test_idx: Tensor,
        ):
        r"""
        Forward pass using label usage algorithm.

        Args:
            feat (torch.Tensor): Node feature tensor of dimension (N,F)
                where N is the number of nodes and 
                F is the number of features per node
            edge_index (torch.Tensor or SparseTensor): The edge indices with
                dimension (N,N) for a dense adjacency matrix or (2,E) for a
                sparse adjacency representation where E is the number of edges
            y (torch.Tensor): Node ground-truth labels tensor of dimension
                of (N,) for 1D tensor or (N,1) for 2D tensor
            mask (torch.Tensor): Global indices of all nodes in training 
                set or mini-batch with dimension (N_train,) where N_train
                is the number of nodes in training set or mini-batch
            val_idx (torch.Tensor): Node indices in validation set with 
                dimension (N_val,) where N_val is the number of nodes
                in validation set 
            test_idx (torch.Tensor): Node indices in test set with dimension
                (N_test,) where N_test is the number of nodes in test set
        """
        assert feat.dim() == 2, "feat must be 2D but got shape {feat.shape}"

        # random split mask based on split ratio
        split_mask = torch.rand(mask.shape) < self.split_ratio
        train_labels_idx = mask[split_mask]  # D_L: nodes with features and labels
        train_pred_idx = mask[~split_mask]  # D_U: nodes to predict labels 

        # add labels to features for train_labels_idx nodes
        # zero value nodes in train_pred_idx
        onehot = torch.zeros([feat.shape[0], self.num_classes]).to(feat.device)
        # create a one-hot encoding according to tensor dim
        if y.dim() == 2:
            onehot[train_labels_idx, y[train_labels_idx,0]] = 1  
        else:
            onehot[train_labels_idx, y[train_labels_idx]] = 1 
        feat = torch.cat([feat, onehot], dim=-1)

        # set predictions for all unlabeled indices
        unlabeled_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        # label reuse procedure
        pred = self.base_model(feat, edge_index)
        for _ in range(max(1, self.num_recycling_iterations+1)):
            pred = pred.detach()
            feat[unlabeled_idx, -self.num_classes:] = F.softmax(pred[unlabeled_idx], dim=-1)
            pred = self.base_model(feat, edge_index)

        # return tuples if specified
        if self.return_tuple:
            return output, train_labels_idx, train_pred_idx
        return output
