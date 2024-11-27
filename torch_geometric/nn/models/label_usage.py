import torch
import torch.nn.functional as F


class LabelUsage(torch.nn.Module):
    r"""The label usage operator for semi-supervised node classification,
    as introduced in `"Bag of Tricks for Node Classification"
    <https://arxiv.org/abs/2103.13355>`_ paper.

    .. note::

        When using the :class:`LabelUsage`, adjust the model's input dimension
        accordingly to include both features and classes.

    Args:
        split_ratio (float): Proportion of true labels to use as features 
            during training.
        num_recycling_iterations (int): Number of iterations for the label
            reuse procedure to cycle predicted soft labels
        return_tuple (bool): If true, returns (output, train_label_idx, 
        train_pred_idx) otherwise returns prediction output
        base_model: An instance of the model that will do the 
            inner forward pass.
        num_classes (int): Number of classes in dataset
    """

    def __init__(
        self,
        split_ratio: float,
        num_recycling_iterations: int,
        return_tuple: bool,
        base_model: torch.nn.Module,
    ):

        super(LabelUsage, self).__init__()
        self.split_ratio = split_ratio
        self.num_recycling_iterations = num_recycling_iterations
        self.return_tuple = return_tuple
        self.base_model = base_model

    def forward(self, x, edge_index, y, train_idx):
        r"""
        Forward pass using label usage algorithm.

        Args:
          x: Node feature tensor 
          edge_index: The edge connectivity
          y: Node label tensor 
          train_idx: Training index tensor with labels

        """
        # random masking to split train_idx based on split ratio
        mask = torch.rand(train_idx.shape) < self.split_ratio
        train_labels_idx = train_idx[mask]  # D_L: nodes with features and labels
        train_pred_idx = train_idx[~mask]  # D_U: nodes to predict labels in training

        # add labels to features for train_labels_idx nodes
        onehot = torch.zeros([x.shape[0], len(torch.unique(y))]).to(x.device)
        onehot[train_labels_idx, y[train_labels_idx]] = 1  # create a one-hot encoding
        feat = torch.cat([x, onehot], dim=-1)

        # label reuse procedure
        for _ in range(max(1, self.num_recycling_iterations)):
            output = self.base_model(feat, edge_index)
            pred_labels = F.softmax(output, dim=1)
            feat[train_pred_idx] = torch.cat(
                [x[train_pred_idx], pred_labels[train_pred_idx]], dim=1
            )

        # return tuples if specified
        if self.return_tuple:
            return output, train_labels_idx, train_pred_idx
        return output
