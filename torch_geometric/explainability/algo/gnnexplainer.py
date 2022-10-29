from math import sqrt
from typing import Optional, Tuple
from warnings import warn

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.explainability.algo import ExplainerAlgorithm
from torch_geometric.explainability.algo.utils import clear_masks, set_masks
from torch_geometric.explainability.explanations import Explanation

EPS = 1e-15

# TODO: rewrite more properly, just POC for now.


class GNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        num_hops (int, optional): The number of hops the :obj:`model` is
            aggregating information from.
            If set to :obj:`None`, will automatically try to detect this
            information based on the number of
            :class:`~torch_geometric.nn.conv.message_passing.MessagePassing`
            layers inside :obj:`model`. (default: :obj:`None`)
        return_type (str, optional): Denotes the type of output from
            :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model
            returns the logarithm of probabilities), :obj:`"prob"` (the
            model returns probabilities), :obj:`"raw"` (the model returns raw
            scores) and :obj:`"regression"` (the model returns scalars).
            (default: :obj:`"log_prob"`)
        feat_mask_type (str, optional): Denotes the type of feature mask
            that will be learned. Valid inputs are :obj:`"feature"` (a single
            feature-level mask for all nodes), :obj:`"individual_feature"`
            (individual feature-level masks for each node), and :obj:`"scalar"`
            (scalar mask for each each node). (default: :obj:`"feature"`)
        allow_edge_mask (boolean, optional): If set to :obj:`False`, the edge
            mask will not be optimized. (default: :obj:`True`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in :attr:`~torch_geometric.nn.models.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        num_hops: Optional[int] = None,
        return_type: str = 'regression',
        feat_mask_type: str = 'feature',
        allow_edge_mask: bool = True,
        **kwargs,
    ):
        super().__init__()
        if feat_mask_type not in ['feature', 'individual_feature', 'scalar']:
            raise ValueError(f'Invalid feature mask type: {feat_mask_type}')
        self.allow_edge_mask = allow_edge_mask
        self.feat_mask_type = feat_mask_type
        self.coeffs.update(kwargs)
        self.epochs = epochs
        self.lr = lr
        self.num_hops = num_hops
        self.return_type = return_type

    @property
    def accept_new_loss(self) -> bool:
        return False

    def supports(self, explanation_type: str, mask_type: str) -> bool:
        if mask_type != "layers":
            return False
        if self.feat_mask_type in ["feature", "individual_feature"]:
            return explanation_type in ["node_feat", "node_and_edge_feat"]
        if self.feat_mask_type == "scalar":
            return explanation_type in ["nodes"]

        return True

    def loss(
        self,
        y_hat: Tensor,
        y: Tensor,
    ) -> Tensor:
        warn('The loss function is wrong for now.')
        loss = torch.nn.functional.mse_loss(y_hat, y)

        if self.allow_edge_mask:
            m = self.edge_mask.sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def explain(self, g: Data, model: torch.nn.Module, target: torch.Tensor,
                target_index: Optional[int] = None,
                batch: Optional[torch.Tensor] = None,
                task_level: str = "graph", **kwargs) -> Explanation:
        if task_level == "graph_level":
            attributions = self.explain_graph(g=g, model=model, target=target,
                                              target_index=target_index,
                                              batch=batch, **kwargs)
        elif task_level == "node_level":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid task level: {task_level}")
        return self._create_explanation_from_masks(g, attributions,
                                                   self.feat_mask_type)

    def _initialize_masks(
        self,
        x: Tensor,
        edge_index: Tensor,
    ):
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1

        if self.feat_mask_type == 'individual_feature':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, F) * std)
        elif self.feat_mask_type == 'scalar':
            self.node_feat_mask = torch.nn.Parameter(torch.randn(N, 1) * std)
        else:
            self.node_feat_mask = torch.nn.Parameter(torch.randn(1, F) * std)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if self.allow_edge_mask:
            self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

    def _clear_masks(self, model: torch.nn.Module):
        clear_masks(model)
        self.node_feat_masks = None
        self.edge_mask = None

    def _create_explanation_from_masks(self, g, attributions, mask_type):
        if mask_type in ['feature', 'individual_feature']:
            return Explanation(x=g.x, edge_index=g.edge_index,
                               node_features_mask=attributions[0],
                               edge_mask=attributions[1])
        elif mask_type == 'scalar':
            return Explanation(x=g.x, edge_index=g.edge_index,
                               node_mask=attributions[0],
                               edge_mask=attributions[1])
        else:
            raise ValueError(f"Invalid mask type: {mask_type}")

    def explain_graph(
        self,
        model: torch.nn.Module,
        g: Data,
        target: torch.Tensor,
        target_index: Optional[int] = 0,
        batch: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        g_copy = g.clone()
        initial_pred = model(g_copy, **kwargs)
        model.eval()
        self._clear_masks(model)
        self._initialize_masks(g_copy.x, g_copy.edge_index)
        self.to(g_copy.x.device)

        if self.allow_edge_mask:
            set_masks(model, self.edge_mask, g_copy.edge_index,
                      apply_sigmoid=True)
            parameters = [self.node_feat_mask, self.edge_mask]
        else:
            parameters = [self.node_feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for _ in range(1, self.epochs + 1):
            optimizer.zero_grad()
            g_copy.x = g.x * self.node_feat_mask.sigmoid()
            out = model(g_copy, batch=batch, **kwargs)

            loss = self.objective(
                out,
                initial_pred,
                target,
                target_index,
            )
            loss.backward(retain_graph=True)
            optimizer.step()

        node_feat_mask = self.node_feat_mask.detach().sigmoid().squeeze()
        if self.allow_edge_mask:
            edge_mask = self.edge_mask.detach().sigmoid()
        else:
            edge_mask = torch.ones(g_copy.edge_index.size(1))

        self._clear_masks(model)
        return node_feat_mask, edge_mask
