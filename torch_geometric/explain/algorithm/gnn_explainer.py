from math import sqrt
from typing import Dict, Optional, Tuple, Union, overload

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import (
    ExplainerConfig,
    Explanation,
    HeteroExplanation,
    ModelConfig,
)
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import (
    clear_masks,
    set_hetero_masks,
    set_masks,
)
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
from torch_geometric.typing import EdgeType, NodeType


class GNNExplainer(ExplainerAlgorithm):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and node features that play a crucial role in the predictions
    made by a GNN.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    .. note::

        The :obj:`edge_size` coefficient is multiplied by the number of nodes
        in the explanation at every iteration, and the resulting value is added
        to the loss as a regularization term, with the goal of producing
        compact explanations.
        A higher value will push the algorithm towards explanations with less
        elements.
        Consider adjusting the :obj:`edge_size` coefficient according to the
        average node degree in the dataset, especially if this value is bigger
        than in the datasets used in the original paper.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    default_coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs = dict(self.default_coeffs)
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
        self.is_hetero = False

    @overload
    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        ...

    @overload
    def forward(
        self,
        model: torch.nn.Module,
        x: Dict[NodeType, Tensor],
        edge_index: Dict[EdgeType, Tensor],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> HeteroExplanation:
        ...

    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:
        self.is_hetero = isinstance(x, dict)
        self._train(model, x, edge_index, target=target, index=index, **kwargs)
        explanation = self._create_explanation()
        self._clean_model(model)
        return explanation

    def _create_explanation(self) -> Union[Explanation, HeteroExplanation]:
        """Create an explanation object from the current masks."""
        if self.is_hetero:
            # For heterogeneous graphs, process each type separately
            node_mask_dict = {}
            edge_mask_dict = {}

            for node_type, mask in self.node_mask.items():
                if mask is not None:
                    node_mask_dict[node_type] = self._post_process_mask(
                        mask,
                        self.hard_node_mask[node_type],
                        apply_sigmoid=True,
                    )

            for edge_type, mask in self.edge_mask.items():
                if mask is not None:
                    edge_mask_dict[edge_type] = self._post_process_mask(
                        mask,
                        self.hard_edge_mask[edge_type],
                        apply_sigmoid=True,
                    )

            # Create heterogeneous explanation
            explanation = HeteroExplanation()
            explanation.set_value_dict('node_mask', node_mask_dict)
            explanation.set_value_dict('edge_mask', edge_mask_dict)

        else:
            # For homogeneous graphs, process single masks
            node_mask = self._post_process_mask(
                self.node_mask,
                self.hard_node_mask,
                apply_sigmoid=True,
            )
            edge_mask = self._post_process_mask(
                self.edge_mask,
                self.hard_edge_mask,
                apply_sigmoid=True,
            )

            # Create homogeneous explanation
            explanation = Explanation(node_mask=node_mask, edge_mask=edge_mask)

        return explanation

    def supports(self) -> bool:
        return True

    @overload
    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> None:
        ...

    @overload
    def _train(
        self,
        model: torch.nn.Module,
        x: Dict[NodeType, Tensor],
        edge_index: Dict[EdgeType, Tensor],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> None:
        ...

    def _train(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> None:
        # Initialize masks based on input type
        self._initialize_masks(x, edge_index)

        # Collect parameters for optimization
        parameters = self._collect_parameters(model, edge_index)

        # Create optimizer
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        # Training loop
        for i in range(self.epochs):
            optimizer.zero_grad()

            # Forward pass with masked inputs
            y_hat = self._forward_with_masks(model, x, edge_index, **kwargs)
            y = target

            # Handle index if provided
            if index is not None:
                y_hat, y = y_hat[index], y[index]

            # Calculate loss
            loss = self._loss(y_hat, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # In the first iteration, collect gradients to identify important
            # nodes/edges
            if i == 0:
                self._collect_gradients()

    def _collect_parameters(self, model, edge_index):
        """Collect parameters for optimization."""
        parameters = []

        if self.is_hetero:
            # For heterogeneous graphs, collect parameters from all types
            for mask in self.node_mask.values():
                if mask is not None:
                    parameters.append(mask)
            if any(v is not None for v in self.edge_mask.values()):
                set_hetero_masks(model, self.edge_mask, edge_index)
            for mask in self.edge_mask.values():
                if mask is not None:
                    parameters.append(mask)
        else:
            # For homogeneous graphs, collect single parameters
            if self.node_mask is not None:
                parameters.append(self.node_mask)
            if self.edge_mask is not None:
                set_masks(model, self.edge_mask, edge_index,
                          apply_sigmoid=True)
                parameters.append(self.edge_mask)

        return parameters

    @overload
    def _forward_with_masks(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tensor:
        ...

    @overload
    def _forward_with_masks(
        self,
        model: torch.nn.Module,
        x: Dict[NodeType, Tensor],
        edge_index: Dict[EdgeType, Tensor],
        **kwargs,
    ) -> Tensor:
        ...

    def _forward_with_masks(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        **kwargs,
    ) -> Tensor:
        """Forward pass with masked inputs."""
        if self.is_hetero:
            # Apply masks to heterogeneous inputs
            h_dict = {}
            for node_type, features in x.items():
                if node_type in self.node_mask and self.node_mask[
                        node_type] is not None:
                    h_dict[node_type] = features * self.node_mask[
                        node_type].sigmoid()
                else:
                    h_dict[node_type] = features

            # Forward pass with masked features
            return model(h_dict, edge_index, **kwargs)
        else:
            # Apply mask to homogeneous input
            h = x if self.node_mask is None else x * self.node_mask.sigmoid()

            # Forward pass with masked features
            return model(h, edge_index, **kwargs)

    def _initialize_masks(
        self,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
    ) -> None:
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        if self.is_hetero:
            # Initialize dictionaries for heterogeneous masks
            self.node_mask = {}
            self.hard_node_mask = {}
            self.edge_mask = {}
            self.hard_edge_mask = {}

            # Initialize node masks for each node type
            for node_type, features in x.items():
                device = features.device
                N, F = features.size()
                self._initialize_node_mask(node_mask_type, node_type, N, F,
                                           device)

            # Initialize edge masks for each edge type
            for edge_type, indices in edge_index.items():
                device = indices.device
                E = indices.size(1)
                N = max(indices.max().item() + 1,
                        max(feat.size(0) for feat in x.values()))
                self._initialize_edge_mask(edge_mask_type, edge_type, E, N,
                                           device)
        else:
            # Initialize masks for homogeneous graph
            device = x.device
            (N, F), E = x.size(), edge_index.size(1)

            # Initialize homogeneous node and edge masks
            self._initialize_homogeneous_masks(node_mask_type, edge_mask_type,
                                               N, F, E, device)

    def _initialize_node_mask(
        self,
        node_mask_type,
        node_type,
        N,
        F,
        device,
    ) -> None:
        """Initialize node mask for a specific node type."""
        std = 0.1
        if node_mask_type is None:
            self.node_mask[node_type] = None
            self.hard_node_mask[node_type] = None
        elif node_mask_type == MaskType.object:
            self.node_mask[node_type] = Parameter(
                torch.randn(N, 1, device=device) * std)
            self.hard_node_mask[node_type] = None
        elif node_mask_type == MaskType.attributes:
            self.node_mask[node_type] = Parameter(
                torch.randn(N, F, device=device) * std)
            self.hard_node_mask[node_type] = None
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask[node_type] = Parameter(
                torch.randn(1, F, device=device) * std)
            self.hard_node_mask[node_type] = None
        else:
            raise ValueError(f"Invalid node mask type: {node_mask_type}")

    def _initialize_edge_mask(self, edge_mask_type, edge_type, E, N, device):
        """Initialize edge mask for a specific edge type."""
        if edge_mask_type is None:
            self.edge_mask[edge_type] = None
            self.hard_edge_mask[edge_type] = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask[edge_type] = Parameter(
                torch.randn(E, device=device) * std)
            self.hard_edge_mask[edge_type] = None
        else:
            raise ValueError(f"Invalid edge mask type: {edge_mask_type}")

    def _initialize_homogeneous_masks(self, node_mask_type, edge_mask_type, N,
                                      F, E, device):
        """Initialize masks for homogeneous graph."""
        # Initialize node mask
        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            raise ValueError(f"Invalid node mask type: {node_mask_type}")

        # Initialize edge mask
        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)
        else:
            raise ValueError(f"Invalid edge mask type: {edge_mask_type}")

    def _collect_gradients(self) -> None:
        if self.is_hetero:
            self._collect_hetero_gradients()
        else:
            self._collect_homo_gradients()

    def _collect_hetero_gradients(self):
        """Collect gradients for heterogeneous graph."""
        for node_type, mask in self.node_mask.items():
            if mask is not None:
                if mask.grad is None:
                    raise ValueError(
                        f"Could not compute gradients for node masks of type "
                        f"'{node_type}'. Please make sure that node masks are "
                        f"used inside the model or disable it via "
                        f"`node_mask_type=None`.")

                self.hard_node_mask[node_type] = mask.grad != 0.0

        for edge_type, mask in self.edge_mask.items():
            if mask is not None:
                if mask.grad is None:
                    raise ValueError(
                        f"Could not compute gradients for edge masks of type "
                        f"'{edge_type}'. Please make sure that edge masks are "
                        f"used inside the model or disable it via "
                        f"`edge_mask_type=None`.")
                self.hard_edge_mask[edge_type] = mask.grad != 0.0

    def _collect_homo_gradients(self):
        """Collect gradients for homogeneous graph."""
        if self.node_mask is not None:
            if self.node_mask.grad is None:
                raise ValueError("Could not compute gradients for node "
                                 "features. Please make sure that node "
                                 "features are used inside the model or "
                                 "disable it via `node_mask_type=None`.")
            self.hard_node_mask = self.node_mask.grad != 0.0

        if self.edge_mask is not None:
            if self.edge_mask.grad is None:
                raise ValueError("Could not compute gradients for edges. "
                                 "Please make sure that edges are used "
                                 "via message passing inside the model or "
                                 "disable it via `edge_mask_type=None`.")
            self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        # Calculate base loss based on model configuration
        loss = self._calculate_base_loss(y_hat, y)

        # Apply regularization based on graph type
        if self.is_hetero:
            # Apply regularization for heterogeneous graph
            loss = self._apply_hetero_regularization(loss)
        else:
            # Apply regularization for homogeneous graph
            loss = self._apply_homo_regularization(loss)

        return loss

    def _calculate_base_loss(self, y_hat, y):
        """Calculate base loss based on model configuration."""
        if self.model_config.mode == ModelMode.binary_classification:
            return self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            return self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            return self._loss_regression(y_hat, y)
        else:
            raise ValueError(f"Invalid model mode: {self.model_config.mode}")

    def _apply_hetero_regularization(self, loss):
        """Apply regularization for heterogeneous graph."""
        # Apply regularization for each edge type
        for edge_type, mask in self.edge_mask.items():
            if (mask is not None
                    and self.hard_edge_mask[edge_type] is not None):
                loss = self._add_mask_regularization(
                    loss, mask, self.hard_edge_mask[edge_type],
                    self.coeffs['edge_size'], self.coeffs['edge_reduction'],
                    self.coeffs['edge_ent'])

        # Apply regularization for each node type
        for node_type, mask in self.node_mask.items():
            if (mask is not None
                    and self.hard_node_mask[node_type] is not None):
                loss = self._add_mask_regularization(
                    loss, mask, self.hard_node_mask[node_type],
                    self.coeffs['node_feat_size'],
                    self.coeffs['node_feat_reduction'],
                    self.coeffs['node_feat_ent'])

        return loss

    def _apply_homo_regularization(self, loss):
        """Apply regularization for homogeneous graph."""
        # Apply regularization for edge mask
        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None
            loss = self._add_mask_regularization(loss, self.edge_mask,
                                                 self.hard_edge_mask,
                                                 self.coeffs['edge_size'],
                                                 self.coeffs['edge_reduction'],
                                                 self.coeffs['edge_ent'])

        # Apply regularization for node mask
        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            loss = self._add_mask_regularization(
                loss, self.node_mask, self.hard_node_mask,
                self.coeffs['node_feat_size'],
                self.coeffs['node_feat_reduction'],
                self.coeffs['node_feat_ent'])

        return loss

    def _add_mask_regularization(self, loss, mask, hard_mask, size_coeff,
                                 reduction_name, ent_coeff):
        """Add size and entropy regularization for a mask."""
        m = mask[hard_mask].sigmoid()
        reduce_fn = getattr(torch, reduction_name)
        # Add size regularization
        loss = loss + size_coeff * reduce_fn(m)
        # Add entropy regularization
        ent = -m * torch.log(m + self.coeffs['EPS']) - (
            1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss = loss + ent_coeff * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None


class GNNExplainer_:
    r"""Deprecated version for :class:`GNNExplainer`."""

    coeffs = GNNExplainer.default_coeffs

    conversion_node_mask_type = {
        'feature': 'common_attributes',
        'individual_feature': 'attributes',
        'scalar': 'object',
    }

    conversion_return_type = {
        'log_prob': 'log_probs',
        'prob': 'probs',
        'raw': 'raw',
        'regression': 'raw',
    }

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        return_type: str = 'log_prob',
        feat_mask_type: str = 'feature',
        allow_edge_mask: bool = True,
        **kwargs,
    ):
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']

        explainer_config = ExplainerConfig(
            explanation_type='model',
            node_mask_type=self.conversion_node_mask_type[feat_mask_type],
            edge_mask_type=MaskType.object if allow_edge_mask else None,
        )
        model_config = ModelConfig(
            mode='regression'
            if return_type == 'regression' else 'multiclass_classification',
            task_level=ModelTaskLevel.node,
            return_type=self.conversion_return_type[return_type],
        )

        self.model = model
        self._explainer = GNNExplainer(epochs=epochs, lr=lr, **kwargs)
        self._explainer.connect(explainer_config, model_config)

    @torch.no_grad()
    def get_initial_prediction(self, *args, **kwargs) -> Tensor:

        training = self.model.training
        self.model.eval()

        out = self.model(*args, **kwargs)
        if (self._explainer.model_config.mode ==
                ModelMode.multiclass_classification):
            out = out.argmax(dim=-1)

        self.model.train(training)

        return out

    def explain_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self._explainer.model_config.task_level = ModelTaskLevel.graph

        explanation = self._explainer(
            self.model,
            x,
            edge_index,
            target=self.get_initial_prediction(x, edge_index, **kwargs),
            **kwargs,
        )
        return self._convert_output(explanation, edge_index)

    def explain_node(
        self,
        node_idx: int,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        self._explainer.model_config.task_level = ModelTaskLevel.node
        explanation = self._explainer(
            self.model,
            x,
            edge_index,
            target=self.get_initial_prediction(x, edge_index, **kwargs),
            index=node_idx,
            **kwargs,
        )
        return self._convert_output(explanation, edge_index, index=node_idx,
                                    x=x)

    def _convert_output(self, explanation, edge_index, index=None, x=None):
        node_mask = explanation.get('node_mask')
        edge_mask = explanation.get('edge_mask')

        if node_mask is not None:
            node_mask_type = self._explainer.explainer_config.node_mask_type
            if node_mask_type in {MaskType.object, MaskType.common_attributes}:
                node_mask = node_mask.view(-1)

        if edge_mask is None:
            if index is not None:
                _, edge_mask = self._explainer._get_hard_masks(
                    self.model, index, edge_index, num_nodes=x.size(0))
                edge_mask = edge_mask.to(x.dtype)
            else:
                edge_mask = torch.ones(edge_index.size(1),
                                       device=edge_index.device)

        return node_mask, edge_mask
