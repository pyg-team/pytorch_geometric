import logging
from typing import Dict, Optional, Tuple, Union, overload

import torch
from torch import Tensor
from torch.nn import ReLU, Sequential

from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import (
    clear_masks,
    set_hetero_masks,
    set_masks,
)
from torch_geometric.explain.config import (
    ExplanationType,
    ModelMode,
    ModelTaskLevel,
)
from torch_geometric.nn import HANConv, HeteroConv, HGTConv, Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import get_embeddings, get_embeddings_hetero


class PGExplainer(ExplainerAlgorithm):
    r"""The PGExplainer model from the `"Parameterized Explainer for Graph
    Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper.

    Internally, it utilizes a neural network to identify subgraph structures
    that play a crucial role in the predictions made by a GNN.
    Importantly, the :class:`PGExplainer` needs to be trained via
    :meth:`~PGExplainer.train` before being able to generate explanations:

    .. code-block:: python

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.003),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=ModelConfig(...),
        )

        # Train against a variety of node-level or graph-level predictions:
        for epoch in range(30):
            for index in [...]:  # Indices to train against.
                loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                                 target=target, index=index)

        # Get the final explanations:
        explanation = explainer(x, edge_index, target=target, index=0)

    Args:
        epochs (int): The number of epochs to train.
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.003`).
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.PGExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [5.0, 2.0],
        'bias': 0.01,
    }

    # NOTE: Add more in the future as needed.
    SUPPORTED_HETERO_MODELS = [
        HGTConv,
        HANConv,
        HeteroConv,
    ]

    def __init__(self, epochs: int, lr: float = 0.003, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.mlp = Sequential(
            Linear(-1, 64),
            ReLU(),
            Linear(64, 1),
        )
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self._curr_epoch = -1
        self.is_hetero = False

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.mlp)

    @overload
    def train(
        self,
        epoch: int,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> float:
        ...

    @overload
    def train(
        self,
        epoch: int,
        model: torch.nn.Module,
        x: Dict[NodeType, Tensor],
        edge_index: Dict[EdgeType, Tensor],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> float:
        ...

    def train(
        self,
        epoch: int,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> float:
        r"""Trains the underlying explainer model.
        Needs to be called before being able to make predictions.

        Args:
            epoch (int): The current epoch of the training phase.
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor or Dict[str, torch.Tensor]): The input node
                features. Can be either homogeneous or heterogeneous.
            edge_index (torch.Tensor or Dict[Tuple[str, str, str]): The input
                edge indices. Can be either homogeneous or heterogeneous.
            target (torch.Tensor): The target of the model.
            index (int or torch.Tensor, optional): The index of the model
                output to explain. Needs to be a single index.
                (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """
        self.is_hetero = isinstance(x, dict)
        if self.is_hetero:
            assert isinstance(edge_index, dict)

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

        # Get embeddings based on whether the graph is homogeneous or
        # heterogeneous
        node_embeddings = self._get_embeddings(model, x, edge_index, **kwargs)

        # Train the model
        self.optimizer.zero_grad()
        temperature = self._get_temperature(epoch)

        # Process embeddings and generate edge masks
        edge_mask = self._generate_edge_masks(node_embeddings, edge_index,
                                              index, temperature)

        # Apply masks to the model
        if self.is_hetero:
            set_hetero_masks(model, edge_mask, edge_index, apply_sigmoid=True)

            # For node-level tasks, we can compute hard masks
            if self.model_config.task_level == ModelTaskLevel.node:
                # Process each edge type separately
                for edge_type, mask in edge_mask.items():
                    # Get the edge indices for this edge type
                    edges = edge_index[edge_type]
                    src_type, _, dst_type = edge_type

                    # Get hard masks for this specific edge type
                    _, hard_mask = self._get_hard_masks(
                        model, index, edges,
                        num_nodes=max(x[src_type].size(0),
                                      x[dst_type].size(0)))

                    edge_mask[edge_type] = mask[hard_mask]
        else:
            # Apply masks for homogeneous graphs
            set_masks(model, edge_mask, edge_index, apply_sigmoid=True)

            # For node-level tasks, we may need to apply hard masks
            hard_edge_mask = None
            if self.model_config.task_level == ModelTaskLevel.node:
                _, hard_edge_mask = self._get_hard_masks(
                    model, index, edge_index, num_nodes=x.size(0))
                edge_mask = edge_mask[hard_edge_mask]

        # Forward pass with masks applied
        y_hat, y = model(x, edge_index, **kwargs), target

        if index is not None:
            y_hat, y = y_hat[index], y[index]

        # Calculate loss
        loss = self._loss(y_hat, y, edge_mask)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        # Clean up
        clear_masks(model)
        self._curr_epoch = epoch

        return float(loss)

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

        if self._curr_epoch < self.epochs - 1:  # Safety check:
            raise ValueError(f"'{self.__class__.__name__}' is not yet fully "
                             f"trained (got {self._curr_epoch + 1} epochs "
                             f"from {self.epochs} epochs). Please first train "
                             f"the underlying explainer model by running "
                             f"`explainer.algorithm.train(...)`.")

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

        # Get embeddings
        node_embeddings = self._get_embeddings(model, x, edge_index, **kwargs)

        # Generate explanations
        if self.is_hetero:
            # Generate edge masks for each edge type
            edge_masks = {}

            # Generate masks for each edge type
            for edge_type, edge_idx in edge_index.items():
                src_node_type, _, dst_node_type = edge_type

                assert src_node_type in node_embeddings
                assert dst_node_type in node_embeddings

                inputs = self._get_inputs_hetero(node_embeddings, edge_type,
                                                 edge_idx, index)
                logits = self.mlp(inputs).view(-1)

                # For node-level explanations, get hard masks for this
                # specific edge type
                hard_edge_mask = None
                if self.model_config.task_level == ModelTaskLevel.node:
                    _, hard_edge_mask = self._get_hard_masks(
                        model, index, edge_idx,
                        num_nodes=max(x[src_node_type].size(0),
                                      x[dst_node_type].size(0)))

                # Apply hard mask if available and it has any True values
                edge_masks[edge_type] = self._post_process_mask(
                    logits, hard_edge_mask, apply_sigmoid=True)

            explanation = HeteroExplanation()
            explanation.set_value_dict('edge_mask', edge_masks)
            return explanation
        else:
            hard_edge_mask = None
            if self.model_config.task_level == ModelTaskLevel.node:
                # We need to compute hard masks to properly clean up edges
                _, hard_edge_mask = self._get_hard_masks(
                    model, index, edge_index, num_nodes=x.size(0))

            inputs = self._get_inputs(node_embeddings, edge_index, index)
            logits = self.mlp(inputs).view(-1)

            edge_mask = self._post_process_mask(logits, hard_edge_mask,
                                                apply_sigmoid=True)

            return Explanation(edge_mask=edge_mask)

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.phenomenon:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"phenomenon explanations "
                          f"got (`explanation_type={explanation_type.value}`)")
            return False

        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node-level or graph-level explanations "
                          f"got (`task_level={task_level.value}`)")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False

        return True

    ###########################################################################

    def _get_embeddings(self, model: torch.nn.Module, x: Union[Tensor,
                                                               Dict[NodeType,
                                                                    Tensor]],
                        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
                        **kwargs) -> Union[Tensor, Dict[NodeType, Tensor]]:
        """Get embeddings from the model based on input type."""
        if self.is_hetero:
            # For heterogeneous graphs, get embeddings for each node type
            embeddings_dict = get_embeddings_hetero(
                model,
                self.SUPPORTED_HETERO_MODELS,
                x,
                edge_index,
                **kwargs,
            )

            # Use the last layer's embeddings for each node type
            last_embedding_dict = {
                node_type: embs[-1] if embs and len(embs) > 0 else None
                for node_type, embs in embeddings_dict.items()
            }

            # Skip if no embeddings were captured
            if not any(emb is not None
                       for emb in last_embedding_dict.values()):
                raise ValueError(
                    "No embeddings were captured from the model. "
                    "Please check if the model architecture is supported.")

            return last_embedding_dict
        else:
            # For homogeneous graphs, get embeddings directly
            return get_embeddings(model, x, edge_index, **kwargs)[-1]

    def _generate_edge_masks(
            self, emb: Union[Tensor, Dict[NodeType, Tensor]],
            edge_index: Union[Tensor,
                              Dict[EdgeType,
                                   Tensor]], index: Optional[Union[int,
                                                                   Tensor]],
            temperature: float) -> Union[Tensor, Dict[EdgeType, Tensor]]:
        """Generate edge masks based on embeddings."""
        if self.is_hetero:
            # For heterogeneous graphs, generate masks for each edge type
            edge_masks = {}

            for edge_type, edge_idx in edge_index.items():
                src, _, dst = edge_type

                assert src in emb and dst in emb
                # Generate inputs for this edge type
                inputs = self._get_inputs_hetero(emb, edge_type, edge_idx,
                                                 index)
                logits = self.mlp(inputs).view(-1)
                edge_masks[edge_type] = self._concrete_sample(
                    logits, temperature)

            # Ensure we have at least one valid edge mask
            if not edge_masks:
                raise ValueError(
                    "Could not generate edge masks for any edge type. "
                    "Please ensure the model architecture is supported.")

            return edge_masks
        else:
            # For homogeneous graphs, generate a single mask
            inputs = self._get_inputs(emb, edge_index, index)
            logits = self.mlp(inputs).view(-1)
            return self._concrete_sample(logits, temperature)

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor,
                    index: Optional[int] = None) -> Tensor:
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            zs.append(embedding[index].view(1, -1).repeat(zs[0].size(0), 1))
        return torch.cat(zs, dim=-1)

    def _get_inputs_hetero(self, embedding_dict: Dict[NodeType, Tensor],
                           edge_type: Tuple[str, str, str], edge_index: Tensor,
                           index: Optional[int] = None) -> Tensor:
        src, _, dst = edge_type

        # Get embeddings for source and destination nodes
        src_emb = embedding_dict[src]
        dst_emb = embedding_dict[dst]

        # Source and destination node embeddings
        zs = [src_emb[edge_index[0]], dst_emb[edge_index[1]]]

        # For node-level explanations, add the target node embedding
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            # Assuming index refers to a node of type 'src'
            target_emb = src_emb[index].view(1, -1).repeat(zs[0].size(0), 1)
            zs.append(target_emb)

        return torch.cat(zs, dim=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor,
                         temperature: float = 1.0) -> Tensor:
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss(self, y_hat: Tensor, y: Tensor,
              edge_mask: Union[Tensor, Dict[EdgeType, Tensor]]) -> Tensor:
        # Calculate base loss based on model configuration
        loss = self._calculate_base_loss(y_hat, y)

        # Apply regularization based on graph type
        if self.is_hetero:
            loss = self._apply_hetero_regularization(loss, edge_mask)
        else:
            loss = self._apply_homo_regularization(loss, edge_mask)

        return loss

    def _calculate_base_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """Calculate base loss based on model configuration."""
        if self.model_config.mode == ModelMode.binary_classification:
            return self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            return self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            return self._loss_regression(y_hat, y)
        else:
            raise ValueError(
                f"Unsupported model mode: {self.model_config.mode}")

    def _apply_hetero_regularization(
            self, loss: Tensor, edge_mask: Dict[EdgeType, Tensor]) -> Tensor:
        """Apply regularization for heterogeneous graph."""
        for _, mask in edge_mask.items():
            loss = self._add_mask_regularization(loss, mask)

        return loss

    def _apply_homo_regularization(self, loss: Tensor,
                                   edge_mask: Tensor) -> Tensor:
        """Apply regularization for homogeneous graph."""
        return self._add_mask_regularization(loss, edge_mask)

    def _add_mask_regularization(self, loss: Tensor, mask: Tensor) -> Tensor:
        """Add size and entropy regularization for a mask."""
        # Apply sigmoid for mask values
        mask = mask.sigmoid()

        # Size regularization
        size_loss = mask.sum() * self.coeffs['edge_size']

        # Entropy regularization
        masked = 0.99 * mask + 0.005
        mask_ent = -masked * masked.log() - (1 - masked) * (1 - masked).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']

        return loss + size_loss + mask_ent_loss
