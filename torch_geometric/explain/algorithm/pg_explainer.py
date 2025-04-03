import logging
from typing import Dict, Optional, Union

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
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import get_embeddings
from torch_geometric.utils.embedding import get_hetero_embeddings


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
        self.edge_mask_dict = None
        self.hard_edge_mask_dict = None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.mlp)

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
    ):
        r"""Trains the underlying explainer model.
        Needs to be called before being able to make predictions.

        Args:
            epoch (int): The current epoch of the training phase.
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor or dict): The input node features of a
                homogeneous or heterogeneous graph.
            edge_index (torch.Tensor or dict): The input edge indices of a
                homogeneous or heterogeneous graph.
            target (torch.Tensor): The target of the model.
            index (int or torch.Tensor, optional): The index of the model
                output to explain. Needs to be a single index.
                (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """
        if isinstance(x, dict):
            assert isinstance(edge_index, dict)
            self.is_hetero = True

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

        # Get embeddings
        if self.is_hetero:
            # For heterogeneous graphs, use get_hetero_embeddings to get embeddings by node type
            # import pdb; pdb.set_trace()
            z_dict = get_hetero_embeddings(model, x, edge_index, **kwargs)
            # Use the last layer embeddings for each node type
            # import pdb; pdb.set_trace()
            z_last_dict = {
                node_type: embs[-1]
                for node_type, embs in z_dict.items() if embs
            }
        else:
            # For homogeneous graphs, use the original get_embeddings
            z = get_embeddings(model, x, edge_index, **kwargs)[-1]

        self.optimizer.zero_grad()
        temperature = self._get_temperature(epoch)

        if self.is_hetero:
            # Process each edge type separately
            self.edge_mask_dict = {}

            # Apply masks for each edge type
            for edge_type, indices in edge_index.items():
                # Generate inputs for each edge type using node embeddings
                src_type, _, dst_type = edge_type

                # Get node embeddings for this edge type
                src_emb = z_last_dict[src_type]
                dst_emb = z_last_dict[dst_type]

                zs = [src_emb[indices[0]], dst_emb[indices[1]]]

                # If this is a node-level explanation, include target node embedding
                if self.model_config.task_level == ModelTaskLevel.node:
                    # For node-level tasks in heterogeneous graphs, we need to know which node type
                    # the index refers to. There are several ways to determine this:
                    if not hasattr(self, 'target_node_type'):
                        # First, check if any destination node type has the target index
                        potential_types = []
                        for node_type, features in x.items():
                            if index < features.size(0):
                                potential_types.append(node_type)

                        # If we found possible types, use the first one
                        if potential_types:
                            self.target_node_type = potential_types[0]
                        # Otherwise, use the first node type as fallback
                        else:
                            self.target_node_type = list(x.keys())[0]

                    # Get embedding for the target node (with safety check)
                    target_emb_tensor = z_last_dict[self.target_node_type]
                    safe_index = min(index, target_emb_tensor.size(0) - 1)
                    target_emb = target_emb_tensor[safe_index]
                    zs.append(
                        target_emb.view(1, -1).expand(indices.size(1), -1))

                # Concatenate the embeddings
                inputs = torch.cat(zs, dim=-1)

                # Get edge mask logits
                logits = self.mlp(inputs).view(-1)
                edge_mask = self._concrete_sample(logits, temperature)

                # Set masks for current edge type
                self.edge_mask_dict[edge_type] = edge_mask

            # Apply all masks at once using set_hetero_masks
            set_hetero_masks(model, self.edge_mask_dict, edge_index,
                             apply_sigmoid=True)

            # Forward pass with all masks applied
            y_hat = model(x, edge_index, **kwargs)
        else:
            # Process homogeneous graph
            inputs = self._get_inputs(z, edge_index, index)
            logits = self.mlp(inputs).view(-1)
            edge_mask = self._concrete_sample(logits, temperature)
            set_masks(model, edge_mask, edge_index, apply_sigmoid=True)

            if self.model_config.task_level == ModelTaskLevel.node:
                _, hard_edge_mask = self._get_hard_masks(
                    model, index, edge_index, num_nodes=x.size(0))
                edge_mask = edge_mask[hard_edge_mask]

            y_hat = model(x, edge_index, **kwargs)

        # Get loss
        y = target
        if index is not None:
            y_hat, y = y_hat[index], y[index]

        # Calculate loss
        if self.is_hetero:
            # Calculate combined loss for all edge types
            loss = self._loss_hetero(y_hat, y)
        else:
            # Calculate loss for homogeneous graph
            loss = self._loss(y_hat, y, edge_mask)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Clear masks
        clear_masks(model)
        self._curr_epoch = epoch

        return float(loss)

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
        self.is_hetero = isinstance(x, dict) and isinstance(edge_index, dict)

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
        if self.is_hetero:
            # For heterogeneous graphs, use get_hetero_embeddings to get embeddings by node type
            z_dict = get_hetero_embeddings(model, x, edge_index, **kwargs)
            # Use the last layer embeddings for each node type
            z_last_dict = {
                node_type: embs[-1]
                for node_type, embs in z_dict.items() if embs
            }
        else:
            # For homogeneous graphs, use the original get_embeddings
            z = get_embeddings(model, x, edge_index, **kwargs)[-1]

        if self.is_hetero:
            # Process each edge type separately
            edge_mask_dict = {}

            # Process each edge type
            for edge_type, indices in edge_index.items():
                # Generate inputs for each edge type using node embeddings
                src_type, _, dst_type = edge_type

                # Get node embeddings for this edge type
                src_emb = z_last_dict[src_type]
                dst_emb = z_last_dict[dst_type]

                # Create model inputs: concatenate source and destination node embeddings
                # Add safety check for index bounds
                safe_src_indices = torch.clamp(indices[0], 0,
                                               src_emb.size(0) - 1)
                safe_dst_indices = torch.clamp(indices[1], 0,
                                               dst_emb.size(0) - 1)

                zs = [src_emb[safe_src_indices], dst_emb[safe_dst_indices]]

                # If this is a node-level explanation, include target node embedding
                if self.model_config.task_level == ModelTaskLevel.node:
                    # For node-level tasks in heterogeneous graphs, we need to know which node type
                    # the index refers to. There are several ways to determine this:
                    if not hasattr(self, 'target_node_type'):
                        # First, check if any destination node type has the target index
                        potential_types = []
                        for node_type, features in x.items():
                            if index < features.size(0):
                                potential_types.append(node_type)

                        # If we found possible types, use the first one
                        if potential_types:
                            self.target_node_type = potential_types[0]
                        # Otherwise, use the first node type as fallback
                        else:
                            self.target_node_type = list(x.keys())[0]

                    # Get embedding for the target node (with safety check)
                    target_emb_tensor = z_last_dict[self.target_node_type]
                    safe_index = min(index, target_emb_tensor.size(0) - 1)
                    target_emb = target_emb_tensor[safe_index]
                    zs.append(
                        target_emb.view(1, -1).expand(indices.size(1), -1))

                # Concatenate the embeddings
                inputs = torch.cat(zs, dim=-1)

                # Get edge mask logits and process them
                logits = self.mlp(inputs).view(-1)

                # Apply sigmoid to get final mask
                edge_mask = torch.sigmoid(logits)
                edge_mask_dict[edge_type] = edge_mask

            # Create heterogeneous explanation
            explanation = HeteroExplanation()
            explanation.set_value_dict('edge_mask', edge_mask_dict)
            return explanation

        else:
            # Homogeneous graph processing
            hard_edge_mask = None
            if self.model_config.task_level == ModelTaskLevel.node:
                # We need to compute hard masks to properly clean up edges and
                # nodes attributions not involved during message passing:
                _, hard_edge_mask = self._get_hard_masks(
                    model, index, edge_index, num_nodes=x.size(0))

            inputs = self._get_inputs(z, edge_index, index)
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

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor,
                    index: Optional[int] = None) -> Tensor:
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            zs.append(embedding[index].view(1, -1).repeat(zs[0].size(0), 1))
        return torch.cat(zs, dim=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor,
                         temperature: float = 1.0) -> Tensor:
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss_hetero(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """Calculate loss for heterogeneous graphs combining all edge types."""
        # Calculate base loss based on task
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)

        # Add regularization for each edge type
        for edge_type, edge_mask in self.edge_mask_dict.items():
            # Size regularization
            mask = edge_mask.sigmoid()
            size_loss = mask.sum() * self.coeffs['edge_size']

            # Entropy regularization
            mask = 0.99 * mask + 0.005
            mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
            mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']

            loss = loss + size_loss + mask_ent_loss

        return loss

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)

        # Regularization loss:
        mask = edge_mask.sigmoid()
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']

        return loss + size_loss + mask_ent_loss
