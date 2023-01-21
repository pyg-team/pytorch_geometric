import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, sigmoid
from torch.nn import LayerNorm, Linear, Parameter, ReLU, Sequential, init
from tqdm import tqdm

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.config import (
    MaskType,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)
from torch_geometric.nn import MessagePassing


def explain_message(self, out, x_i, x_j):
    norm = Sequential(LayerNorm(out.size(-1)).to(out.device), ReLU())
    basis_messages = norm(out)

    if getattr(self, 'message_scale', None) is not None:
        basis_messages = basis_messages * self.message_scale.unsqueeze(-1)

        if self.message_replacement is not None:
            if basis_messages.shape == self.message_replacement.shape:
                basis_messages = (basis_messages +
                                  (1 - self.message_scale).unsqueeze(-1) *
                                  self.message_replacement)
            else:
                basis_messages = (basis_messages +
                                  ((1 - self.message_scale).unsqueeze(-1) *
                                   self.message_replacement.unsqueeze(0)))

    self.latest_messages = basis_messages
    self.latest_source_embeddings = x_j
    self.latest_target_embeddings = x_i

    return basis_messages


class GraphMaskExplainer(ExplainerAlgorithm):
    r"""The GraphMask-Explainer model from the `"Interpreting Graph Neural
    Networks for NLP With Differentiable Edge Masking"
    <https://arxiv.org/abs/2010.00577>`_ paper for identifying layer-wise
    compact subgraph structures and node features that play a crucial role in
    the predictions made by a GNN.

    .. note::

        For an example of using :class:`GraphMaskExplainer`,
        see `examples/contrib/graphmask_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        /contrib/graphmask_explainer.py>`_.

    Args:
        num_layers (int): The number of layers to use.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        penalty_scaling (int, optional): Scaling value of penalty term. Value
            must lie between 0 and 10. (default: :obj:`5`)
        lambda_optimizer_lr (float, optional): The learning rate to optimize
            the Lagrange multiplier. (default: :obj:`1e-2`)
        init_lambda (float, optional): The Lagrange multiplier. Value must lie
            between :obj:`0` and `1`. (default: :obj:`0.55`)
        allowance (float, optional): A float value between :obj:`0` and
            :obj:`1` denotes tolerance level. (default: :obj:`0.03`)
        layer_type (str, optional): The type of GNN layer being used in the GNN
            model. (default: :obj:`GCN`)
        log (bool, optional): If set to :obj:`False`, will not log any
            learning progress. (default: :obj:`True`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.nn.models.GraphMaskExplainer.coeffs`.
    """
    coeffs = {
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(
        self,
        num_layers: int,
        epochs: int = 100,
        lr: float = 0.01,
        penalty_scaling: int = 5,
        lambda_optimizer_lr: int = 1e-2,
        init_lambda: int = 0.55,
        allowance: int = 0.03,
        layer_type: str = 'GCN',
        allow_multiple_explanations: bool = False,
        log: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert layer_type in ['GCN', 'GAT', 'FastRGCN']
        assert 0 <= penalty_scaling <= 10
        assert 0 <= init_lambda <= 1
        assert 0 <= allowance <= 1

        self.num_layers = num_layers
        self.init_lambda = init_lambda
        self.lambda_optimizer_lr = lambda_optimizer_lr
        self.penalty_scaling = penalty_scaling
        self.allowance = allowance
        self.layer_type = layer_type
        self.allow_multiple_explanations = allow_multiple_explanations
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.coeffs.update(kwargs)

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

        hard_node_mask = None

        if self.model_config.task_level == ModelTaskLevel.node:
            hard_node_mask, hard_edge_mask = self._get_hard_masks(
                model, index, edge_index, num_nodes=x.size(0))
        self.train_explainer(model, x, edge_index, target=target, index=index,
                             **kwargs)
        node_mask = self._post_process_mask(self.node_feat_mask,
                                            hard_node_mask, apply_sigmoid=True)
        edge_mask = self.explain(model, index=index)

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def supports(self) -> bool:
        return True

    def hard_concrete(self, input_element, summarize_penalty=True, beta=1 / 3,
                      gamma=-0.2, zeta=1.2, loc_bias=2, min_val=0, max_val=1,
                      training=True) -> Union[Tensor, Tensor]:
        input_element = input_element + loc_bias

        if training:
            u = torch.empty_like(input_element).uniform_(1e-6, 1.0 - 1e-6)

            s = sigmoid(
                (torch.log(u) - torch.log(1 - u) + input_element) / beta)

            penalty = sigmoid(input_element -
                              beta * np.math.log(-gamma / zeta))
        else:
            s = sigmoid(input_element)
            penalty = torch.zeros_like(input_element)

        if summarize_penalty:
            penalty = penalty.mean()

        s = s * (zeta - gamma) + gamma

        clipped_s = s.clamp(min_val, max_val)

        clip_value = (torch.min(clipped_s) + torch.max(clipped_s)) / 2
        hard_concrete = (clipped_s > clip_value).float()
        clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        return clipped_s, penalty

    def set_masks(self, i_dim, j_dim, h_dim, x, device):
        if self.layer_type == 'GCN' or self.layer_type == 'GAT':
            i_dim = j_dim
        (num_nodes, num_feat), std = x.size(), 0.1
        self.feat_mask_type = self.explainer_config.node_mask_type

        if self.feat_mask_type == MaskType.attributes:
            self.node_feat_mask = torch.nn.Parameter(
                torch.randn(num_nodes, num_feat, device=device) * std)
        elif self.feat_mask_type == MaskType.object:
            self.node_feat_mask = torch.nn.Parameter(
                torch.randn(num_nodes, 1, device=device) * std)
        else:
            self.node_feat_mask = torch.nn.Parameter(
                torch.randn(1, num_feat, device=device) * std)

        baselines, self.gates, full_biases = [], torch.nn.ModuleList(), []

        for v_dim, m_dim, h_dim in zip(i_dim, j_dim, h_dim):
            self.transform, self.layer_norm = [], []
            input_dims = [v_dim, m_dim, v_dim]
            for _, input_dim in enumerate(input_dims):
                self.transform.append(
                    Linear(input_dim, h_dim, bias=False).to(device))
                self.layer_norm.append(LayerNorm(h_dim).to(device))

            self.transforms = torch.nn.ModuleList(self.transform)
            self.layer_norms = torch.nn.ModuleList(self.layer_norm)

            self.full_bias = Parameter(
                torch.tensor(h_dim, dtype=torch.float, device=device))
            full_biases.append(self.full_bias)

            self.reset_parameters(input_dims, h_dim)

            self.non_linear = ReLU()
            self.output_layer = Linear(h_dim, 1).to(device)

            gate = [
                self.transforms, self.layer_norms, self.non_linear,
                self.output_layer
            ]
            self.gates.extend(gate)

            baseline = torch.tensor(m_dim, dtype=torch.float, device=device)
            stdv = 1. / math.sqrt(m_dim)
            baseline.uniform_(-stdv, stdv)
            baseline = torch.nn.Parameter(baseline)
            baselines.append(baseline)

        full_biases = torch.nn.ParameterList(full_biases)
        self.full_biases = full_biases

        baselines = torch.nn.ParameterList(baselines)
        self.baselines = baselines

        for parameter in self.parameters():
            parameter.requires_grad = False

    def enable_layer(self, layer):
        for d in range(layer * 4, (layer * 4) + 4):
            for parameter in self.gates[d].parameters():
                parameter.requires_grad = True
        self.full_biases[layer].requires_grad = True
        self.baselines[layer].requires_grad = True

    def reset_parameters(self, input_dims, h_dim):
        fan_in = sum(input_dims)

        std = math.sqrt(2.0 / float(fan_in + h_dim))
        a = math.sqrt(3.0) * std

        for transform in self.transforms:
            init._no_grad_uniform_(transform.weight, -a, a)

        init.zeros_(self.full_bias)

        for layer_norm in self.layer_norms:
            layer_norm.reset_parameters()

    def _loss_regression(self, y_hat: Tensor, y: Tensor) -> Tensor:
        assert self.model_config.return_type == ModelReturnType.raw
        return F.mse_loss(y_hat, y)

    def _loss_binary_classification(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.return_type == ModelReturnType.raw:
            loss_fn = F.binary_cross_entropy_with_logits
        elif self.model_config.return_type == ModelReturnType.probs:
            loss_fn = F.binary_cross_entropy
        else:
            assert False

        return loss_fn(y_hat.view_as(y), y.float())

    def _loss_multiclass_classification(
        self,
        y_hat: Tensor,
        y: Tensor,
    ) -> Tensor:
        if self.model_config.return_type == ModelReturnType.raw:
            loss_fn = F.cross_entropy
        elif self.model_config.return_type == ModelReturnType.probs:
            loss_fn = F.nll_loss
            y_hat = y_hat.log()
        elif self.model_config.return_type == ModelReturnType.log_probs:
            loss_fn = F.nll_loss
        else:
            assert False

        return loss_fn(y_hat, y)

    def _loss(self, y_hat: Tensor, y: Tensor, penalty) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False

        g = torch.relu(loss - self.allowance).mean()
        f = penalty * self.penalty_scaling

        loss = f + F.softplus(self.lambda_op) * g

        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + self.coeffs['EPS']) - (
            1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def freeze_model(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def __set_flags__(self, model):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.explain_message = explain_message.__get__(
                    module, MessagePassing)
                module.explain = True

    def __inject_messages__(self, model: torch.nn.Module, message_scale,
                            message_replacement, set=False):
        i = 0
        for module in model.modules():
            if isinstance(module, MessagePassing):
                if not set:
                    module.message_scale = message_scale[i]
                    module.message_replacement = message_replacement[i]
                    i = i + 1
                else:
                    module.message_scale = None
                    module.message_replacement = None

    def train_explainer(self, model: torch.nn.Module, x: Tensor,
                        edge_index: Tensor, *, target: Tensor,
                        index: Optional[Union[int, Tensor]] = None, **kwargs):

        if not isinstance(index, Tensor) and not isinstance(index, int) \
                and index is not None:
            raise ValueError("'index' parameter can only be a 'Tensor', "
                             "'integer' or set to 'None' instead.")

        self.freeze_model(model)
        self.__set_flags__(model)

        input_dims, output_dims = [], []
        for module in model.modules():
            if isinstance(module, MessagePassing):
                input_dims.append(module.in_channels)
                output_dims.append(module.out_channels)

        self.set_masks(input_dims, output_dims, output_dims, x, x.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for layer in reversed(list(range(self.num_layers))):
            if self.log:
                pbar = tqdm(total=self.epochs)
                if self.model_config.task_level == ModelTaskLevel.node:
                    pbar.set_description(
                        f'Train explainer for node(s) {index} with layer '
                        f'{layer}')
                elif self.model_config.task_level == ModelTaskLevel.edge:
                    pbar.set_description(
                        f"Train explainer for 'edge' Task-level with layer "
                        f"{layer}")
                else:
                    pbar.set_description(
                        f'Train explainer for graph {index} with layer '
                        f'{layer}')
            self.enable_layer(layer)
            for epoch in range(self.epochs):
                with torch.no_grad():
                    model(x, edge_index, **kwargs)
                gates, total_penalty = [], 0
                latest_source_embeddings, latest_messages = [], []
                latest_target_embeddings = []
                for module in model.modules():
                    if isinstance(module, MessagePassing):
                        latest_source_embeddings.append(
                            module.latest_source_embeddings)
                        latest_messages.append(module.latest_messages)
                        latest_target_embeddings.append(
                            module.latest_target_embeddings)
                gate_input = [
                    latest_source_embeddings, latest_messages,
                    latest_target_embeddings
                ]
                for i in range(self.num_layers):
                    output = self.full_biases[i]
                    for j in range(len(gate_input)):
                        partial = self.gates[i * 4][j](gate_input[j][i])
                        result = self.gates[(i * 4) + 1][j](partial)
                        output = output + result
                    relu_output = self.gates[(i * 4) + 2](output /
                                                          len(gate_input))
                    sampling_weights = self.gates[(i * 4) +
                                                  3](relu_output).squeeze(
                                                      dim=-1)
                    sampling_weights, penalty = self.hard_concrete(
                        sampling_weights)
                    gates.append(sampling_weights)
                    total_penalty += penalty

                self.__inject_messages__(model, gates, self.baselines)

                self.lambda_op = torch.tensor(self.init_lambda,
                                              requires_grad=True)
                optimizer_lambda = torch.optim.RMSprop(
                    [self.lambda_op], lr=self.lambda_optimizer_lr,
                    centered=True)

                optimizer.zero_grad()
                optimizer_lambda.zero_grad()

                h = x * self.node_feat_mask.sigmoid()
                y_hat, y = model(x=h, edge_index=edge_index, **kwargs), target

                if self.model_config.task_level == ModelTaskLevel.node \
                        or self.model_config.task_level == ModelTaskLevel.edge:
                    if index is not None:
                        y_hat, y = y_hat[index], y[index]

                self.__inject_messages__(model, gates, self.baselines, True)

                loss = self._loss(y_hat, y, total_penalty)

                loss.backward()
                optimizer.step()
                self.lambda_op.grad *= -1
                optimizer_lambda.step()

                if self.lambda_op.item() < -2:
                    self.lambda_op.data = torch.full_like(
                        self.lambda_op.data, -2)
                elif self.lambda_op.item() > 30:
                    self.lambda_op.data = torch.full_like(
                        self.lambda_op.data, 30)

                if self.log:
                    pbar.update(1)

            if self.log:
                pbar.close()

    def explain(self, model: torch.nn.Module, *,
                index: Optional[Union[int, Tensor]] = None) -> Tensor:

        if not isinstance(index, Tensor) and not isinstance(index, int) \
                and index is not None:
            raise ValueError("'index' parameter can only be a 'Tensor', "
                             "'integer' or set to 'None' instead.")

        self.freeze_model(model)
        self.__set_flags__(model)

        with torch.no_grad():
            latest_source_embeddings, latest_messages = [], []
            latest_target_embeddings = []
            for module in model.modules():
                if isinstance(module, MessagePassing):
                    latest_source_embeddings.append(
                        module.latest_source_embeddings)
                    latest_messages.append(module.latest_messages)
                    latest_target_embeddings.append(
                        module.latest_target_embeddings)
            gate_input = [
                latest_source_embeddings, latest_messages,
                latest_target_embeddings
            ]
            if self.log:
                pbar = tqdm(total=self.num_layers)
            for i in range(self.num_layers):
                if self.log:
                    if self.model_config.task_level == ModelTaskLevel.node:
                        pbar.set_description(f'Explain node(s) {index}')
                    elif self.model_config.task_level == ModelTaskLevel.edge:
                        pbar.set_description("Explain 'edge' Task-level")
                    else:
                        pbar.set_description(f'Explain graph {index}')
                output = self.full_biases[i]
                for j in range(len(gate_input)):
                    partial = self.gates[i * 4][j](gate_input[j][i])
                    result = self.gates[(i * 4) + 1][j](partial)
                    output = output + result
                relu_output = self.gates[(i * 4) + 2](output / len(gate_input))
                sampling_weights = self.gates[(i * 4) +
                                              3](relu_output).squeeze(dim=-1)
                sampling_weights, _ = self.hard_concrete(
                    sampling_weights, training=False)
                if i == 0:
                    edge_weight = sampling_weights
                else:
                    if (edge_weight.size(-1) != sampling_weights.size(-1)
                            and self.layer_type == 'GAT'):
                        sampling_weights = F.pad(
                            input=sampling_weights,
                            pad=(0, edge_weight.size(-1) -
                                 sampling_weights.size(-1), 0, 0),
                            mode='constant', value=0)
                    edge_weight = torch.cat((edge_weight, sampling_weights), 0)
                if self.log:
                    pbar.update(1)
        if self.log:
            pbar.close()

        edge_mask = edge_weight.view(-1,
                                     edge_weight.size(0) // self.num_layers)
        edge_mask = torch.mean(edge_mask, 0)

        return edge_mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'
