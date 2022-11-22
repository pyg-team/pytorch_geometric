import math
from inspect import signature
from math import sqrt
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, sigmoid
from torch.nn import LayerNorm, Linear, Parameter, ReLU, Sequential, init
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import Explainer
from torch_geometric.utils import k_hop_subgraph, to_networkx

EPS = 1e-15


def explain_message(self, out, x_i, x_j):
    basis_messages = Sequential(LayerNorm(out.size(-1)), ReLU())(out)

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


class GraphMaskExplainer(Explainer):
    r"""The GraphMask-Explainer model from the `"Interpreting Graph Neural
    Networks for NLP With Differentiable Edge Masking"
    <https://arxiv.org/abs/2010.00577>`_ paper for identifying layer-wise
    compact subgraph structures and small subsets node features that play
    a crucial role in a GNN’s node-level and graph-level predictions.

    .. note::

        For an example of using GraphMask-Explainer,
        see `examples/graphmask_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        graphmask_explainer.py>`_.

    Args:
        num_layers (int): The number of layers to use.
        model_to_explain (torch.nn.Module): The GNN module to explain.
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
        penalty_scaling (int, optional): Scaling value of penalty term. Value
            must lie between 0 and 10. (default: :obj:`5`)
        lambda_optimizer_lr (float): The learning rate to optimize the
            Lagrange multiplier. (default: :obj:`1e-2`)
        init_lambda (float): The Lagrange multiplier. Value must lie between
            0 and 1. (default: :obj:`0.55`)
        allowance (float): A float value between 0 and 1 denotes tolerance
            level. (default: :obj:`0.03`)
        feat_mask_type (str, optional): Denotes the type of feature mask
            that will be learned. Valid inputs are :obj:`"feature"` (a single
            feature-level mask for all nodes), :obj:`"individual_feature"`
            (individual feature-level masks for each node), and
            :obj:`"scalar"` (scalar mask for each node).
            (default: :obj:`"feature"`)
        layer_type (str): The type of GNN layer being used in the GNN model.
            (default: :obj:`GCN`)
        allow_multiple_explanations: Switch to allow explainer to explain
            node-level predictions for two or more nodes. Must set to
            :obj:`False` while explaining graph-level predictions and
            if only one node-level prediction is to be explained.
            (default: :obj:`False`)
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
    }

    def __init__(self, num_layers, model_to_explain=None, epochs: int = 100,
                 lr: float = 0.01, num_hops: Optional[int] = 3,
                 return_type: str = 'log_prob', penalty_scaling: int = 5,
                 lambda_optimizer_lr: int = 1e-2, init_lambda: int = 0.55,
                 allowance: int = 0.03, feat_mask_type: str = 'scalar',
                 layer_type: str = 'GCN',
                 allow_multiple_explanations: bool = False, log: bool = True,
                 **kwargs):
        super().__init__(model_to_explain, lr, epochs, num_hops, return_type,
                         log)
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']
        assert layer_type in ['GCN', 'GAT', 'FastRGCN']
        assert 0 <= penalty_scaling <= 10
        assert 0 <= init_lambda <= 1
        assert 0 <= allowance <= 1

        self.feat_mask_type = feat_mask_type
        self.num_layers = num_layers
        self.init_lambda = init_lambda
        self.lambda_optimizer_lr = lambda_optimizer_lr
        self.penalty_scaling = penalty_scaling
        self.allowance = allowance
        self.layer_type = layer_type
        self.allow_multiple_explanations = allow_multiple_explanations
        self.coeffs.update(kwargs)

    def hard_concrete(self, input_element, summarize_penalty=True, beta=1 / 3,
                      gamma=-0.2, zeta=1.2, loc_bias=2, min_val=0, max_val=1,
                      training=True):
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

    def set_masks(self, i_dim, j_dim, h_dim, x):
        if self.layer_type == 'GCN' or self.layer_type == 'GAT':
            i_dim = j_dim
        (num_nodes, num_feat) = x.size()

        if self.feat_mask_type == 'individual_feature':
            self.node_feat_mask = torch.nn.Parameter(
                torch.randn(num_nodes, num_feat) * 0.1)
        elif self.feat_mask_type == 'scalar':
            self.node_feat_mask = torch.nn.Parameter(
                torch.randn(num_nodes, 1) * 0.1)
        else:
            self.node_feat_mask = torch.nn.Parameter(
                torch.randn(1, num_feat) * 0.1)

        baselines, self.gates, full_biases = [], torch.nn.ModuleList(), []

        for v_dim, m_dim, h_dim in zip(i_dim, j_dim, h_dim):
            self.transform, self.layer_norm = [], []
            input_dims = [v_dim, m_dim, v_dim]
            for _, input_dim in enumerate(input_dims):
                self.transform.append(Linear(input_dim, h_dim, bias=False))
                self.layer_norm.append(LayerNorm(h_dim))

            self.transforms = torch.nn.ModuleList(self.transform)
            self.layer_norms = torch.nn.ModuleList(self.layer_norm)

            self.full_bias = Parameter(torch.Tensor(h_dim))
            full_biases.append(self.full_bias)

            self.reset_parameters(input_dims, h_dim)

            self.non_linear = ReLU()
            self.output_layer = Linear(h_dim, 1)

            gate = [
                self.transforms, self.layer_norms, self.non_linear,
                self.output_layer
            ]
            self.gates.extend(gate)

            baseline = torch.FloatTensor(m_dim)
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

    def _loss(self, node_idx, log_logits, pred_label, penalty):
        if self.return_type == 'regression':
            if -1 not in node_idx:
                loss = torch.cdist(log_logits[node_idx], pred_label[node_idx])
            else:
                loss = torch.cdist(log_logits, pred_label)
        else:
            if -1 not in node_idx:
                loss = -log_logits[node_idx, pred_label[node_idx]]
            else:
                loss = -log_logits[0, pred_label[0]]
        g = torch.relu(loss - self.allowance).mean()
        f = penalty * self.penalty_scaling

        loss = f + F.softplus(self.lambda_op) * g

        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
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

    def __inject_messages__(self, message_scale, message_replacement,
                            set=False):
        i = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                if not set:
                    module.message_scale = message_scale[i]
                    module.message_replacement = message_replacement[i]
                    i = i + 1
                else:
                    module.message_scale = None
                    module.message_replacement = None

    def train_node_explainer(self, node_idx, x, edge_index, **kwargs):
        r"""Learns a node feature mask and an edge mask and returns only the
        learned node feature mask that plays a crucial role to explain the
        prediction made by the GNN for node(s) :attr:`node_idx`.

        Args:
            node_idx (list, int): Node(s) to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`)
        """

        if not isinstance(node_idx, list) and not isinstance(node_idx, int):
            raise ValueError("'node_idx' parameter can either be a 'list' or "
                             "an 'integer'.")
        if not self.allow_multiple_explanations and isinstance(node_idx, list):
            raise ValueError("'node_idx' parameter must be of type 'integer' "
                             "while not allowing multiple explanations.")
        if self.allow_multiple_explanations and isinstance(node_idx, int):
            raise ValueError("'node_idx' parameter must be of type 'list' "
                             "while allowing multiple explanations.")
        if self.allow_multiple_explanations and len(node_idx) < 2:
            raise ValueError(
                "More than 1 node index must be passed to 'node_idx' list "
                "while allowing multiple explanations.")

        num_nodes = x.size(0)
        self.freeze_model(self.model)
        self.__set_flags__(self.model)

        x, edge_index, mapping, hard_edge_mask, subset, kwargs = \
            self.subgraph(node_idx, x, edge_index, **kwargs)
        if hard_edge_mask[hard_edge_mask].size(0) == 0:
            raise ValueError('no edge has been preserved.')

        input_dims, output_dims = [], []
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                input_dims.append(module.in_channels)
                output_dims.append(module.out_channels)

        self.set_masks(input_dims, output_dims, output_dims, x)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for layer in reversed(list(range(self.num_layers))):
            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.epochs)
                if self.allow_multiple_explanations:
                    pbar.set_description(
                        f'Train explainer for nodes {node_idx} with layer '
                        f'{layer}')
                else:
                    pbar.set_description(
                        f'Train explainer for node {node_idx} with layer '
                        f'{layer}')
            self.enable_layer(layer)
            for epoch in range(self.epochs):
                self.model.eval()
                prediction = self.get_initial_prediction(
                    x, edge_index, **kwargs)
                self.model.train()
                gates, total_penalty = [], 0
                latest_source_embeddings, latest_messages = [], []
                latest_target_embeddings = []
                for module in self.model.modules():
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

                self.__inject_messages__(gates, self.baselines)

                self.lambda_op = torch.tensor(self.init_lambda,
                                              requires_grad=True)
                optimizer_lambda = torch.optim.RMSprop(
                    [self.lambda_op], lr=self.lambda_optimizer_lr,
                    centered=True)

                optimizer.zero_grad()
                optimizer_lambda.zero_grad()

                h = x * self.node_feat_mask.sigmoid()
                out = self.model(x=h, edge_index=edge_index, **kwargs)

                self.__inject_messages__(gates, self.baselines, True)

                if self.return_type == 'regression':
                    loss = self._loss(mapping, out, prediction, total_penalty)
                else:
                    log_logits = self._to_log_prob(out)
                    loss = self._loss(mapping, log_logits, prediction,
                                      total_penalty)
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

                if self.log:  # pragma: no cover
                    pbar.update(1)

            if self.log:  # pragma: no cover
                pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        if self.feat_mask_type == 'individual_feature':
            new_mask = x.new_zeros(num_nodes, x.size(-1))
            new_mask[subset] = node_feat_mask
            node_feat_mask = new_mask
        elif self.feat_mask_type == 'scalar':
            new_mask = x.new_zeros(num_nodes, 1)
            new_mask[subset] = node_feat_mask
            node_feat_mask = new_mask
        node_feat_mask = node_feat_mask.squeeze()
        return node_feat_mask

    def explain_node(self, node_idx, x, edge_index):
        r"""Returns only the learned edge mask that plays a crucial role to explain the
        prediction made by the GNN for node(s) :attr:`node_idx`.

        Args:
            node_idx (list, int): Node(s) to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`)
        """

        if not isinstance(node_idx, list) and not isinstance(node_idx, int):
            raise ValueError("'node_idx' parameter can either be a 'list' or "
                             "an 'integer'.")
        if not self.allow_multiple_explanations and isinstance(node_idx, list):
            raise ValueError("'node_idx' parameter must be of type 'integer' "
                             "while not allowing multiple explanations.")
        if self.allow_multiple_explanations and isinstance(node_idx, int):
            raise ValueError("'node_idx' parameter must be of type 'list' "
                             "while allowing multiple explanations.")
        if self.allow_multiple_explanations and len(node_idx) < 2:
            raise ValueError(
                "More than 1 node index must be passed to 'node_idx' list "
                "while allowing multiple explanations.")

        self.freeze_model(self.model)
        self.__set_flags__(self.model)

        self.model.eval()
        _, _, _, hard_edge_mask, _, _ = self.subgraph(node_idx, x, edge_index)
        if hard_edge_mask[hard_edge_mask].size(0) == 0:
            raise ValueError('no edge has been preserved.')

        with torch.no_grad():
            latest_source_embeddings, latest_messages = [], []
            latest_target_embeddings = []
            for module in self.model.modules():
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
            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.num_layers)
            for i in range(self.num_layers):
                if self.log:  # pragma: no cover
                    if self.allow_multiple_explanations:
                        pbar.set_description(f'Explain nodes {node_idx}')
                    else:
                        pbar.set_description(f'Explain node {node_idx}')
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
                if self.log:  # pragma: no cover
                    pbar.update(1)
        if self.log:  # pragma: no cover
            pbar.close()

        edge_mask = edge_weight.view(-1,
                                     edge_weight.size(0) // self.num_layers)
        edge_mask = torch.mean(edge_mask, 0)

        return edge_mask

    def train_graph_explainer(self, graph_idx, x, edge_index, **kwargs):
        r"""Learns a node feature mask and an edge mask and returns only the
        learned node feature mask that plays a crucial role to explain the
        prediction made by the GNN for graph :attr:`graph_idx`.

        Args:
            graph_idx (int): Graph to explain.
            x (Tensor): The node feature matrix for graph :attr:`graph_idx`.
            edge_index (LongTensor): The edge indices for graph
                :attr:`graph_idx`.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`)
        """

        if not isinstance(graph_idx, int):
            raise ValueError(
                "'graph_idx' parameter must be of type 'integer'.")

        self.freeze_model(self.model)
        self.__set_flags__(self.model)
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)

        input_dims, output_dims = [], []
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                input_dims.append(module.in_channels)
                output_dims.append(module.out_channels)

        self.set_masks(input_dims, output_dims, output_dims, x)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for layer in reversed(list(range(self.num_layers))):
            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.epochs)
                pbar.set_description(
                    f'Train explainer for graph {graph_idx} with layer '
                    f'{layer}')
            self.enable_layer(layer)
            for epoch in range(self.epochs):
                self.model.eval()
                prediction = self.get_initial_prediction(
                    x, edge_index, batch=batch, **kwargs)
                self.model.train()
                gates, total_penalty = [], 0
                latest_source_embeddings, latest_messages = [], []
                latest_target_embeddings = []
                for module in self.model.modules():
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

                self.__inject_messages__(gates, self.baselines)

                self.lambda_op = torch.tensor(self.init_lambda,
                                              requires_grad=True)
                optimizer_lambda = torch.optim.RMSprop(
                    [self.lambda_op], lr=self.lambda_optimizer_lr,
                    centered=True)

                optimizer.zero_grad()
                optimizer_lambda.zero_grad()

                h = x * self.node_feat_mask.sigmoid()
                out = self.model(x=h, edge_index=edge_index, batch=batch,
                                 **kwargs)

                self.__inject_messages__(gates, self.baselines, True)

                if self.return_type == 'regression':
                    loss = self._loss([-1], out, prediction, total_penalty)
                else:
                    log_logits = self._to_log_prob(out)
                    loss = self._loss([-1], log_logits, prediction,
                                      total_penalty)
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

                if self.log:  # pragma: no cover
                    pbar.update(1)

            if self.log:  # pragma: no cover
                pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid().squeeze()

        return node_feat_mask

    def explain_graph(self, graph_idx):
        r"""Returns only the learned edge mask that plays a crucial role to explain the
        prediction made by the GNN for graph :attr:`graph_idx`.

        Args:
            graph_idx (int): Graph to explain.

        :rtype: (:class:`Tensor`)
        """

        if not isinstance(graph_idx, int):
            raise ValueError(
                "'graph_idx' parameter must be of type 'integer'.")

        self.freeze_model(self.model)
        self.__set_flags__(self.model)
        self.model.eval()

        with torch.no_grad():
            latest_source_embeddings, latest_messages = [], []
            latest_target_embeddings = []
            for module in self.model.modules():
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
            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.num_layers)
                pbar.set_description(f'Explain graph {graph_idx}')
            for i in range(self.num_layers):
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
                if self.log:  # pragma: no cover
                    pbar.update(1)
        if self.log:  # pragma: no cover
            pbar.close()

        edge_mask = edge_weight.view(-1,
                                     edge_weight.size(0) // self.num_layers)
        edge_mask = torch.mean(edge_mask, 0)

        return edge_mask

    def visualize_subgraph(self, node_idx: Optional[int], edge_index: Tensor,
                           edge_mask: Tensor, y: Optional[Tensor] = None,
                           edge_y: Optional[Tensor] = None,
                           node_alpha: Optional[Tensor] = None, seed: int = 10,
                           **kwargs):
        r"""Visualizes the subgraph given an edge mask :attr:`edge_mask`.

        Args:
            node_idx (list, int): Node id(s) whose explanation is to be
                displayed. It may consist of either one node id or
                more (to collectively generate explanation of multiple nodes).
                Must be set to :obj:`None` to explain a graph.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. All nodes will have the same color
                if :attr:`node_idx` is set to :obj:`None`.
                (default: :obj:`None`)
            edge_y (Tensor, optional): The edge labels used as edge colorings.
                (default: :obj:`None`)
            node_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating
                transparency of each node. (default: :obj:`None`)
            seed (int, optional): Random seed of the :obj:`networkx` node
                placement algorithm. (default: :obj:`10`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """

        import matplotlib.pyplot as plt
        import networkx as nx

        if (node_alpha is not None
                and (self.feat_mask_type == 'feature'
                     or self.feat_mask_type == 'individual_feature')):
            raise ValueError("'visualize_subgraph' functionality does not "
                             "support feature mask types 'feature' and "
                             "'individual_feature'. Please either set "
                             "'node_alpha' to None or set 'feat_mask_type' "
                             "to 'scalar'.")

        if node_idx is None:
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1))
            subset = torch.arange(edge_index.max().item() + 1,
                                  device=edge_index.device)
            y = None
        else:
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self._flow())

        threshold = (torch.min(edge_mask) + torch.max(edge_mask)) / 2
        edge_mask = (edge_mask >= threshold).to(torch.float)
        print(edge_mask)
        if 0 in edge_mask:
            print('0 available')

        if edge_mask.size(0) == 1:
            edge_mask = edge_mask.numpy()

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        if edge_y is None:
            edge_color = ['black'] * edge_index.size(1)
        else:
            colors = list(plt.rcParams['axes.prop_cycle'])
            edge_color = [
                colors[i % len(colors)]['color']
                for i in edge_y[hard_edge_mask]
            ]
        if len(edge_color) == 1:
            edge_color = np.array(edge_color)

        data = Data(edge_index=edge_index, att=edge_mask,
                    edge_color=edge_color, y=y, num_nodes=y.size(0)).to('cpu')

        G = to_networkx(data, node_attrs=['y'],
                        edge_attrs=['att', 'edge_color'])

        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
        node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
        label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G, seed=seed)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.1),
                    color=data['edge_color'],
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))

        if node_alpha is None:
            nx.draw_networkx_nodes(G, pos, node_color=y.tolist(),
                                   **node_kwargs)
        else:
            node_alpha_subset = node_alpha[subset]
            assert ((node_alpha_subset >= 0) & (node_alpha_subset <= 1)).all()
            nx.draw_networkx_nodes(G, pos, alpha=node_alpha_subset.tolist(),
                                   node_color=y.tolist(), **node_kwargs)

        nx.draw_networkx_labels(G, pos, **label_kwargs)
        return ax, G

    def __repr__(self):
        return f'{self.__class__.__name__}()'
