from typing import Optional

import math
from math import sqrt
from inspect import signature
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch.nn import Linear, LayerNorm, ReLU, Parameter, init
import torch.nn.functional as F
from torch import sigmoid
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx

EPS = 1e-15


class GraphMaskExplainer(torch.nn.Module):
    r"""The GraphMask-Explainer model from the `"Interpreting Graph Neural
    Networks for NLP With Differentiable Edge Masking"
    <https://arxiv.org/abs/2010.00577>`_ paper for identifying layer-wise
    compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-level and graph-level predictions.

    .. note::

        For an example of using GraphMask-Explainer,
        see `examples/graphmask_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        graphmask_explainer.py>`_.
    Args:
        in_channels (list): List of input channels for all the layers being
            used in a GNN module.
        out_channels (list): List of output channels for all the layers being
            used in a GNN module.
        num_layers (int): The number of layers to use.
        num_relations (int, optional): The number of relations for
            heterogeneous graphs. Must be set to :obj:`None` for homogeneous
            graphs. (default: :obj:`None`)
        num_bases (list, optional): If set to not :obj:`None`, this layer will
            use the basis-decomposition regularization scheme where
            :obj:`num_bases` denotes the number of bases to use.
            (default: :obj:`None`)
        num_blocks (list, optional): If set to not :obj:`None`, this layer
            will use the block-diagonal-decomposition regularization
            scheme where :obj:`num_blocks` denotes the number of blocks to
            use. (default: :obj:`None`)
        model_to_explain (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        pooling (str, optional): Pooling layer to use. Possible values are
            :obj:`add` and :obj:`mean`. (default: :obj:`add`)
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
        type_task (str): The type of graph being explained. Must set to
            :obj:`homogeneous` for non-relational graphs and set to
            :obj:`heterogeneous` for relational graphs.
            (default: :obj:`heterogeneous`)
        allow_multiple_explanations: Switch to allow explainer to explain
            node-level predictions for two or more nodes. Must set to
            :obj:`False` while explaining graph-level predictions and
            if only one node-level prediction is to be explained.
            (default: :obj:`False`)
        allow_edge_mask (boolean, optional): If set to :obj:`False`, the edge
            mask will not be optimized. (default: :obj:`True`)
        log (bool, optional): If set to :obj:`False`, will not log any
            learning progress. (default: :obj:`True`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.nn.models.GNNExplainer.coeffs`.
    """

    coeffs = {
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'node_feat_ent': 0.1,
    }

    def __init__(
            self,
            in_channels,
            out_channels,
            num_layers,
            num_relations=None,
            num_bases=None,
            num_blocks=None,
            model_to_explain=None,
            epochs: int = 5,
            lr: float = 0.01,
            pooling='add',
            num_hops: Optional[int] = None,
            return_type: str = 'log_prob',
            penalty_scaling: int = 5,
            lambda_optimizer_lr: int = 1e-2,
            init_lambda: int = 0.55,
            allowance: int = 0.03,
            feat_mask_type: str = 'scalar',
            type_task: str = 'heterogeneous',
            allow_multiple_explanations: bool = False,
            allow_edge_mask: bool = True,
            log: bool = True,
            **kwargs):
        super().__init__()
        assert return_type in ['log_prob', 'prob', 'raw', 'regression']
        assert feat_mask_type in ['feature', 'individual_feature', 'scalar']
        assert type_task in ['heterogeneous', 'homogeneous']
        if type_task == 'heterogeneous' and num_relations is None:
            raise ValueError('num_relations parameter cannot be None.')
        if num_bases is not None and not isinstance(num_bases, list):
            raise ValueError('num_bases parameter must be list-typed.')
        if num_bases is not None and len(num_bases) != num_layers:
            raise ValueError(
                'Length of num_bases list must be '
                'equal to the number of layers being used in a GNN module.')
        if num_bases is None:
            num_bases = [None] * num_layers
        if num_blocks is not None and not isinstance(num_blocks, list):
            raise ValueError('num_blocks parameter must be list-typed.')
        if num_blocks is not None and len(num_blocks) != num_layers:
            raise ValueError(
                'Length of num_blocks list must be '
                'equal to the number of layers being used in a GNN module.')
        if num_blocks is None:
            num_blocks = [None] * num_layers

        self.model_to_explain = model_to_explain
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.pooling = pooling
        self.return_type = return_type
        self.log = log
        self.allow_edge_mask = allow_edge_mask
        self.feat_mask_type = feat_mask_type
        self.type_task = type_task
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.init_lambda = init_lambda
        self.lambda_optimizer_lr = lambda_optimizer_lr
        self.penalty_scaling = penalty_scaling
        self.allowance = allowance
        self.allow_multiple_explanations = allow_multiple_explanations
        self.coeffs.update(kwargs)

    # function to compute hard_concrete_distribution to drop edges
    def __hard_concrete__(
            self,
            input_element,
            summarize_penalty=True,
            beta=1 / 3,
            gamma=-0.2,
            zeta=1.2,
            loc_bias=2,
            min_val=0,
            max_val=1,
            training=True):
        input_element = input_element + loc_bias

        if training:
            u = torch.empty_like(input_element).uniform_(1e-6, 1.0 - 1e-6)

            s = sigmoid((torch.log(u) - torch.log(1 - u) + input_element) /
                        beta)

            penalty = sigmoid(input_element - beta *
                              np.math.log(-gamma / zeta))
        else:
            s = sigmoid(input_element)
            penalty = torch.zeros_like(input_element)

        if summarize_penalty:
            penalty = penalty.mean()

        s = s * (zeta - gamma) + gamma

        clipped_s = s.clamp(min_val, max_val)

        if True:
            clip_value = (torch.min(clipped_s) + torch.max(clipped_s)) / 2
            hard_concrete = (clipped_s > clip_value).float()
            clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        return clipped_s, penalty

    def __set_masks__(self, i_dim, j_dim, h_dim, x, edge_index):
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
                self.transforms,
                self.layer_norms,
                self.non_linear,
                self.output_layer]
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

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model_to_explain.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model_to_explain.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(
            self,
            node_idx,
            x,
            edge_index,
            edge_type=None,
            edge_weight=None,
            **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        if self.type_task == "heterogeneous" and edge_type is not None:
            edge_type = edge_type[edge_mask]
        if self.type_task == "homogeneous" and edge_weight is not None:
            edge_weight = edge_weight[edge_mask]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return (
            x,
            edge_index,
            edge_type,
            mapping,
            edge_mask,
            subset,
            kwargs) if edge_type is not None else (
            x,
            edge_index,
            edge_weight,
            mapping,
            edge_mask,
            subset,
            kwargs) if edge_weight is not None else (
            x,
            edge_index,
            mapping,
            edge_mask,
            subset,
            kwargs)

    def __loss__(self, node_idx, log_logits, pred_label, penalty):
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

    def __to_log_prob__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.log_softmax(dim=-1) if self.return_type == 'raw' else x
        x = x.log() if self.return_type == 'prob' else x
        return x

    def freeze_model(self, module):
        for param in module.parameters():
            param.requires_grad = False

    # function to set message_scale which determines if edge has to be dropped
    def __inject_message_scale__(self, message_scale, set=False):
        if not set:
            self.injected_message_scale = message_scale
        self.injected_message_scale = None

    # function to supersede dropped edges with the learned baseline
    def __inject_message_replacement__(self, message_replacement, set=False):
        if not set:
            self.injected_message_replacement = [message_replacement]
        self.injected_message_replacement = None

    def __train_node_explainer__(
            self,
            node_idx,
            x1,
            edge_index1,
            edge_type1=None,
            edge_weight1=None,
            **kwargs):
        r"""Learns a node feature mask and an edge mask and returns only the
        learned node feature mask that plays a crucial role to explain the
        prediction made by the GNN for node(s) :attr:`node_idx`.
        Args:
            node_idx (list): List of node(s) to explain.
            x1 (Tensor): The node feature matrix.
            edge_index1 (LongTensor): The edge indices.
            edge_type1 (Tensor, optional): The one-dimensional relation
                type/index for each edge in :attr:`edge_index1`.
                Must be set to :obj:`None` only for homogeneous graphs
                related node-level predictions. (default: :obj:`None`)
            edge_weight1 (list, optional): List of layer-wise weights for
                each edge in the input graph. Must be set to :obj:`None` only
                when edge_weight is not passed to a particular layer
                or for heterogeneous graphs related node-level predictions.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to the GNN
                module.
        :rtype: (:class:`Tensor`)
        """

        if not isinstance(node_idx, list):
            raise ValueError('node_idx parameter must be list-typed.')
        if not self.allow_multiple_explanations and len(node_idx) != 1:
            raise ValueError('Length of node_idx parameter must be equal to '
                             '1 while not allowing multiple explanations.')
        if self.allow_multiple_explanations and len(node_idx) < 2:
            raise ValueError(
                'More than 1 node must be passed to node_idx list '
                'while allowing multiple explanations.')
        if self.type_task == 'heterogeneous' and edge_type1 is None:
            raise ValueError('Please provide edge types of the input graph.')

        if self.type_task == 'homogeneous':
            if edge_weight1 is not None and not isinstance(edge_weight1, list):
                raise ValueError('edge_weight1 parameter must be list-typed.')
            if edge_weight1 is not None and len(
                    edge_weight1) != self.model_to_explain.num_layers:
                raise ValueError(
                    'Length of edge_weight1 list must be '
                    'equal to the number of layers being used in a GNN '
                    'module.')
            if edge_weight1 is None:
                edge_weight1 = [None] * self.model_to_explain.num_layers

        num_nodes = x1.size(0)
        self.freeze_model(self.model_to_explain)

        if self.allow_multiple_explanations:
            if self.type_task == 'heterogeneous':
                x, edge_index, edge_type, mapping, hard_edge_mask, subset, \
                    kwargs = self.__subgraph__(node_idx, x1, edge_index1,
                                               edge_type1, **kwargs)
            elif self.type_task == 'homogeneous':
                e_weights = []
                for number_layer in range(self.model_to_explain.num_layers):
                    if edge_weight1[number_layer] is not None:
                        (
                            x,
                            edge_index,
                            edge_weight,
                            mapping,
                            hard_edge_mask,
                            subset,
                            kwargs,
                        ) = self.__subgraph__(node_idx, x1, edge_index1,
                                              edge_weight1[number_layer],
                                              **kwargs)
                        e_weights.append(edge_weight[hard_edge_mask])
                    elif edge_weight1[number_layer] is None:
                        x, edge_index, mapping, hard_edge_mask, subset, \
                            kwargs = self.__subgraph__(node_idx, x1,
                                                       edge_index1, **kwargs)
                        e_weights.append(None)
            if hard_edge_mask[hard_edge_mask].size(0) == 0:
                raise ValueError('no edge has been preserved.')
        else:
            if self.type_task == 'heterogeneous':
                x, edge_index, edge_type, mapping, hard_edge_mask, subset, \
                    kwargs = self.__subgraph__(node_idx[0], x1, edge_index1,
                                               edge_type1, **kwargs)
            elif self.type_task == 'homogeneous':
                e_weights = []
                for number_layer in range(self.model_to_explain.num_layers):
                    if edge_weight1[number_layer] is not None:
                        (
                            x,
                            edge_index,
                            edge_weight,
                            mapping,
                            hard_edge_mask,
                            subset,
                            kwargs,
                        ) = self.__subgraph__(node_idx[0], x1, edge_index1,
                                              edge_weight1[number_layer],
                                              **kwargs)
                        e_weights.append(edge_weight[hard_edge_mask])
                    elif edge_weight1[number_layer] is None:
                        x, edge_index, mapping, hard_edge_mask, subset, \
                            kwargs = self.__subgraph__(node_idx[0], x1,
                                                       edge_index1, **kwargs)
                        e_weights.append(None)
            if hard_edge_mask[hard_edge_mask].size(0) == 0:
                raise ValueError('no edge has been preserved.')

        input_dims = np.array(
            [layer.in_channels for layer in self.model_to_explain.gnn_layers])
        output_dims = np.array(
            [layer.out_channels for layer in self.model_to_explain.gnn_layers])

        self.__set_masks__(input_dims, output_dims, output_dims, x, edge_index)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for layer in reversed(list(range(self.model_to_explain.num_layers))):
            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.epochs)
                if self.allow_multiple_explanations:
                    pbar.set_description(
                        f'Train explainer for nodes {node_idx} with layer '
                        f'{layer}')
                else:
                    pbar.set_description(
                        f'Train explainer for node {node_idx[0]} with layer '
                        f'{layer}')
            self.enable_layer(layer)
            for epoch in range(self.epochs):
                self.model_to_explain.eval()
                with torch.no_grad():
                    if self.type_task == 'heterogeneous':
                        out = self.model_to_explain(
                            x=x, edge_index=edge_index, edge_type=edge_type)
                    elif self.type_task == 'homogeneous' and \
                            not all(v is None for v in edge_weight1):
                        out = self.model_to_explain(
                            x=x, edge_index=edge_index, edge_weight=e_weights)
                    elif self.type_task == 'homogeneous' and \
                            all(v is None for v in edge_weight1):
                        out = self.model_to_explain(
                            x=x, edge_index=edge_index,
                            edge_weight=edge_weight1)
                    if self.return_type == 'regression':
                        prediction = out
                    else:
                        log_logits = self.__to_log_prob__(out)
                        pred_label = log_logits.argmax(dim=-1)
                self.model_to_explain.train()
                gates, total_penalty = [], 0
                latest_source_embeddings = [layer.get_latest_source_embeddings(
                ) for layer in self.model_to_explain.gnn_layers]
                latest_messages = [layer.get_latest_messages()
                                   for layer in
                                   self.model_to_explain.gnn_layers]
                latest_target_embeddings = [layer.get_latest_target_embeddings(
                ) for layer in self.model_to_explain.gnn_layers]
                gate_input = [
                    latest_source_embeddings,
                    latest_messages,
                    latest_target_embeddings]
                for i in range(self.model_to_explain.num_layers):
                    output = self.full_biases[i]
                    for j in range(len(gate_input)):
                        partial = self.gates[i * 4][j](gate_input[j][i])
                        result = self.gates[(i * 4) + 1][j](partial)
                        output = output + result
                    relu_output = self.gates[(
                        i * 4) + 2](output / len(gate_input))
                    sampling_weights = self.gates[(
                        i * 4) + 3](relu_output).squeeze(dim=-1)
                    sampling_weights, penalty = self.__hard_concrete__(
                        sampling_weights)
                    gates.append(sampling_weights)
                    total_penalty += penalty

                self.__inject_message_scale__(gates)
                self.__inject_message_replacement__(self.baselines)

                self.lambda_op = torch.tensor(
                    self.init_lambda, requires_grad=True)
                optimizer_lambda = torch.optim.RMSprop(
                    [self.lambda_op], lr=self.lambda_optimizer_lr,
                    centered=True)

                optimizer.zero_grad()
                optimizer_lambda.zero_grad()

                h = x * self.node_feat_mask.sigmoid()
                if self.type_task == 'heterogeneous':
                    new_out = self.model_to_explain(
                        x=h, edge_index=edge_index, edge_type=edge_type,
                        message_scale=self.injected_message_scale,
                        message_replacement=self.injected_message_replacement)
                elif self.type_task == 'homogeneous' and \
                        not all(v is None for v in edge_weight1):
                    new_out = self.model_to_explain(
                        x=h, edge_index=edge_index, edge_weight=e_weights,
                        message_scale=self.injected_message_scale,
                        message_replacement=self.injected_message_replacement)
                elif self.type_task == 'homogeneous' and \
                        all(v is None for v in edge_weight1):
                    new_out = self.model_to_explain(
                        x=h, edge_index=edge_index, edge_weight=edge_weight1,
                        message_scale=self.injected_message_scale,
                        message_replacement=self.injected_message_replacement)

                self.__inject_message_scale__(gates, True)
                self.__inject_message_replacement__(self.baselines, True)

                if self.return_type == 'regression':
                    loss = self.__loss__(
                        mapping, new_out, prediction, total_penalty)
                else:
                    log_logits = self.__to_log_prob__(new_out)
                    loss = self.__loss__(
                        mapping, log_logits, pred_label, total_penalty)
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

    def __explain_node__(self, node_idx, x, edge_index, **kwargs):
        r"""Returns only the learned edge mask that plays a crucial role to explain the
        prediction made by the GNN for node(s) :attr:`node_idx`.
        Args:
            node_idx (list): List of node(s) to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.
        :rtype: (:class:`Tensor`)
        """

        if not isinstance(node_idx, list):
            raise ValueError('node_idx parameter must be list-typed.')
        if not self.allow_multiple_explanations and len(node_idx) != 1:
            raise ValueError('Length of node_idx parameter must be equal to 1 '
                             'while not allowing multiple explanations.')
        if self.allow_multiple_explanations and len(node_idx) < 2:
            raise ValueError(
                'More than 1 node must be passed to node_idx list '
                'while allowing multiple explanations.')

        self.freeze_model(self.model_to_explain)

        self.model_to_explain.eval()
        if self.allow_multiple_explanations:
            _, _, _, hard_edge_mask, _, _ = \
                self.__subgraph__(node_idx, x, edge_index, **kwargs)
        else:
            _, _, _, hard_edge_mask, _, _ = \
                self.__subgraph__(node_idx[0], x, edge_index, **kwargs)

        with torch.no_grad():
            latest_source_embeddings = [layer.get_latest_source_embeddings(
            ) for layer in self.model_to_explain.gnn_layers]
            latest_messages = [layer.get_latest_messages()
                               for layer in self.model_to_explain.gnn_layers]
            latest_target_embeddings = [layer.get_latest_target_embeddings(
            ) for layer in self.model_to_explain.gnn_layers]
            gate_input = [
                latest_source_embeddings,
                latest_messages,
                latest_target_embeddings]
            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.model_to_explain.num_layers)
            for i in range(self.model_to_explain.num_layers):
                if self.log:  # pragma: no cover
                    if self.allow_multiple_explanations:
                        pbar.set_description(f'Explain nodes {node_idx}')
                    else:
                        pbar.set_description(f'Explain node {node_idx[0]}')
                output = self.full_biases[i]
                for j in range(len(gate_input)):
                    partial = self.gates[i * 4][j](gate_input[j][i])
                    result = self.gates[(i * 4) + 1][j](partial)
                    output = output + result
                relu_output = self.gates[(i * 4) + 2](output / len(gate_input))
                sampling_weights = self.gates[(
                    i * 4) + 3](relu_output).squeeze(dim=-1)
                sampling_weights, penalty = self.__hard_concrete__(
                    sampling_weights, training=False)
                if i == 0:
                    edge_weight = sampling_weights
                else:
                    edge_weight = torch.cat((edge_weight, sampling_weights), 0)
                if self.log:  # pragma: no cover
                    pbar.update(1)
        if self.log:  # pragma: no cover
            pbar.close()

        edge_mask = edge_weight.view(-1, edge_weight.size(0) //
                                     self.model_to_explain.num_layers)
        edge_mask = torch.mean(edge_mask, 0)

        return edge_mask

    def __train_graph_explainer__(
            self,
            graph_idx,
            x,
            edge_index,
            edge_type=None,
            edge_weight=None):
        r"""Learns a node feature mask and an edge mask and returns only the
        learned node feature mask that plays a crucial role to explain the
        prediction made by the GNN for graph :attr:`graph_idx`.
        Args:
            graph_idx (list): List of graph id to explain.
            x (Tensor): The node feature matrix for graph :attr:`graph_idx`.
            edge_index (LongTensor): The edge indices for graph
                :attr:`graph_idx`.
            edge_type (Tensor, optional): The one-dimensional relation
                type/index for each edge in :attr:`edge_index`.
                Must be set to :obj:`None` only for homogeneous graphs
                related graph-level predictions. (default: :obj:`None`)
            edge_weight (list, optional): List of layer-wise weights for
                each edge in :attr:`edge_index`. Must be set to :obj:`None`
                only when edge_weight is not passed to a particular layer
                or for heterogeneous graphs related graph-level predictions.
                (default: :obj:`None`)
        :rtype: (:class:`Tensor`)
        """

        if not isinstance(graph_idx, list) or len(graph_idx) > 1:
            raise ValueError(
                'graph_idx parameter must be list-typed with length '
                'must not exceed 1.')
        if self.type_task == 'heterogeneous' and edge_type is None:
            raise ValueError('Please provide edge types of the input graph.')
        if self.type_task == 'homogeneous':
            if edge_weight is not None and not isinstance(edge_weight, list):
                raise ValueError('edge_weight parameter must be list-typed.')
            if edge_weight is None:
                edge_weight = [None] * self.model_to_explain.num_layers

        self.freeze_model(self.model_to_explain)
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)

        input_dims = np.array(
            [layer.in_channels for layer in self.model_to_explain.gnn_layers])
        output_dims = np.array(
            [layer.out_channels for layer in self.model_to_explain.gnn_layers])

        self.__set_masks__(input_dims, output_dims, output_dims, x, edge_index)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for layer in reversed(list(range(self.model_to_explain.num_layers))):
            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.epochs)
                pbar.set_description(
                    f'Train explainer for graph {graph_idx[0]} with layer '
                    f'{layer}')
            self.enable_layer(layer)
            for epoch in range(self.epochs):
                self.model_to_explain.eval()
                with torch.no_grad():
                    if self.type_task == 'heterogeneous':
                        out = self.model_to_explain(
                            x=x,
                            edge_index=edge_index,
                            edge_type=edge_type,
                            batch=batch,
                            pooling=self.pooling,
                            task='graph')
                    elif self.type_task == 'homogeneous':
                        out = self.model_to_explain(
                            x=x,
                            edge_index=edge_index,
                            edge_weight=edge_weight,
                            batch=batch,
                            pooling=self.pooling,
                            task='graph')
                    if self.return_type == 'regression':
                        prediction = out
                    else:
                        log_logits = self.__to_log_prob__(out)
                        pred_label = log_logits.argmax(dim=-1)
                self.model_to_explain.train()
                gates, total_penalty = [], 0
                latest_source_embeddings = [layer.get_latest_source_embeddings(
                ) for layer in self.model_to_explain.gnn_layers]
                latest_messages = [layer.get_latest_messages()
                                   for layer in
                                   self.model_to_explain.gnn_layers]
                latest_target_embeddings = [layer.get_latest_target_embeddings(
                ) for layer in self.model_to_explain.gnn_layers]
                gate_input = [
                    latest_source_embeddings,
                    latest_messages,
                    latest_target_embeddings]
                for i in range(self.model_to_explain.num_layers):
                    output = self.full_biases[i]
                    for j in range(len(gate_input)):
                        partial = self.gates[i * 4][j](gate_input[j][i])
                        result = self.gates[(i * 4) + 1][j](partial)
                        output = output + result
                    relu_output = self.gates[(
                        i * 4) + 2](output / len(gate_input))
                    sampling_weights = self.gates[(
                        i * 4) + 3](relu_output).squeeze(dim=-1)
                    sampling_weights, penalty = self.__hard_concrete__(
                        sampling_weights)
                    gates.append(sampling_weights)
                    total_penalty += penalty

                self.__inject_message_scale__(gates)
                self.__inject_message_replacement__(self.baselines)

                self.lambda_op = torch.tensor(
                    self.init_lambda, requires_grad=True)
                optimizer_lambda = torch.optim.RMSprop(
                    [self.lambda_op], lr=self.lambda_optimizer_lr,
                    centered=True)

                optimizer.zero_grad()
                optimizer_lambda.zero_grad()

                h = x * self.node_feat_mask.sigmoid()
                if self.type_task == 'heterogeneous':
                    new_out = self.model_to_explain(
                        x=h,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        message_scale=self.injected_message_scale,
                        message_replacement=self.injected_message_replacement,
                        batch=batch,
                        pooling=self.pooling,
                        task='graph')
                elif self.type_task == 'homogeneous':
                    new_out = self.model_to_explain(
                        x=h,
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        message_scale=self.injected_message_scale,
                        message_replacement=self.injected_message_replacement,
                        batch=batch,
                        pooling=self.pooling,
                        task='graph')

                self.__inject_message_scale__(gates, True)
                self.__inject_message_replacement__(self.baselines, True)

                if self.return_type == 'regression':
                    loss = self.__loss__(
                        [-1], new_out, prediction, total_penalty)
                else:
                    log_logits = self.__to_log_prob__(new_out)
                    loss = self.__loss__(
                        [-1], log_logits, pred_label, total_penalty)
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

    def __explain_graph__(self, graph_idx):
        r"""Returns only the learned edge mask that plays a crucial role to explain the
        prediction made by the GNN for graph :attr:`graph_idx`.
        Args:
            graph_idx (list): List of graph id to explain.
        :rtype: (:class:`Tensor`)
        """

        if not isinstance(graph_idx, list) or len(graph_idx) > 1:
            raise ValueError(
                'graph_idx parameter must be list-typed with length '
                'must not exceed 1.')

        self.freeze_model(self.model_to_explain)
        self.model_to_explain.eval()

        with torch.no_grad():
            latest_source_embeddings = [layer.get_latest_source_embeddings(
            ) for layer in self.model_to_explain.gnn_layers]
            latest_messages = [layer.get_latest_messages()
                               for layer in self.model_to_explain.gnn_layers]
            latest_target_embeddings = [layer.get_latest_target_embeddings(
            ) for layer in self.model_to_explain.gnn_layers]
            gate_input = [
                latest_source_embeddings,
                latest_messages,
                latest_target_embeddings]
            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.model_to_explain.num_layers)
                pbar.set_description(f'Explain graph {graph_idx[0]}')
            for i in range(self.model_to_explain.num_layers):
                output = self.full_biases[i]
                for j in range(len(gate_input)):
                    partial = self.gates[i * 4][j](gate_input[j][i])
                    result = self.gates[(i * 4) + 1][j](partial)
                    output = output + result
                relu_output = self.gates[(i * 4) + 2](output / len(gate_input))
                sampling_weights = self.gates[(
                    i * 4) + 3](relu_output).squeeze(dim=-1)
                sampling_weights, penalty = self.__hard_concrete__(
                    sampling_weights, training=False)
                if i == 0:
                    edge_weight = sampling_weights
                else:
                    edge_weight = torch.cat((edge_weight, sampling_weights), 0)
                if self.log:  # pragma: no cover
                    pbar.update(1)
        if self.log:  # pragma: no cover
            pbar.close()

        edge_mask = edge_weight.view(-1, edge_weight.size(0) //
                                     self.model_to_explain.num_layers)
        edge_mask = torch.mean(edge_mask, 0)

        return edge_mask

    def visualize_subgraph(
            self,
            node_idx,
            edge_index,
            edge_mask,
            y=None,
            edge_y=None,
            node_alpha=None,
            seed=10,
            task='node',
            **kwargs):
        r"""Visualizes the subgraph given an edge mask :attr:`edge_mask`.

        Args:
            node_idx (list): List of node id(s), the explanation of which
                is to be displayed. It may consist of either one node id or
                more. Set to some graph id to visualize explanations of that
                particular graph.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. All nodes will have the same color
                if :attr:`node_idx` is set to some graph id.
                (default: :obj:`None`)
            edge_y (Tensor, optional): The edge labels used as edge colorings.
                (default: :obj:`None`)
            node_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating
                transparency of each node. (default: :obj:`None`)
            seed (int, optional): Random seed of the :obj:`networkx` node
                placement algorithm. (default: :obj:`10`)
            task (str, optional): Switch to decide either node-level or
                graph-level task explanations are going to be displayed.
                Must set to :obj:`graph` to visualize graph-level task
                explanations. (default: :obj:`node`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """

        if (self.feat_mask_type == 'feature' or self.feat_mask_type ==
                'individual_feature') and node_alpha is not None:
            raise ValueError(
                'Visualizing explainable subgraph with feature mask type as '
                'either individual_feature or feature is not supported. '
                'Please either set node_alpha to None or set feat_mask_type '
                'to scaler instead.')
        if not isinstance(node_idx, list):
            raise ValueError('node_idx parameter must be list-typed.')
        if task == 'node':
            if not self.allow_multiple_explanations and len(node_idx) != 1:
                raise ValueError(
                    'Length of node_idx parameter must be equal to 1 '
                    'while not allowing multiple explanations.')
            if self.allow_multiple_explanations and len(node_idx) < 2:
                raise ValueError(
                    'More than 1 node must be passed to node_idx list '
                    'while allowing multiple explanations.')
        else:
            if len(node_idx) != 1:
                raise ValueError(
                    'While visualizing graph-level task explanations, '
                    'length of node_idx list must not exceed 1.')

        if task == 'graph':
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1))
            subset = torch.arange(edge_index.max().item() + 1,
                                  device=edge_index.device)
            y = None
        else:
            if self.allow_multiple_explanations:
                subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                    node_idx, self.num_hops, edge_index, relabel_nodes=True,
                    num_nodes=None, flow=self.__flow__())

            else:
                # Only operate on a k-hop subgraph around `node_idx`.
                subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                    node_idx[0], self.num_hops, edge_index, relabel_nodes=True,
                    num_nodes=None, flow=self.__flow__())

        threshold = (torch.min(edge_mask) + torch.max(edge_mask)) / 2
        edge_mask = (edge_mask >= threshold).to(torch.float)

        if edge_mask.size(0) == 1:
            edge_mask = edge_mask.numpy()

        if task == 'graph':
            if y is None:
                y = torch.zeros(edge_index.max().item() + 1,
                                device=edge_index.device)
            else:
                y = y[subset].to(torch.float) / y.max().item()
        else:
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

        if task == 'graph':
            data = Data(
                edge_index=edge_index,
                att=edge_mask,
                edge_color=edge_color,
                y=y,
                num_nodes=y.size(0)).to('cpu')
        else:
            data = Data(
                edge_index=edge_index,
                att=edge_mask,
                edge_color=edge_color,
                y=y,
                num_nodes=y.size(0)).to('cpu')

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
