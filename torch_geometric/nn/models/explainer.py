from abc import abstractmethod
from inspect import signature
from math import sqrt
from typing import Optional

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx


def set_masks(model: torch.nn.Module, mask: Tensor, edge_index: Tensor,
              apply_sigmoid: bool = True):
    """Apply mask to every graph layer in the model."""
    loop_mask = edge_index[0] != edge_index[1]

    # Loop over layers and set masks on MessagePassing layers:
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = mask
            module.__loop_mask__ = loop_mask
            module.__apply_sigmoid__ = apply_sigmoid


def clear_masks(model: torch.nn.Module):
    """Clear all masks from the model."""
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None
            module.__loop_mask__ = None
            module.__apply_sigmoid__ = True
    return module


class CaptumModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mask_type: str = "edge",
                 output_idx: Optional[int] = None):
        super().__init__()
        assert mask_type in ['edge', 'node', 'node_and_edge']

        self.mask_type = mask_type
        self.model = model
        self.output_idx = output_idx

    def forward(self, mask, *args):
        """"""
        # The mask tensor, which comes from Captum's attribution methods,
        # contains the number of samples in dimension 0. Since we are
        # working with only one sample, we squeeze the tensors below.
        assert mask.shape[0] == 1, "Dimension 0 of input should be 1"
        if self.mask_type == "edge":
            assert len(args) >= 2, "Expects at least x and edge_index as args."
        if self.mask_type == "node":
            assert len(args) >= 1, "Expects at least edge_index as args."
        if self.mask_type == "node_and_edge":
            assert args[0].shape[0] == 1, "Dimension 0 of input should be 1"
            assert len(args[1:]) >= 1, "Expects at least edge_index as args."

        # Set edge mask:
        if self.mask_type == 'edge':
            set_masks(self.model, mask.squeeze(0), args[1],
                      apply_sigmoid=False)
        elif self.mask_type == 'node_and_edge':
            set_masks(self.model, args[0].squeeze(0), args[1],
                      apply_sigmoid=False)
            args = args[1:]

        if self.mask_type == 'edge':
            x = self.model(*args)

        elif self.mask_type == 'node':
            x = self.model(mask.squeeze(0), *args)

        else:
            x = self.model(mask[0], *args)

        # Clear mask:
        if self.mask_type in ['edge', 'node_and_edge']:
            clear_masks(self.model)

        if self.output_idx is not None:
            x = x[self.output_idx].unsqueeze(0)

        return x


def to_captum(model: torch.nn.Module, mask_type: str = "edge",
              output_idx: Optional[int] = None) -> torch.nn.Module:
    r"""Converts a model to a model that can be used for
    `Captum.ai <https://captum.ai/>`_ attribution methods.

    .. code-block:: python

        from captum.attr import IntegratedGradients
        from torch_geometric.nn import GCN, from_captum

        model = GCN(...)
        ...  # Train the model.

        # Explain predictions for node `10`:
        output_idx = 10

        captum_model = to_captum(model, mask_type="edge",
                                 output_idx=output_idx)
        edge_mask = torch.ones(num_edges, requires_grad=True, device=device)

        ig = IntegratedGradients(captum_model)
        ig_attr = ig.attribute(edge_mask.unsqueeze(0),
                               target=int(y[output_idx]),
                               additional_forward_args=(x, edge_index),
                               internal_batch_size=1)

    .. note::
        For an example of using a Captum attribution method within PyG, see
        `examples/captum_explainability.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        captum_explainability.py>`_.

    Args:
        model (torch.nn.Module): The model to be explained.
        mask_type (str, optional): Denotes the type of mask to be created with
            a Captum explainer. Valid inputs are :obj:`"edge"`, :obj:`"node"`,
            and :obj:`"node_and_edge"`:

            1. :obj:`"edge"`: The inputs to the forward function should be an
               edge mask tensor of shape :obj:`[1, num_edges]`, a regular
               :obj:`x` matrix and a regular :obj:`edge_index` matrix.

            2. :obj:`"node"`: The inputs to the forward function should be a
               node feature tensor of shape :obj:`[1, num_nodes, num_features]`
               and a regular :obj:`edge_index` matrix.

            3. :obj:`"node_and_edge"`: The inputs to the forward function
               should be a node feature tensor of shape
               :obj:`[1, num_nodes, num_features]`, an edge mask tensor of
               shape :obj:`[1, num_edges]` and a regular :obj:`edge_index`
               matrix.

            For all mask types, additional arguments can be passed to the
            forward function as long as the first arguments are set as
            described. (default: :obj:`"edge"`)
        output_idx (int, optional): Index of the output element (node or link
            index) to be explained. With :obj:`output_idx` set, the forward
            function will return the output of the model for the element at
            the index specified. (default: :obj:`None`)
    """
    return CaptumModel(model, mask_type, output_idx)


class Explainer(torch.nn.Module):
    r"""Base Explainer class for Graph Neural Networks, e.g.
    PGExplainer and GNNExplainer."""

    # TODO: Change docstring
    # TODO: Rearange args
    def __init__(self, model: torch.nn.Module, lr: Optional[float] = None,
                 epochs: Optional[int] = None, num_hops: Optional[int] = None,
                 return_type: str = 'log_prob', log: bool = False, **kwargs):
        super().__init__()
        assert return_type in ['log_prob', 'prob', 'raw', 'regression']

        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.num_hops = num_hops
        self.return_type = return_type
        self.log = log

    @property
    def num_hops(self):
        return self._num_hops

    @num_hops.setter
    def num_hops(self, num_hops):
        if num_hops is not None:
            self._num_hops = num_hops
        else:
            # Could this be a general utils function?
            k = 0
            for module in self.model.modules():
                if isinstance(module, MessagePassing):
                    k += 1
            self._num_hops = k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    # TODO: Find out, whether to change variable name node_idx here as well?
    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        r"""Returns the subgraph of the given node.
        Args:
            node_idx (int): The node idx to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.
        :rtype: (Tensor, Tensor, LongTensor, LongTensor, LongTensor, dict)
        """
        num_nodes, num_edges = x.size(0), edge_index.size(0)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        kwargs_new = {}
        for key, value in kwargs.items():
            if torch.is_tensor(value) and value.size(0) == num_nodes:
                kwargs_new[key] = value[subset]
            elif torch.is_tensor(value) and value.size(0) == num_edges:
                kwargs_new[key] = value[edge_mask]
            kwargs_new[key] = value  # TODO: this is not in PGExplainer
        return x, edge_index, mapping, edge_mask, subset, kwargs_new

    def __to_log_prob__(self, x):
        x = x.log_softmax(dim=-1) if self.return_type == 'raw' else x
        x = x.log() if self.return_type == 'prob' else x
        return x

    def get_initial_prediction(self, x, edge_index, batch=None, **kwargs):
        with torch.no_grad():
            if batch is not None:
                out = self.model(x, edge_index, batch, **kwargs)
            else:
                out = self.model(x, edge_index, **kwargs)
            # Is this if necessary?
            if self.return_type == 'regression':
                prediction = out
            else:
                log_logits = self.__to_log_prob__(out)
                prediction = log_logits.argmax(dim=-1)
        return prediction

    def get_loss(self, out, prediction, node_idx):
        if self.return_type == 'regression':
            loss = self.__loss__(node_idx, out, prediction)
        else:
            log_logits = self.__to_log_prob__(out)
            loss = self.__loss__(node_idx, log_logits, prediction)
        return loss

    @abstractmethod
    def __loss__(self):
        # TODO: Check if loss is used in most GNN Xai methods.
        raise NotImplementedError

    # TODO: Should this method be outside the class?
    # TODO: Enhance for negative attributions
    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, edge_y=None, node_alpha=None,
                           seed=10, **kwargs):
        r"""Visualizes the subgraph given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
                Set to :obj:`-1` to explain graph.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. All nodes will have the same color
                if :attr:`node_idx` is :obj:`-1`.(default: :obj:`None`).
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            edge_y (Tensor, optional): The edge labels used as edge colorings.
            node_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating
                transparency of each node.
            seed (int, optional): Random seed of the :obj:`networkx` node
                placement algorithm. (default: :obj:`10`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx == -1:
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                              device=edge_mask.device)
            subset = torch.arange(edge_index.max().item() + 1,
                                  device=edge_index.device)
            y = None

        else:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

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
