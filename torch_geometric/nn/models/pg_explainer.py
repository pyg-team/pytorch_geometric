from typing import Optional

import torch
from torch import nn
from tqdm import tqdm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from torch.nn.functional import nll_loss
from torch_scatter import scatter_mean
EPS = 1e-15


class PGExplainer(torch.nn.Module):
    r"""The PGExplainer model from the `"Parameterized Explainer for Graph Neural
    Network" <https://arxiv.org/abs/2011.04573>`_ paper. It uses a neural
    network :obj:`explainer_model` , to predict edges that are crucial to
    a GNNs node or graph prediction.

    .. note::

        For an example of using PGExplainer, see `examples/pg_explainer.py <
        >`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        out_channels(int): Size of output of the last :class:`MessagePassing`
            layer in :obj:`model`.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`30`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.003`)
        num_hops (int, optional): The number of hops the :obj:`model` is
            aggregating information from.
            If set to :obj:`None`, will automatically try to detect this
            information based on the number of
            :class:`~torch_geometric.nn.conv.message_passing.MessagePassing`
            layers inside :obj:`model`. (default: :obj:`None`)
        task (str): Denotes the type of task that needs explanation. Valid
            inputs are :obj:`"node"` (for node classification) and
            :obj:`"graph"` (for graph classification). (default: :obj:`"node"`)
        return_type (str, optional): Denotes the type of output from
            :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model returns
            the logarithm of probabilities), :obj:`"prob"` (the model returns
            probabilities) and :obj:`"raw"` (the model returns raw scores).
            (default: :obj:`"log_prob"`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """

    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [5.0, 2.0],
        'bias': 0
    }

    def __init__(self, model, out_channels: int, epochs: int = 30,
                 lr: float = 0.003, num_hops: Optional[int] = None,
                 task: str = 'node', return_type: str = 'log_prob',
                 log: bool = True):
        super(PGExplainer, self).__init__()
        assert return_type in ['log_prob', 'prob', 'raw']
        assert task in ['node', 'graph']

        self.model = model
        self.out_channels = out_channels
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.task = task
        self.return_type = return_type
        self.log = log

        self.exp_in_channels = 2 * out_channels if (
            task == 'graph') else 3 * out_channels
        self.explainer_model = nn.Sequential(
            nn.Linear(self.exp_in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

    def __to_log_prob__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.log_softmax(dim=-1) if self.return_type == 'raw' else x
        x = x.log() if self.return_type == 'prob' else x
        return x

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __get_temp__(self, e: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * ((temp[1] / temp[0])**(e / self.epochs))

    def __subgraph__(self, node_idx, x, z, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        z = z[subset]

        kwargs_new = {}  # all changes are made to kwargs_new and not kwargs.
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs_new[key] = item

        return x, z, edge_index, mapping, edge_mask, kwargs_new

    def __create_explainer_input__(self, edge_index, x, node_id=None):

        rows, cols = edge_index
        x_j, x_i = x[rows], x[cols]
        if self.task == 'node':
            x_node = x[node_id].repeat(rows.size(0), 1)
            explainer_input = torch.cat([x_i, x_j, x_node], 1)
        else:
            explainer_input = torch.cat([x_i, x_j], 1)
        return explainer_input

    def __compute_edge_mask__(self, edge_weight, temperature=1.0, bias=0.0,
                              training=True):

        if training:  # noise is added to edge_weight.
            bias += 0.0001
            eps = (bias -
                   (1 - bias)) * torch.rand(edge_weight.size()) + (1 - bias)
            eps = eps.to(edge_weight.device)
            return (eps.log() -
                    (1 - eps).log() + edge_weight).squeeze() / temperature

        else:
            return edge_weight.squeeze()

    def __set_masks__(self, edge_mask):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = edge_mask

    def __loss__(self, edge_mask, log_logits, pred_label, batch=None,
                 edge_index=None):
        cross_ent_loss = nll_loss(log_logits, pred_label, reduction='sum')
        mask = edge_mask.sigmoid().squeeze()

        # Regularization losses
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask_ent_reg = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent_reg.mean() if batch is None else scatter_mean(
            mask_ent_reg, batch[edge_index[0]]).sum()
        mask_ent_loss *= self.coeffs['edge_ent']

        return cross_ent_loss + size_loss + mask_ent_loss

    def train_explainer(self, x, z, edge_index, node_idxs=None, batch=None,
                        **kwargs):
        r"""Trains the :obj:`explainer_model` to predict an
        edge mask that is crucial to explain the predictions of
        the :obj:`model`.

        Args:
            x (Tensor): The node feature matrix.
            z (Tensor): Node embedding from last :class:`MessagePassing` layer
                in :obj:`model`.
            edge_index (LongTensor): The edge indices.
            node_idxs (Optional, LongTensor): The nodes used to train
                :obj:`explainer_model`. Only required if :obj:`task` is
                :obj:`"node"`.(default: :obj:`None`)
            batch (optional, LongTensor): Batch vector :math:`\mathbf{b} \in
                {\{ 0, \ldots, B-1\}}^N`, which assigns each node to a specific
                example. Only required if :obj:`task` is :obj:`"graph"`. All
                graphs in :attr:`batch` are used to train the explainer.
                (default: :obj:`None`).
            **kwargs (optional): Additional arguments passed to the GNN module.

        """
        assert x.shape[0] == z.shape[0]
        assert ~(batch is None and node_idxs is None)

        self.model.eval()
        self.__clear_masks__()
        self.to(x.device)
        optimizer = torch.optim.Adam(self.explainer_model.parameters(),
                                     lr=self.lr)
        self.explainer_model.train()

        # Get initial prediction.
        with torch.no_grad():
            out = self.model(x=x, edge_index=edge_index, **
                             kwargs) if batch is None else self.model(
                                 x=x, edge_index=edge_index, batch=batch, **
                                 kwargs)
            log_logits = self.__to_log_prob__(out)
            pred_label = log_logits.argmax(dim=-1)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Training Explainer')

        bias = self.coeffs['bias']
        if self.task == "graph":
            assert (batch is not None) and x.shape[0] == batch.shape[0]
            assert batch.unique().shape[0] == pred_label.shape[0]
            batch = batch.squeeze()
            explainer_in = self.__create_explainer_input__(edge_index,
                                                           z).detach()

            for e in range(0, self.epochs):
                optimizer.zero_grad()
                t = self.__get_temp__(e)
                edge_mask = self.__compute_edge_mask__(
                    self.explainer_model(explainer_in), t, bias=bias)
                self.__set_masks__(edge_mask)
                out = self.model(x=x, edge_index=edge_index, batch=batch,
                                 **kwargs)
                log_logits = self.__to_log_prob__(out)
                self.__loss__(edge_mask, log_logits, pred_label.squeeze(),
                              batch=batch, edge_index=edge_index).backward()
                optimizer.step()
                if self.log:  # pragma: no cover
                    pbar.update(1)

        else:
            assert node_idxs.unique().shape[0] == node_idxs.shape[0]
            for e in range(0, self.epochs):
                loss = torch.tensor([0.0], device=x.device).detach()
                t = self.__get_temp__(e)
                optimizer.zero_grad()

                for n in node_idxs:
                    n = int(n)
                    (x_n, z_n, edge_index_n, mapping, _,
                     kwargs_n) = self.__subgraph__(n, x, z, edge_index,
                                                   **kwargs)
                    explainer_in = self.__create_explainer_input__(
                        edge_index_n, z_n, mapping).detach()
                    edge_mask = self.__compute_edge_mask__(
                        self.explainer_model(explainer_in), t, bias=bias)
                    self.__set_masks__(edge_mask)
                    out = self.model(x=x_n, edge_index=edge_index_n,
                                     **kwargs_n)
                    log_logits = self.__to_log_prob__(out)
                    loss += self.__loss__(edge_mask, log_logits[mapping],
                                          pred_label[[n]])

                loss.backward()
                optimizer.step()

                if self.log:  # pragma: no cover
                    pbar.update(1)

        if self.log:
            pbar.close()
        self.__clear_masks__()

    def explain(self, x, z, edge_index, node_id=None, **kwargs):
        r"""Returns an :obj:`edge_mask` that explains :obj:`model` prediction.

        Args:
            x (Tensor): The node feature matrix.
            z (Tensor): Node embedding from last :class:`MessagePassing` layer.
            edge_index (LongTensor): The edge indices.
            node_id (Optional, int): The node id to explain.
                Only required if :obj:`task` is :obj:`"node"`.
            **kwargs (optional): Additional arguments passed to the GNN module.
        :rtype: :class:`Tensor`
        """
        self.explainer_model.eval()
        with torch.no_grad():
            if self.task == "graph":
                explainer_in = self.__create_explainer_input__(edge_index, z)
                edge_mask = self.__compute_edge_mask__(
                    self.explainer_model(explainer_in), training=False)
                return edge_mask.sigmoid()

            else:
                num_edges = edge_index.shape[1]
                x, z, edge_index, mapping, hop_mask, _ = self.__subgraph__(
                    node_id, x, z, edge_index, **kwargs)
                explainer_in = self.__create_explainer_input__(
                    edge_index, z, mapping)
                edge_mask = self.__compute_edge_mask__(
                    self.explainer_model(explainer_in), training=False)

                # edges outside k-hop subgraph of node_id have edge_mask=0.
                full_edge_mask = edge_mask.new_zeros(num_edges)
                full_edge_mask[hop_mask] = edge_mask.sigmoid()
                return full_edge_mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'
