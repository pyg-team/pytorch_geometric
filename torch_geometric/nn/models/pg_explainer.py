from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.functional import nll_loss
from torch_scatter import scatter_mean
from tqdm import tqdm

from torch_geometric.nn.models.explainer import (Explainer, clear_masks,
                                                 set_masks)

EPS = 1e-15


class PGExplainer(Explainer):
    r"""The PGExplainer model from the `"Parameterized Explainer for Graph Neural
    Network" <https://arxiv.org/abs/2011.04573>`_ paper. It uses a neural
    network :obj:`explainer_model` , to predict which edges are crucial to
    a GNNs node or graph prediction.

    .. note::

        For an example of using PGExplainer, see `examples/pg_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        pg_explainer.py>`_.

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
            :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model
            returns the logarithm of probabilities), :obj:`"prob"` (the
            model returns probabilities), :obj:`"raw"` (the model returns raw
            scores) and :obj:`"regression"` (the model returns scalars).
            (default: :obj:`"log_prob"`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in :attr:`~torch_geometric.nn.models.GNNExplainer.coeffs`.
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
                 log: bool = True, **kwargs):
        super().__init__(model, lr, epochs, num_hops, return_type, log)
        assert task in ['node', 'graph']

        self.out_channels = out_channels
        self.task = task
        self.coeffs.update(kwargs)

        self.exp_in_channels = 2 * out_channels if (
            task == 'graph') else 3 * out_channels
        self.explainer_model = nn.Sequential(
            nn.Linear(self.exp_in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _get_temp(self, e: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * ((temp[1] / temp[0])**(e / self.epochs))

    def _create_explainer_input(self, edge_index, x, node_idx=None) -> Tensor:
        rows, cols = edge_index
        x_j, x_i = x[rows], x[cols]
        if self.task == 'node':
            x_node = x[node_idx].repeat(rows.size(0), 1)
            return torch.cat([x_i, x_j, x_node], 1)
        else:
            return torch.cat([x_i, x_j], 1)

    def _compute_edge_mask(self, edge_weight, temperature=1.0, bias=0.0,
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

    def _loss(self, log_logits, prediction, node_idx=None, batch=None,
              edge_index=None):
        if self.return_type == 'regression':
            if node_idx is not None:
                loss = torch.cdist(log_logits[node_idx],
                                   prediction[node_idx]).squeeze()
            else:
                loss = torch.cdist(log_logits, prediction).mean()
        else:
            if node_idx is not None:
                loss = nll_loss(log_logits[node_idx], prediction[node_idx])
            else:
                loss = nll_loss(log_logits, prediction, reduction='sum')
        mask = self.edge_mask.sigmoid().squeeze()

        # Regularization losses
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask_ent_reg = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent_reg.mean() if batch is None else scatter_mean(
            mask_ent_reg, batch[edge_index[0]]).sum()
        mask_ent_loss *= self.coeffs['edge_ent']

        return loss + size_loss + mask_ent_loss

    def train_explainer(self, x: Tensor, z: Tensor, edge_index: Tensor,
                        node_idxs: Optional[LongTensor] = None,
                        batch: Tensor = None, **kwargs):
        r"""Trains the :obj:`explainer_model` to predict an
        edge mask that is crucial to explain the predictions of
        the :obj:`model`.

        Args:
            x (Tensor): The node feature matrix.
            z (Tensor): Node embedding from last :class:`MessagePassing` layer
                in :obj:`model`.
            edge_index (LongTensor): The edge indices.
            node_idxs (Optional, LongTensor): The nodes used for training.
                Only required if :obj:`task` is :obj:`"node"`.
                (default: :obj:`None`)
            batch (optional, LongTensor): Batch vector :math:`\mathbf{b} \in
                {\{ 0, \ldots, B-1\}}^N`, which assigns each node to a specific
                example. Only required if :obj:`task` is :obj:`"graph"`. All
                graphs in :attr:`batch` are used for training.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to the GNN module.

        """
        assert x.shape[0] == z.shape[0]
        assert ~(batch is None and node_idxs is None)

        self.model.eval()
        clear_masks(self.model)
        self.to(x.device)
        optimizer = torch.optim.Adam(self.explainer_model.parameters(),
                                     lr=self.lr)
        self.explainer_model.train()

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Training Explainer')

        bias = self.coeffs['bias']
        if self.task == "graph":
            # Get initial prediction.
            prediction = self.get_initial_prediction(x, edge_index, batch,
                                                     **kwargs)

            assert (batch is not None) and x.shape[0] == batch.shape[0]
            assert batch.unique().shape[0] == prediction.shape[0]
            batch = batch.squeeze()
            explainer_in = self._create_explainer_input(edge_index, z).detach()

            for e in range(0, self.epochs):
                optimizer.zero_grad()
                t = self._get_temp(e)
                self.edge_mask = self._compute_edge_mask(
                    self.explainer_model(explainer_in), t, bias=bias)
                set_masks(self.model, self.edge_mask, edge_index)
                out = self.model(x=x, edge_index=edge_index, batch=batch,
                                 **kwargs)
                self.get_loss(out, prediction.squeeze(), node_idx=None,
                              batch=batch, edge_index=edge_index).backward()
                optimizer.step()
                if self.log:  # pragma: no cover
                    pbar.update(1)

        else:
            assert node_idxs.unique().shape[0] == node_idxs.shape[0]
            for e in range(0, self.epochs):
                loss = torch.tensor([0.0], device=x.device).detach()
                t = self._get_temp(e)
                optimizer.zero_grad()

                for n in node_idxs:
                    n = int(n)
                    kwargs['z'] = z
                    (x_n, edge_index_n, mapping, _, _,
                     kwargs_n) = self.subgraph(n, x, edge_index, **kwargs)
                    z_n = kwargs_n.pop('z')
                    prediction = self.get_initial_prediction(
                        x_n, edge_index_n, **kwargs_n)

                    explainer_in = self._create_explainer_input(
                        edge_index_n, z_n, mapping).detach()
                    self.edge_mask = self._compute_edge_mask(
                        self.explainer_model(explainer_in), t, bias=bias)
                    # Should this be edge_index_n or edge_index?
                    set_masks(self.model, self.edge_mask, edge_index_n)
                    out = self.model(x=x_n, edge_index=edge_index_n,
                                     **kwargs_n)
                    loss += self.get_loss(out, prediction, mapping)
                    clear_masks(self.model)

                loss.backward()
                optimizer.step()

                if self.log:  # pragma: no cover
                    pbar.update(1)

        if self.log:
            pbar.close()
        clear_masks(self.model)

    def explain(self, x: Tensor, z: Tensor, edge_index: Tensor,
                node_idx: Optional[int] = None, **kwargs) -> Tensor:
        r"""Returns an :obj:`edge_mask` that explains :obj:`model` prediction.

        Args:
            x (Tensor): The node feature matrix.
            z (Tensor): Node embedding from last :class:`MessagePassing` layer.
            edge_index (LongTensor): The edge indices.
            node_idx (Optional, int): The node id to explain.
                Only required if :obj:`task` is :obj:`"node"`.
            **kwargs (optional): Additional arguments passed to the GNN module.
        :rtype: :class:`Tensor`
        """
        self.explainer_model.eval()
        with torch.no_grad():
            if self.task == "graph":
                explainer_in = self._create_explainer_input(edge_index, z)
                self.edge_mask = self._compute_edge_mask(
                    self.explainer_model(explainer_in), training=False)
                return self.edge_mask.sigmoid()

            else:
                num_edges = edge_index.shape[1]
                kwargs['z'] = z
                (x, edge_index, mapping, hop_mask, _,
                 kwargs_n) = self.subgraph(node_idx, x, edge_index, **kwargs)
                z = kwargs_n.pop('z')
                explainer_in = self._create_explainer_input(
                    edge_index, z, mapping)
                self.edge_mask = self._compute_edge_mask(
                    self.explainer_model(explainer_in), training=False)

                # edges outside k-hop subgraph of node_id have edge_mask=0.
                full_edge_mask = self.edge_mask.new_zeros(num_edges)
                full_edge_mask[hop_mask] = self.edge_mask.sigmoid()
                return full_edge_mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'
