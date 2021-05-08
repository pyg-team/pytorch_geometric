from typing import Optional

import torch
from torch import nn
from tqdm import tqdm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch.nn.functional import nll_loss
from torch_scatter import scatter_mean
EPS = 1e-15


class PGExplainer(torch.nn.Module):
    r"""The PGExplainer model from the `"Parameterized Explainer for Graph Neural
    Network" <https://arxiv.org/abs/2011.04573>`_ paper for identifying compact
    subgraph structures that play a crucial role in a GNN’s node/graph
    predictions.

    .. note::

        For an example of using PGExplainer, see `<>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        out_channels(int): Size of output of last embedding layer in
            :obj:`model`.
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
        task (str): Denotes type of task that needs explanation. Valid inputs
            are :obj:`"node"` (for node classification) and :obj:`"graph"` (for
            graph classification). (default: :obj:`"node"`)
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
                 lr: float = 0.004, num_hops: Optional[int] = None,
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
        self.edge_mask = None
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

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

        return x, z, edge_index, mapping, kwargs_new

    def __create_explainer_input__(self, edge_index, x, node_id=None):
        r"""Given the embeddign of the sample by the :obj:`model`, this method
        construct the input to the :obj:`explainer_model`.
        """

        rows = edge_index[0]
        cols = edge_index[1]
        x_i = x[rows]
        x_j = x[cols]
        if self.task == 'node':
            x_node = x[node_id].repeat(rows.size(0), 1)
            explainer_input = torch.cat([x_i, x_j, x_node], 1)
        else:
            explainer_input = torch.cat([x_i, x_j], 1)
        return explainer_input

    def __set_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __compute_edge_mask__(self, edge_weight, temperature=1.0, bias=0.0,
                              training=True):
        r"""Compute :obj:`self.edge_mask` (edge probabilities)
        from :attr:`edge_weight`.
        """

        if training:
            bias = bias + 0.0001
            eps = (bias -
                   (1 - bias)) * torch.rand(edge_weight.size()) + (1 - bias)
            eps = eps.to(edge_weight.device)
            self.edge_mask = (eps.log() - (1 - eps).log() +
                              edge_weight).squeeze() / temperature

        else:
            self.edge_mask = edge_weight.squeeze()

    def __loss__(self, log_logits, pred_label, batch=None, edge_index=None):
        cross_ent_loss = nll_loss(log_logits, pred_label, reduction='sum')
        mask = self.edge_mask.sigmoid().squeeze()

        # Regularization losses
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask_ent_reg = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent_reg.mean() if batch is None else scatter_mean(
            mask_ent_reg, batch[edge_index[0]]).sum()
        mask_ent_loss *= self.coeffs['edge_ent']

        return cross_ent_loss + size_loss + mask_ent_loss

    def __to_log_prob__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.log_softmax(dim=-1) if self.return_type == 'raw' else x
        x = x.log() if self.return_type == 'prob' else x
        return x

    def __get_temp__(self, e: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * ((temp[1] / temp[0])**(e / self.epochs))

    def train_explainer(self, index, x, z, edge_index, batch=None, **kwargs):
        r"""Trains a fully connected network :obj:`self.explainer_model` to predict an
        edge mask that is important to explain the predictions made by
        :obj:`self.model`. The nodes/graphs in :attr:`index` are used for
        training.

        Args:
            index (LongTensor): The nodes/graphs used to train explainer.
            x (Tensor): The node feature matrix.
            z (Tensor): Node embedding from last :class:`MessagePassing` layer.
            edge_index (LongTensor): The edge indices.
            batch (optional, LongTensor): Tensor denotes the graph each node
                belongs to. Only required if :obj:`task`=:obj:`"graph"`.
            **kwargs (optional): Additional arguments passed to the GNN module.

        """
        assert x.shape[0] == z.shape[0]
        self.model.eval()
        self.__clear_masks__()
        self.to(x.device)
        optimizer = torch.optim.Adam(self.explainer_model.parameters(),
                                     lr=self.lr)
        self.explainer_model.train()
        index, _ = index.sort()
        index = index.to(x.device)

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
            pbar.set_description(f'Explain indicies {index}')

        bias = self.coeffs['bias']
        if self.task == "graph":
            assert (batch is not None) and x.shape[0] == batch.shape[0]
            assert batch.unique().shape[0] == pred_label.shape[0]
            batch = batch.squeeze()

            train = torch.tensor([i in index for i in batch]).to(
                x.device)  # graph indicies in training.
            x = x[train]
            z = z[train]
            if kwargs.get('edge_attr', None) is None:
                edge_index, _ = subgraph(train, edge_index, relabel_nodes=True)
            else:
                edge_index, kwargs['edge_attr'] = subgraph(
                    train, edge_index, kwargs['edge_attr'], relabel_nodes=True)
            pred_label = pred_label[index]

            # creating new batch tensor, for graphs in training.
            batch = torch.tensor([
                torch.searchsorted(index, i) for i in batch if i in index
            ]).to(x.device)
            explainer_in = self.__create_explainer_input__(edge_index,
                                                           z).detach()

            for e in range(0, self.epochs):
                optimizer.zero_grad()
                t = self.__get_temp__(e)
                self.__compute_edge_mask__(self.explainer_model(explainer_in),
                                           t, bias=bias)
                self.__set_masks__()
                out = self.model(x=x, edge_index=edge_index, batch=batch,
                                 **kwargs)
                log_logits = self.__to_log_prob__(out)
                self.__loss__(log_logits, pred_label.squeeze(), batch=batch,
                              edge_index=edge_index).backward()
                optimizer.step()
                # print(self.__loss__(log_logits, pred_label.squeeze(),
                # batch=batch,
                # edge_index=edge_index))
                if self.log:  # pragma: no cover
                    pbar.update(1)

        else:
            for e in range(0, self.epochs):
                loss = torch.tensor([0.0], device=x.device).detach()
                t = self.__get_temp__(e)
                for n in index:
                    n = int(n)
                    optimizer.zero_grad()
                    x_n, z_n, edge_index_n, mapping, kwargs_n = self.__subgraph__(
                        n, x, z, edge_index, **kwargs)
                    explainer_in = self.__create_explainer_input__(
                        edge_index_n, z_n, mapping).detach()
                    self.__compute_edge_mask__(
                        self.explainer_model(explainer_in), t, bias=bias)
                    self.__set_masks__()
                    out = self.model(x=x_n, edge_index=edge_index_n,
                                     **kwargs_n)
                    log_logits = self.__to_log_prob__(out)
                    loss += self.__loss__(log_logits[mapping], pred_label[[n]])
                # print(loss)
                loss.backward()
                optimizer.step()

                if self.log:  # pragma: no cover
                    pbar.update(1)

        if self.log:
            pbar.close()
        self.__clear_masks__()

    def explain(self, index, x, edge_index, batch=None):
        r"""Get :obj:`edge_mask` for node/graph in index.

        Args:
            index (int): The node/graph id to explain.
            edge_index (LongTensor): The edge indices.
            x (Tensor): The edge mask.
            batch (LongTensor):

        :rtype: :class:`Tensor`
        """
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}()'
