import abc
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import to_undirected
import torch_sparse
from tqdm import tqdm


LOSS_TYPE = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]


def margin(score: Tensor, labels: Tensor,
           idx_mask: Optional[Tensor] = None) -> Tensor:
    r"""Calculate margin between true score and highest non-target score:

    .. math::
        $m = - s_{y} + max_{y' \ne y} s_{y'}$

    where :math:`m` is the margin `s` the score and `y` the labels.

    Args:
        score (Tensor): Some score (e.g. logits) of shape :obj:`[n_elem, dim]`.
        labels (LongTensor): The labels of shape :obj:`[n_elem]`.
        idx_mask (Tensor, optional): To select subset of `score` and `labels`
            of shape :obj:`[n_elem]`. Defaults to None.

    :rtype: (Tensor)
    """
    if idx_mask is not None:
        score = score[idx_mask]
        labels = labels[idx_mask]

    linear_idx = torch.arange(score.size(0), device=score.device)
    true_score = score[linear_idx, labels]

    score = score.clone()
    score[linear_idx, labels] = float('-Inf')
    best_non_target_score = score.amax(dim=-1)

    margin_ = best_non_target_score - true_score
    return margin_


def tanh_margin_loss(prediction: Tensor, labels: Tensor,
                     idx_mask: Optional[Tensor] = None) -> Tensor:
    """Calculate tanh margin loss, a node-classification loss that focuses 
    on nodes next to decision boundary.

    Args:
        prediction (Tensor): Prediction of shape :obj:`[n_elem, dim]`.
        labels (LongTensor): The labels of shape :obj:`[n_elem]`.
        idx_mask (Tensor, optional): To select subset of `score` and `labels`
            of shape :obj:`[n_elem]`. Defaults to None.

    :rtype: (Tensor)
    """
    log_logits = F.log_softmax(prediction, dim=-1)
    margin_ = margin(log_logits, labels, idx_mask)
    loss = torch.tanh(margin_).mean()
    return loss


def probability_margin_loss(prediction: Tensor, labels: Tensor,
                            idx_mask: Optional[Tensor] = None) -> Tensor:
    """Calculate probability margin loss, a node-classification loss that 
    focuses  on nodes next to decision boundary. See `Are Defenses for Graph
    Neural  Networks Robust? 
    <https://www.cs.cit.tum.de/daml/are-gnn-defenses-robust/>` for details.

    Args:
        prediction (Tensor): Prediction of shape :obj:`[n_elem, dim]`.
        labels (LongTensor): The labels of shape :obj:`[n_elem]`.
        idx_mask (Tensor, optional): To select subset of `score` and `labels`
            of shape :obj:`[n_elem]`. Defaults to None.

    :rtype: (Tensor)
    """
    logits = F.softmax(prediction, dim=-1)
    margin_ = margin(logits, labels, idx_mask)
    return margin_.mean()


def masked_cross_entropy(log_logits: Tensor, labels: Tensor,
                         idx_mask: Optional[Tensor] = None) -> Tensor:
    """Calculate masked cross entropy loss, a node-classification loss that
    focuses on nodes next to decision boundary.

    Args:
        log_logits (Tensor): Log logits of shape :obj:`[n_elem, dim]`.
        labels (LongTensor): The labels of shape :obj:`[n_elem]`.
        idx_mask (Tensor, optional): To select subset of `score` and `labels`
            of shape :obj:`[n_elem]`. Defaults to None.

    :rtype: (Tensor)
    """
    if idx_mask is not None:
        log_logits = log_logits[idx_mask]
        labels = labels[idx_mask]

    not_flipped = log_logits.argmax(-1) == labels
    loss = F.cross_entropy(log_logits[not_flipped], labels[not_flipped])
    return loss


class Attack(torch.nn.Module):
    """Abstract class for an adversarial attack that perturbs edges:

    aims to answer the question what (small) perturbation changes the model's 
    prediction the most."""

    @abc.abstractmethod
    def attack(self, x: Tensor, edge_index: Tensor, labels: Tensor,
               budget: int, idx_attack: Optional[Tensor] = None,
               **kwargs) -> Tuple[Tensor, Tensor]:
        """Attack the predictions for the provided model and graph.

        A subset of predictions may be specified with :attr:`idx_attack`. The 
        attack is allowed to flip (i.e. add or delete) :attr:`budget` edges and
        will return the strongest perturbation it can find. It returns both the
        resulting perturbed  :attr:`edge_index` as well as the perturbations.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            labels (Tensor): The labels.
            budget (int): The number of allowed perturbations (i.e. 
                number of edges that are flipped at most).
            idx_attack (Tensor, optional): Filter for predictions/labels. 
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        raise NotImplementedError('Abstractmethod needs to be implemented')

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)


class RBCDAttack(Attack):
    """Abstract class for an adversarial attack for the sparse greedy/projected
    gradient attack based on Randomized Block Coordinate Descent (RBCD) from
    the `Robustness of Graph Neural Networks at Scale 
    <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale/>` paper. Both
    attacks use an efficient gradient based approach that (during the attack)
    relaxed the discrete entries in the adjacency matrix :math:`\{0, 1\}`
    to :math:`[0, 1]`. Thus, they support all models that can handle weighted
    graphs are are differentiabile w.r.t. these edge weights. The memory
    overhead is driven by the additional edges (at most :attr:`block_size`).

    .. note::
        For examples of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        rbcd_attack.py>`.

    Args:
        model (torch.nn.Module): The GNN module to assess.
        block_size (int, optional): Number of randomly selected elements in the
            adjacency matrix to consider. (default: :obj:`1_000_000`)
        loss (LOSS_TYPE, optional): A loss to measure the "strength" of an
            attack.  Note that this function must match the output format of 
            :attr:`model`. By default, it is assumed that the task is
            classification, that the model returns raw predictions (i.e. no
            output activation) or uses :obj:`logsoftmax`, and that the number 
            of predictions matches the number labels passed to :attr:`attack`. 
            (default: :obj:`probability_margin_loss`)
        is_undirected_graph (bool, optional): If :obj:`True` the graph is 
            assumed to be undirected. (default: :obj:`True`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """
    coeffs = {
        'max_final_samples': 20,
        'max_trials_sampling': 20,
        'eps': 1e-7
    }

    def __init__(self,
                 model: torch.nn.Module,
                 block_size: int = 1_000_000,
                 # Target class has lowest score
                 loss: LOSS_TYPE = probability_margin_loss,
                 is_undirected_graph: bool = True,
                 log: bool = True,
                 **kwargs) -> None:
        super().__init__()

        self.model = model
        self.block_size = block_size
        self.loss = loss
        self.is_undirected_graph = is_undirected_graph
        self.log = log

        self.coeffs.update(kwargs)

        # Need to contain the perturbations (shape [2, # perts.])
        self.flipped_edges = None
        # A sequence the attack is iteration over (values are passed to `step`)
        self.step_sequence = tuple()

    def attack(self, x: Tensor, edge_index: Tensor, labels: Tensor,
               budget: int, idx_attack: Optional[Tensor] = None,
               **kwargs) -> Tuple[Tensor, Tensor]:
        """Attack the predictions for the provided model and graph.

        A subset of predictions may be specified with :attr:`idx_attack`. The 
        attack is allowed to flip (i.e. add or delete) :attr:`budget` edges and
        will return the strongest perturbation it can find. It returns both the
        resulting perturbed  :attr:`edge_index` as well as the perturbations.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            labels (Tensor): The labels.
            budget (int): The number of allowed perturbations (i.e. 
                number of edges that are flipped at most).
            idx_attack (Tensor, optional): Filter for predictions/labels. 
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        assert self.block_size > budget, (
            f'The search space size ({self.block_size}) must be '
            f'greater than the number of permutations ({budget})')

        self.model.eval()

        self.device = x.device
        assert kwargs.get('edge_weight', None) is None, \
            '`edge_weight` is not supported'
        edge_weight = torch.ones(edge_index.size(1), device=self.device)
        self.edge_index = edge_index.cpu()
        self.edge_weight = edge_weight.cpu()
        self.n = x.size(0)

        # For collecting attack statistics
        self.attack_statistics = defaultdict(list)

        # Prepare attack and define `self.iterable` to iterate over
        self.prepare(x, edge_index, labels, budget, idx_attack, **kwargs)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=len(self.step_sequence))
            pbar.set_description('Attack model')

        # Loop over the epochs (Algorithm 1, line 5)
        for step in self.step_sequence:
            scalars = self.step(step, x, edge_index, labels,
                                budget, idx_attack, **kwargs)
            self._append_statistics(scalars)

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        ret = self.close(x, edge_index, labels, budget, idx_attack, **kwargs)

        assert self.flipped_edges.shape[1] <= budget, (
            f'# perturbed edges {self.flipped_edges.shape[1]} '
            f'exceeds budget {budget}')

        return ret

    @abc.abstractmethod
    def prepare(self, x: Tensor, edge_index: Tensor, labels: Tensor,
                budget: int, idx_attack: Optional[Tensor] = None, **kwargs):
        """Prepare attack."""
        pass

    @abc.abstractmethod
    def step(self, step: Any, x: Tensor, edge_index: Tensor, labels: Tensor,
             budget: int, idx_attack: Optional[Tensor] = None,
             **kwargs) -> Dict[str, Any]:
        """Step attack. Returned dict is added to statistics."""
        pass

    @abc.abstractmethod
    def close(self, x: Tensor, edge_index: Tensor, labels: Tensor,
              budget: int, idx_attack: Optional[Tensor] = None,
              **kwargs) -> Tuple[Tensor, Tensor]:
        """Clean up and prepare return argument."""
        pass

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                **kwargs) -> Tensor:
        """Forward model."""
        # return self.model(x, (edge_index, edge_weight), **kwargs)
        adj = torch_sparse.SparseTensor.from_edge_index(
            edge_index, edge_weight, (self.n, self.n), is_sorted=True)
        return self.model(x, adj, **kwargs)

    def _sample_random_block(self, budget: int = 0) -> None:
        for _ in range(self.coeffs['max_trials_sampling']):
            self.current_block = torch.randint(
                RBCDAttack.num_possible_edges(
                    self.n, self.is_undirected_graph),
                (self.block_size,), device=self.device)
            self.current_block = torch.unique(self.current_block, sorted=True)
            if self.is_undirected_graph:
                self.block_edge_index = RBCDAttack.linear_to_triu_idx(
                    self.n, self.current_block)
            else:
                self.block_edge_index = RBCDAttack.linear_to_full_idx(
                    self.n, self.current_block)
                self._filter_self_loops(with_weight=False)

            self.block_edge_weight = torch.full_like(
                self.current_block, self.coeffs['eps'], dtype=torch.float32)
            if self.current_block.size(0) >= budget:
                return
        raise RuntimeError(
            'Sampling random block was not successful. '
            'Please decrease `budget`.')

    def _resample_random_block(self, budget: int) -> None:
        # Keep at most half of the block (i.e. resample low weights)
        sorted_idx = torch.argsort(self.block_edge_weight)
        keep_above = (self.block_edge_weight <=
                      self.coeffs['eps']).sum().long()
        if keep_above < sorted_idx.size(0) // 2:
            keep_above = sorted_idx.size(0) // 2
        sorted_idx = sorted_idx[keep_above:]

        self.current_block = self.current_block[sorted_idx]
        self.block_edge_index = self.block_edge_index[:, sorted_idx]
        self.block_edge_weight = self.block_edge_weight[sorted_idx]

        # Sample until enough edges were drawn
        for _ in range(self.coeffs['max_trials_sampling']):
            n_edges_resample = self.block_size - self.current_block.size(0)
            lin_index = torch.randint(
                RBCDAttack.num_possible_edges(
                    self.n, self.is_undirected_graph),
                (n_edges_resample,), device=self.device)

            current_block = torch.cat((self.current_block, lin_index))
            self.current_block, unique_idx = torch.unique(
                current_block, sorted=True, return_inverse=True)

            if self.is_undirected_graph:
                self.block_edge_index = RBCDAttack.linear_to_triu_idx(
                    self.n, self.current_block)
            else:
                self.block_edge_index = RBCDAttack.linear_to_full_idx(
                    self.n, self.current_block)

            # Merge existing weights with new edge weights
            block_edge_weight_prev = self.block_edge_weight.clone()
            self.block_edge_weight = torch.full(
                self.current_block.shape, self.coeffs['eps'],
                device=self.device)
            self.block_edge_weight[
                unique_idx[:sorted_idx.size(0)]] = block_edge_weight_prev

            if not self.is_undirected_graph:
                self._filter_self_loops(with_weight=True)

            if self.current_block.size(0) > budget:
                return
        raise RuntimeError(
            'Sampling random block was not successful.'
            'Please decrease `budget`.')

    def _get_modified_adj(self) -> Tuple[Tensor, Tensor]:
        if self.is_undirected_graph:
            modified_edge_idx, modified_edge_weight = to_undirected(
                self.block_edge_index, self.block_edge_weight,
                self.n, reduce='mean')
        else:
            modified_edge_idx = self.block_edge_index
            modified_edge_weight = self.block_edge_weight
        edge_index = torch.cat(
            (self.edge_index.to(self.device), modified_edge_idx), dim=-1)
        edge_weight = torch.cat(
            (self.edge_weight.to(self.device), modified_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(
            edge_index, edge_weight, m=self.n, n=self.n, op='sum')

        # Allow (soft) removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]

        return edge_index, edge_weight

    def _append_statistics(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.attack_statistics[key].append(value)

    @staticmethod
    def num_possible_edges(n: int, is_undirected_graph: bool) -> int:
        """Determine number of possible edges for graph."""
        if is_undirected_graph:
            return n * (n - 1) // 2
        else:
            return int(n ** 2)  # We filter self-loops later

    @staticmethod
    def linear_to_triu_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Convert a linear index to index of upper triangular matrix."""
        row_idx = (
            n
            - 2
            - torch.floor(torch.sqrt(-8 * lin_idx.double() +
                                     4 * n * (n - 1) - 7) / 2.0 - 0.5)
        ).long()
        col_idx = (
            lin_idx
            + row_idx
            + 1 - n * (n - 1) // 2
            + torch.div((n - row_idx) * ((n - row_idx) - 1),
                        2, rounding_mode='floor')
        )
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def linear_to_full_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Convert a linear index to index of matrix."""
        row_idx = lin_idx // n
        col_idx = lin_idx % n
        return torch.stack((row_idx, col_idx))


class GRBCDAttack(RBCDAttack):
    """Greedy Randomized Block Coordinate Descent (GRBCD) from the `Robustness 
    of Graph Neural Networks at Scale 
    <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale/>` paper.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 block_size: int = 1_000_000,
                 # Target class has lowest score
                 loss: LOSS_TYPE = masked_cross_entropy,
                 is_undirected_graph: bool = True,
                 epochs: int = 100,
                 log: bool = True,
                 **kwargs):
        super().__init__(
            model, block_size, loss, is_undirected_graph, log, **kwargs)
        self.epochs = epochs

    def prepare(self, x: Tensor, edge_index: Tensor, labels: Tensor,
                budget: int, idx_attack: Optional[Tensor] = None,
                **kwargs) -> None:
        """Prepare attack."""
        self.flipped_edges = torch.empty(
            (2, 0), dtype=edge_index.dtype, device=self.device)

        # Determine the number of edges to be flipped in each attach step / epoch
        step_size = budget // self.epochs
        if step_size > 0:
            steps = self.epochs * [step_size]
            for i in range(budget % self.epochs):
                steps[i] += 1
        else:
            steps = [1] * budget

        self.step_sequence = steps

    def step(self, step_size: int, x: Tensor, edge_index: Tensor,
             labels: Tensor, budget: int, idx_attack: Optional[Tensor] = None,
             **kwargs) -> Dict[str, Any]:
        """Step attack. Returned dict is added to statistics."""
        # Sample initial search space (Algorithm 2, line 3-4)
        self._sample_random_block(step_size)
        self.block_edge_weight.requires_grad = True

        # Retrieve sparse perturbed adjacency matrix `A \oplus p_{t-1}`
        # (Algorithm 2, line 7)
        edge_index, edge_weight = self._get_modified_adj()

        # Get predictions (Algorithm 2, line 7)
        predictions = self.forward(x, edge_index, edge_weight, **kwargs)
        # Calculate loss for attacked nodes (Algorithm 2, line 8)
        loss = self.loss(predictions, labels, idx_attack)
        # Retrieve gradient towards the current block (Algorithm 2, line 8)
        gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]

        # Greedy update of edges (Algorithm 2, line 8)
        self._greedy_update(step_size, gradient)

        return dict(loss=loss.item())

    def close(self, *args, **kwargs) -> Any:
        """Clean up and prepare return argument."""
        return self.edge_index, self.flipped_edges

    @torch.no_grad()
    def _greedy_update(self, step_size: int, gradient: torch.Tensor):
        _, topk_edge_index = torch.topk(gradient, step_size)

        add_edge_index = self.block_edge_index[:, topk_edge_index]
        add_edge_weight = torch.ones_like(
            add_edge_index[0], dtype=torch.float32)

        self.flipped_edges = torch.cat(
            (self.flipped_edges, add_edge_index), axis=-1)

        if self.is_undirected_graph:
            add_edge_index, add_edge_weight = to_undirected(
                add_edge_index, add_edge_weight, self.n, reduce='mean')
        edge_index = torch.cat(
            (self.edge_index.to(self.device), add_edge_index.to(self.device)),
            dim=-1)
        edge_weight = torch.cat(
            (self.edge_weight.to(self.device), add_edge_weight.to(self.device)))
        edge_index, edge_weight = torch_sparse.coalesce(
            edge_index, edge_weight, m=self.n, n=self.n, op='sum'
        )

        is_one_mask = torch.isclose(edge_weight, torch.tensor(1.))
        self.edge_index = edge_index[:, is_one_mask]
        self.edge_weight = edge_weight[is_one_mask]
        # self.edge_weight = torch.ones_like(self.edge_weight)
        assert self.edge_index.size(1) == self.edge_weight.size(0)


class PRBCDAttack(RBCDAttack):
    """Projected Randomized Block Coordinate Descent (PRBCD) from the
    `Robustness of Graph Neural Networks at Scale
    <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale/>` paper.
    """

    coeffs = {
        'max_final_samples': 20,
        'max_trials_sampling': 20,
        'with_early_stopping': True,
        'eps': 1e-7
    }

    def __init__(self,
                 model: torch.nn.Module,
                 block_size: int = 1_000_000,
                 epochs_resampling: int = 100,
                 epochs_finetuning: int = 25,
                 # Target class has lowest score
                 loss: LOSS_TYPE = probability_margin_loss,
                 metric: LOSS_TYPE = probability_margin_loss,
                 lr: float = 100,
                 is_undirected_graph: bool = True,
                 log: bool = True,
                 **kwargs) -> None:
        super().__init__(
            model, block_size, loss, is_undirected_graph, log, **kwargs)

        self.epochs_resampling = epochs_resampling
        self.epochs = epochs_resampling + epochs_finetuning
        if metric is not None:
            self.metric = metric
        else:
            self.metric = loss
        self.lr = lr

        self.coeffs.update(kwargs)

        self.budget = 0

    def prepare(self, x: Tensor, edge_index: Tensor, labels: Tensor,
                budget: int, idx_attack: Optional[Tensor] = None, **kwargs):
        """Prepare attack."""
        self.step_sequence = list(range(self.epochs))

        # For early stopping (not explicitly covered by pseudo code)
        self.best_metric = float('-Inf')

        # Sample initial search space (Algorithm 1, line 3-4)
        self._sample_random_block(budget)

    def step(self, epoch: int, x: Tensor, edge_index: Tensor, labels: Tensor,
             budget: int, idx_attack: Optional[Tensor] = None,
             **kwargs) -> Dict[str, Any]:
        """Step attack. Returned dict is added to statistics."""
        self.block_edge_weight.requires_grad = True

        # Retrieve sparse perturbed adjacency matrix `A \oplus p_{t-1}`
        # (Algorithm 1, line 6)
        edge_index, edge_weight = self._get_modified_adj()

        # Get prediction (Algorithm 1, line 6)
        prediction = self.forward(x, edge_index, edge_weight, **kwargs)
        # Calculate loss combining all each node (Algorithm 1, line 7)
        loss = self.loss(prediction, labels, idx_attack)
        # Retrieve gradient towards the current block (Algorithm 1, line 7)
        gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]

        with torch.no_grad():
            # Gradient update step (Algorithm 1, line 7)
            edge_weight = self._update_edge_weights(
                budget, epoch, gradient)[1]
            # For monitoring
            pmass_update = torch.clamp(
                self.block_edge_weight, 0, 1).sum().item()
            # Projection to stay within relaxed `L_0` budget
            # (Algorithm 1, line 8)
            self.block_edge_weight = PRBCDAttack.project(
                budget, self.block_edge_weight, self.coeffs['eps'])
            # For monitoring
            pmass_projected = self.block_edge_weight.sum().item()

            # Calculate metric after the current epoch (overhead
            # for monitoring and early stopping)
            edge_index, edge_weight = self._get_modified_adj()
            prediction = self.forward(x, edge_index, edge_weight, **kwargs)
            metric = self.metric(prediction, labels, idx_attack)
            del edge_index, edge_weight, prediction

            # Save best epoch for early stopping
            # (not explicitly covered by pseudo code)
            if self.coeffs['with_early_stopping'] and self.best_metric < metric:
                self.best_metric = metric
                self.best_block = self.current_block.cpu()
                self.best_edge_index = self.block_edge_index.cpu()
                best_pert_edge_weight = self.block_edge_weight.detach()
                self.best_pert_edge_weight = best_pert_edge_weight.cpu()

            # Resampling of search space (Algorithm 1, line 9-14)
            if epoch < self.epochs_resampling - 1:
                self._resample_random_block(budget)
            elif (self.coeffs['with_early_stopping']
                    and epoch == self.epochs_resampling - 1):
                # Retrieve best epoch if early stopping is active
                # (not explicitly covered by pseudo code)
                self.current_block = self.best_block.to(self.device)
                self.block_edge_index = self.best_edge_index.to(self.device)
                self.block_edge_weight = self.best_pert_edge_weight.to(
                    self.device)

        return dict(loss=loss.item(),
                    metric=metric.item(),
                    prob_mass_after_update=pmass_update,
                    prob_mass_after_projection=pmass_projected)

    def close(self, x: Tensor, edge_index: Tensor, labels: Tensor,
              budget: int, idx_attack: Optional[Tensor] = None,
              **kwargs) -> Any:
        """Clean up and prepare return argument."""
        # Retrieve best epoch if early stopping is active
        # (not explicitly covered by pseudo code)
        if self.coeffs['with_early_stopping']:
            self.current_block = self.best_block.to(self.device)
            self.block_edge_index = self.best_edge_index.to(self.device)
            self.block_edge_weight = self.best_pert_edge_weight.to(self.device)

        # Sample final discrete graph (Algorithm 1, line 16)
        edge_index, flipped_edges = self._sample_final_edges(
            x, labels, budget, idx_attack=idx_attack, **kwargs)

        return edge_index, flipped_edges

    @torch.no_grad()
    def _sample_final_edges(self, x: Tensor, labels: Tensor, budget: int,
                            idx_attack: Optional[Tensor] = None,
                            **kwargs) -> Tuple[Tensor, Tensor]:
        best_metric = float('-Inf')
        block_edge_weight = self.block_edge_weight
        block_edge_weight[block_edge_weight <= self.coeffs['eps']] = 0

        for i in range(self.coeffs['max_final_samples']):
            if i == 0:
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(block_edge_weight)
                sampled_edges[torch.topk(
                    block_edge_weight, budget).indices] = 1
            else:
                sampled_edges = torch.bernoulli(block_edge_weight).float()

            if sampled_edges.sum() > budget:
                # Allowed budget is exceeded
                continue
            self.block_edge_weight = sampled_edges

            edge_index, edge_weight = self._get_modified_adj()
            prediction = self.forward(x, edge_index, edge_weight, **kwargs)
            metric = self.metric(prediction, labels, idx_attack)

            # Save best sample
            if best_metric < metric:
                best_metric = metric
                best_edge_weight = self.block_edge_weight.clone().cpu()

        # Recover best sample
        self.block_edge_weight = best_edge_weight.to(self.device)
        self.flipped_edges = self.block_edge_index[
            :, torch.where(best_edge_weight)[0]]

        edge_index, edge_weight = self._get_modified_adj()
        edge_mask = edge_weight == 1

        assert self.flipped_edges.shape[1] <= budget, (
            f'# perturbed edges {self.flipped_edges.shape[1]} '
            f'exceeds budget {budget}')

        self.edge_index = edge_index[:, edge_mask]
        self.edge_weight = edge_weight[edge_mask]

        return self.edge_index, self.flipped_edges

    def _update_edge_weights(self, budget: int, epoch: int,
                             gradient: Tensor) -> Tuple[Tensor, Tensor]:
        # The learning rate is refined heuristically, s.t. (1) it is
        # independent of the number of perturbations (assuming an undirected
        # adjacency matrix) and (2) to decay learning rate during fine-tuning
        # (i.e. fixed search space).
        lr = (budget / self.n * self.lr
              / np.sqrt(max(0, epoch - self.epochs_resampling) + 1))
        self.block_edge_weight.data.add_(lr * gradient)

        return self._get_modified_adj()

    def _filter_self_loops(self, with_weight: bool):
        is_not_sl = self.block_edge_index[0] != self.block_edge_index[1]
        self.current_block = self.current_block[is_not_sl]
        self.block_edge_index = self.block_edge_index[:, is_not_sl]
        if with_weight:
            self.block_edge_weight = self.block_edge_weight[is_not_sl]

    @staticmethod
    def project(budget: int, values: Tensor, eps: float = 1e-7):
        r"""Projects `values`: $budget \ge \sum \Pi_{[0, 1]}(\text{values})$."""
        if torch.clamp(values, 0, 1).sum() > budget:
            left = (values - 1).min()
            right = values.max()
            miu = PRBCDAttack.bisection(values, left, right, budget)
            values = values - miu
        return torch.clamp(values, min=eps, max=1 - eps)

    @staticmethod
    def bisection(edge_weights, a, b, n_pert, eps=1e-5, max_iter=1e3):
        """Bisection search for projection."""
        def shift(offset):
            return (torch.clamp(edge_weights - offset, 0, 1).sum() - n_pert)

        miu = a
        for i in range(int(max_iter)):
            miu = (a + b) / 2
            # Check if middle point is root
            if (shift(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (shift(miu) * shift(a) < 0):
                b = miu
            else:
                a = miu
            if ((b - a) <= eps):
                break
        return miu
