import functools
import math
from functools import partial
from random import randint
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import torch
import tqdm
from sklearn import metrics
from torch.linalg import norm
from torch.nn import Module, Parameter
from torch.nn.functional import cross_entropy
from torch.nn.init import trunc_normal_
from torch.optim import Adam, Optimizer

from torch_geometric.data import Data


def _remove_prefix(param_name, prefix):
    if param_name.startswith(prefix):
        return param_name[len(prefix):]
    return param_name


class GLTModel(Module):
    r"""Pruning model Wrapper from the `A Unified Lottery Ticket Hypothesis
    for Graph Neural Networks <https://arxiv.org/abs/2102.06790>`_ paper.

    Notably, the model requires additional memory overhead as masks are
    constructed for all named parameters. Masks are not applied to param
    names ending in ignore_keys.

    We support any model that takes edge_weights in the forward call.

    Args: module (torch.nn.Module): The GNN module to prune. graph (
    torch_geometric.data.Data): Graph to perform GLT over. ignore_keys (set,
    optional): Set of keys to ignore when injecting masks into the model.
    """
    MASK = "_mask"
    MASK_FIXED = "_mask_fixed"
    ORIG = "_orig"
    EDGE_MASK = "adj"

    def __init__(self, module: Module, graph: Data,
                 ignore_keys: Optional[Set[str]] = None):
        super().__init__()
        if ignore_keys is None:
            ignore_keys = set()
        self.module = module
        self.graph = graph
        self.ignore_keys = ignore_keys

    def get_params(self) -> List[Parameter]:
        return [
            param for param_name, param in self.named_parameters(recurse=True)
        ]

    def get_masks(self) -> Dict[str, Parameter]:
        return {
            _remove_prefix(param_name, "module."): param
            for param_name, param in self.named_parameters(recurse=True)
            if param_name.endswith(GLTModel.MASK)
        }

    def rewind(self, state_dict: Dict[str, Any]) -> None:
        """rewind model state dict"""
        for k, v in state_dict.items():
            module_path, _, param_name = k.rpartition(".")

            sub_mod = self.get_submodule(module_path)
            curr_param = getattr(sub_mod, param_name)
            curr_param.data = v

            if param_name.endswith(GLTModel.MASK):
                curr_param.requires_grad = False
                curr_param.grad = None

            setattr(sub_mod, param_name, curr_param)

    def apply_mask(self, mask_dict: Dict[str, Any]) -> None:
        """Inject GLTMask into model params and adjacency mask. Ignore keys
        specified by ignore_keys. """
        # Input validation
        param_names = [
            name for name, _ in self.module.named_parameters()
            if not GLTModel._is_injected_parameter(name)
        ] + [GLTModel.EDGE_MASK]
        missing_keys = [
            name for name in param_names
            if name + GLTModel.MASK not in mask_dict.keys()
        ]

        if len(missing_keys):
            raise ValueError(
                f"Masks for {missing_keys} missing from mask_dict!")

        # Injecting module masks
        for param_path in self._get_parameter_paths():
            module_path, _, param_name = param_path.rpartition(".")
            if param_name in self.ignore_keys:
                continue
            sub_mod = self.module.get_submodule(module_path)

            weight = getattr(sub_mod, param_name)
            del sub_mod._parameters[param_name]
            mask = mask_dict[param_path + GLTModel.MASK]

            sub_mod.register_parameter(
                param_name + GLTModel.MASK,
                Parameter(mask.data),
            )
            sub_mod.register_parameter(
                param_name + GLTModel.ORIG,
                Parameter(weight.data),
            )

            sub_mod.register_forward_pre_hook(
                partial(
                    GLTModel._hook,
                    param_name=param_name,
                ))

        adj_mask_name = GLTModel.EDGE_MASK + GLTModel.MASK
        self.register_parameter(adj_mask_name,
                                Parameter(mask_dict[adj_mask_name].data))
        self.register_buffer(
            GLTModel.EDGE_MASK + GLTModel.MASK_FIXED,
            torch.ones_like(mask_dict[adj_mask_name]),
        )
        self.register_forward_pre_hook(GLTModel._compute_masked_adjacency)

    def forward(self):
        """model forward, uses graph edges if present (link prediction)"""
        adj_mask = getattr(self, GLTModel.EDGE_MASK)

        if not isinstance(adj_mask, torch.Tensor):
            raise RuntimeError("invalid adjacency mask!")

        pruned_graph = self.graph.clone()
        pruned_graph.edge_weight = adj_mask  # type: ignore
        if hasattr(pruned_graph, "edges"):
            return self.module(pruned_graph.x, pruned_graph.edge_index,
                               edge_weight=pruned_graph.edge_weight,
                               edges=pruned_graph.edges)
        else:
            return self.module(pruned_graph.x, pruned_graph.edge_index,
                               edge_weight=pruned_graph.edge_weight)

    @staticmethod
    def _hook(sub_mod: Module, input: Any, param_name: str) -> None:
        setattr(
            sub_mod,
            param_name,
            GLTModel._compute_masked_param(sub_mod, param_name),
        )

    @staticmethod
    def _compute_masked_param(sub_mod: Module, param_name: str):
        mask = getattr(sub_mod, param_name + GLTModel.MASK)
        param = getattr(sub_mod, param_name + GLTModel.ORIG)

        masked_param = param * mask
        return masked_param

    @staticmethod
    def _compute_masked_adjacency(sub_mod: Module, input: Any):
        mask = getattr(sub_mod, GLTModel.EDGE_MASK + GLTModel.MASK)
        param = getattr(sub_mod, GLTModel.EDGE_MASK + GLTModel.MASK_FIXED)
        masked_param = torch.relu(param * mask)
        setattr(sub_mod, GLTModel.EDGE_MASK, masked_param)

    @staticmethod
    def _is_injected_parameter(param_name: str):
        return (param_name.endswith(GLTModel.MASK)
                or param_name.endswith(GLTModel.ORIG)
                or param_name.endswith(GLTModel.MASK_FIXED))

    def _get_parameter_paths(self):
        return [
            param_path
            for param_path, _ in self.module.named_parameters(recurse=True)
            if not self._is_injected_parameter(param_path)
        ]


EDGE_MASK = GLTModel.EDGE_MASK + GLTModel.MASK
INIT_FUNC = functools.partial(trunc_normal_, mean=1, a=1 - 1e-3, b=1 + 1e-3)


class GLTMask:
    r"""Generate UGLT masks for pruning model according to named parameters.
    Args: module (torch.nn.Module): The GNN module to make masks for. graph (
    torch_geometric.data.Data): Graph to make adjacency mask for. device (
    torch.device): Torch device to place masks on.
    """
    def __init__(self, module: Module, graph: Data,
                 device: torch.device) -> None:
        self.graph_mask = INIT_FUNC(
            torch.ones((graph.edge_index.shape[1] or graph.num_edges),
                       device=device))
        self.weight_mask = {
            param_name + GLTModel.MASK:
            INIT_FUNC(torch.ones_like(param, device=device))
            for param_name, param in module.named_parameters() if param_name
        }

    def sparsity(self) -> Tuple[float, float]:
        """calculate sparsity for masks"""
        norm_graph_mask = float(torch.count_nonzero(self.graph_mask))
        norm_graph = torch.numel(self.graph_mask)
        graph_sparsity = 1 - norm_graph_mask / norm_graph

        norm_weight_mask = 0
        norm_weight = 0

        for v in self.weight_mask.values():
            norm_weight_mask += float(torch.count_nonzero(v))
            norm_weight += torch.numel(v)

        weight_sparsity = 1 - norm_weight_mask / norm_weight
        return graph_sparsity, weight_sparsity

    def to_dict(self, weight_prefix=False) -> Dict[str, Any]:
        """convert masks to dict of masks"""
        pref = "module." if weight_prefix else ""

        return {
            EDGE_MASK: self.graph_mask.detach().clone(),
            **{
                pref + k: v.detach().clone()
                for k, v in self.weight_mask.items()
            },
        }

    def load_and_binarize(
        self,
        model_masks: Dict[str, Parameter],
        p_theta: float,
        p_g: float,
    ) -> None:
        """Parse masks and set bottom fraction of weights equal to 0 and rest
        equal to 1 for all masks. Number of zeroed out params is determined
        by pruning rates p_theta and p_g (model and graph, respectively). """
        # Validation
        missing_masks = [
            name for name in [
                EDGE_MASK,
                *self.weight_mask.keys(),
            ] if name not in model_masks.keys()
        ]

        if len(missing_masks):
            raise ValueError(
                f"Model has no masks for the following parameters:"
                f" {missing_masks} ")

        # splitting out m_g and m_theta
        graph_mask = model_masks[EDGE_MASK]
        del model_masks[EDGE_MASK]

        # process graph mask
        self.graph_mask = torch.where(self.graph_mask > 0, 1.0,
                                      0.)  # needed to support non-binary inits
        all_weights_graph = graph_mask[self.graph_mask == 1]
        num_prune_graph = min(math.floor(p_g * len(all_weights_graph)),
                              len(all_weights_graph) - 1)
        threshold_graph = all_weights_graph.sort()[0][num_prune_graph]
        self.graph_mask = torch.where(graph_mask > threshold_graph,
                                      self.graph_mask, 0.)

        # process weight masks
        self.weight_mask = {
            k: torch.where(v > 0, 1.0, 0.0)
            for k, v in self.weight_mask.items()
        }  # needed to support non-binary inits
        all_weights_model = torch.concat(
            [v[self.weight_mask[k] == 1] for k, v in model_masks.items()])
        num_prune_weights = min(math.floor(p_theta * len(all_weights_model)),
                                len(all_weights_model) - 1)
        threshold_model = all_weights_model.sort()[0][num_prune_weights]

        self.weight_mask = {
            k: torch.where(v > threshold_model, self.weight_mask[k], 0.0)
            for k, v in model_masks.items()
        }


def score_link_prediction(targets, preds, val_mask, test_mask):
    """helper for getting AUC for link preds"""
    val_preds = preds[val_mask]
    val_gt = targets[val_mask].cpu().detach().numpy()
    val_score = metrics.auc(val_gt,
                            torch.sigmoid(val_preds).cpu().detach().numpy())

    test_preds = preds[test_mask]
    test_gt = targets[test_mask].cpu().detach().numpy()
    test_score = metrics.auc(test_gt,
                             torch.sigmoid(test_preds).cpu().detach().numpy())
    return val_score, test_score


def score_node_classification(targets, preds, val_mask, test_mask):
    """helper for getting accuracy on node classification"""
    correct_val = (preds[val_mask] == targets[val_mask]).sum()
    val_score = int(correct_val) / int(val_mask.sum())

    correct_test = (preds[test_mask] == targets[test_mask]).sum()
    test_score = int(correct_test) / int(test_mask.sum())
    return val_score, test_score


class GLTSearch:
    r"""The Unified Graph Lottery Ticket search algorithm from the `A Unified
    Lottery Ticket Hypothesis for Graph Neural Networks
    <https://arxiv.org/abs/2102.06790>`_ paper.

    This paper presents a unified GNN sparsification (UGS) framework that
    simultaneously prunes the graph adjacency matrix and the model weights,
    for accelerating GNN inference on large-scale graphs. This pruning
    wrapper supports all models that can handle weighted graphs that are
    differentiable w.r.t. these edge weights, *e.g.*,
    :class:`~torch_geometric.nn.conv.GCNConv` or
    :class:`~torch_geometric.nn.conv.GraphConv`.

    GLTSearch class provides an interface to prune a model to a specified
    graph/model pruning rate.

    This methodology is built for both node classification and link
    prediction, but any other tasks should be easily extensible given the
    appropriate loss function and data.

    .. note::
        For examples of using the GLTSearch, see
        `examples/contrib/graph_lottery_ticket.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/
        examples/contrib/graph_lottery_ticket.py>`_.

    Args:
        module (torch.nn.Module): The GNN module to prune.
        graph (torch_geometric.data.Data): Graph to perform GLT search over.
        lr (float): Learning rate for training. Is propagated to masks if
        individual lr not specified.
        reg_graph (float): L2 regularization for graph mask.
        reg_model (float): L2 regularization for model masks.
        optim_args (dict): Args for internal optimizer.
        task (str): Specifies to use node or link train loop.
        lr_mask_model (float, optional):  LR to apply to model mask only.
        lr_mask_graph Optional[float] LR to apply to graph mask only.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        prune_rate_model (float): Sparsity level induced in model masks.
        prune_rate_graph (float): Sparsity level induced in graph mask.
        max_train_epochs (int): max number of epochs to train.
        loss_fn (callable): loss function to train with.
        save_all_masks (bool): toggles saving all masks.
        seed (int): random seed.
        verbose (bool): toggles trainer verbosity.
        ignore_keys (set, optional): Set of keys to ignore when injecting
        masks into the model.
    """
    __match_args__ = ('module', 'graph', 'lr', 'reg_graph', 'reg_model',
                      'task', 'optim_args', 'lr_mask_model', 'lr_mask_graph',
                      'optimizer', 'prune_rate_model', 'prune_rate_graph',
                      'max_train_epochs', 'loss_fn', 'save_all_masks', 'seed',
                      'verbose', 'ignore_keys')

    def __init__(self, module: Module, device: torch.device, graph: Data,
                 lr: float, reg_graph: float, reg_model: float, task: str,
                 optim_args: Dict[str, Any] = {},
                 lr_mask_model: Optional[float] = None,
                 lr_mask_graph: Optional[float] = None,
                 optimizer: Type[Optimizer] = Adam,
                 prune_rate_model: float = 0.2, prune_rate_graph: float = 0.05,
                 max_train_epochs: int = 200,
                 loss_fn: Callable = cross_entropy,
                 save_all_masks: bool = False, seed: Optional[int] = None,
                 verbose: bool = False,
                 ignore_keys: Optional[set] = None) -> None:
        if seed is None:
            seed = randint(1, 9999)
        if ignore_keys is None:
            ignore_keys = set()
        self.module = module
        self.graph = graph
        self.lr = lr
        self.reg_graph = reg_graph
        self.reg_model = reg_model
        self.task = task
        self.optim_args = optim_args
        self.lr_mask_model = lr_mask_model
        self.lr_mask_graph = lr_mask_graph
        self.optimizer = optimizer
        self.prune_rate_model = prune_rate_model
        self.prune_rate_graph = prune_rate_graph
        self.max_train_epochs = max_train_epochs
        self.loss_fn = loss_fn
        self.save_all_masks = save_all_masks
        self.seed = seed
        self.verbose = verbose
        self.ignore_keys = ignore_keys
        'fixes internal hyperparams and generates GLTMask from\n        :obj:`GLTMask` from model '
        torch.manual_seed(self.seed)
        if not self.lr_mask_graph:
            self.lr_mask_graph = self.lr
        if not self.lr_mask_model:
            self.lr_mask_model = self.lr
        self.optim_args = {'lr': self.lr, **self.optim_args}
        self.mask = GLTMask(self.module, self.graph, device)

    def __repr__(self):
        cls = type(self).__name__
        return f'{cls}(module={self.module!r}, graph={self.graph!r}, lr={self.lr!r}, reg_graph={self.reg_graph!r}, reg_model={self.reg_model!r}, task={self.task!r}, optim_args={self.optim_args!r}, lr_mask_model={self.lr_mask_model!r}, lr_mask_graph={self.lr_mask_graph!r}, optimizer={self.optimizer!r}, prune_rate_model={self.prune_rate_model!r}, prune_rate_graph={self.prune_rate_graph!r}, max_train_epochs={self.max_train_epochs!r}, loss_fn={self.loss_fn!r}, save_all_masks={self.save_all_masks!r}, seed={self.seed!r}, verbose={self.verbose!r}, ignore_keys={self.ignore_keys!r})'

    def __eq__(self, other):
        if not isinstance(other, GLTSearch):
            return NotImplemented
        return (self.module, self.graph, self.lr, self.reg_graph,
                self.reg_model, self.task, self.optim_args, self.lr_mask_model,
                self.lr_mask_graph, self.optimizer, self.prune_rate_model,
                self.prune_rate_graph, self.max_train_epochs, self.loss_fn,
                self.save_all_masks, self.seed, self.verbose,
                self.ignore_keys) == (other.module, other.graph, other.lr,
                                      other.reg_graph, other.reg_model,
                                      other.task, other.optim_args,
                                      other.lr_mask_model, other.lr_mask_graph,
                                      other.optimizer, other.prune_rate_model,
                                      other.prune_rate_graph,
                                      other.max_train_epochs, other.loss_fn,
                                      other.save_all_masks, other.seed,
                                      other.verbose, other.ignore_keys)

    def prune(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
        """UGS algorithm. Train model with UGS to train masks and params.
        Disretize masks by pruning rates. Retrain without UGS for final
        performance. """
        initial_params = {
            "module." + k + GLTModel.ORIG if k.rpartition(".")[-1]
            not in self.ignore_keys else "module." + k: v.detach().clone()
            for k, v in self.module.state_dict().items()
        }

        ticket = GLTModel(self.module, self.graph,
                          ignore_keys=self.ignore_keys)
        ticket.apply_mask(self.mask.to_dict())

        unrewound_test_score, masks = self.train(ticket, True)
        if self.verbose:
            print("[UNREWOUND] Final test performance:", unrewound_test_score)
        self.mask.load_and_binarize(masks, self.prune_rate_model,
                                    self.prune_rate_graph)

        mask_dict = self.mask.to_dict(weight_prefix=True)
        ticket.rewind({**mask_dict, **initial_params})
        fixed_test_score, masks = self.train(ticket, False)

        if self.verbose:
            print("[FIXED MASK] Final test performance:", fixed_test_score)
        current_sparsity = self.mask.sparsity()
        if self.verbose:
            print(
                "Graph sparsity:",
                round(current_sparsity[0], 4),
                "Model sparsity:",
                round(current_sparsity[1], 4),
            )
        results_dict = {
            "unrewound_test": unrewound_test_score,
            "fixed_test": fixed_test_score,
            "graph_sparsity": current_sparsity[0],
            "model_sparsity": current_sparsity[1]
        }
        return initial_params, self.mask.to_dict(), results_dict

    def train(self, ticket: GLTModel,
              ugs: bool) -> Tuple[float, Dict[str, Parameter]]:
        """train loop. If ugs flag, use mask regularization."""
        best_val_score = 0.0
        final_test_score = 0.0
        best_masks = {}
        optimizer = self.optimizer(ticket.parameters(), **self.optim_args)

        with tqdm.trange(self.max_train_epochs, disable=not self.verbose) as t:
            for epoch in t:
                ticket.train()
                optimizer.zero_grad()

                output = ticket()

                if self.task == "node_classification":
                    loss = self.loss_fn(output[self.graph.train_mask],
                                        self.graph.y[self.graph.train_mask])
                elif self.task == "link_prediction":
                    edge_mask = self.graph.train_mask[self.graph.edges[
                        0]] & self.graph.train_mask[self.graph.edges[1]]
                    loss = self.loss_fn(
                        output[edge_mask],
                        self.graph.edge_labels[edge_mask].float())
                else:
                    raise ValueError(
                        f"{self.task} must be one of node class. or link pred."
                    )

                if ugs:
                    for mask_name, mask in ticket.get_masks().items():
                        if mask_name.startswith("adj"):
                            loss += self.reg_graph * norm(
                                mask.flatten(), ord=1)
                        else:
                            loss += self.reg_model * norm(
                                mask.flatten(), ord=1)

                loss.backward()
                optimizer.step()

                ticket.eval()
                if self.task == "node_classification":
                    preds = ticket().argmax(dim=1)
                    val_score, test_score = score_node_classification(
                        self.graph.y, preds, self.graph.val_mask,
                        self.graph.test_mask)
                elif self.task == "link_prediction":
                    preds = ticket()
                    val_mask = self.graph.val_mask[self.graph.edges[
                        0]] & self.graph.val_mask[self.graph.edges[1]]
                    test_mask = self.graph.test_mask[self.graph.edges[
                        0]] & self.graph.test_mask[self.graph.edges[1]]
                    val_score, test_score = score_link_prediction(
                        self.graph.edge_labels, preds, val_mask, test_mask)
                else:
                    raise ValueError(
                        f"{self.task} must be one of node class. or link pred."
                    )
                if val_score > best_val_score:
                    best_val_score = val_score
                    final_test_score = test_score

                    if ugs:
                        best_masks = ticket.get_masks()

                t.set_postfix({
                    "loss": loss.item(),
                    "val_score": val_score,
                    "test_score": test_score
                })
        return final_test_score, best_masks
