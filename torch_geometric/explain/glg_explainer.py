from typing import Dict, Optional, Union, Callable, List, Tuple
from collections import defaultdict

import random
import math
import torch
import copy
from sympy import to_dnf, lambdify
from torch import Tensor, LongTensor

from torch_geometric.explain.metric import (
    test_logic_explanation,
    formulas_accuracy
)
from torch_geometric.explain import LogicExplanation, LogicExplanations
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_scatter import scatter
import torch.nn.functional as F


class GLGExplainer(torch.nn.Module):
    r"""Implementation of GLGExplainer `"Global Explainability of GNNs via
    Logic Combination of Learned Concepts" <https://arxiv.org/abs/2210.07147>`_

    Importantly, the :class:`GLGExplainer` needs to be trained via
    :meth:`~GLGExplainer.do_epoch` before providing any Global Explanation.

    For an example of using :class:`GLGExplainer`,
        see `examples/glg_explainer.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        /glg_explainer.py>`_.

    Args:
        le_embedder (torch.nn.Module): The Local Explanations Embedder.
            It must be a :class:`torch.nn.Module` providing an embedding
            for each input local explanation
        num_classes (int): Number of output classes
        num_prototypes (int): Number of prototypes to use
        dim_prototypes (int, optional): Latent dimensionality of each
            prototype. (default :obj:`10`)
        dim_len_hidden (int, optional): Latent dimensionality of the E-LEN.
            (default :obj:`10`)
        activation_func (str or Callable): The type of activation function to
            use for prototypes activations. The possible values are:

                - :obj:`"discrete"`: Discretize the vector of prototype
                  activations as described in the paper

                - :obj:`"inverse"`: Inverse of the Euclidean distance

                - :obj:`"sim"`: The distance function as defined in
                  `"This Looks Like That: Deep Learning for
                  Interpretable Image Recognition"
                  <https://arxiv.org/abs/1806.10574>`_
        len_lr (float, optional): Learning rate for the parameters of the
            E-LEN. (default :obj:`0.0005`)
        prototypes_lr (float, optional): Learning rate for the prototypes.
            (default :obj:`0.001`)
        le_embedder_lr (float, optional): Learning rate for the parameters
            of the :attr:`le_embedder`. (default :obj:`0.001`)
        len_loss (str or Callable, optional): The type of loss for training the
            E-LEN. The possible values are:

                - :obj:`"focal_loss"`: Entropy loss with focal regularization.

                - :obj:`"entropy"`: Standard entropy loss.

            (default :obj:`"focal_loss"`)
        focal_alpha (float, optional): :obj:`"alpha"` hyper-parameter for the
            focal regularization. (default :obj:`-1`)
        focal_gamma (float, optional): :obj:`"gamma"` hyper-parameter for the
            focal regularization. (default :obj:`2`)
        coeff_r1 (float, optional): Scaling weight for the distance loss
            :obj:`R1`. (default :obj:`0.09`)
        coeff_r2 (float, optional): Scaling weight for the distance loss
            :obj:`R2`. (default :obj:`0.00099`)
        min_delta (float, optional): :obj:`"min_delta"` hyper-parameter for
            early stopping. (default :obj:`0.`)
        patience (int, optional): :obj:`"patient"` hyper-parameter for early
            stopping. (default :obj:`100`)

    """
    def __init__(
        self,
        le_embedder: torch.nn.Module,
        num_classes: int,
        num_prototypes: int,
        dim_prototypes: Optional[int] = 10,
        dim_len_hidden: Optional[int] = 10,
        activation_func: Union[Callable, str] = "discrete",
        len_lr: Optional[float] = 0.0005,
        prototypes_lr: Optional[float] = 0.001,
        le_embedder_lr: Optional[float] = 0.001,
        len_loss: Union[Callable, str] = "focal_loss",
        focal_alpha: Optional[float] = -1,
        focal_gamma: Optional[float] = 2,
        coeff_r1: Optional[float] = 0.09,
        coeff_r2: Optional[float] = 0.00099,
        min_delta: Optional[float] = 0.,
        patience: Optional[int] = 100,
    ):
        super().__init__()

        self.le_embedder = le_embedder
        self.num_classes = num_classes
        self.activation_func = activation_func
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.coeff_r1 = coeff_r1
        self.coeff_r2 = coeff_r2
        self.global_explanation = LogicExplanations()

        self.len_model = _ELEN(
            input_shape=num_prototypes,
            hidden_dim=dim_len_hidden,
            n_classes=num_classes,
            no_attn=True
        )
        self.prototype_vectors = torch.nn.Parameter(
            torch.rand((num_prototypes,
                        dim_prototypes)), requires_grad=True)

        self.optimizer = torch.optim.Adam(
            self.le_embedder.parameters(),
            lr=le_embedder_lr
        )
        self.optimizer.add_param_group({
            'params': self.len_model.parameters(),
            'lr': len_lr
        })
        self.optimizer.add_param_group({
            'params': self.prototype_vectors,
            'lr': prototypes_lr
        })

        self.len_loss = self._resolve_len_loss(len_loss)
        self.early_stopping = _EarlyStopping(min_delta, patience)

    def do_epoch(self, loader: DataLoader, train: bool = False) -> Dict:
        r"""Perform one-epoch training.

        Args:
            loader (:class:`DataLoader`): Input data.
            train (bool, optional): If :obj:`False`, no training
                is perforfmed, and only the resulting metrics are returned.
                (default :obj:`False`)
        rtype: :class:`Dict`"""
        assert isinstance(loader.batch_sampler, _GroupBatchSampler), \
            "The input DataLoader must use '_GroupBatchSampler' as"\
            "'batch_sampler'"

        device = self.prototype_vectors.device
        if train:
            self.train()
        else:
            self.eval()

        preds = torch.tensor([], device=device)
        trues = torch.tensor([], device=device)
        le_classes = torch.tensor([], device=device)
        total_prototype_assignements = torch.tensor([], device=device)
        total_losses = defaultdict(lambda: torch.tensor(0., device=device))

        with torch.set_grad_enabled(train):
            for data in loader:
                self.optimizer.zero_grad()
                data = data.to(device)
                le_embeddings = self.le_embedder(
                    data.x,
                    data.edge_index,
                    data.batch
                )

                new_belonging = torch.tensor(
                    self.normalize_belonging(data.graph_id),
                    dtype=torch.long,
                    device=device
                )
                y = scatter(data.y, new_belonging, dim=0, reduce="max")
                y_train_1h = torch.nn.functional.one_hot(
                    y.long(),
                    num_classes=self.num_classes
                ).float().to(device)

                proto_assignements = self.prototype_assignement(le_embeddings)
                total_prototype_assignements = torch.cat([
                    total_prototype_assignements,
                    proto_assignements], dim=0
                )
                le_classes = torch.concat([le_classes, data.le_label], dim=0)

                concept_vector = scatter(
                    proto_assignements,
                    new_belonging,
                    dim=0,
                    reduce="max"
                )
                y_pred = self.len_model(concept_vector).squeeze(-1)

                loss = self._compute_losses(
                    le_embeddings,
                    total_losses,
                    y_pred,
                    y_train_1h,
                    device,
                )
                preds = torch.cat([preds, y_pred], dim=0)
                trues = torch.cat([trues, y_train_1h], dim=0)

                if train:
                    loss.backward()
                    self.optimizer.step()

            cluster_acc = self.concept_purity(
                total_prototype_assignements.argmax(1).detach(),
                le_classes
            )

            metrics = {
                k: v.item() / len(loader)
                for k, v in total_losses.items()
            }
            metrics["fidelity"] = self.fidelity(preds, trues)
            metrics["cluster_acc_mean"] = torch.mean(cluster_acc)
            metrics["cluster_acc_std"] = torch.std(cluster_acc)
            metrics["concept_vector_entropy"] = self._entropy(
                proto_assignements
            ).detach()
            return metrics

    @torch.no_grad()
    def get_concept_vector(
        self,
        loader: DataLoader
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Returns the final concept vector of input data, along
        with the relative encoded embeddings, raw prototype assignements,
        and target values.

        Args:
            loader (:class:`~torch_geometric.loader.DataLoader`): Input data.
        rtype: :class:`Tuple`"""
        assert isinstance(loader.batch_sampler, _GroupBatchSampler), \
            "The input DataLoader must use '_GroupBatchSampler' as"\
            "'batch_sampler'"

        device = self.prototype_vectors.device
        le_embeddings = torch.tensor([], device=device)
        belonging = torch.tensor([], device=device, dtype=torch.long)
        y = torch.tensor([], device=device)
        le_classes = torch.tensor([], device=device)
        le_idxs = torch.tensor([], device=device)
        for data in loader:
            data = data.to(device)
            le_idxs = torch.concat([le_idxs, data.le_id], dim=0)
            embs = self.le_embedder(data.x, data.edge_index, data.batch)
            le_embeddings = torch.concat([le_embeddings, embs], dim=0)
            belonging = torch.concat([belonging, data.graph_id], dim=0)
            le_classes = torch.concat([le_classes, data.le_label], dim=0)
            y = torch.concat([y, data.y], dim=0)

        y = scatter(y, belonging, dim=0, reduce="max")
        y = torch.nn.functional.one_hot(y.long()).float().to(device)

        assignments = self.prototype_assignement(le_embeddings)
        concept_vector = scatter(assignments, belonging, dim=0, reduce="max")
        return (
            concept_vector,
            le_embeddings,
            assignments,
            y,
            le_classes,
            le_idxs,
            belonging
        )

    @torch.no_grad()
    def inspect(
        self,
        loader: DataLoader,
        update_global_explanation: Optional[bool] = False,
        plot: Optional[bool] = True,
        plot_path: Optional[str] = None,
        le_classes_names: Optional[List[str]] = None
    ) -> Dict:
        r"""Compute evaluation metrics for the explainer. Optionally,
        visualize the embedding with the learned prototypes.

        Args:
            loader (:class:`~torch_geometric.loader.DataLoader`): Input data.
            update_global_explanation (bool, optional): If :obj:`True`,
                the stored Global Explanation is updated by extracting new
                formulas. If :obj:`False`, :attr:`self.global_explanation`
                is evaluated. (default :obj:`False`)
            plot (bool, optional): If :obj:`True`, the plot of the embeddings
                and the prototypes is generated. (default :obj:`False`)
            plot_path (str, optional): If :attr:`plot` is :obj:`True` and
                :attr:`plot_path` is not :obj:`None`, the generated plot is
                saved as PDF in the folder pointed by :attr:`plot_path`,
                otherways the plot is generated on-fly. (default :obj:`None`)
            le_classes_names (:class:`List[str]`, optional): If
                :obj:`plot=True`, specify a label for each type local
                explanations. If :obj:`None`, incremental labels
                will be assigned. (default :obj:`None`)
        rtype: :class:`Dict`"""
        assert isinstance(loader.batch_sampler, _GroupBatchSampler), \
            "The input DataLoader must use '_GroupBatchSampler' as"\
            "'batch_sampler'"

        self.eval()
        (
            x,
            emb,
            concepts_assignement,
            y_1h,
            le_classes,
            _,
            _,
        ) = self.get_concept_vector(loader)
        y_pred = self.len_model(x).squeeze(-1).cpu()

        emb = emb.detach().cpu().numpy()
        concept_predictions = concepts_assignement.argmax(1).cpu()

        if plot:
            if le_classes_names is None:
                le_classes_names = [
                    str(i)
                    for i in range(torch.unique(le_classes).shape[0])
                ]

            pca = PCA(n_components=2, random_state=42)
            if self.prototype_vectors.shape[1] == 2:
                emb2d = emb
            else:
                emb2d = pca.fit_transform(emb)

            fig = plt.figure(figsize=(17, 4))
            plt.subplot(1, 2, 1)
            plt.title("local explanations embeddings", size=23)
            for c in torch.unique(le_classes):
                plt.scatter(
                    emb2d[le_classes == c, 0],
                    emb2d[le_classes == c, 1],
                    label=le_classes_names[int(c)],
                    alpha=0.7
                )

            if self.prototype_vectors.shape[1] == 2:
                proto_2d = self.prototype_vectors.cpu().numpy()
            else:
                proto_2d = pca.transform(self.prototype_vectors.cpu().numpy())

            plt.scatter(
                proto_2d[:, 0],
                proto_2d[:, 1],
                marker="x",
                s=60,
                c="black"
            )
            for i, txt in enumerate(range(proto_2d.shape[0])):
                plt.annotate(
                    "p" + str(i),
                    (proto_2d[i, 0] + 0.01, proto_2d[i, 1] + 0.01),
                    size=27
                )
            plt.legend(bbox_to_anchor=(0.04, 1), prop={'size': 17})
            plt.subplot(1, 2, 2)
            plt.title("prototype assignments", size=23)
            for c in range(self.prototype_vectors.shape[0]):
                plt.scatter(
                    emb2d[concept_predictions == c, 0],
                    emb2d[concept_predictions == c, 1],
                    label="p" + str(c)
                )
            plt.legend(prop={'size': 17})

            fig.supxlabel('principal comp. 1', size=20)
            fig.supylabel('principal comp. 2', size=20)
            if plot_path:
                plt.savefig(plot_path)
                plt.close()
            else:
                plt.show()

        device = self.prototype_vectors.device
        self.len_model.to("cpu")
        x = x.detach().cpu()
        y_1h = y_1h.cpu()

        cluster_accs = self.concept_purity(concept_predictions, le_classes)

        if update_global_explanation:
            self.global_explanation = self.len_model.explain_classes(
                x,
                y_1h,
                self.num_classes
            )
        accuracy = formulas_accuracy(self.global_explanation, x, y_1h)

        self.len_model.to(device)
        return {
            "fidelity": self.fidelity(y_pred, y_1h),
            "formula_accuracy": accuracy,
            "concept_purity": torch.mean(cluster_accs),
            "concept_purity_std": torch.std(cluster_accs),
            "concepts_distribution": torch.unique(
                concept_predictions, return_counts=True
            ),
            "explanations": self.global_explanation,
        }

    @torch.no_grad()
    def __call__(
        self,
        loader: DataLoader,
        threshold: float = 0.5
    ) -> Tuple[Tensor, Tensor]:
        r"""Returns the neural preictions of the E-LEN and the predictions of
        the extracted Global Logic Explanations

        .. note::

            GLGExplainer must be trained first.

        Args:
            loader (:class:`~torch_geometric.loader.DataLoader`): Input data
            threshold (float, optional): Threshold to get truth values
                for class predictions
                (i.e. :obj:`pred < threshold = False`,
                :obj:`pred > threshold = True`). (default :obj:`0.5`)

        type: :class:`Tuple[Tensor, Tensor]`"""
        assert isinstance(loader.batch_sampler, _GroupBatchSampler), \
            "The input DataLoader must use '_GroupBatchSampler' as"\
            "'batch_sampler'"
        assert self.global_explanation != LogicExplanations, \
            "No Global Explanation available." \
            "You have to first train the model"

        self.eval()
        (
            x,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.get_concept_vector(loader)
        y_pred = self.len_model(x).squeeze(-1).argmax(1)

        # formula predictions
        x = x.cpu().detach().numpy()
        concept_list = [f"feature{i:010}" for i in range(x.shape[1])]
        class_predictions = torch.zeros(x.shape[0], self.num_classes)
        for i, formula in enumerate(self.global_explanation.get_formulas()):
            explanation = to_dnf(formula)
            f = lambdify(concept_list, explanation, 'numpy')
            class_predictions[:, i] = torch.LongTensor(
                f(*[x[:, i] > threshold for i in range(x.shape[1])])
            )
        return y_pred, class_predictions

    @torch.no_grad()
    def concept_purity(self, concept_predictions, classes):
        uniques = torch.unique(concept_predictions)
        accs = torch.zeros(uniques.shape[0], device=concept_predictions.device)
        for i, cl in enumerate(uniques):
            _, counts = torch.unique(
                classes[concept_predictions == cl],
                return_counts=True
            )
            accs[i] = torch.max(counts) / torch.sum(counts)
        return accs

    @torch.no_grad()
    def fidelity(self, preds, trues):
        return sum(
            trues[:, :].eq(preds[:, :] > 0).sum(1) == self.num_classes
        ) / len(preds)

    def prototype_assignement(self, le_embeddings):
        if not isinstance(self.activation_func, str):
            le_assignments = self.activation_func(
                le_embeddings,
                self.prototype_vectors
            )
        elif self.activation_func == "inverse":
            le_assignments = self._inverse(
                torch.cdist(le_embeddings, self.prototype_vectors, p=2)
            )
        elif self.activation_func == "sim":  # from ProtoPNet
            dist = torch.cdist(le_embeddings, self.prototype_vectors, p=2)**2
            sim = torch.log((dist + 1) / (dist + 1e-6))
            le_assignments = F.softmax(sim, dim=-1)
        elif self.activation_func == "discrete":
            dist = torch.cdist(le_embeddings, self.prototype_vectors, p=2)**2
            sim = torch.log((dist + 1) / (dist + 1e-6))
            y_soft = F.softmax(sim, dim=-1)

            # reparametrization trick from Gumbel-Softmax
            index = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
            le_assignments = y_hard - y_soft.detach() + y_soft
        else:
            raise NotImplementedError(f"'activation_func' can have only values"
                                      f"'discrete', 'sim' or 'inverse' "
                                      f"got {self.activation_func}"
                                      )
        return le_assignments

    def load_state_dict(self, loader: DataLoader, **kwargs) -> None:
        r"""Load the state of a trained model and extract
        its relative Global Explanation.

        Args:
            loader (:class:`~torch_geometric.loader.DataLoader`): Input data
                 the Global Explanation
            **kwargs (optional): Parameters for
                :meth:`torch.nn.load_state_dict`
        """

        assert isinstance(loader.batch_sampler, _GroupBatchSampler), \
            "The input DataLoader must use '_GroupBatchSampler' as"\
            "'batch_sampler'"

        ret = super().load_state_dict(kwargs)
        assert ret[0] == [] and ret[1] == [], \
            "Some keys in the state dict were not succesfully matched"

        # Extract and store the resulting Global Explanation
        # for the loaded model
        (
            x,
            _,
            _,
            y_1h,
            _,
            _,
            _,
        ) = self.get_concept_vector(loader)
        self.global_explanation = self.len_model.explain_classes(
            x,
            y_1h,
            self.num_classes
        )
        return ret

    def _compute_losses(
            self,
            le_embeddings,
            total_losses,
            y_pred,
            y_train_1h,
            device
    ):
        len_loss = 0.5 * self.len_loss(y_pred, y_train_1h)  # LEN loss
        total_losses["len_loss"] += len_loss.detach()

        # R1 loss: push each prototype to be close to at least one example
        if self.coeff_r1 > 0:
            sample_prototype_distance = torch.cdist(
                le_embeddings,
                self.prototype_vectors,
                p=2
            )**2  # num_sample x num_prototypes
            min_proto_sample_dist = sample_prototype_distance.T.min(-1).values
            avg_prototype_sample_distance = torch.mean(min_proto_sample_dist)
            r1_loss = self.coeff_r1 * avg_prototype_sample_distance
            total_losses["r1_loss"] += r1_loss.detach()
        else:
            r1_loss = torch.tensor(0., device=device)

        # R2 loss: Push every example to be close to a sample
        if self.coeff_r2 > 0:
            sample_prototype_distance = torch.cdist(
                le_embeddings,
                self.prototype_vectors,
                p=2
            )**2
            min_proto_sample_dist = sample_prototype_distance.min(-1).values
            avg_sample_prototype_distance = torch.mean(min_proto_sample_dist)
            r2_loss = self.coeff_r2 * avg_sample_prototype_distance
            total_losses["r2_loss"] += r2_loss.detach()
        else:
            r2_loss = torch.tensor(0., device=device)

        loss = len_loss + r1_loss + r2_loss
        total_losses["loss"] += loss.detach()
        return loss

    def _resolve_len_loss(self, loss):
        if isinstance(loss, str):
            if loss == "focal_loss":
                return self._focal_loss
            elif loss == "entropy":
                if self.num_classes == 2:
                    return F.binary_cross_entropy_with_logits
                elif self.num_classes == 3:
                    return F.cross_entropy
                else:
                    raise NotImplementedError(f"num_classes implemented <= 3"
                                              f"got {self.num_classes}."
                                              )
            else:
                raise NotImplementedError(f"'loss' can have only values"
                                          f"'focal_loss' or 'entropy' "
                                          f"got ('loss={loss}')."
                                          )
        else:
            return loss

    def _focal_loss(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets \
                + (1 - self.focal_alpha) * (1 - targets)
            loss = alpha_t * loss
        loss = loss.mean()
        return loss

    @staticmethod
    def normalize_belonging(belonging):
        ret = []
        i = -1
        for j, elem in enumerate(belonging):
            if len(ret) > 0 and elem == belonging[j - 1]:
                ret.append(i)
            else:
                i += 1
                ret.append(i)
        return ret

    def _inverse(self, x):
        x = 1 / (x + 0.0000001)
        return x / x.sum(-1).unsqueeze(1)

    def _entropy(self, logits):
        logp = torch.log(logits + 0.0000000001)
        entropy = torch.sum(-logits * logp, dim=1)
        entropy = torch.mean(entropy)
        return entropy

    @staticmethod
    def get_sampler(num_input_graphs: int, drop_last: bool,
                    belonging: LongTensor) -> Callable:
        return _GroupBatchSampler(num_input_graphs, drop_last, belonging)


class _EarlyStopping():
    def __init__(self, min_delta=0, patience=0):
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.best = torch.tensor(float("Inf"))
        self.best_epoch = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, current_value: float) -> bool:
        if current_value + self.min_delta < self.best:
            self.best = current_value
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
        return self.stop_training


class _GroupBatchSampler():
    r"""A custom Torch sampler in order to sample in
    the same batch all disconnected local explanations
    belonging to the same input sample, identified by
    the :class:`torch_geometric.data.Data`
    parameter of the :class:`torch_geometric.loader.Dataloader`."""

    def __init__(
            self,
            num_input_graphs: int,
            drop_last: bool,
            belonging: LongTensor
    ):
        self.batch_size = num_input_graphs
        self.drop_last = drop_last
        self.belonging = belonging
        self.num_input_graphs = num_input_graphs

        torch.manual_seed(42)
        random.seed(42)

    def __iter__(self):
        batch = []
        num_graphs_added = 0
        graph_idxs = random.sample(
            torch.unique(self.belonging).tolist(),
            torch.unique(self.belonging).shape[0]
        )
        for graph_id in graph_idxs:
            le_idxs = torch.where(self.belonging == graph_id)[0]
            batch.extend(le_idxs.tolist())
            if num_graphs_added >= self.batch_size:
                yield batch
                batch = []
                num_graphs_added = 0
            num_graphs_added += 1

        if len(batch) > 1 and not self.drop_last:
            yield batch

    def __len__(self):
        length = len(self.belonging) // self.batch_size
        if self.drop_last:
            return length
        else:
            return length + 1


class _EntropyLinear(torch.nn.Module):
    r"""Adaptation of the implementation of an Entropy-based
    Logic Explained Network from `"torch_explain"
    <https://pypi.org/project/torch-explain/>`_ ."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            n_classes: int,
            temperature: float = 0.6,
            bias: bool = True,
            remove_attention: bool = False
    ) -> None:
        super(_EntropyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temp = temperature
        self.alpha = None
        self.remove_attention = remove_attention
        self.weight = torch.nn.Parameter(
            torch.Tensor(n_classes, out_features, in_features)
        )
        self.has_bias = bias
        if bias:
            self.bias = torch.nn.Parameter(
                torch.Tensor(n_classes, 1, out_features)
            )
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight
            )
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)

        # compute concept-awareness scores
        gamma = self.weight.norm(dim=1, p=1)
        self.alpha = torch.exp(gamma / self.temp) \
            / torch.sum(torch.exp(gamma / self.temp), dim=1, keepdim=True)

        # weight the input concepts by awareness scores
        self.alpha_norm = self.alpha / self.alpha.max(dim=1)[0].unsqueeze(1)
        if self.remove_attention:
            self.concept_mask = torch.ones_like(self.alpha_norm, dtype=bool)
            x = input
        else:
            self.concept_mask = self.alpha_norm > 0.5
            x = input.multiply(self.alpha_norm.unsqueeze(1))

        # compute linear map
        x = x.matmul(self.weight.permute(0, 2, 1))
        if self.has_bias:
            x += self.bias
        return x.permute(1, 0, 2)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_classes={}'.format(
            self.in_features, self.out_features, self.n_classes
        )


class _ELEN(torch.nn.Module):
    def __init__(
            self,
            input_shape,
            n_classes,
            temperature=1,
            hidden_dim=10,
            no_attn=False
    ) -> None:
        super().__init__()

        self.layers = torch.nn.Sequential(*[
            _EntropyLinear(
                input_shape,
                hidden_dim,
                n_classes=n_classes,
                temperature=temperature,
                remove_attention=no_attn
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, int(hidden_dim / 2)),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(
                int(hidden_dim / 2),
                1 if n_classes == 2 else n_classes
            ),
        ])

    def forward(self, x):
        return self.layers(x)

    def explain_classes(
            self,
            x: torch.Tensor,
            y_1h: torch.Tensor,
            num_classes: int
    ) -> LogicExplanations:
        r"""Generate the Global Explanation by
        extracting a logic formula for each class.

        Args:
            x (torch.Tensor): Input concept vectors
            y_1h (torch.Tensor): One-hot encoded target labels
            num_classes (int): Number of classes to explain
        rtype: :class:`LogicExplanations`"""

        explanations = LogicExplanations()
        for c in range(num_classes):
            explanation = self.explain_class(x, y_1h, target_class=c)
            explanations.add_explanation(explanation)
        return explanations

    def explain_class(
            self,
            c: torch.Tensor,
            y: torch.Tensor,
            target_class: int,
            train_mask: torch.Tensor = None,
            val_mask: torch.Tensor = None,
            edge_index: torch.Tensor = None,
            max_minterm_complexity: int = None,
            max_accuracy: bool = True,
            concept_names: List[str] = None,
            try_all: bool = False,
            c_threshold: float = 0.5,
            y_threshold: float = 0.
    ) -> LogicExplanation:
        r"""Generate a local explanation for a single sample.

        Args:
            c (torch.Tensor): Input concept vectors
            y (torch.Tensor): One-hot encoded target labels
            target_class (int): Target class
            train_mask (torch.Tensor, optional): Train mask. If :obj:`None`
                every concept vector will be selected. (default :obj:`None`)
            val_mask (torch.Tensor, optional): Validation mask. If :obj:`None`
                every concept vector will be selected. (default :obj:`None`)
            edge_index (torch.Tensor, optional): Edge index for graph data
                used in graph-based models. (default :obj:`None`)
            max_minterm_complexity (int, optional): Maximum number
                of concepts per logic formula (per sample).
                (default :obj:`None`)
            max_accuracy (bool, optional): If True a formula is simplified
                only if the simplified formula gets 100% accuracy.
                (default :obj:`True`)
            concept_names (List[str], optional): List containing the names of
                the input concepts. (default :obj:`None`)
            try_all (bool, optional): If True, then tries all possible
                conjunctions of the top k explanations. (default :obj:`False`)
            c_threshold (float, optional): Threshold to get truth values
                for concept predictions (i.e. :obj:`pred < threshold = False`,
                :obj:`pred > threshold = True`). (default :obj:`0.5`)
            y_threshold (float, optional): Threshold to get truth values
                for class predictions
                (i.e. :obj:`pred < threshold = False`,
                :obj:`pred > threshold = True`). (default :obj:`0.`)
        rtype: :class:`LogicExplanation`"""

        if train_mask is None:
            train_mask = torch.arange(c.shape[0]).long()
        if val_mask is None:
            val_mask = torch.arange(c.shape[0]).long()

        (
            c_correct,
            y_correct,
            correct_mask,
            active_mask
        ) = self._get_correct_data(c,
                                   y,
                                   train_mask,
                                   target_class,
                                   edge_index,
                                   y_threshold
                                   )
        if c_correct is None:
            return LogicExplanation("")

        feature_names = [f"feature{j:010}" for j in range(c_correct.size(1))]
        class_explanation = ""
        local_explanations = []
        local_explanations_accuracies = {}
        local_explanations_raw = {}
        idx_and_exp = []
        for layer_id, module in enumerate(self.layers.children()):
            if isinstance(module, _EntropyLinear):
                # look at the "positive" rows of the truth table only
                positive_samples = torch.nonzero(
                    y_correct[:, target_class]
                ).numpy().ravel()

                for positive_sample in positive_samples:
                    (
                        le,
                        le_raw
                    ) = self._local_explanation(
                        module,
                        feature_names,
                        positive_sample,
                        local_explanations_raw,
                        c_correct,
                        y_correct,
                        target_class,
                        max_accuracy,
                        max_minterm_complexity,
                        c_threshold=c_threshold,
                        y_threshold=y_threshold,
                    )

                    if (le and le_raw and
                            le_raw not in local_explanations_raw):
                        idx_and_exp.append(
                            (positive_sample, le_raw)
                        )
                        local_explanations_raw[
                            le_raw
                        ] = le_raw
                        local_explanations.append(le)

                for positive_sample, le_raw in idx_and_exp:
                    # test explanation accuracy
                    if le_raw not in local_explanations_accuracies and le_raw:
                        accuracy = test_logic_explanation(
                            le_raw,
                            c,
                            y,
                            target_class,
                            val_mask,
                            c_threshold
                        )
                        local_explanations_accuracies[
                            le_raw
                        ] = accuracy

                # aggregate local explanations and replace
                # concept names in the final formula
                if try_all:
                    raise NotImplementedError("Parameter 'try_all'"
                                              "disabled for this"
                                              "implementation.")
                else:
                    explanations = []
                    for expl, acc in local_explanations_accuracies.items():
                        explanation = self._simplify_formula(
                            expl,
                            c,
                            y,
                            target_class,
                            max_accuracy,
                            val_mask,
                            c_threshold
                        )
                        explanations.append("(" + explanation + ")")
                    aggr_explanation = "(" + " | ".join(explanations) + ")"
                class_explanation_raw = str(aggr_explanation)
                class_explanation = class_explanation_raw
                if concept_names is not None:
                    class_explanation = self.replace_names(
                        class_explanation,
                        concept_names
                    )
                break
        return LogicExplanation(class_explanation[1:-1])

    def _local_explanation(
        self,
        module,
        feature_names,
        neuron_id,
        neuron_explanations_raw,
        c_validation,
        y_target,
        target_class,
        max_accuracy,
        max_minterm_complexity,
        c_threshold=0.5,
        y_threshold=0.,
    ):
        # explanation is the conjunction of non-pruned features
        explanation_raw = ""
        if max_minterm_complexity:
            concepts_to_retain = torch.argsort(
                module.alpha[target_class],
                descending=True)[:max_minterm_complexity]
        else:
            non_pruned_concepts = module.concept_mask[target_class]
            concepts_sorted = torch.argsort(module.alpha[target_class])
            concepts_to_retain = concepts_sorted[
                non_pruned_concepts[concepts_sorted]
            ]

        for j in concepts_to_retain:
            if feature_names[j] not in ["()", ""]:
                if explanation_raw:
                    explanation_raw += " & "
                if c_validation[neuron_id, j] > c_threshold:
                    # if non_pruned_neurons[j] > 0:
                    explanation_raw += feature_names[j]
                else:
                    explanation_raw += f"~{feature_names[j]}"

        explanation_raw = str(explanation_raw)
        if explanation_raw in ["", "False", "True", "(False)", "(True)"]:
            return None, None

        if explanation_raw in neuron_explanations_raw:
            explanation = neuron_explanations_raw[explanation_raw]
        else:
            explanation = explanation_raw

        if explanation in ["", "False", "True", "(False)", "(True)"]:
            return None, None

        return explanation, explanation_raw

    def _get_correct_data(
            self,
            c,
            y,
            train_mask,
            target_class,
            edge_index,
            threshold=0.
    ):
        active_mask = y[train_mask, target_class] == 1

        # get model's predictions
        if edge_index is None:
            preds = self.layers(c).squeeze(-1)
        else:
            preds = self.layers(c, edge_index).squeeze(-1)

        # identify samples correctly classified of the target class
        correct_mask = y[train_mask, target_class].eq(
            preds[train_mask, target_class] > threshold
        )
        if ((sum(correct_mask & ~active_mask) < 2) or
                (sum(correct_mask & active_mask) < 2)):
            return None, None, None, None

        # select correct samples from both classes
        c_target_correct = c[train_mask][correct_mask & active_mask]
        y_target_correct = y[train_mask][correct_mask & active_mask]
        c_opposite_correct = c[train_mask][correct_mask & ~active_mask]
        y_opposite_correct = y[train_mask][correct_mask & ~active_mask]

        # merge correct samples in the same dataset
        c_validation = torch.cat([c_opposite_correct, c_target_correct], dim=0)
        y_validation = torch.cat([y_opposite_correct, y_target_correct], dim=0)

        return c_validation, y_validation, correct_mask, active_mask

    def _simplify_formula(
        self,
        explanation: str,
        x: torch.Tensor,
        y: torch.Tensor,
        target_class: int,
        max_accuracy: bool,
        mask: torch.Tensor = None,
        c_threshold: float = 0.5,
        y_threshold: float = 0.
    ) -> str:
        r"""Simplify formula to a simpler one that is still coherent.

        Args:
            explanation (str): Local formula to be simplified
            x (torch.Tensor): Input data
            y (torch.Tensor): Categorical target labels(not one-hot encoded)
            target_class (int): Target class
            max_accuracy (bool): Drop term only if it gets max accuracy
            mask (torch.Tensor, optional): Drop term only if it gets
                max accuracy. (default: :obj:`None`)
            c_threshold (float, optional): Threshold to get truth values
                for concept predictions
                (i.e. :obj:`pred < threshold = False`,
                :obj:`pred > threshold = True`). (default: :obj:`0.5`)
            y_threshold (float, opitonal): threshold to get truth values
                for class predictions
                (i.e. :obj:`pred < threshold = False`,
                :obj:`pred > threshold = True`). (default: :obj:`0.`)
        rtype: :class:`str`"""

        base_accuracy = test_logic_explanation(
            explanation,
            x,
            y,
            target_class,
            mask,
            c_threshold
        )
        for term in explanation.split(" & "):
            explanation_simplified = copy.deepcopy(explanation)

            if explanation_simplified.endswith(f"{term}"):
                explanation_simplified = explanation_simplified.replace(
                    f" & {term}", ""
                )
            else:
                explanation_simplified = explanation_simplified.replace(
                    f"{term} & ", ""
                )

            if explanation_simplified:
                accuracy = test_logic_explanation(
                    explanation_simplified,
                    x,
                    y,
                    target_class,
                    mask,
                    c_threshold
                )
                if (max_accuracy and accuracy == 1.0) or (
                        not max_accuracy and accuracy >= base_accuracy
                ):
                    explanation = copy.deepcopy(explanation_simplified)
                    base_accuracy = accuracy
        return explanation

    def replace_names(
            self,
            explanation: str,
            concept_names: List[str]
    ) -> str:
        r"""Replace names of concepts in a formula.

        Args:
            explanation (str): formula where to replace names
            concept_names (List[str]): New concept names

        rtype: :class:`str`"""

        feature_abbreviations = [
            f'feature{i:010}' for i in range(len(concept_names))
        ]
        mapping = []
        for f_abbr, f_name in zip(feature_abbreviations, concept_names):
            mapping.append((f_abbr, f_name))

        for k, v in mapping:
            explanation = explanation.replace(k, v)

        return explanation
