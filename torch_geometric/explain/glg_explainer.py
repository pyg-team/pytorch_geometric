from typing import Dict, Optional, Tuple
from collections import defaultdict
import math

import torch
from sympy import to_dnf, lambdify
from torch import Tensor
from torch_geometric.explain.metric import (
    formulas_accuracy
)
from torch_geometric.explain import LogicExplanation, LogicExplanations
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
import torch.nn.functional as F


class GLGExplainer(torch.nn.Module):
    r"""Implementation of GLGExplainer `"Global Explainability of GNNs via
    Logic Combination of Learned Concepts"
    <https://openreview.net/forum?id=OTbRTIY4YS>`_

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
        len_lr (float, optional): Learning rate for the parameters of the
            E-LEN. (default :obj:`0.0005`)
        prototypes_lr (float, optional): Learning rate for the prototypes.
            (default :obj:`0.001`)
        le_embedder_lr (float, optional): Learning rate for the parameters
            of the :attr:`le_embedder`. (default :obj:`0.001`)
        coeff_r1 (float, optional): Scaling weight for the distance loss
            :obj:`R1`. (default :obj:`0.09`)
        coeff_r2 (float, optional): Scaling weight for the distance loss
            :obj:`R2`. (default :obj:`0.00099`)
    """
    def __init__(
        self,
        le_embedder: torch.nn.Module,
        num_classes: int,
        num_prototypes: int,
        dim_prototypes: Optional[int] = 10,
        dim_len_hidden: Optional[int] = 10,
        len_lr: Optional[float] = 0.0005,
        prototypes_lr: Optional[float] = 0.001,
        le_embedder_lr: Optional[float] = 0.001,
        coeff_r1: Optional[float] = 0.09,
        coeff_r2: Optional[float] = 0.00099,
    ):
        super().__init__()

        self.le_embedder = le_embedder
        self.num_classes = num_classes
        self.coeff_r1 = coeff_r1
        self.coeff_r2 = coeff_r2
        self.global_explanation = LogicExplanations()
        self.len_loss = self._resolve_len_loss()

        self.len_model = _ELEN(
            input_shape=num_prototypes,
            hidden_dim=dim_len_hidden,
            n_classes=num_classes
        )
        self.prototype_vectors = torch.nn.Parameter(
            torch.rand((num_prototypes,
                        dim_prototypes)),
            requires_grad=True
        )
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

    def do_epoch(self, loader: DataLoader, train: bool = False) -> Dict:
        r"""Perform one-epoch training.

        Args:
            loader (:class:`DataLoader`): Input data.
            train (bool, optional): If :obj:`False`, no training
                is perforfmed, and only the resulting metrics are returned.
                (default :obj:`False`)
        rtype: :class:`Dict`"""

        if train:
            self.train()
        else:
            self.eval()

        device = self.prototype_vectors.device
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

        device = self.prototype_vectors.device
        le_embs = torch.tensor([], device=device)
        belonging = torch.tensor([], device=device, dtype=torch.long)
        y = torch.tensor([], device=device)
        le_lbl = torch.tensor([], device=device)
        le_idxs = torch.tensor([], device=device)
        for data in loader:
            data = data.to(device)
            le_idxs = torch.concat([le_idxs, data.le_id], dim=0)
            embs = self.le_embedder(data.x, data.edge_index, data.batch)
            le_embs = torch.concat([le_embs, embs], dim=0)
            belonging = torch.concat([belonging, data.graph_id], dim=0)
            le_lbl = torch.concat([le_lbl, data.le_label], dim=0)
            y = torch.concat([y, data.y], dim=0)

        y = scatter(y, belonging, dim=0, reduce="max")
        y = torch.nn.functional.one_hot(y.long()).float().to(device)
        assign = self.prototype_assignement(le_embs)
        concept_vec = scatter(assign, belonging, dim=0, reduce="max")
        return (concept_vec, le_embs, assign, y, le_lbl, le_idxs, belonging)

    @torch.no_grad()
    def inspect(
        self,
        loader: DataLoader,
        update_global_explanation: Optional[bool] = False,
    ) -> Dict:
        r"""Compute evaluation metrics for the explainer. Optionally,
        visualize the embedding with the learned prototypes.

        Args:
            loader (:class:`~torch_geometric.loader.DataLoader`): Input data.
            update_global_explanation (bool, optional): If :obj:`True`,
                the stored Global Explanation is updated by extracting new
                formulas. If :obj:`False`, :attr:`self.global_explanation`
                is evaluated. (default :obj:`False`)
        rtype: :class:`Dict`"""
        self.eval()
        (
            x, emb, concepts_assignement, y_1h, le_classes, _, _,
        ) = self.get_concept_vector(loader)
        y_pred = self.len_model(x).squeeze(-1).cpu()

        emb = emb.detach().cpu().numpy()
        concept_predictions = concepts_assignement.argmax(1).cpu()

        device = self.prototype_vectors.device
        self.len_model.to("cpu")
        x = x.detach().cpu()
        y_1h = y_1h.cpu()

        if update_global_explanation:
            self.global_explanation = self.len_model.explain_classes(
                x, y_1h, self.num_classes
            )
        accuracy = formulas_accuracy(self.global_explanation, x, y_1h)
        cluster_accs = self.concept_purity(concept_predictions, le_classes)
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
        loader: DataLoader
    ) -> Tuple[Tensor, Tensor]:
        r"""Returns the neural preictions of the E-LEN and the predictions of
        the extracted Global Logic Explanations

        Args:
            loader (:class:`~torch_geometric.loader.DataLoader`): Input data

        type: :class:`Tuple[Tensor, Tensor]`"""
        assert self.global_explanation != LogicExplanations, \
            "No Global Explanation available." \
            "You have to first train the model"

        self.eval()
        x, _, _, _, _, _, _, = self.get_concept_vector(loader)
        y_pred = self.len_model(x).squeeze(-1).argmax(1)

        # formula predictions
        x = x.cpu().detach().numpy()
        concept_list = [f"feature{i:010}" for i in range(x.shape[1])]
        class_predictions = torch.zeros(x.shape[0], self.num_classes)
        for i, formula in enumerate(self.global_explanation.get_formulas()):
            f = lambdify(concept_list, to_dnf(formula), 'numpy')
            class_predictions[:, i] = torch.LongTensor(
                f(*[x[:, i] > 0.5 for i in range(x.shape[1])])
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
        dist = torch.cdist(le_embeddings, self.prototype_vectors, p=2)**2
        sim = torch.log((dist + 1) / (dist + 1e-6))
        y_soft = F.softmax(sim, dim=-1)

        # reparametrization trick from Gumbel-Softmax
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        le_assignments = y_hard - y_soft.detach() + y_soft
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
        ret = super().load_state_dict(kwargs)
        assert ret[0] == [] and ret[1] == [], \
            "Some keys in the state dict were not succesfully matched"

        # Extract and store the resulting Global Explanation
        # for the loaded model
        (
            x, _, _, y_1h, _, _, _,
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
        sample_prototype_distance = torch.cdist(
            le_embeddings,
            self.prototype_vectors,
            p=2
        )**2  # num_sample x num_prototypes

        # R1 loss: push each prototype to be close to at least one example
        if self.coeff_r1 > 0:
            min_proto_sample_dist = sample_prototype_distance.T.min(-1).values
            avg_prototype_sample_distance = torch.mean(min_proto_sample_dist)
            r1_loss = self.coeff_r1 * avg_prototype_sample_distance
            total_losses["r1_loss"] += r1_loss.detach()
        else:
            r1_loss = torch.tensor(0., device=device)

        # R2 loss: Push every example to be close to a sample
        if self.coeff_r2 > 0:
            min_proto_sample_dist = sample_prototype_distance.min(-1).values
            avg_sample_prototype_distance = torch.mean(min_proto_sample_dist)
            r2_loss = self.coeff_r2 * avg_sample_prototype_distance
            total_losses["r2_loss"] += r2_loss.detach()
        else:
            r2_loss = torch.tensor(0., device=device)

        loss = len_loss + r1_loss + r2_loss
        total_losses["loss"] += loss.detach()
        return loss

    def _resolve_len_loss(self):
        if self.num_classes == 2:
            return F.binary_cross_entropy_with_logits
        elif self.num_classes == 3:
            return F.cross_entropy
        else:
            raise NotImplementedError(f"num_classes implemented <= 3"
                                      f"got {self.num_classes}.")

    @staticmethod
    def normalize_belonging(belonging):
        ret = []
        i = -1
        for j, elem in enumerate(belonging):
            if not (len(ret) > 0 and elem == belonging[j - 1]):
                i += 1
            ret.append(i)
        return ret

    def _entropy(self, logits):
        logp = torch.log(logits + 1e-10)
        entropy = torch.mean(torch.sum(-logits * logp, dim=1))
        return entropy


class _EntropyLinear(torch.nn.Module):
    r"""Adaptation of the implementation of an Entropy-based
    Logic Explained Network from `"torch_explain"
    <https://pypi.org/project/torch-explain/>`_ ."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            n_classes: int,
            temperature: float
    ) -> None:
        super(_EntropyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes
        self.temp = temperature
        self.weight = torch.nn.Parameter(
            torch.Tensor(n_classes, out_features, in_features)
        )
        self.bias = torch.nn.Parameter(
            torch.Tensor(n_classes, 1, out_features)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
            self.weight
        )
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if len(input.shape) == 2:
            input = input.unsqueeze(0)

        gamma = self.weight.norm(dim=1, p=1)
        self.alpha = torch.exp(gamma / self.temp) \
            / torch.sum(torch.exp(gamma / self.temp), dim=1, keepdim=True)
        self.alpha_norm = self.alpha / self.alpha.max(dim=1)[0].unsqueeze(1)
        self.concept_mask = torch.ones_like(self.alpha_norm, dtype=bool)
        x = input.matmul(self.weight.permute(0, 2, 1)) + self.bias
        return x.permute(1, 0, 2)


class _ELEN(torch.nn.Module):
    def __init__(
            self, input_shape, n_classes, temperature=1, hidden_dim=10,
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(*[
            _EntropyLinear(
                input_shape,
                hidden_dim,
                n_classes=n_classes,
                temperature=temperature,
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
            val_mask: torch.Tensor = None
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
        rtype: :class:`LogicExplanation`"""

        if train_mask is None:
            train_mask = torch.arange(c.shape[0]).long()
        if val_mask is None:
            val_mask = torch.arange(c.shape[0]).long()
        c_correct, y_correct = self._get_correct_data(
            c,
            y,
            train_mask,
            target_class
        )
        if c_correct is None:
            return LogicExplanation("")

        feature_names = [f"feature{j:010}" for j in range(c_correct.size(1))]
        class_explanation = ""
        local_expls_raw = {}
        for _, module in enumerate(self.layers.children()):
            if isinstance(module, _EntropyLinear):
                # look at the "positive" rows of the truth table only
                positive_samples = torch.nonzero(
                    y_correct[:, target_class]
                ).numpy().ravel()

                for positive_sample in positive_samples:
                    le_raw = self._local_explanation(
                        module,
                        feature_names,
                        positive_sample,
                        c_correct,
                        target_class
                    )
                    if (le_raw and le_raw not in local_expls_raw):
                        local_expls_raw[le_raw] = le_raw

                explanations = ["(" + e + ")" for e in local_expls_raw.keys()]
                class_explanation = "(" + " | ".join(explanations) + ")"
                break
        return LogicExplanation(class_explanation[1:-1])

    def _local_explanation(
        self,
        module,
        feature_names,
        neuron_id,
        c_validation,
        target_class,
        c_threshold=0.5,
    ):
        # explanation is the conjunction of non-pruned features
        explanation_raw = ""
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
                    explanation_raw += feature_names[j]
                else:
                    explanation_raw += f"~{feature_names[j]}"

        explanation_raw = str(explanation_raw)
        if explanation_raw in ["", "False", "True", "(False)", "(True)"]:
            return None, None

        return explanation_raw

    def _get_correct_data(
            self, c, y, train_mask, target_class, threshold=0.
    ):
        active_mask = y[train_mask, target_class] == 1
        preds = self.layers(c).squeeze(-1)

        # identify samples correctly classified of the target class
        correct_mask = y[train_mask, target_class].eq(
            preds[train_mask, target_class] > threshold
        )
        if ((sum(correct_mask & ~active_mask) < 2) or
                (sum(correct_mask & active_mask) < 2)):
            return None, None

        # select correct samples from both classes
        c_target_correct = c[train_mask][correct_mask & active_mask]
        y_target_correct = y[train_mask][correct_mask & active_mask]
        c_opposite_correct = c[train_mask][correct_mask & ~active_mask]
        y_opposite_correct = y[train_mask][correct_mask & ~active_mask]

        # merge correct samples in the same dataset
        c_validation = torch.cat([c_opposite_correct, c_target_correct], dim=0)
        y_validation = torch.cat([y_opposite_correct, y_target_correct], dim=0)

        return c_validation, y_validation
