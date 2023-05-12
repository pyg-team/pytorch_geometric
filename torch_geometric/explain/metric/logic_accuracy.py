from typing import List, Tuple, Union

import torch

from sklearn.metrics import accuracy_score
from sympy import to_dnf, lambdify

from torch_geometric.explain.explanation import (
    LogicExplanation,
    LogicExplanations
)


def test_logic_explanation(
    explanation: Union[str, LogicExplanation],
    x: torch.Tensor,
    y: torch.Tensor,
    target_class: int,
    mask: torch.Tensor = None,
    threshold: float = 0.5,
    material: bool = False,
) -> Tuple[float, torch.Tensor]:
    r"""Compute the accuracy of an input logic formula.
        Args:
            explanation (Union[str, LogicExplanation]): The
                logic formula to be evaluated
            x (torch.Tensor): input data
            y (torch.Tensor): input labels (MUST be one-hot encoded)
            target_class (int): target class
            mask (torch.Tensor, optional): sample mask. (default: :obj:`None`)
            threshold (float, optional): threshold to get concept truth values.
                (default: :obj:`0.5`)
            material (bool, optional): if :obj:`True` the formula is evaluated
                via a material implication, material biconditional otherways.
                (default: :obj:`False`)

        :rtype: :class:`float`"""

    if isinstance(explanation, LogicExplanation):
        formula = explanation.explanation
    else:
        formula = explanation

    if mask is None:
        mask = torch.arange(x.shape[0]).long()

    if formula in ['True', 'False', ''] or formula is None:
        return 0.0
    else:
        assert len(y.shape) == 2
        y2 = y[:, target_class]
        concept_list = [f"feature{i:010}" for i in range(x.shape[1])]
        # get predictions using sympy
        explanation = to_dnf(formula)
        fun = lambdify(concept_list, explanation, 'numpy')
        x = x.cpu().detach().numpy()
        preds = fun(*[x[:, i] > threshold for i in range(x.shape[1])])
        preds = torch.LongTensor(preds)
        if material:
            # material implication: (p=>q) <=> (not p or q)
            accuracy = torch.sum(
                torch.logical_or(
                    torch.logical_not(preds[mask]),
                    y2[mask]
                )
            ) / len(y2[mask])
            accuracy = accuracy.item()
        else:
            # material biconditional:
            # (p<=>q) <=> (p and q) or (not p and not q)
            accuracy = accuracy_score(preds[mask], y2[mask])
        return accuracy


def formulas_accuracy(
    explanations: Union[LogicExplanations, List[str]],
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor = None,
    threshold: float = 0.5
) -> Tuple[float, torch.Tensor]:
    r"""Test all together the logic formulas of different classes.
    When a sample fires more than one formula, consider the sample
    as wrongly predicted.

        Args:
            explanations (Union[LogicExplanations, List[str]]): The
            logic formulas to be evaluated, one for each class
            x (torch.Tensor): input data
            y (torch.Tensor): input labels (MUST be one-hot encoded)
            mask (torch.Tensor, optional): sample mask. (default: :obj:`None`)
            threshold (float): threshold to get concept truth values.
            (default: :obj:`0.5`)

        :rtype: :class:`float`"""

    if isinstance(explanations, LogicExplanations):
        formulas = explanations.get_formulas()
    else:
        formulas = explanations

    if mask is None:
        mask = torch.arange(x.shape[0]).long()

    if formulas is None or formulas == []:
        return 0.0
    for formula in formulas:
        if formula in ['True', 'False', '']:
            return 0.0
    assert len(y.shape) == 2

    y2 = y.argmax(-1)
    x = x.cpu().detach().numpy()
    concept_list = [f"feature{i:010}" for i in range(x.shape[1])]

    # get predictions using sympy
    class_predictions = torch.zeros(len(formulas), x.shape[0])
    for i, formula in enumerate(formulas):
        explanation = to_dnf(formula)
        fun = lambdify(concept_list, explanation, 'numpy')

        predictions = fun(*[x[:, i] > threshold for i in range(x.shape[1])])
        predictions = torch.LongTensor(predictions)
        class_predictions[i] = predictions

    class_predictions_filtered = torch.zeros(class_predictions.shape[1])
    for i in range(class_predictions.shape[1]):
        if sum(class_predictions[:, i]) != 1:
            class_predictions_filtered[i] = -1  # consider as an error
        else:
            class_predictions_filtered[i] = class_predictions[:, i].argmax(-1)

    # material biconditional:
    # (p<=>q) <=> (p and q) or (not p and not q)
    accuracy = accuracy_score(
        class_predictions_filtered[mask],
        y2[mask]
    )
    return accuracy
