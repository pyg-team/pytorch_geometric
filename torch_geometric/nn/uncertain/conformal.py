from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .calibrate import ETS, GATS, TS, VS, BaseCalibration
from .predict import aps, cqr, qr, raps, tps


@dataclass
class ConformalPrediction:
    """Container for conformal prediction results.

    :param prediction_sets: The predicted set or interval for each sample
    :type prediction_sets: Union[np.ndarray, List[np.ndarray]]
    :param coverage: Empirical coverage on validation set
    :type coverage: float
    :param efficiency: Average size of prediction sets
    :type efficiency: float
    :param confidence_scores: Optional confidence scores for predictions
    :type confidence_scores: Optional[np.ndarray]
    """

    prediction_sets: Union[np.ndarray, List[np.ndarray]]
    coverage: float
    efficiency: float
    confidence_scores: Optional[np.ndarray] = None


class Conformal:
    """A wrapper class for conformal prediction methods.

    :param model: The base model to make predictions
    :type model: torch.nn.Module
    :param prediction: The strategy ('aps', 'tps', 'raps', 'cqr', 'qr')
    :type prediction: str
    :param calibration: The method ('ts', 'vs', 'ets', 'gats')
    :type calibration: str, optional
    :param device: Device to use for computation ('cpu' or 'cuda')
    :type device: str
    """
    def __init__(
        self,
        model: torch.nn.Module,
        prediction: str,
        calibration: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.prediction = prediction.lower()
        self.calibration = calibration.lower() if calibration else None

        # Initialize state
        self.is_fitted = False
        self.alpha = None
        self.n_calibration = None
        self.n_validation = None

    def _initialize_prediction(self) -> callable:
        """Initialize the prediction strategy."""
        prediction_methods = {
            "aps": aps,
            "tps": tps,
            "raps": raps,
            "cqr": cqr,
            "qr": qr,
        }

        if self.prediction not in prediction_methods:
            raise ValueError(f"prediction must be one of "
                             f"{list(prediction_methods.keys())}")

        method = prediction_methods[self.prediction]
        return method

    def _initialize_calibration(self) -> Optional[BaseCalibration]:
        """Initialize the calibration model if specified."""
        if not self.calibration:
            return None

        calibration_methods = {"ts": TS, "vs": VS, "ets": ETS, "gats": GATS}

        if self.calibration not in calibration_methods:
            raise ValueError(f"calibration must be one of "
                             f"{list(calibration_methods.keys())}")

        method = calibration_methods[self.calibration]
        return method(self.model, device=self.device)

    def fit(
        self,
        calibration_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        """Fit the conformal predictor using calibration data.

        Args:
            calibration_data: Input data containing features and labels
            alpha: The desired miscoverage level (default: 0.1)
            **kwargs: Additional arguments for calibration and prediction
                Required for specific calibration methods:
                - num_classes: Required for VS, ETS
                - num_nodes: Required for GATS
                - edge_index: Required for GATS
                - train_mask: Required for calibration
                - test_mask: Required for calibration
                Optional parameters:
                - heads: Optional for GATS (default: 8)
                - bias: Optional for GATS (default: 1.0)
                - wdecay: Weight decay for optimization (default: 0.01)
        """
        # Initialize and fit calibration model if specified
        if self.calibration:
            self.calibrated_model = self._initialize_calibration()
            self.calibrated_model.fit(data=calibration_data, **kwargs)
        else:
            self.calibrated_model = None

        self.is_fitted = True

    def predict(
        self,
        data: object,
        calibration,
        validation,
        labels,
        alpha: float = 0.1,
        **kwargs,
    ) -> ConformalPrediction:
        """Make predictions with conformal prediction intervals.

        :param data: Input data object containing features and graph structure
        :type data: object
        :param calibration: Mask for calibration set
        :type calibration: torch.Tensor
        :param validation: Mask for validation set
        :type validation: torch.Tensor
        :param labels: Ground truth labels
        :type labels: torch.Tensor
        :param alpha: Desired miscoverage level
        :type alpha: float
        :param kwargs: Additional arguments for specific prediction methods
        :type kwargs: dict
        :return: Conformal prediction results
        :rtype: ConformalPrediction
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Conformal predictor must be fitted before making predictions")

        # Initialize alpha
        self.alpha = alpha

        # Initialize prediction strategy
        prediction_strategy = self._initialize_prediction()

        # Use calibrated model if available, otherwise use base model
        model = self.calibrated_model if self.calibrated_model else self.model

        # Get masks and compute sizes
        calibration_mask = calibration
        validation_mask = validation
        self.n_calibration = calibration_mask.sum().item()
        self.n_validation = validation_mask.sum().item()

        # Get predictions
        with torch.no_grad():
            pred = model(data.x, data.edge_index)

        # Handle classification predictions (aps, tps, raps)
        if self.prediction in ["aps", "tps", "raps"]:
            logits = torch.nn.Softmax(dim=1)(pred).cpu().numpy()

            prediction_sets, coverage, efficiency = prediction_strategy(
                cal_smx=logits[calibration_mask],
                val_smx=logits[validation_mask],
                cal_labels=labels[calibration_mask].cpu().numpy(),
                val_labels=labels[validation_mask].cpu().numpy(),
                n=self.n_calibration,
                alpha=self.alpha,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ["lam_reg", "k_reg", "rand", "disallow_zero_sets"]
                },
            )

            confidence_scores = logits.max(axis=1)

        # Handle regression predictions (cqr, qr)
        else:
            pred = pred.cpu().numpy()

            prediction_sets, coverage, efficiency = prediction_strategy(
                cal_labels=labels[calibration_mask].cpu().numpy(),
                cal_lower=pred[calibration_mask][:, 0],
                cal_upper=pred[calibration_mask][:, 1],
                val_labels=labels[validation_mask].cpu().numpy(),
                val_lower=pred[validation_mask][:, 0],
                val_upper=pred[validation_mask][:, 1],
                n=self.n_calibration,
                alpha=self.alpha,
            )

            confidence_scores = None

        return ConformalPrediction(
            prediction_sets=prediction_sets,
            coverage=coverage,
            efficiency=efficiency,
            confidence_scores=confidence_scores,
        )
