from typing import Any, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from torch_geometric.nn import GATConv


class CalibAttentionLayer(nn.Module):
    """Graph Attention Layer specialized for calibration.

    :param in_channels: Number of input features
    :type in_channels: int
    :param out_channels: Number of output features
    :type out_channels: int
    :param edge_index: Graph connectivity in COO format
    :type edge_index: torch.Tensor
    :param num_nodes: Number of nodes in the graph
    :type num_nodes: int
    :param train_mask: Mask indicating training nodes
    :type train_mask: torch.Tensor
    :param heads: Number of attention heads
    :type heads: int
    :param bias: Initial bias value
    :type bias: float
    :param dist_to_train: Optional distance to training nodes
    :type dist_to_train: Optional[torch.Tensor]
    :param device: Device to use for computation
    :type device: str
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_index: torch.Tensor,
        num_nodes: int,
        train_mask: torch.Tensor,
        heads: int = 8,
        bias: float = 1.0,
        dist_to_train: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=True,
            bias=True,
        ).to(device)

        self.linear = nn.Linear(heads, 1, bias=True).to(device)
        self.bias = nn.Parameter(torch.ones(1) * bias)
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.train_mask = train_mask
        self.dist_to_train = dist_to_train
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the calibration attention layer."""
        # Apply GAT and Linear Layer
        h = self.gat(x, self.edge_index)
        temperature = F.relu(self.linear(h)) + self.bias

        # Apply distance-based scaling if available
        if self.dist_to_train is not None:
            temperature = temperature * (1 + self.dist_to_train.view(-1, 1))

        return temperature


class BaseCalibration(nn.Module):
    """Base class for all calibration methods.

    :param model: The underlying model to be calibrated
    :type model: nn.Module
    :param device: Device to use for computation ('cpu' or 'cuda')
    :type device: str
    """
    def __init__(self, model: nn.Module, device: str):
        super().__init__()
        self.model = model
        self.device = device
        self.is_fitted = False

    def validate_fit_data(self, data: Any, **kwargs) -> None:
        """Validate the data and parameters required for fitting."""
        if (not hasattr(data, "x") or not hasattr(data, "edge_index")
                or not hasattr(data, "y")):
            raise ValueError("data must have attributes: x, edge_index, and y")

        if "train_mask" not in kwargs:
            raise ValueError("Missing parameter: train_mask")


class TS(BaseCalibration):
    """Temperature Scaling calibration method.

    Applies a single temperature parameter to scale model logits.

    :param model: The underlying model to be calibrated
    :type model: nn.Module
    :param device: Device to use for computation ('cpu' or 'cuda')
    :type device: str
    """
    def __init__(self, model: nn.Module, device: str):
        super().__init__(model, device)
        self.temperature = nn.Parameter(torch.ones(1))

    def validate_fit_data(self, data: Any, **kwargs) -> None:
        # TS doesn't require additional parameters
        super().validate_fit_data(data, **kwargs)

    def fit(self, data: Any, **kwargs) -> "TS":
        """Fit the temperature scaling model."""
        self.validate_fit_data(data, **kwargs)
        self.to(self.device)

        train_mask = kwargs["train_mask"]
        wdecay = kwargs.get("wdecay", 0.01)

        optimizer = optim.Adam(
            [self.temperature],
            lr=0.01,
            weight_decay=wdecay,
        )
        fit_calibration(self, optimizer, data, train_mask)
        return self

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Apply temperature scaling to the model's logits."""
        logits = self.model(x, edge_index)
        return logits / self.temperature.expand(logits.size())


class VS(BaseCalibration):
    """Vector Scaling calibration method.

    Applies a vector of temperatures and biases to scale model logits.

    :param model: The underlying model to be calibrated
    :type model: nn.Module
    :param device: Device to use for computation ('cpu' or 'cuda')
    :type device: str
    """
    def __init__(self, model: nn.Module, device: str):
        super().__init__(model, device)
        self.temperature = None
        self.bias = None

    def validate_fit_data(self, data: Any, **kwargs) -> None:
        super().validate_fit_data(data, **kwargs)
        if "num_classes" not in kwargs:
            raise ValueError("VS requires num_classes parameter")

        num_classes = kwargs["num_classes"]
        self.temperature = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def fit(self, data: Any, **kwargs) -> "VS":
        """Fit the vector scaling model."""
        self.validate_fit_data(data, **kwargs)
        self.to(self.device)

        train_mask = kwargs["train_mask"]
        wdecay = kwargs.get("wdecay", 0.01)

        optimizer = optim.Adam([self.temperature, self.bias], lr=0.01,
                               weight_decay=wdecay)
        fit_calibration(self, optimizer, data, train_mask)
        return self

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Apply vector scaling to the model's logits."""
        logits = self.model(x, edge_index)
        return logits * self.temperature + self.bias


class ETS(BaseCalibration):
    """Ensemble Temperature Scaling calibration.

    Combines multiple calibration methods through learned weights.

    :param model: The underlying model to be calibrated
    :type model: nn.Module
    :param device: Device to use for computation ('cpu' or 'cuda')
    :type device: str
    """
    def __init__(self, model: nn.Module, device: str):
        super().__init__(model, device)
        self.temp_model = None
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.zeros(1))
        self.weight3 = nn.Parameter(torch.zeros(1))

    def validate_fit_data(self, data: Any, **kwargs) -> None:
        super().validate_fit_data(data, **kwargs)
        if "num_classes" not in kwargs:
            raise ValueError("ETS requires num_classes parameter")

        self.num_classes = kwargs["num_classes"]
        self.temp_model = TS(self.model, self.device)

    def ensemble_scaling(self, logits: np.ndarray, labels: np.ndarray,
                         temp: float) -> np.ndarray:
        """Compute optimal ensemble weights."""
        from scipy import optimize

        # Compute the three probability distributions
        p1 = torch.softmax(torch.tensor(logits), dim=1)[:, None].numpy()
        p0 = torch.softmax(torch.tensor(logits) / temp, dim=1)[:, None].numpy()
        p2 = np.ones_like(p0) / self.num_classes

        # Define optimization constraints
        bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        # Optimize weights using cross-entropy loss
        result = optimize.minimize(
            self._cross_entropy_loss,
            x0=(1.0, 0.0, 0.0),
            args=(p0, p1, p2, labels),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=1e-12,
            options={"disp": False},
        )

        return result.x

    @staticmethod
    def _cross_entropy_loss(weights: np.ndarray, *args) -> float:
        """Compute cross-entropy loss for ensemble weights."""
        p0, p1, p2, labels = args
        p = weights[0] * p0 + weights[1] * p1 + weights[2] * p2
        return -np.sum(labels * np.log(p)) / p.shape[0]

    def fit(self, data: Any, **kwargs) -> "ETS":
        """Fit the ensemble temperature scaling model."""
        self.validate_fit_data(data, **kwargs)
        self.to(self.device)

        # First fit the temperature scaling model
        self.temp_model.fit(data, **kwargs)

        train_mask = kwargs["train_mask"]
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)[train_mask]
            labels = data.y[train_mask]
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, labels.unsqueeze(-1), 1)

        temp = self.temp_model.temperature.cpu().detach().numpy()
        w = self.ensemble_scaling(logits.cpu().detach().numpy(),
                                  one_hot.cpu().detach().numpy(), temp)

        self.weight1.data = torch.tensor(w[0])
        self.weight2.data = torch.tensor(w[1])
        self.weight3.data = torch.tensor(w[2])

        self.is_fitted = True
        return self

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ensemble temperature scaling to the model's logits."""
        logits = self.model(x, edge_index)
        temp = self.temp_model.temperature

        # Compute ensemble prediction
        p = (self.weight1 * F.softmax(logits / temp, dim=1) +
             self.weight2 * F.softmax(logits, dim=1) + self.weight3 *
             (1 / self.num_classes))

        return torch.log(p)


class GATS(BaseCalibration):
    """Graph Attention Temperature Scaling calibration.

    Uses graph attention networks to learn node-specific temperature scaling.

    :param model: The underlying model to be calibrated
    :type model: nn.Module
    :param device: Device to use for computation ('cpu' or 'cuda')
    :type device: str
    """
    def __init__(self, model: nn.Module, device: str):
        super().__init__(model, device)
        self.cagat = None

    def validate_fit_data(self, data: Any, **kwargs) -> None:
        super().validate_fit_data(data, **kwargs)
        required = ["num_nodes", "num_classes", "edge_index"]
        missing = [param for param in required if param not in kwargs]
        if missing:
            raise ValueError(f"GATS requires parameters: {missing}")

        self.cagat = CalibAttentionLayer(
            in_channels=kwargs["num_classes"],
            out_channels=1,
            edge_index=kwargs["edge_index"],
            num_nodes=kwargs["num_nodes"],
            train_mask=kwargs["train_mask"],
            dist_to_train=kwargs.get("dist_to_train"),
            heads=kwargs.get("heads", 8),
            bias=kwargs.get("bias", 1.0),
            device=self.device,
        )

    def fit(self, data: Any, **kwargs) -> "GATS":
        """Fit the graph attention temperature scaling model."""
        self.validate_fit_data(data, **kwargs)
        self.to(self.device)

        train_mask = kwargs["train_mask"]
        wdecay = kwargs.get("wdecay", 0.01)

        optimizer = optim.Adam(
            self.cagat.parameters(),
            lr=0.01,
            weight_decay=wdecay,
        )
        fit_calibration(self, optimizer, data, train_mask)
        return self

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Apply graph attention temperature scaling to the model's logits."""
        logits = self.model(x, edge_index)
        temperature = self.cagat(logits).view(logits.size(0), -1)
        return logits / temperature.expand(logits.size())


def fit_calibration(
    model: BaseCalibration,
    optimizer: optim.Optimizer,
    data: Any,
    train_mask: torch.Tensor,
    patience: int = 100,
    max_epochs: int = 2000,
) -> None:
    """Train a calibration model using early stopping.

    :param model: Calibration model to train
    :type model: BaseCalibration
    :param optimizer: Optimizer for training
    :type optimizer: optim.Optimizer
    :param data: Training data
    :type data: Any
    :param train_mask: Mask indicating training nodes
    :type train_mask: torch.Tensor
    :param patience: Number of epochs to wait for improvement
    :type patience: int
    :param max_epochs: Maximum number of training epochs
    :type max_epochs: int
    """
    with torch.no_grad():
        labels = data.y

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        model.train()
        model.model.eval()

        calibrated = model(data.x, data.edge_index)
        loss = F.cross_entropy(calibrated[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
