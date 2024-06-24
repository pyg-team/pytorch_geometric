from dataclasses import dataclass
from typing import Literal, Optional

import torch
from tape.dataset.dataset import GraphDataset
from tape.gnn_model import NodeClassifier, NodeClassifierArgs

from torch_geometric.data import Data


@dataclass
class GnnTrainerArgs:
    epochs: int
    lr: float
    weight_decay: float = 0.0
    early_stopping_patience: int = 50
    device: Optional[str] = None


@dataclass
class GnnTrainerOutput:
    loss: float
    accuracy: float
    logits: torch.Tensor


class GnnTrainer:
    def __init__(self, trainer_args: GnnTrainerArgs,
                 graph_dataset: GraphDataset,
                 model_args: NodeClassifierArgs) -> None:

        self.trainer_args = trainer_args
        self.dataset: Data = graph_dataset.dataset
        self.device = trainer_args.device or (
            'cuda' if torch.cuda.is_available() else 'cpu')

        use_predictions = graph_dataset.feature_type == 'prediction'
        if use_predictions:
            # The node features will be the `topk` classes
            #   ➜ Shape: (num_nodes, topk)
            # It will get passed to an embedding lookup layer
            #   ➜ Shape: (num_nodes, topk, hidden_dim)
            # And the last two dims will be flattened
            #   ➜ Shape: (num_nodes, topk * hidden_dim)
            model_args.in_channels = graph_dataset.topk * \
                model_args.hidden_channels
        else:
            model_args.in_channels = self.dataset.num_node_features
        model_args.out_channels = graph_dataset.num_classes
        model_args.use_predictions = use_predictions
        self.model = NodeClassifier(model_args).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=trainer_args.lr,
            weight_decay=trainer_args.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self) -> GnnTrainerOutput:
        patience = self.trainer_args.early_stopping_patience
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(1, self.trainer_args.epochs + 1):
            train_output = self._train_eval(self.dataset, stage='train')
            val_output = self._train_eval(self.dataset, stage='val')
            print(f'Epoch: {epoch:03d} | Train loss: {train_output.loss:.4f}, '
                  f'Val loss: {val_output.loss:.4f}, '
                  f'Train accuracy: {train_output.accuracy:.4f}, '
                  f'Val accuracy: {val_output.accuracy:.4f}')
            if val_output.loss < best_val_loss:
                best_val_loss = val_output.loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping on epoch {epoch} due to no improvement'
                      f' in validation loss for {patience} epochs.')
                break

        output = self._train_eval(self.dataset, stage='test')
        return output

    def _train_eval(self, data: Data, stage: Literal['train', 'val', 'test']):
        if stage == 'train':
            self.model.train()
        else:
            self.model.eval()

        data = data.to(self.device)
        mask = getattr(data, f'{stage}_mask')
        if stage == 'train':
            self.optimizer.zero_grad()
            logits = self.model(data.x, data.edge_index)
            loss = self.criterion(logits[mask], data.y[mask].flatten())
            loss.backward()
            self.optimizer.step()
        else:
            with torch.inference_mode():
                logits = self.model(data.x, data.edge_index)
            loss = self.criterion(logits[mask], data.y[mask].flatten())

        accuracy = GnnTrainer.compute_accuracy(logits, data.y, mask)
        return GnnTrainerOutput(loss=float(loss), accuracy=accuracy,
                                logits=logits)

    @staticmethod
    def compute_accuracy(logits: torch.Tensor, y_true: torch.Tensor,
                         mask: torch.Tensor) -> float:
        y_pred = logits.argmax(dim=1)
        correct = y_pred[mask] == y_true[mask]
        return int(correct.sum()) / int(mask.sum())
