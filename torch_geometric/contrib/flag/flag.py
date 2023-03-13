import numbers
from typing import List, Tuple

import torch

from .flag_callback import FLAGCallback


class FLAG(torch.nn.Module):
    """
    The Free Large-scale Adversarial Augmentation on Graphs (FLAG) algorithm
    from the `Robust Optimization as Data Augmentation for Large-scale Graphs
    <https://arxiv.org/pdf/2010.09891.pdf>`_ paper.

    FLAG is a model and task free data augmentation strategy for graphs. It
    runs :code:`n_ascent_step` forward and backward passes, perturbing the
    node features, performing gradient ascent on this perturbation, and
    accumulating the gradients with respect to the model parameters. At the
    end of the steps, the model parameters are updated with these accumulated
    gradients.

    FLAG can be used in node classification, link prediction, and graph
    classification. However, this implementation is only meant for data with
    continuous node features (while the original paper handles discrete node
    features, it does so through changes to the model's :code:`forward`
    function).

    Args:
        model (torch.nn.Module): The GNN module to train.
        optimizer (torch.optim.Optimizer): The optimizer object.
        loss_fn (torch.nn.Module): The loss function that is used for
            calculating the gradients.
        device (torch.device): The device to use.
        callbacks ([FLAGCallback]): List of callbacks to apply during FLAG
            algorithm.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device,
        callbacks: List[FLAGCallback] = [],
    ) -> None:  # None return type as per https://peps.python.org/pep-0484/
        super(FLAG, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.device = device
        self.callbacks = callbacks

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y_true: torch.Tensor,
        train_idx: torch.Tensor,
        step_size: float,
        n_ascent_steps: int = 3,
    ) -> Tuple[torch.nn.Module, torch.Tensor]:
        """
        Perform the FLAG forward and backward passes. Note that users should
        not invoke :code:`loss.backward()` or :code:`optimizer.step()`
        externally to this function in their epoch loop.

        Args:
            x (torch.Tensor): Node feature matrix with shape
                [num_nodes, num_node_features].
            edge_index (torch.Tensor): Graph connectivity in COO format with
                shape [2, num_edges].
            y_true (torch.Tensor): Graph-level, node-level, or edge-level
                ground-truth labels.
            train_idx (torch.Tensor): Indices of training data.
            step_size (float): The step size to take during the gradient
                ascent on the perturbation. This is also used as the ranges
                when initializing the perturbation.
            n_ascent_steps (int): The number of forward and backward passes
                to take. Note that the FLAG paper uses 3 for experiments.

        :returns:
            - self.loss (torch.nn.Module) - The loss after the final \
                iteration of the FLAG algorithm.
            - out (torch.Tensor) - The model output on the perturbed inputs \
                from the last iteration of the FLAG algorithm.
        """

        if not isinstance(n_ascent_steps, int) or n_ascent_steps <= 0:
            raise ValueError(f"Invalid n_ascent_steps: {n_ascent_steps}." +
                             " n_ascent_steps should be a positive integer.")

        if not isinstance(step_size, numbers.Number) or step_size <= 0:
            raise ValueError(f"Invalid step_size: {step_size}." +
                             " step_size should be a positive float.")

        # Code below adapted from:
        #   https://github.com/devnkong/FLAG/blob/main/deep_gcns_torch
        #     /examples/ogb/ogbn_arxiv/main.py#L56
        #   and
        #   https://github.com/devnkong/FLAG/blob/main/deep_gcns_torch
        #     /examples/ogb/attacks.py

        training_labels = y_true.squeeze(1)[train_idx]

        self.model.train()
        self.optimizer.zero_grad()

        self.perturb = torch.nn.Parameter(
            torch.zeros(
                *x.shape,
                dtype=torch.float,
                device=self.device,
            ).uniform_(-step_size, step_size))

        [c.on_ascent_step_begin(0, None) for c in self.callbacks]

        out = self.model(x + self.perturb, edge_index)[train_idx]
        self.loss = self.loss_fn(out, training_labels)
        self.loss /= n_ascent_steps

        [
            c.on_ascent_step_end(0, self.loss, self.perturb)
            for c in self.callbacks
        ]

        for i in range(n_ascent_steps - 1):

            [c.on_ascent_step_begin(i + 1, self.loss) for c in self.callbacks]

            self.loss.backward()

            # Update the perturbation at each step using gradient ascent
            perturb_data = self.perturb.detach() + step_size * torch.sign(
                self.perturb.grad.detach())
            self.perturb.data = perturb_data.data
            self.perturb.grad[:] = 0

            # Calculate the loss on the perturbed input
            out = self.model(x + self.perturb, edge_index)[train_idx]
            self.loss = self.loss_fn(out, training_labels)

            self.loss /= n_ascent_steps

            [
                c.on_ascent_step_end(i + 1, self.loss, self.perturb)
                for c in self.callbacks
            ]

        [c.on_optimizer_step_begin(self.loss) for c in self.callbacks]

        self.loss.backward()
        self.optimizer.step()

        [c.on_optimizer_step_end(self.loss) for c in self.callbacks]

        # Users should not be invoking loss.backward() or optimizer.step()
        # externally to this function in their epoch loop.
        #
        # However, we zero out the gradients in the optimizer here to
        # mitigate the undesirable effects if they do.
        self.optimizer.zero_grad()

        return self.loss, out

    def get_model(self) -> torch.nn.Module:
        """
        :returns:
            - self.model (torch.nn.Module): a reference to the underlying \
                model that was trained by the FLAG module.
        """
        return self.model
