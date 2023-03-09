import numbers

import torch

from .flag_callback import FLAGCallback

class FLAG(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.nn.Module,
        loss_fn: torch.nn.Module,
        device: torch.device,
        callbacks: [FLAGCallback] = [],
    ) -> None:  # None return type as per https://peps.python.org/pep-0484/
        super(FLAG, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.device = device
        self.callbacks = callbacks

    def forward(
        self,
        x: torch.tensor,
        edge_index: torch.tensor,
        y_true: torch.tensor,
        train_idx: torch.tensor,
        step_size: float,
        n_ascent_steps: int = 3,
    ) -> tuple[torch.nn.Module, torch.tensor]:
        if not isinstance(n_ascent_steps, int) or n_ascent_steps <= 0:
            raise ValueError(f"Invalid n_ascent_steps: {n_ascent_steps}." +
                             " n_ascent_steps should be a positive integer.")

        if not isinstance(step_size, numbers.Number) or step_size <= 0:
            raise ValueError(f"Invalid step_size: {step_size}." +
                             " step_size should be a positive float.")

        # Code below adapted from:
        #   https://github.com/devnkong/FLAG/blob/main/deep_gcns_torch
        #     /examples/ogb/ogbn_arxiv/main.py#L56
        #
        # This example does not assume that the model forward function has a
        #   `perturb` argument
        # forward = lambda perturb : model(x+perturb, edge_index)[train_idx]
        # model_forward = (model, forward)

        # TODO
        # Tyler, did you mean to have .squeeze(1)??
        training_labels = y_true[train_idx]

        # loss, out = flag(model_forward, x.shape, target,
        #   args, optimizer, device, F.nll_loss)
        #
        # return loss.item()

        ########################################################

        # Code below adapted from:
        #   https://github.com/devnkong/FLAG/blob/main/deep_gcns_torch
        #     /examples/ogb/attacks.py
        #
        # model, forward = model_forward
        self.model.train()
        self.optimizer.zero_grad()

        # perturb = torch.FloatTensor(*x.shape).uniform_(
        #   -args.step_size, args.step_size).to(self.device)

        # perturb = torch.FloatTensor(*x.shape).uniform_(
        #     -step_size, step_size).to(self.device)
        # perturb.requires_grad_()

        # TODO
        # If this implementation doesn't work as we expect, try changing
        # this back to to original implementation using tensor.uniform_()
        self.perturb = torch.nn.Parameter(
            torch.zeros(
                *x.shape,
                dtype=torch.float,
                device=self.device,
            ).uniform_(-step_size, step_size))

        [c.on_ascent_step_begin(0, None) for c in self.callbacks]

        # out = forward(perturb)
        out = self.model(x + self.perturb, edge_index)[train_idx]
        self.loss = self.loss_fn(out, training_labels)

        # loss /= args.m
        self.loss /= n_ascent_steps

        [c.on_ascent_step_end(0, self.loss, self.perturb) for c in self.callbacks]

        # for _ in range(args.m - 1):
        for i in range(n_ascent_steps - 1):
            [c.on_ascent_step_begin(i + 1, self.loss) for c in self.callbacks]

            self.loss.backward()
            # perturb_data = perturb.detach() + args.step_size *
            #   torch.sign(perturb.grad.detach())
            perturb_data = self.perturb.detach() + step_size * torch.sign(
                self.perturb.grad.detach())
            self.perturb.data = perturb_data.data
            self.perturb.grad[:] = 0

            # out = forward(perturb)
            out = self.model(x + self.perturb, edge_index)[train_idx]
            self.loss = self.loss_fn(out, training_labels)

            # loss /= args.m
            self.loss /= n_ascent_steps

            [c.on_ascent_step_end(i + 1, self.loss, self.perturb) for c in self.callbacks]


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
        returns: a reference to the underlying model that was trained by
                 the FLAG module.
        """
        return self.model

