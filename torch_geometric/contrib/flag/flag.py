import torch


class FLAG(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.nn.Module,
        loss_fn: torch.nn.Module,
        device: str,
    ) -> None:  # None return type as per https://peps.python.org/pep-0484/
        super(FLAG, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.device = device

    def forward(
        self,
        x: torch.tensor,
        edge_index: torch.tensor,
        y_true: torch.tensor,
        train_idx: torch.tensor,
        step_size: float,
        n_ascent_steps: int,
    ) -> tuple[torch.nn.Module, torch.tensor]:
        # Code below adapted from:
        #   https://github.com/devnkong/FLAG/blob/main/deep_gcns_torch
        #     /examples/ogb/ogbn_arxiv/main.py#L56
        #
        # This example does not assume that the model forward function has a
        #   `perturb` argument
        # forward = lambda perturb : model(x+perturb, edge_index)[train_idx]
        # model_forward = (model, forward)
        training_labels = y_true.squeeze(1)[train_idx]

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
            torch.rand(
                *x.shape,
                dtype=torch.float,
                device=self.device,
            ))
        self.perturb.detach().uniform_(-step_size, step_size)

        # out = forward(perturb)
        out = self.model(x + self.perturb, edge_index)[train_idx]
        self.loss = self.loss_fn(out, training_labels)

        # loss /= args.m
        self.loss /= n_ascent_steps

        # for _ in range(args.m - 1):
        for _ in range(n_ascent_steps - 1):
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

        self.loss.backward()
        self.optimizer.step()

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
