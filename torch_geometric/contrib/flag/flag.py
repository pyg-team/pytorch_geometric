import torch


class FLAG:
    def __init__(
        self,
        device: str,
    ):
        self.device = device

    # TODO
    # Should we make this class a torch.nn.Module and change this function
    #   to be the forward function?
    #
    # The constructor could be parameterized with the external model, data
    #   optimizer, loss_fn, etc
    def augment_train(
        self,
        model: torch.nn.Module,
        x: torch.tensor,
        edge_index: torch.tensor,
        y_true: torch.tensor,
        train_idx: torch.tensor,
        step_size: float,
        n_ascent_steps: int,
        optimizer: torch.nn.Module,
        loss_fn: torch.nn.Module,
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
        model.train()
        optimizer.zero_grad()

        # TODO
        #   torch.FloatTensor is deprecated
        #   Ensure that *x.shape works as we expect
        # perturb = torch.FloatTensor(*x.shape).uniform_(
        #   -args.step_size, args.step_size).to(self.device)
        perturb = torch.FloatTensor(*x.shape).uniform_(
            -step_size, step_size).to(self.device)
        perturb.requires_grad_()
        # out = forward(perturb)
        out = model(x + perturb, edge_index)[train_idx]
        loss = loss_fn(out, training_labels)

        # TODO Should this be loss += loss / args.m instead?
        # loss /= args.m
        loss /= n_ascent_steps

        # for _ in range(args.m - 1):
        for _ in range(n_ascent_steps - 1):
            loss.backward()
            # perturb_data = perturb.detach() + args.step_size *
            #   torch.sign(perturb.grad.detach())
            perturb_data = perturb.detach() + step_size * torch.sign(
                perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0

            # out = forward(perturb)
            out = model(x + perturb, edge_index)[train_idx]
            loss = loss_fn(out, training_labels)

            # TODO Should this be loss += loss / args.m instead?
            # loss /= args.m
            loss /= n_ascent_steps

        loss.backward()
        optimizer.step()

        return loss, out
