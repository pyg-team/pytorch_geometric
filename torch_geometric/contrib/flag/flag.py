import torch


class FLAG:
    def __init__(
        self,
        device: str,
    ):
        self.device = device

    def augment_train(
            self,
            model,
            x,
            edge_index,
            y_true,
            train_idx,
            step_size,
            optimizer,
            loss_fn,
            args,  # TODO ????
    ):
        # TODO Code below adapted from:
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

        # TODO What is args.m??
        loss /= args.m

        for _ in range(args.m - 1):
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

            # TODO What is args.m??
            loss /= args.m

        loss.backward()
        optimizer.step()

        return loss, out
