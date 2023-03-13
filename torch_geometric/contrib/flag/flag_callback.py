from typing import Optional

import torch


class FLAGCallback:
    """
    Base class for callbacks to use in order to hook into the various stages
    of the model training with the
    `FLAG data augmentation strategy
    <https://github.com/pyg-team/pytorch_geometric/blob/master/
    torch_geometric/contrib/flag/flag.py>`_.

    To create a custom callback, subclass
    `torch_geometric.contrib.flag.FLAGCallback` and override the method
    associated with the stage of interest.
    """
    def on_ascent_step_begin(self, step_index: int,
                             loss: Optional[torch.nn.Module]):
        """
        Called at the start of an ascent step.

        Subclasses should override for any actions to run. This function
        should only be called during TRAIN mode.

        Args:
            step_index (int): The index of step.
            loss (Optional[torch.nn.Module]): The loss before the ascent
                step. Will be None before the first ascent step.
        """
        pass

    def on_ascent_step_end(self, step_index: int, loss: torch.nn.Module,
                           perturb_data: torch.nn.Parameter):
        """
        Called at the end of an ascent step.

        Subclasses should override for any actions to run. This function
        should only be called during TRAIN mode.

        Args:
            step_index (int): The index of step.
            loss (torch.nn.Module): The loss after the ascent step.
            perturb_data (torch.nn.Parameter): The tensor containing the input
                perturbations after the ascent step.
        """
        pass

    def on_optimizer_step_begin(self, loss: torch.nn.Module):
        """
        Called at the start of an optimizer step.

        Subclasses should override for any actions to run. This function
        should only be called during TRAIN mode.

        Args:
            loss (torch.nn.Module): The loss after the ascent step.
        """
        pass

    def on_optimizer_step_end(self, loss: torch.nn.Module):
        """
        Called at the end of an optimizer step.

        Subclasses should override for any actions to run. This function
        should only be called during TRAIN mode.

        Args:
            loss (torch.nn.Module): The loss after the ascent step.
        """
        pass


class FLAGLossHistoryCallback(FLAGCallback):
    """
    Callback that collects the loss metric from each step.
    """
    def __init__(self):
        self.loss_history = []

    def on_ascent_step_end(self, step_index: int, loss: torch.nn.Module,
                           perturb_data: torch.nn.Parameter):
        """
        Called at the end of an ascent step. Appends the loss from the
        previous step to the loss_history.

        Args:
            step_index (int): The index of step.
            loss (torch.nn.Module): The loss after the ascent step.
            perturb_data (torch.nn.Parameter): The tensor containing the input
                perturbations after the ascent step.
        """
        self.loss_history.append(loss)


class FLAGPerturbHistoryCallback(FLAGCallback):
    """
    Callback that collects the perturbation from each step.
    """
    def __init__(self):
        self.perturb_history = []

    def on_ascent_step_end(self, step_index: int, loss: torch.nn.Module,
                           perturb_data: torch.nn.Parameter):
        """
        Called at the end of an ascent step. Appends the perturbation from the
        previous step to the perturb_history.

        Args:
            step_index (int): The index of step.
            loss (torch.nn.Module): The loss after the ascent step.
            perturb_data (torch.nn.Parameter): The tensor containing the input
                perturbations after the ascent step.
        """
        self.perturb_history.append(perturb_data.cpu().detach().numpy())
