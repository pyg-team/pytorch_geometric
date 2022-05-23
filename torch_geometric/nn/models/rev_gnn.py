import copy
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor, nn

from torch_geometric.typing import Adj


class InvertibleFunction(torch.autograd.Function):
    r"""Invertible autograd function. This allows doing autograd in a
    reversible fashion so that the memory of intermediate results can
    be freed during the forward pass and be constructed on-the-fly
    during the bachward pass.

    Args:
        ctx (torch.autograd.function.InvertibleFunctionBackward):
            A context object that can be used
            to stash information for backward computation.
        fn (nn.Module): The forward function.
        fn_inverse (nn.Module): The inverse function to
            recompute the freed input node features.
        num_bwd_passes (int): Number of backward passes to retain a link
            with the output. After the last backward pass the output is
            discarded and memory is freed.
        num_inputs (int): The number of inputs to the forward function.
        inputs_and_weights (tuple): inputs and weights for autograd.
    """
    @staticmethod
    def forward(ctx, fn, fn_inverse, num_bwd_passes, num_inputs,
                *inputs_and_weights):
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.weights = inputs_and_weights[num_inputs:]
        ctx.num_bwd_passes = num_bwd_passes
        ctx.num_inputs = num_inputs
        inputs = inputs_and_weights[:num_inputs]
        ctx.input_requires_grad = []

        with torch.no_grad():
            # Make a detached copy which shares the storage
            x = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    x.append(element.detach())
                    ctx.input_requires_grad.append(element.requires_grad)
                else:
                    x.append(element)
                    ctx.input_requires_grad.append(None)
            outputs = ctx.fn(*x)

        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        # Detaches outputs in-place, allows discarding the intermedate result
        detached_outputs = tuple(element.detach_() for element in outputs)

        # Clear memory of node features
        inputs[0].storage().resize_(0)

        # Store these tensor nodes for backward passes
        ctx.inputs = [inputs] * num_bwd_passes
        ctx.outputs = [detached_outputs] * num_bwd_passes

        return detached_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Retrieve input and output tensor nodes
        if len(ctx.outputs) == 0:
            raise RuntimeError(
                "Trying to perform backward on the 'InvertibleFunction'",
                f"for more than '{ctx.num_bwd_passes}' times.",
                "Try raising 'num_bwd_passes'.")
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        # Recompute input
        with torch.no_grad():
            # Need edge_index and other args
            inputs_inverted = ctx.fn_inverse(*(outputs + inputs[1:]))
            # Clear memory from outputs
            if len(ctx.outputs) == 0:
                for element in outputs:
                    element.storage().resize_(0)

            if not isinstance(inputs_inverted, tuple):
                inputs_inverted = (inputs_inverted, )
            for element_original, element_inverted in zip(
                    inputs, inputs_inverted):
                element_original.storage().resize_(
                    int(np.prod(element_original.size())))
                element_original.set_(element_inverted)

        # Compute gradients with grad enabled
        with torch.set_grad_enabled(True):
            detached_inputs = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    detached_inputs.append(element.detach())
                else:
                    detached_inputs.append(element)
            detached_inputs = tuple(detached_inputs)
            for det_input, requires_grad in zip(detached_inputs,
                                                ctx.input_requires_grad):
                if isinstance(det_input, torch.Tensor):
                    det_input.requires_grad = requires_grad
            temp_output = ctx.fn(*detached_inputs)

        if not isinstance(temp_output, tuple):
            temp_output = (temp_output, )
        filtered_detached_inputs = tuple(
            filter(
                lambda x: x.requires_grad
                if isinstance(x, torch.Tensor) else False,
                detached_inputs,
            ))
        gradients = torch.autograd.grad(
            outputs=temp_output,
            inputs=filtered_detached_inputs + ctx.weights,
            grad_outputs=grad_outputs,
        )

        input_gradients = []
        i = 0
        for rg in ctx.input_requires_grad:
            if rg:
                input_gradients.append(gradients[i])
                i += 1
            else:
                input_gradients.append(None)

        gradients = tuple(input_gradients) + gradients[-len(ctx.weights):]

        return (None, None, None, None) + gradients


class InvertibleModule(nn.Module, ABC):
    r"""An abstract class for implementing invertible modules.

    Args:
        disable (bool, optional): This disables using the InvertibleFunction.
            Therefore, it executes functions without memory savings.
            (default: :obj:`False`)
        num_bwd_passes (int, optional):
            Number of backward passes to retain a link with the output.
            After the last backward pass the output is discarded
            and memory is freed. (default: :obj:`1`)
    """
    def __init__(self, disable=False, num_bwd_passes=1):
        super().__init__()
        self.disable = disable
        self.num_bwd_passes = num_bwd_passes

    def forward(self, *args):
        y = self._fn_apply(args, self._forward, self._inverse)
        return y

    def inverse(self, *args):
        x = self._fn_apply(args, self._inverse, self._forward)
        return x

    @abstractmethod
    def _forward(self):
        pass

    @abstractmethod
    def _inverse(self):
        pass

    def _fn_apply(self, args, fn, fn_inverse):
        if not self.disable:
            out = InvertibleFunction.apply(
                fn,
                fn_inverse,
                self.num_bwd_passes,
                len(args),
                *args,
                *tuple(p for p in self.parameters() if p.requires_grad),
            )
        else:
            out = fn(*args)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(out, tuple) and len(out) == 1:
            return out[0]
        return out


class GroupAddRev(InvertibleModule):
    r"""InvertibleModule of Grouped Reversible GNNs for saving GPU memory.
    Proposed in Training Graph Neural Networks with 1000 Layers:
    https://arxiv.org/abs/2106.07476.
    This module enables training arbitary deep GNNs with memory complexity
    that is independent of the number of layers.

    Args:
        gnn (Union[nn.Module, nn.ModuleList]):
            A seed gnn for building gnn groups or a groups of gnns.
        split_dim (int optional): The dimension to spilt groups.
            (default: :obj:`-1`)
        num_groups (Optional[int], optional): The number of groups for
            group additive reverse.
            (default: :obj:`None`)
        disable (bool, optional): This disables using the InvertibleFunction.
            Therefore, it executes functions without memory savings.
            (default: :obj:`False`)
        num_bwd_passes (int, optional):
            Number of backward passes to retain a link with the output.
            After the last backward pass the output is discarded
            and memory is freed. (default: :obj:`1`)
    """
    def __init__(
        self,
        gnn: Union[nn.Module, nn.ModuleList],
        split_dim: int = -1,
        num_groups: Optional[int] = None,
        disable: bool = False,
        num_bwd_passes: int = 1,
    ):
        super().__init__(disable, num_bwd_passes)

        if isinstance(gnn, nn.ModuleList):
            n_groups = len(gnn)
            if num_groups is not None:
                assert (n_groups == num_groups
                        ), "Number of GNN groups mismatch with num_group"
            num_groups = n_groups
            gnn_groups = gnn
        else:
            assert num_groups is not None, "Please specific number of groups"
            gnn_groups = self._create_groups(gnn, num_groups)

        assert num_groups >= 2, "Number of groups should not be smaller than 2"

        self.gnn_groups = gnn_groups
        self.split_dim = split_dim
        self.num_groups = num_groups

    def reset_parameters(self):
        for gnn in self.gnn_groups:
            gnn.reset_parameters()

    def _forward(self, x: Tensor, edge_index: Adj, *args):
        xs = torch.chunk(x, self.num_groups, dim=self.split_dim)
        args_chunks = self._get_arg_chucks(args, x.size())
        y_in = sum(xs[1:])
        ys = []
        for i in range(self.num_groups):
            y_in = xs[i] + self.gnn_groups[i](y_in, edge_index, *
                                              args_chunks[i])
            ys.append(y_in)

        y = torch.cat(ys, dim=self.split_dim)

        return y

    def _inverse(self, y: Tensor, edge_index: Adj, *args):
        ys = torch.chunk(y, self.num_groups, dim=self.split_dim)
        args_chunks = self._get_arg_chucks(args, y.size())
        xs = []
        for i in range(self.num_groups - 1, -1, -1):
            if i != 0:
                y_in = ys[i - 1]
            else:
                y_in = sum(xs)
            x = ys[i] - self.gnn_groups[i](y_in, edge_index, *args_chunks[i])
            xs.append(x)

        x = torch.cat(xs[::-1], dim=self.split_dim)

        return x

    def _get_arg_chucks(self, args, size):
        # The size of edge related tensors should not be the
        # same as node features.
        if len(args) == 0:
            args_chunks = [()] * self.num_groups
        else:

            def chunk_arg(arg):
                if isinstance(arg, Tensor) and arg.size() == size:
                    return torch.chunk(arg, self.num_groups,
                                       dim=self.split_dim)
                return [arg] * self.num_groups

            chunked_args = map(chunk_arg, args)
            args_chunks = list(zip(*chunked_args))
        return args_chunks

    @staticmethod
    def _create_groups(seed_block, num_groups):
        blocks = nn.ModuleList()
        for i in range(num_groups):
            if i == 0:
                blocks.append(seed_block)
            else:
                new_block = copy.deepcopy(seed_block)
                if hasattr(new_block, 'reset_parameters'):
                    new_block.reset_parameters()
                blocks.append(new_block)

        return blocks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_groups={self.num_groups}, \
            disable={self.disable}"
