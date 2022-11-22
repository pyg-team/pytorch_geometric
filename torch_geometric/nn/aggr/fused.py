from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.nn import (
    Aggregation,
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    MulAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
)

# We can fuse all aggregations together that rely on `scatter` directives.
FUSABLE_AGGRS = {
    SumAggregation,
    MeanAggregation,
    MaxAggregation,
    MinAggregation,
    MulAggregation,
    VarAggregation,
    StdAggregation,
}

# All aggregations that rely on computing the degree of indices.
DEGREE_BASED_AGGRS = {
    MeanAggregation,
    VarAggregation,
    StdAggregation,
}

# Map aggregations to `reduce` options in `scatter` directives.
REDUCE = {
    SumAggregation: 'sum',
    MeanAggregation: 'sum',
    MaxAggregation: 'amax',
    MinAggregation: 'amin',
    MulAggregation: 'prod',
    VarAggregation: 'pow_sum',
    StdAggregation: 'pow_sum',
}


class FusedAggregation(Aggregation):
    r"""Helper class to fuse computation of multiple aggregations together.
    Used internally in :class:`~torch_geometric.nn.aggr.MultiAggregation` to
    speed-up computation.

    Args:
        aggrs (list): The list of aggregation schemes to use.
    """
    def __init__(self, aggrs: List[Aggregation]):
        super().__init__()

        self.aggr_cls = [aggr.__class__ for aggr in aggrs]
        self.aggr_index = {cls: i for i, cls in enumerate(self.aggr_cls)}

        for cls in self.aggr_cls:
            if cls not in FUSABLE_AGGRS:
                raise ValueError(f"Received aggregation '{cls.__name__}' in "
                                 f"'{self.__class__.__name__}' which is not "
                                 f"fusable")

        # Check whether we need to compute degree information:
        self.need_degree = False
        for cls in self.aggr_cls:
            if cls in DEGREE_BASED_AGGRS:
                self.need_degree = True

        # Determine which reduction to use for each aggregator:
        # An entry of `None` means that this operator re-uses intermediate
        # outputs from other aggregators.
        self.reduce_ops: List[Optional[str]] = []
        # Determine which `(Aggregator, index)` to use as intermediate output:
        self.lookup_ops: List[Optional[Tuple[Any, int]]] = []

        for cls in self.aggr_cls:
            if cls == MeanAggregation:
                # Directly use output of `SumAggregation`:
                if SumAggregation in self.aggr_index:
                    self.reduce_ops.append(None)
                    self.lookup_ops.append(
                        (SumAggregation, self.aggr_index[SumAggregation]))
                else:
                    self.reduce_ops.append(REDUCE[cls])
                    self.lookup_ops.append(None)

            elif cls == VarAggregation:
                if MeanAggregation in self.aggr_index:
                    self.reduce_ops.append(REDUCE[cls])
                    self.lookup_ops.append(
                        (MeanAggregation, self.aggr_index[MeanAggregation]))
                elif SumAggregation in self.aggr_index:
                    self.reduce_ops.append(REDUCE[cls])
                    self.lookup_ops.append(
                        (SumAggregation, self.aggr_index[SumAggregation]))
                else:
                    self.reduce_ops.append(REDUCE[cls])
                    self.lookup_ops.append(None)

            elif cls == StdAggregation:
                # Directly use output of `VarAggregation`:
                if VarAggregation in self.aggr_index:
                    self.reduce_ops.append(None)
                    self.lookup_ops.append(
                        (VarAggregation, self.aggr_index[VarAggregation]))
                elif MeanAggregation in self.aggr_index:
                    self.reduce_ops.append(REDUCE[cls])
                    self.lookup_ops.append(
                        (MeanAggregation, self.aggr_index[MeanAggregation]))
                elif SumAggregation in self.aggr_index:
                    self.reduce_ops.append(REDUCE[cls])
                    self.lookup_ops.append(
                        (SumAggregation, self.aggr_index[SumAggregation]))
                else:
                    self.reduce_ops.append(REDUCE[cls])
                    self.lookup_ops.append(None)

            else:
                self.reduce_ops.append(REDUCE[cls])
                self.lookup_ops.append(None)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        # Assert two-dimensional input for now to simplify computation:
        self.assert_index_present(index)
        self.assert_two_dimensional_input(x, dim)

        if self.need_degree:
            count = x.new_zeros(dim_size)
            count.scatter_add_(0, index, x.new_ones(x.size(0)))
            count = count.view(-1, 1)

        # Mask to set non-existing indicses to zero:
        mask = x.new_ones(dim_size, dtype=torch.bool)
        mask[index] = False

        F = x.size(-1)
        index = index.view(-1, 1).expand(-1, F)
        out = x.new_empty(dim_size, len(self.aggr_cls) * F)

        #######################################################################

        # Iterate over all reduction ops to compute first results:
        for i, reduce in enumerate(self.reduce_ops):
            if reduce is None:
                continue

            src = x * x if reduce == 'pow_sum' else x
            reduce = 'sum' if reduce == 'pow_sum' else reduce

            offset = slice(i * F, (i + 1) * F)
            out[:, offset].scatter_reduce_(0, index, src, reduce=reduce,
                                           include_self=False)

        #######################################################################

        # Compute `MeanAggregation` first to be able to re-use it:
        i = self.aggr_index.get(MeanAggregation)
        if i is not None:
            if self.lookup_ops[i] is None:
                sum_ = out[:, i * F:(i + 1) * F]
            else:
                tmp_aggr, j = self.lookup_ops[i]
                assert tmp_aggr == SumAggregation
                sum_ = out[:, j * F:(j + 1) * F]

            out[:, i * F:(i + 1) * F] = sum_ / count

        # Compute `VarAggregation` second to be able to re-use it:
        i = self.aggr_index.get(VarAggregation)
        if i is not None:
            if self.lookup_ops[i] is None:
                mean = x.new_empty(dim_size, F)
                mean.scatter_reduce_(0, index, src, reduce='sum',
                                     include_self=False)
                mean = mean / count
            else:
                tmp_aggr, j = self.lookup_ops[i]
                if tmp_aggr == SumAggregation:
                    mean = out[:, j * F:(j + 1) * F] / count
                elif tmp_aggr == MeanAggregation:
                    mean = out[:, j * F:(j + 1) * F]
                else:
                    raise NotImplementedError

            pow_sum = out[:, i * F:(i + 1) * F]
            out[:, i * F:(i + 1) * F] = (pow_sum / count) - (mean * mean)

        # Compute `StdAggregation` last:
        i = self.aggr_index.get(StdAggregation)
        if i is not None:
            var = None
            if self.lookup_ops[i] is None:
                pow_sum = out[:, i * F:(i + 1) * F]
                mean = x.new_empty(dim_size, F)
                mean.scatter_reduce_(0, index, src, reduce='sum',
                                     include_self=False)
                mean = mean / count
            else:
                tmp_aggr, j = self.lookup_ops[i]
                if tmp_aggr == VarAggregation:
                    var = out[:, j * F:(j + 1) * F]
                elif tmp_aggr == SumAggregation:
                    pow_sum = out[:, i * F:(i + 1) * F]
                    mean = out[:, j * F:(j + 1) * F] / count
                elif tmp_aggr == MeanAggregation:
                    pow_sum = out[:, i * F:(i + 1) * F]
                    mean = out[:, j * F:(j + 1) * F]
                else:
                    raise NotImplementedError

            if var is None:
                var = (pow_sum / count) - (mean * mean)

            out[:, i * F:(i + 1) * F] = (var.relu() + 1e-5).sqrt()

        #######################################################################

        out[mask] = 0.  # Set non-existing indices to zero.

        return out
