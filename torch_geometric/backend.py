from typing import Optional

# If set to `True`, PyG is configured to use the `segment_matmul` and
# `grouped_matmul` kernels from `pyg-lib` to parallelize matrix multiplication
# across segments/groups of potentially varying size.
# If set to `None`, will automatically decide whether to utilize
# `segment_matmul` and `grouped_matmul` based on input sizes.
# Requires `pyg-lib` to be installed.
use_segment_matmul: Optional[bool] = None
