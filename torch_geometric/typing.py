from typing import Tuple, Optional

from torch import Tensor

PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
Size = Optional[Tuple[int, int]]
