import numpy as np
import numpy.testing as npt
import torch
from torch_geometric.transforms import GridSampling
from torch_geometric.data import Data


def test_grid_sampling():
    assert GridSampling(0.01, 12).__repr__() == 'GridSampling(0.01, 12)'

    num_points = 5
    num_classes = 2

    pos = torch.Tensor([[0, 0, 0.01], [0.01, 0, 0], [0, 0.01, 0], [0, 0.01, 0], [0.01, 0, 0.01]])
    
    batch = torch.from_numpy(np.zeros(num_points)).long()
    
    y = np.asarray(np.random.randint(0, num_classes, 5))
    uniq, counts = np.unique(y, return_counts=True)
    answer = uniq[np.argmax(counts)]
    
    y = torch.from_numpy(y)

    out_data = GridSampling(0.04, 2)()(Data(pos=pos, batch=batch, y=y))

    out_y = out_data.y.detach().cpu().numpy()

    npt.assert_array_almost_equal(answer, out_y)