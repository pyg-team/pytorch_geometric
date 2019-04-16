import torch
from torch_geometric.transforms import GenerateMeshNormals
from torch_geometric.data import Data

def test_generate_normals():
    pos = torch.Tensor  ([ [0,0,0], [0,1,0], [1,1,1], [2,1,0], [2,0,0] ])
    face = torch.Tensor ([ [0,1,2], [2,3,4] ]).long().t()
    
    data = Data(pos=pos)
    data.face = face
    data = GenerateMeshNormals()(data)	
     
    assert data.norm.size(0) == data.pos.size(0)
    
    expected = torch.Tensor([   [ 0.3536,  0.0000, -0.3536],
                                [ 0.3536,  0.0000, -0.3536],
                                [ 0.0000,  0.0000, -0.7071],
                                [-0.3536,  0.0000, -0.3536],
                                [-0.3536,  0.0000, -0.3536] ])
                                
                          
    diff = torch.abs(torch.add(data.norm, -expected))
    assert torch.all(torch.lt(diff, 1e-4))