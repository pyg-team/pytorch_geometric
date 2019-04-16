import torch
from torch_scatter import scatter_mean

class GenerateMeshNormals(object):
    r"""Generate a normal for each mesh node based on the neighboring faces."""

    def __init__(self):
        pass

    def __call__(self, data):
        pos, face = data.pos, data.face
        assert not pos.is_cuda and not face.is_cuda
        assert pos.size(1) == 3 and face.size(0) == 3
                 
        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]
        fn = torch.nn.functional.normalize(vec1.cross(vec2), p=2)
        numFaces = data.face.size(1)

        fn_x = (fn.t()[0]).reshape(numFaces, 1).expand(numFaces, 3).flatten()
        fn_y = (fn.t()[1]).reshape(numFaces, 1).expand(numFaces, 3).flatten()
        fn_z = (fn.t()[2]).reshape(numFaces, 1).expand(numFaces, 3).flatten()        
        faceIdxList = face.flatten()   
        res_x = scatter_mean(fn_x, faceIdxList)
        res_y = scatter_mean(fn_y, faceIdxList)
        res_z = scatter_mean(fn_z, faceIdxList)                       
        res = torch.stack([res_x, res_y, res_z]).t()
        res = torch.nn.functional.normalize(res, p=0)
        data.norm = res
        return data        

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)
