import torch

r"""Rotates all points given in the data object so that the eigenvectors of the positions
    overlie the axes of the cartesian coordinate system. If the data object contains normals 
    saved in data.norm these will be also rotated.

    Args:
        center (bool, optional): If set to :obj:`True` the points are also shifted so that their mean lies in zero. (default: :obj:`False`)
        maxPointsPCA (int, optional): If set to a value greater than 0, only a maximum of :obj: points 
            taken from the original points to calculate the eigenvectors in order to save memory and computation time (default: :obj:`-1`)
"""
class NormalizeRotation(object):
    def __init__(self, center=False, maxPointsPCA=-1):
        self.center = center
        self.maxPointsPCA = maxPointsPCA

    def __call__(self, data):
        pos = data.pos
        numPoints = data.pos.shape[0]           
        
        if self.maxPointsPCA > 0 and numPoints > self.maxPointsPCA:
            prob = torch.ones(numPoints) / numPoints
            randSamples = torch.multinomial(prob, self.maxPointsPCA, replacement=False)        
            pos = data.pos[randSamples]
                        
        hasNormals = False
        if hasattr(data, 'norm'):
            assert data.pos.shape == data.norm.shape
            hasNormals = True

        X_mean, eigenvec = PCA(pos)        
        data.pos = data.pos - X_mean
        transf = eigenvec.inverse()  

        for i in range(data.pos.shape[0]):
            data.pos[i] = torch.mv(transf, data.pos[i])
            if hasNormals:
                data.norm[i] = torch.mv(transf, data.norm[i])
                
        if not self.center:
            data.pos = data.pos + X_mean            

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)

    
def PCA(pos):
     pos_mean = torch.mean(pos,0)
     pos_centered = pos - pos_mean.expand_as(pos)
     U,_,_ = torch.svd(torch.t(pos_centered))
     eigenvec = U[:,:3]
     return (pos_mean, eigenvec)