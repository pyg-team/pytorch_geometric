import os
import h5py
import numpy as np
import torch
import scipy.sparse as sp


def read_matlab_sparse(path, name):
    db = h5py.File(path, 'r')
    ds = db[name]

    if 'ir' in ds.keys():
        data = np.asarray(ds['data'])
        ir = np.asarray(ds['ir'])
        jc = np.asarray(ds['jc'])
        db.close()
        return sp.csc_matrix((data, ir, jc)).astype(np.float32)


def read_matlab(path, name):
    db = h5py.File(path, 'r')
    ds = db[name]
    out = np.asarray(ds).astype(np.float32).T.squeeze()
    db.close()
    return torch.FloatTensor(out)


root = os.path.expanduser('~/cvpr')
geo_path = os.path.join(root, 'geodistances', 'p=005_cs=010_ada=0')
patch_path = os.path.join(root, 'patches_coord', 'p=005_cs=010_ada=0')

file_path = os.path.join(patch_path, 'tr_reg_000.mat')
out = read_matlab_sparse(file_path, 'patch_coord')
rho = out[:, :out.shape[0]].tocoo()
theta = out[:, out.shape[0]:].tocoo()

row = torch.from_numpy(rho.row).long()
col = torch.from_numpy(rho.col).long()
rho = torch.from_numpy(rho.data)
theta = torch.from_numpy(theta.data)

index = torch.stack([row, col], dim=0)
value = torch.stack([rho, theta], dim=1)
adj = torch.sparse.FloatTensor(index, value, torch.Size([6890, 6890, 2]))

outputs = []
for i in range(0, 100):
    file_path = os.path.join(geo_path, 'tr_reg_{:03d}.mat'.format(i))
    out = read_matlab(file_path, 'distance_matrix')
    torch.save(out, os.path.join(root, 'geodesic/{:02d}.pt'.format(i)))
