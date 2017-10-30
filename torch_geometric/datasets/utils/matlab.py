import h5py
import torch
import numpy as np


def read_matlab(path, name):
    db = h5py.File(path, 'r')
    ds = db[name]
    out = np.asarray(ds).astype(np.float32).T.squeeze()
    db.close()
    return torch.FloatTensor(out)
