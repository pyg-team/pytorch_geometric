from __future__ import print_function

import os.path as osp
import tarfile
import zipfile
import gzip
import shutil


def maybe_log(path, log=True):
    if log:
        print('Extracting', path)


def extract_tar(path, folder, mode='r:gz', log=True):
    maybe_log(path, log)
    with tarfile.open(path, mode) as f:
        f.extractall(folder)


def extract_zip(path, folder, log=True):
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def extract_gz(path, folder, name, log=True):
    maybe_log(path, log)
    with gzip.open(path, 'rb') as f_in:
        with open(osp.join(folder, name), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
