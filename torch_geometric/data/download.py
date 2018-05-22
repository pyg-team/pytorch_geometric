import os.path as osp
from six.moves import urllib

from .makedirs import makedirs


def download_url(url, folder, log=True):
    if log:
        print('Downloading', url)

    makedirs(folder)

    data = urllib.request.urlopen(url)
    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path
