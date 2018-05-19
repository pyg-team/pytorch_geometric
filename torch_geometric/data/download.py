import os.path as osp
from six.moves import urllib
from torch_geometric.data import makedirs


def download_url(url, folder):
    print('Downloading', url)

    makedirs(folder)

    data = urllib.request.urlopen(url)
    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path
