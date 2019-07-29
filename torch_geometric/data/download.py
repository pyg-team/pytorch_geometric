from __future__ import print_function

import os.path as osp
from six.moves import urllib

from .makedirs import makedirs


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    if log:
        print('Downloading', url)
    
    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)
    
    # check if file exists
    if osp.exists(path):
        pass
    else:
        makedirs(folder)
        data = urllib.request.urlopen(url)
        with open(path, 'wb') as f:
            f.write(data.read())

    return path
