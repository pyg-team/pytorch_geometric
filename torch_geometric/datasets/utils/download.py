from __future__ import print_function

import os
from six.moves import urllib

from .dir import make_dirs


def download_url(url, dir):
    make_dirs(dir)

    filename = url.rpartition('/')[2]
    file_path = os.path.join(dir, filename)
    data = urllib.request.urlopen(url)

    with open(file_path, 'wb') as f:
        f.write(data.read())

    return file_path
