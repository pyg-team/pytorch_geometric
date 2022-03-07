<<<<<<< HEAD
import sys
import ssl
# import os.path as osp
from pathlib import Path
from six.moves import urllib
=======
import os.path as osp
import ssl
import sys
import urllib
>>>>>>> 4557254c849eda62ce1860a56370eb4a54aa76dd

from .makedirs import makedirs


def download_url(url: str, folder: str, log: bool = True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

<<<<<<< HEAD
    filename = url.rpartition('/')[2]
    filename = filename if filename[0] == '?' else filename.split('?')[0]
    path = osp.join(folder, filename)
=======
    filename = url.rpartition('/')[2].split('?')[0]
    path = Path.joinpath(Path(folder), filename)
    # path = osp.join(folder, filename)
>>>>>>> f1c34427cce0562cd57fc797231077e4eab06ff1

    if Path(path).exists():  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path
