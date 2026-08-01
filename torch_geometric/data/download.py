import os
import os.path as osp
import ssl
import sys
import urllib
import warnings
from typing import Optional

import fsspec

from torch_geometric.io import fs

INSECURE_ENV_VAR = 'PYG_INSECURE_DOWNLOADS'


def get_ssl_context() -> ssl.SSLContext:
    r"""Returns the :class:`ssl.SSLContext` used to download datasets.

    Server certificates are verified by default. Set the environment variable
    :obj:`PYG_INSECURE_DOWNLOADS=1` to skip verification, e.g., for a dataset
    host with a misconfigured certificate chain. Note that doing so leaves
    downloads open to tampering by anyone able to intercept the connection.
    """
    if os.getenv(INSECURE_ENV_VAR, '0') == '1':
        warnings.warn(
            f'Downloading with TLS certificate verification disabled '
            f"('{INSECURE_ENV_VAR}=1'). Downloaded files cannot be trusted to "
            f'originate from the expected host.',
            stacklevel=2,
        )
        return ssl._create_unverified_context()

    # `certifi` is installed alongside the `requests` dependency, and its CA
    # bundle is more reliable than the system trust store on some platforms:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_url(
    url: str,
    folder: str,
    log: bool = True,
    filename: Optional[str] = None,
):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
        filename (str, optional): The filename of the downloaded file. If set
            to :obj:`None`, will correspond to the filename given by the URL.
            (default: :obj:`None`)
    """
    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if fs.exists(path):  # pragma: no cover
        if log and 'PYTEST_CURRENT_TEST' not in os.environ:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log and 'PYTEST_CURRENT_TEST' not in os.environ:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder, exist_ok=True)

    data = urllib.request.urlopen(url, context=get_ssl_context())

    with fsspec.open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def download_google_url(
    id: str,
    folder: str,
    filename: str,
    log: bool = True,
):
    r"""Downloads the content of a Google Drive ID to a specific folder."""
    url = f'https://drive.usercontent.google.com/download?id={id}&confirm=t'
    return download_url(url, folder, log, filename)
