import os
import os.path as osp

from torch_geometric import get_home_dir, set_home_dir
from torch_geometric.home import DEFAULT_CACHE_DIR


def test_home():
    print("===================")
    print(os.getenv('GITHUB_ACTIONS'))
    print(os.getenv('GITHUB_ACTIONS') is True)
    print(os.getenv('GITHUB_ACTIONS') == 'true')
    print("===================")

    os.environ.pop('PYG_HOME', None)
    home_dir = osp.expanduser(DEFAULT_CACHE_DIR)
    assert get_home_dir() == home_dir

    home_dir = '/tmp/test_pyg1'
    os.environ['PYG_HOME'] = home_dir
    assert get_home_dir() == home_dir

    home_dir = '/tmp/test_pyg2'
    set_home_dir(home_dir)
    assert get_home_dir() == home_dir
