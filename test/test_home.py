import os
import os.path as osp
from tempfile import gettempdir

from torch_geometric import get_home_dir, set_home_dir

ENV_PYG_HOME = 'PYG_HOME'
DEFAULT_CACHE_DIR = osp.join('~', '.cache', 'pyg')


def test_home():
    home_dir = os.getenv(ENV_PYG_HOME, DEFAULT_CACHE_DIR)
    home_dir = osp.expanduser(home_dir)
    assert get_home_dir() == home_dir

    temp_test_path = osp.join(gettempdir(), 'test_pyg')
    set_home_dir(temp_test_path)
    assert get_home_dir() == temp_test_path
