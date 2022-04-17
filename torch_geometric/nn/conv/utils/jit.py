import os.path as osp
import sys
from getpass import getuser
from importlib.util import module_from_spec, spec_from_file_location
from tempfile import NamedTemporaryFile as TempFile
from tempfile import gettempdir

from torch_geometric.data.makedirs import makedirs


def class_from_module_repr(cls_name, module_repr):
    path = osp.join(gettempdir(), f'{getuser()}_pyg')
    makedirs(path)
    with TempFile(mode='w+', suffix='.py', delete=False, dir=path) as f:
        f.write(module_repr)
    spec = spec_from_file_location(cls_name, f.name)
    mod = module_from_spec(spec)
    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)
