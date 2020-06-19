import sys
from tempfile import NamedTemporaryFile
from importlib.util import spec_from_file_location, module_from_spec


def class_from_module_repr(cls_name, module_repr):
    with NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as f:
        f.write(module_repr)
    spec = spec_from_file_location(cls_name, f.name)
    mod = module_from_spec(spec)
    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)
