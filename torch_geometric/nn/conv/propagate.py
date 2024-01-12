import importlib
import os
import os.path as osp
import sys
from typing import Any

from jinja2 import Template

from torch_geometric import get_home_dir


def module_from_template(
    module_name: str,
    template_path: str,
    **kwargs: Any,
) -> Any:

    if module_name in sys.modules:  # If module is already loaded, return it:
        return sys.modules[module_name]

    with open(template_path, 'r') as f:
        template = Template(f.read())
    module_repr = template.render(**kwargs)
    print()
    print(module_repr)

    instance_dir = osp.join(get_home_dir(), 'propagate')
    os.makedirs(instance_dir, exist_ok=True)
    instance_path = osp.join(instance_dir, f'{module_name}.py')
    with open(instance_path, 'w') as f:
        f.write(module_repr)

    spec = importlib.util.spec_from_file_location(module_name, instance_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
