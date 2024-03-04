import importlib
import os.path as osp
import sys
import tempfile
from typing import Any

from jinja2 import Environment, FileSystemLoader


def module_from_template(
    module_name: str,
    template_path: str,
    tmp_dirname: str,
    **kwargs: Any,
) -> Any:

    if module_name in sys.modules:  # If module is already loaded, return it:
        return sys.modules[module_name]

    env = Environment(loader=FileSystemLoader(osp.dirname(template_path)))
    template = env.get_template(osp.basename(template_path))
    module_repr = template.render(**kwargs)

    with tempfile.NamedTemporaryFile(
            mode='w',
            prefix=f'{module_name}_',
            suffix='.py',
            delete=False,
    ) as tmp:
        tmp.write(module_repr)

    spec = importlib.util.spec_from_file_location(module_name, tmp.name)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
