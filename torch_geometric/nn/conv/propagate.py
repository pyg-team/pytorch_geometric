import importlib
import os
import os.path as osp
import re
import sys
from typing import Any, Optional

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


def type_hint_to_str(type_hint: Any) -> str:
    type_repr = str(type_hint)
    type_repr = re.sub(r'<class \'(.*)\'>', r'\1', type_repr)
    return type_repr


def find_parenthesis_content(text: str, prefix: str) -> Optional[str]:
    match = re.search(prefix, text)
    if match is None:
        return

    offset = text[match.start():].find('(')
    if offset < 0:
        return

    content = text[match.start() + offset:]

    num_opened = num_closed = 0
    for end, char in enumerate(content):
        if char == '(':
            num_opened += 1
        if char == ')':
            num_closed += 1
        if num_opened > 0 and num_opened == num_closed:
            content = content[1:end]
            content = content.replace('\n', ' ').replace('#', ' ')
            return re.sub(' +', ' ', content.replace('\n', ' ')).strip()
