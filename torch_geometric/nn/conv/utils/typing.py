import inspect
import re
from collections import OrderedDict
from itertools import product
from typing import Callable, Dict, List, Tuple

import pyparsing as pp


def split_types_repr(types_repr: str) -> List[str]:
    out = []
    i = depth = 0
    for j, char in enumerate(types_repr):
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
        elif char == ',' and depth == 0:
            out.append(types_repr[i:j].strip())
            i = j + 1
    out.append(types_repr[i:].strip())
    return out


def sanitize(type_repr: str):
    type_repr = re.sub(r'<class \'(.*)\'>', r'\1', type_repr)
    type_repr = type_repr.replace('typing.', '')
    type_repr = type_repr.replace('torch_sparse.tensor.', '')
    type_repr = type_repr.replace('Adj', 'Union[Tensor, SparseTensor]')
    type_repr = type_repr.replace('torch_geometric.SparseTensor',
                                  'SparseTensor')

    # Replace `Union[..., NoneType]` by `Optional[...]`.
    sexp = pp.nestedExpr(opener='[', closer=']')
    tree = sexp.parseString(f'[{type_repr.replace(",", " ")}]').asList()[0]

    def union_to_optional_(tree):
        for i in range(len(tree)):
            e, n = tree[i], tree[i + 1] if i + 1 < len(tree) else []
            if e == 'Union' and n[-1] == 'NoneType':
                tree[i] = 'Optional'
                tree[i + 1] = tree[i + 1][:-1]
            elif e == 'Union' and 'NoneType' in n:
                idx = n.index('NoneType')
                n[idx] = [n[idx - 1]]
                n[idx - 1] = 'Optional'
            elif isinstance(e, list):
                tree[i] = union_to_optional_(e)
        return tree

    tree = union_to_optional_(tree)
    type_repr = re.sub(r'\'|\"', '', str(tree)[1:-1]).replace(', [', '[')

    return type_repr


def param_type_repr(param) -> str:
    if param.annotation is inspect.Parameter.empty:
        return 'torch.Tensor'
    return sanitize(re.split(r':|='.strip(), str(param))[1])


def return_type_repr(signature) -> str:
    return_type = signature.return_annotation
    if return_type is inspect.Parameter.empty:
        return 'torch.Tensor'
    elif str(return_type)[:6] != '<class':
        return sanitize(str(return_type))
    elif return_type.__module__ == 'builtins':
        return return_type.__name__
    else:
        return f'{return_type.__module__}.{return_type.__name__}'


def parse_types(func: Callable) -> List[Tuple[Dict[str, str], str]]:
    source = inspect.getsource(func)
    signature = inspect.signature(func)

    # Parse `# type: (...) -> ...` annotation. Note that it is allowed to pass
    # multiple `# type:` annotations in `forward()`.
    iterator = re.finditer(r'#\s*type:\s*\((.*)\)\s*->\s*(.*)\s*\n', source)
    matches = list(iterator)

    if len(matches) > 0:
        out = []
        args = list(signature.parameters.keys())
        for match in matches:
            arg_types_repr, return_type = match.groups()
            arg_types = split_types_repr(arg_types_repr)
            arg_types = OrderedDict((k, v) for k, v in zip(args, arg_types))
            return_type = return_type.split('#')[0].strip()
            out.append((arg_types, return_type))
        return out

    # Alternatively, parse annotations using the inspected signature.
    else:
        ps = signature.parameters
        arg_types = OrderedDict((k, param_type_repr(v)) for k, v in ps.items())
        return [(arg_types, return_type_repr(signature))]


def resolve_types(arg_types: Dict[str, str],
                  return_type_repr: str) -> List[Tuple[List[str], str]]:
    out = []
    for type_repr in arg_types.values():
        if type_repr[:5] == 'Union':
            out.append(split_types_repr(type_repr[6:-1]))
        else:
            out.append([type_repr])
    return [(x, return_type_repr) for x in product(*out)]
