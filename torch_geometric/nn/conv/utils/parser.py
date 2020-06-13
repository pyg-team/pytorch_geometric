from typing import List


def split_type_repr(type_repr: str) -> List[str]:
    out = []
    i = depth = 0
    for j, char in enumerate(type_repr):
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
        elif char == ',' and depth == 0:
            out.append(type_repr[i:j].strip())
            i = j + 1
    out.append(type_repr[i:].strip())
    return out


if __name__ == '__main__':
    out = 'Tuple[Tensor, torch.Tensor], OptPairTensor, Dict[str, Tensor]'
    out = split_type_repr(out)
    print(out)
