import collections
import torch
from torch import Tensor
from typing import Union, List, overload, Optional, Tuple, NamedTuple
import torch
import inspect
from torch_sparse import SparseTensor
from jinja2 import Template

# def my_conv(x, y=2.0):  # noqa: F811
#     if isinstance(y, str):
#         if y == "hi":
#             return 4.0 - x
#         else:
#             return 5.0 - x
#     else:
#         return 2.0 + x


@torch.jit.script
class Test(object):
    @torch.jit._overload_method
    def size(self, src: torch.Tensor, dim: str) -> List[int]:
        pass

    @torch.jit._overload_method
    def size(self, src: torch.Tensor, dim: int) -> int:
        pass

    def size(self, src, dim):
        if isinstance(dim, str):
            return [1, 1]
        else:
            return 1

    def bla(self, src: torch.Tensor):
        pass
        # bla = self.size(src, "awd")
        # bla = self.size(src, 0)


@torch.jit._overload
def bla(x: str, y: str) -> str:
    pass


@torch.jit._overload
def bla(x: int, y: int) -> int:
    pass


def bla(x, y):
    if isinstance(x, str):
        return x + y
    else:
        return x - y


print()
print(bla('w', 'a'))
print(bla(4, 1))


@torch.jit.script
def haha():
    print(bla('w', 'a'))
    print(bla(4, 2))


# Kwargs = collections.namedtuple('kwargs', ['x', 'y': int])
# Point = collections.namedtuple('Point', ['x', 'y'])


class Kwargs(NamedTuple):
    x: torch.Tensor
    y: int


class MessagePassing(torch.nn.Module):
    def __init__(self):
        super(MessagePassing, self).__init__()

    @torch.jit._overload_method
    def propagate(self, adj_type, kwargs):
        # type: (torch.Tensor, Kwargs) -> Tensor
        pass

    @torch.jit._overload_method
    def propagate(self, adj_type, kwargs):
        # type: (SparseTensor, Kwargs) -> Tensor
        pass

    def propagate(self, adj_type, kwargs):
        return kwargs.x

    @torch.jit._overload_method
    def forward(self, x: torch.Tensor, adj_type: SparseTensor) -> torch.Tensor:
        pass

    @torch.jit._overload_method
    def forward(self, x: torch.Tensor, adj_type: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x, adj_type):
        kwargs: Kwargs = Kwargs(x=x, y=10)
        return self.propagate(adj_type, kwargs)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = MessagePassing()

    def forward(self, x: torch.Tensor, adj_type: SparseTensor):
        return self.conv(x, adj_type)


print('---------')
x = torch.randn(2, 2)
adj = SparseTensor.from_dense(x)
conv = MessagePassing()
# print(conv.propagate(x))
# print(conv.propagate(adj))
print('---------')

model = Net()
print(model(x, adj))

conv = torch.jit.script(model)
print(conv(x, adj))

# bla = torch.jit.script(conv)
# print(bla(x))
# print(bla(adj))

# bla = Test()
# x = torch.zeros(5)
# print(bla.bla(x))
# print(bla.size())
# print(bla.size(0))

# # @torch._jit_internal._overload_method
# def size(dim: int) -> int:
#     return 1

# # @torch._jit_internal._overload_method
# def size() -> List[int]:
#     return [1, 1]

# @torch.jit.script
# def test():
#     size()
#     size(1)

# class Over(torch.nn.Module):
#     def __init__(self):
#         super(Over, self).__init__()

#     @torch.jit._overload_method  # noqa: F811
#     def forward(self, x):  # noqa: F811
#         # type: (Tuple[Tensor, Tensor]) -> Tensor
#         pass

#     @torch.jit._overload_method  # noqa: F811
#     def forward(self, x):  # noqa: F811
#         # type: (Tensor) -> Tensor
#         pass

#     def forward(self, x):  # noqa: F811
#         if isinstance(x, Tensor):
#             return x + 20
#         else:
#             return x[0] + 5

# class S(torch.jit.ScriptModule):
#     def __init__(self):
#         super(S, self).__init__()
#         self.weak = Over()

#     @torch.jit.script_method
#     def forward(self, x):
#         return self.weak(x) + self.weak((x, x))


@torch.jit._overload  # noqa: F811
def my_conv(x, y):  # noqa: F811
    # type: (float, str) -> (float)
    pass


@torch.jit._overload  # noqa: F811
def my_conv(x, y=2.0):  # noqa: F811
    # type: (float, float) -> (float)
    pass


def my_conv(x, y=2.0):  # noqa: F811
    if isinstance(y, str):
        if y == "hi":
            return 4.0 - x
        else:
            return 5.0 - x
    else:
        return 2.0 + x


def test_jit():
    # s_mod = S()
    # x = torch.ones(1)
    # print(s_mod(x))
    # print(s_mod)

    x = torch.zeros(5)

    print(my_conv(x, y=2.0))
    print(my_conv(x, y='hii'))

    # torch.jit.script(my_conv)
    # self.assertEqual(s_mod(x), x + 20 + 5 + x)

    # bla = Over()
    # jitted_bla = torch.jit.script(bla)
    # print(bla(torch.zeros(2)))
    # print(jitted_bla(torch.zeros(2)))

    # test()

    # bla = test()
    # print(bla.size(1))
    # print(bla.size())

    # conv = MyConv()
    # jitted_conv = torch.jit.script(conv)
    # jitted_conv(x)
    # print(jitted_conv(x, x))

    # bla = inspect.getsource(MyConv)
    # print(bla)

    # eval(bla)
