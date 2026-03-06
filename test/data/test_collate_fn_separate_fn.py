import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate


class Foo:
    def __init__(self, x: Tensor):
        self.x = x


class FooChild(Foo):
    pass


def foo_collate(
    *,
    key: str,
    values: list[Foo],
    data_list,
    stores,
    increment: bool,
    collate_fn_map,
):
    xs = [v.x for v in values]
    sizes = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    slices = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)], dim=0)
    return Foo(torch.cat(xs, dim=0)), slices, None


def foo_separate(
    *,
    key: str,
    values: Foo,
    idx: int,
    slices: Tensor,
    incs,
    batch,
    store,
    decrement: bool,
    separate_fn_map,
):
    start, end = int(slices[idx]), int(slices[idx + 1])
    return Foo(values.x[start:end])


class FooBatch(Foo):
    pass


def foobatch_collate(
    *,
    key: str,
    values: list[Foo],
    data_list,
    stores,
    increment: bool,
    collate_fn_map,
):
    xs = [v.x for v in values]
    sizes = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    slices = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)], dim=0)
    return FooBatch(torch.cat(xs, dim=0)), slices, None


def foobatch_separate(
    *,
    key: str,
    values: Foo,
    idx: int,
    slices: Tensor,
    incs,
    batch,
    store,
    decrement: bool,
    separate_fn_map,
):
    start, end = int(slices[idx]), int(slices[idx + 1])
    return FooBatch(values.x[start:end])


class MapObj:
    def __init__(self, x: Tensor):
        self.x = x


def mapobj_collate(
    *,
    key: str,
    values: list[MapObj],
    data_list,
    stores,
    increment: bool,
    collate_fn_map,
):
    xs = [v.x for v in values]
    sizes = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    slices = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)], dim=0)
    value = {"x": torch.cat(xs, dim=0)}
    slice_dict = {"x": slices}
    return value, slice_dict, None


class SeqObj:
    def __init__(self, a: Tensor, b: Tensor):
        self.a = a
        self.b = b


def seqobj_collate(
    *,
    key: str,
    values: list[SeqObj],
    data_list,
    stores,
    increment: bool,
    collate_fn_map,
):
    as_ = [v.a for v in values]
    bs_ = [v.b for v in values]
    a_sizes = torch.tensor([x.size(0) for x in as_], dtype=torch.long)
    b_sizes = torch.tensor([x.size(0) for x in bs_], dtype=torch.long)
    a_slices = torch.cat([torch.zeros(1, dtype=torch.long), a_sizes.cumsum(0)], dim=0)
    b_slices = torch.cat([torch.zeros(1, dtype=torch.long), b_sizes.cumsum(0)], dim=0)
    return [torch.cat(as_, dim=0), torch.cat(bs_, dim=0)], [a_slices, b_slices], None


def test_collate_separate_fn_map_roundtrip():
    data_list = [
        Data(foo=Foo(torch.tensor([1, 2]))),
        Data(foo=Foo(torch.tensor([3]))),
        Data(foo=Foo(torch.tensor([4, 5, 6]))),
    ]

    batch, slice_dict, inc_dict = collate(
        Data,
        data_list=data_list,
        increment=True,
        add_batch=False,
        collate_fn_map={Foo: foo_collate},
    )

    for i, ref in enumerate(data_list):
        out = separate(
            Data,
            batch=batch,
            idx=i,
            slice_dict=slice_dict,
            inc_dict=inc_dict,
            decrement=True,
            separate_fn_map={Foo: foo_separate},
        )
        assert isinstance(out.foo, Foo)
        assert torch.equal(out.foo.x, ref.foo.x)


def test_collate_separate_fn_map_isinstance_dispatch_and_recursion():
    data_list = [
        Data(
            foo=FooChild(torch.tensor([1])),
            nested={"foo": Foo(torch.tensor([2, 3])), "t": torch.tensor([4])},
        ),
        Data(
            foo=FooChild(torch.tensor([5, 6])),
            nested={"foo": Foo(torch.tensor([7])), "t": torch.tensor([8, 9])},
        ),
    ]

    batch, slice_dict, inc_dict = collate(
        Data,
        data_list=data_list,
        increment=True,
        add_batch=False,
        collate_fn_map={Foo: foo_collate},
    )

    out0 = separate(
        Data,
        batch=batch,
        idx=0,
        slice_dict=slice_dict,
        inc_dict=inc_dict,
        decrement=True,
        separate_fn_map={Foo: foo_separate},
    )

    assert torch.equal(out0.foo.x, data_list[0].foo.x)
    assert isinstance(out0.nested["foo"], Foo)
    assert torch.equal(out0.nested["foo"].x, data_list[0].nested["foo"].x)
    assert torch.equal(out0.nested["t"], data_list[0].nested["t"])


def test_separate_mapping_handles_incs_none_from_custom_collate():
    data_list = [
        Data(m=MapObj(torch.tensor([1, 2]))),
        Data(m=MapObj(torch.tensor([3]))),
    ]

    batch, slice_dict, inc_dict = collate(
        Data,
        data_list=data_list,
        increment=True,
        add_batch=False,
        collate_fn_map={MapObj: mapobj_collate},
    )

    out0 = separate(
        Data,
        batch=batch,
        idx=0,
        slice_dict=slice_dict,
        inc_dict=inc_dict,
        decrement=True,
    )

    assert torch.equal(out0.m["x"], torch.tensor([1, 2]))


def test_separate_sequence_handles_incs_none_from_custom_collate():
    data_list = [
        Data(s=SeqObj(torch.tensor([1]), torch.tensor([2, 3]))),
        Data(s=SeqObj(torch.tensor([4, 5]), torch.tensor([6]))),
    ]

    batch, slice_dict, inc_dict = collate(
        Data,
        data_list=data_list,
        increment=True,
        add_batch=False,
        collate_fn_map={SeqObj: seqobj_collate},
    )

    out1 = separate(
        Data,
        batch=batch,
        idx=1,
        slice_dict=slice_dict,
        inc_dict=inc_dict,
        decrement=True,
    )

    assert torch.equal(out1.s[0], torch.tensor([4, 5]))
    assert torch.equal(out1.s[1], torch.tensor([6]))


def test_separate_fn_map_isinstance_dispatch_for_subclass_batch_value():
    data_list = [
        Data(foo=Foo(torch.tensor([1, 2]))),
        Data(foo=Foo(torch.tensor([3]))),
    ]

    batch, slice_dict, inc_dict = collate(
        Data,
        data_list=data_list,
        increment=True,
        add_batch=False,
        collate_fn_map={Foo: foobatch_collate},
    )

    out0 = separate(
        Data,
        batch=batch,
        idx=0,
        slice_dict=slice_dict,
        inc_dict=inc_dict,
        decrement=True,
        separate_fn_map={Foo: foobatch_separate},
    )

    assert isinstance(batch.foo, FooBatch)
    assert isinstance(out0.foo, FooBatch)
    assert torch.equal(out0.foo.x, data_list[0].foo.x)


def test_collate_fn_map_prefers_exact_type_match():
    data_list = [
        Data(foo=FooChild(torch.tensor([1]))),
        Data(foo=FooChild(torch.tensor([2, 3]))),
    ]

    batch, _, _ = collate(
        Data,
        data_list=data_list,
        increment=True,
        add_batch=False,
        collate_fn_map={
            Foo: foo_collate,
            FooChild: foobatch_collate,
        },
    )

    assert isinstance(batch.foo, FooBatch)


def test_separate_fn_map_prefers_exact_type_match():
    data_list = [
        Data(foo=Foo(torch.tensor([1, 2]))),
        Data(foo=Foo(torch.tensor([3]))),
    ]

    batch, slice_dict, inc_dict = collate(
        Data,
        data_list=data_list,
        increment=True,
        add_batch=False,
        collate_fn_map={Foo: foobatch_collate},
    )

    out0 = separate(
        Data,
        batch=batch,
        idx=0,
        slice_dict=slice_dict,
        inc_dict=inc_dict,
        decrement=True,
        separate_fn_map={
            Foo: foo_separate,
            FooBatch: foobatch_separate,
        },
    )

    assert isinstance(out0.foo, FooBatch)
    assert torch.equal(out0.foo.x, data_list[0].foo.x)
