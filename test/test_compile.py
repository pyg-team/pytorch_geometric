import torch_geometric.typing
from torch_geometric import enable_compile


def test_compile_context_manager():
    WITH_PYG_LIB = torch_geometric.typing.WITH_PYG_LIB
    WITH_TORCH_SCATTER = torch_geometric.typing.WITH_TORCH_SCATTER
    WITH_TORCH_SPARSE = torch_geometric.typing.WITH_TORCH_SPARSE

    with enable_compile():
        assert not torch_geometric.typing.WITH_PYG_LIB
        assert not torch_geometric.typing.WITH_TORCH_SCATTER
        assert not torch_geometric.typing.WITH_TORCH_SPARSE

    assert torch_geometric.typing.WITH_PYG_LIB == WITH_PYG_LIB
    assert torch_geometric.typing.WITH_TORCH_SCATTER == WITH_TORCH_SCATTER
    assert torch_geometric.typing.WITH_TORCH_SPARSE == WITH_TORCH_SPARSE


def test_compile_decorator():
    @enable_compile()
    def main():
        assert not torch_geometric.typing.WITH_PYG_LIB
        assert not torch_geometric.typing.WITH_TORCH_SCATTER
        assert not torch_geometric.typing.WITH_TORCH_SPARSE

    WITH_PYG_LIB = torch_geometric.typing.WITH_PYG_LIB
    WITH_TORCH_SCATTER = torch_geometric.typing.WITH_TORCH_SCATTER
    WITH_TORCH_SPARSE = torch_geometric.typing.WITH_TORCH_SPARSE

    main()

    assert torch_geometric.typing.WITH_PYG_LIB == WITH_PYG_LIB
    assert torch_geometric.typing.WITH_TORCH_SCATTER == WITH_TORCH_SCATTER
    assert torch_geometric.typing.WITH_TORCH_SPARSE == WITH_TORCH_SPARSE
