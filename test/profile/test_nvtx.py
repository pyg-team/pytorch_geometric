from unittest.mock import call, patch

from torch_geometric.profile import nvtxit


def _setup_mock(torch_cuda_mock):
    torch_cuda_mock.is_available.return_value = True
    torch_cuda_mock.cudart.return_value.cudaProfilerStart.return_value = None
    torch_cuda_mock.cudart.return_value.cudaProfilerStop.return_value = None
    return torch_cuda_mock


@patch('torch_geometric.profile.nvtx.torch.cuda')
def test_nvtxit_base(torch_cuda_mock):
    torch_cuda_mock = _setup_mock(torch_cuda_mock)

    # dummy func calls a calls b

    @nvtxit()
    def call_b():
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 1  # noqa: E501
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 0  # noqa: E501
        return 42

    @nvtxit()
    def call_a():
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 1  # noqa: E501
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 0  # noqa: E501
        return call_b()

    def dummy_func():
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 0  # noqa: E501
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 0  # noqa: E501
        return call_a()

    assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 0  # noqa: E501
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 0  # noqa: E501
    dummy_func()

    assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 1  # noqa: E501
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 1  # noqa: E501
    assert torch_cuda_mock.nvtx.range_push.call_args_list == [
        call('call_a_0'), call('call_b_0')
    ]


@patch('torch_geometric.profile.nvtx.torch.cuda')
def test_nvtxit_rename(torch_cuda_mock):
    torch_cuda_mock = _setup_mock(torch_cuda_mock)

    # dummy func calls a calls b

    @nvtxit()
    def call_b():
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 1  # noqa: E501
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 0  # noqa: E501
        return 42

    @nvtxit('a_nvtx')
    def call_a():
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 1  # noqa: E501
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 0  # noqa: E501
        return call_b()

    def dummy_func():
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 0  # noqa: E501
        assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 0  # noqa: E501
        return call_a()

    assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 0  # noqa: E501
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 0  # noqa: E501
    dummy_func()

    assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 1  # noqa: E501
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 1  # noqa: E501
    assert torch_cuda_mock.nvtx.range_push.call_args_list == [
        call('a_nvtx_0'), call('call_b_0')
    ]


@patch('torch_geometric.profile.nvtx.torch.cuda')
def test_nvtxit_iters(torch_cuda_mock):
    torch_cuda_mock = _setup_mock(torch_cuda_mock)

    # dummy func calls a calls b

    @nvtxit(n_iters=1)
    def call_b():
        return 42

    @nvtxit()
    def call_a():
        return call_b()

    assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 0  # noqa: E501
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 0  # noqa: E501

    call_b()
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 1  # noqa: E501
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 1  # noqa: E501
    call_a()
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 2  # noqa: E501
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 2  # noqa: E501

    assert torch_cuda_mock.nvtx.range_push.call_args_list == [
        call('call_b_0'), call('call_a_0')
    ]


@patch('torch_geometric.profile.nvtx.torch.cuda')
def test_nvtxit_warmups(torch_cuda_mock):
    torch_cuda_mock = _setup_mock(torch_cuda_mock)

    # dummy func calls a calls b

    @nvtxit(n_warmups=1)
    def call_b():
        return 42

    @nvtxit()
    def call_a():
        return call_b()

    assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 0  # noqa: E501
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 0  # noqa: E501

    call_b()
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 0  # noqa: E501
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 0  # noqa: E501
    call_a()
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStart.call_count == 1  # noqa: E501
    assert torch_cuda_mock.cudart.return_value.cudaProfilerStop.call_count == 1  # noqa: E501

    assert torch_cuda_mock.nvtx.range_push.call_args_list == [
        call('call_a_0'), call('call_b_1')
    ]
