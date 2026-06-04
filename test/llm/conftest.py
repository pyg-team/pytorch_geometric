import pathlib

import pytest

LLM_DIR = pathlib.Path(__file__).parent


def pytest_collection_modifyitems(items):
    for item in items:
        if pathlib.Path(item.fspath).is_relative_to(LLM_DIR):
            item.add_marker(pytest.mark.rag)
