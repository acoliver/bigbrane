import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls real models; runs only with -m live or BIGBRANE_LIVE=1")


def pytest_collection_modifyitems(config, items):
    if "live" in (config.getoption("-m") or "") or os.environ.get("BIGBRANE_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live test: run with -m live (spends model quota)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_routing_override(monkeypatch):
    """A BIGBRANE_ROUTING exported in the caller's shell must not change which table the tests load."""
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)
