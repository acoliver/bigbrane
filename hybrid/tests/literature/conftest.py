import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls the real keyless literature APIs; needs BIGBRANE_LIVE=1")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("BIGBRANE_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test: set BIGBRANE_LIVE=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
