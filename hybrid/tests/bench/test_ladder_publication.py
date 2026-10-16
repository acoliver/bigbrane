"""Evaluation cache publication must not expose a delayed writer's empty archive to staging."""
from __future__ import annotations

import io
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from hybrid.bench import ladder


CFG = {"test_seed": 11, "tasks": {"m": {"module": "mqar", "vocab": 128, "seq_len": 64,
                                          "train_k": [4], "eval_k": [4], "n_eval": 8}}}


@pytest.mark.parametrize("oracle", [False, True])
def test_delayed_cache_writer_cannot_truncate_published_inputs(work_dir, monkeypatch, oracle):
    ready, primary_done, opened, release = (threading.Event() for _ in range(4))
    original = np.savez
    filename = "K4.oracle.npz" if oracle else "K4.npz"
    field = "targets" if oracle else "tokens"

    def paused_savez(file, **arrays):
        if threading.current_thread().name.startswith("delayed") and field in arrays:
            ready.set()
            assert primary_done.wait(10)
            # The old implementation opens the published path in wb mode here. The fixed one
            # passes a private file handle and only replaces the published name after closing it.
            if isinstance(file, (str, Path)):
                with open(file, "wb") as handle:
                    opened.set()
                    assert release.wait(10)
                    original(handle, **arrays)
            else:
                opened.set()
                assert release.wait(10)
                original(file, **arrays)
        else:
            original(file, **arrays)

    monkeypatch.setattr(np, "savez", paused_savez)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="delayed") as pool:
        delayed = pool.submit(ladder.build_eval_sets, CFG, "m", work_dir / "data")
        try:
            assert ready.wait(10)
            assert ladder.build_eval_sets(CFG, "m", work_dir / "data") == ["K4"]
            primary_done.set()
            assert opened.wait(10)
            assert ladder.build_eval_sets(CFG, "m", work_dir / "data") == ["K4"]
            published = work_dir / "data" / "m" / filename
            staged = work_dir / filename
            shutil.copyfile(published, staged)
            with np.load(staged, allow_pickle=False) as data:
                assert int(data["positions"].sum()) == 32
            before = staged.read_bytes()
        finally:
            primary_done.set()
            release.set()
        assert delayed.result(timeout=10) == ["K4"]
    assert published.read_bytes() == before
    assert not list(published.parent.glob("*.part"))


def test_atomic_npz_preserves_archive_bytes(work_dir):
    arrays = {"tokens": np.arange(32, dtype=np.int16), "positions": np.ones(32, dtype=bool)}
    expected = io.BytesIO()
    np.savez(expected, **arrays)
    target = work_dir / "set.npz"
    ladder._write_npz(target, **arrays)
    assert target.read_bytes() == expected.getvalue()


def test_interrupted_npz_write_keeps_published_archive(work_dir, monkeypatch):
    target = work_dir / "set.npz"
    np.savez(target, tokens=np.arange(4))
    before = target.read_bytes()

    def interrupted(file, **arrays):
        file.write(b"partial archive")
        raise OSError("interrupted archive write")

    monkeypatch.setattr(np, "savez", interrupted)
    with pytest.raises(OSError, match="interrupted archive write"):
        ladder._write_npz(target, tokens=np.arange(8))
    assert target.read_bytes() == before
    assert not list(work_dir.glob("*.part"))
