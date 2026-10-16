"""`run` and `plumb` refuse to start, with exit 2 and the measurement printed, when the workspace volume is low."""
import pytest

from hybrid import diskguard
from hybrid.research.__main__ import main


@pytest.fixture
def low_workspace(monkeypatch):
    monkeypatch.setattr(diskguard, "path_free_bytes", lambda path, statvfs=None: 3 * diskguard.GB)


def test_run_refuses_before_building_anything(low_workspace, monkeypatch, tmp_path, capsys):
    import hybrid.research.orchestrator as orchestrator
    monkeypatch.setattr(orchestrator, "build_orchestrator", lambda *a, **k: pytest.fail("orchestrator built"))
    assert main(["run", "--run-dir", str(tmp_path / "run")]) == 2
    err = capsys.readouterr().err
    assert "refusing to start the research run" in err and "3.0 GB free, threshold 20 GB" in err
    assert not (tmp_path / "run").exists()


def test_plumb_refuses_before_reading_the_source(low_workspace, monkeypatch, tmp_path, capsys):
    import hybrid.research.plumb as plumb
    monkeypatch.setattr(plumb, "plumb_command", lambda args: pytest.fail("plumb started"))
    code = main(["plumb", "--source-run", str(tmp_path / "src"), "--card", "card-00000000",
                 "--run-dir", str(tmp_path / "out"), "--reason", "test"])
    assert code == 2
    assert "refusing to start the research plumb" in capsys.readouterr().err


def test_inspect_is_not_guarded(low_workspace, monkeypatch, tmp_path, capsys):
    # inspect only reads a run; a full disk must not block looking at results.
    import hybrid.research.__main__ as cli
    monkeypatch.setattr(cli, "inspect_run", lambda run_dir: 0)
    assert main(["inspect", str(tmp_path)]) == 0
    assert "refusing" not in capsys.readouterr().err
