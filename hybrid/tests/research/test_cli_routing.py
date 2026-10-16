"""`python -m hybrid.research run --routing gateway` fails fast, before the run directory exists."""
from hybrid.research.__main__ import build_parser, main


def test_routing_flag_is_parsed():
    args = build_parser().parse_args(["run", "--routing", "gateway"])
    assert args.routing == "gateway"
    assert build_parser().parse_args(["run"]).routing is None


def test_gateway_routing_without_a_gateway_prints_serve_and_login_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BIGBRANE_GATEWAY_HOME", str(tmp_path / "gateway-home"))   # no run-token there
    monkeypatch.delenv("BIGBRANE_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)
    run_dir = tmp_path / "run-20261010T000000Z"
    assert main(["run", "--routing", "gateway", "--run-dir", str(run_dir)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("cannot start the run: gateway backend 'gateway' is not usable at http://127.0.0.1:8765/v1")
    assert ".venv-hybrid/bin/python -m hybrid.gateway serve --port 8765" in err
    assert ".venv-hybrid/bin/python -m hybrid.gateway login anthropic" in err
    assert ".venv-hybrid/bin/python -m hybrid.gateway login chatgpt" in err
    assert not run_dir.exists()


def test_env_routing_is_checked_the_same_way(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BIGBRANE_GATEWAY_HOME", str(tmp_path / "gateway-home"))
    monkeypatch.delenv("BIGBRANE_GATEWAY_TOKEN", raising=False)
    monkeypatch.setenv("BIGBRANE_ROUTING", "gateway")
    assert main(["run", "--run-dir", str(tmp_path / "run-x")]) == 2
    assert "hybrid.gateway serve --port 8765" in capsys.readouterr().err


def test_unknown_routing_is_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)
    assert main(["run", "--routing", "nope", "--run-dir", str(tmp_path / "run-y")]) == 2
    assert "--routing = 'nope' names no routing table" in capsys.readouterr().err
