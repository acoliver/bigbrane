"""The planted-card test (DISCOVERY-PIPELINE-PLAN.md section 13, Phase 9) with fake models: sealing and the leak check,
the entry point and the whole line on the fixture sets, the held-back set and batches, T6 seeding, and the refusal
paths. The metrics are tested on their own in test_planted_metrics.py."""
import asyncio
import copy
import json
import os
import stat
from pathlib import Path

import pytest

from hybrid import diskguard, localmodels
from hybrid.research import planted as pl
from hybrid.research import planted_seal as ps
from hybrid.research.__main__ import main
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.outcomes import OutcomeStore
from hybrid.tests.research import planted_fakes as pf

RUN_TOML = ROOT / "hybrid" / "config" / "run.toml"
MAIN = {"T1": "pc-a1c0de01", "T2": "pc-a1c0de02", "T3": "pc-a1c0de03", "T4": "pc-a1c0de04", "T5": "pc-a1c0de05",
        "T6": "pc-a1c0de06", "T7": "pc-a1c0de07"}
HELD = {"T1": "pc-b2d0e001", "T2": "pc-b2d0e002", "T3": "pc-b2d0e003", "T4": "pc-b2d0e004", "T5": "pc-b2d0e005",
        "T6": "pc-b2d0e006", "T7": "pc-b2d0e007"}


def cfg():
    return load_run_config(RUN_TOML)


@pytest.fixture(autouse=True)
def workspace_for_sealing(monkeypatch):
    """pytest's temporary directories are inside the repository; the rule that sealed files live outside the
    workspace is tested against the hybrid/ subtree instead."""
    monkeypatch.setattr(ps, "WORKSPACE", ROOT / "hybrid")


@pytest.fixture(autouse=True)
def gateway_calls(monkeypatch):
    """The gateway check never reaches the gateway in these tests; it records that it was called."""
    import hybrid.llm as llm

    calls = []
    monkeypatch.setattr(llm, "check_gateways", lambda models: calls.append(models) or [])
    return calls


@pytest.fixture
def world(tmp_path, monkeypatch):
    """An earlier run with the two dead ideas the T2 and T6 fixtures vary, and a source store holding both. The
    command resolves store origins under this runs root instead of the workspace's runs/."""
    runs_root = tmp_path / "runs"
    pf.earlier_run(runs_root)
    store = pf.store_with_origins(tmp_path, runs_root)
    monkeypatch.setattr(pl, "RUNS_ROOT", runs_root)
    return {"runs_root": runs_root, "store": store}


def run_test(tmp_path, world, name, *, runs=1, keys=None, overrides=None, calls=None):
    planted, sealed = pf.sealed_set(tmp_path, name, store=world["store"])
    return asyncio.run(pl.planted_test(
        planted=planted, sealed=sealed, runs=runs, run_root=tmp_path / "planted-runs", out_dir=tmp_path / "out",
        cfg=cfg(), reason="planted fixture test", deps_factory=pf.deps_factory(overrides=overrides, calls=calls),
        store_source=world["store"].dir, runs_root=world["runs_root"], proposer_model="fake-explorer", keys=keys))


def results_of(report, run=0):
    return {key: card["results"][run] for key, card in report["cards"].items()}


# ----- the fixture set ----------------------------------------------------------------------------------------------

def test_the_fixture_sets_are_anonymous_and_pass_the_set_checks(world):
    for name in ("main", "held_back"):
        cards, _ = ps.read_cards(pf.FIXTURES / name)
        data = json.loads((pf.FIXTURES / f"{name}.expectations.json").read_text())
        assert ps.expectation_problems(data, set(cards), f"fixture_{name}") == []
        assert {e["type"] for e in data["cards"].values()} == set(ps.TYPES)
        assert ps.leak_problems(cards, data["cards"]) == {}
        assert ps.set_problems(cards, data["cards"], store=world["store"]) == []
        assert ps.store_origin_problems(data["cards"], world["store"], world["runs_root"]) == []


# ----- sealing and the leak check -------------------------------------------------------------------------------------

def test_sealing_moves_the_expectations_to_an_owner_only_file_and_commits_every_hash(tmp_path, world):
    planted, sealed = pf.sealed_set(tmp_path, "main", store=world["store"])
    assert not (tmp_path / "main.expectations.json").exists()
    mode = stat.S_IMODE(sealed.path.stat().st_mode)
    assert mode == 0o600 and stat.S_IMODE(sealed.path.parent.stat().st_mode) == 0o700
    seal = json.loads((planted.directory / ps.SEAL_FILE).read_text())
    assert seal["sealed_sha256"] == ps.sha256_bytes(sealed.path.read_bytes()) == sealed.sha256
    assert seal["cards"] == planted.card_hashes() and len(seal["cards"]) == 7
    # Nothing in the set directory or the cards says what any card is.
    text = "".join(path.read_text() for path in planted.directory.rglob("*.json"))
    # The type is matched as a JSON string: bare "T1" also occurs in a seal timestamp written between 10:00 and
    # 19:59 UTC ("2026-10-13T11:19:12"), which made this test fail by the clock.
    for entry in sealed.entries.values():
        assert entry["spec"] not in text and json.dumps(entry["type"]) not in text
    # A sealed set is never sealed again.
    source = tmp_path / "again.json"
    source.write_text((pf.FIXTURES / "main.expectations.json").read_text())
    with pytest.raises(ps.SealError, match="already sealed"):
        ps.seal(set_name="fixture_main", directory=planted.directory, expectations_path=source,
                sealed_directory=tmp_path / "sealed")


def test_a_changed_card_or_sealed_file_breaks_the_seal(tmp_path, world):
    planted, sealed = pf.sealed_set(tmp_path, "main", store=world["store"])
    os.chmod(sealed.path, 0o644)
    with pytest.raises(ps.SealError, match="readable by group or others"):
        ps.read_sealed(planted, tmp_path / "sealed")
    os.chmod(sealed.path, 0o600)
    data = json.loads(sealed.path.read_text())
    data["cards"][MAIN["T3"]]["type"] = "T7"
    sealed.path.write_text(json.dumps(data))
    with pytest.raises(ps.SealError, match="differs from the hash"):
        ps.read_sealed(planted, tmp_path / "sealed")
    card_path = planted.directory / "cards" / f"{MAIN['T1']}.json"
    card = json.loads(card_path.read_text())
    card["title"] += " (edited)"
    card_path.write_text(json.dumps(card))
    with pytest.raises(ps.SealError, match="card file changed since sealing"):
        ps.load_set(planted.directory)


def test_the_sealed_file_must_live_outside_the_workspace(tmp_path):
    directory = pf.copy_set("main", tmp_path / "set")
    source = tmp_path / "e.json"
    source.write_text((pf.FIXTURES / "main.expectations.json").read_text())
    with pytest.raises(ps.SealError, match="outside the workspace"):
        ps.seal(set_name="fixture_main", directory=directory, expectations_path=source,
                sealed_directory=ROOT / "hybrid" / "planted-sealed-never")
    assert source.exists() and not (directory / ps.SEAL_FILE).exists()


@pytest.mark.parametrize("field,text,finding", [
    ("summary", "A T3 card that rests on a free index.", "names a planted type: 'T3'"),
    ("summary", "The segmenter is an oracle.", "'oracle'"),
    ("title", "A renamed recurrence", "'renamed'"),
    ("summary", "Planted to test the challenger.", "'planted'"),
    ("summary", "Like a gated recurrent unit.", "'gated recurrent'"),
    ("summary", "In the spirit of RWKV.", "'rwkv'"),
    ("summary", "It is a bucket tally.", "'bucket tally'"),
    ("summary", "As shown in 2021.", "'2021'"),
])
def test_the_leak_check_catches_type_tags_planted_terms_family_names_and_citations(field, text, finding):
    card = copy.deepcopy(pf.CARDS["quorrel"])
    card[field] = text
    entry = {"source_family": "bucket tally", "aliases": []}
    problems = ps.leak_problems({"pc-00000001": card}, {"pc-00000001": entry})["pc-00000001"]
    assert any(finding in p for p in problems), problems


def test_a_card_may_not_carry_origin_or_stage_fields():
    card = copy.deepcopy(pf.CARDS["corvane"])
    card["lens"] = "kernel_gaussian_processes"
    card["arm"] = "A1"
    card["difference_statement"] = {}
    card["status"] = {"stage": "S2", "verdicts": {}}
    problems = ps.card_problems(card)
    assert "lens must be 'neutral'" in problems
    assert "field arm must be absent" in problems and "field difference_statement must be absent" in problems
    assert "status must be {stage: S0, verdicts: {}}" in problems


def test_expectations_need_the_plan_counts_origins_and_subtypes():
    keys = {f"pc-{i:08x}" for i in range(45)}
    kinds = ["T1"] * 10 + ["T2"] * 6 + ["T3"] * 8 + ["T4"] * 9 + ["T5"] * 6 + ["T6"] * 6
    subtypes = iter(["sign", "cost_exponent", "capacity"] * 3)
    cards = {}
    for key, kind in zip(sorted(keys), kinds):
        entry = {"type": kind, "spec": "s", "source_family": "f", "aliases": [], "batch": 1}
        if kind == "T4":
            entry["subtype"] = next(subtypes)
        if kind in ("T2", "T6"):
            entry["origin"] = {"run": "run-x", "card_id": "card-0000d002"}
        cards[key] = entry
    data = {"format": ps.FORMAT, "set": "main", "cards": cards}
    assert ps.expectation_problems(data, keys, "main") == []
    broken = copy.deepcopy(data)
    first_t5 = next(k for k, e in broken["cards"].items() if e["type"] == "T5")
    broken["cards"][first_t5]["type"] = "T7"
    assert any("needs" in p for p in ps.expectation_problems(broken, keys, "main"))
    broken = copy.deepcopy(data)
    t6 = next(k for k, e in broken["cards"].items() if e["type"] == "T6")
    del broken["cards"][t6]["origin"]
    assert any(p.startswith(f"{t6}: origin is required") for p in ps.expectation_problems(broken, keys, "main"))
    broken["cards"][t6]["origin"] = {"planted": t6}
    assert any("earlier batch" in p for p in ps.expectation_problems(broken, keys, "main"))
    t4 = next(k for k, e in broken["cards"].items() if e["type"] == "T4")
    broken["cards"][t4]["subtype"] = "rounding"
    assert any(f"{t4}: subtype must be one of" in p for p in ps.expectation_problems(broken, keys, "main"))


def test_set_checks_catch_collisions_and_origins_that_test_nothing(world):
    cards, _ = ps.read_cards(pf.FIXTURES / "main")
    data = json.loads((pf.FIXTURES / "main.expectations.json").read_text())["cards"]
    # A copy of the T7 card under another key collides on signature and wording.
    cards["pc-0000ffff"] = copy.deepcopy(cards[MAIN["T7"]])
    data["pc-0000ffff"] = {"type": "T7", "spec": "s", "source_family": "f", "aliases": [], "batch": 1}
    problems = ps.set_problems(cards, data, store=world["store"])
    assert f"pc-0000ffff and {MAIN['T7']} share a mechanism signature" in problems
    assert any("TF-IDF cosine" in p and "pc-0000ffff" in p for p in problems)
    del cards["pc-0000ffff"], data["pc-0000ffff"]
    # T6 must keep its origin's signature; T2 must change it.
    t2, t6 = copy.deepcopy(cards[MAIN["T2"]]), copy.deepcopy(cards[MAIN["T6"]])
    cards[MAIN["T6"]]["mechanism"]["update"][0]["plain"] += " then sort"
    cards[MAIN["T2"]] = copy.deepcopy(pf.origin_card("card-0000d002"))
    problems = ps.set_problems(cards, data, store=world["store"])
    assert f"{MAIN['T6']}: a T6 card must keep its origin's mechanism signature" in problems
    assert any(p.startswith(f"{MAIN['T2']}: a T2 card must change an equation") for p in problems)
    # Without its origin in the store or the runs, a T2 or T6 card cannot run.
    empty = OutcomeStore(world["store"].dir.parent / "empty")
    found = ps.store_origin_problems(data, empty, world["runs_root"].parent / "no-runs")
    assert any(MAIN["T6"] in p and "neither in the store" in p for p in found)
    cards[MAIN["T2"]], cards[MAIN["T6"]] = t2, t6


# ----- the whole line on the fixture set ------------------------------------------------------------------------------

def test_each_type_stops_where_its_fake_verdicts_send_it_and_two_runs_agree(tmp_path, world):
    llms = []
    report = run_test(tmp_path, world, "main", runs=2, calls=llms)
    assert report["accepted"], json.dumps({k: [(r["stop"], r["kill_codes"], r["notes"]) for r in c["results"]] for k, c in report["cards"].items()}, indent=1)
    for run in (0, 1):
        got = results_of(report, run)
        assert (got[MAIN["T1"]]["stop"], got[MAIN["T1"]]["kill_codes"]) == ("G", ["excluded_family:linear_attention"])
        assert (got[MAIN["T2"]]["stop"], got[MAIN["T2"]]["kill_codes"]) == ("P", ["not_radical:R3"])
        assert (got[MAIN["T3"]]["stop"], got[MAIN["T3"]]["kill_codes"]) == ("C", ["magic_x:A1"])
        assert got[MAIN["T4"]]["stop"] == "M"
        assert {"obligation_failed:P2.nesting", "obligation_failed:P2.composition"} <= set(
            got[MAIN["T4"]]["kill_codes"])
        assert (got[MAIN["T5"]]["stop"], got[MAIN["T5"]]["kill_codes"]) == ("P", ["reduces_to_known:other_known"])
        assert got[MAIN["T5"]]["override"]["M"] == "pass" and got[MAIN["T5"]]["override"]["C"] == "pass"
        assert (got[MAIN["T6"]]["stop"], got[MAIN["T6"]]["kill_codes"]) == ("G", ["duplicate:card-0000d006"])
        assert (got[MAIN["T7"]]["stop"], got[MAIN["T7"]]["outcome"]) == ("T", "reached")
        assert all(r["model_calls"] <= 80 and not r["over_budget"] for r in got.values())
    assert report["agreement"]["rate"] == 1.0 and report["audit"]["differences"] == []
    assert report["audit"]["math_checks"] > 0
    # Cards get new ids in every run; the run ledgers commit the seal first and map keys to ids.
    ids = [{k: r["card_id"] for k, r in results_of(report, run).items()} for run in (0, 1)]
    assert set(ids[0].values()).isdisjoint(ids[1].values())
    run_dir = Path(report["meta"]["runs_meta"][0]["run_dir"])
    events = list(Ledger(run_dir).iter_events())
    assert events[0]["type"] == "planted_seal" and events[0]["data"]["cards"] == ids[0]
    assert events[0]["data"]["sealed_sha256"] == report["meta"]["sealed_sha256"]
    created = [e["data"] for e in events if e["type"] == "card_created"]
    assert {c["role"] for c in created} == {"explorer"} and {c["op"] for c in created} == {"new"}
    assert [e["data"]["stage"] for e in events if e["type"] == "planted_reached"] == ["T"]
    # The live switches were on.
    start = next(e["data"] for e in events if e["type"] == "run_start")
    assert start["stages"] == ["S0", "S1", "R", "S2", "M", "C", "S4", "S5", "S6"]
    assert start["math_profile"] and start["target"]["enabled"] and start["refine"]["enabled"]
    assert start["outcomes"]["enabled"] and start["outcomes"]["enforce_r3"] and start["generation"]["mode"] == "arms"
    # No prompt carried a type tag, a specification from the sealed file, its format name or a set key.
    sealed_entries = json.loads((pf.FIXTURES / "main.expectations.json").read_text())["cards"]
    words = {e["spec"] for e in sealed_entries.values()} | {ps.FORMAT}
    for llm in llms:
        for call in llm.calls:
            text = call["system"] + call["user"]
            assert not any(word in text for word in words) and "pc-a1c0de" not in text
            assert not any(f"T{i}" in call["user"].split() for i in range(1, 8))
    # The planted run never wrote the source store.
    assert len(world["store"].current()) == 2
    out = tmp_path / "out"
    assert len((out / "outcomes.jsonl").read_text().splitlines()) == 14
    text = (out / "report.md").read_text()
    assert "Accepted (plan section 13.2): yes" in text and "| T4 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |" in text
    assert json.loads((out / "report.json").read_text())["accepted"] is True


def test_a_t6_card_seeds_its_origin_from_the_earlier_run_when_the_store_lacks_it(tmp_path, world):
    source = pf.store_with_origins(tmp_path / "partial", world["runs_root"], cards=("card-0000d002",))
    before = (source.dir / "outcomes.jsonl").read_bytes()
    planted, sealed = pf.sealed_set(tmp_path, "main", store=world["store"])
    report = asyncio.run(pl.planted_test(
        planted=planted, sealed=sealed, runs=1, run_root=tmp_path / "r", out_dir=tmp_path / "o", cfg=cfg(),
        reason="seeding", deps_factory=pf.deps_factory(), store_source=source.dir, runs_root=world["runs_root"],
        keys=[MAIN["T6"], MAIN["T7"]]))
    store_meta = report["meta"]["runs_meta"][0]["store"]
    assert [s["card_id"] for s in store_meta["seeded"]] == ["card-0000d006"] and store_meta["present"] == []
    assert results_of(report)[MAIN["T6"]]["kill_codes"] == ["duplicate:card-0000d006"]
    assert (source.dir / "outcomes.jsonl").read_bytes() == before
    copy_store = OutcomeStore(tmp_path / "r" / "run1-outcomes")
    seeded = next(s for s in copy_store.current() if s.record["card_id"] == "card-0000d006")
    assert seeded.record["written_by"] == "planted_seed" and seeded.dead


def test_prepare_store_refuses_an_origin_it_cannot_find_or_that_did_not_die(tmp_path, world):
    origins = {"pc-00000001": {"run": "run-20250101T000000Z", "card_id": "card-0000d006"}}
    with pytest.raises(pl.PlantedError, match="neither in the store nor under"):
        pl.prepare_store(world["store"].dir, tmp_path / "copy1", origins, world["runs_root"])
    alive = pf.build_run(world["runs_root"] / "run-20260902T000000Z",
                         [(pf.origin_card("card-0000d006"), [("S0", "pass", ["ok"]), ("S1", "pass", ["ok"])], {})])
    origins = {"pc-00000001": {"run": alive.name, "card_id": "card-0000d006"}}
    with pytest.raises(pl.PlantedError, match="did not die"):
        pl.prepare_store(world["store"].dir, tmp_path / "copy2", origins, world["runs_root"])


# ----- the held-back set and batches ---------------------------------------------------------------------------------

def test_the_held_back_set_runs_separately_and_a_later_batch_meets_its_planted_origin(tmp_path, world):
    report = run_test(tmp_path, world, "held_back")
    got = results_of(report)
    assert report["meta"]["set"] == "fixture_held_back" and report["meta"]["cards"] == 7
    t1_id = got[HELD["T1"]]["card_id"]
    assert got[HELD["T6"]]["kill_codes"] == [f"duplicate:{t1_id}"] and got[HELD["T6"]]["stop"] == "G"
    assert got[HELD["T6"]]["notes"] == []
    order = report["meta"]["runs_meta"][0]["order"]
    assert order[1] == [HELD["T6"]] and HELD["T6"] not in order[0]
    events = list(Ledger(Path(report["meta"]["runs_meta"][0]["run_dir"])).iter_events())
    check = next(e["data"] for e in events if e["type"] == "planted_origin_check")
    assert check == {"key": HELD["T6"], "origin": HELD["T1"], "origin_card_id": t1_id, "origin_dead": True}
    admitted = [e for e in events if e["type"] == "planted_admit"]
    assert admitted[-1]["data"]["key"] == HELD["T6"]
    last_t1 = max(e["seq"] for e in events if e["data"].get("card_id") == t1_id)
    assert admitted[-1]["seq"] > last_t1
    # One run: repeatability cannot be measured, so the test is not accepted, though every per-run metric is met.
    assert report["runs"][0]["passed"] and report["agreement"]["rate"] is None and not report["accepted"]


def test_a_planted_origin_that_survived_is_reported_on_its_variant(tmp_path, world):
    report = run_test(tmp_path, world, "held_back", keys=[HELD["T1"], HELD["T6"]],
                      overrides={"arm_classifier": {"T1": lambda m, i, u: pf.classification("none")}})
    got = results_of(report)
    assert got[HELD["T1"]]["stop"] == "T"
    assert got[HELD["T6"]]["notes"] == [f"origin {HELD['T1']} had not died when this card entered"]


def test_add_types_merges_the_failing_types_original_cards(tmp_path, world, monkeypatch):
    held, _ = pf.sealed_set(tmp_path / "h", "held_back", store=world["store"])
    main_set, _ = pf.sealed_set(tmp_path / "m", "main", store=world["store"])
    monkeypatch.setattr(ps, "set_dir", lambda name: main_set.directory)
    args = type("Args", (), {"sealed_dir": None, "add_types": "T3,T4"})()
    monkeypatch.setenv(ps.SEALED_ENV, str(tmp_path / "h" / "sealed"))
    with pytest.raises(ps.SealError, match="sealed expectations not found"):
        pl.selected_keys(args, held)
    # Both sealed files in one directory, as in ~/.bigbrane/planted.
    (tmp_path / "m" / "sealed" / "fixture_main.sealed.json").rename(tmp_path / "h" / "sealed" /
                                                                   "fixture_main.sealed.json")
    merged, sealed, keys = pl.selected_keys(args, held)
    assert keys == sorted(HELD.values()) + [MAIN["T3"], MAIN["T4"]]
    assert sealed.entries[MAIN["T3"]]["type"] == "T3" and MAIN["T3"] in merged.cards


# ----- refusal paths --------------------------------------------------------------------------------------------------

def cli_args(tmp_path, set_dir, world, *extra):
    return ["planted", "--set", "fixture_main", "--cards-dir", str(set_dir), "--sealed-dir", str(tmp_path / "sealed"),
            "--out", str(tmp_path / "o"), "--run-root", str(tmp_path / "r"), "--store", str(world["store"].dir),
            *extra, "--reason", "refusal test"]


def assert_nothing_created(tmp_path):
    assert not (tmp_path / "o").exists() and not (tmp_path / "r").exists()


def test_refuses_on_low_disk(tmp_path, world, monkeypatch, capsys):
    planted, _ = pf.sealed_set(tmp_path, "main", store=world["store"])
    monkeypatch.setattr(diskguard, "path_free_bytes", lambda path, statvfs=None: 10**9)
    assert main(cli_args(tmp_path, planted.directory, world)) == 2
    assert "the research planted" in capsys.readouterr().err
    assert_nothing_created(tmp_path)


def test_refuses_when_a_local_model_is_not_loaded(tmp_path, world, monkeypatch, capsys):
    planted, _ = pf.sealed_set(tmp_path, "main", store=world["store"])
    monkeypatch.setattr(localmodels, "fetch_models", lambda server, **kwargs: [])
    assert main(cli_args(tmp_path, planted.directory, world)) == 2
    assert "local model check failed" in capsys.readouterr().err
    assert_nothing_created(tmp_path)


def test_refuses_when_the_gateway_check_fails(tmp_path, world, monkeypatch, capsys):
    import hybrid.llm as llm
    from hybrid.llm import GatewayUnavailable

    planted, _ = pf.sealed_set(tmp_path, "main", store=world["store"])

    def down(models):
        raise GatewayUnavailable("gateway backend 'gateway' is not usable: connection refused")

    monkeypatch.setattr(llm, "check_gateways", down)
    assert main(cli_args(tmp_path, planted.directory, world)) == 2
    assert "cannot start the planted test: gateway backend" in capsys.readouterr().err
    assert_nothing_created(tmp_path)


def test_refuses_a_broken_seal_an_open_sealed_file_a_leak_and_a_missing_reason(tmp_path, world, monkeypatch, capsys,
                                                                                gateway_calls):
    planted, sealed = pf.sealed_set(tmp_path, "main", store=world["store"])
    args = cli_args(tmp_path, planted.directory, world)
    assert main(args[:-2]) == 2
    assert "--reason is required" in capsys.readouterr().err
    os.chmod(sealed.path, 0o640)
    assert main(args) == 2
    assert "readable by group or others" in capsys.readouterr().err
    os.chmod(sealed.path, 0o600)
    # A leak found at run time refuses too (here a word of the T1 card made a banned term).
    terms = ps.PLANTED_TERMS
    monkeypatch.setattr(ps, "PLANTED_TERMS", terms + ("ledger",))
    assert main(args) == 2
    assert "'ledger'" in capsys.readouterr().err
    monkeypatch.setattr(ps, "PLANTED_TERMS", terms)
    card_path = planted.directory / "cards" / f"{MAIN['T5']}.json"
    card_path.write_text(card_path.read_text().replace("push and pop", "push, pop"))
    assert main(args) == 2
    assert "card file changed since sealing" in capsys.readouterr().err
    assert_nothing_created(tmp_path)
    # Every refusal above came from an offline check, before the gateway was contacted.
    assert gateway_calls == []


def test_refuses_check_files_that_differ_from_the_manifest(tmp_path, world, monkeypatch, capsys, gateway_calls):
    from hybrid.math import manifest

    planted, _ = pf.sealed_set(tmp_path, "main", store=world["store"])

    def changed(directory=None, path=None):
        raise manifest.ManifestMismatch(["cost.py: hash differs"])

    monkeypatch.setattr(manifest, "verify", changed)
    assert main(cli_args(tmp_path, planted.directory, world)) == 2
    assert "check files differ from MANIFEST.json: cost.py: hash differs" in capsys.readouterr().err
    assert_nothing_created(tmp_path)
    assert gateway_calls == []


def test_refuses_an_output_directory_that_is_not_empty_and_a_set_collision(tmp_path, world, capsys, gateway_calls):
    planted, _ = pf.sealed_set(tmp_path, "main", store=world["store"])
    args = cli_args(tmp_path, planted.directory, world)
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "old.json").write_text("{}")
    assert main(args) == 2
    assert "is not empty" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()
    (tmp_path / "o" / "old.json").unlink()
    # A store record that shares the T7 card's mechanism signature would kill it as a duplicate: refused.
    store = OutcomeStore(world["store"].dir)
    t7 = pf.CARDS["corvane"]
    run = pf.build_run(world["runs_root"] / "run-20260903T000000Z",
                       [(dict(t7, id="card-0000d007"),
                         [("S0", "pass", ["ok"]), ("S1", "fail", ["gate: fatal finding: x"])], {})])
    from hybrid.research.outcome_records import RunView, build_record

    store.append(build_record(RunView(run), "card-0000d007", "backfill"))
    assert main(args) == 2
    assert f"{MAIN['T7']}: mechanism signature equals that of the stored record" in capsys.readouterr().err
    assert gateway_calls == []


def test_the_command_runs_the_whole_test_end_to_end(tmp_path, world, monkeypatch, capsys, gateway_calls):
    """`python -m hybrid.research planted` with fake dependencies: preflights, two runs, the report, exit 0."""
    planted, _ = pf.sealed_set(tmp_path, "main", store=world["store"])
    slots = {"limit": 3, "requested": "auto", "clamped": [], "slot_mb": 4096, "directory": str(tmp_path / "slots"),
             "vm": {"cpus": 6, "memory_mb": 16384}}
    monkeypatch.setattr(pl, "job_slots_from_config", lambda cfg: ("slots", slots))
    monkeypatch.setattr(pl, "real_deps_factory", lambda cfg, routing, job_slots: pf.deps_factory())
    assert main(cli_args(tmp_path, planted.directory, world)) == 0
    out = capsys.readouterr().out
    assert out.startswith("PLANTED-CARD TEST: ") and '"accepted": true' in out
    assert "runs at once: 1; job slots: 3 of 4096 MB" in out
    assert len(gateway_calls) == 1
    report = json.loads((tmp_path / "o" / "report.json").read_text())
    assert report["meta"]["runs"] == 2 and report["agreement"]["rate"] == 1.0
    assert report["meta"]["parallel"] == 1 and report["meta"]["job_slots"] == slots
    assert sorted(p.name for p in (tmp_path / "r").iterdir()) == [
        "run1", "run1-outcomes", "run1-override", "run2", "run2-outcomes", "run2-override"]
    assert sorted(p.name for p in (tmp_path / "r" / "run1-override").iterdir()) == [MAIN["T5"]]


def test_parallel_runs_give_the_serial_result_in_their_own_directories(tmp_path, world):
    """`--parallel 2`: both whole-set runs and their T5 override runs at once, each with its own store copy and run
    directory; the per-card results, the score and outcomes.jsonl are those of the serial test."""
    keys = [MAIN["T1"], MAIN["T5"], MAIN["T7"]]

    def run(name, parallel):
        planted, sealed = pf.sealed_set(tmp_path / name, "main", store=world["store"])
        return asyncio.run(pl.planted_test(
            planted=planted, sealed=sealed, runs=2, run_root=tmp_path / name / "r", out_dir=tmp_path / name / "o",
            cfg=cfg(), reason="parallel test", deps_factory=pf.deps_factory(), store_source=world["store"].dir,
            runs_root=world["runs_root"], keys=keys, parallel=parallel))

    serial, both = run("serial", 1), run("parallel", 2)
    strip = {"card_id", "override", "executor_s"}     # executor_s is measured time
    for key in keys:
        for index in (0, 1):
            got, want = both["cards"][key]["results"][index], serial["cards"][key]["results"][index]
            assert {k: v for k, v in got.items() if k not in strip} == {k: v for k, v in want.items() if k not in strip}
            if want["override"] is not None:
                assert {k: v for k, v in got["override"].items() if k != "run_dir"} == \
                    {k: v for k, v in want["override"].items() if k != "run_dir"}
    assert both["accepted"] == serial["accepted"] and both["agreement"] == serial["agreement"]
    assert both["meta"]["parallel"] == 2 and both["audit"]["differences"] == []
    lines = [json.loads(line) for line in (tmp_path / "parallel" / "o" / "outcomes.jsonl").read_text().splitlines()]
    assert [line["run"] for line in lines] == [1, 1, 1, 2, 2, 2]
    run_dirs = [Path(meta["run_dir"]) for meta in both["meta"]["runs_meta"]]
    assert [d.name for d in run_dirs] == ["run1", "run2"]
    spans = []
    for run_dir in run_dirs:
        events = list(Ledger(run_dir).iter_events())
        spans.append((events[0]["ts"], events[-1]["ts"]))
    assert spans[1][0] < spans[0][1]           # the second run started before the first ended
    with pytest.raises(pl.PlantedError, match="--parallel"):
        run("bad", 0)


# ----- check-hash audit -----------------------------------------------------------------------------------------------

def test_a_file_changed_during_the_test_fails_the_audit(tmp_path, world):
    planted, sealed = pf.sealed_set(tmp_path, "main", store=world["store"])
    base = pf.deps_factory()
    made = []

    def tampering(ledger):
        made.append(ledger.run_dir)
        if len(made) == 1:   # during run 1, after the snapshot taken before the first run
            (planted.directory / "cards" / "stray.json").write_text("{}")
        return base(ledger)

    report = asyncio.run(pl.planted_test(
        planted=planted, sealed=sealed, runs=2, run_root=tmp_path / "r", out_dir=tmp_path / "o", cfg=cfg(),
        reason="audit", deps_factory=tampering, store_source=world["store"].dir, runs_root=world["runs_root"],
        keys=[MAIN["T1"], MAIN["T7"]]))
    # Each run is compared with the snapshot taken before the first, so the change shows after both runs.
    differences = report["audit"]["differences"]
    assert [d.split(":", 1)[0] for d in differences] == ["after run 1", "after run 2"], differences
    assert all(d.endswith("set-main/cards/stray.json added") for d in differences), differences
    assert all(r["passed"] for r in report["runs"]) and report["agreement"]["ok"]
    assert not report["audit"]["ok"] and not report["accepted"]
    assert "Check integrity: " in (tmp_path / "o" / "report.md").read_text()


def test_a_math_check_that_staged_a_different_file_is_a_difference(tmp_path, world):
    from hybrid.math import manifest

    planted, sealed = pf.sealed_set(tmp_path, "main", store=world["store"])
    report = asyncio.run(pl.planted_test(
        planted=planted, sealed=sealed, runs=1, run_root=tmp_path / "r", out_dir=tmp_path / "o", cfg=cfg(),
        reason="audit", deps_factory=pf.deps_factory(), store_source=world["store"].dir,
        runs_root=world["runs_root"], keys=[MAIN["T7"]]))
    run_dir = Path(report["meta"]["runs_meta"][0]["run_dir"])
    recorded = manifest.verify()
    count, found = pl.math_check_differences(run_dir, recorded)
    assert count == report["audit"]["math_checks"] > 0 and found == []
    path = sorted((run_dir / "artifacts" / "math_check").glob("*.json"))[0]
    check = json.loads(path.read_text())
    staged = sorted(check["check_hashes"])[0]
    check["check_hashes"][staged] = "0" * 64
    path.write_text(json.dumps(check))
    count, found = pl.math_check_differences(run_dir, recorded)
    assert len(found) == 1 and f"{staged} hashed 000000000000" in found[0]


def test_planted_config_turns_on_every_switch_section_13_tests():
    base = cfg()
    live = pl.planted_config(base, store_path=Path("results/x"), n_cards=49)
    assert live["s3"]["mode"] == "math" and live["math"]["profile"] and live["target"]["enabled"]
    assert live["refine"]["enabled"] and live["s2"]["family_exclusion"] and live["s2"]["difference_statement"]
    assert live["outcomes"]["enabled"] and live["outcomes"]["enforce_r3"]
    assert live["outcomes"]["path"] == str(Path("results/x").resolve())
    assert live["generation"]["mode"] == "arms" and live["explore"]["p_escape"] == 0.0
    cap = base["budgets"]["idea"]["model_calls"]
    assert live["budgets"]["max_cards"] == 49 and live["budgets"]["max_calls"] >= 49 * cap
    # Per-idea and per-lineage caps are the config's, untouched.
    assert live["budgets"]["idea"] == base["budgets"]["idea"] and live["budgets"]["lineage"] == base["budgets"]["lineage"]
    assert base["s3"]["mode"] == cfg()["s3"]["mode"]   # the input is not modified


def test_seal_and_verify_commands(tmp_path, world, capsys):
    directory = pf.copy_set("main", tmp_path / "set")
    source = tmp_path / "e.json"
    source.write_text((pf.FIXTURES / "main.expectations.json").read_text())
    base = ["planted", "--set", "fixture_main", "--cards-dir", str(directory), "--sealed-dir", str(tmp_path / "s"),
            "--store", str(world["store"].dir)]
    assert main(["planted", "seal", *base[1:]]) == 2
    assert "--expectations" in capsys.readouterr().err
    # check: every sealing check, nothing written.
    assert main(["planted", "check", *base[1:], "--expectations", str(source)]) == 0
    assert "7 cards pass every sealing check" in capsys.readouterr().out
    assert source.exists() and not (directory / ps.SEAL_FILE).exists() and not (tmp_path / "s").exists()
    card_path = directory / "cards" / f"{MAIN['T4']}.json"
    original = card_path.read_text()
    card_path.write_text(original.replace("saturating integrator", "saturating integrator (a T4 card)", 1))
    assert main(["planted", "check", *base[1:], "--expectations", str(source)]) == 2
    assert f"{MAIN['T4']}: names a planted type: 'T4'" in capsys.readouterr().err
    card_path.write_text(original)
    assert main(["planted", "seal", *base[1:], "--expectations", str(source)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cards"] == 7 and (tmp_path / "s" / "fixture_main.sealed.json").is_file()
    assert main(["planted", "verify", *base[1:]]) == 0
    assert "7 cards verify" in capsys.readouterr().out
