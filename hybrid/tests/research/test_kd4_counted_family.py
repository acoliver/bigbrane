"""KD4: the excluded-family gate at S2 is decided from the comparator(s) the host counted, not from the reduction
critic's free `family` label (results/kd4-20261017/REPORT.md).

The novelty probe (results/novelty-probe-20261017) killed card-17100003, a Minsky two-counter machine, as
`excluded_family:ssm`: the counted comparator was in family other_known (exact, 305 inputs), the lifted-state ssm
comparator was "not counted (declared in another family)", and the critic's `family` said ssm. The three probe
records are replayed from their saved reduction_critic artifacts (fixtures/kd4_novelty_probe_20261017, sha256 in
SHA256SUMS), with no model call. The synthetic cases run S2 end to end with scripted models."""
from hybrid.tests.research.reference_replays import kd8_record
from hybrid.tests.research.critic_answers import KNOWN, NOT_KNOWN
import asyncio
import json
from pathlib import Path

import pytest

from hybrid.research.outcomes import OutcomeStore
from hybrid.research.stages import difference
from hybrid.research.stages.s2_prior_art import counted_comparisons
from hybrid.tests.research.test_difference import config, critic, mine
from hybrid.tests.research.test_orchestrator_fakes import comparison_check, declared, make_orchestrator, synthetic_declared

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "kd4_novelty_probe_20261017"
FAMILIES, KEYWORDS = difference._families("hybrid/config/families.toml", "hybrid/config/archive.toml")


def probe(card_id: str) -> dict:
    historical = json.loads((FIXTURES / f"reduction_critic-{card_id}.json").read_text(encoding="utf-8"))
    replay = kd8_record({"card": card_id, "comparators": historical["comparators"],
                         "comparisons": historical["reduction_check"]["comparisons"]})
    return dict(historical, comparators=replay["comparators"],
                reduction_check=dict(historical["reduction_check"], comparisons=replay["comparisons"]))


def excluded_id(artifact: dict) -> str | None:
    """The gate's decision for a saved critic artifact: its recorded outcome and check report, no model call."""
    counted = counted_comparisons(artifact, artifact["reduction_check"])
    found = difference.excluded_reduction(artifact, artifact["reduction_outcome"], FAMILIES, KEYWORDS, counted)
    return found.id if found else None


# ----- replay of the probe records --------------------------------------------------------------------------------------


def test_card_17100003_is_no_longer_an_excluded_kill():
    artifact = probe("card-17100003")
    assert artifact["family"] == "ssm" and artifact["reduces_to"] == "other_known"      # the critic's label, as recorded
    assert artifact["reduction_outcome"] == "verified"
    assert [c["declared_id"] for c in counted_comparisons(artifact, artifact["reduction_check"])] == ["minsky_two_counter"]
    assert excluded_id(artifact) is None


@pytest.mark.parametrize("card_id, family", [("card-17100009", "ssm"), ("card-1710000a", "delta_rule")])
def test_the_probe_controls_stay_excluded_kills(card_id, family):
    assert excluded_id(probe(card_id)) == family


def test_the_counted_comparators_are_the_ones_that_verified():
    """card-1710000a declared two linear_attention comparators; only the delta rule one is within the tolerance."""
    counted = counted_comparisons(probe("card-1710000a"), probe("card-1710000a")["reduction_check"])
    assert [c["declared_id"] for c in counted] == ["deltanet_delta_rule"]
    assert {c["family"] for c in counted} == {"linear_attention"}


# ----- the decision on its own ------------------------------------------------------------------------------------------


def counted(family, text="", declared_id="c"):
    return {"declared_id": declared_id, "family": family, "comparator": text}


def decide(crit, outcome, comparators):
    found = difference.excluded_reduction(crit, outcome, FAMILIES, KEYWORDS, comparators)
    return found.id if found else None


def test_the_critic_naming_an_excluded_family_does_not_kill_when_an_in_scope_comparator_was_counted():
    ssm = critic(reduces_to="other_known", family="ssm")
    assert decide(ssm, "verified", [counted("other_known")]) is None


def test_a_counted_excluded_comparator_kills_whatever_the_critic_names():
    for named in ("none", "exact_match_cache", "hopfield_associative_memory", "transformer"):
        assert decide(critic(reduces_to="ssm", family=named), "verified", [counted("ssm")]) == "ssm"


def test_without_a_verified_reduction_nothing_is_excluded():
    for outcome in ("refuted", "unverified", "none"):
        assert decide(critic(reduces_to="ssm", family="ssm"), outcome, [counted("ssm")]) is None


def test_an_excluded_comparator_among_several_counted_families_decides():
    """Any counted comparator in an excluded family decides; the families.toml order breaks a tie among them."""
    crit = critic(reduces_to="other_known", family="none")
    assert decide(crit, "verified", [counted("other_known"), counted("ssm")]) == "ssm"
    assert decide(crit, "verified", [counted("ssm"), counted("other_known")]) == "ssm"
    assert decide(crit, "verified", [counted("hopfield_or_kernel_memory"), counted("other_known")]) is None
    assert decide(crit, "verified", [counted("ssm"), counted("transformer_variant")]) == "transformer"


@pytest.mark.parametrize("named, text, expected", [
    ("linear_attention", "", "linear_attention"),            # the critic's label picks among the counted family's siblings
    ("delta_rule", "", "delta_rule"),
    ("none", "Delta-rule fast weight programmer", "delta_rule"),     # the host's own comparator text decides otherwise
    ("none", "Unnormalized additive linear attention", "linear_attention"),
    ("none", "", "linear_attention"),                        # a tie takes the family listed first in families.toml
    ("ssm", "Delta-rule fast weight programmer", "delta_rule"),       # a label outside the counted family is ignored
    ("transformer", "", "linear_attention"),
])
def test_siblings_sharing_the_counted_family_are_told_apart_but_never_added(named, text, expected):
    crit = critic(reduces_to="linear_attention", family=named, mapping="m", why="w", difference_if_any="d")
    assert decide(crit, "verified", [counted("linear_attention", text)]) == expected


# ----- the whole gate, with scripted models ------------------------------------------------------------------------------


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261017T080000Z"


@pytest.fixture
def empty(tmp_path):
    return OutcomeStore(tmp_path / "empty-outcomes")


def s2_of(orch):
    return orch.store.get_card(orch.store.list_cards()[0])["status"]["verdicts"]["S2"]


def kill_codes(orch, run_dir, empty):
    return mine(empty, run_dir)[orch.store.list_cards()[0]]["kill_codes"]


LIFTED = declared("ssm", "onehot_lifted_ssm", ("lifted_state",))


def run_s2(run_dir, empty, crit):
    orch, _ = make_orchestrator(run_dir, cfg=config(empty), script={
        "reduction_critic": crit, "reduction_check": comparison_check(0.0)})
    asyncio.run(orch.run())
    return orch


def test_an_ssm_label_with_only_an_in_scope_comparator_counted_takes_the_verified_reduction_path(run_dir, empty):
    crit = critic(reduces_to="other_known", family="ssm", **KNOWN, mapping="a synthetic Dyck-2 routing control, not the recorded Minsky card",
                  family_why="A lifted-state repair over reachable configurations is an ssm.",
                  comparators=[synthetic_declared("other_known"), LIFTED])
    orch = run_s2(run_dir, empty, crit)
    s2 = s2_of(orch)
    assert s2["verdict"] == "reduces_to_known"
    assert s2["reasons"][-1].startswith("gate: update rule reduces to other_known:")
    assert any("not counted (declared in another family than other_known)" in r for r in s2["reasons"])
    assert not any("excluded_family" in r for r in s2["reasons"])
    assert kill_codes(orch, run_dir, empty) == ["reduces_to_known:other_known"]
    assert orch.ctx.latest_json_artifact(orch.store.list_cards()[0], "reduction_critic")["family"] == "ssm"


def test_an_excluded_comparator_that_was_counted_kills_with_its_family(run_dir, empty):
    control = dict(synthetic_declared("ssm"), id="diag_ssm")
    crit = critic(reduces_to="ssm", family="none", **KNOWN, mapping="h_t = a h_{t-1} + b x_t",
                  comparators=[control])
    orch = run_s2(run_dir, empty, crit)
    s2 = s2_of(orch)
    assert s2["verdict"] == "reduces_to_known" and s2["reasons"][-1].startswith("gate: excluded_family:ssm:")
    assert "counted comparator: diag_ssm" in s2["reasons"][-1]
    assert kill_codes(orch, run_dir, empty) == ["excluded_family:ssm"]


@pytest.mark.parametrize("reduces_to, family", [("other_known", "ssm"), ("ssm", "none")])
def test_confidence_without_execution_cannot_create_a_counted_family(run_dir, empty, reduces_to, family):
    crit = critic(reduces_to=reduces_to, family=family, confidence=0.95, **KNOWN, mapping="a mapping")
    orch, _ = make_orchestrator(run_dir, cfg=config(empty, statement=False), script={
        "reduction_critic": crit, "reduction_check": {"check_code": None, "explanation": "no numeric check is possible"}})
    asyncio.run(orch.run())
    verdict = s2_of(orch)
    assert verdict["verdict"] == "pass"
    assert any("confidence cannot establish verification" in reason for reason in verdict["reasons"])
    assert not any("gate: excluded_family" in reason or "gate: update rule reduces" in reason
                   for reason in verdict["reasons"])
    record = orch.ctx.latest_json_artifact(orch.store.list_cards()[0], "reduction_critic")
    assert record["reduction_outcome"] == "unverified" and record["reduction_check"] is None
    assert counted_comparisons(record, record["reduction_check"]) == []
