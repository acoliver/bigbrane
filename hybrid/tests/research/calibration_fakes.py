"""Fakes for the Phase 4 tests (profile obligations in stage M and the calibration driver).

``FastExecutor`` answers library jobs by running the staged driver (main.py, the copy of hybrid/math/jobmain.py that
MathRunner staged, with the staged bbmath modules) inside the test process instead of a container or a subprocess,
and caches the printed result by request, so the replay and later runs cost nothing. Lean and Z3 jobs are not
needed by these fakes (P3 uses SymPy, answered by the scripted default of math_fakes).

Decision P4P: calibration cards K1 and K2 supply measured learners for every P4 task, which train for minutes on
m = 1200 to 4000 examples. The full check of such a learner (its exact source hash, on the same experiment: task,
size, m, seeds, threshold, layout, widths evaluated at the size, architecture contract) is answered with the result the
host measured for it on check seeds 0 to 2 (fixtures/p4_protocols, copied from results/p4-protocols-20261016 and
results/p4-protocols2-20261016; the K1 composition learner's recorded verdict is violated, decision P4K). The
research tests set the P4 test count to 40 (conftest.py SCRIPTED_PROTOCOL), so the test count is not compared: the
measured result at 500 test instances stands in for it. The timing probe of such a learner is answered with the host's
recorded seconds for one seed (the real probe trains the learner, and its projection against the runner's cap moves with
the load of the machine), and the slow replay in test_p4p_protocols.py trains the recall learner through the same
runner.

The scripted roles write the eleven profile obligations for a calibration card, the prover answers with reference
code from hybrid/math/references.py (unbounded constructions for a card that passes P2, capped ones for a card that
fails it) and small learners that really learn from their examples.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from hybrid.math import learnability as learnability_check
from hybrid.math import profile, references
from hybrid.math.runner import CHECK_FILE
from hybrid.research.mathrecords import formal_claims
from hybrid.research.stages.falsify import claim_items
from hybrid.tests.research import math_fakes as mf

_CACHE: dict[str, str] = {}
_JOB_LOCK = threading.Lock()
P4_PROTOCOLS = Path(__file__).parent / "fixtures" / "p4_protocols"


def experiment_key(spec: dict, architecture) -> str:
    """A learnability experiment as the check runs it, without its test count: task, size, m, seeds, threshold, the
    generator layout and the widths evaluated at the size, and the architecture contract. The layout is compared
    without its echo of the given params: {"V": 16} and {"V": 16, "n_keys": 16} build the same recall layout."""
    from hybrid.math import constructions
    from hybrid.research.stages import card_resources
    task, size = spec["task"], spec["size"]
    name = constructions.SIZE_NAME[task]
    widths = {key: card_resources.evaluate(value, {name: size}) for key, value in (spec.get("widths") or {}).items()}
    if constructions.length_formula(task) is None:
        # The host leaves widths.positions out for a task whose stream length varies per instance (dyck) and the
        # harness uses the longest stream; a recorded request that wrote the longest length states the same experiment.
        widths.pop("positions", None)
    layout = {key: value for key, value in constructions.layout(task, size, spec.get("params")).items()
              if key != "params"}
    return json.dumps([task, size, spec["examples"], sorted(spec["seeds"]), spec.get("threshold"), layout, widths,
                       architecture], sort_keys=True, default=str)


def measured_protocols() -> dict[tuple[str, str], dict]:
    """The recorded host results of the card-supplied learners, by (source sha256, experiment_key)."""
    found = {}
    for path in sorted(P4_PROTOCOLS.glob("*-confirmation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        found[(record["sha256"], experiment_key(record["spec"], record["architecture_contract"]))] = record["result"]
    return found


def recorded_probe_walls() -> dict[tuple[str, str], float]:
    """The seconds the host measured for one seed of each card-supplied learner (the first check seed, controls
    included), by (source sha256, experiment_key without the architecture contract, which a probe does not carry)."""
    found = {}
    for path in sorted(P4_PROTOCOLS.glob("*-confirmation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        found[(record["sha256"], experiment_key(record["spec"], None))] = record["per_seed_wall"][0]["total_s"]
    return found


class FastExecutor:
    def __init__(self, tmp_path: Path):
        self.tmp_path = Path(tmp_path).resolve()
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        self.jobs: list = []
        # Jobs answered with a recorded protocol result (decision P4P).
        self.measured: list = []
        self._measured = measured_protocols()
        self._walls = recorded_probe_walls()

    def reach_threshold(self, task: str) -> None:
        """Answer the recorded learners of ``task`` as if every seed had reached the threshold: a run in which the
        card's own learner learns (the K1 composition learner recorded a failing seed, decision P4K)."""
        for key, result in list(self._measured.items()):
            if result["task"] == task and result["status"] != "discharged":
                rows = [dict(row, accuracy=1.0, passed=True, reason=None, first_error=None) for row in result["results"]]
                evidence = f"{task} at size {result['size']} from m={result['examples']} examples: every seed reaches the threshold (scripted)"
                self._measured[key] = dict(result, status="discharged", counterexample=None, results=rows,
                                           evidence=evidence)

    @staticmethod
    def _recorded_probe(spec: dict, wall_s: float) -> dict:
        """A timing probe report whose projection of the full check is the recorded seconds per seed."""
        n = learnability_check.probe_examples(spec)
        phases = {phase: 0.0 for phase in learnability_check.SCALED_PHASES + learnability_check.FIXED_PHASES}
        phases["train"] = wall_s * n / spec["examples"]
        timing = {"seed": spec["seeds"][0], "examples": n, "total_s": wall_s, "phases": phases,
                  "probe_accuracy": 0.0}
        projected = learnability_check.projected_seconds(spec, timing)
        return {"status": "unknown", "tier": "executed", "counterexample": None, "probe": True, "timing": timing,
                "projected_s": round(projected, 3),
                "evidence": (f"timing probe only, decides nothing: recorded host time of seed {timing['seed']}, "
                             f"{wall_s:.1f} s; projected full check {projected:.1f} s (scripted)")}

    def _answer(self, job, stdout: str):
        out, err = self.tmp_path / f"{job.job_id}.stdout", self.tmp_path / f"{job.job_id}.stderr"
        out.write_text(stdout)
        err.write_text("")
        return SimpleNamespace(status="ok", detail="", exit_code=0, run_s=0.01, stdout_path=str(out),
                               stderr_path=str(err))

    def run(self, job):
        self.jobs.append(job)
        staged = Path(job.input_dir)
        if (staged / CHECK_FILE).is_file():
            raise AssertionError("no Lean statements in the Phase 4 fakes")
        raw = (staged / "request.json").read_text()
        request = json.loads(raw)
        if request["kind"] == "sympy":
            return self._answer(job, json.dumps(mf.sympy_default(request), sort_keys=True) + "\n")
        if request["kind"] in ("z3", "predicate_agreement"):
            return self._answer(job, json.dumps(mf.z3_default(request), sort_keys=True) + "\n")
        if request["kind"] == "learnability_probe":
            sha = hashlib.sha256(request["source"].encode("utf-8")).hexdigest()
            if sha in {recorded_sha for recorded_sha, _ in self._measured}:
                # The probe of a card-supplied learner trains it for real, and its projection depends on how busy the
                # machine is. The recorded host time of one seed stands in, so the verdict does not.
                wall = self._walls.get((sha, experiment_key(request["spec"], None)))
                if wall is None:
                    raise AssertionError("no recorded timing for this card protocol learner on this experiment (a "
                                         "unit test must not train it): "
                                         + experiment_key(request["spec"], request.get("architecture")))
                return self._answer(job, json.dumps(self._recorded_probe(request["spec"], wall)) + "\n")
        if request["kind"] == "learnability":
            sha = hashlib.sha256(request["source"].encode("utf-8")).hexdigest()
            result = self._measured.get((sha, experiment_key(request["spec"], request.get("architecture"))))
            if result is not None:
                self.measured.append(job)
                return self._answer(job, json.dumps(result, default=str, sort_keys=True) + "\n")
            if sha in {recorded_sha for recorded_sha, _ in self._measured}:
                # A card-supplied learner trains for minutes. A unit test must never run one: it either asks for an
                # experiment that has a recorded result, or it uses a card copy without protocols.
                raise AssertionError(
                    "no recorded result for this card protocol learner on this experiment (a unit test must not "
                    f"train it): {experiment_key(request['spec'], request.get('architecture'))}")
        # Stages call the executor through asyncio.to_thread, so cards in M and C at once run jobs on several threads;
        # redirect_stdout swaps the process-wide sys.stdout, so one job at a time.
        with _JOB_LOCK:
            if raw not in _CACHE:
                spec = importlib.util.spec_from_file_location(f"bbjob_{job.job_id.replace('-', '_')}",
                                                              staged / "main.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    module.main()
                _CACHE[raw] = buffer.getvalue()
            printed = _CACHE[raw]
        return self._answer(job, printed)


# ----- prover sources ------------------------------------------------------------------------------------------------

CAPPED_COMPOSITION = '''
CAP = 2

def init(layout):
    return {"g": {}, "f": {}, "phase": 0, "key": -1, "seps": [layout["GSEP"], layout["FSEP"], layout["QSEP"]]}

def update(state, token):
    if token in state["seps"]:
        state["phase"] = state["seps"].index(token) + 1
        state["key"] = -1
    elif state["phase"] == 3:
        state["key"] = token
    elif state["key"] < 0:
        state["key"] = token
    else:
        table = state["g" if state["phase"] == 1 else "f"]
        if len(table) < CAP:
            table[state["key"]] = token
        state["key"] = -1
    return state

def predict(state):
    return state["f"].get(state["g"].get(state["key"], -1), 0)
'''

COPY_LEARNER = '''
RECIPE = {"method": "exhaustive search over a small rule class", "init": "none"}
def learn(layout, examples, seed):
    if not examples:
        return {"offset": 0}
    best, best_score = 0, -1
    for offset in range(1, 2 * layout["size"] + 2):
        score = 0
        for ex in examples:
            toks = ex["tokens"]
            for pos, ans in ex["queries"]:
                src = pos - offset
                if 0 <= src < len(toks) and toks[src] == ans:
                    score += 1
        if score > best_score:
            best, best_score = offset, score
    return {"offset": best}

def init(layout, params):
    return {"seen": [], "offset": params["offset"]}

def update(state, token):
    state["seen"] = state["seen"] + [token]
    return state

def predict(state):
    src = len(state["seen"]) - 1 - state["offset"]
    if state["offset"] > 0 and 0 <= src < len(state["seen"]):
        return state["seen"][src]
    return 0
'''

RECALL_LEARNER = '''
RECIPE = {"method": "exhaustive search over a small rule class", "init": "none"}
SHIFTS = [-2, -1, 2, 3, 1]

def answer(toks, pos, shift):
    first = -1
    for i in range(pos):
        if toks[i] == toks[pos]:
            first = i
            break
    j = first + shift
    if first >= 0 and 0 <= j < pos:
        return toks[j]
    return -1

def learn(layout, examples, seed):
    if not examples:
        return {"shift": 0}
    best, best_score = 0, -1
    for shift in SHIFTS:
        score = sum(1 for ex in examples for pos, ans in ex["queries"] if answer(ex["tokens"], pos, shift) == ans)
        if score > best_score:
            best, best_score = shift, score
    return {"shift": best}

def init(layout, params):
    return {"seen": [], "shift": params["shift"]}

def update(state, token):
    state["seen"] = state["seen"] + [token]
    return state

def predict(state):
    return answer(state["seen"], len(state["seen"]) - 1, state["shift"])
'''

DYCK_LEARNER = '''
RECIPE = {"method": "exhaustive search over a small rule class", "init": "none"}
def learn(layout, examples, seed):
    counts = {}
    for ex in examples:
        wanted = dict((pos, ans) for pos, ans in ex["queries"])
        stack = []
        for pos, tok in enumerate(ex["tokens"]):
            if tok in layout["OPEN"]:
                stack = stack + [tok]
            elif stack:
                stack = stack[:-1]
            if stack and pos in wanted:
                row = counts.get(stack[-1], {})
                row[wanted[pos]] = row.get(wanted[pos], 0) + 1
                counts[stack[-1]] = row
    return dict((top, max(row, key=lambda c: row[c])) for top, row in counts.items())

def init(layout, params):
    return {"stack": [], "close_of": params, "open": list(layout["OPEN"])}

def update(state, token):
    if token in state["open"]:
        state["stack"] = state["stack"] + [token]
    elif state["stack"]:
        state["stack"] = state["stack"][:-1]
    return state

def predict(state):
    if not state["stack"]:
        return 2
    return state["close_of"].get(state["stack"][-1], 2)
'''

COMPOSITION_LEARNER = '''
RECIPE = {"method": "exhaustive search over a small rule class", "init": "none"}
RULES = [0, 1, 2, 3]   # g(x), f(x), g(f(x)), f(g(x)); the state holds numbers only

def init(layout, params):
    return {"g": {}, "f": {}, "phase": 0, "key": -1, "rule": params["rule"],
            "seps": [layout["GSEP"], layout["FSEP"], layout["QSEP"]]}

def update(state, token):
    if token in state["seps"]:
        state["phase"] = state["seps"].index(token) + 1
        state["key"] = -1
    elif state["phase"] == 3 or state["key"] < 0:
        state["key"] = token
    else:
        state["g" if state["phase"] == 1 else "f"][state["key"]] = token
        state["key"] = -1
    return state

def apply(rule, g, f, x):
    if rule == 0:
        return g.get(x, -1)
    if rule == 1:
        return f.get(x, -1)
    if rule == 2:
        return g.get(f.get(x, -1), -1)
    return f.get(g.get(x, -1), -1)

def predict(state):
    return apply(state["rule"], state["g"], state["f"], state["key"])

def learn(layout, examples, seed):
    best, best_score = RULES[0], -1
    for rule in RULES:
        score = 0
        for ex in examples:
            wanted = dict((pos, ans) for pos, ans in ex["queries"])
            state = init(layout, {"rule": rule})
            for pos, tok in enumerate(ex["tokens"]):
                state = update(state, tok)
                if pos in wanted and predict(state) == wanted[pos]:
                    score += 1
        if score > best_score:
            best, best_score = rule, score
    return {"rule": best}
'''

# Decision P4K: K1's two-layer attention model does not reliably learn T=4 composition from the stated examples (a
# measured result, not a pipeline output). This learner stands for a run that does not: it settles on the most frequent
# answer, which a trained model that memorizes or keeps a frequency shortcut also does on unseen composition queries.
SHORTCUT_COMPOSITION_LEARNER = '''
RECIPE = {"method": "most frequent answer", "init": "none"}

def init(layout, params):
    return {"answer": params["answer"]}

def update(state, token):
    return state

def predict(state):
    return state["answer"]

def learn(layout, examples, seed):
    counts = {}
    for ex in examples:
        for pos, ans in ex["queries"]:
            counts[ans] = counts.get(ans, 0) + 1
    return {"answer": max(counts, key=lambda a: counts[a]) if counts else 0}
'''

LEARNERS = {"copy": COPY_LEARNER, "recall": RECALL_LEARNER, "dyck": DYCK_LEARNER, "composition": COMPOSITION_LEARNER}
OPEN = {"copy": references.construction_source(references.COPY_BUFFER),
        "recall": references.construction_source(references.RECALL_TABLE),
        "dyck": references.construction_source(references.DYCK_STACK),
        "composition": references.COMPOSITION_TABLES, "binding": references.BINDING_ENV}
# A four-bit or two-symbol mechanism: copy and recall keep almost nothing, the stack holds four levels, the tables
# two entries. Binding still passes (P2.composition fails on the two-hop lookup).
CAPPED = {"copy": references.construction_source(references.COPY_BUFFER, 2),
          "recall": references.construction_source(references.RECALL_TABLE, 1),
          "dyck": references.construction_source(references.DYCK_STACK, 4),
          "composition": CAPPED_COMPOSITION, "binding": references.BINDING_ENV}
STATE_WORDS = {"copy": "2*L + 8", "recall": "4*K + 8", "dyck": "2*depth + 8", "composition": "8*T + 8",
               "binding": "8*hops + 16"}
NS = [16, 32, 64, 128]

# Per calibration card: the reference cost code and its formula, which constructions it gets, its P3 limit, and the
# numbers it gives P5.
COST = {
    "k1": (references.SOFTMAX_ATTENTION, dict(references.REFERENCES["softmax_attention"][1], ns=NS,
                                              widths={"d": "d", "rounds": "1"})),
    "k2": (references.ELMAN_RNN, dict(references.REFERENCES["elman_rnn"][1], ns=NS, widths={"d": "d", "rounds": "1"})),
    "k3": (references.HMM_FORWARD, dict(references.REFERENCES["hmm_forward"][1], ns=NS,
                                        widths={"d": "k", "rounds": "1"})),
    "k4": (references.DECAYED_UNIGRAM, dict(references.REFERENCES["decayed_unigram"][1], ns=NS,
                                            widths={"d": "V", "rounds": "1"})),
}
# The shared architecture contract every calibration card freezes (FIX-4): square feed-forward matrices, binary32;
# the scripted learners keep no array wider than 64.
ARCHITECTURE = {"hidden": "d", "precision_bits": 32}
LEARNER_WIDTHS = {"d": "64", "rounds": "1"}
HYPOTHESIS_CLASS = {"definition": "every readout table from 4 symbols to 4 answers",
                    "size_justification": "4**4 = 256 tables; the scripted mechanisms supply no prior bits",
                    "log2_size": 8}
# FIX-17 (decision P3): each limit names the part of the mechanism it follows from and quotes the card's text there.
# FIX-18 (decision L): the limits are the card's own (hybrid/fixtures/calibration/k*.json, field limits), copied with
# the card's task, size, resources and why; the scripted fixer adds only the basis.
LIMIT_BASES = {
    "k1": {"part": "update", "anchor": "mechanism.update.embed", "quote": "a symbol table row plus a position table row"},
    "k2": {"part": "state", "anchor": "mechanism.state.s", "quote": "The store carried from token to token"},
    "k3": {"part": "state", "anchor": "mechanism.state.z", "quote": "Index of the current latent state, one of k = 16"},
    "k4": {"part": "state", "anchor": "mechanism.state.w", "quote": "The last two symbols read"},
}
CALIBRATION_CARDS = Path(__file__).resolve().parents[2] / "fixtures" / "calibration"


def copied_limits(card_limits: list[dict], basis: dict) -> list[dict]:
    """A fixer's copy of a card's stated limits: the card's task, size, resources and why, with ``basis`` (its part
    is the limit's kind)."""
    return [{"task": limit["task"], "size": limit["size"], "resources": limit["resources"], "why": limit["why"],
             "basis": dict(basis, part=limit["kind"])} for limit in card_limits]


LIMITS = {key: copied_limits(json.loads((CALIBRATION_CARDS / f"{key}.json").read_text())["limits"], basis)
          for key, basis in LIMIT_BASES.items()}
PASSES_P2 = {"k1": True, "k2": True, "k3": False, "k4": False}
EXAMPLES = 40


# FIX-11: the card claim each card's copy, recall and dyck numbers cite (its capability claim for that task).
P5_CLAIMS = {"k1": {"copy": "capability.1", "recall": "capability.2", "dyck": "capability.3"},
             "k2": {"copy": "capability.1", "recall": "capability.2", "dyck": "capability.3"},
             "k3": {"copy": "capability.3", "recall": "capability.4", "dyck": "capability.1"},
             "k4": {"copy": "capability.1", "recall": "capability.2", "dyck": "capability.3"}}


def card_resources_of(card: dict | None, key: str | None = None) -> dict:
    """FIX-16: the card's resources. Without a card (or for a card without them) the resources of calibration card
    ``key``, else K2's, so a scripted answer for k3 never mixes in K2's values."""
    if card is not None and isinstance(card.get("resources"), dict):
        return card["resources"]
    from hybrid.research.calibration import FIXTURES
    return json.loads((FIXTURES / f"{key if key in PASSES_P2 else 'k2'}.json").read_text())["resources"]


def without_protocols(card: dict) -> dict:
    """A copy of the card with the learning protocols of decision P4P removed, so stage M asks the prover for every
    learner as it did before the calibration cards supplied theirs. For a test that uses a calibration card as the
    vehicle for prover rounds, budgets, contracts, premises or tiers, and never checks the card's own learners."""
    card = copy.deepcopy(card)
    for entry in card["resources"]["learning"].values():
        entry.pop("protocol", None)
    return card


def frozen(claim_id: str, statement: dict, card: dict | None = None, key: str | None = None) -> dict:
    """FIX-16: the spec of a scripted statement as the host freezes it: every protocol, card and derived value filled
    in from the card's resources (``card_resources_of``). The scripted statement copies or leaves out those values,
    so the host finds nothing to refuse."""
    from hybrid.research.stages import card_resources
    form = statement["form"]
    spec, problems = card_resources.normalize_statement({"resources": card_resources_of(card, key)}, claim_id, form,
                                                        statement[form])
    assert problems == [], problems
    return spec


def p5_numbers(key: str, card: dict | None = None) -> dict:
    # FIX-8: an entry at every P2 size (profile.p5_coverage_problems). FIX-11: every numeric accuracy cites the card
    # claim that states it (profile.resolve_p5_accuracies).
    cites = P5_CLAIMS[key]
    if PASSES_P2[key]:
        accuracy, bits = 0.99, 1_000_000
        dyck = [{"depth": depth, "accuracy": accuracy} for depth in profile.REQUIRED_SIZES["dyck"]]
    else:
        # Chance level for recall: an answer that is always wrong would carry information too (Fano).
        accuracy, bits = 0.0, 4
        dyck = [{"depth": 2, "accuracy": 0.99}, {"depth": 4, "accuracy": 0.99}, {"depth": 8, "accuracy": 0.0},
                {"depth": 16, "accuracy": 0.0}]
    # FIX-16: each entry's state_bits is derived by the host from the card's resources; the fixer leaves it out.
    examples = sorted({entry["examples"] for entry in card_resources_of(card, key)["learning"].values()})
    return {"copy": [{"D": 4, "L": L, "accuracy": accuracy, "claim": cites["copy"]}
                     for L in profile.REQUIRED_SIZES["copy"]],
            "recall": [{"K": K, "V": 16, "accuracy": accuracy or 1 / 16, "claim": cites["recall"]}
                       for K in profile.REQUIRED_SIZES["recall"]],
            "dyck": [dict(entry, claim=cites["dyck"]) for entry in dyck],
            "sample": [{"log2_H": 8, "prior_bits": 0, "bits_per_example": 2, "examples": m,
                        "hypothesis_class": dict(HYPOTHESIS_CLASS)} for m in examples]}


def construction(task: str) -> dict:
    # FIX-16: count, threshold, params, widths and state_words come from the protocol and the card; the host fills
    # them in.
    return {"form": "construction", "construction": {"task": task, "sizes": list(profile.REQUIRED_SIZES[task])}}


def learnability(task: str, size: int | None = None, card: dict | None = None, key: str | None = None) -> dict:
    # FIX-16: the P4 case (size, examples m) and the widths are the card's; seeds and test_count the protocol's.
    learned = card_resources_of(card, key)["learning"][task]
    return {"form": "learnability", "learnability": {"task": task, "size": learned["size"],
                                                     "examples": learned["examples"], "seeds": [0, 1, 2]}}


def quote(slot: str) -> str:
    return profile.SLOT_TEXT[slot][:40]


def slot_obligation(oid: str, slot: str, statements: list[dict]) -> dict:
    return mf.obligation(oid, slot, quote(slot), statements)


def fixer_output(key: str, card: dict, *, drop_slots=(), extra=None) -> dict:
    """The statement fixer's answer for a calibration card: the eleven profile slots, and the card's own claims as
    not formalized (calibration scores the profile only)."""
    _, cost_spec = COST[key]
    resources = card_resources_of(card, key)
    # FIX-16: the cost sizes are the card's names on the protocol grid; the widths are the card's; the lengths are
    # the protocol's (left out when it runs the plan's lengths).
    cost_spec = dict(cost_spec, sizes={name: list(profile.PROTOCOL["cost_grid"]) for name in resources["cost"]["sizes"]},
                     widths={"d": resources["cost"]["d"], "rounds": resources["cost"]["rounds"]})
    cost_spec.pop("ns", None)
    if profile.PROTOCOL["p1_ns"] is not None:
        cost_spec["ns"] = list(profile.PROTOCOL["p1_ns"])
    obligations = [
        slot_obligation("O1", "P1", [{"form": "cost", "cost": dict(cost_spec)}]),
        slot_obligation("O2", "P2.copy", [construction("copy")]),
        slot_obligation("O3", "P2.recall", [construction("recall")]),
        slot_obligation("O4", "P2.nesting", [construction("dyck")]),
        slot_obligation("O5", "P2.composition", [construction("composition"), construction("binding")]),
        slot_obligation("O6", "P3", [mf.sympy_statement("2**(n + 4)", "16*2**n")]),
        slot_obligation("O7", "P4.copy", [learnability("copy", card=card, key=key)]),
        slot_obligation("O8", "P4.recall", [learnability("recall", card=card, key=key)]),
        slot_obligation("O9", "P4.nesting", [learnability("dyck", card=card, key=key)]),
        slot_obligation("O10", "P4.composition", [learnability("composition", card=card, key=key)]),
        slot_obligation("O11", "P5", [{"form": "conservation", "conservation": p5_numbers(key, card)}]),
    ]
    obligations = [o for o in obligations if o["claim_id"] not in drop_slots]
    claims = [c["id"] for c in formal_claims(claim_items(card))]
    # FIX-11: a card with other claim ids (the planted cards have only capability.1) cites its first capability.
    for o in obligations:
        if o["claim_id"] == "P5":
            for entry in (e for s in o["statements"] for section in ("copy", "recall", "dyck")
                          for e in s["conservation"].get(section, [])):
                if entry["claim"] not in claims:
                    entry["claim"] = next(cid for cid in claims if cid.startswith("capability."))
    answer = {"obligations": obligations, "definitions": [], "premises": [],
              "not_formalized": [{"claim_id": cid, "why": "calibration scores the profile only"} for cid in claims],
              "limits": LIMITS[key], "architecture": architecture_of(card, key)}
    answer.update(extra or {})
    return answer


def architecture_of(card: dict | None, key: str | None = None) -> dict:
    """FIX-16: the architecture contract copied from the card's resources."""
    stated = card_resources_of(card, key)["architecture"]
    return {key: value for key, value in stated.items() if value is not None}


def _shown(user: str) -> list[dict]:
    return json.JSONDecoder().raw_decode(user.split("FROZEN STATEMENTS:\n", 1)[1])[0]


def prover_output(key: str, *, passes_p2: bool | None = None, learns_composition: bool = True) -> callable:
    """``learns_composition`` False gives the prover the shortcut learner for composition (decision P4K); the other
    learners and every other test keep the learner that reaches the threshold."""
    passes = PASSES_P2[key] if passes_p2 is None else passes_p2
    learners = dict(LEARNERS)
    if not learns_composition:
        learners["composition"] = SHORTCUT_COMPOSITION_LEARNER
    cost_source, _ = COST[key]
    sources = OPEN if passes else CAPPED

    def answer(index: int, user: str) -> dict:
        attempts = []
        for item in _shown(user):
            spec = item["frozen_statement"]
            if item["form"] == "cost":
                source = cost_source
            elif item["form"] == "construction":
                source = sources[spec["task"]]
            elif item["form"] == "learnability":
                source = learners[spec["task"]]
            else:
                raise AssertionError(f"no scripted source for {item['form']}")
            attempts.append({"statement_id": item["statement_id"], "proof": None, "source": source,
                             "notes": "scripted"})
        return {"attempts": attempts}

    return answer


def script(key: str, card: dict, *, passes_p2: bool | None = None, learns_composition: bool = True,
           fixer=None) -> dict:
    return {"statement_fixer": fixer or fixer_output(key, card),
            "prover": prover_output(key, passes_p2=passes_p2, learns_composition=learns_composition),
            "challenger": mf.challenger_output(), "statement_backtranslate": mf.backtranslate_output,
            "statement_match": mf.match_output()}


def card_key(ledger) -> tuple[str, int]:
    data = next(e["data"] for e in ledger.iter_events("operator_override"))
    return data["calibration"]["card"], data["calibration"]["run"]

