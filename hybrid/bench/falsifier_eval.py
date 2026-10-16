"""Card falsifier tasks: scoring and pre-registered rule decisions. NumPy and hybrid.bench.metrics only.

Runs in two places, always as staged code: the evaluator job (scores every entrant's falsifier predictions
and decides each task's rule) and the falsifier validation job (checks that the oracle scores itself
perfectly and that a trivial predictor does not meet a rule that would support the card).

A falsifier task is designed per card by the `falsifier_designer` role. Its pass rules (one to three) are
pre-registered before any entrant runs. Each rule names the card claims it decides and quotes the card text
it comes from:

    {"claims": ["falsifier.3", ...], "quote": "<card text>",
     "conditions": [{"entrant": "candidate", "minus": "linear_attention" | null, "metric": "acc" | "kl",
                     "group": "<group label>" | null, "op": ">=" | "<=" | ">" | "<", "value": float}, ...],
     "if_met": "supported" | "refuted"}

A condition compares metric(entrant, group) - metric(minus, group) (or metric(entrant, group) when minus is
null) with value. "entrant" and "minus" name "candidate", a bench baseline, or the trivial references
"uniform" and "unigram". group null means the whole test set. The rule is met when every condition holds;
the claim it decides is then `if_met`, and otherwise the opposite.

A rule may also carry "control_rule": the index of another rule of the same task that the candidate must meet
first. A rule that can support the card from a low score (a limit such as "accuracy at most 0.6", or a falsifier
that was not observed) says nothing about capacity unless the same candidate shows the capability in the same
run; the control rule is that capability.

The decision is "inconclusive" instead when the candidate has no valid predictions, when the task's protocol
does not fit the candidate (see task_blocked), when it did not train on the task (it does not beat the better
of the uniform and unigram references by TRAINED_KL_GAIN nats of KL or TRAINED_ACC_GAIN of accuracy), when the
rule's control rule is not met, or when a condition names an entrant without valid predictions.
"""
from __future__ import annotations

import math

import numpy as np

from hybrid.bench import metrics

TRAINED_KL_GAIN = 0.05
TRAINED_ACC_GAIN = 0.05
REFERENCES = ("uniform", "unigram")
OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b}
OPPOSITE = {"supported": "refuted", "refuted": "supported"}


def score(kind, oracle, pred, group, group_labels):
    """Task metrics with a per-group block always present (also for a single group)."""
    out = metrics.task_metrics(kind, oracle, pred, group, group_labels)
    if "groups" not in out:
        out["groups"] = {str(group_labels[0]): {"kl": out["kl"], "acc": out["acc"], "n": out["n"]}}
    return out


def uniform_pred(n, vocab):
    return np.full((n, vocab), 1.0 / vocab)


def unigram_dist(train, vocab):
    """Add-one smoothed token frequencies of the training sequences."""
    counts = np.zeros(vocab)
    for seq in train:
        np.add.at(counts, np.asarray(seq, dtype=np.int64), 1.0)
    return (counts + 1.0) / (counts.sum() + vocab)


def references(kind, oracle, group, group_labels, unigram):
    n, vocab = len(group), len(unigram)
    return {"uniform": score(kind, oracle, uniform_pred(n, vocab), group, group_labels),
            "unigram": score(kind, oracle, np.tile(unigram, (n, 1)), group, group_labels)}


def metric_of(scores, metric, group):
    return scores[metric] if group is None else scores["groups"][str(group)][metric]


def evaluate_rule(rule, scores):
    """scores: {entrant name: metrics or None}. Returns {"met": bool | None, "clauses": [...], "missing": [...]}."""
    clauses, missing = [], []
    for cond in rule["conditions"]:
        names = [cond["entrant"]] + ([cond["minus"]] if cond.get("minus") else [])
        absent = [name for name in names if scores.get(name) is None]
        if absent:
            missing += absent
            clauses.append(dict(cond, lhs=None, holds=None))
            continue
        lhs = metric_of(scores[cond["entrant"]], cond["metric"], cond["group"])
        if cond.get("minus"):
            lhs -= metric_of(scores[cond["minus"]], cond["metric"], cond["group"])
        clauses.append(dict(cond, lhs=float(lhs), holds=bool(OPS[cond["op"]](lhs, cond["value"]))))
    met = None if missing else all(c["holds"] for c in clauses)
    return {"met": met, "clauses": clauses, "missing": sorted(set(missing))}


def trained(candidate, refs):
    """Did the candidate learn the task: does it beat the better of the uniform and unigram references?

    The unigram reference can be worse than uniform (a frequent separator token takes most of its mass), so
    comparing with it alone lets a predictor that knows nothing pass."""
    best_kl = min(refs["uniform"]["kl"], refs["unigram"]["kl"])
    best_acc = max(refs["uniform"]["acc"], refs["unigram"]["acc"])
    kl_gain = best_kl - candidate["kl"]
    acc_gain = candidate["acc"] - best_acc
    return {"trained": bool(kl_gain >= TRAINED_KL_GAIN or acc_gain >= TRAINED_ACC_GAIN),
            "kl_gain_over_best_reference": float(kl_gain), "acc_gain_over_best_reference": float(acc_gain),
            "kl_gain_needed": TRAINED_KL_GAIN, "acc_gain_needed": TRAINED_ACC_GAIN}


def task_blocked(task, describe):
    """Why no rule of this task can be decided for this candidate, or None.

    task: the evaluator's entry for the task (falsifier.eval_entry): max_train_len and protocol_params.
    describe: the candidate's describe() as reported by the falsifier job (train_max_len, hyper).
    Two things must hold. The candidate trains on every training sequence in full, otherwise the answer at the
    end of a longer sequence is never trained. Each model-side protocol parameter (d, M) equals the value the
    card or the task states, otherwise the rule tests a different mechanism than the card's."""
    if not isinstance(describe, dict):
        return "the candidate reported no readable describe(), so its training length and sizes are unknown"
    limit = describe.get("train_max_len")
    if not isinstance(limit, int) or isinstance(limit, bool):
        return ("the candidate's describe() has no integer train_max_len, so the length it trains on is unknown")
    if task["max_train_len"] > limit:
        return ("the task has training sequences of %d tokens but the candidate trains on at most %d tokens, so the "
                "answers at the end of longer sequences are never trained" % (task["max_train_len"], limit))
    hyper = describe.get("hyper")
    for param in task.get("protocol_params", []):
        if not param.get("model_side"):
            continue
        name, want = param["name"], param["task"]
        got = hyper.get(name) if isinstance(hyper, dict) else None
        if not isinstance(got, int) or isinstance(got, bool):
            return ("the protocol needs %s = %g and the candidate's describe()['hyper'] does not report an integer %s"
                    % (name, want, name))
        if got != want:
            return "the protocol needs %s = %g and the candidate was built with %s = %d" % (name, want, name, got)
    return None


def decide(rule, scores, rules=None, blocked=None):
    """scores holds the entrants and the two references. rules is the task's rule list (needed to resolve a
    control_rule); blocked is the task_blocked reason, if any. Returns the decision record."""
    candidate = scores.get("candidate")
    evaluation = evaluate_rule(rule, scores)
    if candidate is None:
        return {"outcome": "inconclusive", "reason": "the candidate produced no valid predictions on this task",
                "rule_met": evaluation["met"], **evaluation}
    if blocked:
        return {"outcome": "inconclusive", "reason": blocked, "rule_met": evaluation["met"], **evaluation}
    training = trained(candidate, scores)
    if not training["trained"]:
        return {"outcome": "inconclusive", "training": training, "rule_met": evaluation["met"], **evaluation,
                "reason": "the candidate did not train on this task: it does not beat the better of the uniform and "
                          "unigram references by %.2f nats of KL or %.2f accuracy" % (TRAINED_KL_GAIN, TRAINED_ACC_GAIN)}
    control = rule.get("control_rule")
    control_record = None
    if control is not None:
        if rules is None or not 0 <= control < len(rules):
            raise ValueError("control_rule %r does not name a rule of the task" % (control,))
        control_eval = evaluate_rule(rules[control], scores)
        control_record = {"rule_index": control, "met": control_eval["met"], "clauses": control_eval["clauses"]}
        if not control_eval["met"]:
            return {"outcome": "inconclusive", "training": training, "control": control_record,
                    "rule_met": evaluation["met"], **evaluation,
                    "reason": "the positive control (rule %d) was not met, so a low score here does not separate a "
                              "capacity limit from a model that cannot do the task" % control}
    if evaluation["met"] is None:
        return {"outcome": "inconclusive", "training": training, "rule_met": None, **evaluation,
                "reason": "no valid predictions from %s, named in the rule" % ", ".join(evaluation["missing"])}
    outcome = rule["if_met"] if evaluation["met"] else OPPOSITE[rule["if_met"]]
    record = {"outcome": outcome, "training": training, "rule_met": evaluation["met"], **evaluation,
              "reason": "rule %s, so the claim is %s" % ("met" if evaluation["met"] else "not met", outcome)}
    if control_record is not None:
        record["control"] = control_record
    return record


def better_than(cond):
    """True when the condition asks the candidate to do at least this well (acc above, kl below a value)."""
    return (cond["metric"] == "acc") == (cond["op"] in (">=", ">"))


def needs_control(rule):
    """True when the rule can produce "supported" from a low score and so must name a control_rule: a
    "supported" rule with a condition that a weak model meets (a limit), or a "refuted" rule whose conditions all
    ask for performance (when the observation is not made, the falsifier claim reads supported)."""
    if rule["if_met"] == "supported":
        return any(not better_than(c) for c in rule["conditions"])
    return all(better_than(c) for c in rule["conditions"])


def trivially_met(rule, refs):
    """Names of the trivial references that meet a rule supporting the card when put in the candidate's place.

    Only rules whose conditions all name the candidate alone and all ask for performance above a bar are
    checked. A rule that predicts a limit (accuracy at most 0.6 at K = 256) is met by any weak predictor by
    design; such a rule must name a control_rule (needs_control), which decide() requires the candidate to meet
    first."""
    if rule["if_met"] != "supported":
        return []
    if any(c["entrant"] != "candidate" or c.get("minus") or not better_than(c) for c in rule["conditions"]):
        return []
    return [name for name in REFERENCES if evaluate_rule(rule, {"candidate": refs[name]})["met"]]


def analytic_reference_levels(kind, oracle, unigram):
    """Uniform and unigram scores computed directly from their definitions, without metrics.task_metrics."""
    vocab = len(unigram)
    if kind == "answer":
        top = np.flatnonzero(unigram >= unigram.max() - metrics.TIE)
        uni_kl = -np.log(np.maximum(unigram[oracle], metrics.FLOOR))
        uni_acc = np.isin(oracle, top).astype(float) / len(top)
        uniform = {"kl": math.log(vocab), "acc": 1.0 / vocab}
    else:
        safe = np.where(oracle > 0, oracle, 1.0)
        entropy = -(oracle * np.log(safe)).sum(1)
        best = oracle >= oracle.max(1, keepdims=True) - metrics.TIE
        uniform = {"kl": float(np.mean(math.log(vocab) - entropy)), "acc": float(np.mean(best.sum(1) / vocab))}
        top = unigram >= unigram.max() - metrics.TIE
        uni_kl = -entropy - (oracle * np.log(np.maximum(unigram, metrics.FLOOR))).sum(1)
        uni_acc = (best & top).sum(1) / top.sum()
    return {"uniform": uniform, "unigram": {"kl": float(np.mean(uni_kl)), "acc": float(np.mean(uni_acc))}}


def evaluate(spec, in_dir, verify_outputs):
    """The `falsifiers` block of bench_result.json. in_dir holds falsifier_oracle/<task>.npz and
    falsifier_entrants/<entrant>/<task>/{receipt.json, out/}. verify_outputs is evaluator.verify_outputs."""
    import json

    block = {}
    for task_id, task in spec["falsifiers"].items():
        with np.load(in_dir / "falsifier_oracle" / ("%s.npz" % task_id), allow_pickle=False) as data:
            oracle, group, unigram = data["oracle"], data["group"], data["unigram"]
        if len(group) != task["n_prefixes"]:
            raise ValueError("falsifier oracle for %s does not match the spec" % task_id)
        labels = task["group_labels"]
        scores, entrants = {}, {}
        candidate_describe = None
        for name in spec["entrants"]:
            entrant_dir = in_dir / "falsifier_entrants" / name / task_id
            receipt = json.loads((entrant_dir / "receipt.json").read_text())
            entry = {"job_id": receipt["job_id"], "job_status": receipt["status"],
                     "job_detail": receipt.get("detail", "")}
            scores[name] = None
            if receipt["status"] != "ok":
                entry["status"] = "job_failed"
            else:
                verify_outputs(entrant_dir / "out", receipt["outputs"])
                try:
                    pred = np.load(entrant_dir / "out" / "pred.npy", allow_pickle=False)
                    problems = metrics.validate(pred, len(group), task["vocab_size"])
                except (OSError, ValueError) as error:
                    problems = ["cannot load: %s" % error]
                if problems:
                    entry.update(status="invalid", problems=problems)
                else:
                    scores[name] = score(task["oracle_kind"], oracle, pred.astype(np.float64), group, labels)
                    entry.update(status="ok", metrics=scores[name])
                    if name == "candidate":
                        try:   # written by the candidate's job, so unreadable content is a result, not a crash
                            candidate_describe = json.loads((entrant_dir / "out" / "describe.json").read_text())
                        except (OSError, ValueError):
                            candidate_describe = None
                        entry["describe"] = candidate_describe
            entrants[name] = entry
        refs = references(task["oracle_kind"], oracle, group, labels, unigram)
        scores.update(refs)
        block[task_id] = {key: task[key] for key in ("title", "protocol", "deviations", "vocab_size", "oracle_kind",
                                                     "group_labels", "n_prefixes", "n_train", "train_tokens",
                                                     "max_train_len", "protocol_params")}
        blocked = None if scores.get("candidate") is None else task_blocked(task, candidate_describe)
        decisions = [{"rule_index": index, "claims": rule["claims"], "quote": rule["quote"], "rule": rule,
                      **decide(rule, scores, task["rules"], blocked)} for index, rule in enumerate(task["rules"])]
        block[task_id].update(entrants=entrants, references=refs, decisions=decisions)
    return block
