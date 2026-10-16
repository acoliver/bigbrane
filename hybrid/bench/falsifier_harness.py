"""Falsifier task validation and data generation job. Standard library, NumPy and staged bench code only.

Runs in the executor (Podman: no network, read-only root, one CPU) as `python3 /in/main.py`, working
directory /out. /in holds:
    main.py        this file
    task.py        the designer's program; it defines generate(seed) -> dict (see FIELDS)
    harness.json   {"seed": int, "caps": {...}, "rules": [{...}], "child_timeout_s": float}
    code/hybrid/bench/{metrics,falsifier_eval,falsifier_reference}.py   the bench's own scoring code

Steps:
1. generate(seed) runs in two child processes with different PYTHONHASHSEED values. In each child an audit
   hook is active while task.py is imported and generate runs; it records and blocks file writes and
   deletions, reads outside the Python installation and /in, network use, subprocesses and ctypes ("escape"
   violations).
2. Each child checks the structure and the size caps and prints the sha256 of the sorted-key compact JSON of the
   output. Equal digests mean the generator is deterministic on replay.
3. The first child also checks that the oracle scores itself perfectly (KL 0, accuracy 1), that every
   next token agrees with the oracle (the answer itself, or a token of nonzero oracle probability), that
   the uniform and unigram predictors score at the levels computed from their definitions, and that the
   rules' group labels exist and a trivial predictor does not meet a rule that would support the card.
   Gates on the task (check_structure, check_scoring): every test group is covered by training (a training
   sequence at least as long as the longest test prefix plus its answer, and none longer than the candidate's
   train_max_len); the trivial references fail the training check; generic leak controls (recency, n-gram)
   do not meet a candidate condition; a task flagged solvable_by "lookup" is solved by lookup in every group;
   a task that declares a reference mechanism (hybrid/bench/falsifier_reference.py) has its rules checked
   against the reference's accuracy (reference_problems).
   It writes train.json, prefixes.json, oracle.npz (oracle, group, next_tokens) and meta.json to /out.
4. The parent prints the report as the last JSON line: {"ok": bool, "problems": [...], ...}. It always
   exits 0; a failed check is a report, not a crash.
"""
import hashlib
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD_ENV = "BB_FALSIFIER_CHILD"
FIELDS = ("vocab_size", "train", "prefixes", "next_tokens", "oracle_kind", "oracle", "group", "group_labels")
ESCAPE_EVENTS = ("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty",
                 "os.spawn", "os.kill", "os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace",
                 "os.mkdir", "os.chmod", "os.chown", "os.truncate", "os.symlink", "os.link", "shutil.", "ctypes.",
                 "urllib.", "http.", "ftplib.", "smtplib.", "webbrowser.", "pty.", "fcntl.")
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _load_cfg():
    with open(os.path.join(HERE, "harness.json")) as handle:
        return json.load(handle)


def _audit(allowed_read, violations, active):
    def hook(event, args):
        if not active[0]:
            return
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int):
                return
            writing = (mode is not None and any(ch in str(mode) for ch in "wax+")) or (
                mode is None and isinstance(flags, int) and flags & WRITE_FLAGS)
            if writing:
                violation = "write: %s" % path
            else:
                real = os.path.realpath(os.fsdecode(path))
                if any(real == root or real.startswith(root + os.sep) for root in allowed_read):
                    return
                violation = "read outside the Python installation and /in: %s" % real
        elif event.startswith(ESCAPE_EVENTS):
            violation = "%s %s" % (event, str(args)[:200])
        else:
            return
        violations.append(violation)
        raise PermissionError("blocked by the falsifier harness: " + violation)
    return hook


def _stable_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _int_list(value, where, problems):
    if not isinstance(value, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        problems.append("%s must be a list of ints" % where)
        return False
    return True


def coverage_problems(train, prefixes, group, labels, train_max_len):
    """Every test group is covered by training, and training fits what the candidate trains on.

    A group whose longest test prefix (plus its answer) is longer than every training sequence is never seen
    during training, so no model learns the task at that length. A training sequence longer than the candidate's
    guaranteed training length loses its end, which is where the answer is."""
    problems = []
    longest_train = max(len(s) for s in train)
    group = list(group)
    for g, label in enumerate(labels):
        longest = max(len(p) for p, gg in zip(prefixes, group) if gg == g)
        if longest + 1 > longest_train:
            problems.append("group %s has test prefixes of %d tokens but the longest training sequence has %d, so no "
                            "model trains at this length; include training episodes from every group"
                            % (label, longest, longest_train))
    if train_max_len is not None and longest_train > train_max_len:
        over = sum(1 for s in train if len(s) > train_max_len)
        problems.append("%d training sequences are longer than the %d tokens the candidate trains on (longest %d), "
                        "so the answer at their end is never trained; shorten the episodes to at most %d tokens"
                        % (over, train_max_len, longest_train, train_max_len))
    return problems


def check_structure(data, caps, train_max_len=None):
    """Returns (normalized data or None, problems). train_max_len: the candidate's guaranteed training length."""
    problems = []
    if not isinstance(data, dict):
        return None, ["generate(seed) must return a dict, got %s" % type(data).__name__]
    missing = [k for k in FIELDS if k not in data]
    extra = sorted(set(data) - set(FIELDS))
    if missing or extra:
        return None, ["generate(seed) keys: missing %s, unexpected %s" % (missing, extra)]
    import numpy as np

    out = {}
    vocab = data["vocab_size"]
    if not isinstance(vocab, int) or isinstance(vocab, bool) or not 2 <= vocab <= caps["max_vocab"]:
        return None, ["vocab_size must be an int in [2, %d], got %r" % (caps["max_vocab"], vocab)]
    out["vocab_size"] = vocab
    for key, n_cap, tok_cap in (("train", caps["max_train_sequences"], caps["max_train_tokens"]),
                                ("prefixes", caps["max_prefixes"], caps["max_prefix_tokens"])):
        seqs = data[key]
        if isinstance(seqs, np.ndarray):
            seqs = seqs.tolist()
        if not isinstance(seqs, list) or not seqs:
            problems.append("%s must be a non-empty list of sequences" % key)
            continue
        seqs = [s.tolist() if isinstance(s, np.ndarray) else s for s in seqs]
        if not all(_int_list(s, "%s[i]" % key, problems) and s for s in seqs):
            problems.append("%s holds an empty or non-integer sequence" % key)
            continue
        total = sum(len(s) for s in seqs)
        longest = max(len(s) for s in seqs)
        if len(seqs) > n_cap:
            problems.append("%s has %d sequences, cap %d" % (key, len(seqs), n_cap))
        if total > tok_cap:
            problems.append("%s has %d tokens, cap %d" % (key, total, tok_cap))
        if longest > caps["max_sequence_length"]:
            problems.append("%s has a sequence of %d tokens, cap %d" % (key, longest, caps["max_sequence_length"]))
        if min(min(s) for s in seqs) < 0 or max(max(s) for s in seqs) >= vocab:
            problems.append("%s has tokens outside [0, vocab_size)" % key)
        out[key] = seqs
    if problems:
        return None, problems
    n = len(out["prefixes"])
    if n < caps["min_prefixes"]:
        problems.append("only %d test prefixes, need at least %d" % (n, caps["min_prefixes"]))
    labels = data["group_labels"]
    if (not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels)
            or len(set(labels)) != len(labels) or len(labels) > caps["max_groups"]):
        problems.append("group_labels must be 1 to %d distinct non-empty strings" % caps["max_groups"])
        return None, problems
    out["group_labels"] = labels
    group = np.asarray(data["group"])
    nxt = np.asarray(data["next_tokens"])
    if group.shape != (n,) or group.dtype.kind not in "iu" or group.min() < 0 or group.max() >= len(labels):
        problems.append("group must hold one index into group_labels per prefix")
    elif min(np.bincount(group, minlength=len(labels))) < caps["min_group_size"]:
        problems.append("every group needs at least %d prefixes: %s" % (
            caps["min_group_size"], np.bincount(group, minlength=len(labels)).tolist()))
    else:
        problems += coverage_problems(out["train"], out["prefixes"], group, labels, train_max_len)
    if nxt.shape != (n,) or nxt.dtype.kind not in "iu" or nxt.min() < 0 or nxt.max() >= vocab:
        problems.append("next_tokens must hold one token in [0, vocab_size) per prefix")
    kind = data["oracle_kind"]
    oracle = np.asarray(data["oracle"])
    if kind == "answer":
        if oracle.shape != (n,) or oracle.dtype.kind not in "iu" or oracle.min() < 0 or oracle.max() >= vocab:
            problems.append("an answer oracle must hold one token in [0, vocab_size) per prefix")
    elif kind == "dist":
        if oracle.shape != (n, vocab) or oracle.dtype.kind != "f":
            problems.append("a dist oracle must be a float array of shape (n_prefixes, vocab_size)")
        elif oracle.size > caps["max_oracle_floats"]:
            problems.append("dist oracle has %d floats, cap %d" % (oracle.size, caps["max_oracle_floats"]))
        elif (not np.isfinite(oracle).all() or (oracle < 0).any()
              or np.max(np.abs(oracle.sum(1) - 1)) > 1e-9):
            problems.append("dist oracle rows must be finite, non-negative and sum to 1 within 1e-9")
    else:
        problems.append("oracle_kind must be 'answer' or 'dist', got %r" % (kind,))
    if problems:
        return None, problems
    out.update(oracle_kind=kind, oracle=oracle.tolist(), group=group.astype(int).tolist(),
               next_tokens=nxt.astype(int).tolist())
    return out, []


ALL = "(all)"
RECENCY_MAX = 8
NGRAM_ORDERS = (1, 2)
LOOKUP_MIN_ACC = 0.99


def control_predictions(train, prefixes):
    """Next-token guesses (-1: no guess) of generic predictors that know nothing about the task's mechanism:
    recency_j (the j-th last prefix token), ngram_n (the most frequent continuation of the last n tokens in the
    training sequences) and lookup (the token that followed the most recent earlier occurrence of the last
    prefix token)."""
    preds = {"recency_%d" % j: [p[-j] if len(p) >= j else -1 for p in prefixes] for j in range(1, RECENCY_MAX + 1)}
    for n in NGRAM_ORDERS:
        table = {}
        for seq in train:
            for i in range(n, len(seq)):
                counts = table.setdefault(tuple(seq[i - n:i]), {})
                counts[seq[i]] = counts.get(seq[i], 0) + 1
        best = {key: min(counts, key=lambda tok: (-counts[tok], tok)) for key, counts in table.items()}
        preds["ngram_%d" % n] = [best.get(tuple(p[-n:]), -1) if len(p) >= n else -1 for p in prefixes]
    lookup = []
    for p in prefixes:
        last = {}
        for i in range(len(p) - 1):
            last[p[i]] = i
        lookup.append(p[last[p[-1]] + 1] if p[-1] in last else -1)
    preds["lookup"] = lookup
    return preds


def hits(data, guess):
    """Per prefix: 1.0 when the guessed token is the answer (a token of top oracle probability for a dist
    oracle), else 0.0. A guess of -1 is never right."""
    import numpy as np

    from hybrid.bench import metrics

    guess = np.asarray(guess, dtype=np.int64)
    if data["oracle_kind"] == "answer":
        return (guess == np.asarray(data["oracle"], dtype=np.int64)).astype(float)
    oracle = np.asarray(data["oracle"], dtype=np.float64)
    row = np.arange(len(guess))
    return ((guess >= 0) & (oracle[row, np.maximum(guess, 0)] >= oracle.max(1) - metrics.TIE)).astype(float)


def accuracies_by_group(data, right):
    """{ALL: mean, group label: mean} of a per-prefix right/wrong (or fractional) array."""
    import numpy as np

    group = np.asarray(data["group"], dtype=np.int64)
    out = {ALL: float(right.mean())}
    for g, label in enumerate(data["group_labels"]):
        out[label] = float(right[group == g].mean())
    return out


def control_accuracies(data):
    """{control name: {group label or ALL: accuracy}} on the test prefixes."""
    return {name: accuracies_by_group(data, hits(data, guess))
            for name, guess in control_predictions(data["train"], data["prefixes"]).items()}


def control_problems(rules, controls, solvable_by, labels):
    """Leak and solvability gates from the controls' accuracies."""
    from hybrid.bench import falsifier_eval as fe

    problems = []
    for index, rule in enumerate(rules):
        flagged = set()
        for cond in rule["conditions"]:
            if cond["entrant"] != "candidate" or cond.get("minus") or cond["metric"] != "acc":
                continue
            if not fe.better_than(cond):
                continue
            key = ALL if cond["group"] is None else cond["group"]
            if key not in labels and key != ALL:
                continue   # an unknown group is already reported by check_scoring
            for name, accs in controls.items():
                if name != "lookup" and name not in flagged and fe.OPS[cond["op"]](accs[key], cond["value"]):
                    flagged.add(name)
                    problems.append("rule %d is met by the %s control, so the answer is predictable without the "
                                    "mechanism" % (index, name))
    if solvable_by == "lookup":
        weak = {label: controls["lookup"][label] for label in labels if controls["lookup"][label] < LOOKUP_MIN_ACC}
        if weak:
            problems.append("the task is declared solvable_by lookup, but lookup scores %s (need %.2f in every group)"
                            % (weak, LOOKUP_MIN_ACC))
    return problems


NEGATED_OP = {"<=": ">", "<": ">=", ">=": "<", ">": "<="}


def reference_accuracies(data, reference):
    """{variant: {group label or ALL: accuracy}} of the declared reference mechanism, averaged over its lifts."""
    import numpy as np

    from hybrid.bench import falsifier_reference as fr

    guesses = fr.pair_memory_guesses(data["prefixes"], data["vocab_size"], reference["d"], reference["min_token"])
    return {variant: accuracies_by_group(data, np.mean([hits(data, g) for g in per_seed], axis=0))
            for variant, per_seed in guesses.items()}


def reference_problems(rules, accuracies):
    """Gates from the reference mechanism's accuracy, "rls" (the card's own operator). Only rules whose
    conditions all ask the candidate alone for an accuracy can be scored this way; the others are skipped.

    - a rule with if_met "refuted" must not be met by the reference (the card's mechanism would fire it);
    - a "supported" rule that is a limit must be met by the reference (otherwise it refutes a correct build);
    - a rule named as another rule's control_rule must be met by the reference (it can be passed in principle)."""
    from hybrid.bench import falsifier_eval as fe

    block = {"acc": accuracies[ALL],
             "groups": {label: {"acc": acc} for label, acc in accuracies.items() if label != ALL}}

    def evaluated(rule):
        if any(c["entrant"] != "candidate" or c.get("minus") or c["metric"] != "acc" for c in rule["conditions"]):
            return None
        return fe.evaluate_rule(rule, {"candidate": block})

    problems = []
    controls = {rule["control_rule"] for rule in rules if rule.get("control_rule") is not None}
    for index, rule in enumerate(rules):
        result = evaluated(rule)
        if result is None:
            continue
        if rule["if_met"] == "refuted" and result["met"]:
            problems.append("the card's own d x d operator meets falsifier rule %d, so observing it cannot mean "
                            "something other than the operator stores the bindings" % index)
        if rule["if_met"] == "supported" and fe.needs_control(rule) and not result["met"]:
            failed = next(c for c in result["clauses"] if not c["holds"])
            problems.append("the card's own mechanism violates limit rule %d (reference %s %.2f %s %.2f), so the "
                            "rule would refute a correct implementation"
                            % (index, failed["metric"], failed["lhs"], NEGATED_OP[failed["op"]], failed["value"]))
        if index in controls and not result["met"]:
            problems.append("the card's own mechanism does not meet control rule %d, so no build of the card can "
                            "pass it" % index)
    return problems


def check_scoring(data, rules, solvable_by=None, reference=None):
    """Oracle self-score, next-token consistency, analytic trivial levels, the training check on the trivial
    references, rule groups, vacuity, the leak and solvability controls, and (when the task declares a
    reference) the reference-mechanism capacity control."""
    import numpy as np

    sys.path.insert(0, os.path.join(HERE, "code"))
    from hybrid.bench import falsifier_eval as fe

    problems, checks = [], {}
    vocab, kind, labels = data["vocab_size"], data["oracle_kind"], data["group_labels"]
    group = np.asarray(data["group"], dtype=np.int64)
    nxt = np.asarray(data["next_tokens"], dtype=np.int64)
    if kind == "answer":
        oracle = np.asarray(data["oracle"], dtype=np.int64)
        perfect = np.eye(vocab)[oracle]
        consistent = oracle == nxt
    else:
        oracle = np.asarray(data["oracle"], dtype=np.float64)
        perfect = oracle
        consistent = oracle[np.arange(len(nxt)), nxt] > 0
    self_score = fe.score(kind, oracle, perfect, group, labels)
    checks["oracle_self_score"] = {"kl": self_score["kl"], "acc": self_score["acc"]}
    worst_kl = max(g["kl"] for g in self_score["groups"].values())
    worst_acc = min(g["acc"] for g in self_score["groups"].values())
    if worst_kl > 1e-9 or worst_acc < 1 - 1e-9:
        problems.append("the oracle does not score itself perfectly: KL %.3g, accuracy %.6f" % (worst_kl, worst_acc))
    checks["next_tokens_consistent"] = int(consistent.sum())
    if not consistent.all():
        bad = np.flatnonzero(~consistent)[:5].tolist()
        problems.append("%d of %d next tokens disagree with the oracle (%s at prefixes %s)" % (
            int((~consistent).sum()), len(nxt), "not the answer" if kind == "answer" else "zero oracle probability",
            bad))
    unigram = fe.unigram_dist(data["train"], vocab)
    refs = fe.references(kind, oracle, group, labels, unigram)
    analytic = fe.analytic_reference_levels(kind, oracle, unigram)
    checks["references"] = {name: {"kl": refs[name]["kl"], "acc": refs[name]["acc"]} for name in refs}
    checks["analytic"] = analytic
    for name in fe.REFERENCES:
        for metric in ("kl", "acc"):
            got, want = refs[name][metric], analytic[name][metric]
            if not math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12):
                problems.append("%s %s scores %.12g, analytic level %.12g" % (name, metric, got, want))
    for name in fe.REFERENCES:
        if fe.trained(refs[name], refs)["trained"]:
            problems.append("a trivial reference passes the training check: %s" % name)
    for index, rule in enumerate(rules):
        for cond in rule["conditions"]:
            if cond["group"] is not None and cond["group"] not in labels:
                problems.append("rule %d names group %r, which is not in group_labels %s"
                                % (index, cond["group"], labels))
    if not problems:
        for index, rule in enumerate(rules):
            met_by = fe.trivially_met(rule, refs)
            if met_by:
                problems.append("rule %d supports the card when the candidate is replaced by %s, so it tests "
                                "nothing" % (index, " or ".join(met_by)))
    controls = control_accuracies(data)
    checks["controls"] = controls
    problems += control_problems(rules, controls, solvable_by, labels)
    if reference is not None:
        checks["reference"] = dict(reference, accuracy=reference_accuracies(data, reference))
        problems += reference_problems(rules, checks["reference"]["accuracy"]["rls"])
    return problems, checks


def write_outputs(data):
    import numpy as np

    out = os.getcwd()
    with open(os.path.join(out, "train.json"), "w") as handle:
        json.dump({"sequences": data["train"]}, handle)
    with open(os.path.join(out, "prefixes.json"), "w") as handle:
        json.dump({"prefixes": data["prefixes"]}, handle)
    oracle = np.asarray(data["oracle"], dtype=np.int64 if data["oracle_kind"] == "answer" else np.float64)
    np.savez(os.path.join(out, "oracle.npz"), oracle=oracle, group=np.asarray(data["group"], dtype=np.int64),
             next_tokens=np.asarray(data["next_tokens"], dtype=np.int64))
    with open(os.path.join(out, "meta.json"), "w") as handle:
        json.dump({key: data[key] for key in ("vocab_size", "oracle_kind", "group_labels")}, handle)


def child(primary):
    import sysconfig

    import numpy  # noqa: F401  (loaded before the hook so its own file reads are not audited)

    sys.dont_write_bytecode = True
    cfg = _load_cfg()
    paths = sysconfig.get_paths()
    allowed = tuple(sorted({os.path.realpath(paths[k]) for k in ("stdlib", "platstdlib", "purelib", "platlib")}
                           | {os.path.realpath(HERE)}))
    violations, active = [], [True]
    sys.addaudithook(_audit(allowed, violations, active))
    report = {"primary": primary, "violations": violations, "problems": []}
    try:
        sys.path.insert(0, HERE)
        import task
        data = task.generate(int(cfg["seed"]))
    except BaseException as error:  # the designer's code is under test; its failure is the result
        active[0] = False
        report["problems"].append("generate(seed) raised %s: %s" % (type(error).__name__, str(error)[:500]))
        print(json.dumps(report))
        return
    active[0] = False
    data, problems = check_structure(data, cfg["caps"], cfg.get("train_max_len"))
    report["problems"] += problems
    if data is not None:
        report["digest"] = hashlib.sha256(_stable_json(data)).hexdigest()
        if primary:
            problems, checks = check_scoring(data, cfg["rules"], cfg.get("solvable_by"), cfg.get("reference"))
            report["problems"] += problems
            report["checks"] = checks
            report["sizes"] = {"vocab_size": data["vocab_size"], "n_train": len(data["train"]),
                               "train_tokens": sum(len(s) for s in data["train"]),
                               "n_prefixes": len(data["prefixes"]),
                               "prefix_tokens": sum(len(s) for s in data["prefixes"])}
            if not report["problems"] and not violations:
                write_outputs(data)
    print(json.dumps(report))


def run_child(primary, hash_seed, timeout):
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed), PYTHONDONTWRITEBYTECODE="1")
    env[CHILD_ENV] = "primary" if primary else "replay"
    try:
        done = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"problems": ["generate(seed) did not finish within %.0f s" % timeout], "violations": []}
    for line in reversed(done.stdout.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    return {"problems": ["the generator child exited %d without a report; stderr: %s"
                         % (done.returncode, done.stderr[-1500:])], "violations": []}


def main():
    cfg = _load_cfg()
    timeout = float(cfg["child_timeout_s"])
    first = run_child(True, 1, timeout)
    second = run_child(False, 2, timeout)
    problems = list(first["problems"])
    violations = sorted(set(first["violations"]) | set(second["violations"]))
    if violations:
        problems.append("escape: the generator touched files, the network or processes: %s" % violations[:10])
    deterministic = "digest" in first and first.get("digest") == second.get("digest")
    if "digest" in first and not deterministic:
        problems.append("the generator is not deterministic on replay (sha256 %s then %s)"
                        % (first.get("digest"), second.get("digest") or "none: " + "; ".join(second["problems"])))
    report = {"ok": not problems, "problems": problems, "deterministic": deterministic,
              "digest": first.get("digest"), "violations": violations, "checks": first.get("checks"),
              "sizes": first.get("sizes")}
    print(json.dumps(report))


if __name__ == "__main__":
    mode = os.environ.get(CHILD_ENV)
    if mode:
        child(mode == "primary")
    else:
        main()
