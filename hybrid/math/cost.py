"""Cost verifier (DISCOVERY-PIPELINE-PLAN.md section 4.4, P1): counts the operations a candidate's update and
predict code actually performs and compares them with the card's claimed cost formula.

Candidate contract. The candidate is Python source with no imports that defines

    def init(sizes, ops):                 -> (params, state)
    def update(params, state, token, ops): -> state
    def predict(params, state, ops):       -> anything

``ops`` is an ``OpCounter``. Its ``ops.np`` namespace holds the only array functions the candidate may use; every
array it returns is a ``Counted`` wrapper whose arithmetic is counted. A run feeds n tokens (deterministic,
uniform over ``sizes["V"]``, default 16), calling update then predict for each, so the counted total is the
sequence cost of update plus predict over n tokens. init is counted separately.

Counting rules (one unit each):

* ``mac``: a matrix product of shapes (..., k) and (k, ...) counts k per output element (``matmul``, ``dot``, ``@``);
* ``arith``: one per output element of an elementwise ufunc (add, subtract, multiply, divide, power, negative,
  absolute, ...), and one per binary step of a sum reduction (a sum of k elements counts k - 1);
* ``compare``: one per output element of greater, less, equal, maximum, minimum, where and the like, and one per
  step of a max, min, argmax or argmin reduction (k - 1 for k elements);
* ``transcendental``: one per output element of exp, log, tanh, sqrt, sin, cos and the like, and of
  ``ops.np.sigmoid``, the logistic sigma(x) = 1/(1+exp(-x)) as one unit like tanh.

Creating, indexing, slicing, copying and concatenating arrays count nothing; they show up in the state size.
Python scalar arithmetic is not counted; instead every executed line of the candidate is counted with
``sys.settrace``, and the fitted slope of executed lines in n may not exceed the claimed exponent, so a Python
loop that does uncounted work still shows its growth.

The verifier runs the candidate at every n in ``ns`` crossed with every listed value of the other size
parameters, and requires (plan section 4.4):

1. counted operations equal the formula at every point, or differ by at most the declared remainder;
2. the least-squares slope of log(ops) against log(n) is within ``tolerance`` (default 0.05) of the formula's
   slope over the same points, for every setting of the other sizes, and likewise the two-point slope in every
   other size parameter; with a declared remainder the formula's slope widens to the band of slopes that counts
   within the remainder can have on those points, so admitted lower-order work does not fail a finite-grid slope;
3. the formula's asymptotic exponent in n equals the claimed exponent, when the card states one;
4. the slope of executed lines in n is at most the formula's slope plus ``tolerance``;
5. the peak state size in words equals the state formula at every point (or within its declared remainder);
6. the parameters do not grow during the sequence.

A remainder is part of the frozen claim and is checked before any run (``remainder_problems``): it must be
nonnegative and below its formula at every grid point, and of strictly lower order than the formula in every size it
declares (``ops_remainder_lower_in``, ``state_remainder_lower_in``), so it can admit lower-order implementation detail
but never missing leading work. In the 2026-10-13 smoke card k2's MAC-counted dots fell 8nd^2 short of its
scalar-dot formula; no lower-order remainder can absorb that (results/phase4-20261013T073748Z/k2-smoke-diag).

Precision policy (FIX-20, decision PR). ``OpCounter(precision_bits)`` evaluates the candidate at the card's declared
precision (resources.architecture.precision_bits, frozen as the profile's architecture contract): every float array
it makes (zeros, ones, full, eye, arange, array, normal, sigmoid) has that dtype, an integer or bool array that meets
a float operand is converted to it first, and a result at another float dtype raises ``PrecisionError``. Parameters,
state and every intermediate are therefore binary32 under a 32-bit policy and binary64 under a 64-bit one. In K2
run 2 of calibration 20261016T051933Z the counted reference ran in float64 while the frozen equations rounded state to
float32, and conformance check O2.a failed on the 6.34e-06 difference (k2-o2-diag/REPORT.md). Without a policy
(``precision_bits=None``, a record with no declared precision) NumPy's defaults apply as before.

Wall time is recorded and never decides. Self-contained (stdlib, NumPy, SymPy) so it runs in the tools image as
part of ``bbmath``.
"""
from __future__ import annotations

import ast
import builtins
import itertools
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

import numpy as np
import sympy as sp
from numpy.lib.mixins import NDArrayOperatorsMixin

DEFAULT_NS = (64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_TOLERANCE = 0.05
DEFAULT_VOCAB = 16
CANDIDATE_FILE = "<candidate>"
DISCHARGED = "discharged"
VIOLATED = "violated"

COMPARE_UFUNCS = frozenset({"greater", "greater_equal", "less", "less_equal", "equal", "not_equal", "maximum",
                            "minimum", "fmax", "fmin", "sign", "logical_and", "logical_or", "logical_not",
                            "logical_xor", "isnan", "isinf", "isfinite"})
TRANSCENDENTAL_UFUNCS = frozenset({"exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "tanh", "sinh",
                                   "cosh", "sqrt", "cbrt", "sin", "cos", "tan", "arctan", "arcsin", "arccos",
                                   "arctanh", "logaddexp", "logaddexp2"})
UFUNC_NAMES = ("add", "subtract", "multiply", "divide", "true_divide", "floor_divide", "remainder", "power",
               "negative", "absolute", "square", "reciprocal", "floor", "ceil", "rint", "bitwise_and",
               "bitwise_or", "bitwise_xor", "left_shift", "right_shift") + tuple(sorted(COMPARE_UFUNCS)) \
    + tuple(sorted(TRANSCENDENTAL_UFUNCS))
SAFE_BUILTINS = {name: getattr(builtins, name)
                 for name in ("range", "len", "min", "max", "abs", "int", "float", "bool", "list", "tuple", "dict",
                              "enumerate", "zip", "sum", "reversed", "sorted", "any", "all", "round",
                              "ValueError", "AssertionError")}
# NumPy's C methods import helper modules on first use (ndarray.max, .mean, .std and, in the tools image, .sum reach
# numpy._core._methods through PyImport_Import), and that lookup reads __import__ from the builtins of the calling
# frame, which is the candidate's namespace: without it a learner's ``A.sum(axis=1, keepdims=True)`` raised
# KeyError('__import__') in the 2026-10-13 calibration (card k3). audit_source refuses the name and every import statement in candidate
# code, so only the interpreter's own lazy imports can reach it.
SAFE_BUILTINS["__import__"] = builtins.__import__
# isinstance tests a value the code already holds against a type it can already name (the allowed int, float, bool,
# list, tuple and dict, or an array type reached through np); it reads no attribute and reaches no object, so it opens
# nothing that getattr, type or object (all refused) would. The k3 learner of the 2026-10-13 calibration used it to
# walk the plain layout and example data and failed with NameError on attempt 2 of both learnability statements;
# that was legitimate data handling, so the allowlist was too narrow there.
SAFE_BUILTINS["isinstance"] = builtins.isinstance
FORBIDDEN_NAMES = frozenset({"eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
                             "getattr", "setattr", "delattr", "type", "object", "breakpoint", "input", "super"})


# Order in which failed checks are reported; the first becomes the counterexample.
CHECK_ORDER = ("claimed_exponent", "exponent_n", "lines_exponent_n", "ops_exact", "state_exact", "params_grew",
               "architecture")


class CandidateError(ValueError):
    """The candidate breaks the contract (imports, private attributes, missing functions, uncounted arrays)."""


class PrecisionError(CandidateError):
    """A computation that leaves the declared precision policy (FIX-20)."""


# FIX-20: the float dtype of each declared precision (architecture.PRECISIONS).
FLOAT_DTYPES = {16: np.dtype(np.float16), 32: np.dtype(np.float32), 64: np.dtype(np.float64)}


def policy_dtype(precision_bits: int | None) -> np.dtype | None:
    """The float dtype of a declared precision, None without one. An undeclared width is refused."""
    if precision_bits is None:
        return None
    if isinstance(precision_bits, bool) or precision_bits not in FLOAT_DTYPES:
        raise ValueError(f"precision_bits must be one of {sorted(FLOAT_DTYPES)}, not {precision_bits!r}")
    return FLOAT_DTYPES[precision_bits]


def _kind(ufunc) -> str:
    if ufunc.__name__ in COMPARE_UFUNCS:
        return "compare"
    if ufunc.__name__ in TRANSCENDENTAL_UFUNCS:
        return "transcendental"
    return "arith"


class Counted(NDArrayOperatorsMixin):
    """A NumPy array whose arithmetic is charged to an OpCounter. The raw array is not reachable from candidate
    code (attributes starting with an underscore are forbidden there)."""

    __slots__ = ("_a", "_ops")

    def __init__(self, array, ops: "OpCounter"):
        self._a = np.asarray(array)
        self._ops = ops

    # -- NumPy protocol -------------------------------------------------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self._ops._ufunc(ufunc, method, inputs, kwargs)

    def __array__(self, dtype=None, copy=None):
        raise CandidateError("a Counted array cannot be turned into a plain NumPy array; use ops.np")

    # -- shape and data movement (not counted) -------------------------------------------------------------------
    @property
    def shape(self):
        return self._a.shape

    @property
    def ndim(self):
        return self._a.ndim

    @property
    def size(self):
        return int(self._a.size)

    @property
    def T(self):
        return Counted(self._a.T, self._ops)

    def __len__(self):
        return len(self._a)

    def __iter__(self):
        return (Counted(row, self._ops) for row in self._a)

    def __getitem__(self, key):
        return Counted(self._a[self._ops._unwrap_key(key)], self._ops)

    def __setitem__(self, key, value):
        self._a[self._ops._unwrap_key(key)] = self._ops._unwrap(value)

    def reshape(self, *shape):
        return Counted(self._a.reshape(*shape), self._ops)

    def copy(self):
        return Counted(self._a.copy(), self._ops)

    def item(self):
        return self._a.item()

    def __int__(self):
        return int(self._a)

    def __index__(self):
        return int(self._a)

    def __float__(self):
        return float(self._a)

    def __bool__(self):
        return bool(self._a)

    def __repr__(self):
        return f"Counted({self._a!r})"

    # -- counted reductions --------------------------------------------------------------------------------------
    def sum(self, axis=None):
        return self._ops._ufunc(np.add, "reduce", (self,), {"axis": axis})

    def max(self, axis=None):
        return self._ops._ufunc(np.maximum, "reduce", (self,), {"axis": axis})

    def min(self, axis=None):
        return self._ops._ufunc(np.minimum, "reduce", (self,), {"axis": axis})

    def argmax(self, axis=None):
        return self._ops._arg(np.argmax, self, axis)

    def argmin(self, axis=None):
        return self._ops._arg(np.argmin, self, axis)


class OpCounter:
    """Counts operations by kind. ``ops.np`` is the candidate's array namespace; ``ops.charge`` is the explicit
    counter for work done outside NumPy."""

    KINDS = ("mac", "arith", "compare", "transcendental")

    def __init__(self, precision_bits: int | None = None):
        self.counts: Counter = Counter()
        # FIX-20: the declared precision policy; None leaves NumPy's defaults (no declared precision).
        self.precision_bits = precision_bits
        self.dtype = policy_dtype(precision_bits)
        self.np = self._namespace()

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    def charge(self, kind: str, amount: int = 1) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown op kind {kind!r}; expected one of {self.KINDS}")
        if amount < 0:
            raise ValueError("amount must be non-negative")
        self.counts[kind] += int(amount)

    def reset(self) -> Counter:
        counts, self.counts = self.counts, Counter()
        return counts

    # -- precision policy (FIX-20) -------------------------------------------------------------------------------
    def _policy_operands(self, raw: list) -> list:
        """Operands under the precision policy: a list of numbers becomes an array, a float one at the policy
        dtype, and an integer or bool array that meets a float operand is converted to the policy dtype (NumPy
        would otherwise promote int64 with float32 to float64). Python numbers stay weak scalars, which keep the
        array's dtype; a NumPy float scalar (never weak) is taken at the policy dtype. A float array at another
        precision is refused."""
        if self.dtype is None:
            return raw
        arrays = [np.asarray(x, dtype=self.dtype) if isinstance(x, np.floating)
                  else x if isinstance(x, (bool, int, float)) else np.asarray(x) for x in raw]
        arrays = [x.astype(self.dtype) if isinstance(x, np.ndarray) and x.dtype.kind == "f"
                  and not isinstance(r, np.ndarray) else x for x, r in zip(arrays, raw)]
        for x in arrays:
            if isinstance(x, np.ndarray) and x.dtype.kind == "f" and x.dtype != self.dtype:
                raise PrecisionError(f"a {x.dtype} operand under the declared {self.precision_bits}-bit precision")
        floating = any(isinstance(x, float) or (isinstance(x, np.ndarray) and x.dtype.kind == "f") for x in arrays)
        if not floating:
            return arrays
        return [x.astype(self.dtype) if isinstance(x, np.ndarray) and x.dtype.kind in "biu" else x for x in arrays]

    def _policy_result(self, result):
        """A result is held to the policy: a float result at another dtype raises (fail fast, never a cast)."""
        if self.dtype is not None:
            for value in (result if isinstance(result, tuple) else (result,)):
                kind = np.asarray(value).dtype
                if kind.kind == "f" and kind != self.dtype:
                    raise PrecisionError(f"an operation computed at {kind} under the declared "
                                         f"{self.precision_bits}-bit precision")
        return result

    def _float(self, maker, args, kwargs):
        """An allocation under the policy: a float result is made at the policy dtype; an explicit float dtype other
        than the policy's is refused."""
        if self.dtype is None:
            return maker(*args, **kwargs)
        if "dtype" in kwargs:
            asked = np.dtype(kwargs["dtype"])
            if asked.kind == "f" and asked != self.dtype:
                raise PrecisionError(f"dtype {asked} under the declared {self.precision_bits}-bit precision")
            return maker(*args, **kwargs)
        result = maker(*args, **kwargs)
        if np.asarray(result).dtype.kind == "f":
            result = maker(*args, **kwargs, dtype=self.dtype)
        return result

    # -- wrapping ------------------------------------------------------------------------------------------------
    def _unwrap(self, value):
        if isinstance(value, Counted):
            if value._ops is not self:
                raise CandidateError("a Counted array from another counter")
            return value._a
        if isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, (list, tuple)):
            return type(value)(self._unwrap(v) for v in value)
        raise CandidateError(f"uncounted operand of type {type(value).__name__}; create arrays with ops.np")

    def _unwrap_key(self, key):
        if isinstance(key, tuple):
            return tuple(self._unwrap_key(k) for k in key)
        if isinstance(key, Counted):
            return key._a
        return key

    def _wrap(self, value):
        if isinstance(value, tuple):
            return tuple(self._wrap(v) for v in value)
        return Counted(value, self)

    def _ufunc(self, ufunc, method, inputs, kwargs):
        if kwargs.get("out") is not None:
            raise CandidateError("out= is not supported")
        raw = self._policy_operands([self._unwrap(x) for x in inputs])
        if method == "__call__":
            result = ufunc(*raw, **kwargs)
            if ufunc is np.matmul:
                k = np.shape(raw[0])[-1] if np.ndim(raw[0]) else 1
                self.counts["mac"] += int(np.size(result)) * int(k)
            else:
                self.counts[_kind(ufunc)] += int(np.size(result))
        elif method == "reduce":
            result = ufunc.reduce(*raw, **kwargs)
            self.counts[_kind(ufunc)] += int(np.size(raw[0])) - int(np.size(result))
        elif method == "outer":
            result = ufunc.outer(*raw, **kwargs)
            self.counts[_kind(ufunc)] += int(np.size(result))
        else:
            raise CandidateError(f"ufunc method {method!r} is not supported")
        return self._wrap(self._policy_result(result))

    def _arg(self, function, array, axis):
        raw = self._unwrap(array)
        result = function(raw, axis=axis)
        self.counts["compare"] += int(np.size(raw)) - int(np.size(result))
        return self._wrap(result)

    def _as_counted(self, value):
        """A Counted operand. Under a policy a Python float becomes a 0-d array at the policy dtype, since a 0-d
        array is not a weak scalar (``ops.np.add(x, 0.5)`` keeps x's binary32)."""
        if isinstance(value, Counted):
            return value
        raw = self._unwrap(value)
        if self.dtype is not None and isinstance(raw, float):
            return Counted(np.asarray(raw, dtype=self.dtype), self)
        return Counted(self._policy_operands([raw])[0], self)

    def _namespace(self) -> SimpleNamespace:
        def alloc(maker):
            return lambda *args, **kwargs: Counted(self._float(maker, args, kwargs), self)

        def ufunc_entry(ufunc):
            return lambda *args: self._ufunc(ufunc, "__call__", tuple(self._as_counted(a) for a in args), {})

        def dot(a, b):
            return self._ufunc(np.matmul, "__call__", (self._as_counted(a), self._as_counted(b)), {})

        def where(condition, a, b):
            raw = [self._unwrap(self._as_counted(x)) for x in (condition, a, b)]
            raw = raw[:1] + self._policy_operands(raw[1:])
            result = self._policy_result(np.where(*raw))
            self.counts["compare"] += int(np.size(result))
            return self._wrap(result)

        def join(function):
            return lambda arrays, axis=0: Counted(self._policy_result(
                function(self._policy_operands([self._unwrap(a) for a in arrays]), axis=axis)), self)

        def normal(shape, seed, scale=1.0):
            # Drawn as before and rounded to the policy, so a seed gives the same draws under every precision.
            value = np.random.default_rng(seed).normal(0.0, scale, size=shape)
            return Counted(value if self.dtype is None else value.astype(self.dtype), self)

        def array(data):
            return Counted(np.array(self._unwrap(data), dtype=self.dtype or float), self)

        def sigmoid(a):
            # One unit per element, as tanh: a card whose accounting charges the logistic one operation can execute it
            # (card k2 in the 2026-10-13 smoke froze that convention, found no sigmoid here, and its reference code
            # fell back to 1/(1+exp(x)), which is sigma(-x)). Under a policy it is evaluated at the policy dtype.
            raw = np.asarray(self._unwrap(self._as_counted(a)), dtype=self.dtype or float)
            with np.errstate(over="ignore"):
                result = 1.0 / (1.0 + np.exp(-raw))
            self.counts["transcendental"] += int(np.size(result))
            return self._wrap(self._policy_result(result))

        namespace = {name: ufunc_entry(getattr(np, name)) for name in UFUNC_NAMES}
        namespace.update(sigmoid=sigmoid)
        namespace.update(
            zeros=alloc(np.zeros), ones=alloc(np.ones), full=alloc(np.full), eye=alloc(np.eye),
            arange=alloc(np.arange), array=array, normal=normal, dot=dot, matmul=dot, where=where,
            concatenate=join(np.concatenate), stack=join(np.stack),
            outer=lambda a, b: self._ufunc(np.multiply, "outer", (self._as_counted(a), self._as_counted(b)), {}),
            sum=lambda a, axis=None: self._as_counted(a).sum(axis),
            max=lambda a, axis=None: self._as_counted(a).max(axis),
            min=lambda a, axis=None: self._as_counted(a).min(axis),
            argmax=lambda a, axis=None: self._as_counted(a).argmax(axis),
            argmin=lambda a, axis=None: self._as_counted(a).argmin(axis),
        )
        return SimpleNamespace(**namespace)


WORD_BITS = 64


def words(value) -> int:
    """Size in words of a state or parameter structure: one word per array element, per Python number and per
    numeric dict key. A Python integer wider than a 64-bit word counts one word per 64 bits it needs (FIX-8: one
    unbounded integer could otherwise carry any amount of state as one word), and an object array counts its
    elements' words. Strings and other objects are refused, so nothing in the state goes unmeasured."""
    if value is None:
        return 0
    if isinstance(value, Counted):
        return value.size
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return sum(words(v) for v in value.flat)
        return int(value.size)
    if isinstance(value, int) and not isinstance(value, bool):
        return max(1, -(-(abs(value).bit_length() + 1) // WORD_BITS))
    if isinstance(value, (bool, float)):
        return 1
    if isinstance(value, (list, tuple)):
        return sum(words(v) for v in value)
    if isinstance(value, dict):
        # Numeric keys hold information (a key-value table); string keys are field names.
        return sum(words(v) + (0 if isinstance(k, str) else words(k)) for k, v in value.items())
    raise CandidateError(f"cannot measure the size of a {type(value).__name__} in state or params")



# ----- carried configurations of a finite-state mechanism (FIX-8) ---------------------------------------------------
#
# words() measures how many numbers a state holds, not how many values they take: one Python integer is one word
# whether it ranges over 16 values or 2**60. A mechanism that declares a finite carried-state budget (architecture
# state_bits) is held to it through these: the state must be made of integers, its packed width must fit the budget,
# and the harness counts the distinct whole configurations it reaches.

class StateDomainError(ValueError):
    """A carried state that is not a finite integer configuration."""


def _integers(value, out: list, shape: list) -> None:
    if value is None:
        shape.append("none")
    elif isinstance(value, (bool, np.bool_)):
        shape.append("bool")
        out.append(int(value))
    elif isinstance(value, (int, np.integer)):
        shape.append("int")
        out.append(int(value))
    elif isinstance(value, (Counted, np.ndarray)):
        raw = value._a if isinstance(value, Counted) else value
        if raw.dtype.kind not in "biu":
            raise StateDomainError(f"the carried state holds a {raw.dtype} array of shape {raw.shape}; a finite "
                                   "carried state is made of integers")
        shape.append(("array", raw.dtype.str, raw.shape))
        out.extend(int(v) for v in raw.flat)
    elif isinstance(value, (list, tuple)):
        shape.append((type(value).__name__, len(value)))
        for item in value:
            _integers(item, out, shape)
    elif isinstance(value, dict):
        keys = sorted(value, key=repr)
        shape.append(("dict", tuple(k for k in keys if isinstance(k, str))))
        for key in keys:
            if not isinstance(key, str):
                _integers(key, out, shape)
            _integers(value[key], out, shape)
    else:
        raise StateDomainError(f"the carried state holds a {type(value).__name__} ({value!r:.60}); a finite carried "
                               "state is made of integers")


def configuration(state) -> tuple:
    """The whole carried state as one hashable configuration: its structure (container types, lengths, field names,
    array dtypes and shapes) and every integer in it, in a fixed order. Raises StateDomainError when the state holds
    anything but integers (a float, a string, an object)."""
    out: list[int] = []
    shape: list = []
    _integers(state, out, shape)
    return tuple(shape), tuple(out)


def packed_bits(fields) -> int:
    """Bits that hold the integers ``fields`` side by side: each nonnegative v needs v.bit_length() bits, a negative
    one a sign bit more. The joint accounting of every carried field, not each field against the budget alone."""
    return sum(v.bit_length() if v >= 0 else (-v).bit_length() + 1 for v in fields)

# ----- loading candidate code -------------------------------------------------------------------------------------

def audit_source(source: str) -> list[str]:
    """Constructs that would let candidate work escape the counter: imports, private or dunder attributes, and
    builtins that reach outside the namespace."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            problems.append(f"line {node.lineno}: imports are not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {node.lineno}: global and nonlocal are not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            problems.append(f"line {node.lineno}: private attribute {node.attr!r}")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            problems.append(f"line {node.lineno}: name {node.id!r} is not allowed")
    return problems


def load_functions(source: str, names: tuple) -> SimpleNamespace:
    """Audit and execute counted code; return the named functions. Executed lines are attributed to
    CANDIDATE_FILE, which the line tracer follows."""
    problems = audit_source(source)
    if problems:
        raise CandidateError("; ".join(problems))
    namespace = {"__builtins__": dict(SAFE_BUILTINS)}
    exec(compile(source, CANDIDATE_FILE, "exec"), namespace)
    missing = [name for name in names if not callable(namespace.get(name))]
    if missing:
        raise CandidateError(f"code does not define {missing}")
    return SimpleNamespace(**{name: namespace[name] for name in names})


def load_candidate(source: str) -> SimpleNamespace:
    return load_functions(source, ("init", "update", "predict"))


class LineTracer:
    def __init__(self):
        self.lines = 0

    def __enter__(self):
        self._previous = sys.gettrace()
        sys.settrace(self._global)
        return self

    def __exit__(self, *exc):
        sys.settrace(self._previous)
        return False

    def _global(self, frame, event, arg):
        if event == "call" and frame.f_code.co_filename == CANDIDATE_FILE:
            return self._local
        return None

    def _local(self, frame, event, arg):
        if event == "line":
            self.lines += 1
        return self._local


# ----- measurement ------------------------------------------------------------------------------------------------

@dataclass
class Measurement:
    point: dict
    ops: int
    by_kind: dict
    lines: int
    state_peak_words: int
    init_ops: int
    param_words_start: int
    param_words_end: int
    seconds: float


def token_stream(n: int, vocab: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(vocab) for _ in range(n)]


def measure(candidate: SimpleNamespace, point: dict, *, vocab: int = DEFAULT_VOCAB, seed: int = 0,
            precision_bits: int | None = None) -> Measurement:
    """Run the candidate over n = point["n"] tokens with the other sizes from ``point``, under the declared precision
    policy when one is given (FIX-20)."""
    sizes = {k: v for k, v in point.items() if k != "n"}
    counter = OpCounter(precision_bits)
    started = time.perf_counter()
    params, state = candidate.init(dict(sizes), counter)
    init_ops = counter.total
    counter.reset()
    params_start = words(params)
    peak = words(state)
    tokens = token_stream(point["n"], sizes.get("V", vocab), seed)
    with LineTracer() as tracer:
        for token in tokens:
            state = candidate.update(params, state, token, counter)
            peak = max(peak, words(state))
            candidate.predict(params, state, counter)
    return Measurement(dict(point), counter.total, dict(counter.counts), tracer.lines, peak, init_ops,
                       params_start, words(params), round(time.perf_counter() - started, 4))


# ----- the claim and the verdict ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class CostClaim:
    """A card's cost claim. ``ops`` and ``state`` are SymPy expressions in n and the names in ``sizes``: total
    counted operations of update plus predict over n tokens, and peak state words. ``sizes`` lists the values of
    every other size parameter (the plan asks for two each). ``exponent`` is the claimed exponent in n (1 for
    linear). A remainder, when declared, bounds |counted - formula| at every point."""
    ops: str
    state: str
    sizes: dict = field(default_factory=dict)
    exponent: float | None = None
    ops_remainder: str | None = None
    state_remainder: str | None = None
    ops_remainder_lower_in: list | None = None
    state_remainder_lower_in: list | None = None


CLAIM_FIELDS = ("ops", "state", "sizes", "exponent", "ops_remainder", "state_remainder", "ops_remainder_lower_in",
                "state_remainder_lower_in")


def _symbols(claim: CostClaim) -> dict:
    return {name: sp.Symbol(name, positive=True) for name in ("n", *claim.sizes)}


def exponent_in(expr: sp.Expr, symbols: dict, name: str, fixed: dict) -> float:
    """lim log f / log x as the size ``name`` grows, the other sizes fixed at ``fixed``."""
    x = symbols[name]
    others = {symbols[k]: v for k, v in fixed.items() if k != name}
    return float(sp.limit(sp.log(expr.subs(others)) / sp.log(x), x, sp.oo))


def remainder_problems(claim: CostClaim, ns=DEFAULT_NS) -> list[str]:
    """What is wrong with a claim's declared remainders, found without running anything. A remainder admits
    lower-order work the formula does not state exactly (plan section 4.4); it must be declared before the proof
    attempts with the sizes in which it is of lower order, be nonnegative and below its formula at every grid point,
    and have a strictly smaller asymptotic exponent than its formula in every declared size. A remainder of the same
    order as the formula could absorb missing leading work, as card k2's 8nd^2 MAC shortfall in the 2026-10-13
    smoke would have needed."""
    symbols = _symbols(claim)
    names = ["n", *claim.sizes]
    settings = [dict(zip(claim.sizes, combo)) for combo in itertools.product(*claim.sizes.values())] or [{}]
    points = [dict(setting, n=n) for setting in settings for n in ns]
    problems = []
    for label, formula, remainder, lower_in in (
            ("ops", claim.ops, claim.ops_remainder, claim.ops_remainder_lower_in),
            ("state", claim.state, claim.state_remainder, claim.state_remainder_lower_in)):
        field_name = f"{label}_remainder_lower_in"
        if not remainder:
            if lower_in:
                problems.append(f"{field_name} is set but no {label}_remainder is declared")
            continue
        try:
            rem, main = parse_formula(remainder, symbols), parse_formula(formula, symbols)
        except (ValueError, TypeError, sp.SympifyError) as exc:
            problems.append(f"{label}_remainder {remainder!r} cannot be read: {exc}")
            continue
        if rem == 0:
            continue
        if not lower_in:
            problems.append(f"{label}_remainder {remainder} needs {field_name}: the sizes (from {names}) in which it "
                            f"is of strictly lower order than the {label} formula, stated before freezing")
            continue
        unknown = [name for name in lower_in if name not in names]
        if unknown:
            problems.append(f"{field_name} names {unknown}, which are not sizes of this claim ({names})")
            continue
        for point in points:
            r, f = _value(rem, symbols, point), _value(main, symbols, point)
            if r < 0:
                problems.append(f"{label}_remainder {remainder} is {r} at {point}: a remainder bounds |counted - "
                                "formula| and cannot be negative")
                break
            if r >= f:
                problems.append(f"{label}_remainder {remainder} is {r} at {point}, not below the {label} formula's "
                                f"{f}: a remainder that large can absorb leading work")
                break
        for name in lower_in:
            for setting in settings:
                fixed = dict(setting, n=min(ns))
                e_rem, e_main = exponent_in(rem, symbols, name, fixed), exponent_in(main, symbols, name, fixed)
                if not e_rem < e_main:
                    problems.append(f"{label}_remainder {remainder} grows as {name}^{e_rem:g}, not slower than the "
                                    f"{label} formula's {name}^{e_main:g} (other sizes {setting}): it is not of lower "
                                    f"order in {name}")
                    break
    return problems


def parse_formula(text: str, symbols: dict) -> sp.Expr:
    expr = sp.sympify(text, locals=dict(symbols, log=sp.log, sqrt=sp.sqrt))
    unknown = {str(s) for s in expr.free_symbols} - set(symbols)
    if unknown:
        raise ValueError(f"formula {text!r} uses symbols {sorted(unknown)} that are not size parameters")
    return expr


def asymptotic_exponent(expr: sp.Expr, symbols: dict, fixed: dict) -> float:
    """lim_{n -> oo} log f / log n with the other sizes fixed at ``fixed``."""
    n = symbols["n"]
    value = sp.limit(sp.log(expr.subs({symbols[k]: v for k, v in fixed.items()})) / sp.log(n), n, sp.oo)
    return float(value)


def loglog_slope(xs, ys) -> float:
    if len(xs) < 2:
        raise ValueError("a slope needs at least two points")
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.maximum(np.asarray(ys, dtype=float), 1.0))
    return float(np.polyfit(lx, ly, 1)[0])


def slope_band(xs, formula, remainder) -> tuple[float, float]:
    """Least and greatest log-log slope (as ``loglog_slope`` fits it) of any counts within ``remainder`` of
    ``formula`` at the points ``xs``. The fitted slope is sum c_i log y_i with c_i = (lx_i - mean) / Sxx, so its
    extremes put every y_i at the end of its interval that the sign of c_i favours."""
    lx = np.log(np.asarray(xs, dtype=float))
    c = (lx - lx.mean()) / float(np.sum((lx - lx.mean()) ** 2))
    f, r = np.asarray(formula, dtype=float), np.asarray(remainder, dtype=float)
    low_y, high_y = np.log(np.maximum(f - r, 1.0)), np.log(np.maximum(f + r, 1.0))
    low = float(np.sum(np.where(c > 0, c * low_y, c * high_y)))
    high = float(np.sum(np.where(c > 0, c * high_y, c * low_y)))
    return low, high


def _value(expr, symbols, point) -> sp.Expr:
    return expr.subs({symbols[k]: v for k, v in point.items()})


def verify_cost(source: str, claim: CostClaim, *, ns=DEFAULT_NS, tolerance: float = DEFAULT_TOLERANCE,
                vocab: int = DEFAULT_VOCAB, seed: int = 0, architecture: dict | None = None,
                precision_bits: int | None = None) -> dict:
    """Run the candidate on the grid and compare with the claim. Returns the obligation-check contract
    (status, evidence, counterexample) plus ``failures`` and every ``measurement``. ``architecture``
    ({"shared": the profile's contract, "widths": the statement's widths}, FIX-4) adds the architecture check: at every
    setting of the sizes the parameters and state ``init`` returns must stay within the declared dimensions, at the
    declared precision, and realize the declared hidden width (architecture.py). ``precision_bits`` (FIX-20) runs the
    candidate under the card's declared precision policy, the same counter the conformance audit uses; the
    architecture check then also holds the stored arrays to that precision."""
    if len(set(ns)) < 2:
        raise ValueError("the verifier needs at least two values of n")
    single = [name for name, values in claim.sizes.items() if len(set(values)) < 2]
    if single:
        raise ValueError(f"size parameters {single} need at least two values each (plan section 4.4, P1)")
    declared = remainder_problems(claim, ns)
    if declared:
        raise ValueError("the claim's remainder is not admissible: " + "; ".join(declared))
    candidate = load_candidate(source)
    symbols = _symbols(claim)
    ops_expr, state_expr = parse_formula(claim.ops, symbols), parse_formula(claim.state, symbols)
    ops_rem = parse_formula(claim.ops_remainder, symbols) if claim.ops_remainder else sp.Integer(0)
    state_rem = parse_formula(claim.state_remainder, symbols) if claim.state_remainder else sp.Integer(0)
    names = list(claim.sizes)
    settings = [dict(zip(names, combo)) for combo in itertools.product(*(claim.sizes[k] for k in names))]
    points = [dict(setting, n=n) for setting in settings for n in ns]
    policy_dtype(precision_bits)
    measurements = [measure(candidate, p, vocab=vocab, seed=seed, precision_bits=precision_bits) for p in points]
    failures = []

    for m in measurements:
        expected = _value(ops_expr, symbols, m.point)
        if expected <= 0:
            raise ValueError(f"the ops formula is not positive at {m.point}: {expected}")
        if abs(m.ops - expected) > _value(ops_rem, symbols, m.point):
            # The units by kind show a convention mismatch at a glance: in the 2026-10-13 smoke, card k2's formula
            # counted a length-k dot product as 2k-1 while its source's ops.np.dot was charged k mac units.
            failures.append({"check": "ops_exact", "point": m.point, "counted": m.ops, "formula": str(expected),
                             "difference": str(m.ops - expected), "counted_by_kind": dict(m.by_kind)})
        expected_state = _value(state_expr, symbols, m.point)
        if abs(m.state_peak_words - expected_state) > _value(state_rem, symbols, m.point):
            failures.append({"check": "state_exact", "point": m.point, "counted": m.state_peak_words,
                             "formula": str(expected_state),
                             "difference": str(m.state_peak_words - expected_state)})
        if m.param_words_end != m.param_words_start:
            failures.append({"check": "params_grew", "point": m.point, "start": m.param_words_start,
                             "end": m.param_words_end})

    def band(row, xs):
        values = [float(_value(ops_expr, symbols, m.point)) for m in row]
        slope = loglog_slope(xs, values)
        if not claim.ops_remainder:
            return slope, slope, slope
        low, high = slope_band(xs, values, [float(_value(ops_rem, symbols, m.point)) for m in row])
        return slope, low, high

    def slope_entry(base: dict, slope: float, low: float, high: float) -> dict:
        entry = dict(base, formula=round(slope, 4))
        if claim.ops_remainder:
            entry["formula_band"] = [round(low, 4), round(high, 4)]
        return entry

    def slope_failure(check: str, sizes: dict, measured: float, slope: float, low: float, high: float) -> dict:
        failure = {"check": check, "sizes": sizes, "measured_slope": round(measured, 4),
                   "formula_slope": round(slope, 4), "tolerance": tolerance}
        if claim.ops_remainder:
            failure["formula_band"] = [round(low, 4), round(high, 4)]
        return failure

    slopes = []
    for setting in settings:
        row = [m for m in measurements if all(m.point[k] == v for k, v in setting.items())]
        xs = [m.point["n"] for m in row]
        formula_slope, low, high = band(row, xs)
        measured = loglog_slope(xs, [m.ops for m in row])
        lines = loglog_slope(xs, [m.lines for m in row])
        slopes.append(slope_entry({"sizes": setting, "parameter": "n", "measured": round(measured, 4),
                                   "lines": round(lines, 4)}, formula_slope, low, high))
        if measured < low - tolerance or measured > high + tolerance:
            failures.append(slope_failure("exponent_n", setting, measured, formula_slope, low, high))
        if lines > high + tolerance:
            failures.append({"check": "lines_exponent_n", "sizes": setting, "lines_slope": round(lines, 4),
                             "formula_slope": round(high, 4), "tolerance": tolerance})
    n_max = max(ns)
    for name in names:
        values = sorted(claim.sizes[name])
        if len(values) < 2:
            continue
        others = [k for k in names if k != name]
        for combo in itertools.product(*(claim.sizes[k] for k in others)):
            fixed = dict(zip(others, combo), n=n_max)
            row = sorted((m for m in measurements if all(m.point[k] == v for k, v in fixed.items())),
                         key=lambda m: m.point[name])
            xs = [m.point[name] for m in row]
            formula_slope, low, high = band(row, xs)
            measured = loglog_slope(xs, [m.ops for m in row])
            slopes.append(slope_entry({"sizes": fixed, "parameter": name, "measured": round(measured, 4)},
                                      formula_slope, low, high))
            if measured < low - tolerance or measured > high + tolerance:
                failures.append(slope_failure(f"exponent_{name}", fixed, measured, formula_slope, low, high))

    arch_points = []
    if architecture is not None:
        from . import architecture as arch
        for setting in settings:
            params, state = candidate.init(dict(setting), OpCounter(precision_bits))
            found = arch.survey(params=params, state=state)
            point = arch.check_point(architecture["shared"], architecture["widths"], dict(setting, n=max(ns)), found,
                                     positions=max(ns), vocab=setting.get("V", vocab),
                                     precision=precision_bits is not None, realize_hidden=True)
            arch_points.append(point)
            if point["violations"]:
                failures.append({"check": "architecture", "sizes": setting, "dimensions": point["dimensions"],
                                 "violations": point["violations"]})

    formula_exponent = asymptotic_exponent(ops_expr, symbols, settings[0])
    if claim.exponent is not None and abs(formula_exponent - claim.exponent) > 1e-9:
        failures.append({"check": "claimed_exponent", "claimed": claim.exponent, "formula_exponent": formula_exponent})

    failures.sort(key=lambda f: CHECK_ORDER.index(f["check"]) if f["check"] in CHECK_ORDER else len(CHECK_ORDER))
    status = VIOLATED if failures else DISCHARGED
    evidence = (f"{len(points)} points, n in {list(ns)}, sizes {claim.sizes}; formula exponent in n "
                f"{formula_exponent:g}; slopes in n (measured/formula): "
                + ", ".join(f"{s['measured']}/{s['formula']}" for s in slopes if s["parameter"] == "n")
                + (f"; failed checks: {sorted({f['check'] for f in failures})}" if failures else "; all checks pass"))
    first_by_check: dict = {}
    for failure in failures:
        first_by_check.setdefault(failure["check"], failure)
    result = {"status": status, "evidence": evidence, "counterexample": failures[0] if failures else None,
              "failures": failures, "failed_checks": first_by_check, "formula_exponent": formula_exponent,
              "slopes": slopes, "tolerance": tolerance, "measurements": [asdict(m) for m in measurements],
              "precision_bits": precision_bits}
    if architecture is not None:
        from . import architecture as arch
        result["architecture"] = arch.summary(arch_points)
    return result


# ----- what the prover and the statement fixer are told (host side) -----------------------------------------------

# Names a source of each sandbox finds besides its builtins: learnability, simulation and construction sources get
# NumPy as ``np``; a cost source gets nothing and reaches arrays only through its ``ops`` argument.
SANDBOX_GLOBALS = {"cost": (), "learnability": ("np",), "simulation": ("np",), "construction": ("np",)}
# Names that exist in some sandbox but not in every one, flagged when a source uses them unbound.
_INJECTED = frozenset({"np"})


def _bound_names(tree: ast.AST) -> set[str]:
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
    return bound


def sandbox_problems(source: str, form: str) -> list[str]:
    """Names a source uses that its sandbox does not have, found without running it: a builtin absent from the
    checker's namespace (``str``, ``print``, ``set``, ``Exception``), ``np`` in a cost source, or a class statement
    (it needs ``__build_class__``). Only names bound nowhere in the source are reported, so a local ``str = ...`` is
    not flagged. Construction sources run with the standard builtins, so only ``np``-style names apply there. In the
    2026-10-13 smoke all four final card k1 learners called ``str`` in init and each spent an executor job to learn
    it from a NameError."""
    if form not in SANDBOX_GLOBALS:
        raise ValueError(f"no sandbox for form {form!r}")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    available = set(dir(builtins)) if form == "construction" else set(SAFE_BUILTINS)
    provided = set(SANDBOX_GLOBALS[form])
    bound = _bound_names(tree)
    guarded = _guarded_nodes(tree)
    problems, seen = [], set()
    if form != "construction":
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                problems.append(f"line {node.lineno}: class statements do not work in this sandbox (it has no "
                                "__build_class__); use functions and dicts")
    names = sorted((node for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)),
                   key=lambda node: (node.lineno, node.col_offset))
    for node in names:
        name = node.id
        if name in bound or name in available or name in provided or name in seen or id(node) in guarded:
            continue
        if hasattr(builtins, name) or name in _INJECTED:
            seen.add(name)
            what = "builtin" if hasattr(builtins, name) else "name"
            problems.append(f"line {node.lineno}: {what} {name!r} does not exist in the {form} sandbox; it would "
                            "raise NameError when that line runs")
    if form == "cost":
        problems += _counted_namespace_problems(tree, guarded)
    return problems


def _guarded_nodes(tree: ast.AST) -> set[int]:
    """Nodes inside the body of a try statement: an absent name there may be caught on purpose."""
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for statement in node.body:
                guarded.update(id(inner) for inner in ast.walk(statement))
    return guarded


def _counted_namespace_problems(tree: ast.AST, guarded: set[int]) -> list[str]:
    """Attributes of ``ops.np`` (or of a name assigned from ``<x>.np``) that the counted namespace does not have. In the
    2026-10-13 smoke card k2's first cost source called ops.np.sigmoid and spent a job on the AttributeError."""
    available = set(vars(OpCounter().np))
    aliases = {target.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and isinstance(node.value, ast.Attribute) and node.value.attr == "np"
               for target in node.targets if isinstance(target, ast.Name)}
    problems, seen = [], set()
    for node in sorted((n for n in ast.walk(tree) if isinstance(n, ast.Attribute)),
                       key=lambda n: (n.lineno, n.col_offset)):
        base = node.value
        counted = ((isinstance(base, ast.Attribute) and base.attr == "np")
                   or (isinstance(base, ast.Name) and base.id in aliases))
        if not counted or node.attr in available or node.attr.startswith("_") or id(node) in guarded:
            continue
        if node.attr not in seen:
            seen.add(node.attr)
            problems.append(f"line {node.lineno}: ops.np has no {node.attr!r}; the counted namespace has "
                            + ", ".join(sorted(available)))
    return problems


def crash_line(exc: BaseException, source: str | None, filename: str) -> str | None:
    """Where in the submitted source an exception was raised: the deepest traceback frame compiled from
    ``filename``, with its line number, function and the line's text. None when no frame is in the source (the
    exception came from the harness itself)."""
    found = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == filename:
            found = (tb.tb_lineno, tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    if found is None:
        return None
    lineno, name = found
    lines = (source or "").splitlines()
    text = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
    return f"line {lineno} in {name}(): {text}"[:400]


def describe_exception(exc: BaseException) -> str:
    """``Type: message``, with an empty message said so: a bare ``assert`` raised ``AssertionError:`` and nothing
    else in the smoke's recall construction."""
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else f"{type(exc).__name__} (no message)"


def counting_text() -> str:
    """The P1 counting convention, for the statement fixer and the prover. Every unit count below is measured by
    running this module's counter on small arrays when the text is built, and the sizes ``init`` receives by calling
    ``measure``, so the text states what the checker does. In the 2026-10-13 smoke the fixer froze a scalar-dot
    convention (a length-k dot product as k multiplications and k-1 additions) while ``ops.np.dot`` charges k mac
    units, so card k2's reference code failed ops_exact at every point (counted 44736 against 74176 at n=64, d=8,
    V=4) and card k1's prover alternated between the two conventions."""
    def units(fn) -> dict:
        ops = OpCounter()
        fn(ops.np)
        return {kind: count for kind, count in ops.counts.items() if count}

    k = 8
    seen = {}

    class Probe:
        def init(self, sizes, ops):
            seen["sizes"] = sorted(sizes)
            return {}, {"t": 0}

        def update(self, params, state, token, ops):
            return {"t": state["t"] + 1}

        def predict(self, params, state, ops):
            return 0

    measure(Probe(), {"n": 3, "d": 2, "V": 4})
    dot = units(lambda np_: np_.dot(np_.ones(k), np_.ones(k)))
    matvec = units(lambda np_: np_.ones((4, k)) @ np_.ones(k))
    scalar_dot = units(lambda np_: np_.sum(np_.multiply(np_.ones(k), np_.ones(k))))
    add = units(lambda np_: np_.add(np_.ones(k), np_.ones(k)))
    exp = units(lambda np_: np_.exp(np_.ones(k)))
    reduce_sum = units(lambda np_: np_.sum(np_.ones(k)))
    reduce_max = units(lambda np_: np_.max(np_.ones(k)))
    argmax = units(lambda np_: np_.argmax(np_.ones(k)))
    condition = units(lambda np_: np_.greater(np_.ones(k), 0.5))
    where = Counter(units(lambda np_: np_.where(np_.greater(np_.ones(k), 0.5), np_.ones(k), np_.zeros(k))))
    where.subtract(condition)
    where = {kind: count for kind, count in where.items() if count}
    logistic = units(lambda np_: np_.divide(1.0, np_.add(1.0, np_.exp(np_.negative(np_.ones(k))))))
    free = units(lambda np_: np_.concatenate([np_.zeros(k), np_.ones(k).copy()[1:]]).reshape(1, 2 * k - 1))
    by_kind = {kind: sorted(name for name in UFUNC_NAMES if _kind(getattr(np, name)) == kind)
               for kind in ("arith", "compare", "transcendental")}

    def show(counts: dict) -> str:
        return " + ".join(f"{count} {kind}" for kind, count in sorted(counts.items())) or "0"

    sigmoid = units(lambda np_: np_.sigmoid(np_.ones(k)))
    probe = OpCounter()
    at_two = float(probe.np.sigmoid(probe.np.array([2.0])).item())
    reversed_two = float(probe.np.divide(1.0, probe.np.add(1.0, probe.np.exp(probe.np.array([2.0])))).item())
    absent = [name for name in ("expit", "softmax", "einsum", "tensordot", "sort", "mean") if not hasattr(
        OpCounter().np, name)]
    return ("P1 COUNTING CONVENTION (measured from hybrid/math/cost.py's counter when this text was built; the counter "
            "decides, and a definition cannot change how it counts, so write the P1 formulas and the accounting "
            "definitions in these units). Every unit counts 1 toward the total, whatever its kind. "
            f"Matrix and dot products (`@`, ops.np.dot, ops.np.matmul) count k mac units per output element for a "
            f"contraction of length k: a length-{k} dot product counts {show(dot)}, a 4 x {k} matrix times a "
            f"length-{k} vector counts {show(matvec)}. The same dot product written as multiply then sum counts "
            f"{show(scalar_dot)} (k multiplications and k-1 additions); the two spellings are different formulas, so "
            "the formula must follow the spelling the reference code uses, and the reference code must use one "
            f"spelling consistently. Elementwise functions count one per output element (ops.np.add on two length-"
            f"{k} arrays: {show(add)}; ops.np.exp on length {k}: {show(exp)}): arith {', '.join(by_kind['arith'])}; "
            f"compare {', '.join(by_kind['compare'])}; transcendental {', '.join(by_kind['transcendental'])}. "
            f"Reductions count k-1 for k elements: the sum of {k} counts {show(reduce_sum)}, the max "
            f"{show(reduce_max)}, argmax {show(argmax)}. ops.np.where counts one compare per output element "
            f"({show(where)} for length {k}, besides the {show(condition)} of the comparison that made its "
            f"condition). ops.np.sigmoid(x) is the logistic sigma(x) = 1/(1+exp(-x)) and counts one transcendental "
            f"per element, as tanh ({show(sigmoid)} on length {k}; sigma(2) = {at_two:.4f}): a definition that charges "
            "a sigmoid one operation must be implemented with ops.np.sigmoid. Written out as 1/(1+exp(-x)) the same "
            f"function costs {show(logistic)} on length {k} (negative, exp, add, divide), and a formula must then count "
            f"those units. Mind the sign: 1/(1+exp(x)) is sigma(-x) ({reversed_two:.4f} at x = 2), not sigma(x); a "
            "reference implementation computes the activation the card states, it does not flip the sign and rely on "
            f"negated weights. Not in ops.np: {', '.join(absent)}; compose them from the functions above. "
            f"Creating, indexing, slicing, copying, reshaping and concatenating arrays count {show(free)}. Python "
            "arithmetic on plain numbers is not counted, but every executed source line is traced and its growth in "
            "n may not exceed the formula's. The total is update plus predict over n tokens; init is not counted. "
            f"init(sizes, ops) receives the sizes {seen['sizes']} for a point with n, d and V: n is never passed, so "
            "nothing can be reserved per position at init; state that grows with n grows as tokens arrive. State "
            "words are measured with cost.words on the state after init and after every update, and the peak is "
            "compared with the state formula: one word per array element, per Python number (a counter such as "
            f"state['t'] is one word: the probe state {{'t': 3}} measures {words({'t': 3})}) and per numeric dict key; "
            "string keys are free and string values are refused. Parameters are not state, but they may not grow "
            "during the sequence. The checks are exact: counted operations and state words must equal the formulas "
            "at every point. A formula may declare a remainder (ops_remainder, state_remainder) for lower-order work "
            "it cannot state exactly, only with the sizes in which that remainder is of strictly lower order "
            "(ops_remainder_lower_in, state_remainder_lower_in); before freezing the host refuses a remainder that is "
            "negative, not below its formula at some grid point, or not of lower order in a declared size, and the "
            "finite-grid slope checks then allow the slopes that counts within the remainder can have. A remainder "
            "never covers a difference in the leading term, such as dots counted as k mac units against a formula "
            "that counts 2k-1.")
