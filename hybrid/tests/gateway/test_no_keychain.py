"""The gateway must never touch the macOS keychain: guards every Python module under hybrid/.

Keychain prompts interrupt the owner, and probing the keychain was a likely cause of earlier prompts. Tokens live in
private files (``fsstore.py``). Docs may say so in words; code may not reference keychain APIs at all.
"""
import ast
import re
from pathlib import Path

HYBRID = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()
SKIPPED_DIRS = {"__pycache__"}

# Built from fragments so this file does not match its own patterns when read by a human grepping for them.
KEYCHAIN_WORD = re.compile("key" + "chain", re.IGNORECASE)
SECURITY_BINDINGS = re.compile(
    r"Security\.framework|CoreFoundation\.framework|\bSecItem\w*|\bkSec[A-Z]\w*|\bSecKeychain\w*|"
    r"(?:find|add|delete|dump)-(?:generic|internet)-password|\bsecurity\s+(?:find|add|delete|unlock|dump|list)",
    re.IGNORECASE)


def python_sources():
    for path in sorted(HYBRID.rglob("*.py")):
        if path.resolve() == THIS_FILE or SKIPPED_DIRS & set(path.relative_to(HYBRID).parts):
            continue
        yield path


def security_cli_calls(tree: ast.AST):
    """Calls such as subprocess.run(["security", ...]) or os.system("security ...")."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for arg in node.args[:1]:
            first = arg.elts[0] if isinstance(arg, (ast.List, ast.Tuple)) and arg.elts else arg
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                words = first.value.split()
                if words and Path(words[0]).name == "security":
                    yield node.lineno


def ctypes_security_imports(tree: ast.AST):
    """ctypes.CDLL / cdll.LoadLibrary of a Security or CoreFoundation framework path."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(
                r"(Security|CoreFoundation)(\.framework)?(/|$)", node.value) and "/" in node.value:
            yield node.lineno


def test_the_scan_covers_the_gateway_modules():
    names = {p.name for p in python_sources()}
    assert {"store.py", "fsstore.py", "refresher.py", "login.py"} <= names
    assert "keychain.py" not in names


def test_no_module_under_hybrid_references_the_keychain():
    offenders = []
    for path in python_sources():
        text = path.read_text()
        tree = ast.parse(text)
        for number, line in enumerate(text.splitlines(), 1):
            if KEYCHAIN_WORD.search(line) or SECURITY_BINDINGS.search(line):
                offenders.append(f"{path.relative_to(HYBRID)}:{number}: {line.strip()[:80]}")
        offenders += [f"{path.relative_to(HYBRID)}:{n}: security CLI call" for n in security_cli_calls(tree)]
        offenders += [f"{path.relative_to(HYBRID)}:{n}: ctypes Security binding" for n in ctypes_security_imports(tree)]
    assert not offenders, "keychain access is forbidden:\n" + "\n".join(offenders)


def test_the_detector_catches_each_kind_of_keychain_use():
    samples = {
        "ctypes": 'import ctypes\nsec = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")\n',
        "cli": 'import subprocess\nsubprocess.run(["security", "find-generic-password", "-s", "x"])\n',
        "cli_string": 'import os\nos.system("/usr/bin/security add-generic-password -s x")\n',
        "api": "status = SecItemCopyMatching(query, None)\n",
        "word": "class Keychain:\n    pass\n",
    }
    for name, source in samples.items():
        tree = ast.parse(source)
        hit = (KEYCHAIN_WORD.search(source) or SECURITY_BINDINGS.search(source) or list(security_cli_calls(tree))
               or list(ctypes_security_imports(tree)))
        assert hit, f"detector missed the {name} sample"


def test_gateway_has_no_keychain_module():
    assert not (HYBRID / "gateway" / ("key" + "chain.py")).exists()
