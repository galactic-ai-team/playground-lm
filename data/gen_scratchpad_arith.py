#!/usr/bin/env python3
"""Stage G: procedurally generated step-by-step (scratchpad) arithmetic SFT data.

Every row is machine-generated and machine-verified (MIT, this repository).
No external text, no model outputs.

Answer format (fixed, so a scorer can extract it):

    User: What is 47 + 85?
    Assistant: Work: Ones: 7 + 5 + 0 = 12, write 2, carry 1. Tens: 4 + 8 + 1 = 13,
    write 3, carry 1. Hundreds: carry 1, write 1. Answer: 132

(The real text is one line; ``format_example`` collapses whitespace exactly as
the Stage E/F builders do, so it matches the demo's ``User:/Assistant:`` form.)

Operations: add, sub (borrowing, negative results), mul (k-digit x 1-digit
column steps, and k-digit x 2-digit via partial products), div (exact long
division, quotient only). Every step equation is re-checked by
``verify_steps`` and the final ``Answer:`` by ``extract_answer``.

Split: hash of the canonical problem key ``op:a:b`` (not the template), so a
problem never appears in both train and val under different phrasings.
``--exclude-keys`` takes a file of canonical keys (one per line) that must
never be emitted -- the owner runs this locally with keys derived from
private suites; the keys file itself is never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

LICENSE = "MIT (this repository)"
SOURCE = "generated-scratchpad-arithmetic"
SCHEMA = 1
ANSWER_PREFIX = "Answer:"
PLACES = ["Ones", "Tens", "Hundreds", "Thousands", "Ten-thousands", "Hundred-thousands",
          "Millions"]

PHRASINGS = {
    "add": ["What is {a} + {b}?", "Compute {a} plus {b}.", "Add {a} and {b}.",
            "{a} + {b} = ?", "Find the sum of {a} and {b}.",
            "Work out {a} + {b} step by step."],
    "sub": ["What is {a} - {b}?", "Compute {a} minus {b}.", "Subtract {b} from {a}.",
            "{a} - {b} = ?", "Find the difference {a} - {b}.",
            "Work out {a} - {b} step by step."],
    "mul": ["What is {a} * {b}?", "Compute {a} times {b}.", "Multiply {a} by {b}.",
            "{a} x {b} = ?", "Find the product of {a} and {b}.",
            "Work out {a} * {b} step by step."],
    "div": ["What is {a} / {b}?", "Compute {a} divided by {b}.", "Divide {a} by {b}.",
            "{a} / {b} = ?", "Find the quotient of {a} and {b}.",
            "Work out {a} / {b} step by step."],
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def format_example(user: str, assistant: str) -> str:
    """Same as data/build_stage_e_sft.format_example (kept local: no numpy import)."""
    user, assistant = normalize_space(user), normalize_space(assistant)
    if not user or not assistant:
        raise ValueError("empty turn")
    return f"User: {user}\nAssistant: {assistant}"


def _digits(n: int) -> list[int]:
    """Least-significant digit first."""
    return [int(c) for c in reversed(str(n))]


def _digits_line(n: int) -> str:
    """Spelled top-down so the model reads the column writes back in order."""
    return "Digits from the top place down: " + " ".join(str(abs(n))) + "."


def _place(i: int) -> str:
    return PLACES[i] if i < len(PLACES) else f"Column {i + 1}"


# ---- step writers --------------------------------------------------------------

def add_steps(a: int, b: int) -> tuple[list[str], int]:
    if a < 0 or b < 0:
        raise ValueError("add_steps expects non-negative operands")
    da, db = _digits(a), _digits(b)
    carry, steps = 0, []
    for i in range(max(len(da), len(db))):
        x = da[i] if i < len(da) else 0
        y = db[i] if i < len(db) else 0
        s = x + y + carry
        steps.append(f"{_place(i)}: {x} + {y} + {carry} = {s}, write {s % 10}, carry {s // 10}.")
        carry = s // 10
    if carry:
        steps.append(f"{_place(max(len(da), len(db)))}: carry {carry}, write {carry}.")
    steps.append(_digits_line(a + b))
    return steps, a + b


def sub_steps(a: int, b: int) -> tuple[list[str], int]:
    if a < 0 or b < 0:
        raise ValueError("sub_steps expects non-negative operands")
    steps: list[str] = []
    neg = b > a
    if neg:
        steps.append(f"{b} is bigger than {a}, so compute {b} - {a} and make it negative.")
        a, b = b, a
    da, db = _digits(a), _digits(b)
    borrow = 0
    for i in range(len(da)):
        x = da[i]
        y = db[i] if i < len(db) else 0
        d = x - y - borrow
        if d < 0:
            steps.append(f"{_place(i)}: {x} - {y} - {borrow} is below 0, borrow 1: "
                         f"{x + 10} - {y} - {borrow} = {d + 10}, write {d + 10}.")
            borrow = 1
        else:
            steps.append(f"{_place(i)}: {x} - {y} - {borrow} = {d}, write {d}.")
            borrow = 0
    result = a - b
    steps.append(_digits_line(result))
    return steps, (-result if neg else result)


def mul1_steps(a: int, m: int) -> tuple[list[str], int]:
    """a (k digits) x m (one digit), column by column with carries."""
    if not 0 <= m <= 9 or a < 0:
        raise ValueError("mul1_steps expects a >= 0 and a one-digit multiplier")
    da = _digits(a)
    carry, steps = 0, []
    for i, x in enumerate(da):
        p = x * m + carry
        steps.append(f"{_place(i)}: {x} * {m} + {carry} = {p}, write {p % 10}, carry {p // 10}.")
        carry = p // 10
    if carry:
        steps.append(f"{_place(len(da))}: carry {carry}, write {carry}.")
    steps.append(_digits_line(a * m))
    return steps, a * m


def mul_steps(a: int, b: int) -> tuple[list[str], int]:
    if a < 0 or b < 0:
        raise ValueError("mul_steps expects non-negative operands")
    if b < 10:
        return mul1_steps(a, b)
    if b >= 100:
        raise ValueError("multiplier must be at most two digits")
    ones, tens = b % 10, b // 10
    s1, p1 = mul1_steps(a, ones)
    s2, p2 = mul1_steps(a, tens)
    steps = [f"Split {b} into {tens * 10} + {ones}."]
    steps.append(f"First {a} * {ones}:")
    steps += s1
    steps.append(f"So {a} * {ones} = {p1}.")
    steps.append(f"Next {a} * {tens}:")
    steps += s2
    steps.append(f"So {a} * {tens} = {p2}, and {a} * {tens * 10} = {p2 * 10}.")
    steps.append(f"Add {p1} + {p2 * 10}:")
    s3, total = add_steps(p1, p2 * 10)
    steps += s3
    return steps, total


def div_steps(a: int, b: int) -> tuple[list[str], int]:
    """Exact long division a / b with a one-digit divisor (2..9)."""
    if not 2 <= b <= 9 or a < 0 or a % b:
        raise ValueError("div_steps expects an exact division by 2..9")
    steps, rem, q_digits = [], 0, []
    for c in str(a):
        cur = rem * 10 + int(c)
        q, rem = divmod(cur, b)
        steps.append(f"Bring down {c}: {cur} / {b} = {q} remainder {rem}.")
        q_digits.append(str(q))
    return steps, a // b


WRITERS = {"add": add_steps, "sub": sub_steps, "mul": mul_steps, "div": div_steps}


def solve(op: str, a: int, b: int) -> str:
    steps, result = WRITERS[op](a, b)
    return "Work: " + " ".join(steps) + f" {ANSWER_PREFIX} {result}"


# ---- verification --------------------------------------------------------------

_EQ = re.compile(r"(-?\d+)((?:\s*[-+*/]\s*-?\d+)+)\s*=\s*(-?\d+)")
_DIV_REM = re.compile(r"(\d+) / (\d+) = (\d+) remainder (\d+)")
_ANS = re.compile(r"Answer:\s*(-?\d+)\s*$")


def extract_answer(text: str) -> int | None:
    """Final answer after the fixed ``Answer:`` cue (last occurrence, end of text)."""
    m = _ANS.search(normalize_space(text))
    return int(m.group(1)) if m else None


def verify_steps(text: str) -> int:
    """Re-check every ``x op y [op z] = r`` and division-remainder step. Returns the
    number of checked equations; raises ValueError on the first wrong one."""
    checked = 0
    for m in _DIV_REM.finditer(text):
        cur, d, q, r = map(int, m.groups())
        if cur != d * q + r or not 0 <= r < d:
            raise ValueError(f"bad division step: {m.group(0)}")
        checked += 1
    stripped = _DIV_REM.sub("", text)
    for m in _EQ.finditer(stripped):
        expr = m.group(1) + m.group(2)
        if "/" in expr:
            raise ValueError(f"unexpected bare division: {m.group(0)}")
        tokens = re.findall(r"[-+*]|\d+", expr.replace(" ", ""))
        # left-to-right with * precedence (only a*b+c appears)
        value = _eval_simple(tokens)
        if value != int(m.group(3)):
            raise ValueError(f"bad step: {m.group(0)}")
        checked += 1
    return checked


def _eval_simple(tokens: list[str]) -> int:
    # tokens alternate number, op, number ...; leading '-' handled by regex groups
    nums, ops = [], []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "-" and (i == 0 or tokens[i - 1] in "+-*"):
            nums.append(-int(tokens[i + 1]))
            i += 2
            continue
        if t in "+-*":
            ops.append(t)
        else:
            nums.append(int(t))
        i += 1
    # apply * first
    out_n, out_o = [nums[0]], []
    for op, n in zip(ops, nums[1:]):
        if op == "*":
            out_n[-1] *= n
        else:
            out_o.append(op)
            out_n.append(n)
    total = out_n[0]
    for op, n in zip(out_o, out_n[1:]):
        total = total + n if op == "+" else total - n
    return total


def check_row(op: str, a: int, b: int, assistant: str) -> None:
    if op == "div" and (b == 0 or a % b):
        raise ValueError("division rows must be exact with a non-zero divisor")
    truth = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b,
             "div": lambda: a // b}[op]()
    if extract_answer(assistant) != truth:
        raise ValueError(f"final answer mismatch for {op}:{a}:{b}")
    if verify_steps(assistant) == 0:
        raise ValueError("no checkable steps")


# ---- sampling / split ----------------------------------------------------------

def problem_key(op: str, a: int, b: int) -> str:
    return f"{op}:{a}:{b}"


def is_val(key: str, val_frac: float, salt: str = "stage-g-scratch") -> bool:
    h = hashlib.sha256(f"{salt}|{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < val_frac


def sample_problem(rng: random.Random, op: str, max_digits: int) -> tuple[int, int]:
    def num(lo_d: int, hi_d: int) -> int:
        d = rng.randint(lo_d, hi_d)
        return rng.randint(0 if d == 1 else 10 ** (d - 1), 10 ** d - 1)

    if op in ("add", "sub"):
        return num(1, max_digits), num(1, max_digits)
    if op == "mul":
        if rng.random() < 0.6:
            return num(1, max(1, max_digits - 1)), rng.randint(2, 9)
        # two-digit multiplier: keep the multiplicand <= 2 digits (~200 tokens of work)
        return num(1, min(2, max_digits)), rng.randint(10, 99)
    b = rng.randint(2, 9)
    q = num(1, max(1, max_digits - 1))
    return q * b, b


def generate(n: int, seed: int, *, max_digits: int = 4, val_frac: float = 0.05,
             ops: tuple[str, ...] = ("add", "sub", "mul", "div"),
             exclude_keys: frozenset[str] = frozenset()) -> tuple[list[dict], list[dict]]:
    """Return (train, val) rows. Deterministic for (n, seed, max_digits, val_frac, ops)."""
    rng = random.Random(seed)
    seen: set[str] = set()
    train, val = [], []
    attempts = 0
    while len(train) + len(val) < n and attempts < n * 50:
        attempts += 1
        op = ops[(len(train) + len(val)) % len(ops)]
        a, b = sample_problem(rng, op, max_digits)
        key = problem_key(op, a, b)
        if key in seen or key in exclude_keys:
            continue
        seen.add(key)
        tmpl = rng.choice(PHRASINGS[op])
        user = tmpl.format(a=a, b=b)
        assistant = solve(op, a, b)
        check_row(op, a, b, assistant)
        split_val = is_val(key, val_frac)
        bucket = val if split_val else train
        bucket.append({
            "id": f"scratch-{op}-{len(train) + len(val)}",
            "source": SOURCE,
            "license": LICENSE,
            "category": f"arithmetic-scratchpad-{op}",
            "problem_key": key,
            "user": user,
            "assistant": assistant,
            "text": format_example(user, assistant),
        })
    if len(train) + len(val) < n:
        raise RuntimeError(f"could only generate {len(train) + len(val)} unique problems")
    return train, val


def load_exclude_keys(path: Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    keys = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()}
    return frozenset(k for k in keys if k and not k.startswith("#"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--max-digits", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--ops", default="add,sub,mul,div")
    ap.add_argument("--exclude-keys", type=Path, default=None,
                    help="owner-only file of op:a:b keys never to emit (not committed)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    ops = tuple(o.strip() for o in args.ops.split(",") if o.strip())
    bad = [o for o in ops if o not in WRITERS]
    if bad:
        ap.error(f"unknown ops: {bad}")
    excl = load_exclude_keys(args.exclude_keys)
    train, val = generate(args.n, args.seed, max_digits=args.max_digits,
                          val_frac=args.val_frac, ops=ops, exclude_keys=excl)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": SCHEMA, "kind": "stage_g_scratchpad_arith",
                "source": SOURCE, "license": LICENSE, "seed": args.seed, "n": args.n,
                "max_digits": args.max_digits, "val_frac": args.val_frac, "ops": list(ops),
                "excluded_keys_count": len(excl), "answer_format": f"... {ANSWER_PREFIX} <int>",
                "files": {}}
    for name, rows in (("train", train), ("val", val)):
        p = args.out_dir / f"scratchpad_{name}.jsonl"
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["files"][name] = {"path": p.name, "rows": len(rows), "sha256": _sha256(p)}
    (args.out_dir / "scratchpad_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
