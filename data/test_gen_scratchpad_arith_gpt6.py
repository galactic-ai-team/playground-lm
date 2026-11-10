"""Regression checks for generated annotation integrity (Stage G GPT-6 review).

The pre-patch verifier accepted an inconsistent write digit, carry, or final
'digits from the top place down' text as long as equations and final Answer
were right. Generated corpus rows must match the complete canonical solver.
"""
import pytest
from data.gen_scratchpad_arith import solve, verify_steps, check_row


def test_corrupted_write_digit_rejected_even_when_equations_valid():
    s = solve("add", 47, 85)
    bad = s.replace("write 2, carry 1", "write 9, carry 1", 1)
    assert bad != s
    assert verify_steps(bad) > 0  # An equation-only verifier misses this.
    with pytest.raises(ValueError, match="noncanonical|annotation|scratchpad"):
        check_row("add", 47, 85, bad)


def test_corrupted_digit_assembly_rejected():
    s = solve("add", 47, 85)
    bad = s.replace("Digits from the top place down: 1 3 2.",
                    "Digits from the top place down: 9 9 9.", 1)
    assert bad != s
    assert verify_steps(bad) > 0
    with pytest.raises(ValueError, match="noncanonical|annotation|scratchpad"):
        check_row("add", 47, 85, bad)


def test_corrupted_carry_annotation_rejected():
    s = solve("add", 47, 85)
    bad = s.replace("write 2, carry 1", "write 2, carry 4", 1)
    assert bad != s
    assert verify_steps(bad) > 0
    with pytest.raises(ValueError, match="noncanonical|annotation|scratchpad"):
        check_row("add", 47, 85, bad)


def test_original_solver_remains_accepted():
    for op, a, b in (("add", 47, 85), ("sub", 302, 47),
                     ("mul", 47, 36), ("div", 3584, 7)):
        check_row(op, a, b, solve(op, a, b))
