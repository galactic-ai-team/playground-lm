"""Lock the demo's Facts/Question line to the grounded SFT builder (Lumi glue).

demo/test_fact_retrieval.mjs asserts the same literal against buildFactsQuestion().
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stage_g_grounded import facts_prompt  # noqa: E402

EXPECTED = ("Facts: [1] Paris is the capital of France. [2] Bees make honey. "
            "[3] Mars is called the red planet. Question: What is the capital of France?")


def test_facts_prompt_matches_demo_literal():
    got = facts_prompt(["Paris is the capital of France.", "Bees  make honey.",
                        "Mars is called the red planet."], "  What is the\ncapital of France? ")
    assert got == EXPECTED
