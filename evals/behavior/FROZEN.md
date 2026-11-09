# Frozen behavior eval (Stage E)

Held-out probes frozen **before** Stage E hyperparameter iteration. Not used for training.

## Files

| File | Audience | sha256 |
|------|----------|--------|
| `behavior_eval.prompts.frozen.json` | All lanes (prompt text only) | `e0f7feb4af2cf6854f852318cbeb983d45a0c7dc85785ae5f46df16f4cf67d39` |
| `behavior_eval.expect.frozen.json` | **Eval lane only (GPT-6)** | `caf5a86f823f3918eaebb827963fec16e2eaa220b0245e57bf3d8eb235f8c8b6` |

Legacy combined blob sha256 (pre-split, for continuity): `ccaf97895ed3dda307cdee0f2ae50b9e023000684892c2af0b0e1b459f1007c3`.

**Do not edit** either file during candidate iteration.  
**Data / license / UI lanes:** use the prompts file only — do not design training data from expect scorers.  
**Eval lane:** `evals/run_behavior_eval.py` merges prompts + expects at score time.
