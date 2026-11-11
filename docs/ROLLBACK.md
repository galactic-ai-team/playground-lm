# PlaygroundLM live demo — one-step rollback

**Current live:** PlaygroundLM **v0.6 Chat** (Stage G candidate G2, step **31505**) with BM25 fact lookup.  
**Rollback target:** PlaygroundLM **v0.5 Chat** (Stage E Round 3, step **31140**).

Stephon approved the v0.6 promotion at 4:54 PM ET on Nov 11, 2026. v0.5 remains the rollback and is still listed in the demo checkpoint slider. Do not delete v0.5 or v0.6 files.

## Fastest rollback (one git step)

Revert the v0.6 promotion PR on `main` (the merge that pointed production at G2_31505). Vercel will redeploy https://playground-lm.vercel.app from `main`. The site will again default to v0.5 Chat (E31140). Fact lookup will be off because that checkpoint has no `fact_lookup: true` flag.

If the promotion landed as a single commit, `git revert` of that commit on `main` is enough.

## One-line manifest rollback (same effect, no full revert)

In `demo/manifest.json`:

1. Set `model_hosting.live_step` to `31140`.
2. Remove the step `31505` entry from `checkpoints` (or move it off the end — the demo defaults to the last checkpoint).
3. Leave the step `31140` entry as it is:

`https://huggingface.co/Gteam04/playground-lm-onnx/resolve/27778a045bee172529170330ada959131037da64/step_00031140.q8.onnx`

SHA256 `0fb5504d0e98e4eeb2032a3c450ffd2b0d996bb11c8d27912be1d79b821fdee8`.

Commit, merge to `main`, and wait for the production Vercel deploy.

## Temporary switch (no deploy)

On the live page, open **Advanced** and move the checkpoint slider back to **v0.5 Chat (Stage E)**. That only lasts for that browser session and does not change what new visitors get.

## What must stay in place

- Hugging Face `Gteam04/playground-lm-onnx` root file `step_00031140.q8.onnx` (revision `27778a045bee172529170330ada959131037da64`) — do not overwrite or delete.
- Stage D `step_00031104.q8.onnx` (revision `9732b2c8318190b43b5f97c9f7299559efd46b26`) remains a further rollback behind v0.5.
- The v0.6 live copy is `live/v0.6_g2_31505/step_00031505_g2.q8.onnx` at revision `d9e1629d821e7a9550fac9c54c56b132cd9327b5`. The preview copy under `previews/stage_g_g2_31505/` is also left in place.

Never delete model files without Stephon's explicit yes.
