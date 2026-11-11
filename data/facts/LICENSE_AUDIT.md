# Fact sheet v1 — license & provenance audit (Nova, Stage G)

**File:** `data/facts/fact_sheet.jsonl` — 526 entries, frozen by SHA-256
(see the Stage G thread / PR description for the v1 hash).
**License of the sheet itself:** MIT (this repository). Every entry is
marked `"license": "MIT (this repository)"`.

## Method

- Every fact is **one short sentence written for this repository** by
  Nova (`generate_fact_sheet.py` holds the source tables and emits the
  JSONL deterministically). Facts themselves are not copyrightable
  (facts/ideas are outside copyright; only expression is protected), and
  the expression here is original — **no sentence was copied** from any
  reference page, dataset, or book.
- Facts were **checked** (not copied) against the reference pages below
  on 2026-11-10. Each entry's `verify_url` names the page its topic was
  checked against.
- **Stable facts only:** no current office holders, no population or
  sports records that change, no opinions. Deliberately excluded as
  unstable or disputed: Israel/Palestine capitals, Yemen's capital
  (civil war), Sri Lanka's dual capitals, Nauru (no official capital).
  Multi-capital states are phrased precisely (Netherlands: Amsterdam
  official; Bolivia: Sucre constitutional / La Paz seat of government;
  South Africa: Pretoria administrative; Switzerland: Bern de facto).
- Contested superlatives are hedged in the text itself (e.g. the Nile is
  "*generally considered* the longest river").

## Sources checked (reference only — nothing scraped or copied)

| Topic | Page checked |
|---|---|
| World capitals | https://simple.wikipedia.org/wiki/List_of_national_capitals |
| U.S. state capitals | https://simple.wikipedia.org/wiki/List_of_U.S._state_capitals |
| Planets / solar system | https://science.nasa.gov/solar-system/planet-sizes-and-locations-in-our-solar-system/ (NASA) |
| Element symbols | https://simple.wikipedia.org/wiki/List_of_elements_by_symbol |
| Geography superlatives | https://dailygk.indgovtjobs.net/gk/geography/world-geography-basics/ |
| Animals / human body | https://www.syvum.com/cgi/online/serve.cgi/quiz/livingworld.html?list |
| Calendar / history / language | Definitional and standard textbook facts (Gregorian calendar conventions, widely published historical dates, dictionary-level language facts), phrased originally; spot items cross-checked in the same sessions as the pages above. |
| PlaygroundLM self-facts | The project record (this repository: 124M config, 12 layers / 12 heads / width 768 / 1024 context, trained from scratch by the Galactic AI Team; demo at playground-lm.vercel.app). |

## Confirmations

- **No CC0 / CC-BY text is used**, so no attribution file is required
  and `data/facts/ATTRIBUTION.md` is intentionally absent (the plan
  requires it only if CC0/CC-BY text were included).
- **No NonCommercial material. No share-alike material.** Nothing from
  TriviaQA, SQuAD, Wikipedia article text, or any dataset dump was
  packed into the sheet; Wikipedia/NASA pages were read as references
  for verification only.
- **No scraped paragraphs.** The generator contains only short curated
  data points (country→capital pairs, element→symbol pairs, dates) and
  original sentence templates.
- **Held-out split:** ~20% of entries per topic are `heldout: true`
  (102/526 = 19.4%), chosen deterministically (every 5th entry within
  each topic) so the holdout spreads across all topics. Held-out facts
  are never trained as answers; they test reading from context.
