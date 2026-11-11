"""Deterministic question phrasings for fact-sheet entries (Stage G, Lumi glue).

Nova's fact sheet v1 has no ``questions`` field and Claude's grounded builder skips
entries without one. This writer adds 2-4 rule-based phrasings per fact so the builder
can run (``build_stage_g_grounded.py --auto-questions``). No model output, no external
text: templates for the four table-generated shapes (capitals, continents, element
symbols, month lengths), a "What is the ...?" form for "<answer> is the ..." sentences,
and fill-in-the-blank forms for everything. The fact sheet file itself is unchanged.

Fresh private-suite questions must be written independently (G5 checks exact overlap).
"""
from __future__ import annotations

import re

_TEMPLATES = [
    (re.compile(r"^(?P<a>.+?) is the (?:official |administrative |constitutional )?capital of (?P<x>.+)\.$"),
     ["What is the capital of {x}?", "Which city is the capital of {x}?", "Name the capital of {x}."]),
    (re.compile(r"^(?P<x>.+?) is a country in (?P<a>.+)\.$"),
     ["Which continent is {x} in?", "On what continent is {x}?"]),
    (re.compile(r"^The chemical symbol for (?P<x>.+?) is (?P<a>.+)\.$"),
     ["What is the chemical symbol for {x}?", "Which symbol stands for {x}?"]),
    (re.compile(r"^(?P<x>[A-Z][a-z]+) has (?P<a>\d+) days\.$"),
     ["How many days does {x} have?", "How many days are in {x}?"]),
]


def _blank(text: str, alias: str) -> str | None:
    m = re.search(re.escape(alias), text, flags=re.IGNORECASE)
    if not m:
        return None
    return text[: m.start()] + "___" + text[m.end():]


def questions_for(fact: dict) -> list[str]:
    text, answer = fact["text"].strip(), fact["aliases"][0]
    qs: list[str] = []
    for pat, forms in _TEMPLATES:
        m = pat.match(text)
        if m and m.group("a").strip().casefold() == answer.strip().casefold():
            qs += [f.format(x=m.group("x")) for f in forms]
            break
    if not qs:
        m = re.match(rf"^{re.escape(answer)} is the (.+)\.$", text, flags=re.IGNORECASE)
        if m:
            qs.append(f"What is the {m.group(1)}?")
    blank = _blank(text, answer)
    if blank:
        qs += [f"Fill in the blank: {blank}", f"Complete the sentence: {blank}"]
    seen, out = set(), []
    for q in qs:
        if q.casefold() not in seen:
            seen.add(q.casefold())
            out.append(q)
    return out[:4]
