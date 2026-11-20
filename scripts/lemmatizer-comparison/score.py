"""Score Stanza, spaCy and simplemma against results/gold_corpus_types.csv (token-weighted)."""
import json
from pathlib import Path

import polars as pl

RES = Path(__file__).resolve().parent / "results"
SYSTEMS = ["stanza", "spacy", "simplemma"]

g = pl.read_csv(RES / "gold_corpus_types.csv").filter(pl.col("gold") != "?")


def correct(lemma, gold, tolerant):
    options = gold.split("|")
    if tolerant:  # old orthography (ß for ss) counts as correct
        lemma = lemma.replace("ß", "ss")
        options = [o.replace("ß", "ss") for o in options]
    return lemma in options


result = {"types_scored": g.height, "tokens_scored": int(g["n"].sum())}
for tolerant in (False, True):
    key = "orthography_tolerant" if tolerant else "strict"
    result[key] = {}
    for s in SYSTEMS:
        ok = [correct(l, gd, tolerant) for l, gd in zip(g[s], g["gold"])]
        tokens = int(g.filter(pl.Series(ok))["n"].sum())
        result[key][s] = {"tokens_correct": tokens, "share": round(tokens / result["tokens_scored"], 4)}

# error classes per system (strict)
def err_class(form, upos, lemma, gold):
    if correct(lemma, gold, True):
        return "alte Rechtschreibung"
    if lemma == form:
        return "unverändert gelassen"
    if upos == "NOUN" and lemma[:1].islower():
        return "Nomen klein geschrieben"
    return "sonstiger Fehler"


errs = {}
for s in SYSTEMS:
    rows = [
        (err_class(f, u, l, gd), n)
        for f, u, l, gd, n in zip(g["form"], g["upos"], g[s], g["gold"], g["n"])
        if not correct(l, gd, False)
    ]
    errs[s] = pl.DataFrame(rows, schema=["class", "n"], orient="row").group_by("class").agg(pl.col("n").sum()).sort("n", descending=True).to_dicts()
result["errors_strict_by_class"] = errs
(RES / "score.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
print(json.dumps(result, indent=1, ensure_ascii=False))
