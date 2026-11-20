"""
Stanza vs. spaCy vs. simplemma: lemma quality and speed for pipeline_pre_ppmi.py.

Run from the repo root (simplemma is not a project dependency, install it outside the venv):
    uv pip install --python .venv/bin/python --target <dir> simplemma
    PYTHONPATH=<dir> .venv/bin/python internal/lemmatizer-comparison/compare.py [n_sample]

Writes to internal/lemmatizer-comparison/results/:
    hard_cases.csv        curated hard cases (testset.py) with the lemma of every system
    speed.json            run times on the corpus sample
    agreement.json        lemma agreement per POS (Stanza tokens), POS/gender agreement Stanza-spaCy
    diff_types.csv        content-word types where the three lemmas are not all equal, with frequency
Scoring against the hand-made gold lemmas: score.py.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import polars as pl
import simplemma
import spacy
import stanza
from spacy.tokens import Doc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from testset import CASES  # noqa: E402

REPO = HERE.parents[1]
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
CONTENT = {"NOUN", "PROPN", "ADJ", "VERB"}

# Stanza: identical to the pipeline (pipeline_pre_ppmi.py, nlp_st2)
nlp_st = stanza.Pipeline(
    "de",
    processors="tokenize,mwt,pos,lemma",
    tokenize_no_ssplit=True,
    dir=str(REPO / "assets" / "stanza_resources"),
    download_method=None,
    logging_level="WARN",
)
# spaCy: as in the (deactivated) spaCy cell of the pipeline
SPACY_DISABLE = ["parser", "ner", "senter"]
nlp_sp = spacy.load(REPO / "assets" / "spacy" / "de_core_news_md")


def sm(word, greedy=False):
    return simplemma.lemmatize(word, lang="de", greedy=greedy)


def st_gender(word):
    for feat in (word.feats or "").split("|"):
        if feat.startswith("Gender="):
            return feat[7:]
    return None


def sp_gender(tok):
    g = tok.morph.get("Gender")
    return g[0] if g else None


def spacy_on_words(words):
    """Run spaCy on a pre-tokenized word list (Stanza's words), so both see the same tokens."""
    doc = Doc(nlp_sp.vocab, words=words)
    for name, proc in nlp_sp.pipeline:
        if name not in SPACY_DISABLE:
            doc = proc(doc)
    return doc


# ---------- A: curated hard cases ----------
rows = []
st_docs = nlp_st([stanza.Document([], text=s) for _, s, _, _ in CASES])
sp_docs = list(nlp_sp.pipe([s for _, s, _, _ in CASES], disable=SPACY_DISABLE))
for (cat, sent, target, gold), sd, pd in zip(CASES, st_docs, sp_docs):
    start = sent.index(target)
    end = start + len(target)
    # all words inside the target span; a split token gets its piece lemmas joined with "+"
    st_words = [w for t in sd.sentences[0].tokens if t.start_char >= start and t.end_char <= end for w in t.words]
    sp_toks = [t for t in pd if t.idx >= start and t.idx + len(t) <= end]
    r = dict(
        cat=cat, sentence=sent, target=target, gold="|".join(gold),
        stanza="+".join(w.lemma or w.text for w in st_words),
        stanza_pos="+".join(w.upos for w in st_words),
        spacy="+".join(t.lemma_ for t in sp_toks),
        spacy_pos="+".join(t.pos_ for t in sp_toks),
        simplemma=sm(target),
        simplemma_greedy=sm(target, True),
    )
    for k in ("stanza", "spacy", "simplemma", "simplemma_greedy"):
        r[f"ok_{k}"] = r[k] in gold
    rows.append(r)
hard = pl.DataFrame(rows)
hard.write_csv(OUT / "hard_cases.csv")
ok_cols = [c for c in hard.columns if c.startswith("ok_")]
with pl.Config(tbl_rows=50):
    print(hard.group_by("cat", maintain_order=True).agg(pl.len().alias("n"), *[pl.col(c).sum() for c in ok_cols]))
    print(hard.select(pl.len().alias("n"), *[pl.col(c).sum() for c in ok_cols]))

# ---------- B: corpus sample ----------
kwic = pl.read_csv(
    REPO / "stages/01_preprocessing/output/punkt/kwic_hits_bundestag_punkt.csv", infer_schema_length=0
)
sample = kwic.sample(N_SAMPLE, seed=42)
texts = [
    f"{(b or '').strip()} {(t or '').strip()} {(a or '').strip()}"
    for b, t, a in sample.select("context_before", "target_form", "context_after").iter_rows()
]

# speed, each system end to end on the raw texts (own tokenizer)
t0 = time.perf_counter()
st_docs = []
for i in range(0, len(texts), 500):  # chunks of 500 documents, as in lemmatize_batch
    st_docs += nlp_st([stanza.Document([], text=x) for x in texts[i : i + 500]])
t_stanza = time.perf_counter() - t0

t0 = time.perf_counter()
_ = [[t.lemma_ for t in d] for d in nlp_sp.pipe(texts, disable=SPACY_DISABLE)]
t_spacy = time.perf_counter() - t0

t0 = time.perf_counter()
_ = [list(simplemma.text_lemmatizer(x, lang="de")) for x in texts]
t_sm = time.perf_counter() - t0

n_words = sum(len(s.words) for d in st_docs for s in d.sentences)
speed = dict(
    n_texts=len(texts),
    n_chars=sum(map(len, texts)),
    n_words_stanza=n_words,
    seconds={"stanza": round(t_stanza, 2), "spacy": round(t_spacy, 2), "simplemma": round(t_sm, 3)},
    words_per_second={k: round(n_words / v) for k, v in [("stanza", t_stanza), ("spacy", t_spacy), ("simplemma", t_sm)]},
    speedup_vs_stanza={"spacy": round(t_stanza / t_spacy, 1), "simplemma": round(t_stanza / t_sm, 1)},
    note="CPU only, torch threads=%d; spaCy single process" % __import__("torch").get_num_threads(),
)
print(json.dumps(speed, indent=1))
(OUT / "speed.json").write_text(json.dumps(speed, indent=1))

# per word on Stanza's tokenization (isolates lemmatizer quality from tokenization);
# punctuation is skipped, as in the pipeline
lem_agree = {k: Counter() for k in ("stanza=spacy", "stanza=simplemma", "spacy=simplemma", "all3")}
total = Counter()
pos_agree, gen_agree, gen_total = Counter(), 0, 0
types = Counter()
for d in st_docs:
    for s in d.sentences:
        words = [w for t in s.tokens for w in t.words]
        spd = spacy_on_words([w.text for w in words])
        for w, t in zip(words, spd):
            if w.upos == "PUNCT":
                continue
            a, b, c = w.lemma or w.text, t.lemma_, sm(w.text)
            total[w.upos] += 1
            for key, eq in (("stanza=spacy", a == b), ("stanza=simplemma", a == c), ("spacy=simplemma", b == c), ("all3", a == b == c)):
                lem_agree[key][w.upos] += eq
            pos_agree[w.upos] += w.upos == t.pos_
            if w.upos == "NOUN":
                gen_total += 1
                gen_agree += st_gender(w) == sp_gender(t)
            if w.upos in CONTENT and not a == b == c:
                types[(w.text, w.upos, a, b, c)] += 1


def share(counter, keys):
    a = sum(counter[k] for k in keys)
    n = sum(total[k] for k in keys)
    return {"agree": a, "n": n, "share": round(a / n, 4)}


all_pos = list(total)
agreement = {
    "lemma": {
        key: {"all": share(c, all_pos), "content": share(c, CONTENT), **{p: share(c, [p]) for p, _ in total.most_common()}}
        for key, c in lem_agree.items()
    },
    "pos_stanza_vs_spacy": {"all": share(pos_agree, all_pos), "content": share(pos_agree, CONTENT)},
    "gender_nouns_stanza_vs_spacy": {"agree": gen_agree, "n": gen_total, "share": round(gen_agree / gen_total, 4)},
}
(OUT / "agreement.json").write_text(json.dumps(agreement, indent=1, ensure_ascii=False))
for key in lem_agree:
    x, y = agreement["lemma"][key]["all"], agreement["lemma"][key]["content"]
    print(f"Lemma {key:18s} alle {x['share']:6.1%}   Inhaltswörter {y['share']:6.1%}")
print("POS Stanza=spaCy:", agreement["pos_stanza_vs_spacy"])
print("Genus Nomen Stanza=spaCy:", agreement["gender_nouns_stanza_vs_spacy"])

diff_types = pl.DataFrame(
    [(*k, v) for k, v in types.most_common()],
    schema=["form", "upos", "stanza", "spacy", "simplemma", "n"],
    orient="row",
)
diff_types.write_csv(OUT / "diff_types.csv")
print("content-word types with disagreement:", diff_types.height, "tokens:", diff_types["n"].sum())
