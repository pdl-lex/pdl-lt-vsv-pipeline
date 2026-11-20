"""
Speedtest für die VSV-Pipeline: misst die rechenintensiven Schritte der Notebooks
auf einem festen Datenauszug, damit verschiedene Rechner vergleichbar sind.

Schritte (Reihenfolge wie in der Pipeline):
  1 kwic_window   KWIC-Extraktion, festes Fenster (20 Wörter)  -> reine Extraktion + CSV-Lesen
  2 kwic_punkt    KWIC-Extraktion mit Satzgrenzen (NLTK Punkt) -> Differenz zu 1 = Satzgrenzen
  3 stanza_load   Laden der Stanza-Pipeline (Plattenzugriff, Modellinitialisierung)
  4 lemmatize     Lemmatisierung der Kontexte mit Stanza (inkl. Kontext-Mapping)
  5 ppmi          PPMI aus den Kookkurrenzen (polars) und Aufbau der dünnen Matrix
  6 svd           abgeschnittene SVD (scipy svds, k = 300)

Aufruf (aus dem Repo-Wurzelverzeichnis):
  uv run python internal/speedtest/speedtest.py --prepare   # einmal: Datenauszug erzeugen
  uv run python internal/speedtest/speedtest.py             # Test laufen lassen

Die Funktionen kommen, wo möglich, direkt aus den Notebooks (@app.function). Was in den
Notebooks nur in Zellen steht, ist unten als Kopie markiert ("KOPIE aus ...").
"""

import argparse
import csv
import json
import os
import platform
import random
import shutil
import socket
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"
RESULTS = HERE / "results"
ASSETS = REPO / "assets"
PRE_DIR = REPO / "stages" / "01_preprocessing"

# Umfang des Tests. Auf dem Referenzrechner (Ryzen AI Max+ 395, 32 Threads) ergibt das ca. 3 min.
# Ändert man diese Werte, sind die Ergebnisse nicht mehr mit älteren Läufen vergleichbar ->
# DATA_VERSION hochzählen.
DATA_VERSION = 1
CORPUS_EVERY_NTH = 2  # jede 2. Rede des Bundestagskorpus (~39.000 Reden, ~190 MB)
N_LEMMA_HITS = 12_000  # zufällige KWIC-Treffer (Seed 0) für die Lemmatisierung
SVD_K = 300  # wie pipeline_emb_ppmi.py
SEED = 0

SRC_CORPUS = PRE_DIR / "input" / "CPP-BT_2026-01-17_DE_CSV_Reden_Gesamt.csv"
SRC_KWIC = PRE_DIR / "output" / "punkt" / "kwic_hits_bundestag_punkt.csv"
SRC_COOC = REPO / "stages" / "02_embedding" / "input" / "punkt" / "cooc_counts_bundestag_punkt_halfyear.csv"
SRC_TARGETS = PRE_DIR / "input" / "target_forms.csv"

# Notebook-Funktionen importierbar machen (marimo-Notebooks sind normale Python-Module)
sys.path.insert(0, str(PRE_DIR))


# ---------------------------------------------------------------------------------------------
# Datenauszug
# ---------------------------------------------------------------------------------------------
def prepare():
    import polars as pl

    for p in (SRC_CORPUS, SRC_KWIC, SRC_COOC, SRC_TARGETS):
        if not p.is_file():
            sys.exit(f"Quelldatei fehlt: {p}\n(--prepare nur auf einem Rechner mit vollständigen Pipeline-Daten)")
    DATA.mkdir(parents=True, exist_ok=True)

    # 1) Korpus: jede n-te Rede, nur die Spalten, die die KWIC-Extraktion liest; als CSV, weil die
    #    Pipeline das Korpus als CSV zeilenweise liest (read_corpus_rows)
    print("Korpusauszug ...", flush=True)
    csv.field_size_limit(2**31 - 1)
    keep = ["rede_id", "sitzung_datum", "rede_text", "redner_fraktion"]
    n_in = n_out = 0
    with open(SRC_CORPUS, encoding="utf-8-sig", newline="") as fin, open(
        DATA / "corpus.csv", "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        header = next(reader)
        idx = [header.index(c) for c in keep]
        writer = csv.writer(fout)
        writer.writerow(keep)
        for row in reader:
            if n_in % CORPUS_EVERY_NTH == 0:
                writer.writerow([row[i] for i in idx])
                n_out += 1
            n_in += 1
    print(f"  {n_out:,} von {n_in:,} Reden")

    # 2) KWIC-Treffer für die Lemmatisierung: Zufallsstichprobe über das ganze Korpus
    print("KWIC-Auszug ...", flush=True)
    hits = pl.read_csv(SRC_KWIC, null_values="NA")
    hits.sample(n=N_LEMMA_HITS, seed=SEED).write_parquet(DATA / "kwic_hits.parquet")
    print(f"  {N_LEMMA_HITS:,} von {hits.height:,} Treffern")

    # 3) Kookkurrenzen vollständig (die SVD hängt von der ganzen Matrix ab)
    print("Kookkurrenzen ...", flush=True)
    pl.read_csv(SRC_COOC, schema_overrides={"slice": pl.Utf8}).write_parquet(DATA / "cooc_counts.parquet")

    shutil.copy(SRC_TARGETS, DATA / "target_forms.csv")
    (DATA / "version.json").write_text(
        json.dumps({"data_version": DATA_VERSION, "corpus_every_nth": CORPUS_EVERY_NTH,
                    "n_lemma_hits": N_LEMMA_HITS, "created": datetime.now().isoformat(timespec="seconds")}),
        encoding="utf-8",
    )
    size = sum(p.stat().st_size for p in DATA.iterdir()) / 1e6
    print(f"Fertig: {DATA} ({size:.0f} MB)")


# ---------------------------------------------------------------------------------------------
# KOPIE aus pipeline_pre_ppmi.py (Zellen "strip_punctuation", "kwic_window", "kwic_sentence").
# Diese Funktionen stehen im Notebook in Zellen und sind deshalb nicht importierbar.
# ---------------------------------------------------------------------------------------------
def make_kwic_functions(target_forms):
    import re
    from bisect import bisect_left, bisect_right

    import nltk
    from nltk.tokenize import PunktTokenizer

    nltk.data.path.insert(0, str(ASSETS / "nltk_data"))
    punkt = PunktTokenizer("german")

    TOKEN = re.compile(r"\S+")
    collapse_whitespace = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731

    def strip_punctuation(word):
        return re.sub(r"^\W+|\W+$", "", word)

    def kwic_window(doc_id, text, window=20):
        words = text.split()
        hits = []
        for i, word_raw in enumerate(words):
            word = strip_punctuation(word_raw)
            if word in target_forms:
                hits.append({
                    "id": doc_id, "target_form": word, "position": i,
                    "context_before": " ".join(words[max(0, i - window): i]),
                    "context_after": " ".join(words[i + 1: i + 1 + window]),
                    "sentence_based": False,
                })
        return hits

    def punkt_spans(snip):
        return punkt.span_tokenize(snip)

    def kwic_sentence(doc_id, text, spans=punkt_spans, pad=600, max_words=20):
        toks = [m.span() for m in TOKEN.finditer(text)]
        starts = [s for s, _ in toks]
        ends = [e for _, e in toks]
        hits = []
        for i, (hs, he) in enumerate(toks):
            word = strip_punctuation(text[hs:he])
            if word not in target_forms:
                continue
            lo, hi = max(0, hs - pad), min(len(text), he + pad)
            snip = text[lo:hi]
            a, b = 0, len(text)
            sentence_based = False
            for s, e in spans(snip):
                if s <= hs - lo < e:
                    sentence_based = True
                    if not (s == 0 and lo > 0):
                        a = lo + s
                    if not (e == len(snip) and hi < len(text)):
                        b = lo + e
                    break
            j0 = max(bisect_left(starts, a), i - max_words)
            j1 = min(bisect_right(ends, b) - 1, i + max_words)
            hits.append({
                "id": doc_id, "target_form": word, "position": i,
                "context_before": collapse_whitespace(text[starts[j0]:hs]) if j0 < i else "",
                "context_after": collapse_whitespace(text[he:ends[j1]]) if j1 > i else "",
                "sentence_based": sentence_based,
            })
        return hits

    return kwic_window, kwic_sentence


def load_target_form_lemmas():
    # wie Zelle "target_form_lemmas" in pipeline_pre_ppmi.py
    from pipeline_pre_dataimport import normalize_chars

    form_lemmas = {}
    with open(DATA / "target_forms.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            form = normalize_chars(row["form"].strip())
            if form:
                form_lemmas[form] = normalize_chars(row["lemma"].strip())
    return form_lemmas


# ---------------------------------------------------------------------------------------------
# Testschritte
# ---------------------------------------------------------------------------------------------
def step_kwic(extract, out_path):
    from pipeline_pre_dataimport import normalize_chars
    from pipeline_pre_ppmi import read_corpus_rows, write_kwic_hits

    rows = read_corpus_rows(
        [DATA / "corpus.csv"], "rede_id", "sitzung_datum", "rede_text",
        normalize=normalize_chars, facet_cols=["redner_fraktion"],
    )
    return write_kwic_hits(rows, extract, out_path, facet_names=["facet_1"])


def step_stanza_load():
    import stanza

    # wie Zelle "nlp_st2" in pipeline_pre_ppmi.py
    return stanza.Pipeline(
        "de", processors="tokenize,mwt,pos,lemma", tokenize_no_ssplit=True,
        dir=str(ASSETS / "stanza_resources"), download_method=None, logging_level="WARN",
    )


def step_lemmatize(nlp, form_lemmas):
    import polars as pl
    from pipeline_pre_ppmi import lemmatize_batch, make_context_mapper

    hits = pl.read_parquet(DATA / "kwic_hits.parquet")
    mapper = make_context_mapper(form_lemmas, gender_rule=True)
    out = lemmatize_batch(hits, nlp, form_lemmas, context_mapper=mapper)
    return out.height


def step_ppmi():
    # KOPIE (verdichtet) aus pipeline_emb_ppmi.py, Abschnitt PPMI und Matrixaufbau
    import polars as pl
    from scipy.sparse import csr_matrix

    cooc = pl.read_parquet(DATA / "cooc_counts.parquet")
    t = cooc.group_by(["slice", "target_lemma"]).agg(pl.col("n_cooc").sum().alias("n_t"))
    c = cooc.group_by(["slice", "context_lemma"]).agg(pl.col("n_cooc").sum().alias("n_c"))
    s = cooc.group_by("slice").agg(pl.col("n_cooc").sum().alias("n_s"))
    ppmi = (
        cooc.join(t, on=["slice", "target_lemma"], how="left")
        .join(c, on=["slice", "context_lemma"], how="left")
        .join(s, on="slice", how="left")
        .with_columns((pl.col("n_t") * pl.col("n_c") / pl.col("n_s")).alias("expected"))
        .with_columns((pl.col("n_cooc") / pl.col("expected")).log(2).clip(lower_bound=0).alias("ppmi"))
        .filter(pl.col("ppmi") > 0)
    )
    rows = ppmi.select("target_lemma", "slice").unique().sort("target_lemma", "slice").with_row_index("r")
    cols = ppmi.select("context_lemma").unique().sort("context_lemma").with_row_index("c")
    ppmi = ppmi.join(rows, on=["target_lemma", "slice"]).join(cols, on="context_lemma")
    return csr_matrix(
        (ppmi["ppmi"].to_numpy(), (ppmi["r"].to_numpy(), ppmi["c"].to_numpy())),
        shape=(rows.height, cols.height),
    )


def step_svd(matrix):
    import numpy as np
    from scipy.sparse.linalg import svds

    U, S, Vt = svds(matrix, k=SVD_K, random_state=0)
    order = np.argsort(S)[::-1]
    return S[order]


# ---------------------------------------------------------------------------------------------
# Systeminfo
# ---------------------------------------------------------------------------------------------
def system_info():
    info = {
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": None,
    }
    try:  # Linux
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["ram_gb"] = round(int(line.split()[1]) / 1024**2, 1)
                break
    except OSError:
        if platform.system() == "Windows":
            try:
                import subprocess

                info["cpu"] = subprocess.run(
                    ["powershell", "-NoProfile", "(Get-CimInstance Win32_Processor).Name"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip() or info["cpu"]
            except Exception:
                pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:  # pragma: no cover
        info["torch"] = f"nicht verfügbar ({e})"
    try:
        import numpy as np

        blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
        info["blas"] = blas.get("name")
    except Exception:
        info["blas"] = None
    import polars as pl

    info["polars_threads"] = pl.thread_pool_size()
    return info


# ---------------------------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------------------------
def run(label):
    version_file = DATA / "version.json"
    if not version_file.is_file():
        sys.exit(f"Datenauszug fehlt ({DATA}). Erst mit --prepare erzeugen oder den Ordner data/ kopieren.")
    data_meta = json.loads(version_file.read_text(encoding="utf-8"))
    if data_meta["data_version"] != DATA_VERSION:
        sys.exit(f"Datenauszug hat Version {data_meta['data_version']}, Skript erwartet {DATA_VERSION}.")

    info = system_info()
    print(f"Rechner: {info['host']} | {info['cpu']} | {info['logical_cpus']} Threads | {info['ram_gb']} GB RAM")
    print(f"GPU (Stanza): {info.get('gpu') or 'keine, CPU'} | torch-Threads: {info.get('torch_threads')}")
    print()

    times, details = {}, {}

    def timed(name, title, fn):
        print(f"  {title:<42}", end="", flush=True)
        t0 = time.perf_counter()
        result = fn()
        times[name] = round(time.perf_counter() - t0, 2)
        print(f"{times[name]:8.1f} s", flush=True)
        return result

    t_start = time.perf_counter()
    form_lemmas = load_target_form_lemmas()
    kwic_window, kwic_sentence = make_kwic_functions(frozenset(form_lemmas))
    corpus_mb = (DATA / "corpus.csv").stat().st_size / 1e6

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        n_docs, n_hits = timed("kwic_window", "1 KWIC-Extraktion, festes Fenster",
                               lambda: step_kwic(kwic_window, tmp / "w.csv"))
        details["kwic_docs"], details["kwic_window_hits"] = n_docs, n_hits
        _, n_hits = timed("kwic_punkt", "2 KWIC-Extraktion, Satzgrenzen (Punkt)",
                          lambda: step_kwic(kwic_sentence, tmp / "p.csv"))
        details["kwic_punkt_hits"] = n_hits

    nlp = timed("stanza_load", "3 Stanza laden", step_stanza_load)
    n = timed("lemmatize", f"4 Lemmatisierung ({N_LEMMA_HITS:,} Treffer)", lambda: step_lemmatize(nlp, form_lemmas))
    details["lemma_hits"] = n
    del nlp
    matrix = timed("ppmi", "5 PPMI + dünne Matrix", step_ppmi)
    details["matrix_shape"], details["matrix_nnz"] = list(matrix.shape), int(matrix.nnz)
    S = timed("svd", f"6 SVD (k = {SVD_K})", lambda: step_svd(matrix))
    details["svd_s0"] = float(S[0])  # Plausibilitätskontrolle: muss auf allen Rechnern gleich sein
    total = round(time.perf_counter() - t_start, 2)

    # abgeleitete Kennzahlen und Hochrechnung auf den vollen Bundestagslauf
    rates = {
        "kwic_window_mb_s": round(corpus_mb / times["kwic_window"], 2),
        "kwic_punkt_mb_s": round(corpus_mb / times["kwic_punkt"], 2),
        "satzgrenzen_s": round(times["kwic_punkt"] - times["kwic_window"], 2),
        "lemma_hits_s": round(N_LEMMA_HITS / times["lemmatize"], 1),
    }
    full_hits = 692_136  # Bundestag, punkt (internal/Dauer der Rechenschritte)
    proj_lemma_min = full_hits / rates["lemma_hits_s"] / 60
    proj_kwic_min = times["kwic_punkt"] * CORPUS_EVERY_NTH / 60

    print(f"  {'Gesamt':<42}{total:8.1f} s\n")
    print(f"  KWIC (Punkt): {rates['kwic_punkt_mb_s']} MB/s, Satzgrenzen kosten {rates['satzgrenzen_s']} s")
    print(f"  Lemmatisierung: {rates['lemma_hits_s']} Treffer/s")
    print(f"  Hochrechnung Bundestag voll: KWIC ~{proj_kwic_min:.1f} min, Lemmatisierung ~{proj_lemma_min:.0f} min")
    print(f"  Kontrolle: {details['kwic_punkt_hits']:,} Treffer (Punkt), Matrix {details['matrix_shape']}, "
          f"S[0] = {details['svd_s0']:.4f}")

    result = {
        "label": label, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_version": DATA_VERSION, "system": info, "times_s": times, "total_s": total,
        "rates": rates, "details": details,
        "projection_bundestag_min": {"kwic_punkt": round(proj_kwic_min, 1), "lemmatize": round(proj_lemma_min, 1)},
    }
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f"{info['host']}_{label + '_' if label else ''}{stamp}.json"
    (RESULTS / name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # eine Zeile pro Lauf in results.csv, zum direkten Vergleich der Rechner
    summary = RESULTS / "results.csv"
    fields = ["timestamp", "host", "label", "cpu", "logical_cpus", "ram_gb", "gpu", "data_version",
              *times, "total_s", *rates]
    row = {"timestamp": result["timestamp"], "host": info["host"], "label": label, "cpu": info["cpu"],
           "logical_cpus": info["logical_cpus"], "ram_gb": info["ram_gb"], "gpu": info.get("gpu") or "",
           "data_version": DATA_VERSION, **times, "total_s": total, **rates}
    new = not summary.exists()
    with summary.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter=";")
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"\nGespeichert: {RESULTS / name}\n             {summary}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="Datenauszug aus den Pipeline-Daten erzeugen")
    ap.add_argument("--label", default="", help="Zusatz für den Ergebnisnamen, z. B. 'akku' oder 'gpu'")
    args = ap.parse_args()
    random.seed(SEED)
    if args.prepare:
        prepare()
    else:
        run(args.label)


if __name__ == "__main__":
    main()
