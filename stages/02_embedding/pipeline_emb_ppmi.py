import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np

    return mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pipeline zur Erstellung von Bundle-Daten für VSV: PPMI-Embedding
    Dieses Notebook bietet einen vollständige, kommentierte Pipeline zur Vektorisierung von Daten mittels PPMI/SVD, ausgehend von zuvor erzeugten Kookkurenz-Daten hin zu dem für das "Visualisierung Semantischer Vektoren"-Tool nötigen Daten-Bundle.
    Als Beispiel-Datensatz werden die Reden-Texte der Bundestagssitzungen zwischen 2013 und 2026 genutzt.



    * Vorbereitung der Daten / Preprocessing (pipeline_pre.py)
    * **Embedding bzw. Vektorisierung (pipeline_emb.py)**
    * Nachbearbeitung und Generierung des Bundles (pipeline_post.py)

    Die hier verwendeten Daten werden aus dem Output-Verzeichnis des Preprocessing-Schritts geladen.

    Zusätzlich zu diesem Pipeline-Schritt gibt es eine interaktive Demo, die die Funktionsweise von PPMI detailliert erläutert. Diese findet sich im Notebook *pipeline_emb_ppmi_demo.py*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Input

    Von den im Pre-Processing erzeugten Daten werden nur die folgenden benötigt:

    * **kwic_hits_sliced_[kwic_variant]_[slice_grid].parquet: Pro Zeitscheibe aggregierte Version der lemmatisierten KWIC.
    * cooc_counts_[kwic_variant]_[slice_grid]: Kookkurenzen für Slice x Zielwort x Kontextwort.

    Folgende Varianten stehen zur Verfügung:

    * KWIC mit einem Fenster von N Wörtern vor und nach der Fundstelle (Standard 20): "window20"
    * KWIC mit Fenster identisch zum Satz, geparst mit NLTK  Punkt: "punkt"
    * KWIC mit Fenster identisch zum Satz, geparst mit SoMaJo: "somajo"

    * Zeitscheibengröße Jahre: "year"
    * Zeitscheibengröße Halbjahre: "halfyear"
    * Zeitscheibengröße Quartale: "quarter"

    Standardmäßg werden im folgenden die Daten für "punkt" und "halfyear" verarbeitet, also die Kookkurenzen mit Stanza-Punkt-Lemmatisierung und halbjährlicher Auflösung. In der folgenden Zelle können diese Variablen angepasst werden.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PPMI - Positive Pointwise Mutual Information
    ### Vorbereitung
    """)
    return


@app.cell
def _():
    # input selection, same naming as the output files of pipeline_pre.py
    # both values are passed on to pipeline_post.py via output/emb_meta.json
    kwic_variant = "punkt"
    slice_grid = "halfyear"
    return kwic_variant, slice_grid


@app.cell
def _(kwic_variant, pl, slice_grid):
    # slice as string, otherwise year slices ("2018") are inferred as i64 and don't match the parquet
    cooc_counts = pl.read_csv(
        f"../01_preprocessing/output/{kwic_variant}/cooc_counts_{kwic_variant}_{slice_grid}.csv",
        schema_overrides={"slice": pl.Utf8},
    )
    cooc_counts
    return (cooc_counts,)


@app.cell
def _(kwic_variant, pl, slice_grid):
    # lazy: with large corpora the hits do not fit into memory comfortably; only the two
    # aggregations below (hits per row, dominant POS) read them, in streaming mode
    kwic_hits_sliced = pl.scan_parquet(f"../01_preprocessing/output/{kwic_variant}/kwic_hits_sliced_{kwic_variant}_{slice_grid}.parquet")
    kwic_hits_sliced.head(100).collect()
    return (kwic_hits_sliced,)


@app.cell
def _(cooc_counts, pl):
    # row total per target word and slice = sum of occurences via all context words
    target_slice_cooc_totals = (
        cooc_counts
        .group_by(["slice", "target_lemma"])
        .agg(
            pl.col("n_cooc").sum().alias("n_cooc_target_slice")
        )
    )

    # column total per context word and slice = sum of occurences via all target words
    context_slice_cooc_totals = (
        cooc_counts
        .group_by(["slice", "context_lemma"])
        .agg(
            pl.col("n_cooc").sum().alias("n_cooc_context_slice")
        )
    )

    # total per slice = sum of all cooccurrences in this slice
    slice_cooc_totals = (
        cooc_counts
        .group_by(["slice"])
        .agg(
            pl.col("n_cooc").sum().alias("n_cooc_slice")
        )
    )
    return (
        context_slice_cooc_totals,
        slice_cooc_totals,
        target_slice_cooc_totals,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Beispielausgaben**
    """)
    return


@app.cell
def _(target_slice_cooc_totals):
    target_slice_cooc_totals
    return


@app.cell
def _(context_slice_cooc_totals):
    context_slice_cooc_totals
    return


@app.cell
def _(slice_cooc_totals):
    slice_cooc_totals
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In der folgenden Zelle werden die errechneten Werte wieder an die Langform-Tabelle angehängt, ein Wert pro Zeile (=beobachteter Kookurenz pro Zeitscheibe).

    Der Schlüssel besteht aus "slice" und "target_lemma", bzw. "slice" und "context_lemma", die prinzipbedingt nur einmal auftauchen dürfen und damit eindeutige Schlüssel sind. Check: Die Zeilenzahl in cooc_counts und cooc_marginals muss identisch sein.
    """)
    return


@app.cell
def _(
    context_slice_cooc_totals,
    cooc_counts,
    slice_cooc_totals,
    target_slice_cooc_totals,
):
    # add the target-slice total
    cooc_marginals = cooc_counts.join(
        target_slice_cooc_totals,
        on=["slice", "target_lemma"],
        how="left",
    )

    # add the context-slice total
    cooc_marginals = cooc_marginals.join(
        context_slice_cooc_totals,
        on=["slice", "context_lemma"],
        how="left",
    )

    # add the slice total
    cooc_marginals = cooc_marginals.join(
        slice_cooc_totals,
        on=["slice"],
        how="left",
    )

    cooc_marginals
    return (cooc_marginals,)


@app.cell
def _(cooc_counts, cooc_marginals):
    len(cooc_marginals) == len(cooc_counts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Berechnung des PPMI-Wertes
    Nach all der Vorarbeit folgt hier nun endlich die Berechnung des PPMI-Wertes, in drei Schritten. Jeder Wert wird der Tabelle angehängt.
    * Erwartete Anzahl des gemeinsamen Auftauchens eines Wortpaares unter Unabhängigkeit
      $$
    \text{{erwartet}} = \frac{{\text{{Randsumme}}_{{\text{{Ziel}}}} \cdot \text{{Randsumme}}_{{\text{{Kontext}}}}}}{{\text{{Gesamt}}}}
    $$
    * PMI:
    $$
    \text{{PMI}} = \log_2 \frac{{\text{{beobachtet}}}}{{\text{{erwartet}}}}
    $$

    * PPMI

    $$
    \text{{PPMI}} = \max\left(0,\ \log_2 \frac{{\text{{beobachtet}}}}{{\text{{erwartet}}}}\right)
    $$
    """)
    return


@app.cell
def _(cooc_marginals, pl):
    # expected number of appearances
    cooc_ppmi = cooc_marginals.with_columns(
        (
            (pl.col("n_cooc_target_slice") * pl.col("n_cooc_context_slice"))
            / pl.col("n_cooc_slice")
        ).alias("expected")
    )

    # PMI: log of the ratio
    cooc_ppmi = cooc_ppmi.with_columns(
        (pl.col("n_cooc") / pl.col("expected")).log(2).alias("pmi")
    )

    # PPMI: clip PMI at 0, so only positive values are kept
    cooc_ppmi = cooc_ppmi.with_columns(
        pl.col("pmi").clip(lower_bound=0).alias("ppmi")
    )

    cooc_ppmi
    return (cooc_ppmi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation der Ergebnisse (PPMI)

    **Wie viele Beobachtungen braucht PPMI, um verlässlich zu sein?**

    Kommt ein Zielwort in einer Zeitscheibe nur sehr selten vor, beruht sein
    gesamtes PPMI-Profil auf entsprechend wenig Material – im Extremfall auf
    einem einzigen Satzkontext. Jedes Wort aus diesem einen Fenster wirkt dann
    "besonders eng verbunden" mit dem Zielwort, einfach weil keine zweite
    Beobachtung existiert, die das relativieren könnte. Die Rangliste sieht
    plausibel aus, spiegelt aber eher Zufall als ein wiederkehrendes
    sprachliches Muster wider.

    Ähnliches gilt umgekehrt für seltene Kontextwörter. Ein Wort, das insgesamt
    nur ein- oder zweimal in den Kontextfenstern eines Quartals auftaucht und
    dabei zufällig immer neben demselben Zielwort steht, bekommt einen hohen
    PPMI-Wert – unabhängig davon, ob dahinter eine echte inhaltliche Verbindung
    steckt oder schlicht Zufall (Eigennamen, Tippfehler, die Wortwahl einzelner
    Redner). Erst wenn ein Kontextwort öfter auftaucht, zeigt sich, ob es
    wirklich bevorzugt in der Nähe eines bestimmten Zielworts steht oder
    allgemein häufig vorkommt.

    Praktisch heißt das: PPMI-Werte sind erst ab einer gewissen Mindestanzahl
    an Beobachtungen aussagekräftig – sowohl beim Zielwort als auch bei den
    Kontextwörtern. Ein einzelner auffällig hoher Wert lohnt sich immer
    gegenzuchecken, wie viele tatsächliche Treffer dahinterstehen, bevor man
    ihn als inhaltlichen Befund interpretiert.

    ***Einmalige Kontextwörter (1 Vorkommen)***
    ppmi ist ein Verhältnis: log2(beobachtet / erwartet). Bei einem exklusiven Kontextwort ist randsumme_kontext == beobachtet, und dann kürzt sich beobachtet aus dem Bruch heraus. Übrig bleibt log2(gesamt / randsumme_ziel) — eine Konstante, die nur noch vom Zielwort abhängt. Deshalb sind z.B. 12 Treffer und 7 Treffer nicht mehr unterscheidbar.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SVD - Singular Value Decomposition
    In den folgenden Zellen wird die Langform einer SVD unterzogen - die Langform stellt dabei bereits die Matrix dar.

    Bis hierher ist jedes Zielwort in jeder Zeitscheibe durch eine *Liste* von  Kontextwörtern beschrieben. Die SVD macht daraus einen *Vektor*: sie sucht die 100 Achsen, entlang derer sich die Zeilen am stärksten unterscheiden,    und drückt jede Zeile in diesen 100 Koordinaten aus.
    Entscheidend ist, dass das **eine einzige** Zerlegung über alle Zeitscheiben gemeinsam ist. Eine SVD pro Quartal würde 50 zufällig gedrehte Räume liefern, in denen ein Vergleich *zwischen* Quartalen nichts bedeutet.
    """)
    return


@app.cell
def _(cooc_ppmi, pl):
    # filter out rows where PPMI is 0
    # those word combinations may still have occurred, but less frequent than expected
    ppmi_positive = cooc_ppmi.filter(pl.col("ppmi") > 0)
    print(f"{cooc_ppmi.shape[0] - ppmi_positive.shape[0]} Zeilen mit PPMI = 0 wurden entfernt.")
    ppmi_positive
    return (ppmi_positive,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Die SVD-Matrix ist eine Tabelle mit den Zielwörtern in den Zeilen, und den Kontextwörtern den Spalten - prinzipiell wie bisher auch. Da die Zielwörter pro Zeitscheibe betrachtet werden, ist eine Zeile genaugenommen als Paar aus Zielwort und Zeitscheibe definiert. In den Zellen der Tabelle stehen schließlich die PPMI-Werte.

    Anders gesagt: Die Langform wird in einer Kreutabelle überführt.
    """)
    return


@app.cell
def _(ppmi_positive):
    # get unique target_lemma+slice combinations, sort, and add index
    target_slices = (
        ppmi_positive
        .select(["target_lemma", "slice"])
        .unique()
        .sort(["target_lemma", "slice"])
        .with_row_index("target_slice_idx")
    )

    target_slices
    return (target_slices,)


@app.cell
def _(ppmi_positive):
    # get unique context words, sort, add index
    context_words = (
        ppmi_positive
        .select("context_lemma")
        .unique()
        .sort("context_lemma")
        .with_row_index("context_idx")
    )
    context_words
    return (context_words,)


@app.cell
def _(context_words, ppmi_positive, target_slices):
    # in the long form table, add the indices to each occurrence
    ppmi_positive_indexed = ppmi_positive.join(
        target_slices,
        on=["target_lemma", "slice"],
        how="left",
    )

    ppmi_positive_indexed = ppmi_positive_indexed.join(
        context_words,
        on="context_lemma",
        how="left",
    )
    ppmi_positive_indexed
    return (ppmi_positive_indexed,)


@app.cell
def _(context_words, np, ppmi_positive_indexed, target_slices):
    # initialize matrix with correct dimensions, shortened to float32

    ppmi_matrix = np.zeros(
        (len(target_slices), len(context_words)),
        dtype=np.float32,
    )

    ppmi_matrix[
        ppmi_positive_indexed["target_slice_idx"].to_numpy(),
        ppmi_positive_indexed["context_idx"].to_numpy(),
    ] = ppmi_positive_indexed["ppmi"].to_numpy()

    print(f"Matrix im Speicher: {ppmi_matrix.nbytes / 1e6:.0f} MB")
    ppmi_matrix
    return (ppmi_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Optional: SVD nur an ausreichend belegten Zeilen**

    Mit `MIN_HITS_FIT` lässt sich festlegen, dass nur Zeilen mit mindestens so vielen Treffern (`n_hits`) die SVD bestimmen. Die übrigen Zeilen werden anschließend per *folding-in* in denselben Raum projiziert (Deerwester et al. 1990; Berry, Dumais & O'Brien 1995). Sie bleiben also erhalten, bestimmen die Achsen aber nicht mit, und sind in der Ausgabe mit `is_fitted = false` markiert.

    Voreingestellt ist `None`: Alle Zeilen gehen in die SVD ein. Ob der Schalter für ein Korpus sinnvoll ist und welcher Wert, zeigt `pipeline_emb_ppmi_diagnose.py` (Abschnitt 1: Schwelle; Abschnitt 2: ob der Fit ohne diese Zeilen den Raum verbessert). Nur einsetzen, wenn Abschnitt 2 für den gewählten Wert eine Verbesserung zeigt.
    """)
    return


@app.cell
def _(kwic_hits_sliced, np, pl, target_slices):
    # None = all rows take part in the SVD (default); a number = only rows with at least this many hits
    MIN_HITS_FIT = None

    # n_hits = number of KWIC hits per target word and slice, counted before explode
    target_slice_hits = (
        kwic_hits_sliced
        .group_by(["slice", "target_lemma"])
        .len()
        .rename({"len": "n_hits"})
        .collect(engine="streaming")
    )
    _row_hits = (
        target_slices
        .join(target_slice_hits, on=["slice", "target_lemma"], how="left")
        .sort("target_slice_idx")["n_hits"]
        .to_numpy()
    )
    is_fitted = (
        np.ones(len(_row_hits), dtype=bool) if MIN_HITS_FIT is None else _row_hits >= MIN_HITS_FIT
    )
    print(f"Zeilen in der SVD: {is_fitted.sum()} von {len(is_fitted)}")
    return MIN_HITS_FIT, is_fitted, target_slice_hits


@app.cell
def _(is_fitted, np, ppmi_matrix):
    # Singular Value Decomposition
    #  M = U * S * Vt
    # full_matrices=False: numpy only calculates min(rows, columns) dimensions instead of columns x columns for V
    # with MIN_HITS_FIT set, only the fitted rows enter the decomposition (U then has only those rows)

    U, S, Vt = np.linalg.svd(ppmi_matrix[is_fitted], full_matrices=False)

    print("U :", U.shape)   # rows in new space
    print("S :", S.shape)   # relevance of each axis
    print("Vt:", Vt.shape)  # contextwords, needed for anchors
    return S, U, Vt


@app.cell
def _(mo):
    mo.md(r"""
    **Auf k Dimensionen kürzen**

    Die vollständige SVD ist eine verlustfreie Zerlegung. Die Reduktion entsteht erst durch die Beschränkung auf die
    k größten Singulärwerte.
    Neben k wird der Exponent p der Eigenwertgewichtung festgelegt (Zeilenvektor = U_k · Σ_k^p; Caron 2001;
    Levy, Goldberg & Dagan 2015). p = 1 gewichtet jede Dimension mit ihrem vollen Singulärwert, kleinere Werte
    verringern das Gewicht der führenden Dimensionen gegenüber den übrigen.

    Geeignete Werte für k und p hängen von den Daten ab, etwa von der Zahl der Zielwörter, der Größe der Zeitscheiben
    und der Belegung der Zeilen. Sie sollten für jedes Korpus und jedes Zeitraster gemeinsam geprüft werden.
    Aufschluss darüber gibt das Notebook `pipeline_emb_ppmi_diagnose.py` (Abschnitt "Rang k der SVD und Exponent p
    der Eigenwertgewichtung").

    Die in der folgenden Zelle gesetzten Werte (k = 300, p = 0,5) sind mit Abschnitt 3 des Diagnose-Notebooks
    auf punkt/halfyear bestimmt (2026-11-17, Methodology 1.8). Für ein anderes Korpus, Raster oder eine andere
    KWIC-Variante sind sie neu zu bestimmen.
    """)
    return


@app.cell
def _(MIN_HITS_FIT, S, U, Vt, ppmi_matrix):
    # determined with pipeline_emb_ppmi_diagnose.py, section 3 (punkt/halfyear, 2026-11-17); see
    # "internal/Readme Scientific Methodology.md", 1.8
    k = 300
    eig_p = 0.5

    # row vector = U_k * S_k**p (eigenvalue weighting, Caron 2001)
    # p = 1 weights each dimension by its full singular value; p = 0.5 reduces the weight
    # of the leading dimensions relative to the others
    if MIN_HITS_FIT is None:
        target_slice_vectors = U[:, :k] * S[:k] ** eig_p
    else:
        # folding-in for all rows: m V_k S_k**(p-1); identical to U_k S_k**p for the fitted rows
        target_slice_vectors = (ppmi_matrix @ Vt[:k].T) * S[:k] ** (eig_p - 1)
    return eig_p, k, target_slice_vectors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Output

    Für das Postprocessing werden die unten genannten Dateien übergeben. Da die Numpy-Matrizen nur Vektoren als Zahlenwerte enthalten, sind in den Parquet-Dateien die Zeilen- bzw. Spaltenbeschriftungen sowie weitere Daten (<> Vektoren) enthalten.

    * Zielwörter
     * **target_slice_vectors_[kwic_variant]_[slice_grid].npy**: Zielwortvektoren pro Zeitscheibe (Zeile i gehört zu target_slice_idx in  target_slices.parquet)
     * **target_slices_[kwic_variant]_[slice_grid].parquet**: target_slice_idx, Zielwort, Zeitscheibe, die Trefferzahl (n_hits) je Zeile und ob die Zeile in die SVD eingegangen ist (is_fitted)

    * Kontextwörter
     * **context_vectors_[kwic_variant]_[slice_grid].npy**: Kontextwort-Vektoren (Zeile j gehört zu context_idx in context_words.parquet)
     * **context_words_[kwic_variant]_[slice_grid].parquet**: context_idx, Kontextwort, Belegung (in wie vielen Zeilen PPMI > 0, Summe der Kookkurrenzen) und häufigste Wortart
    * Metadaten
     * **emb_meta_[kwic_variant]_[slice_grid].json**: Quelldatei, Zeitraster, k, p, der Frobenius-Anteil, die Mindesttrefferzahl für die SVD (min_hits_fit, null = alle Zeilen) und die Zahl der Zeilen in der SVD

    U, S und die vollständige Vt werden nicht gespeichert. Insbesondere Vt wäre mit (Zeilen × Kontextwörter) sehr groß, genutzt werden aber nur die ersten k Zeilen.
    """)
    return


@app.cell
def _(kwic_hits_sliced, pl, ppmi_positive_indexed):
    # occupancy per context word: in how many matrix rows does it have PPMI > 0,
    # and how many cooccurrences does that amount to
    # used in pipeline_post.py as minimum occupancy filter for anchors and neighbors
    context_counts = ppmi_positive_indexed.group_by("context_idx").agg(
        [
            pl.col("target_slice_idx").n_unique().alias("n_target_slices"),
            pl.col("n_cooc").sum().alias("n_cooc"),
        ]
    )

    # dominant POS per lemma
    # a lemma can carry several tags ("gut" as ADV and ADJ); only the more frequent one is kept
    # deliberately not via .sort().unique(keep="first"), whose order is not guaranteed
    # ties (e.g. 3x NOUN, 3x PROPN) are broken alphabetically by POS via the second sort key,
    # otherwise the result would depend on the row order of group_by, which polars does not guarantee
    # the counts per (lemma, POS) are made in streaming mode: the exploded list (one row per
    # context word of every hit) never exists as a whole; the result is small
    context_pos_dominant = (
        kwic_hits_sliced
        .select(["context_lemmas", "context_pos_tags"])
        .explode(["context_lemmas", "context_pos_tags"], empty_as_null=True)
        .rename({"context_lemmas": "context_lemma", "context_pos_tags": "pos"})
        .group_by(["context_lemma", "pos"])
        .len()
        .collect(engine="streaming")
        .group_by("context_lemma")
        .agg(pl.col("pos").sort_by(["len", "pos"], descending=[True, False]).first())
    )
    return context_counts, context_pos_dominant


@app.cell
def _(
    MIN_HITS_FIT,
    Vt,
    context_counts,
    context_pos_dominant,
    context_words,
    eig_p,
    is_fitted,
    k,
    kwic_variant,
    np,
    pl,
    retained_frobenius_sq,
    slice_grid,
    target_slice_hits,
    target_slice_vectors,
    target_slices,
):
    import json
    from pathlib import Path

    # context word vectors from the column side of the SVD, one row per context word
    # unweighted, i.e. without S**eig_p (open question, see "internal/Readme Scientific Methodology.md", 1.9)
    context_vectors = Vt[:k].T

    # joins give no order guarantee, so restore matrix order explicitly
    # is_fitted: row took part in the SVD (false only with MIN_HITS_FIT set, see above)
    target_slices_export = (
        target_slices
        .join(target_slice_hits, on=["slice", "target_lemma"], how="left")
        .sort("target_slice_idx")
        .with_columns(pl.Series("is_fitted", is_fitted))
    )

    context_words_export = (
        context_words
        .join(context_counts, on="context_idx", how="left")
        .join(context_pos_dominant, on="context_lemma", how="left")
        .sort("context_idx")
    )

    # position == index, otherwise pipeline_post.py would attach vectors to wrong labels
    assert (target_slices_export["target_slice_idx"].to_numpy() == np.arange(target_slice_vectors.shape[0])).all()
    assert (context_words_export["context_idx"].to_numpy() == np.arange(context_vectors.shape[0])).all()
    # every matrix row has KWIC hits; null would point to a broken join
    assert target_slices_export["n_hits"].null_count() == 0

    # create folder if necessary
    Path(f"output/{kwic_variant}").mkdir(parents=True, exist_ok=True)


    np.save(f"output/{kwic_variant}/target_slice_vectors_{kwic_variant}_{slice_grid}.npy", target_slice_vectors)
    np.save(f"output/{kwic_variant}/context_vectors_{kwic_variant}_{slice_grid}.npy", context_vectors)
    target_slices_export.write_parquet(f"output/{kwic_variant}/target_slices_{kwic_variant}_{slice_grid}.parquet")
    context_words_export.write_parquet(f"output/{kwic_variant}/context_words_{kwic_variant}_{slice_grid}.parquet")

    emb_meta = {
        "kwic_variant": kwic_variant,
        "slice_grid": slice_grid,
        "k": k,
        "eig_p": eig_p,
        "n_target_slices": target_slice_vectors.shape[0],
        "n_context_words": context_vectors.shape[0],
        "retained_frobenius_sq": float(retained_frobenius_sq),
        "min_hits_fit": MIN_HITS_FIT,
        "n_fitted_target_slices": int(is_fitted.sum()),
    }
    with open(f"output/{kwic_variant}/emb_meta_{kwic_variant}_{slice_grid}.json", "w", encoding="utf-8") as _f:
        json.dump(emb_meta, _f, indent=2)

    print(f"target_slice_vectors: {target_slice_vectors.shape}, context_vectors: {context_vectors.shape}")
    print(f"Kontextwörter ohne POS: {context_words_export['pos'].null_count()}")
    emb_meta
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metriken und Checks

    Es folgen einige Metriken und Checks:
    **TODO: besser erläutern oder entfernen**
    """)
    return


@app.cell
def _(S, k, target_slice_vectors):
    # Fraction of the squared Frobenius norm retained by the rank-k approximation:
    #   ||M_k||_F^2 / ||M||_F^2  =  sum(S[:k]**2) / sum(S**2)
    # ||M||_F^2 is the sum of all squared entries of M (here: the PPMI matrix
    # passed to the SVD). Equivalently: 1 - relative squared reconstruction error.
    # Not "explained variance": the PPMI matrix is not mean-centered, so the
    # first singular value largely reflects the overall mean of the rows.
    # Describes compression within one run; comparable across runs only if the
    # input matrix is constructed the same way.
    retained_frobenius_sq = (S[:k] ** 2).sum() / (S ** 2).sum()

    print(f"k = {k}, target_slice_vectors: {target_slice_vectors.shape}")
    print(f"retained fraction of squared Frobenius norm: {retained_frobenius_sq:.1%}")
    return (retained_frobenius_sq,)


@app.cell
def _(np, pl, slice_grid, target_slice_vectors, target_slices):
    # Sanity check via cosine similarity.
    # Normalize rows to unit length; the dot product of unit vectors is the cosine.
    norms = np.linalg.norm(target_slice_vectors, axis=1, keepdims=True)
    assert (norms > 0).all(), f"{(norms == 0).sum()} zero rows: cosine undefined"
    target_slice_vectors_unit = target_slice_vectors / norms

    target_slice_cosines = target_slice_vectors_unit @ target_slice_vectors_unit.T          # (n_target_slices, n_target_slices), cosines

    # The row table must be in matrix row order; otherwise positional lookup
    # below would attach wrong labels to rows.
    assert (target_slices["target_slice_idx"].to_numpy() == np.arange(target_slice_vectors.shape[0])).all()

    # Example row and its nearest neighbors among all rows of the matrix.
    # the example lemma is taken from the data (first target lemma), because the target words
    # are defined in pipeline_pre_ppmi.py and may change
    assert target_slices.height > 0, "target_slices is empty: PPMI matrix has no rows (fewer than two target words?)"
    example_lemma = target_slices["target_lemma"].sort()[0]
    example_rows = target_slices.filter(pl.col("target_lemma") == example_lemma)

    # preferred slice (labels depend on the grid); falls back to the latest slice of the lemma
    example_slice = {"quarter": "2022-Q2", "halfyear": "2022-H1", "year": "2022"}[slice_grid]
    if example_slice not in example_rows["slice"].to_list():
        example_slice = example_rows["slice"].max()

    example_i = example_rows.filter(
        pl.col("slice") == example_slice
    )["target_slice_idx"].item()                                    # fails unless exactly one match
    print(f"Beispiel: {example_lemma}, {example_slice}")

    # argsort sorts ascending; negating gives descending similarity.
    neighbor_idx = np.argsort(-target_slice_cosines[example_i])[:50]

    target_slice_neighbors = target_slices[neighbor_idx.tolist()].with_columns(
        pl.Series("cosine", target_slice_cosines[example_i][neighbor_idx]).round(3)
    )
    target_slice_neighbors
    return


if __name__ == "__main__":
    app.run()
