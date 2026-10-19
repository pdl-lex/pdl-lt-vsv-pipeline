import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    return mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diagnose zu pipeline_emb_ppmi.py

    Dieses Notebook prüft Verfahrensentscheidungen der Embedding-Stufe an den Daten:

    1. Context Distribution Smoothing (CDS)
    2. additive Glättung (add-λ)
    3. Rang k der abgeschnittenen SVD und Exponent p der Eigenwertgewichtung

    Alle Abschnitte verwenden dasselbe Messverfahren, das vorab einmal beschrieben wird. Jeder Abschnitt ist gleich aufgebaut: Verfahren, Fragestellung, Vorprüfung, erwartete Ergebnisse und ihre Deutung, Berechnung, automatische Auswertung mit Empfehlung. Die Auswertung leitet ihre Aussagen ausschließlich aus den Werten ab, die im selben Lauf berechnet werden. Ergebnisse früherer Messungen werden nicht übernommen.

    Laufzeit: einige Minuten, überwiegend für die wiederholten Zufallsteilungen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Daten

    Grundlage ist die Belegliste aus pre (`kwic_hits_sliced_punkt_*.parquet`): eine Zeile je Beleg (KWIC-Treffer) mit Rede-ID, Datum, Zielwort und den Lemmata des Kontexts. Die Zeitscheiben werden aus dem Datum auf dieselbe Weise gebildet wie in pre. Auch die Kookkurrenzen werden auf dieselbe Weise gezählt: Jedes Vorkommen eines Kontextlemmas in einem Beleg zählt einmal.

    Die Belegliste wird an Stelle der Kookkurrenzdatei verwendet, weil die folgenden Prüfungen die Belege nach Reden aufteilen. Die Kookkurrenzdatei enthält nur Summen je Zeitscheibe. Zur Kontrolle vergleicht der Code die eigene Zählung mit der Kookkurrenzdatei aus pre. Weichen die beiden voneinander ab, bricht das Notebook ab.

    Begriffe wie in emb: Eine **Zeile** der Matrix ist ein Paar aus Zielwort und Zeitscheibe, z. B. „Sicherheit, 2022-Q2“. Ein **Kontextwort** ist eine Spalte der Matrix.
    """)
    return


@app.cell
def _(mo, pl):
    # time grid: "quarter", "halfyear" or "year"; labels are built exactly as in pipeline_pre_ppmi.py
    RASTER = "quarter"
    _out = mo.notebook_dir() / ".." / "01_preprocessing" / "output"

    # hit list from pre: one row per KWIC hit with speech id, date and context lemmas.
    # All kwic_hits_sliced_* files contain the same hits and differ only in the slice column,
    # which is rebuilt here for the chosen grid.
    _hits = pl.read_parquet(_out / "kwic_hits_sliced_punkt_year.parquet")
    _year = pl.col("date").dt.year().cast(pl.Utf8)
    _quarter = pl.col("date").dt.quarter()
    if RASTER == "quarter":
        _slice = _year + "-Q" + _quarter.cast(pl.Utf8)
    elif RASTER == "halfyear":
        _slice = _year + "-H" + ((_quarter - 1) // 2 + 1).cast(pl.Utf8)
    else:
        _slice = _year
    _hits = _hits.with_columns(_slice.alias("slice"))

    # one row per context token, as in pre
    _tokens = (
        _hits
        .select(["id", "slice", "target_lemma", "context_lemmas"])
        .explode("context_lemmas", empty_as_null=False)
        .rename({"context_lemmas": "context_lemma"})
        .drop_nulls("context_lemma")
    )

    # consistency check: the counts must equal pre's co-occurrence file exactly
    _keys = ["slice", "target_lemma", "context_lemma"]
    _own = _tokens.group_by(_keys).len().rename({"len": "n_cooc"})
    _pre = pl.read_csv(
        _out / f"cooc_counts_punkt_{RASTER}.csv",
        schema_overrides={"slice": pl.Utf8, "context_lemma": pl.Utf8},
    )
    _mismatch = (
        _own.join(_pre, on=_keys, how="full", coalesce=True, suffix="_pre")
        .filter(pl.col("n_cooc").ne_missing(pl.col("n_cooc_pre")))
    )
    mo.stop(
        _mismatch.height > 0,
        mo.callout(
            mo.md(
                f"**Abbruch:** {_mismatch.height} Zählungen weichen von "
                f"`cooc_counts_punkt_{RASTER}.csv` ab. Belegliste und Kookkurrenzdatei "
                "stammen nicht aus demselben Lauf von pre."
            ),
            kind="danger",
        ),
    )

    # rows in the same order as in emb: target word, then slice
    _row_keys = ["target_lemma", "slice"]
    rows = (
        _tokens.select(_row_keys).unique().sort(_row_keys).with_row_index("i")
        .join(_hits.group_by(_row_keys).len().rename({"len": "hits"}), on=_row_keys, how="left")
        .sort("i")
    )
    _columns = _tokens.select("context_lemma").unique().sort("context_lemma").with_row_index("j")
    n_columns = _columns.height

    # counts per speech: the speech is the unit of the random split below
    counts_by_speech = (
        _tokens.group_by(["id", "slice", "target_lemma", "context_lemma"]).len()
        .join(rows.select(["i", *_row_keys]), on=_row_keys)
        .join(_columns, on="context_lemma")
        .select(["id", "slice", "i", "j", "len"])
    )

    def _de(n):
        return f"{n:,}".replace(",", ".")

    mo.md(
        f"Belege: {_de(_hits.height)} aus {_de(_hits['id'].n_unique())} Reden. "
        f"Matrix: {rows.height} Zeilen (Zielwort × Zeitscheibe), {_de(n_columns)} Kontextwörter. "
        f"Die Zählungen stimmen mit `cooc_counts_punkt_{RASTER}.csv` überein."
    )
    return counts_by_speech, n_columns, rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Messverfahren

    ### Grundgedanke: Split-half-Prüfung

    Die Reden jeder Zeitscheibe werden zufällig in zwei gleich große Hälften A und B geteilt. Aus jeder Hälfte wird mit derselben Variante des Verfahrens eine eigene Matrix berechnet. Beide Hälften stammen aus derselben Zeitscheibe. Unterschiede zwischen ihnen sind deshalb Zufallsschwankungen der Stichprobe und kein Bedeutungswandel. Eine geeignete Variante bildet eine Zeile in beiden Hälften ähnlich ab und unterscheidet sie zugleich von anderen Zeilen. Das Vorgehen folgt der Split-half-Methode zur Schätzung der Reliabilität (Spearman 1910; Brown 1910). Die Stabilität von Wortvektoren wird in der Literatur ebenfalls durch Neuziehen ganzer Dokumente geprüft (Antoniak & Mimno 2018).

    Geteilt wird nach Reden, nicht nach einzelnen Belegen. Belege aus derselben Rede sind nicht unabhängig voneinander (derselbe Redner, dasselbe Thema, wiederholte Formulierungen). Lägen sie in beiden Hälften, wären die Hälften einander künstlich ähnlich.

    Die Teilung wird mehrfach mit verschiedenen Zufallszahlen wiederholt (Anzahl: `N_SPLITS` unten). Alle Varianten werden auf denselben Teilungen berechnet und sind dadurch direkt vergleichbar.

    ### Kennwerte

    Für jede Zeile wird die Kosinusähnlichkeit ihrer Hälfte A zu ihrer eigenen Hälfte B berechnet. Diese Ähnlichkeit wird mit der Ähnlichkeit zu Vergleichszeilen verglichen, ebenfalls über die Hälften hinweg (Mittelwert aus „Hälfte A der Zeile mit Hälfte B der Vergleichszeile“ und umgekehrt). Es gibt zwei Arten von Vergleichszeilen:

    - **a_zeit**: Die Vergleichszeilen sind die anderen Zeitscheiben desselben Zielworts. a_zeit ist der Anteil dieser Vergleichszeilen, denen die Zeile weniger ähnlich ist als ihrer eigenen zweiten Hälfte. Ein hoher Wert bedeutet, dass sich das, was eine Zeitscheibe von den anderen Zeitscheiben desselben Worts unterscheidet, in beiden unabhängigen Hälften zeigt. Nur dann lässt sich Bedeutungswandel von Zufallsschwankungen unterscheiden.
    - **a_wort**: Die Vergleichszeilen sind die anderen Zielwörter derselben Zeitscheibe. a_wort ist der Anteil dieser Vergleichszeilen, denen die Zeile weniger ähnlich ist als ihrer eigenen zweiten Hälfte. Ein hoher Wert bedeutet, dass die Zielwörter unterscheidbar bleiben.

    Beide Werte liegen zwischen 0 und 1. Bei 0,5 ist die eigene zweite Hälfte nicht ähnlicher als eine beliebige Vergleichszeile; bei 1 ist sie ähnlicher als alle Vergleichszeilen. Gleiche Ähnlichkeiten zählen zur Hälfte. Der Wert entspricht der Fläche unter der ROC-Kurve (AUC): der Wahrscheinlichkeit, dass der zutreffende Fall eine höhere Ähnlichkeit erhält als ein zufällig gewählter Vergleichsfall (Hanley & McNeil 1982). Er ist identisch mit der normierten Mann-Whitney-U-Statistik (Mann & Whitney 1947). Die Gegenüberstellung der Ähnlichkeit innerhalb derselben Einheit und zwischen verschiedenen Einheiten über zwei Datenhälften hinweg folgt Haxby et al. (2001).

    Die Kennwerte beruhen auf der Rangfolge der Ähnlichkeiten, nicht auf Differenzen von Kosinuswerten. Der Grund: Die geprüften Varianten verändern die Verteilung der Kosinuswerte insgesamt. Nach einer starken Kürzung durch die SVD sind z. B. alle Zeilen einander ähnlicher. Differenzen von Kosinuswerten sind deshalb zwischen Varianten nicht vergleichbar; die Rangfolge ist von solchen Verschiebungen nicht betroffen.

    Die beiden Kennwerte ergänzen einander. Eine Variante, die die Zeitscheiben eines Worts zu einem einzigen Punkt zusammenzieht, erhöht a_wort, senkt aber a_zeit. Eine Variante, die alle Zeilen einer Zeitscheibe einander angleicht, erhöht a_zeit, senkt aber a_wort.

    ### Zusammenfassung über Zeilen und Teilungen

    Für jede Teilung wird der Mittelwert über die Zeilen gebildet, für alle Zeilen und getrennt für drei Gruppen nach Textmenge. Die **Gruppen nach Textmenge** (Terzile) entstehen, indem die Zeilen nach der Zahl ihrer Belege in drei gleich große Gruppen geteilt werden. Berichtet wird der Mittelwert über die Teilungen. Eine Zeile, die in einer Hälfte keinen Beleg oder keinen positiven Wert hat, entfällt in dieser Teilung.

    Die Werte beziehen sich auf halbe Datenmengen. Für die Entscheidung zählt nur der Vergleich der Varianten untereinander.

    ### Entscheidungsregel

    Verglichen wird jeweils eine Variante mit einer Referenz, dem heutigen Stand von emb. Eine Änderung gilt als **belastbar gestiegen** bzw. **belastbar gesunken**, wenn sie in allen Teilungen positiv bzw. negativ ist. Wechselt das Vorzeichen zwischen den Teilungen, ist die Änderung kleiner als die Schwankung, die allein durch die zufällige Teilung entsteht.

    Jede Variante wird über beide Kennwerte und alle Gruppen hinweg eingestuft:

    | Einstufung | Bedingung |
    |---|---|
    | Verbesserung | Mindestens ein Kennwert steigt über alle Zeilen belastbar, und kein Kennwert sinkt in irgendeiner Gruppe belastbar (Pareto-Verbesserung). |
    | Verschlechterung | Mindestens ein Kennwert sinkt belastbar, keiner steigt belastbar. |
    | Zielkonflikt | Belastbare Anstiege und belastbare Rückgänge treten gleichzeitig auf. |
    | kein belastbarer Unterschied | alle übrigen Fälle |

    Empfohlen wird eine Variante nur mit der Einstufung „Verbesserung“. Kommen mehrere Varianten in Frage, wird diejenige mit dem größten Anstieg von a_zeit über alle Zeilen empfohlen, weil a_zeit die Eigenschaft misst, auf die es bei der Darstellung von Bedeutungswandel ankommt. Varianten, deren a_zeit sich von der empfohlenen nicht belastbar unterscheidet, werden als gleichwertig genannt.

    ### Begriffe

    - **Kosinusähnlichkeit**: Ähnlichkeit zweier Vektoren nach ihrer Richtung, unabhängig von ihrer Länge.
    - **Hapax** (von *hapax legomenon*, „einmal Gesagtes“): ein Kontextwort, das in einer Zeitscheibe über alle Zielwörter hinweg genau einmal vorkommt.
    """)
    return


@app.cell
def _(mo, np):
    def show_table(df):
        """Table with all rows on one page."""
        return mo.ui.table(df, pagination=False, selection=None, show_column_summaries=False, show_data_types=False)

    def count_matrix(counts, n_rows, n_columns):
        """Dense count matrix (rows x context words) from a long table with columns i, j, len."""
        C = np.zeros((n_rows, n_columns))
        np.add.at(C, (counts["i"].to_numpy(), counts["j"].to_numpy()), counts["len"].to_numpy())
        return C

    def ppmi_matrix(C, row_slice, alpha=1.0, add=0.0):
        """PPMI with marginals per time slice, as in pipeline_emb_ppmi.py.
        alpha < 1: context distribution smoothing; add > 0: additive smoothing of the counts.
        Rows without any count stay 0, also with additive smoothing."""
        M = np.zeros_like(C)
        for s in np.unique(row_slice):
            r = np.flatnonzero(row_slice == s)
            Cs = C[r]
            observed = Cs.sum(axis=1) > 0
            c = np.flatnonzero(Cs.sum(axis=0) > 0)      # context words occurring in this slice
            r, Cs = r[observed], Cs[observed][:, c]
            if len(r) == 0:
                continue
            Cs = Cs + add
            total = Cs.sum()
            p_target = Cs.sum(axis=1, keepdims=True) / total
            f = Cs.sum(axis=0) ** alpha
            p_context = f / f.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                pmi = np.log2(Cs / total / (p_target * p_context))
            # cells without (smoothed) count get 0, as in emb
            M[np.ix_(r, c)] = np.where(Cs > 0, np.maximum(pmi, 0), 0.0)
        return M

    def unit_rows(X):
        """Rows scaled to length 1; rows without any non-zero value become NaN (cosine undefined)."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(norms > 0, X / norms, np.nan)

    def separation(A, B, row_word, row_slice):
        """A, B: unit row vectors of half A and half B (same row order).
        Returns a_zeit and a_wort per row: share of comparison rows that are less similar
        to the row than its own other half (ties count half)."""
        X = A.astype(np.float32) @ B.astype(np.float32).T   # X[i, j] = cos(half A of i, half B of j)
        own = np.diag(X).copy()
        comp = (X + X.T) / 2                                 # comparison in both directions
        same_word = row_word[:, None] == row_word[None, :]
        same_slice = row_slice[:, None] == row_slice[None, :]

        def share(mask):
            valid = mask & ~np.isnan(comp)
            less = ((comp < own[:, None]) & valid).sum(axis=1)
            ties = ((comp == own[:, None]) & valid).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                result = (less + 0.5 * ties) / valid.sum(axis=1)
            result[np.isnan(own)] = np.nan
            return result

        return {
            "a_zeit": share(same_word & ~same_slice),
            "a_wort": share(same_slice & ~same_word),
        }

    def mean_per_split(x, mask):
        """x: splits x rows. Mean over the selected rows, one value per split;
        rows without a value in a split are left out of that split."""
        valid = ~np.isnan(x[:, mask])
        with np.errstate(invalid="ignore"):
            return np.where(valid, x[:, mask], 0).sum(axis=1) / valid.sum(axis=1)

    def fmt_num(x, sign=False, digits=3):
        return (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace("-", "−").replace(".", ",")

    def fmt_pct(x):
        return f"{x * 100:.0f} %"

    return (
        count_matrix,
        fmt_num,
        fmt_pct,
        mean_per_split,
        ppmi_matrix,
        separation,
        show_table,
        unit_rows,
    )


@app.cell
def _(fmt_num, mean_per_split, np, pl):
    CRITERIA = ["a_zeit", "a_wort"]

    def change_per_split(variant, reference, group_masks):
        """Mean change (variant minus reference) per split, for each group and criterion."""
        return {
            g: {c: mean_per_split(variant[c] - reference[c], m) for c in CRITERIA}
            for g, m in group_masks.items()
        }

    def verdict(per_split):
        """'gestiegen' or 'gesunken' if the change has the same sign in every split."""
        if (per_split > 0).all():
            return "gestiegen"
        if (per_split < 0).all():
            return "gesunken"
        return "uneinheitlich"

    def classify(change):
        """Classification of a variant against the reference, see the decision rule."""
        verdicts = [verdict(x) for d in change.values() for x in d.values()]
        up_overall = any(verdict(change["alle Zeilen"][c]) == "gestiegen" for c in CRITERIA)
        up_any = "gestiegen" in verdicts
        down_any = "gesunken" in verdicts
        if up_overall and not down_any:
            return "Verbesserung"
        if down_any and not up_any:
            return "Verschlechterung"
        if up_any and down_any:
            return "Zielkonflikt"
        return "kein belastbarer Unterschied"

    def change_table(change, n_splits):
        """Mean change per group and criterion, with the number of splits with a positive change."""
        columns = {"Gruppe": list(change)}
        for c in CRITERIA:
            columns[f"Δ {c}"] = [round(float(np.mean(change[g][c])), 4) for g in change]
            columns[f"Δ {c} > 0"] = [f"{int((change[g][c] > 0).sum())} von {n_splits}" for g in change]
        return pl.DataFrame(columns)

    def describe_change(change):
        """Lists all belastbar changes (criterion, group, mean change)."""
        found = {"gestiegen": [], "gesunken": []}
        for g, d in change.items():
            for c in CRITERIA:
                v = verdict(d[c])
                if v in found:
                    found[v].append(f"{c} ({g}: {fmt_num(float(np.mean(d[c])), sign=True)})")
        return (
            f"Belastbar gestiegen: {', '.join(found['gestiegen']) or 'nichts'}. "
            f"Belastbar gesunken: {', '.join(found['gesunken']) or 'nichts'}."
        )

    def select_variant(results, changes, classes, all_rows):
        """Variant with the largest mean increase of a_zeit among those classified as improvement,
        and the improving variants whose a_zeit does not differ belastbar from it."""
        improving = [v for v in changes if classes[v] == "Verbesserung"]
        if not improving:
            return None, []
        best = max(improving, key=lambda v: float(np.mean(changes[v]["alle Zeilen"]["a_zeit"])))
        equivalent = [
            v for v in improving
            if v != best
            and verdict(mean_per_split(results[v]["a_zeit"] - results[best]["a_zeit"], all_rows)) == "uneinheitlich"
        ]
        return best, equivalent

    REASONS = {
        "Verschlechterung": "Die Variante verschlechtert die Unterscheidbarkeit.",
        "Zielkonflikt": (
            "Die Variante verbessert einzelne Kennwerte und verschlechtert andere; "
            "nach der Entscheidungsregel genügt das nicht."
        ),
        "kein belastbarer Unterschied": (
            "Ein Nutzen ist nicht nachweisbar; das einfachere Verfahren ist vorzuziehen."
        ),
    }
    return (
        CRITERIA,
        REASONS,
        change_per_split,
        change_table,
        classify,
        describe_change,
        select_variant,
        verdict,
    )


@app.cell
def _(np, rows):
    row_slice = rows["slice"].to_numpy()
    row_word = rows["target_lemma"].to_numpy()

    # three groups of equal size by number of hits (terciles); ties are broken by row order
    _hits = rows["hits"].to_numpy()
    row_group = np.argsort(np.argsort(_hits, kind="stable")) * 3 // len(_hits)
    group_labels = [
        f"{_name} ({_hits[row_group == _g].min()}–{_hits[row_group == _g].max()} Belege)"
        for _g, _name in enumerate(["geringe Textmenge", "mittlere Textmenge", "hohe Textmenge"])
    ]
    group_masks = {_label: row_group == _g for _g, _label in enumerate(group_labels)}
    group_masks["alle Zeilen"] = np.ones(len(row_group), dtype=bool)
    return group_labels, group_masks, row_group, row_slice, row_word


@app.cell
def _(
    count_matrix,
    counts_by_speech,
    mo,
    n_columns,
    np,
    pl,
    ppmi_matrix,
    row_slice,
    row_word,
    rows,
    separation,
    unit_rows,
):
    N_SPLITS = 20
    SEED = 42

    # random splits, shared by all sections: half of the speeches of each slice go to half A
    _rng = np.random.default_rng(SEED)
    _speeches = counts_by_speech.select(["id", "slice"]).unique().sort("id")
    _in_a = [
        _speeches
        .with_columns(pl.Series("key", _rng.random(_speeches.height)))
        .with_columns((pl.col("key").rank("ordinal").over("slice") <= pl.len().over("slice") / 2).alias("in_a"))
        .select(["id", "in_a"])
        for _ in range(N_SPLITS)
    ]

    def split_counts(r):
        """Count matrices of half A and half B in split r."""
        counts = counts_by_speech.join(_in_a[r], on="id")
        return (
            count_matrix(counts.filter(pl.col("in_a")), rows.height, n_columns),
            count_matrix(counts.filter(~pl.col("in_a")), rows.height, n_columns),
        )

    def ppmi_split_results(alpha=1.0, add=0.0, title=""):
        """a_zeit and a_wort (splits x rows) for PPMI vectors of one variant."""
        results = {"a_zeit": [], "a_wort": []}
        for r in mo.status.progress_bar(range(N_SPLITS), title=title):
            counts_a, counts_b = split_counts(r)
            res = separation(
                unit_rows(ppmi_matrix(counts_a, row_slice, alpha, add)),
                unit_rows(ppmi_matrix(counts_b, row_slice, alpha, add)),
                row_word,
                row_slice,
            )
            for c in results:
                results[c].append(res[c])
        return {c: np.array(v) for c, v in results.items()}

    # full data: counts and hapax contexts (occurring exactly once in the slice, over all target words)
    counts_full = count_matrix(counts_by_speech, rows.height, n_columns)
    hapax_mask = np.zeros(counts_full.shape, dtype=bool)
    for _s in np.unique(row_slice):
        _r = np.flatnonzero(row_slice == _s)
        hapax_mask[_r] = counts_full[_r].sum(axis=0) == 1
    return N_SPLITS, counts_full, hapax_mask, ppmi_split_results, split_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Referenzwerte: PPMI wie in emb

    Die Abschnitte zu CDS und additiver Glättung vergleichen mit den PPMI-Vektoren ohne Glättung, also mit der Matrix, die emb an die SVD übergibt.
    """)
    return


@app.cell
def _(ppmi_split_results):
    reference_ppmi = ppmi_split_results(title="Referenz: PPMI ohne Glättung")
    return (reference_ppmi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Context Distribution Smoothing (CDS)

    ### Verfahren

    PPMI vergleicht, wie oft ein Kontextwort $c$ zusammen mit einem Zielwort $w$ beobachtet wurde, mit der Häufigkeit, die bei Unabhängigkeit der beiden zu erwarten wäre:

    $$
    \text{PPMI}(w, c) = \max\left(0,\ \log_2 \frac{P(w, c)}{P(w)\,P(c)}\right)
    $$

    PMI ist zugunsten seltener Kontextwörter verzerrt (Turney & Pantel 2010): Weil $P(c)$ klein ist, erhält ein Hapax einen hohen Wert, obwohl ein einzelnes Vorkommen zufällig sein kann. Context Distribution Smoothing (Levy, Goldberg & Dagan 2015) ersetzt $P(c)$ durch eine geglättete Verteilung:

    $$
    P_\alpha(c) = \frac{\#(c)^\alpha}{\sum_{c'} \#(c')^\alpha}, \qquad \alpha = 0{,}75
    $$

    Dabei ist $\#(c)$ die Häufigkeit des Kontextworts in der Zeitscheibe. Mit $\alpha < 1$ steigt die geschätzte Wahrscheinlichkeit seltener Kontextwörter, und ihr PPMI-Wert sinkt. Bei häufigen Kontextwörtern ist es umgekehrt. Der Wert 0,75 stammt aus der Stichprobenverteilung des Negative Sampling in word2vec (Mikolov et al. 2013). Levy, Goldberg & Dagan (2015) berichten mit CDS bessere Ergebnisse bei Wortähnlichkeits- und Analogieaufgaben. Für dieses Korpus steht ein solcher Test mit menschlichen Urteilen nicht zur Verfügung.

    ### Fragestellung

    Verbessert CDS die Unterscheidbarkeit von Zeitscheiben (a_zeit) oder Zielwörtern (a_wort), ohne eine der beiden zu verschlechtern? Referenz: PPMI ohne CDS.

    ### Vorprüfung

    Wirkt CDS wie beschrieben? Für jede Zeile wird der Anteil der Summe der quadrierten PPMI-Werte berechnet, der auf Hapax-Kontexte entfällt, einmal ohne und einmal mit CDS. Maßgeblich sind die quadrierten Werte, weil die Länge eines Vektors, auf die die Kosinusähnlichkeit normiert, aus ihnen berechnet wird. Erwartet wird, dass der Anteil mit CDS sinkt.

    ### Erwartete Ergebnisse und ihre Deutung

    | Befund | Deutung |
    |---|---|
    | a_zeit oder a_wort steigt belastbar, keiner der Werte sinkt | CDS verringert den Einfluss zufälliger Schwankungen: Nutzen |
    | keine belastbare Änderung | CDS verändert die Vektoren, aber nicht ihre Unterscheidbarkeit: kein Nutzen |
    | ein Wert sinkt belastbar | CDS erschwert die Unterscheidung von Zeitscheiben oder Zielwörtern: Schaden |

    Ein Nutzen ist vor allem in der Gruppe nach Textmenge zu erwarten, in der Hapax-Kontexte den größten Anteil des Vektors ausmachen.
    """)
    return


@app.cell
def _(
    counts_full,
    group_masks,
    hapax_mask,
    np,
    pl,
    ppmi_matrix,
    row_slice,
    show_table,
):
    CDS_ALPHA = 0.75

    # share of the squared vector length that falls on hapax contexts, per row
    cds_hapax_share = {}
    for _name, _alpha in {"ohne CDS": 1.0, "mit CDS": CDS_ALPHA}.items():
        _sq = ppmi_matrix(counts_full, row_slice, _alpha) ** 2
        with np.errstate(invalid="ignore"):
            cds_hapax_share[_name] = (_sq * hapax_mask).sum(axis=1) / _sq.sum(axis=1)

    show_table(pl.DataFrame({
        "Gruppe": list(group_masks),
        "Zeilen": [int(_m.sum()) for _m in group_masks.values()],
        **{
            f"Hapax-Anteil {_name}": [round(float(np.nanmean(_v[_m])), 3) for _m in group_masks.values()]
            for _name, _v in cds_hapax_share.items()
        },
    }))
    return CDS_ALPHA, cds_hapax_share


@app.cell
def _(CDS_ALPHA, ppmi_split_results):
    cds_results = ppmi_split_results(alpha=CDS_ALPHA, title="mit CDS")
    return (cds_results,)


@app.cell(hide_code=True)
def _(
    CDS_ALPHA,
    CRITERIA,
    N_SPLITS,
    REASONS,
    cds_hapax_share,
    cds_results,
    change_per_split,
    change_table,
    classify,
    describe_change,
    fmt_num,
    fmt_pct,
    group_labels,
    group_masks,
    mean_per_split,
    mo,
    np,
    pl,
    reference_ppmi,
    show_table,
    verdict,
):
    _change = change_per_split(cds_results, reference_ppmi, group_masks)
    _class = classify(_change)
    _all = group_masks["alle Zeilen"]

    _levels = pl.DataFrame({
        "Variante": ["ohne CDS", f"mit CDS (α = {fmt_num(CDS_ALPHA, digits=2)})"],
        **{
            _c: [round(float(np.mean(mean_per_split(_res[_c], _all))), 4) for _res in (reference_ppmi, cds_results)]
            for _c in CRITERIA
        },
    })

    # pre-check
    _h_ohne = float(np.nanmean(cds_hapax_share["ohne CDS"]))
    _h_mit = float(np.nanmean(cds_hapax_share["mit CDS"]))
    _h_group = [float(np.nanmean(cds_hapax_share["ohne CDS"][group_masks[_l]])) for _l in group_labels]
    _expected = group_labels[int(np.argmax(_h_group))]
    _text = (
        f"**Vorprüfung.** Ohne CDS entfallen im Mittel {fmt_pct(_h_ohne)} der quadrierten PPMI-Werte einer "
        f"Zeile auf Hapax-Kontexte, mit CDS {fmt_pct(_h_mit)}. "
        + ("CDS senkt das Gewicht seltener Kontextwörter also wie beschrieben. " if _h_mit < _h_ohne else
           "CDS senkt das Gewicht seltener Kontextwörter in diesen Daten nicht. ")
        + f"Am höchsten ist der Anteil in der Gruppe „{_expected}“ ({fmt_pct(max(_h_group))})."
    )
    _paragraphs = [_text]

    # split-half
    _paragraphs.append(f"**Split-half-Prüfung.** Einstufung gegenüber „ohne CDS“: **{_class}**. " + describe_change(_change))
    _expected_up = [_c for _c in CRITERIA if verdict(_change[_expected][_c]) == "gestiegen"]
    _paragraphs.append(
        f"In der Gruppe „{_expected}“ steigt belastbar: {', '.join(_expected_up)}. Das entspricht der Erwartung aus der Vorprüfung."
        if _expected_up else
        f"In der Gruppe „{_expected}“ steigt keiner der Kennwerte belastbar. Das widerspricht der Erwartung aus der Vorprüfung."
    )

    if _class == "Verbesserung":
        _kind, _rec = "success", (
            "**Empfehlung: CDS verwenden.** Die Prüfung bezieht sich auf die PPMI-Vektoren; "
            "ob der Nutzen nach der SVD erhalten bleibt, ist gesondert zu prüfen."
        )
    else:
        _kind, _rec = "danger", f"**Empfehlung: CDS nicht verwenden.** {REASONS[_class]}"

    mo.vstack([
        mo.md("**Tabelle 1.1:** Kennwerte je Variante, alle Zeilen"), show_table(_levels),
        mo.md("**Tabelle 1.2:** Änderung durch CDS nach Textmenge"), show_table(change_table(_change, N_SPLITS)),
        mo.md("### Auswertung"),
        mo.md("\n\n".join(_paragraphs)),
        mo.callout(mo.md(_rec), kind=_kind),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Additive Glättung (add-λ)

    ### Verfahren

    Vor der PPMI-Berechnung wird zu jeder Zählung einer Zeitscheibe die Konstante $\lambda$ addiert, auch zu Paaren aus Zielwort und Kontextwort, die nie gemeinsam beobachtet wurden. Randsummen und Gesamtzahl werden aus der geglätteten Tabelle berechnet. Bei $\lambda = 1$ heißt das Verfahren Laplace-Glättung, bei $0 < \lambda < 1$ Lidstone-Glättung (Manning & Schütze 1999, Kap. 6.2). In der Literatur zur distributionellen Semantik wird es als Mittel gegen die Verzerrung der PMI zugunsten seltener Ereignisse genannt (Turney & Pantel 2010). Nicht zu verwechseln ist es mit *Laplacian smoothing*, der Glättung mit dem Laplace-Operator auf Netzen oder Graphen.

    Geglättet werden nur Zeilen, die in der Zeitscheibe mindestens einen Beleg haben, und nur Kontextwörter, die in der Zeitscheibe vorkommen. Geprüft werden mehrere Werte von $\lambda$ (`ADD_LAMBDAS` unten).

    ### Fragestellung

    Verbessert die additive Glättung die Unterscheidbarkeit von Zeitscheiben (a_zeit) oder Zielwörtern (a_wort), ohne eine der beiden zu verschlechtern? Referenz: PPMI ohne Glättung.

    ### Vorprüfung

    Je Zeile werden zwei Anteile an der Summe der quadrierten PPMI-Werte berechnet:

    - **Hapax-Anteil**: wie im Abschnitt zu CDS. Erwartet wird, dass er mit zunehmendem $\lambda$ sinkt.
    - **Anteil nie beobachteter Paare**: der Anteil, der auf Paare entfällt, die im Text nie gemeinsam vorkommen. Ohne Glättung ist er 0. Er gibt an, welcher Teil des Vektors nicht auf Beobachtungen beruht.

    ### Erwartete Ergebnisse und ihre Deutung

    | Befund | Deutung |
    |---|---|
    | a_zeit oder a_wort steigt belastbar, keiner der Werte sinkt | Die Glättung verringert den Einfluss zufälliger Schwankungen: Nutzen |
    | a_zeit steigt, a_wort sinkt | Die zugesetzten Werte hängen nur von den Randsummen der Zeitscheibe ab und gleichen alle Zeilen einer Zeitscheibe einander an: kein Nutzen |
    | ein Wert sinkt belastbar, keiner steigt | Die zugesetzten Werte überdecken beobachtete Unterschiede: Schaden |

    Die Split-half-Prüfung bewertet nur die Reproduzierbarkeit. Werte, die nicht auf Beobachtungen beruhen, aber in beiden Hälften gleich ausfallen, erkennt sie nicht als Nachteil. Der Anteil nie beobachteter Paare ist deshalb bei einer Empfehlung mitzuberücksichtigen.
    """)
    return


@app.cell
def _(counts_full, hapax_mask, np, pl, ppmi_matrix, row_slice, show_table):
    ADD_LAMBDAS = [0.01, 0.1, 1.0]

    # shares of the squared vector length per row: hapax contexts, and pairs never observed together
    add_shares = {}
    _unobserved = counts_full == 0
    for _lam in [0.0, *ADD_LAMBDAS]:
        _sq = ppmi_matrix(counts_full, row_slice, add=_lam) ** 2
        with np.errstate(invalid="ignore"):
            add_shares[_lam] = {
                "hapax": (_sq * hapax_mask).sum(axis=1) / _sq.sum(axis=1),
                "unobserved": (_sq * _unobserved).sum(axis=1) / _sq.sum(axis=1),
            }

    show_table(pl.DataFrame({
        "λ": [0.0, *ADD_LAMBDAS],
        "Hapax-Anteil": [round(float(np.nanmean(_v["hapax"])), 3) for _v in add_shares.values()],
        "Anteil nie beobachteter Paare": [round(float(np.nanmean(_v["unobserved"])), 3) for _v in add_shares.values()],
    }))
    return ADD_LAMBDAS, add_shares


@app.cell
def _(ADD_LAMBDAS, fmt_num, ppmi_split_results):
    add_results = {
        _lam: ppmi_split_results(add=_lam, title=f"additive Glättung, λ = {fmt_num(_lam, digits=2)}")
        for _lam in ADD_LAMBDAS
    }
    return (add_results,)


@app.cell(hide_code=True)
def _(
    ADD_LAMBDAS,
    CRITERIA,
    N_SPLITS,
    REASONS,
    add_results,
    add_shares,
    change_per_split,
    change_table,
    classify,
    describe_change,
    fmt_num,
    fmt_pct,
    group_masks,
    mean_per_split,
    mo,
    np,
    pl,
    reference_ppmi,
    select_variant,
    show_table,
):
    def _lam_label(lam):
        return f"λ = {fmt_num(lam, digits=2)}"

    _all = group_masks["alle Zeilen"]
    _changes = {_lam: change_per_split(add_results[_lam], reference_ppmi, group_masks) for _lam in ADD_LAMBDAS}
    _classes = {_lam: classify(_c) for _lam, _c in _changes.items()}

    _levels = pl.DataFrame({
        "Variante": ["ohne Glättung", *[_lam_label(_l) for _l in ADD_LAMBDAS]],
        **{
            _c: [
                round(float(np.mean(mean_per_split(_res[_c], _all))), 4)
                for _res in (reference_ppmi, *[add_results[_l] for _l in ADD_LAMBDAS])
            ]
            for _c in CRITERIA
        },
        "Einstufung": ["Referenz", *[_classes[_l] for _l in ADD_LAMBDAS]],
    })
    _details = pl.concat([
        pl.DataFrame({"Variante": [_lam_label(_l)] * len(group_masks)}).hstack(change_table(_changes[_l], N_SPLITS))
        for _l in ADD_LAMBDAS
    ])

    # pre-check
    _paragraphs = []
    _hapax_ref = float(np.nanmean(add_shares[0.0]["hapax"]))
    _parts = [
        f"{_lam_label(_l)}: Hapax-Anteil {fmt_pct(float(np.nanmean(add_shares[_l]['hapax'])))}, "
        f"Anteil nie beobachteter Paare {fmt_pct(float(np.nanmean(add_shares[_l]['unobserved'])))}"
        for _l in ADD_LAMBDAS
    ]
    _paragraphs.append(
        f"**Vorprüfung.** Ohne Glättung beträgt der Hapax-Anteil im Mittel {fmt_pct(_hapax_ref)}. "
        + "; ".join(_parts) + "."
    )

    # split-half, per variant
    _lines = [f"- {_lam_label(_l)}: **{_classes[_l]}**. {describe_change(_changes[_l])}" for _l in ADD_LAMBDAS]
    _paragraphs.append("**Split-half-Prüfung** (gegenüber „ohne Glättung“):\n\n" + "\n".join(_lines))

    _best, _equivalent = select_variant(add_results, _changes, _classes, _all)
    if _best is None:
        _kind = "danger"
        _rec = "**Empfehlung: keine additive Glättung verwenden.** Keine der geprüften Konstanten ist als Verbesserung eingestuft. "
        _rec += " ".join(sorted({REASONS[_classes[_l]] for _l in ADD_LAMBDAS}))
    else:
        _kind = "success"
        _rec = (
            f"**Empfehlung: additive Glättung mit {_lam_label(_best)}.** a_zeit steigt über alle Zeilen um "
            f"{fmt_num(float(np.mean(_changes[_best]['alle Zeilen']['a_zeit'])), sign=True)}. "
            f"Dabei entfallen im Mittel {fmt_pct(float(np.nanmean(add_shares[_best]['unobserved'])))} der "
            "quadrierten PPMI-Werte auf nie beobachtete Paare; dieser Anteil beruht nicht auf Beobachtungen."
        )
        if _equivalent:
            _rec += f" Gleichwertig in a_zeit: {', '.join(_lam_label(_l) for _l in _equivalent)}."
        _rec += " Die Prüfung bezieht sich auf die PPMI-Vektoren; ob der Nutzen nach der SVD erhalten bleibt, ist gesondert zu prüfen."

    mo.vstack([
        mo.md("**Tabelle 2.1:** Kennwerte je Variante, alle Zeilen"), show_table(_levels),
        mo.md("**Tabelle 2.2:** Änderung gegenüber „ohne Glättung“ nach Textmenge"), show_table(_details),
        mo.md("### Auswertung"),
        mo.md("\n\n".join(_paragraphs)),
        mo.callout(mo.md(_rec), kind=_kind),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Rang k der SVD und Exponent p der Eigenwertgewichtung

    ### Verfahren

    emb zerlegt die PPMI-Matrix $M$ mit der Singulärwertzerlegung $M = U \Sigma V^\top$ und behält nur die $k$ größten Singulärwerte (abgeschnittene SVD, Deerwester et al. 1990). Nach dem Satz von Eckart & Young (1936) ist $U_k \Sigma_k V_k^\top$ die beste Approximation von $M$ mit Rang $k$. Die Zeilenvektoren sind

    $$
    x = U_k\, \Sigma_k^{\,p}.
    $$

    Der Exponent $p$ der Eigenwertgewichtung (*eigenvalue weighting*; Caron 2001; Levy, Goldberg & Dagan 2015) bestimmt, wie stark die Dimensionen mit großen Singulärwerten gegenüber den übrigen gewichtet werden: bei $p = 1$ mit dem Singulärwert, bei $p = 0{,}5$ mit seiner Quadratwurzel, bei $p = 0$ alle gleich. Werte unter 1 haben sich in mehreren Untersuchungen als günstiger erwiesen (Caron 2001; Bullinaria & Levy 2012; Österlund, Ödling & Sahlgren 2015). Eine Theorie, die einen bestimmten Wert vorschreibt, gibt es nicht; $p$ ist empirisch zu bestimmen.

    Heutiger Stand in emb (Referenz): $k$ = ein Drittel der Zeilenzahl, mindestens 30, höchstens 100; $p = 1$.

    ### Fragestellung

    Welche Kombination aus $k$ und $p$ ergibt die beste Unterscheidbarkeit von Zeitscheiben (a_zeit), ohne die Unterscheidbarkeit der Zielwörter (a_wort) zu verschlechtern? Gerechnet wird mit PPMI ohne Glättung, wie heute in emb.

    ### Ergänzung des Messverfahrens: gemeinsamer Raum für beide Hälften

    Die Hälften müssen im selben Koordinatensystem verglichen werden. Zwei naheliegende Wege scheiden aus:

    - **SVD je Hälfte:** Die Achsen zweier getrennter Zerlegungen sind nicht vergleichbar (Reihenfolge, Vorzeichen und Drehung sind nicht festgelegt).
    - **SVD der vollständigen Daten:** Die Achsen würden an die Zufallsschwankungen beider Hälften angepasst. Die Hälften würden dadurch einander künstlich ähnlich. Eine Stichprobe, an die ein Verfahren angepasst wurde, darf nicht zu seiner Bewertung verwendet werden (Hastie, Tibshirani & Friedman 2009, Kap. 7).

    Stattdessen wird die SVD an einer Hälfte berechnet, genau wie emb sie an den vollständigen Daten berechnet. Die andere Hälfte wird in denselben Raum projiziert (*folding-in*; Deerwester et al. 1990; Berry, Dumais & O'Brien 1995):

    $$
    x = m\, V_k\, \Sigma_k^{\,p-1}
    $$

    Dabei ist $m$ der PPMI-Zeilenvektor. Für die Zeilen der Hälfte, an der die SVD berechnet wurde, ergibt diese Formel genau $U_k \Sigma_k^p$. Danach werden die Rollen der Hälften getauscht, und die Kennwerte beider Richtungen werden gemittelt.

    ### Vorprüfung

    Für jedes $k$ wird angegeben, welchen Anteil der Summe der quadrierten PPMI-Werte die Approximation mit Rang $k$ erhält (Anteil an der quadrierten Frobenius-Norm, berechnet auf den vollständigen Daten). Der Wert beschreibt nur die Stärke der Kürzung, nicht ihre Güte.

    ### Erwartete Ergebnisse und ihre Deutung

    | Befund | Deutung |
    |---|---|
    | a_zeit steigt mit sinkendem k bis zu einem Höchstwert und fällt danach | Die Dimensionen mit kleinen Singulärwerten tragen überwiegend Zufallsschwankungen; eine zu starke Kürzung beseitigt Unterschiede zwischen Zeitscheiben. |
    | a_wort steigt, a_zeit fällt bei kleinem k | Die Kürzung erhält die Identität der Zielwörter, zieht aber ihre Zeitscheiben zusammen. |
    | a_zeit steigt mit wachsendem k durchgehend | Auch Dimensionen mit kleinen Singulärwerten tragen reproduzierbare Information über Zeitscheiben. |
    | p < 1 erhöht a_zeit | Die Dimensionen mit großen Singulärwerten sind bei p = 1 zu stark gewichtet. |

    Liegt die empfohlene Kombination am Rand des untersuchten Bereichs, ist der Bereich zu erweitern.
    """)
    return


@app.cell
def _(counts_full, np, pl, ppmi_matrix, row_slice, rows, show_table):
    SVD_K_REFERENCE = min(100, max(30, rows.height // 3))    # rule in pipeline_emb_ppmi.py
    SVD_P_REFERENCE = 1.0
    # up to the number of rows; ranks that cannot be computed in every half are dropped below
    SVD_KS = sorted({5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, SVD_K_REFERENCE})
    SVD_PS = [1.0, 0.5, 0.0]

    # singular values of the full PPMI matrix via the Gram matrix: M M^T = U S^2 U^T
    _m = ppmi_matrix(counts_full, row_slice)
    _s2 = np.clip(np.linalg.eigvalsh(_m @ _m.T)[::-1], 0, None)
    svd_retained = {_k: float(_s2[:_k].sum() / _s2.sum()) for _k in SVD_KS}

    show_table(pl.DataFrame({
        "k": SVD_KS,
        "erhaltener Anteil": [round(svd_retained[_k], 3) for _k in SVD_KS],
    }))
    return SVD_KS, SVD_K_REFERENCE, SVD_PS, SVD_P_REFERENCE, svd_retained


@app.cell
def _(
    N_SPLITS,
    SVD_KS,
    SVD_PS,
    mo,
    np,
    ppmi_matrix,
    row_slice,
    row_word,
    separation,
    split_counts,
    unit_rows,
):
    _variants = [(_k, _p) for _k in SVD_KS for _p in SVD_PS]
    _collected = {_v: {"a_zeit": [], "a_wort": []} for _v in _variants}
    svd_min_rank = len(row_slice)

    for _r in mo.status.progress_bar(range(N_SPLITS), title="SVD-Varianten"):
        _counts_a, _counts_b = split_counts(_r)
        _halves = [ppmi_matrix(_counts_a, row_slice), ppmi_matrix(_counts_b, row_slice)]
        _empty = [_h.sum(axis=1) == 0 for _h in _halves]
        _per_direction = []
        for _fit in (0, 1):
            _f, _o = _halves[_fit], _halves[1 - _fit]
            # SVD of the fitting half via its Gram matrix: F F^T = U S^2 U^T, V = F^T U S^-1
            _ev, _u = np.linalg.eigh(_f @ _f.T)
            _order = np.argsort(_ev)[::-1]
            _u, _s = _u[:, _order], np.sqrt(np.clip(_ev[_order], 0, None))
            _rank = int((_s > 1e-8 * _s[0]).sum())
            svd_min_rank = min(svd_min_rank, _rank)
            _o_ft = _o @ _f.T
            _res = {}
            for _k, _p in _variants:
                if _k > _rank:
                    _res[(_k, _p)] = None
                    continue
                # fitting half: U_k S_k^p, as in emb
                _x_fit = _u[:, :_k] * _s[:_k] ** _p
                # other half folded in: m V_k S_k^(p-1) = (O F^T) U_k S_k^(p-2)
                _x_other = (_o_ft @ _u[:, :_k]) * _s[:_k] ** (_p - 2)
                _x_fit[_empty[_fit]] = 0
                _x_other[_empty[1 - _fit]] = 0
                _x_a, _x_b = (_x_fit, _x_other) if _fit == 0 else (_x_other, _x_fit)
                _res[(_k, _p)] = separation(unit_rows(_x_a), unit_rows(_x_b), row_word, row_slice)
            _per_direction.append(_res)
        # mean of both directions
        for _v in _variants:
            _d1, _d2 = _per_direction[0][_v], _per_direction[1][_v]
            for _c in ("a_zeit", "a_wort"):
                _collected[_v][_c].append(
                    np.full(len(row_slice), np.nan) if _d1 is None or _d2 is None else (_d1[_c] + _d2[_c]) / 2
                )

    # only ranks that exist in every half of every split
    svd_results = {
        _v: {_c: np.array(_x) for _c, _x in _d.items()}
        for _v, _d in _collected.items()
        if _v[0] <= svd_min_rank
    }
    return svd_min_rank, svd_results


@app.cell(hide_code=True)
def _(
    CRITERIA,
    N_SPLITS,
    REASONS,
    SVD_KS,
    SVD_K_REFERENCE,
    SVD_PS,
    SVD_P_REFERENCE,
    change_per_split,
    change_table,
    classify,
    describe_change,
    fmt_num,
    fmt_pct,
    group_masks,
    mean_per_split,
    mo,
    np,
    pl,
    reference_ppmi,
    select_variant,
    show_table,
    svd_min_rank,
    svd_results,
    svd_retained,
):
    def _label(v):
        return f"k = {v[0]}, p = {fmt_num(v[1], digits=1)}"

    _all = group_masks["alle Zeilen"]
    _ref = (SVD_K_REFERENCE, SVD_P_REFERENCE)
    _ks = sorted({_v[0] for _v in svd_results})
    _dropped = [_k for _k in SVD_KS if _k not in _ks]
    _candidates = [_v for _v in svd_results if _v != _ref]
    _changes = {_v: change_per_split(svd_results[_v], svd_results[_ref], group_masks) for _v in _candidates}
    _classes = {_v: classify(_c) for _v, _c in _changes.items()}

    def _mean_all(res, c):
        return float(np.mean(mean_per_split(res[c], _all)))

    _long = pl.DataFrame({
        "k": [_v[0] for _v in svd_results],
        "p": [f"p = {fmt_num(_v[1], digits=1)}" for _v in svd_results],
        **{_c: [round(_mean_all(svd_results[_v], _c), 4) for _v in svd_results] for _c in CRITERIA},
        "Einstufung": ["Referenz" if _v == _ref else _classes[_v] for _v in svd_results],
    })
    _pivots = [
        _long.pivot(on="p", index="k", values=_col).sort("k")
        for _col in ("a_zeit", "a_wort", "Einstufung")
    ]

    _paragraphs = []
    _best, _equivalent = select_variant(svd_results, _changes, _classes, _all)

    # pre-check
    _text = (
        f"**Vorprüfung.** Die Referenz (k = {SVD_K_REFERENCE}) erhält {fmt_pct(svd_retained[SVD_K_REFERENCE])} "
        "der Summe der quadrierten PPMI-Werte"
    )
    if _best is not None:
        _text += f", die empfohlene Variante (k = {_best[0]}) {fmt_pct(svd_retained[_best[0]])}"
    _paragraphs.append(_text + ".")

    # overview
    _counts = {_c: sum(1 for _v in _candidates if _classes[_v] == _c) for _c in REASONS | {"Verbesserung": ""}}
    _text = (
        f"**Split-half-Prüfung.** Referenz: {_label(_ref)}, a_zeit = {fmt_num(_mean_all(svd_results[_ref], 'a_zeit'))}, "
        f"a_wort = {fmt_num(_mean_all(svd_results[_ref], 'a_wort'))}. "
        f"Von {len(_candidates)} Varianten sind gegenüber der Referenz eingestuft als Verbesserung: {_counts['Verbesserung']}, "
        f"Zielkonflikt: {_counts['Zielkonflikt']}, Verschlechterung: {_counts['Verschlechterung']}, "
        f"kein belastbarer Unterschied: {_counts['kein belastbarer Unterschied']}."
    )
    if _dropped:
        _text += (
            f" Nicht berechenbar, weil der Rang in mindestens einer Hälfte nur {svd_min_rank} beträgt: "
            f"k = {', '.join(str(_k) for _k in _dropped)}."
        )
    _paragraphs.append(_text)

    # comparison with the PPMI vectors without SVD
    _ppmi_zeit = _mean_all(reference_ppmi, "a_zeit")
    _top = max(svd_results, key=lambda v: _mean_all(svd_results[v], "a_zeit"))
    _top_zeit = _mean_all(svd_results[_top], "a_zeit")
    _paragraphs.append(
        f"**Vergleich mit den PPMI-Vektoren ohne SVD:** a_zeit = {fmt_num(_ppmi_zeit)}, "
        f"a_wort = {fmt_num(_mean_all(reference_ppmi, 'a_wort'))}. Der höchste Wert von a_zeit unter den "
        f"SVD-Varianten beträgt {fmt_num(_top_zeit)} ({_label(_top)}). "
        + (
            "Keine SVD-Variante erreicht damit a_zeit der PPMI-Vektoren: Die Kürzung verringert in diesen Daten "
            "die Unterscheidbarkeit der Zeitscheiben."
            if _top_zeit < _ppmi_zeit else
            "Mindestens eine SVD-Variante erreicht oder übertrifft damit a_zeit der PPMI-Vektoren."
        )
    )

    _outputs = [
        mo.md("**Tabelle 3.1:** a_zeit je Kombination (alle Zeilen)"), show_table(_pivots[0]),
        mo.md("**Tabelle 3.2:** a_wort je Kombination (alle Zeilen)"), show_table(_pivots[1]),
        mo.md("**Tabelle 3.3:** Einstufung gegenüber der Referenz"), show_table(_pivots[2]),
    ]

    if _best is None:
        _kind = "danger"
        _rec = (
            f"**Empfehlung: Referenz beibehalten ({_label(_ref)}).** Keine der geprüften Kombinationen ist als "
            "Verbesserung eingestuft."
        )
    else:
        _paragraphs.append(f"Empfohlene Kombination: {_label(_best)}. {describe_change(_changes[_best])}")
        if _equivalent:
            _paragraphs.append("Gleichwertig in a_zeit: " + "; ".join(_label(_v) for _v in _equivalent) + ".")
        _edges = []
        if _best[0] == max(_ks):
            _edges.append(
                "k liegt am oberen Rand des untersuchten Bereichs"
                + (" (größeres k ist in den Hälften nicht berechenbar)" if _dropped else "; der Bereich sollte erweitert werden")
            )
        if _best[0] == min(_ks):
            _edges.append("k liegt am unteren Rand des untersuchten Bereichs; der Bereich sollte erweitert werden")
        if _best[1] == min(SVD_PS):
            _edges.append("p liegt am unteren Rand des untersuchten Bereichs; der Bereich sollte erweitert werden")
        if _edges:
            _paragraphs.append("**Hinweis:** " + "; ".join(_edges) + ".")
        _kind = "success"
        _rec = (
            f"**Empfehlung: {_label(_best)}.** a_zeit steigt gegenüber der Referenz über alle Zeilen um "
            f"{fmt_num(float(np.mean(_changes[_best]['alle Zeilen']['a_zeit'])), sign=True)}, und kein Kennwert "
            "sinkt in einer Gruppe belastbar."
        )
        _outputs += [
            mo.md("**Tabelle 3.4:** Änderung der empfohlenen Kombination gegenüber der Referenz nach Textmenge"),
            show_table(change_table(_changes[_best], N_SPLITS)),
        ]

    mo.vstack(_outputs + [
        mo.md("### Auswertung"),
        mo.md("\n\n".join(_paragraphs)),
        mo.callout(mo.md(_rec), kind=_kind),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Einschränkungen

    - Die Split-half-Prüfung misst, wie gut sich Zeilen unabhängig von der zufälligen Auswahl der Reden reproduzieren und unterscheiden lassen. Ob die Vektoren die Bedeutung zutreffend abbilden (Validität), misst sie nicht.
    - a_zeit erfasst jede reproduzierbare Eigenschaft einer Zeitscheibe, also auch solche, die nicht auf der Bedeutung des Zielworts beruhen, etwa die Themen eines Quartals oder die Textmenge. a_wort begrenzt diesen Einfluss nur teilweise.
    - Die Abschnitte 1 und 2 prüfen die PPMI-Vektoren vor der SVD, Abschnitt 3 prüft die SVD auf PPMI ohne Glättung. Wechselwirkungen zwischen Glättung und SVD sind nicht geprüft.
    - In Abschnitt 3 wird jeweils eine Hälfte in den Raum der anderen projiziert. Die Kennwerte beider Richtungen werden gemittelt.
    - Die Matrix enthält wenige Zielwörter. a_wort beruht je Zeile auf höchstens so vielen Vergleichszeilen, wie es weitere Zielwörter in der Zeitscheibe gibt.

    ## Literatur

    - Antoniak, M. & Mimno, D. (2018): Evaluating the stability of embedding-based word similarities. *Transactions of the Association for Computational Linguistics* 6, 107–119.
    - Berry, M. W., Dumais, S. T. & O'Brien, G. W. (1995): Using linear algebra for intelligent information retrieval. *SIAM Review* 37(4), 573–595.
    - Brown, W. (1910): Some experimental results in the correlation of mental abilities. *British Journal of Psychology* 3(3), 296–322.
    - Bullinaria, J. A. & Levy, J. P. (2012): Extracting semantic representations from word co-occurrence statistics: Stop-lists, stemming, and SVD. *Behavior Research Methods* 44(3), 890–907.
    - Caron, J. (2001): Experiments with LSA scoring: Optimal rank and basis. In: Berry, M. W. (Hrsg.): *Computational Information Retrieval*. SIAM, 157–169.
    - Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. & Harshman, R. (1990): Indexing by latent semantic analysis. *Journal of the American Society for Information Science* 41(6), 391–407.
    - Eckart, C. & Young, G. (1936): The approximation of one matrix by another of lower rank. *Psychometrika* 1(3), 211–218.
    - Hanley, J. A. & McNeil, B. J. (1982): The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 143(1), 29–36.
    - Hastie, T., Tibshirani, R. & Friedman, J. (2009): *The Elements of Statistical Learning*. 2. Aufl. Springer.
    - Haxby, J. V., Gobbini, M. I., Furey, M. L., Ishai, A., Schouten, J. L. & Pietrini, P. (2001): Distributed and overlapping representations of faces and objects in ventral temporal cortex. *Science* 293(5539), 2425–2430.
    - Levy, O., Goldberg, Y. & Dagan, I. (2015): Improving distributional similarity with lessons learned from word embeddings. *Transactions of the Association for Computational Linguistics* 3, 211–225.
    - Mann, H. B. & Whitney, D. R. (1947): On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics* 18(1), 50–60.
    - Manning, C. D. & Schütze, H. (1999): *Foundations of Statistical Natural Language Processing*. MIT Press.
    - Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. (2013): Distributed representations of words and phrases and their compositionality. *Advances in Neural Information Processing Systems* 26, 3111–3119.
    - Österlund, A., Ödling, D. & Sahlgren, M. (2015): Factorization of latent variables in distributional semantic models. *Proceedings of EMNLP 2015*, 227–231.
    - Spearman, C. (1910): Correlation calculated from faulty data. *British Journal of Psychology* 3(3), 271–295.
    - Turney, P. D. & Pantel, P. (2010): From frequency to meaning: Vector space models of semantics. *Journal of Artificial Intelligence Research* 37, 141–188.

    Die bibliografischen Angaben sind vor einer Veröffentlichung an den Originalen zu prüfen.
    """)
    return


if __name__ == "__main__":
    app.run()
