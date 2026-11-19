import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import sys

    # BLAS threads: with the default (one thread per logical core) np.linalg.eigh on the 2169 x 2169 Gram
    # matrices of section 3 took 54 s instead of 5 s with 4 threads (oversubscription of the cores; svds in
    # sections 1 and 2 is hardly affected, 6.6 vs. 7.3 s). OpenBLAS reads the variable only when it is loaded,
    # so it must be set before numpy is imported; a value set from outside takes precedence.
    BLAS_THREADS_PRESET = "numpy" not in sys.modules
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    import marimo as mo
    import numpy as np
    import polars as pl
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds

    return BLAS_THREADS_PRESET, json, mo, np, pl, sp, svds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diagnose zu pipeline_emb_ppmi.py

    Dieses Notebook prüft Verfahrensentscheidungen der Embedding-Stufe an den Daten:

    1. Reliabilität einer Zeile in Abhängigkeit von ihrer Trefferzahl (Mindesttrefferzahl)
    2. Einfluss schwach belegter Zeilen auf den Raum (SVD ohne diese Zeilen, *folding-in*)
    3. Rang k der SVD und Exponent p der Eigenwertgewichtung (Referenz: k und p des letzten Laufs von emb)
    4. Abhängigkeit der Drift zwischen benachbarten Zeitscheiben von der Trefferzahl (Teilstichproben, Permutationserwartung)

    Alle Abschnitte verwenden dasselbe Messverfahren, das vorab einmal beschrieben wird. Jeder Abschnitt ist gleich aufgebaut: Verfahren, Fragestellung, Methode, erwartete Ergebnisse und ihre Deutung, Entscheidungsregel, Vorprüfung, Berechnung, automatische Auswertung mit Empfehlung, Einschränkungen, Literatur. Die Auswertung leitet ihre Aussagen ausschließlich aus den Werten ab, die im selben Lauf berechnet werden. Ergebnisse früherer Messungen werden nicht übernommen.

    Das frühere Diagnose-Notebook (`pipeline_emb_ppmi_diagnose_alt.py`, Glättung und Rang der SVD) bleibt unverändert erhalten.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Daten

    Grundlage ist die Belegliste aus pre (`kwic_hits_sliced_<Variante>_<Raster>.parquet`): eine Zeile je Beleg (KWIC-Treffer) mit Rede-ID, Zeitscheibe, Zielwort und den Lemmata des Kontexts. Die Kookkurrenzen werden auf dieselbe Weise gezählt wie in pre: Jedes Vorkommen eines Kontextlemmas in einem Beleg zählt einmal.

    Die Belegliste wird an Stelle der Kookkurrenzdatei verwendet, weil die folgenden Prüfungen die Belege nach Reden aufteilen und einzelne Belege ziehen. Die Kookkurrenzdatei enthält nur Summen je Zeitscheibe. Zur Kontrolle vergleicht der Code die eigene Zählung mit der Kookkurrenzdatei aus pre. Weichen die beiden voneinander ab, bricht das Notebook ab.

    Rang k der SVD und Exponent p der Eigenwertgewichtung werden aus den Metadaten des letzten Laufs von emb gelesen (`emb_meta_<Projekt>_<Variante>_<Raster>.json`), damit die Diagnose dieselben Werte verwendet wie die Pipeline.

    Begriffe wie in emb: Eine **Zeile** der Matrix ist ein Paar aus Zielwort und Zeitscheibe, z. B. „Sicherheit, 2022-H1“. Ein **Kontextwort** ist eine Spalte der Matrix. Die **Trefferzahl** einer Zeile ist die Zahl ihrer Belege (`n_hits` in emb).
    """)
    return


@app.cell
def _(json, mo, np, pl, sp):
    # input selection, same naming as in pipeline_emb_ppmi.py
    PROJECT = "bundestag"
    KWIC_VARIANT = "punkt"
    SLICE_GRID = "halfyear"
    _run = f"{PROJECT}_{KWIC_VARIANT}_{SLICE_GRID}"

    # the same files as pipeline_emb_ppmi.py reads (input/), and its results (output/)
    _pre = mo.notebook_dir() / "input" / KWIC_VARIANT
    _emb = mo.notebook_dir() / "output" / KWIC_VARIANT

    # k and p as used in the last run of emb
    _meta = json.loads((_emb / f"emb_meta_{_run}.json").read_text(encoding="utf-8"))
    SVD_K = int(_meta["k"])
    SVD_P = float(_meta["eig_p"])

    # one row per KWIC hit; h = hit index
    hits = (
        pl.read_parquet(_pre / f"kwic_hits_sliced_{_run}.parquet")
        .select(["id", "slice", "target_lemma", "context_lemmas"])
        .with_row_index("h")
    )

    # one row per context token, as in pre
    _tokens = (
        hits
        .select(["h", "slice", "target_lemma", "context_lemmas"])
        .explode("context_lemmas", empty_as_null=False)
        .rename({"context_lemmas": "context_lemma"})
        .drop_nulls("context_lemma")
    )

    # consistency check: the counts must equal pre's co-occurrence file exactly
    _keys = ["slice", "target_lemma", "context_lemma"]
    _own = _tokens.group_by(_keys).len().rename({"len": "n_cooc"})
    _pre_counts = pl.read_csv(
        _pre / f"cooc_counts_{_run}.csv",
        schema_overrides={"slice": pl.Utf8, "context_lemma": pl.Utf8},
    )
    _mismatch = (
        _own.join(_pre_counts, on=_keys, how="full", coalesce=True, suffix="_pre")
        .filter(pl.col("n_cooc").ne_missing(pl.col("n_cooc_pre")))
    )
    mo.stop(
        _mismatch.height > 0,
        mo.callout(
            mo.md(
                f"**Abbruch:** {_mismatch.height} Zählungen weichen von "
                f"`cooc_counts_{_run}.csv` ab. Belegliste und Kookkurrenzdatei "
                "stammen nicht aus demselben Lauf von pre."
            ),
            kind="danger",
        ),
    )

    # a speech belongs to exactly one slice; the random split below relies on it
    assert hits.select(["id", "slice"]).unique().height == hits["id"].n_unique()

    # rows: target word x slice, n_hits counted over all hits as in emb
    _row_keys = ["target_lemma", "slice"]
    rows = (
        hits.group_by(_row_keys).len().rename({"len": "n_hits"})
        .sort(_row_keys).with_row_index("i")
    )
    _columns = _tokens.select("context_lemma").unique().sort("context_lemma").with_row_index("j")
    n_rows = rows.height
    n_columns = _columns.height

    _hit_index = hits.join(rows.select(["i", *_row_keys]), on=_row_keys, how="left").sort("h")
    hit_row = _hit_index["i"].to_numpy()
    hit_speech = _hit_index["id"].cast(pl.Categorical).to_physical().to_numpy()

    row_word = rows["target_lemma"].to_numpy()
    slice_labels = sorted(rows["slice"].unique().to_list())
    row_slice = rows["slice"].replace_strict(slice_labels, list(range(len(slice_labels)))).to_numpy()
    row_hits = rows["n_hits"].to_numpy()

    # sparse count matrix hits x context words; every matrix below is built from it
    _hc = _tokens.group_by(["h", "context_lemma"]).len().join(_columns, on="context_lemma")
    hit_counts = sp.csr_matrix(
        (_hc["len"].to_numpy().astype(np.float64), (_hc["h"].to_numpy(), _hc["j"].to_numpy())),
        shape=(hits.height, n_columns),
    )

    def _de(n):
        return f"{n:,}".replace(",", ".")

    mo.md(
        f"Projekt `{PROJECT}`, Variante `{KWIC_VARIANT}`, Raster `{SLICE_GRID}`. "
        f"Belege: {_de(hits.height)} aus {_de(hits['id'].n_unique())} Reden. "
        f"Matrix: {_de(n_rows)} Zeilen ({rows['target_lemma'].n_unique()} Zielwörter × "
        f"{len(slice_labels)} Zeitscheiben, nicht jede Kombination belegt), {_de(n_columns)} Kontextwörter. "
        f"SVD aus emb: k = {SVD_K}, p = {SVD_P:g}. "
        f"Die Zählungen stimmen mit `cooc_counts_{_run}.csv` überein."
    )
    return (
        SVD_K,
        SVD_P,
        hit_counts,
        hit_row,
        hit_speech,
        hits,
        n_rows,
        row_hits,
        row_slice,
        row_word,
        slice_labels,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Messverfahren

    ### Grundgedanke: Split-half-Prüfung

    Die Reden jeder Zeitscheibe werden zufällig in zwei gleich große Hälften A und B geteilt. Aus jeder Hälfte wird eine eigene PPMI-Matrix berechnet, mit denselben Randsummen je Zeitscheibe wie in emb. Beide Hälften stammen aus derselben Zeitscheibe. Unterschiede zwischen ihnen sind deshalb Zufallsschwankungen der Stichprobe und kein Bedeutungswandel. Eine Zeile ist reliabel, wenn sie in beiden Hälften ähnlich abgebildet wird und sich zugleich von anderen Zeilen unterscheidet. Das Vorgehen folgt der Split-half-Methode zur Schätzung der Reliabilität (Spearman 1910; Brown 1910). Die Stabilität von Wortvektoren wird in der Literatur ebenfalls durch Neuziehen ganzer Dokumente geprüft (Antoniak & Mimno 2018).

    Geteilt wird nach Reden, nicht nach einzelnen Belegen. Belege aus derselben Rede sind nicht unabhängig voneinander (derselbe Redner, dasselbe Thema, wiederholte Formulierungen). Lägen sie in beiden Hälften, wären die Hälften einander künstlich ähnlich.

    Die Teilung wird mehrfach mit verschiedenen Zufallszahlen wiederholt (Anzahl: `N_SPLITS` unten). Alle Abschnitte verwenden dieselben Teilungen.

    ### SVD und *folding-in*

    Gemessen wird an den Vektoren nach der SVD, weil sie die Grundlage aller späteren Schritte sind. Die SVD wird an einer Hälfte berechnet (k und p wie in emb). Die Zeilen beider Hälften werden anschließend in diesen Raum projiziert (*folding-in*, Deerwester et al. 1990; Berry, Dumais & O'Brien 1995): Eine Zeile $m$ der PPMI-Matrix erhält den Vektor $m \, V_k \, \Sigma_k^{p-1}$. Für die Zeilen der Hälfte, an der die SVD berechnet wurde, ist das identisch mit $U_k \Sigma_k^p$ wie in emb. Eine SVD der vollständigen Daten würde die beiden Hälften künstlich ähnlich machen, weil beide in dieselbe Zerlegung eingehen. Die Rechnung wird in beiden Richtungen durchgeführt (SVD an A, dann an B), die Kennwerte werden gemittelt.

    ### Kennwerte

    Für jede Zeile wird die Kosinusähnlichkeit ihrer Hälfte A zu ihrer eigenen Hälfte B berechnet. Diese Ähnlichkeit wird mit der Ähnlichkeit zu Vergleichszeilen verglichen, ebenfalls über die Hälften hinweg (Mittelwert aus „Hälfte A der Zeile mit Hälfte B der Vergleichszeile“ und umgekehrt). Es gibt zwei Arten von Vergleichszeilen:

    - **a_wort**: Die Vergleichszeilen sind die anderen Zielwörter derselben Zeitscheibe. a_wort ist der Anteil dieser Vergleichszeilen, denen die Zeile weniger ähnlich ist als ihrer eigenen zweiten Hälfte. Ein hoher Wert bedeutet, dass das Profil der Zeile reproduzierbar ist und sie sich von anderen Zielwörtern unterscheiden lässt.
    - **a_zeit**: Die Vergleichszeilen sind die anderen Zeitscheiben desselben Zielworts. Ein hoher Wert bedeutet, dass sich das, was eine Zeitscheibe von den anderen Zeitscheiben desselben Worts unterscheidet, in beiden unabhängigen Hälften zeigt.

    Beide Werte liegen zwischen 0 und 1. Bei 0,5 ist die eigene zweite Hälfte nicht ähnlicher als eine beliebige Vergleichszeile; bei 1 ist sie ähnlicher als alle Vergleichszeilen. Gleiche Ähnlichkeiten zählen zur Hälfte. Der Wert entspricht der Fläche unter der ROC-Kurve (AUC): der Wahrscheinlichkeit, dass der zutreffende Fall eine höhere Ähnlichkeit erhält als ein zufällig gewählter Vergleichsfall (Hanley & McNeil 1982). Er ist identisch mit der normierten Mann-Whitney-U-Statistik (Mann & Whitney 1947). Die Gegenüberstellung der Ähnlichkeit innerhalb derselben Einheit und zwischen verschiedenen Einheiten über zwei Datenhälften hinweg folgt Haxby et al. (2001).

    Die beiden Kennwerte messen Verschiedenes. a_wort erwartet bei jeder ausreichend belegten Zeile einen hohen Wert, weil verschiedene Zielwörter verschiedene Kontexte haben. a_zeit erwartet einen hohen Wert nur, wenn sich das Wort zwischen den Zeitscheiben tatsächlich unterscheidet. Ein Wort mit gleichbleibender Verwendung hat zu Recht ein a_zeit nahe 0,5, unabhängig von seiner Trefferzahl.

    Eine Zeile, die in einer Hälfte keinen Beleg oder keinen von null verschiedenen Vektor hat, entfällt in dieser Teilung.

    ### Begriffe

    - **Kosinusähnlichkeit**: Ähnlichkeit zweier Vektoren nach ihrer Richtung, unabhängig von ihrer Länge.
    - **Teilung**: eine zufällige Aufteilung der Reden in die Hälften A und B.
    """)
    return


@app.cell
def _(mo, np, row_slice, row_word, slice_labels, sp, svds):
    def show_table(df):
        """Table with all rows on one page."""
        return mo.ui.table(df, pagination=False, selection=None, show_column_summaries=False, show_data_types=False)

    def counts_from_hits(hit_counts, selected, target, n_out):
        """Count matrix (n_out x context words): the counts of hit selected[t] are added to row target[t]."""
        S = sp.csr_matrix(
            (np.ones(len(selected)), (target, selected)), shape=(n_out, hit_counts.shape[0])
        )
        return (S @ hit_counts).tocsr()

    def slice_marginals(C):
        """Column totals per slice (slices x context words, dense) and grand total per slice,
        from a count matrix whose rows are all matrix rows (order as in `rows`)."""
        n = C.shape[0]
        S = sp.csr_matrix((np.ones(n), (row_slice, np.arange(n))), shape=(len(slice_labels), n))
        Z = np.asarray((S @ C).todense())
        return Z, Z.sum(axis=1)

    def ppmi(C, slices, Z, T, replaced=None):
        """PPMI as in pipeline_emb_ppmi.py: log2(observed / expected), clipped at 0.
        C: counts (rows x context words), slices: slice index per row of C,
        Z, T: column totals and grand total per slice (see slice_marginals).
        replaced: counts of the full rows that the rows of C replace (subsamples); the marginals are then
        those of the matrix in which each full row is replaced by its subsample."""
        C = C.tocsr()
        row_total = np.asarray(C.sum(axis=1)).ravel()
        coo = C.tocoo()
        s = slices[coo.row]
        z, t = Z[s, coo.col], T[s]
        if replaced is not None:
            # subtract what the full row contributes beyond its subsample
            rest = (replaced.tocsr() - C).tocsr()
            z = z - np.asarray(rest[coo.row, coo.col]).ravel()
            t = t - np.asarray(rest.sum(axis=1)).ravel()[coo.row]
        with np.errstate(divide="ignore"):
            pmi = np.log2(coo.data * t / (row_total[coo.row] * z))
        M = sp.csr_matrix((np.maximum(pmi, 0), (coo.row, coo.col)), shape=C.shape)
        M.eliminate_zeros()
        return M

    def fit_space(M, k, seed):
        """Truncated SVD of M: first k right singular vectors (context words x k) and singular values,
        in descending order."""
        _, s, vt = svds(M, k=k, rng=np.random.default_rng(seed))
        order = np.argsort(s)[::-1]
        return vt[order].T, s[order]

    def project(M, V, s, p):
        """Folding-in: m V_k S_k^(p-1); equals U_k S_k^p for the rows the SVD was fitted on."""
        return np.asarray(M @ V) * s ** (p - 1)

    def unit_rows(X):
        """Rows scaled to length 1; rows without any non-zero value become NaN (cosine undefined)."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(norms > 0, X / norms, np.nan).astype(np.float32)

    def separation(QA, QB, query_rows, XA, XB, block=2000):
        """a_wort and a_zeit per query.
        QA, QB: unit vectors of the queries in half A and half B; query_rows: their row index.
        XA, XB: unit vectors of all rows in half A and half B (comparison rows).
        Share of comparison rows that are less similar to the query than its own other half;
        ties count half."""
        out = {"a_wort": np.full(len(query_rows), np.nan), "a_zeit": np.full(len(query_rows), np.nan)}
        for b in range(0, len(query_rows), block):
            q = slice(b, b + block)
            r = query_rows[q]
            own = np.einsum("ij,ij->i", QA[q], QB[q])
            comp = (QA[q] @ XB.T + QB[q] @ XA.T) / 2
            same_word = row_word[r][:, None] == row_word[None, :]
            same_slice = row_slice[r][:, None] == row_slice[None, :]
            for name, mask in (("a_wort", same_slice & ~same_word), ("a_zeit", same_word & ~same_slice)):
                valid = mask & ~np.isnan(comp)
                less = ((comp < own[:, None]) & valid).sum(axis=1)
                ties = ((comp == own[:, None]) & valid).sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    res = (less + 0.5 * ties) / valid.sum(axis=1)
                res[np.isnan(own)] = np.nan
                out[name][q] = res
        return out

    def fmt_num(x, digits=3):
        return "–" if x is None or np.isnan(x) else f"{x:.{digits}f}".replace("-", "−").replace(".", ",")

    def fmt_int(n):
        return f"{int(n):,}".replace(",", ".")

    def fmt_pct(x):
        return f"{x * 100:.0f} %".replace(".", ",")

    def first_stable(per_split, level, strict=False, usable=None):
        """Smallest column index g such that every split reaches the level in column g and in every
        usable column after it; None if there is none. per_split: splits x columns."""
        ok = (per_split > level) if strict else (per_split >= level)
        ok = ok.all(axis=0)
        usable = np.ones(per_split.shape[1], dtype=bool) if usable is None else usable
        for g in range(per_split.shape[1]):
            if usable[g] and all(ok[x] for x in range(g, per_split.shape[1]) if usable[x]):
                return g
        return None

    return (
        counts_from_hits,
        first_stable,
        fit_space,
        fmt_int,
        fmt_num,
        fmt_pct,
        ppmi,
        project,
        separation,
        show_table,
        slice_marginals,
        unit_rows,
    )


@app.cell
def _(
    counts_from_hits,
    hit_counts,
    hit_row,
    hits,
    n_rows,
    np,
    pl,
    ppmi,
    row_slice,
    slice_marginals,
):
    N_SPLITS = 20
    SEED = 42

    # random splits, shared by all sections: half of the speeches of each slice go to half A.
    # in_a[r] is a boolean per hit (order of `hits`).
    _rng = np.random.default_rng(SEED)
    _speeches = hits.select(["id", "slice"]).unique().sort("id")
    in_a = []
    for _ in range(N_SPLITS):
        _split = (
            _speeches
            .with_columns(pl.Series("key", _rng.random(_speeches.height)))
            .with_columns(
                (pl.col("key").rank("ordinal").over("slice") <= pl.len().over("slice") / 2).alias("in_a")
            )
        )
        in_a.append(
            hits.select(["h", "id"]).join(_split.select(["id", "in_a"]), on="id", how="left")
            .sort("h")["in_a"].to_numpy()
        )
    assert all(len(_x) == len(hit_row) for _x in in_a)

    def half_matrices(r):
        """Both halves of split r: selected hits, count matrices, slice marginals, PPMI matrices
        and hits per row (each a list [half A, half B])."""
        sel = [np.flatnonzero(in_a[r]), np.flatnonzero(~in_a[r])]
        C = [counts_from_hits(hit_counts, s, hit_row[s], n_rows) for s in sel]
        marg = [slice_marginals(c) for c in C]
        M = [ppmi(C[x], row_slice, *marg[x]) for x in (0, 1)]
        n_half = [np.bincount(hit_row[s], minlength=n_rows) for s in sel]
        return sel, C, marg, M, n_half

    return N_SPLITS, SEED, half_matrices


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Reliabilität in Abhängigkeit von der Trefferzahl

    ### Verfahren

    In emb geht jede Zeile in die SVD ein, unabhängig davon, auf wie vielen Belegen sie beruht. Das PPMI-Profil einer Zeile mit wenigen Belegen hängt stark davon ab, welche Belege zufällig vorliegen. Bei kleiner Randsumme der Zeile erhalten die wenigen beobachteten Kontexte zudem hohe PMI-Werte (zur Überschätzung seltener Ereignisse durch PMI: Levy, Goldberg & Dagan 2015). Solche Zeilen können im Raum als Ausreißer oder als scheinbar starke Verschiebung erscheinen.

    Vorgesehen ist eine dokumentierte Mindesttrefferzahl: Zeilen mit weniger Belegen sollen die SVD nicht mitbestimmen. Ob ein solcher Eingriff die übrigen Zeilen verbessert, prüft Abschnitt 2. Dieser Abschnitt bestimmt, wo die Schwelle liegt.

    ### Fragestellung

    Ab welcher Trefferzahl ist das Profil einer Zeile so reproduzierbar, dass sie sich in zwei unabhängigen Stichproben als dieselbe Zeile wiedererkennen lässt?

    ### Methode

    **Kontrollierte Teilstichproben (Hauptmessung).** Ausgewählt werden die gut belegten Zeilen: Zeilen mit mindestens `H_MAX` Belegen in jeder der beiden Hälften einer Teilung (Prüfzeilen). Für jede Prüfzeile werden in jeder Hälfte genau h Belege gezogen, für jedes h aus `H_GRID`. Aus diesen h Belegen wird die Zeile neu berechnet (PPMI mit den Randsummen der Hälfte, dann *folding-in*) und ihre Kennwerte gegen die übrigen Zeilen bestimmt. Die Vergleichszeilen bleiben dabei unverändert. Dieselbe Zeile wird so bei verschiedenen Trefferzahlen gemessen. Unterschiede zwischen den h gehen damit allein auf die Trefferzahl zurück und nicht darauf, welche Wörter selten sind.

    Gezogen wird nach Reden, wie bei der Teilung: Die Reden der Zeile werden in zufällige Reihenfolge gebracht und ihre Belege so lange übernommen, bis h Belege erreicht sind; aus der letzten Rede werden zufällig so viele Belege genommen, wie noch fehlen. So bleibt die Bündelung der Belege in Reden erhalten, wie sie in schwach belegten Zeilen tatsächlich vorkommt. Die Stichproben sind geschachtelt: Die Stichprobe für ein kleineres h ist Teil der Stichprobe für ein größeres h. Das verringert die Zufallsschwankung zwischen benachbarten Werten von h.

    **Natürliche Zeilen (Gegenprobe).** Alle Zeilen werden mit ihren tatsächlichen Belegen gemessen und nach ihrer Trefferzahl in der Hälfte gruppiert (die kleinere der beiden Hälften; Gruppengrenzen aus `H_GRID`). Diese Messung zeigt, wie sich die tatsächlich vorkommenden schwach belegten Zeilen verhalten. Sie vermischt aber die Trefferzahl mit den Eigenschaften der jeweiligen Wörter.

    **Bezug auf die vollständigen Daten.** Eine Hälfte mit h Belegen ist eine Stichprobe von h Belegen, ebenso wie eine Zeile der vollständigen Daten mit h Belegen. Die Kurve wird deshalb direkt als Funktion der Trefferzahl einer Zeile gelesen; eine Umrechnung von der halben auf die volle Datenmenge ist nicht nötig. Die Spearman-Brown-Formel (Spearman 1910; Brown 1910) wäre dafür auch nicht anwendbar, weil sie für Korrelationskoeffizienten gilt und nicht für AUC-Werte.

    **Kennwert für die Entscheidung: a_zeit.** Das Problem schwach belegter Zeilen besteht darin, dass sie wie eine Verschiebung desselben Worts erscheinen. Ob eine Zeile sich über den Zufall der Stichprobe hinaus von den anderen Zeitscheiben desselben Worts abhebt, misst a_zeit. a_wort misst die leichtere Aufgabe, ein Zielwort von anderen Zielwörtern zu unterscheiden, und wird zur Information mitberichtet. Bei einem einzelnen Wort mit gleichbleibender Verwendung liegt a_zeit auch bei guter Belegung nahe 0,5 (s. Messverfahren). Die Entscheidung beruht deshalb auf dem Mittelwert über die Prüfzeilen, und sie setzt voraus, dass dieser Mittelwert mit allen Belegen der Hälfte deutlich über dem Niveau liegt; die Auswertung prüft das.

    ### Erwartete Ergebnisse und ihre Deutung

    - a_zeit steigt mit h und nähert sich dem Wert, den die Prüfzeilen mit allen Belegen ihrer Hälfte erreichen. Das Ergebnis ist eine Schwelle h\*, ab der das Niveau erreicht wird.
    - Erreicht a_zeit das Niveau schon beim kleinsten h, ist für dieses Niveau keine Mindesttrefferzahl nötig.
    - Erreichen die Prüfzeilen das Niveau auch mit allen Belegen ihrer Hälfte nicht, ist das Niveau für diese Daten und diese SVD nicht erreichbar. Die Schwelle lässt sich dann nach dieser Regel nicht bestimmen.
    - Liegen die natürlichen Zeilen bei gleicher Trefferzahl unter den Teilstichproben, sind die tatsächlich schwach belegten Zeilen schlechter wiederzuerkennen, als ihre Trefferzahl allein erwarten lässt (etwa weil ihre Belege aus wenigen Reden stammen oder weil es Wörter sind, die sich zwischen den Zeitscheiben wenig unterscheiden). h\* aus den Teilstichproben ist dann eine Untergrenze. Liegen sie darüber, ist h\* vorsichtig gewählt.

    ### Entscheidungsregel

    - **Schwelle h\***: der kleinste Wert h in `H_GRID`, bei dem der Mittelwert von a_zeit über die Prüfzeilen in **allen** Teilungen mindestens `AUC_LEVEL` beträgt, und ebenso bei jedem größeren h. Voreingestellt ist 0,7, die übliche Grenze für eine akzeptable Trennschärfe einer AUC (Hosmer & Lemeshow 2000). Das Niveau ist eine Konvention und muss als solche dokumentiert werden.
    - Voraussetzung: Die Prüfzeilen erreichen mit allen Belegen ihrer Hälfte das Niveau in allen Teilungen. Sonst ist keine Schwelle bestimmbar.
    - **Zufallsgrenze h₀** (zur Information): derselbe Wert für das Kriterium „a_zeit über 0,5 in allen Teilungen“, d. h. die Zeile hebt sich überhaupt von den anderen Zeitscheiben desselben Worts ab.
    - Die Gegenprobe wird nach derselben Regel ausgewertet, über die Gruppen statt über h. Gruppen mit im Mittel weniger als `MIN_ROWS_GROUP` Zeilen gehen nicht in die Regel ein. Ergibt die Gegenprobe einen höheren Wert, nennt die Auswertung beide Werte.
    - Empfohlen wird h\* als Mindesttrefferzahl einer Zeile der vollständigen Daten.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vorprüfung

    Wie viele Zeilen und Belege lägen unter einer Schwelle? Die Tabelle zeigt für jeden Wert aus `H_GRID` die Zahl der Zeilen der vollständigen Daten mit weniger Belegen, ihren Anteil an allen Zeilen und den Anteil ihrer Belege an allen Belegen.
    """)
    return


@app.cell
def _(fmt_int, fmt_pct, pl, row_hits, show_table):
    # candidate thresholds (hits per row); the largest value also defines the probe rows
    H_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    H_MAX = max(H_GRID)

    show_table(pl.DataFrame({
        "Schwelle (Belege)": [str(_h) for _h in H_GRID],
        "Zeilen darunter": [fmt_int((row_hits < _h).sum()) for _h in H_GRID],
        "Anteil Zeilen": [fmt_pct((row_hits < _h).mean()) for _h in H_GRID],
        "Anteil Belege": [fmt_pct(row_hits[row_hits < _h].sum() / row_hits.sum()) for _h in H_GRID],
    }))
    return H_GRID, H_MAX


@app.cell
def _(
    H_GRID,
    H_MAX,
    N_SPLITS,
    SEED,
    SVD_K,
    SVD_P,
    counts_from_hits,
    fit_space,
    half_matrices,
    hit_counts,
    hit_row,
    hit_speech,
    mo,
    n_rows,
    np,
    pl,
    ppmi,
    project,
    row_slice,
    row_word,
    separation,
    unit_rows,
):
    AUC_LEVEL = 0.7
    MIN_PROBES = 30        # fewer probe rows in any split: stop, H_MAX too large for these data
    MIN_ROWS_GROUP = 10    # groups of the cross-check with fewer rows (mean over splits) are not used

    _crit = ("a_wort", "a_zeit")

    def _nested_rank(selected, rng):
        """Rank of each selected hit within its row: speeches in random order, hits within a speech
        in random order. The first h hits of a row form its sample of size h."""
        df = pl.DataFrame({
            "pos": np.arange(len(selected)),
            "row": hit_row[selected],
            "speech": hit_speech[selected],
            "k_hit": rng.random(len(selected)),
        })
        sk = df.select(["row", "speech"]).unique().sort(["row", "speech"])
        sk = sk.with_columns(pl.Series("k_speech", rng.random(sk.height)))
        return (
            df.join(sk, on=["row", "speech"])
            .sort(["row", "k_speech", "k_hit"])
            .with_columns(pl.int_range(pl.len()).over("row").alias("rank"))
            .sort("pos")["rank"].to_numpy()
        )

    # probe curve: splits x (H_GRID + "all hits of the half"); cross-check: splits x groups
    probe_curve = {_c: np.full((N_SPLITS, len(H_GRID) + 1), np.nan) for _c in _crit}
    natural_curve = {_c: np.full((N_SPLITS, len(H_GRID)), np.nan) for _c in _crit}
    natural_n = np.zeros((N_SPLITS, len(H_GRID)))
    probe_n = np.zeros(N_SPLITS, dtype=int)
    probe_words = np.zeros(N_SPLITS, dtype=int)
    # per-row values of the reference (SVD on all rows, as in emb), reused in section 2
    reference_scores = {_c: np.full((N_SPLITS, n_rows), np.nan) for _c in _crit}

    for _r in mo.status.progress_bar(range(N_SPLITS), title="Teilungen"):
        _rng = np.random.default_rng(SEED + 1000 + _r)
        _sel, _C, _marg, _M, _n_half = half_matrices(_r)

        # probe rows and their nested samples per half and h
        _probes = np.flatnonzero(np.minimum(*_n_half) >= H_MAX)
        mo.stop(
            len(_probes) < MIN_PROBES,
            mo.callout(
                mo.md(
                    f"**Abbruch:** In Teilung {_r + 1} haben nur {len(_probes)} Zeilen mindestens "
                    f"{H_MAX} Belege in jeder Hälfte (verlangt: {MIN_PROBES}). Den größten Wert in "
                    "`H_GRID` verkleinern."
                ),
                kind="danger",
            ),
        )
        probe_n[_r] = len(_probes)
        probe_words[_r] = len(np.unique(row_word[_probes]))
        _probe_pos = np.full(n_rows, -1)
        _probe_pos[_probes] = np.arange(len(_probes))
        _samples = []
        for _x in (0, 1):
            _ps = _sel[_x][_probe_pos[hit_row[_sel[_x]]] >= 0]
            _rank = _nested_rank(_ps, _rng)
            _samples.append([
                ppmi(
                    counts_from_hits(hit_counts, _ps[_rank < _h], _probe_pos[hit_row[_ps[_rank < _h]]], len(_probes)),
                    row_slice[_probes], *_marg[_x], replaced=_C[_x][_probes],
                )
                for _h in H_GRID
            ])

        # both directions: SVD fitted on half A, then on half B
        _probe_acc = {_c: np.zeros((len(H_GRID) + 1, len(_probes))) for _c in _crit}
        _nat_acc = {_c: np.zeros(n_rows) for _c in _crit}
        for _fit in (0, 1):
            _V, _s = fit_space(_M[_fit], SVD_K, SEED + _r)
            _X = [unit_rows(project(_M[_x], _V, _s, SVD_P)) for _x in (0, 1)]
            _nat = separation(_X[0], _X[1], np.arange(n_rows), _X[0], _X[1])
            for _g in range(len(H_GRID)):
                _QA = unit_rows(project(_samples[0][_g], _V, _s, SVD_P))
                _QB = unit_rows(project(_samples[1][_g], _V, _s, SVD_P))
                _res = separation(_QA, _QB, _probes, _X[0], _X[1])
                for _c in _crit:
                    _probe_acc[_c][_g] += _res[_c] / 2
            for _c in _crit:
                _probe_acc[_c][len(H_GRID)] += _nat[_c][_probes] / 2
                _nat_acc[_c] += _nat[_c] / 2

        # mean over probe rows; cross-check: mean per group of hits in the smaller half
        _group = np.searchsorted(H_GRID, np.minimum(*_n_half), side="right") - 1
        for _c in _crit:
            reference_scores[_c][_r] = _nat_acc[_c]
            probe_curve[_c][_r] = np.nanmean(_probe_acc[_c], axis=1)
            for _g in range(len(H_GRID)):
                _v = _nat_acc[_c][_group == _g]
                _v = _v[~np.isnan(_v)]
                natural_curve[_c][_r, _g] = _v.mean() if len(_v) else np.nan
                if _c == "a_zeit":
                    natural_n[_r, _g] = len(_v)
    return (
        AUC_LEVEL,
        MIN_ROWS_GROUP,
        natural_curve,
        natural_n,
        probe_curve,
        probe_n,
        probe_words,
        reference_scores,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Auswertung

    **Teilstichproben:** je Trefferzahl h der Mittelwert über die Prüfzeilen, gemittelt über die Teilungen, dazu der niedrigste Wert einer einzelnen Teilung und die Zahl der Teilungen, die das Niveau bzw. die Zufallsgrenze überschreiten. Die letzte Zeile zeigt die Prüfzeilen mit allen Belegen ihrer Hälfte (Obergrenze der Kurve).

    **Gegenprobe:** natürliche Zeilen, gruppiert nach der Zahl ihrer Belege in der kleineren Hälfte; Zeilen je Gruppe als Mittel über die Teilungen.
    """)
    return


@app.cell
def _(
    AUC_LEVEL,
    H_GRID,
    H_MAX,
    MIN_ROWS_GROUP,
    N_SPLITS,
    SVD_K,
    SVD_P,
    first_stable,
    fmt_int,
    fmt_num,
    fmt_pct,
    mo,
    natural_curve,
    natural_n,
    np,
    pl,
    probe_curve,
    probe_n,
    probe_words,
    row_hits,
    show_table,
):
    _n = len(H_GRID)
    # _w: criterion of the decision rule (a_zeit), _z: reported for information (a_wort)
    _w, _z = probe_curve["a_zeit"], probe_curve["a_wort"]
    _lvl = fmt_num(AUC_LEVEL, 2)

    def _count(x, level, strict=False):
        return f"{int(((x > level) if strict else (x >= level)).sum())} von {N_SPLITS}"

    _probe_table = pl.DataFrame({
        "Belege je Hälfte": [str(_h) for _h in H_GRID] + [f"alle (≥ {H_MAX})"],
        "a_zeit": [fmt_num(_w[:, _g].mean()) for _g in range(_n + 1)],
        "a_zeit, niedrigste Teilung": [fmt_num(_w[:, _g].min()) for _g in range(_n + 1)],
        f"Teilungen ≥ {_lvl}": [_count(_w[:, _g], AUC_LEVEL) for _g in range(_n + 1)],
        "Teilungen > 0,5": [_count(_w[:, _g], 0.5, strict=True) for _g in range(_n + 1)],
        "a_wort": [fmt_num(_z[:, _g].mean()) for _g in range(_n + 1)],
    })

    _nw = natural_curve["a_zeit"]
    _group_rows = natural_n.mean(axis=0)
    _usable = (_group_rows >= MIN_ROWS_GROUP) & ~np.isnan(_nw).any(axis=0)
    _group_labels = [
        (f"≥ {_h}" if _g == _n - 1 else (str(_h) if H_GRID[_g + 1] - 1 == _h else f"{_h}–{H_GRID[_g + 1] - 1}"))
        for _g, _h in enumerate(H_GRID)
    ]
    _natural_table = pl.DataFrame({
        "Belege in der kleineren Hälfte": _group_labels,
        "Zeilen": [fmt_num(_x, 1) for _x in _group_rows],
        "a_zeit": [fmt_num(np.nanmean(_nw[:, _g])) if _group_rows[_g] > 0 else "–" for _g in range(_n)],
        "a_zeit, niedrigste Teilung": [fmt_num(np.nanmin(_nw[:, _g])) if _group_rows[_g] > 0 else "–" for _g in range(_n)],
        f"Teilungen ≥ {_lvl}": [_count(_nw[:, _g], AUC_LEVEL) for _g in range(_n)],
        "in der Regel": ["ja" if _u else "nein" for _u in _usable],
        "a_wort": [
            fmt_num(np.nanmean(natural_curve["a_wort"][:, _g])) if _group_rows[_g] > 0 else "–" for _g in range(_n)
        ],
    })

    # decision rule
    _ceiling_ok = bool((_w[:, _n] >= AUC_LEVEL).all())
    _g_star = first_stable(_w[:, :_n], AUC_LEVEL)
    _g_zero = first_stable(_w[:, :_n], 0.5, strict=True)
    _g_nat = first_stable(np.nan_to_num(_nw, nan=-1.0), AUC_LEVEL, usable=_usable) if _usable.any() else None
    h_natural = H_GRID[_g_nat] if _g_nat is not None else None

    _text = [
        f"Prüfzeilen: {fmt_int(probe_n.min())} bis {fmt_int(probe_n.max())} je Teilung, aus "
        f"{probe_words.min()} bis {probe_words.max()} Zielwörtern (Zeilen mit mindestens {H_MAX} Belegen "
        f"in jeder Hälfte). SVD mit k = {SVD_K}, p = {SVD_P:g}.",
        f"Mit allen Belegen ihrer Hälfte erreichen die Prüfzeilen a_zeit = {fmt_num(_w[:, _n].mean())} "
        f"(niedrigste Teilung {fmt_num(_w[:, _n].min())}), a_wort = {fmt_num(_z[:, _n].mean())}.",
    ]
    h_star = None
    if not _ceiling_ok:
        _text.append(
            f"Das Niveau {_lvl} wird von den Prüfzeilen auch mit allen Belegen ihrer Hälfte nicht in allen "
            "Teilungen erreicht. Nach der Entscheidungsregel lässt sich keine Schwelle bestimmen; das Niveau "
            "ist für diese Daten und diese SVD zu hoch angesetzt."
        )
        _kind = "warn"
    elif _g_star is None:
        _text.append(
            f"Innerhalb von `H_GRID` (bis {H_MAX} Belege) erreicht a_zeit das Niveau {_lvl} nicht stabil in "
            "allen Teilungen, obwohl die Prüfzeilen es mit allen Belegen ihrer Hälfte erreichen. Die Schwelle "
            f"liegt oberhalb von {H_MAX}; `H_GRID` nach oben erweitern."
        )
        _kind = "warn"
    elif _g_star == 0:
        h_star = H_GRID[0]
        _text.append(
            f"a_zeit liegt schon bei {H_GRID[0]} Beleg(en) je Zeile in allen Teilungen bei mindestens {_lvl}. "
            "Für dieses Niveau ist keine Mindesttrefferzahl nötig."
        )
        _kind = "success"
    else:
        h_star = H_GRID[_g_star]
        _below = row_hits < h_star
        _text.append(
            f"Ab h\\* = {h_star} Belegen liegt a_zeit in allen Teilungen bei mindestens {_lvl} "
            f"(Mittel {fmt_num(_w[:, _g_star].mean())}, niedrigste Teilung {fmt_num(_w[:, _g_star].min())}). "
            f"Bei {H_GRID[_g_star - 1]} Belegen wird es in {_count(_w[:, _g_star - 1], AUC_LEVEL)} Teilungen "
            f"erreicht. Die Schwelle liegt also zwischen {H_GRID[_g_star - 1] + 1} und {h_star}; genauer "
            "bestimmen lässt sie sich mit zusätzlichen Werten in `H_GRID`."
        )
        _text.append(
            f"Unter {h_star} Belegen liegen in den vollständigen Daten {fmt_int(_below.sum())} von "
            f"{fmt_int(len(row_hits))} Zeilen ({fmt_pct(_below.mean())}) mit "
            f"{fmt_pct(row_hits[_below].sum() / row_hits.sum())} der Belege."
        )
        _kind = "success"

    if _g_zero is None:
        _text.append("Die Zufallsgrenze (a_zeit > 0,5 in allen Teilungen) wird innerhalb von `H_GRID` nicht stabil überschritten.")
    else:
        _text.append(
            f"Die Zufallsgrenze (a_zeit > 0,5 in allen Teilungen) wird ab {H_GRID[_g_zero]} Beleg(en) überschritten."
        )

    if _g_nat is None:
        _text.append(
            f"Gegenprobe: Die natürlichen Zeilen erreichen das Niveau {_lvl} in keiner verwendbaren Gruppe stabil."
        )
    else:
        _h_nat = H_GRID[_g_nat]
        _cmp = (
            "derselbe Wert wie bei den Teilstichproben" if h_star == _h_nat
            else (
                "ein höherer Wert als bei den Teilstichproben. Die tatsächlich schwach belegten Zeilen sind "
                "schlechter wiederzuerkennen, als ihre Trefferzahl allein erwarten lässt; h\\* ist als "
                f"Untergrenze zu lesen, {_h_nat} ist die vorsichtigere Wahl"
                if h_star is not None and _h_nat > h_star
                else "ein niedrigerer Wert als bei den Teilstichproben; h\\* ist vorsichtig gewählt"
                if h_star is not None
                else "für die Teilstichproben ist keine Schwelle bestimmt"
            )
        )
        _text.append(
            f"Gegenprobe: Die natürlichen Zeilen erreichen das Niveau ab der Gruppe mit {_group_labels[_g_nat]} "
            f"Belegen in der kleineren Hälfte, {_cmp}."
        )

    if h_star is not None and h_star > H_GRID[0]:
        _text.append(
            f"**Empfehlung:** Mindesttrefferzahl {h_star} Belege je Zeile (`MIN_HITS_FIT = {h_star}` in emb, "
            f"Niveau {_lvl}). Ob der Fit der SVD ohne die Zeilen darunter die übrigen Zeilen verbessert, ist "
            f"damit nicht gezeigt; das prüft Abschnitt 2. Die Schwelle gilt für k = {SVD_K}, "
            f"p = {SVD_P:g} und dieses Raster und ist nach einer Änderung neu zu bestimmen."
        )
    elif h_star is not None:
        _text.append("**Empfehlung:** keine Mindesttrefferzahl.")
    else:
        _text.append("**Empfehlung:** keine; s. o.")

    mo.vstack([
        mo.md("**Teilstichproben**"),
        show_table(_probe_table),
        mo.md("**Gegenprobe: natürliche Zeilen**"),
        show_table(_natural_table),
        mo.callout(mo.md("\n\n".join(_text)), kind=_kind),
    ])
    return h_natural, h_star


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Einschränkungen

    - Gemessen wird an halben Datenmengen: Randsummen und SVD beruhen auf einer Hälfte. In den vollständigen Daten sind beide stabiler. Die Schwelle ist in dieser Hinsicht eher vorsichtig.
    - Die Prüfzeilen sind die gut belegten Zeilen. Häufige Wörter und belegreiche Zeitscheiben können sich in ihren Kontexten systematisch von seltenen unterscheiden. Diesen Unterschied zeigt nur die Gegenprobe, und dort vermischt mit der Trefferzahl.
    - Die Belege einer Teilstichprobe sind in der Hälfte, an der die SVD berechnet wird, Teil der vollständigen Zeile und damit der Daten, aus denen der Raum entsteht. Der Einfluss einer einzelnen Zeile auf die SVD ist bei vielen Zeilen gering, aber nicht null. In jeder Richtung betrifft er eine der beiden Teilstichproben.
    - Die PPMI-Werte einer Teilstichprobe werden mit den Randsummen der Hälfte berechnet, in der die vollständige Zeile durch ihre Teilstichprobe ersetzt ist. Die übrigen Zeilen behalten die Randsummen der unveränderten Hälfte; die Abweichung betrifft je Prüfzeile nur deren eigene Zeitscheibe und ist gering.
    - Die Vergleichszeilen bleiben unverändert, auch schwach belegte. Die Höhe von a_zeit hängt davon ab, wie stark sich die Prüfwörter zwischen den Zeitscheiben tatsächlich unterscheiden, und von der Zahl ihrer Zeitscheiben; der Verlauf über h hängt davon weniger ab. Die Schwelle ist deshalb zwischen Korpora, Zielwortlisten und Rastern nur eingeschränkt übertragbar.
    - Das Niveau 0,7 ist eine Konvention aus der Bewertung diagnostischer Tests und wird hier übertragen. Ein anderes Niveau verschiebt die Schwelle.
    - Die Schwelle ist nur so genau wie `H_GRID`.
    - Gemessen wird Reliabilität, nicht Validität: Eine reproduzierbare Zeile bildet die Verwendung des Worts nicht notwendig zutreffend ab.
    - Die Schwelle hängt von k, p, Raster und KWIC-Variante ab und ist nach jeder Änderung neu zu bestimmen.

    ### Literatur

    - Antoniak, M. & Mimno, D. (2018): Evaluating the stability of embedding-based word similarities. *Transactions of the Association for Computational Linguistics* 6, 107–119.
    - Berry, M. W., Dumais, S. T. & O'Brien, G. W. (1995): Using linear algebra for intelligent information retrieval. *SIAM Review* 37(4), 573–595.
    - Brown, W. (1910): Some experimental results in the correlation of mental abilities. *British Journal of Psychology* 3(3), 296–322.
    - Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. & Harshman, R. (1990): Indexing by latent semantic analysis. *Journal of the American Society for Information Science* 41(6), 391–407.
    - Hanley, J. A. & McNeil, B. J. (1982): The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 143(1), 29–36.
    - Haxby, J. V., Gobbini, M. I., Furey, M. L., Ishai, A., Schouten, J. L. & Pietrini, P. (2001): Distributed and overlapping representations of faces and objects in ventral temporal cortex. *Science* 293(5539), 2425–2430.
    - Hosmer, D. W. & Lemeshow, S. (2000): *Applied Logistic Regression*. 2. Aufl. Wiley.
    - Levy, O., Goldberg, Y. & Dagan, I. (2015): Improving distributional similarity with lessons learned from word embeddings. *Transactions of the Association for Computational Linguistics* 3, 211–225.
    - Mann, H. B. & Whitney, D. R. (1947): On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics* 18(1), 50–60.
    - Spearman, C. (1910): Correlation calculated from faulty data. *British Journal of Psychology* 3(3), 271–295.

    Die bibliografischen Angaben sind vor einer Veröffentlichung an den Originalen zu prüfen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Einfluss schwach belegter Zeilen auf den Raum

    ### Verfahren

    In emb wird die SVD an allen Zeilen berechnet. Die Variante dieses Abschnitts berechnet sie nur an den Zeilen mit mindestens N Belegen. Die übrigen Zeilen werden anschließend per *folding-in* in diesen Raum projiziert (Deerwester et al. 1990; Berry, Dumais & O'Brien 1995). Sie bleiben damit erhalten und erhalten einen Vektor, bestimmen aber die Achsen nicht mit. Für die gefitteten Zeilen ist das Ergebnis dasselbe wie bei einem Ausschluss der übrigen Zeilen vor der SVD. Die PPMI-Werte werden in beiden Fällen an der vollständigen Matrix berechnet; die Randsummen bleiben also unverändert.

    Ein Anlass zu der Vermutung, dass schwach belegte Zeilen den Raum verschlechtern: Die SVD wählt ihre Achsen nach der quadrierten Länge der Zeilen (Eckart & Young 1936). Schwach belegte Zeilen haben bei kleiner Randsumme hohe PPMI-Werte auf wenigen Kontexten (Levy, Goldberg & Dagan 2015) und können so Achsen belegen, die sonst Unterschiede zwischen gut belegten Zeilen abbilden würden.

    ### Fragestellung

    Werden die gut belegten Zeilen reliabler und besser unterscheidbar, wenn die schwach belegten Zeilen die SVD nicht mitbestimmen?

    ### Methode

    **Vorprüfung ohne Teilungen.** An der vollständigen Matrix wird für jede Schwelle N bestimmt, welchen Anteil die Zeilen unter N an der quadrierten Gesamtnorm der PPMI-Matrix und an den Hebelwerten der ersten k Achsen haben. Der Hebelwert (*statistical leverage score*) einer Zeile ist die quadrierte Länge ihrer Zeile in $U_k$. Er misst, wie stark die Zeile die k Achsen der SVD bestimmt; die Summe über alle Zeilen ist k (Mahoney & Drineas 2009). Ist der Anteil der Zeilen unter N an den Hebelwerten deutlich größer als ihr Anteil an den Zeilen, bestimmen sie die Achsen überproportional mit, und eine Wirkung des Eingriffs ist zu erwarten. Ist er klein, ist auch keine Wirkung zu erwarten.

    **Split-half-Vergleich.** Auf denselben Teilungen wie in Abschnitt 1 werden zwei Varianten verglichen:

    | Variante | SVD berechnet an | übrige Zeilen |
    |---|---|---|
    | **R** (Referenz, heutiger Stand von emb) | allen Zeilen der Hälfte | – |
    | **F(N)** | Zeilen mit mindestens N Belegen in der Hälfte | per *folding-in* projiziert |

    Wie in Abschnitt 1 gilt die Trefferzahl der jeweiligen Hälfte, weil eine Hälfte mit h Belegen einer Zeile der vollständigen Daten mit h Belegen entspricht. Die SVD wird in beiden Richtungen berechnet, die Kennwerte werden gemittelt.

    **Bewertungszeilen** sind die Zeilen mit mindestens N Belegen in beiden Hälften; sie werden in beiden Richtungen gefittet. Verglichen werden a_zeit und a_wort dieser Zeilen in F(N) und R. Die Vergleichszeilen sind in beiden Varianten dieselben, nur der Raum unterscheidet sich. Die **Gruppen nach Textmenge** (Terzile) entstehen, indem die Zeilen mit mindestens N Belegen nach ihrer Trefferzahl in den vollständigen Daten in drei gleich große Gruppen geteilt werden.

    Zur Information wird berichtet, wie sich die Kennwerte der Zeilen ändern, die in keiner Richtung gefittet werden (weniger als N Belege in beiden Hälften).

    Geprüft werden die Schwellen aus `N_CANDIDATES`, darunter die Werte h\* und (falls abweichend) der Wert der Gegenprobe aus Abschnitt 1.

    ### Erwartete Ergebnisse und ihre Deutung

    - **Verbesserung**: Schwach belegte Zeilen verschlechtern den Raum für die übrigen. Die SVD ohne sie zu berechnen ist begründet; das lässt sich nur in der Pipeline umsetzen, nicht durch einen Filter im Frontend.
    - **Kein belastbarer Unterschied**: Die schwach belegten Zeilen stören die übrigen nicht messbar. Es genügt, sie zu markieren und die Anzeige dem Frontend zu überlassen.
    - **Verschlechterung**: Die schwach belegten Zeilen tragen zur Struktur des Raums bei, etwa weil sie gemeinsam Kontexte belegen, die einzeln selten sind. Der Eingriff ist dann nicht sinnvoll.
    - Ist die Vorprüfung unauffällig, ist kein Unterschied zu erwarten. Zeigt der Split-half-Vergleich dennoch einen, beruht er nicht auf der Hebelwirkung allein.

    ### Entscheidungsregel

    Eine Änderung (F(N) minus R, gemittelt über die Bewertungszeilen) gilt als **belastbar gestiegen** bzw. **belastbar gesunken**, wenn sie in allen Teilungen positiv bzw. negativ ist. Jede Schwelle N wird über beide Kennwerte und alle Gruppen hinweg eingestuft:

    | Einstufung | Bedingung |
    |---|---|
    | Verbesserung | Mindestens ein Kennwert steigt über alle Bewertungszeilen belastbar, und kein Kennwert sinkt in irgendeiner Gruppe belastbar (Pareto-Verbesserung). |
    | Verschlechterung | Mindestens ein Kennwert sinkt belastbar, keiner steigt belastbar. |
    | Zielkonflikt | Belastbare Anstiege und belastbare Rückgänge treten gleichzeitig auf. |
    | kein belastbarer Unterschied | alle übrigen Fälle |

    Maßgeblich für die Empfehlung ist die Einstufung bei N = h\* aus Abschnitt 1. Bei einer Verbesserung wird empfohlen, die SVD ohne die Zeilen unter h\* zu berechnen (`MIN_HITS_FIT = h*` in emb). In allen anderen Fällen wird empfohlen, die SVD unverändert zu lassen und die Zeilen nur zu markieren. Die übrigen Schwellen dienen zur Information.
    """)
    return


@app.cell
def _(
    SEED,
    SVD_K,
    counts_from_hits,
    fit_space,
    fmt_int,
    fmt_num,
    fmt_pct,
    h_natural,
    h_star,
    hit_counts,
    hit_row,
    n_rows,
    np,
    pl,
    ppmi,
    project,
    row_hits,
    row_slice,
    show_table,
    slice_marginals,
):
    # thresholds to test; h* and the cross-check value from section 1 are always included
    N_CANDIDATES = sorted({3, 5, 7, 10} | {_h for _h in (h_star, h_natural) if _h is not None and _h > 1})

    # full data, as in emb; the space is reused in section 4
    _all = np.arange(len(hit_row))
    full_counts = counts_from_hits(hit_counts, _all, hit_row, n_rows)
    full_ppmi = ppmi(full_counts, row_slice, *slice_marginals(full_counts))
    full_V, full_s = fit_space(full_ppmi, SVD_K, SEED)
    _U = project(full_ppmi, full_V, full_s, 0.0)        # U_k = M V_k S_k^-1
    _leverage = (_U ** 2).sum(axis=1)                    # sums to k
    _sq = np.asarray(full_ppmi.multiply(full_ppmi).sum(axis=1)).ravel()

    precheck = {
        _n: {
            "rows": float((row_hits < _n).mean()),
            "norm": float(_sq[row_hits < _n].sum() / _sq.sum()),
            "leverage": float(_leverage[row_hits < _n].sum() / _leverage.sum()),
        }
        for _n in N_CANDIDATES
    }
    show_table(pl.DataFrame({
        "Schwelle N": [str(_n) for _n in N_CANDIDATES],
        "Zeilen unter N": [fmt_int((row_hits < _n).sum()) for _n in N_CANDIDATES],
        "Anteil Zeilen": [fmt_pct(precheck[_n]["rows"]) for _n in N_CANDIDATES],
        "Anteil quadrierte Norm": [fmt_pct(precheck[_n]["norm"]) for _n in N_CANDIDATES],
        "Anteil Hebelwerte": [fmt_pct(precheck[_n]["leverage"]) for _n in N_CANDIDATES],
        "Hebelwerte / Zeilen": [
            fmt_num(precheck[_n]["leverage"] / precheck[_n]["rows"], 2) if precheck[_n]["rows"] > 0 else "–"
            for _n in N_CANDIDATES
        ],
    }))
    return N_CANDIDATES, full_V, full_counts, full_ppmi, full_s, precheck


@app.cell
def _(
    N_CANDIDATES,
    N_SPLITS,
    SEED,
    SVD_K,
    SVD_P,
    fit_space,
    half_matrices,
    mo,
    n_rows,
    np,
    project,
    separation,
    unit_rows,
):
    _crit = ("a_wort", "a_zeit")
    # per-row values of F(N) (direction mean), splits x rows; NaN where F(N) cannot be computed
    fitted_scores = {_n: {_c: np.full((N_SPLITS, n_rows), np.nan) for _c in _crit} for _n in N_CANDIDATES}
    fitted_rows = {_n: np.zeros((N_SPLITS, 2), dtype=int) for _n in N_CANDIDATES}
    half_min = np.zeros((N_SPLITS, n_rows), dtype=int)
    half_max = np.zeros((N_SPLITS, n_rows), dtype=int)

    for _r in mo.status.progress_bar(range(N_SPLITS), title="Teilungen"):
        _, _, _, _M, _n_half = half_matrices(_r)
        half_min[_r], half_max[_r] = np.minimum(*_n_half), np.maximum(*_n_half)
        for _n in N_CANDIDATES:
            _acc = {_c: np.zeros(n_rows) for _c in _crit}
            for _fit in (0, 1):
                _fit_rows = np.flatnonzero(_n_half[_fit] >= _n)
                fitted_rows[_n][_r, _fit] = len(_fit_rows)
                if len(_fit_rows) <= SVD_K:
                    _acc = None
                    break
                _V, _s = fit_space(_M[_fit][_fit_rows], SVD_K, SEED + _r)
                _X = [unit_rows(project(_M[_x], _V, _s, SVD_P)) for _x in (0, 1)]
                _res = separation(_X[0], _X[1], np.arange(n_rows), _X[0], _X[1])
                for _c in _crit:
                    _acc[_c] += _res[_c] / 2
            if _acc is not None:
                for _c in _crit:
                    fitted_scores[_n][_c][_r] = _acc[_c]
    return fitted_rows, fitted_scores, half_max, half_min


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Auswertung

    **Übersicht:** je Schwelle N die mittlere Änderung F(N) minus R über alle Bewertungszeilen und die Zahl der Teilungen mit positiver Änderung, die Einstufung nach der Entscheidungsregel und zur Information die Änderung bei den Zeilen, die in keiner Richtung gefittet werden. **Gruppen:** dieselben Änderungen getrennt nach Textmenge.
    """)
    return


@app.cell
def _(
    N_CANDIDATES,
    N_SPLITS,
    SVD_K,
    SVD_P,
    fitted_rows,
    fitted_scores,
    fmt_num,
    fmt_pct,
    h_natural,
    h_star,
    half_max,
    half_min,
    mo,
    np,
    pl,
    precheck,
    reference_scores,
    row_hits,
    show_table,
):
    _crit = ("a_zeit", "a_wort")

    def _mean_change(n, c, mask):
        """Mean change F(N) minus R per split over the rows in mask (splits x rows)."""
        d = fitted_scores[n][c] - reference_scores[c]
        valid = mask & ~np.isnan(d)
        with np.errstate(invalid="ignore"):
            return np.where(valid, d, 0).sum(axis=1) / valid.sum(axis=1)

    def _verdict(x):
        if np.isnan(x).any():
            return "nicht berechenbar"
        if (x > 0).all():
            return "gestiegen"
        if (x < 0).all():
            return "gesunken"
        return "uneinheitlich"

    def _positive(x):
        return "–" if np.isnan(x).any() else f"{int((x > 0).sum())} von {N_SPLITS}"

    _overview, _details, classes, _found = [], [], {}, {}
    for _n in N_CANDIDATES:
        _eval = half_min >= _n
        _unfitted = half_max < _n
        # terciles of the rows with at least N hits, by hits in the full data; ties broken by row order
        _above = np.flatnonzero(row_hits >= _n)
        _rank = np.empty(len(_above), dtype=int)
        _rank[np.argsort(row_hits[_above], kind="stable")] = np.arange(len(_above))
        _groups = {}
        for _g, _name in enumerate(["geringe Textmenge", "mittlere Textmenge", "hohe Textmenge"]):
            _members = _above[_rank * 3 // len(_above) == _g]
            _m = np.zeros(len(row_hits), dtype=bool)
            _m[_members] = True
            _groups[f"{_name} ({row_hits[_members].min()}–{row_hits[_members].max()} Belege)"] = _m
        _masks = {"alle Bewertungszeilen": np.ones(len(row_hits), dtype=bool), **_groups}

        _change = {_g: {_c: _mean_change(_n, _c, _eval & _m[None, :]) for _c in _crit} for _g, _m in _masks.items()}
        _verdicts = {_g: {_c: _verdict(_change[_g][_c]) for _c in _crit} for _g in _masks}
        _all_v = [_v for _d in _verdicts.values() for _v in _d.values()]
        if "nicht berechenbar" in _all_v:
            classes[_n] = "nicht berechenbar"
        else:
            _up_overall = any(_verdicts["alle Bewertungszeilen"][_c] == "gestiegen" for _c in _crit)
            _up, _down = "gestiegen" in _all_v, "gesunken" in _all_v
            classes[_n] = (
                "Verbesserung" if _up_overall and not _down
                else "Verschlechterung" if _down and not _up
                else "Zielkonflikt" if _up and _down
                else "kein belastbarer Unterschied"
            )
        _found[_n] = {
            _k: [
                f"{_c} ({_g}: {fmt_num(float(np.mean(_change[_g][_c])), 4)})"
                for _g in _masks for _c in _crit if _verdicts[_g][_c] == _k
            ]
            for _k in ("gestiegen", "gesunken")
        }

        _un = {_c: _mean_change(_n, _c, _unfitted) for _c in _crit}
        _overview.append({
            "N": str(_n),
            "gefittete Zeilen je Hälfte": fmt_num(fitted_rows[_n].mean(), 0),
            "Bewertungszeilen": fmt_num(_eval.sum(axis=1).mean(), 0),
            "Δ a_zeit": fmt_num(float(np.mean(_change["alle Bewertungszeilen"]["a_zeit"])), 4),
            "Δ a_zeit > 0": _positive(_change["alle Bewertungszeilen"]["a_zeit"]),
            "Δ a_wort": fmt_num(float(np.mean(_change["alle Bewertungszeilen"]["a_wort"])), 4),
            "Δ a_wort > 0": _positive(_change["alle Bewertungszeilen"]["a_wort"]),
            "Einstufung": classes[_n],
            "nicht gefittete Zeilen": fmt_num(_unfitted.sum(axis=1).mean(), 0),
            "Δ a_zeit, nicht gefittet": fmt_num(float(np.nanmean(_un["a_zeit"])), 4),
            "Δ a_wort, nicht gefittet": fmt_num(float(np.nanmean(_un["a_wort"])), 4),
        })
        for _g in _masks:
            _details.append({
                "N": str(_n),
                "Gruppe": _g,
                "Δ a_zeit": fmt_num(float(np.mean(_change[_g]["a_zeit"])), 4),
                "Δ a_zeit > 0": _positive(_change[_g]["a_zeit"]),
                "Δ a_wort": fmt_num(float(np.mean(_change[_g]["a_wort"])), 4),
                "Δ a_wort > 0": _positive(_change[_g]["a_wort"]),
            })

    # verdict
    _text = [f"SVD mit k = {SVD_K}, p = {SVD_P:g}."]
    _target = h_star if h_star is not None and h_star > 1 else None
    for _n in N_CANDIDATES:
        _p = precheck[_n]
        _text.append(
            f"N = {_n}: {classes[_n]}. Belastbar gestiegen: {', '.join(_found[_n]['gestiegen']) or 'nichts'}. "
            f"Belastbar gesunken: {', '.join(_found[_n]['gesunken']) or 'nichts'}. "
            f"Vorprüfung: {fmt_pct(_p['rows'])} der Zeilen, {fmt_pct(_p['norm'])} der quadrierten Norm, "
            f"{fmt_pct(_p['leverage'])} der Hebelwerte."
        )
    if _target is None:
        _text.append(
            "**Empfehlung:** Abschnitt 1 ergibt keine Mindesttrefferzahl; dieser Abschnitt dient nur zur Information. "
            "Die SVD bleibt unverändert."
        )
        _kind = "neutral"
    elif classes[_target] == "Verbesserung":
        _text.append(
            f"**Empfehlung:** SVD nur an den Zeilen mit mindestens {_target} Belegen berechnen, die übrigen per "
            f"*folding-in* projizieren und markieren (`MIN_HITS_FIT = {_target}` in emb)."
        )
        if h_natural is not None and h_natural > _target and classes.get(h_natural) == "Verbesserung":
            _text.append(
                f"Auch die vorsichtigere Schwelle {h_natural} aus der Gegenprobe ist eine Verbesserung."
            )
        _kind = "success"
    else:
        _text.append(
            f"**Empfehlung:** Bei N = h\\* = {_target} ergibt sich „{classes[_target]}“. Die SVD bleibt unverändert "
            f"an allen Zeilen; die Zeilen unter {_target} Belegen werden nur markiert (die Trefferzahl `n_hits` "
            "liegt in der Ausgabe von emb bereits vor), die Anzeige regelt das Frontend."
        )
        _kind = "neutral" if classes[_target] == "kein belastbarer Unterschied" else "warn"

    mo.vstack([
        mo.md("**Übersicht**"),
        show_table(pl.DataFrame(_overview)),
        mo.md("**Gruppen nach Textmenge**"),
        show_table(pl.DataFrame(_details)),
        mo.callout(mo.md("\n\n".join(_text)), kind=_kind),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Einschränkungen

    - Gemessen wird an halben Datenmengen. In den vollständigen Daten liegen mehr Zeilen über einer Schwelle N, und der Raum ist stabiler; die Wirkung des Eingriffs kann dort kleiner ausfallen.
    - Bewertet werden nur die gut belegten Zeilen. Eine Verbesserung für sie sagt nichts darüber, ob die schwach belegten Zeilen selbst besser oder schlechter dargestellt werden; das zeigt nur die Angabe zu den nicht gefitteten Zeilen, und auch dort nur ihre Reliabilität.
    - Die Schwelle wird in den Hälften auf die Trefferzahl der jeweiligen Hälfte angewandt. Welche Zeilen gefittet werden, hängt deshalb von der Teilung ab, und eine Zeile kann in einer Richtung gefittet werden und in der anderen nicht. Bewertungszeilen sind nur die Zeilen, die in beiden Richtungen gefittet werden.
    - Die SVD braucht mehr gefittete Zeilen als k. Ist das für eine Schwelle nicht erfüllt, ist sie „nicht berechenbar“.
    - Die Vorprüfung beruht auf einer einzigen SVD der vollständigen Daten, ohne Teilungen; sie zeigt eine mögliche Wirkung an, beweist aber keine.
    - Gemessen wird Reliabilität, nicht Validität.
    - Das Ergebnis gilt für k, p, Raster und KWIC-Variante dieses Laufs.

    ### Literatur

    - Berry, M. W., Dumais, S. T. & O'Brien, G. W. (1995): Using linear algebra for intelligent information retrieval. *SIAM Review* 37(4), 573–595.
    - Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. & Harshman, R. (1990): Indexing by latent semantic analysis. *Journal of the American Society for Information Science* 41(6), 391–407.
    - Eckart, C. & Young, G. (1936): The approximation of one matrix by another of lower rank. *Psychometrika* 1(3), 211–218.
    - Levy, O., Goldberg, Y. & Dagan, I. (2015): Improving distributional similarity with lessons learned from word embeddings. *Transactions of the Association for Computational Linguistics* 3, 211–225.
    - Mahoney, M. W. & Drineas, P. (2009): CUR matrix decompositions for improved data analysis. *Proceedings of the National Academy of Sciences* 106(3), 697–702.

    Weitere Angaben zum Messverfahren s. Abschnitt 1. Die bibliografischen Angaben sind vor einer Veröffentlichung an den Originalen zu prüfen.
    """)
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

    Der Exponent $p$ der Eigenwertgewichtung (*eigenvalue weighting*; Caron 2001; Levy, Goldberg & Dagan 2015) bestimmt, wie stark die Dimensionen mit großen Singulärwerten gegenüber den übrigen gewichtet werden: bei $p = 1$ mit dem Singulärwert, bei $p = 0{,}5$ mit seiner Quadratwurzel, bei $p = 0$ alle gleich. Werte unter 1 haben sich in mehreren Untersuchungen als günstiger erwiesen (Caron 2001; Bullinaria & Levy 2012; Österlund, Ödling & Sahlgren 2015). Eine Theorie, die bestimmte Werte für $k$ oder $p$ vorschreibt, gibt es nicht. Beide sind empirisch zu bestimmen, und zwar für jedes Korpus und jedes Zeitraster, weil der günstige Rang von der Zahl der Zeilen und ihrer Belegung abhängt.

    Referenz ist der Stand von emb: $k$ und $p$ des letzten Laufs (s. Daten). Die Abschnitte 1 und 2 verwenden diese Werte und sind nach einer Änderung von $k$ oder $p$ neu zu berechnen.

    ### Fragestellung

    Welche Kombination aus $k$ und $p$ ergibt die beste Unterscheidbarkeit der Zeitscheiben eines Worts (a_zeit), ohne die Unterscheidbarkeit der Zielwörter (a_wort) zu verschlechtern und ohne die Abhängigkeit der Ähnlichkeit von der Textmenge (Volumeneffekt) zu erhöhen?

    ### Methode

    **Split-half-Prüfung** auf denselben Teilungen wie in den Abschnitten 1 und 2: SVD an einer Hälfte, *folding-in* der anderen, beide Richtungen, Kennwerte gemittelt (s. Messverfahren). Geprüft werden alle Kombinationen aus `KP_KS` und `KP_PS`. Ränge, die in einer Hälfte größer sind als der Rang ihrer PPMI-Matrix, sind nicht berechenbar und entfallen.

    **Rechenweg.** Die SVD einer Hälfte $F$ wird über ihre Gram-Matrix berechnet: $F F^\top = U \Sigma^2 U^\top$ (Golub & Van Loan 2013, Kap. 8). Eine einzige Eigenzerlegung liefert alle Singulärwerte und damit alle $k$ auf einmal; eine abgeschnittene SVD je $k$ wäre bei großem $k$ zu langsam. Für die Zeilen von $F$ ist $x = U_k \Sigma_k^p$ wie in emb. Die Zeilen $m$ der anderen Hälfte $O$ erhalten durch *folding-in* $x = m V_k \Sigma_k^{p-1}$; mit $V_k = F^\top U_k \Sigma_k^{-1}$ ist das $(O F^\top)\, U_k\, \Sigma_k^{p-2}$. Eine Kontrollzelle prüft an einer Teilung, dass dieser Weg bei der Referenz dieselben Kosinusähnlichkeiten ergibt wie die SVD in den Abschnitten 1 und 2.

    **Kennwerte.** a_zeit und a_wort je Zeile wie in den übrigen Abschnitten, gemittelt über alle Zeilen und getrennt für drei Gruppen nach Textmenge (Terzile der Trefferzahl in den vollständigen Daten).

    **Volumeneffekt** (Methodology 1.18): An der Hälfte, an der die SVD berechnet wird, erhält jede Zeile die mittlere Kosinusähnlichkeit zu den übrigen Zeitscheiben desselben Zielworts. Diese wird über alle Zeilen mit dem Logarithmus der Zahl der Kontexttokens der Zeile in dieser Hälfte korreliert (Pearson, gepoolt). Ein hoher positiver Wert bedeutet, dass gut belegte Zeilen den übrigen Zeitscheiben ihres Worts allein wegen ihrer Textmenge ähnlicher sind; ein deutlich negativer Wert, dass sie ihnen allein deswegen unähnlicher sind. Beides ist eine Abhängigkeit von der Textmenge (zu Häufigkeitseffekten als Artefakt: Dubossarsky, Weinshall & Grossman 2017). Bewertet wird deshalb der Betrag $|r|$, je Teilung als Mittel der beiden Richtungen; ein kleinerer Betrag ist besser. Methodology 1.18 nennt nur „niedriger ist besser“, weil dort bisher nur positive Werte auftraten.

    **Vergleich ohne SVD** (zur Information): dieselben Kennwerte mit den PPMI-Zeilenvektoren selbst. Die Kosinusähnlichkeiten werden exakt aus den Skalarprodukten der Hälften berechnet. Der Vergleich zeigt, ob die Kürzung durch die SVD die Unterscheidbarkeit verbessert oder nur erhält.

    ### Erwartete Ergebnisse und ihre Deutung

    | Befund | Deutung |
    |---|---|
    | a_zeit steigt mit sinkendem k bis zu einem Höchstwert und fällt danach | Die Dimensionen mit kleinen Singulärwerten tragen überwiegend Zufallsschwankungen; eine zu starke Kürzung beseitigt Unterschiede zwischen Zeitscheiben. |
    | a_wort steigt, a_zeit fällt bei kleinem k | Die Kürzung erhält die Identität der Zielwörter, zieht aber ihre Zeitscheiben zusammen (Zielkonflikt). |
    | a_zeit steigt mit wachsendem k durchgehend | Auch Dimensionen mit kleinen Singulärwerten tragen reproduzierbare Information über Zeitscheiben. |
    | p < 1 erhöht a_zeit | Die Dimensionen mit großen Singulärwerten sind bei p = 1 zu stark gewichtet. |
    | Der Betrag des Volumeneffekts ändert sich stark mit k oder p | Die hinzukommenden bzw. stärker gewichteten Dimensionen bilden die Textmenge der Zeilen ab. |

    Liegt die empfohlene Kombination am Rand des untersuchten Bereichs, ist der Bereich zu erweitern.

    ### Entscheidungsregel

    Jede Kombination wird mit der Referenz verglichen. Eine Änderung (Kombination minus Referenz, je Teilung gemittelt über die Zeilen) gilt als **belastbar gestiegen** bzw. **belastbar gesunken**, wenn sie in allen Teilungen positiv bzw. negativ ist. Beim Betrag des Volumeneffekts ist ein Anstieg eine Verschlechterung.

    | Einstufung | Bedingung |
    |---|---|
    | Verbesserung | a_zeit oder a_wort steigt über alle Zeilen belastbar, keiner der beiden sinkt in irgendeiner Gruppe belastbar, und der Betrag des Volumeneffekts steigt nicht belastbar (Pareto-Verbesserung). |
    | Verschlechterung | Mindestens ein Kennwert verschlechtert sich belastbar, keiner verbessert sich belastbar. |
    | Zielkonflikt | Belastbare Verbesserungen und belastbare Verschlechterungen treten gleichzeitig auf. |
    | kein belastbarer Unterschied | alle übrigen Fälle |

    Empfohlen wird unter den Verbesserungen die Kombination mit dem größten mittleren Anstieg von a_zeit über alle Zeilen. a_zeit misst die Eigenschaft, auf die es bei der Darstellung von Bedeutungswandel ankommt. Kombinationen, deren a_zeit sich von der empfohlenen nicht belastbar unterscheidet, werden als gleichwertig genannt. Gibt es keine Verbesserung, bleibt die Referenz.

    Ein Zielkonflikt wird nicht automatisch aufgelöst. Zur Information nennt die Auswertung die Kombinationen, die in den Mittelwerten von a_zeit, a_wort und Betrag des Volumeneffekts von keiner anderen übertroffen werden (Pareto-Front). Unter ihnen ist eine begründete Abwägung möglich.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vorprüfung

    Für jedes $k$ wird angegeben, welchen Anteil der Summe der quadrierten PPMI-Werte die Approximation mit Rang $k$ erhält (Anteil an der quadrierten Frobenius-Norm, berechnet auf den vollständigen Daten). Der Wert beschreibt nur die Stärke der Kürzung, nicht ihre Güte. Er ist kein „erklärter Varianzanteil“, weil die PPMI-Matrix nicht zentriert ist.
    """)
    return


@app.cell
def _(
    SVD_K,
    SVD_P,
    counts_from_hits,
    fmt_int,
    fmt_pct,
    hit_counts,
    hit_row,
    mo,
    n_rows,
    np,
    pl,
    ppmi,
    row_slice,
    row_word,
    show_table,
    slice_marginals,
):
    # grid of section 3; the reference (k, p of emb) is always included. Ranks above the rank of a half are dropped.
    KP_KS = sorted({10, 25, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500} | {SVD_K})
    KP_PS = sorted({1.0, 0.5, 0.0} | {SVD_P}, reverse=True)
    KP_REFERENCE = (SVD_K, SVD_P)
    KP_RANK_TOL = 1e-6    # singular values below this fraction of the largest count as zero (rank)

    def kp_spectrum(F):
        """All singular values and left singular vectors of F via its Gram matrix F F^T = U S^2 U^T,
        in descending order, truncated at the numerical rank. Rows of F without any value get exact zero rows
        in U (the eigensolver leaves values near machine precision there), so unit_rows marks them as NaN."""
        ev, u = np.linalg.eigh((F @ F.T).toarray())
        order = np.argsort(ev)[::-1]
        s = np.sqrt(np.clip(ev[order], 0, None))
        rank = int((s > KP_RANK_TOL * s[0]).sum())
        u = u[:, order[:rank]]
        u[F.getnnz(axis=1) == 0] = 0
        return u, s[:rank]

    # comparison rows of a_wort lie in the same slice, those of a_zeit belong to the same word: both measures
    # are computed block by block within these groups (same result as separation, much faster for many variants)
    _word_groups = [np.flatnonzero(row_word == _w) for _w in np.unique(row_word)]
    _slice_groups = [np.flatnonzero(row_slice == _t) for _t in np.unique(row_slice)]

    def kp_separation(XA, XB):
        """a_wort and a_zeit for all rows, as separation(XA, XB, all rows, XA, XB)."""
        out = {}
        for name, groups, other in (("a_wort", _slice_groups, row_word), ("a_zeit", _word_groups, row_slice)):
            res = np.full(len(row_word), np.nan)
            for g in groups:
                own = np.einsum("ij,ij->i", XA[g], XB[g])
                comp = (XA[g] @ XB[g].T + XB[g] @ XA[g].T) / 2
                valid = (other[g][:, None] != other[g][None, :]) & ~np.isnan(comp)
                less = ((comp < own[:, None]) & valid).sum(axis=1)
                ties = ((comp == own[:, None]) & valid).sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = (less + 0.5 * ties) / valid.sum(axis=1)
                r[np.isnan(own)] = np.nan
                res[g] = r
            out[name] = res
        return out

    def kp_volume_effect(X, log_tokens):
        """Volume effect: Pearson correlation between the mean cosine of a row to the other slices of its word
        and the log number of its context tokens. X: unit rows of the fitting half (NaN: no vector)."""
        mean_sim = np.full(len(row_word), np.nan)
        for g in _word_groups:
            sim = X[g] @ X[g].T
            valid = (row_slice[g][:, None] != row_slice[g][None, :]) & ~np.isnan(sim)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_sim[g] = np.where(valid, sim, 0).sum(axis=1) / valid.sum(axis=1)
        use = ~np.isnan(mean_sim) & np.isfinite(log_tokens)
        return float(np.corrcoef(mean_sim[use], log_tokens[use])[0, 1])

    _C = counts_from_hits(hit_counts, np.arange(len(hit_row)), hit_row, n_rows)
    _, _s = kp_spectrum(ppmi(_C, row_slice, *slice_marginals(_C)))
    _share = np.cumsum(_s ** 2) / (_s ** 2).sum()
    kp_rank_full = len(_s)

    mo.vstack([
        show_table(pl.DataFrame({
            "k": [fmt_int(_k) for _k in KP_KS],
            "erhaltener Anteil": [fmt_pct(_share[min(_k, kp_rank_full) - 1]) for _k in KP_KS],
            "": ["Referenz (emb)" if _k == SVD_K else "" for _k in KP_KS],
        })),
        mo.md(f"Rang der PPMI-Matrix der vollständigen Daten: {fmt_int(kp_rank_full)} (Zeilen: {fmt_int(n_rows)})."),
    ])
    return (
        KP_KS,
        KP_PS,
        KP_REFERENCE,
        kp_separation,
        kp_spectrum,
        kp_volume_effect,
    )


@app.cell
def _(
    BLAS_THREADS_PRESET,
    KP_REFERENCE,
    SEED,
    fit_space,
    fmt_num,
    half_matrices,
    kp_separation,
    kp_spectrum,
    mo,
    np,
    project,
    separation,
    unit_rows,
):
    # control: the Gram route of section 3 must reproduce fit_space/project (sections 1 and 2) at the reference,
    # and the exact PPMI cosines without SVD; checked on split 1, SVD fitted on half A
    _, _, _, _M, _ = half_matrices(0)
    _k, _p = KP_REFERENCE
    _u, _s = kp_spectrum(_M[0])
    _cross = (_M[1] @ _M[0].T).toarray() @ _u
    _gram = [unit_rows(_u[:, :_k] * _s[:_k] ** _p), unit_rows(_cross[:, :_k] * _s[:_k] ** (_p - 2))]
    _V, _sv = fit_space(_M[0], _k, SEED)
    _svd = [unit_rows(project(_M[_x], _V, _sv, _p)) for _x in (0, 1)]
    _dev_svd = max(
        float(np.nanmax(np.abs(_gram[_a] @ _gram[_b].T - _svd[_a] @ _svd[_b].T))) for _a, _b in ((0, 0), (0, 1), (1, 1))
    )

    # exact cosines without SVD: full-rank coordinates of half A, half B projected onto the row space of A and
    # divided by its true row norms
    _na = np.sqrt(np.asarray(_M[0].multiply(_M[0]).sum(axis=1)).ravel())
    _nb = np.sqrt(np.asarray(_M[1].multiply(_M[1]).sum(axis=1)).ravel())
    with np.errstate(divide="ignore", invalid="ignore"):
        _xb = (_cross / _s) / _nb[:, None]
        _direct = (_M[0] @ _M[1].T).toarray() / np.outer(_na, _nb)
    _approx = unit_rows(_u * _s) @ _xb.T
    _ok = (_na > 0)[:, None] & (_nb > 0)[None, :]
    _dev_ppmi = float(np.abs(_approx - _direct)[_ok].max())

    # block-wise measures of section 3 against separation (sections 1 and 2), on the vectors of fit_space
    _rows = np.arange(len(_na))
    _ref_sep = separation(_svd[0], _svd[1], _rows, _svd[0], _svd[1])
    _own_sep = kp_separation(_svd[0], _svd[1])
    _same_nan = all(np.array_equal(np.isnan(_ref_sep[_c]), np.isnan(_own_sep[_c])) for _c in _ref_sep)
    _dev_sep = max(float(np.nanmax(np.abs(_ref_sep[_c] - _own_sep[_c]))) for _c in _ref_sep)

    KP_CHECK_TOL = 1e-6
    mo.stop(
        _dev_svd > KP_CHECK_TOL or _dev_ppmi > KP_CHECK_TOL or _dev_sep > KP_CHECK_TOL or not _same_nan,
        mo.callout(
            mo.md(
                f"**Abbruch:** Der Rechenweg von Abschnitt 3 weicht ab (Gram-Matrix gegenüber der SVD: "
                f"{_dev_svd:.1e}, gegenüber den PPMI-Kosinuswerten: {_dev_ppmi:.1e}; a_wort/a_zeit gegenüber "
                f"`separation`: {_dev_sep:.1e}, fehlende Werte gleich: {'ja' if _same_nan else 'nein'}; zulässig "
                f"{KP_CHECK_TOL:.0e})."
            ),
            kind="danger",
        ),
    )
    _msg = mo.md(
        f"Kontrolle (Teilung 1, SVD an Hälfte A): Der Rechenweg über die Gram-Matrix ergibt bei k = {_k}, "
        f"p = {fmt_num(_p, 1)} dieselben Kosinusähnlichkeiten wie die SVD der Abschnitte 1 und 2 (größte Abweichung "
        f"{_dev_svd:.1e}) und ohne Kürzung dieselben wie die PPMI-Vektoren (größte Abweichung {_dev_ppmi:.1e}). "
        f"Die blockweise Berechnung von a_wort und a_zeit stimmt mit `separation` überein (größte Abweichung "
        f"{_dev_sep:.1e})."
    )
    _blas = mo.callout(
        mo.md(
            "Hinweis: numpy war beim Start des Notebooks schon geladen; die Begrenzung der BLAS-Threads (Import-Zelle) "
            "greift dann nicht. Die Ergebnisse sind davon nicht betroffen, die Laufzeit dieses Abschnitts kann sich "
            "aber vervielfachen. Abhilfe: `OPENBLAS_NUM_THREADS=4` vor dem Start setzen."
        ),
        kind="warn",
    )
    mo.vstack([_msg] if BLAS_THREADS_PRESET else [_msg, _blas])
    return


@app.cell
def _(
    KP_KS,
    KP_PS,
    N_SPLITS,
    half_matrices,
    kp_separation,
    kp_spectrum,
    kp_volume_effect,
    mo,
    n_rows,
    np,
    unit_rows,
):
    _crit = ("a_wort", "a_zeit")
    # variants: (k, p) and "PPMI" (without SVD); values per split and row (mean of both directions)
    kp_variants = [(_k, _p) for _k in KP_KS for _p in KP_PS] + ["PPMI"]
    kp_scores = {_v: {_c: np.full((N_SPLITS, n_rows), np.nan) for _c in _crit} for _v in kp_variants}
    kp_volume = {_v: np.full(N_SPLITS, np.nan) for _v in kp_variants}
    kp_rank_min = n_rows

    for _r in mo.status.progress_bar(range(N_SPLITS), title="Teilungen"):
        _, _C, _, _M, _ = half_matrices(_r)
        _acc = {_v: {_c: np.zeros(n_rows) for _c in _crit} for _v in kp_variants}
        _vol = {_v: 0.0 for _v in kp_variants}
        _computable = {_v: True for _v in kp_variants}
        for _fit in (0, 1):
            _u, _s = kp_spectrum(_M[_fit])
            kp_rank_min = min(kp_rank_min, len(_s))
            _cross = (_M[1 - _fit] @ _M[_fit].T).toarray() @ _u    # (O F^T) U
            with np.errstate(divide="ignore"):
                _log_tokens = np.log(np.asarray(_C[_fit].sum(axis=1)).ravel())
            for _v in kp_variants:
                if _v == "PPMI":
                    # exact cosines without SVD (see control cell)
                    _norm_o = np.sqrt(np.asarray(_M[1 - _fit].multiply(_M[1 - _fit]).sum(axis=1)).ravel())
                    _xf = unit_rows(_u * _s)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        _xo = np.where(_norm_o[:, None] > 0, (_cross / _s) / _norm_o[:, None], np.nan)
                else:
                    _k, _p = _v
                    if _k > len(_s):
                        _computable[_v] = False
                        continue
                    _xf = unit_rows(_u[:, :_k] * _s[:_k] ** _p)
                    _xo = unit_rows(_cross[:, :_k] * _s[:_k] ** (_p - 2))
                _XA, _XB = (_xf, _xo) if _fit == 0 else (_xo, _xf)
                _res = kp_separation(_XA, _XB)
                for _c in _crit:
                    _acc[_v][_c] += _res[_c] / 2
                _vol[_v] += kp_volume_effect(_xf, _log_tokens) / 2
        for _v in kp_variants:
            if _computable[_v]:
                for _c in _crit:
                    kp_scores[_v][_c][_r] = _acc[_v][_c]
                kp_volume[_v][_r] = _vol[_v]
    return kp_rank_min, kp_scores, kp_variants, kp_volume


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Auswertung

    **Tabellen 3.1 bis 3.4:** a_zeit, a_wort, Volumeneffekt (mit Vorzeichen; bewertet wird der Betrag) und Einstufung je Kombination (Mittel über die Teilungen, alle Zeilen). **Tabelle 3.5:** Änderung der empfohlenen Kombination gegenüber der Referenz nach Textmenge, mit der Zahl der Teilungen, in denen die Änderung positiv ist.
    """)
    return


@app.cell
def _(
    KP_KS,
    KP_PS,
    KP_REFERENCE,
    N_SPLITS,
    fmt_int,
    fmt_num,
    kp_rank_min,
    kp_scores,
    kp_variants,
    kp_volume,
    mo,
    np,
    pl,
    row_hits,
    show_table,
):
    _crit = ("a_zeit", "a_wort")

    def _label(v):
        return "PPMI ohne SVD" if v == "PPMI" else f"k = {v[0]}, p = {fmt_num(v[1], 1)}"

    # groups by text volume: terciles of the hits per row in the full data, ties broken by row order
    _rank = np.empty(len(row_hits), dtype=int)
    _rank[np.argsort(row_hits, kind="stable")] = np.arange(len(row_hits))
    _groups = {"alle Zeilen": np.ones(len(row_hits), dtype=bool)}
    for _g, _name in enumerate(["geringe Textmenge", "mittlere Textmenge", "hohe Textmenge"]):
        _m = _rank * 3 // len(row_hits) == _g
        _groups[f"{_name} ({row_hits[_m].min()}–{row_hits[_m].max()} Belege)"] = _m

    def _mean(v, c, mask):
        """Mean over the rows in mask, per split."""
        x = kp_scores[v][c]
        valid = mask[None, :] & ~np.isnan(x)
        with np.errstate(invalid="ignore"):
            return np.where(valid, x, 0).sum(axis=1) / valid.sum(axis=1)

    _computed = [_v for _v in kp_variants if not np.isnan(kp_volume[_v]).any()]
    _ref = KP_REFERENCE
    mo.stop(
        _ref not in _computed,
        mo.callout(mo.md("**Abbruch:** Die Referenz ist in mindestens einer Hälfte nicht berechenbar."), kind="danger"),
    )

    # change per split against the reference; volume effect: an increase is a deterioration
    _change = {
        _v: {
            **{(_g, _c): _mean(_v, _c, _m) - _mean(_ref, _c, _m) for _g, _m in _groups.items() for _c in _crit},
            ("alle Zeilen", "|Volumeneffekt|"): np.abs(kp_volume[_v]) - np.abs(kp_volume[_ref]),
        }
        for _v in _computed
    }

    def _better(v):
        """Robust improvements and deteriorations of v against the reference (lists of descriptions)."""
        up, down = [], []
        for (g, c), d in _change[v].items():
            good, bad = ((d < 0).all(), (d > 0).all()) if c == "|Volumeneffekt|" else ((d > 0).all(), (d < 0).all())
            text = f"{c} ({g}: {fmt_num(float(d.mean()), 4)})"
            if good:
                up.append((g, c, text))
            if bad:
                down.append((g, c, text))
        return up, down

    _classes, _found = {}, {}
    for _v in _computed:
        if _v == _ref:
            continue
        _up, _down = _better(_v)
        _found[_v] = (_up, _down)
        _up_overall = any(_g == "alle Zeilen" and _c in _crit for _g, _c, _ in _up)
        _classes[_v] = (
            "Verbesserung" if _up_overall and not _down
            else "Verschlechterung" if _down and not _up
            else "Zielkonflikt" if _up and _down
            else "kein belastbarer Unterschied"
        )

    _all = _groups["alle Zeilen"]
    _avg = {_v: {_c: float(_mean(_v, _c, _all).mean()) for _c in _crit} for _v in _computed}
    for _v in _computed:
        _avg[_v]["Volumeneffekt"] = float(kp_volume[_v].mean())
        _avg[_v]["|Volumeneffekt|"] = float(np.abs(kp_volume[_v]).mean())

    # tables k x p (SVD variants only)
    def _pivot(value):
        _ks = [_k for _k in KP_KS if any((_k, _p) in _computed for _p in KP_PS)]
        return pl.DataFrame({
            "k": [fmt_int(_k) for _k in _ks],
            **{
                f"p = {fmt_num(_p, 1)}": [value((_k, _p)) if (_k, _p) in _computed else "–" for _k in _ks]
                for _p in KP_PS
            },
        })

    _tables = [
        ("Tabelle 3.1: a_zeit", _pivot(lambda v: fmt_num(_avg[v]["a_zeit"]))),
        ("Tabelle 3.2: a_wort", _pivot(lambda v: fmt_num(_avg[v]["a_wort"]))),
        ("Tabelle 3.3: Volumeneffekt (r gepoolt)", _pivot(lambda v: fmt_num(_avg[v]["Volumeneffekt"]))),
        ("Tabelle 3.4: Einstufung gegenüber der Referenz", _pivot(lambda v: "Referenz" if v == _ref else _classes[v])),
    ]

    # recommendation: largest mean rise of a_zeit among the improvements
    _svd = [_v for _v in _computed if _v != "PPMI"]
    _improvements = [_v for _v in _svd if _v != _ref and _classes[_v] == "Verbesserung"]
    kp_best = max(_improvements, key=lambda v: _change[v][("alle Zeilen", "a_zeit")].mean()) if _improvements else None
    _equivalent = []
    if kp_best is not None:
        _best_zeit = _mean(kp_best, "a_zeit", _all)
        for _v in _improvements:
            if _v != kp_best:
                _d = _mean(_v, "a_zeit", _all) - _best_zeit
                if not ((_d > 0).all() or (_d < 0).all()):
                    _equivalent.append(_v)

    # Pareto front of the means (a_zeit, a_wort higher, volume effect lower)
    def _dominates(a, b):
        ge = (_avg[a]["a_zeit"] >= _avg[b]["a_zeit"] and _avg[a]["a_wort"] >= _avg[b]["a_wort"]
              and _avg[a]["|Volumeneffekt|"] <= _avg[b]["|Volumeneffekt|"])
        return ge and _avg[a] != _avg[b]

    _front = [_v for _v in _svd if not any(_dominates(_w, _v) for _w in _svd)]
    _front.sort(key=lambda v: -_avg[v]["a_zeit"])

    _counts = {_c: sum(1 for _v in _classes if _v != "PPMI" and _classes[_v] == _c) for _c in
               ("Verbesserung", "Zielkonflikt", "Verschlechterung", "kein belastbarer Unterschied")}
    _dropped = [_k for _k in KP_KS if not any((_k, _p) in _computed for _p in KP_PS)]
    _text = [
        f"Referenz: {_label(_ref)}, a_zeit = {fmt_num(_avg[_ref]['a_zeit'])}, a_wort = {fmt_num(_avg[_ref]['a_wort'])}, "
        f"Volumeneffekt = {fmt_num(_avg[_ref]['Volumeneffekt'])}. Von {len(_svd) - 1} Kombinationen sind eingestuft als "
        f"Verbesserung: {_counts['Verbesserung']}, Zielkonflikt: {_counts['Zielkonflikt']}, Verschlechterung: "
        f"{_counts['Verschlechterung']}, kein belastbarer Unterschied: {_counts['kein belastbarer Unterschied']}."
        + (
            f" Nicht berechenbar, weil der Rang in mindestens einer Hälfte nur {fmt_int(kp_rank_min)} beträgt: "
            f"k = {', '.join(fmt_int(_k) for _k in _dropped)}."
            if _dropped else ""
        ),
    ]
    if "PPMI" in _computed:
        _top = max(_svd, key=lambda v: _avg[v]["a_zeit"])
        _text.append(
            f"PPMI ohne SVD: a_zeit = {fmt_num(_avg['PPMI']['a_zeit'])}, a_wort = {fmt_num(_avg['PPMI']['a_wort'])}, "
            f"Volumeneffekt = {fmt_num(_avg['PPMI']['Volumeneffekt'])}. Höchstes a_zeit mit SVD: "
            f"{fmt_num(_avg[_top]['a_zeit'])} ({_label(_top)}). "
            + (
                "Keine Kombination erreicht a_zeit der PPMI-Vektoren; die Kürzung verringert in diesen Daten die "
                "Unterscheidbarkeit der Zeitscheiben."
                if _avg[_top]["a_zeit"] < _avg["PPMI"]["a_zeit"]
                else "Mindestens eine Kombination erreicht oder übertrifft a_zeit der PPMI-Vektoren."
            )
        )
    _text.append(
        "Pareto-Front (in den Mittelwerten von keiner anderen Kombination übertroffen): "
        + "; ".join(
            f"{_label(_v)} (a_zeit {fmt_num(_avg[_v]['a_zeit'])}, a_wort {fmt_num(_avg[_v]['a_wort'])}, "
            f"Volumeneffekt {fmt_num(_avg[_v]['Volumeneffekt'])})"
            for _v in _front
        )
        + "."
    )

    _outputs = []
    for _title, _df in _tables:
        _outputs += [mo.md(f"**{_title}**"), show_table(_df)]

    if kp_best is None:
        _kind = "warn"
        _rec = (
            f"**Empfehlung: Referenz beibehalten ({_label(_ref)}).** Keine Kombination ist als Verbesserung eingestuft. "
            "Zielkonflikte sind nach der Pareto-Front abzuwägen."
        )
    else:
        _up, _down = _found[kp_best]
        _text.append(
            f"Empfohlene Kombination: {_label(kp_best)}. Belastbar verbessert: {', '.join(_t for *_, _t in _up)}."
        )
        if _equivalent:
            _text.append("Gleichwertig in a_zeit: " + "; ".join(_label(_v) for _v in _equivalent) + ".")
        _edges = []
        _ks_ok = [_k for _k in KP_KS if _k not in _dropped]
        if kp_best[0] == max(_ks_ok):
            _edges.append(
                "k liegt am oberen Rand des untersuchten Bereichs"
                + (" (größeres k ist in den Hälften nicht berechenbar)" if _dropped else "; `KP_KS` erweitern")
            )
        if kp_best[0] == min(_ks_ok):
            _edges.append("k liegt am unteren Rand des untersuchten Bereichs; `KP_KS` erweitern")
        if kp_best[1] == min(KP_PS):
            _edges.append("p liegt am unteren Rand des untersuchten Bereichs; `KP_PS` erweitern")
        if _edges:
            _text.append("**Hinweis:** " + "; ".join(_edges) + ".")
        _kind = "success"
        _rec = (
            f"**Empfehlung: {_label(kp_best)}** (`k = {kp_best[0]}`, `eig_p = {fmt_num(kp_best[1], 1).replace(',', '.')}` "
            f"in emb). a_zeit steigt gegenüber der Referenz über alle Zeilen um "
            f"{fmt_num(float(_change[kp_best][('alle Zeilen', 'a_zeit')].mean()), 4)}, kein Kennwert verschlechtert sich "
            "belastbar. Danach die Abschnitte 1 und 2 mit den neuen Werten wiederholen."
        )
        _outputs += [
            mo.md("**Tabelle 3.5: Änderung der empfohlenen Kombination gegenüber der Referenz nach Textmenge**"),
            show_table(pl.DataFrame([
                {
                    "Gruppe": _g,
                    "Δ a_zeit": fmt_num(float(_change[kp_best][(_g, "a_zeit")].mean()), 4),
                    "Δ a_zeit > 0": f"{int((_change[kp_best][(_g, 'a_zeit')] > 0).sum())} von {N_SPLITS}",
                    "Δ a_wort": fmt_num(float(_change[kp_best][(_g, "a_wort")].mean()), 4),
                    "Δ a_wort > 0": f"{int((_change[kp_best][(_g, 'a_wort')] > 0).sum())} von {N_SPLITS}",
                }
                for _g in _groups
            ])),
            mo.md(
                f"Betrag des Volumeneffekts: Δ = {fmt_num(float(_change[kp_best][('alle Zeilen', '|Volumeneffekt|')].mean()), 4)}, "
                f"gesunken in {int((_change[kp_best][('alle Zeilen', '|Volumeneffekt|')] < 0).sum())} von {N_SPLITS} "
                "Teilungen."
            ),
        ]

    mo.vstack(_outputs + [mo.md("\n\n".join(_text)), mo.callout(mo.md(_rec), kind=_kind)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Einschränkungen

    - Gemessen wird an halben Datenmengen. Der Rang einer Hälfte ist kleiner als der der vollständigen Daten, und der günstige Rang kann in den vollständigen Daten etwas höher liegen. Für die Entscheidung zählt der Vergleich der Kombinationen untereinander.
    - Die Split-half-Prüfung misst Reliabilität, nicht Validität. a_zeit erfasst jede reproduzierbare Eigenschaft einer Zeitscheibe, also auch solche, die nicht auf der Bedeutung des Zielworts beruhen, etwa die Themen eines Halbjahrs oder die Textmenge. a_wort und der Volumeneffekt begrenzen diesen Einfluss nur teilweise.
    - Der Volumeneffekt wird an der Hälfte berechnet, an der die SVD berechnet wird, und gepoolt über alle Zielwörter. Er unterscheidet nicht zwischen einer Abhängigkeit von der Textmenge und echtem Wandel, der mit einer Zunahme der Textmenge einhergeht (über ein Wort wird oft gerade dann viel geredet, wenn sich sein Kontext ändert; Methodology 1.18).
    - Die Bewertung „belastbar“ verlangt dasselbe Vorzeichen in allen Teilungen. Kleine, aber gleichgerichtete Unterschiede werden dadurch als belastbar eingestuft, auch wenn sie inhaltlich unerheblich sind; die Tabellen zeigen die Größe.
    - Die Empfehlung gilt für die Referenz dieses Laufs: Eine Pareto-Verbesserung gegenüber einer anderen Referenz kann anders ausfallen. Die Pareto-Front hängt nicht von der Referenz ab.
    - Das Ergebnis gilt für dieses Korpus, dieses Raster und diese KWIC-Variante und ist nach jeder Änderung der Daten neu zu bestimmen. Nach einer Änderung von k oder p sind die Abschnitte 1 und 2 zu wiederholen.

    ### Literatur

    - Bullinaria, J. A. & Levy, J. P. (2012): Extracting semantic representations from word co-occurrence statistics: stop-lists, stemming, and SVD. *Behavior Research Methods* 44(3), 890–907.
    - Caron, J. (2001): Experiments with LSA scoring: optimal rank and basis. In: Berry, M. W. (Hg.): *Computational Information Retrieval*. SIAM, 157–169.
    - Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. & Harshman, R. (1990): Indexing by latent semantic analysis. *Journal of the American Society for Information Science* 41(6), 391–407.
    - Dubossarsky, H., Weinshall, D. & Grossman, E. (2017): Outta control: Laws of semantic change and inherent biases in word representation models. In: *Proceedings of EMNLP 2017*, 1136–1145.
    - Eckart, C. & Young, G. (1936): The approximation of one matrix by another of lower rank. *Psychometrika* 1(3), 211–218.
    - Golub, G. H. & Van Loan, C. F. (2013): *Matrix Computations*. 4. Aufl. Johns Hopkins University Press.
    - Levy, O., Goldberg, Y. & Dagan, I. (2015): Improving distributional similarity with lessons learned from word embeddings. *Transactions of the Association for Computational Linguistics* 3, 211–225.
    - Österlund, A., Ödling, D. & Sahlgren, M. (2015): Factorization of latent variables in distributional semantic models. In: *Proceedings of EMNLP 2015*, 227–231.

    Weitere Angaben zum Messverfahren s. Abschnitt 1. Die bibliografischen Angaben sind vor einer Veröffentlichung an den Originalen zu prüfen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Abhängigkeit der Drift von der Trefferzahl

    ### Verfahren

    Die Verschiebung eines Zielworts zwischen zwei benachbarten Zeitscheiben wird als Kosinusdistanz zwischen seinen beiden Zeilenvektoren gemessen, also als 1 minus Kosinusähnlichkeit (zu diesem Maß Hamilton, Leskovec & Jurafsky 2016; Kutuzov et al. 2018). Im Folgenden heißt dieser Wert **Drift**. Die Trefferzahlen der beiden Zeilen eines Paars sind oft sehr verschieden. Das ist ein Problem, wenn die Drift selbst von der Trefferzahl abhängt: Ein Profil aus wenigen Belegen ist eine ungenaue Schätzung, und die Zufallsschwankung der Stichprobe erscheint als Abstand. Ein Teil der gemessenen Drift ist dann ein Stichprobeneffekt und kein Wandel. Dubossarsky, Weinshall & Grossman (2017) zeigen, dass mehrere in der Literatur berichtete Zusammenhänge zwischen Häufigkeit und Bedeutungswandel unter einer solchen Kontrolle weitgehend verschwinden.

    Dieser Abschnitt prüft, wie stark die Drift in diesen Daten von der Trefferzahl abhängt, und vergleicht Verfahren, die diese Abhängigkeit verringern sollen. Er ergänzt Abschnitt 1: Dort wird gefragt, ob eine einzelne Zeile reproduzierbar ist, hier, ob der Abstand zwischen zwei Zeilen von ihrer Trefferzahl abhängt.

    ### Fragestellung

    Hängt die Drift zwischen benachbarten Zeitscheiben desselben Zielworts von der Trefferzahl der beiden Zeilen ab? Welches Verfahren verringert diese Abhängigkeit am stärksten: Teilstichproben auf eine gemeinsame Trefferzahl oder der Abzug der Drift, die ohne Unterschied zwischen den Zeitscheiben zu erwarten ist?

    ### Methode

    **Paare.** Ausgewertet werden alle Paare benachbarter Zeitscheiben desselben Zielworts, deren beide Zeilen mindestens `DRIFT_FLOOR` Belege haben. Voreingestellt ist h\* aus Abschnitt 1, mindestens aber `DRIFT_FLOOR_MIN` = 10 Belege. Die Untergrenze verhindert, dass die Teilstichproben je Wort (s. u.) auf einzelne Belege fallen; sie ist eine Konvention. Die Vektoren stammen wie in emb aus einer SVD der vollständigen Daten (k und p wie in emb). Neu berechnete Zeilen (Teilstichproben, Permutationen) werden per *folding-in* in diesen Raum projiziert (s. Messverfahren).

    **Kennwert: Volumenabhängigkeit.** Für jedes Paar wird die kleinere der beiden Trefferzahlen bestimmt; sie begrenzt die Genauigkeit des Paars. Gemessen wird der Korrelationskoeffizient nach Pearson, r, zwischen der Drift und dem Logarithmus dieser Trefferzahl, und zwar innerhalb der Zielwörter: Von beiden Größen wird je Zielwort der Mittelwert über seine Paare abgezogen, und die Korrelation wird aus diesen Abweichungen über alle Paare berechnet (Korrelation innerhalb von Gruppen, Bland & Altman 1995). r liegt zwischen −1 und 1. Bei 0 hängt die Drift nicht linear von der logarithmierten Trefferzahl ab. Ein negativer Wert bedeutet, dass ein Wort zwischen schwach belegten Zeitscheiben eine größere Drift zeigt als zwischen gut belegten. Die Berechnung innerhalb der Zielwörter schließt aus, dass Unterschiede zwischen häufigen und seltenen Wörtern den Wert bestimmen. Gefragt ist, ob die Trajektorie *eines* Worts von der Belegung seiner Zeitscheiben abhängt. Einbezogen werden Zielwörter mit mindestens zwei Paaren.

    **Unsicherheit.** Die Unsicherheit von r wird mit einem Bootstrap über die Zielwörter geschätzt (Efron & Tibshirani 1993; Ziehen ganzer Gruppen: Davison & Hinkley 1997). Die Zielwörter werden `DRIFT_N_BOOT`-mal mit Zurücklegen neu gezogen, jedes mit allen seinen Paaren, und r wird jedes Mal neu berechnet. Das 95-%-Intervall reicht vom 2,5-%- bis zum 97,5-%-Quantil dieser Werte (Perzentilintervall). Gezogen werden Zielwörter und nicht Paare, weil die Paare desselben Worts nicht unabhängig voneinander sind.

    **Verglichene Verfahren.** Alle Varianten werden an denselben Paaren berechnet.

    | Variante | Berechnung | verwendete Belege |
    |---|---|---|
    | **Rohdrift** (Referenz, heutiger Stand von emb) | Vektoren der vollständigen Daten | alle |
    | **Teilstichprobe je Paar** | Die größere Zeile eines Paars wird auf die Trefferzahl der kleineren gezogen; Drift gemittelt über `DRIFT_N_DRAWS` Ziehungen | je Paar zweimal die kleinere Trefferzahl |
    | **Teilstichprobe je Wort, Distanzen gemittelt** | Jede ausgewertete Zeile eines Worts wird auf die kleinste Trefferzahl unter diesen Zeilen gezogen; Drift je Ziehung, dann gemittelt | je Zeile die kleinste Trefferzahl ihres Worts |
    | **Teilstichprobe je Wort, Vektoren gemittelt** | dieselben Ziehungen; die Einheitsvektoren einer Zeile werden über die Ziehungen gemittelt, die Drift wird aus den gemittelten Vektoren berechnet | wie zuvor, je Ziehung |
    | **Drift abzüglich Permutationserwartung** | beobachtete Drift minus mittlere Drift unter Permutation (s. u.) | alle |

    Die Teilstichproben werden nach Reden gezogen wie in Abschnitt 1: Reden in zufälliger Reihenfolge, aus der letzten Rede so viele Belege, wie noch fehlen. Ihre PPMI-Werte werden wie in Abschnitt 1 mit den Randsummen berechnet, in denen die vollständige Zeile durch ihre Teilstichprobe ersetzt ist. Die beiden Varianten „je Wort“ unterscheiden sich darin, was sich in emb umsetzen ließe: Gemittelte Vektoren ergeben einen Vektor je Zeile und passen in das bestehende Bundle, gemittelte Distanzen gibt es nur je Paar.

    Die Teilstichprobe je Paar gleicht die Trefferzahl nur innerhalb eines Paars an, nicht zwischen den Paaren eines Worts. Sie wird mitgeführt, weil sie das naheliegende Verfahren ist; das Ergebnis zeigt, ob die Angleichung innerhalb des Paars genügt.

    **Permutationserwartung.** Für jedes Paar werden die Belege beider Zeitscheiben zusammengelegt und `DRIFT_N_PERM`-mal zufällig auf zwei Zeilen mit den tatsächlichen Trefferzahlen verteilt, nach Reden wie oben. Jede Verteilung ergibt eine Drift, wie sie bei gleicher Trefferzahl ohne Unterschied zwischen den beiden Zeitscheiben auftritt (Permutationstest, Good 2005; als Kontrollbedingung für die Messung von Bedeutungswandel: Dubossarsky, Weinshall & Grossman 2017). Damit beobachtete und permutierte Drift auf dieselbe Weise berechnet werden, erhalten in dieser Rechnung beide Zeilen eines Paars dieselben Randsummen, nämlich die beider Zeitscheiben zusammen. Die so berechnete beobachtete Drift („Drift mit gemeinsamen Randsummen“) weicht deshalb von der Rohdrift ab und wird zur Information mitberichtet. Je Paar ergibt sich außerdem ein p-Wert: der Anteil der Permutationen mit mindestens so großer Drift wie beobachtet, berechnet als (1 + Anzahl) / (1 + `DRIFT_N_PERM`) (Phipson & Smyth 2010).

    **Sättigungskurve (zur Information).** Für die Paare, deren beide Zeilen mindestens `H_MAX` Belege haben (Prüfpaare), werden aus jeder Zeile genau h Belege gezogen, für jedes h aus `H_GRID` (geschachtelt wie in Abschnitt 1), und die Drift wird mit gemeinsamen Randsummen berechnet. Ebenso werden aus den zusammengelegten Belegen beider Zeilen zwei getrennte Stichproben von je h Belegen gezogen; ihr Abstand ist die Drift ohne Unterschied zwischen den Zeitscheiben bei h Belegen. Das Vorgehen entspricht der Rarefaktion, bei der Kennwerte an Stichproben gleichen Umfangs verglichen werden (Sanders 1968; Hurlbert 1971). Die Kurve zeigt, bis zu welcher Trefferzahl die Drift überwiegend vom Stichprobenumfang bestimmt wird.

    ### Erwartete Ergebnisse und ihre Deutung

    - Ist r der Rohdrift belastbar negativ, zeigen Wörter zwischen schwach belegten Zeitscheiben eine größere Drift. Das ist der erwartete Stichprobeneffekt. Wie groß er ohne jeden Unterschied zwischen den Zeitscheiben wäre, zeigt r der Permutationserwartung.
    - Ist r der Rohdrift nicht belastbar von 0 verschieden, ist in diesen Daten kein Eingriff nötig.
    - Die Teilstichprobe je Paar sollte r wenig verändern, weil die Genauigkeit eines Paars ohnehin von seiner kleineren Zeile begrenzt wird.
    - Die Teilstichprobe je Wort mit gemittelten Distanzen sollte r am stärksten gegen 0 bringen, weil alle Paare eines Worts mit derselben Trefferzahl gemessen werden. Ihr Preis ist der Verlust an Belegen.
    - Mit gemittelten Vektoren kann ein Teil der Abhängigkeit zurückkehren: Bei einer gut belegten Zeile unterscheiden sich die Ziehungen, und ihr Mittel ist genauer als eine einzelne Ziehung; bei der schwächsten Zeile eines Worts sind alle Ziehungen gleich.
    - Die Drift abzüglich Permutationserwartung bringt r gegen 0, soweit sich der Stichprobeneffekt zur Drift addiert. Sie verwirft keine Belege.
    - In der Sättigungskurve fällt die Drift ohne Unterschied mit wachsendem h. Wird sie klein gegenüber der beobachteten Drift, bestimmt die Trefferzahl die Drift nicht mehr wesentlich.

    ### Entscheidungsregel

    - **Voraussetzung:** Das 95-%-Intervall von r der Rohdrift schließt 0 aus. Andernfalls wird kein Eingriff empfohlen.
    - Für jede Variante wird in jeder Bootstrap-Wiederholung der Betrag |r| der Variante minus |r| der Rohdrift berechnet. Liegt das 95-%-Intervall dieser Differenz ganz unter 0, ist die Volumenabhängigkeit **belastbar geringer**, liegt es ganz über 0, **belastbar höher**, sonst gilt **kein belastbarer Unterschied**. Bewertet wird der Betrag, weil auch eine positive Abhängigkeit eine Abhängigkeit von der Textmenge ist (wie in Abschnitt 3).
    - Empfohlen wird unter den belastbar geringeren Varianten die mit dem kleinsten |r|. Varianten, deren |r| sich von dem der empfohlenen nicht belastbar unterscheidet (Intervall der Differenz schließt 0 ein), werden als gleichwertig genannt. Zwischen ihnen entscheiden Belegverlust und Umsetzbarkeit, die die Auswertung je Variante angibt.
    - Ist keine Variante belastbar geringer, bleibt es bei der Rohdrift.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vorprüfung

    Wie ungleich sind die Trefferzahlen benachbarter Zeitscheiben? Die Tabelle zeigt für alle Paare benachbarter Zeitscheiben desselben Zielworts und für die ausgewerteten Paare das Verhältnis der größeren zur kleineren Trefferzahl (1 = gleich viele Belege).
    """)
    return


@app.cell
def _(fmt_int, fmt_num, fmt_pct, h_star, mo, np, pl, row_hits, row_slice, row_word, show_table):
    DRIFT_FLOOR_MIN = 10
    DRIFT_FLOOR = max(DRIFT_FLOOR_MIN, h_star) if h_star is not None else DRIFT_FLOOR_MIN

    # pairs of adjacent slices of the same word; a = earlier slice, b = later slice
    _word_code = np.unique(row_word, return_inverse=True)[1]
    _order = np.lexsort((row_slice, _word_code))
    _a, _b = _order[:-1], _order[1:]
    _adjacent = (_word_code[_a] == _word_code[_b]) & (row_slice[_b] == row_slice[_a] + 1)
    _a, _b = _a[_adjacent], _b[_adjacent]
    _low = np.minimum(row_hits[_a], row_hits[_b])
    _ratio = np.maximum(row_hits[_a], row_hits[_b]) / _low
    _evaluated = _low >= DRIFT_FLOOR
    mo.stop(
        not _evaluated.any(),
        mo.callout(
            mo.md(f"**Abbruch:** Kein Paar benachbarter Zeitscheiben hat in beiden Zeilen mindestens {DRIFT_FLOOR} Belege."),
            kind="danger",
        ),
    )
    drift_ia, drift_ib = _a[_evaluated], _b[_evaluated]

    def _describe(name, mask):
        r = _ratio[mask]
        return {
            "Paare": name,
            "Anzahl": fmt_int(mask.sum()),
            "Zielwörter": fmt_int(len(np.unique(_word_code[_a][mask]))),
            "Verhältnis, Median": fmt_num(float(np.median(r)), 2),
            "Verhältnis, 90-%-Quantil": fmt_num(float(np.quantile(r, 0.9)), 2),
            "Anteil ≥ 2": fmt_pct((r >= 2).mean()),
            "Anteil ≥ 5": fmt_pct((r >= 5).mean()),
            "Anteil ≥ 10": fmt_pct((r >= 10).mean()),
        }

    show_table(pl.DataFrame([
        _describe("alle benachbarten Paare", np.ones(len(_a), dtype=bool)),
        _describe(f"ausgewertet (beide Zeilen ≥ {DRIFT_FLOOR} Belege)", _evaluated),
    ]))
    return DRIFT_FLOOR, drift_ia, drift_ib


@app.cell
def _(
    SVD_P,
    counts_from_hits,
    drift_ia,
    drift_ib,
    full_V,
    full_counts,
    full_ppmi,
    full_s,
    hit_counts,
    hit_row,
    hit_speech,
    n_rows,
    np,
    ppmi,
    project,
    row_hits,
    row_slice,
    slice_marginals,
    sp,
    unit_rows,
):
    drift_Z, drift_T = slice_marginals(full_counts)
    drift_slice_a, drift_slice_b = row_slice[drift_ia], row_slice[drift_ib]
    drift_n_a, drift_n_b = row_hits[drift_ia], row_hits[drift_ib]
    _nq = len(drift_ia)

    # reference: vectors of the full data, as in emb
    _X = unit_rows(project(full_ppmi, full_V, full_s, SVD_P))
    drift_raw = 1 - np.einsum("ij,ij->i", _X[drift_ia], _X[drift_ib])

    def drift_rank(group, speech, rng):
        """Rank of each entry within its group: speeches in random order, entries within a speech in random
        order (as in section 1). The first h entries of a group form its sample of size h."""
        key = group.astype(np.int64) * (int(speech.max()) + 1) + speech
        _, inv = np.unique(key, return_inverse=True)
        k_speech = rng.random(inv.max() + 1)[inv]
        order = np.lexsort((rng.random(len(group)), k_speech, group))
        g = group[order]
        rank = np.empty(len(group), dtype=np.int64)
        rank[order] = np.arange(len(group)) - np.searchsorted(g, g, side="left")
        return rank

    def _pooled_ppmi(C, sa, sb):
        """PPMI as in emb, but with the marginals of slices sa and sb together (per row of C)."""
        C = C.tocsr()
        row_total = np.asarray(C.sum(axis=1)).ravel()
        coo = C.tocoo()
        a, b = sa[coo.row], sb[coo.row]
        z = drift_Z[a, coo.col] + drift_Z[b, coo.col]
        t = drift_T[a] + drift_T[b]
        with np.errstate(divide="ignore"):
            pmi = np.log2(coo.data * t / (row_total[coo.row] * z))
        M = sp.csr_matrix((np.maximum(pmi, 0), (coo.row, coo.col)), shape=C.shape)
        M.eliminate_zeros()
        return M

    def drift_pooled(sel, target, sa, sb):
        """Drift per pair with shared marginals. Hit sel[t] goes to pseudo-row target[t] = 2 * pair + side;
        sa, sb: slices of the pairs."""
        C = counts_from_hits(hit_counts, sel, target, 2 * len(sa))
        M = _pooled_ppmi(C, np.repeat(sa, 2), np.repeat(sb, 2))
        X = unit_rows(project(M, full_V, full_s, SVD_P))
        return 1 - np.einsum("ij,ij->i", X[0::2], X[1::2])

    def drift_vectors(sel, target, rows_of):
        """Unit vectors as in emb for subsamples: hit sel[t] goes to output row target[t], which replaces
        row rows_of[target[t]] of the full data (marginals adjusted as in section 1)."""
        C = counts_from_hits(hit_counts, sel, target, len(rows_of))
        M = ppmi(C, row_slice[rows_of], drift_Z, drift_T, replaced=full_counts[rows_of])
        return unit_rows(project(M, full_V, full_s, SVD_P))

    # pool: every hit of an evaluated pair, once for each pair its row belongs to (at most two)
    _pa = np.full(n_rows, -1)
    _pa[drift_ia] = np.arange(_nq)
    _pb = np.full(n_rows, -1)
    _pb[drift_ib] = np.arange(_nq)
    _ha = np.flatnonzero(_pa[hit_row] >= 0)
    _hb = np.flatnonzero(_pb[hit_row] >= 0)
    drift_pool_hit = np.concatenate([_ha, _hb])
    drift_pool_pair = np.concatenate([_pa[hit_row[_ha]], _pb[hit_row[_hb]]])
    drift_pool_side = np.concatenate([np.zeros(len(_ha), dtype=np.int64), np.ones(len(_hb), dtype=np.int64)])
    drift_pool_speech = hit_speech[drift_pool_hit]
    return (
        drift_n_a,
        drift_n_b,
        drift_pool_hit,
        drift_pool_pair,
        drift_pool_side,
        drift_pool_speech,
        drift_pooled,
        drift_rank,
        drift_raw,
        drift_slice_a,
        drift_slice_b,
        drift_vectors,
    )


@app.cell
def _(
    SEED,
    drift_n_a,
    drift_pool_hit,
    drift_pool_pair,
    drift_pool_side,
    drift_pool_speech,
    drift_pooled,
    drift_rank,
    drift_slice_a,
    drift_slice_b,
    mo,
    np,
):
    DRIFT_N_PERM = 99

    # observed drift with shared marginals, then the same statistic with the hits of each pair redistributed
    # by speech onto two rows of the actual sizes (the first n_a hits of the random order go to side a)
    _rng = np.random.default_rng(SEED + 4000)
    drift_pooled_obs = drift_pooled(
        drift_pool_hit, 2 * drift_pool_pair + drift_pool_side, drift_slice_a, drift_slice_b
    )
    drift_null = np.full((DRIFT_N_PERM, len(drift_slice_a)), np.nan)
    for _x in mo.status.progress_bar(range(DRIFT_N_PERM), title="Permutationen"):
        _rank = drift_rank(drift_pool_pair, drift_pool_speech, _rng)
        _side = (_rank >= drift_n_a[drift_pool_pair]).astype(np.int64)
        drift_null[_x] = drift_pooled(drift_pool_hit, 2 * drift_pool_pair + _side, drift_slice_a, drift_slice_b)
    return DRIFT_N_PERM, drift_null, drift_pooled_obs


@app.cell
def _(
    SEED,
    drift_ia,
    drift_ib,
    drift_n_a,
    drift_n_b,
    drift_pool_hit,
    drift_pool_pair,
    drift_pool_side,
    drift_pool_speech,
    drift_rank,
    drift_vectors,
    hit_row,
    hit_speech,
    mo,
    n_rows,
    np,
    row_hits,
    row_word,
    unit_rows,
):
    DRIFT_N_DRAWS = 20
    _rng = np.random.default_rng(SEED + 5000)
    _nq = len(drift_ia)

    # per pair: the larger row is drawn down to the hits of the smaller one (output rows 2 * pair + side)
    _m_pair = np.minimum(drift_n_a, drift_n_b)
    _group = 2 * drift_pool_pair + drift_pool_side
    _rows_of = np.empty(2 * _nq, dtype=np.int64)
    _rows_of[0::2], _rows_of[1::2] = drift_ia, drift_ib
    _d_pair = np.full((DRIFT_N_DRAWS, _nq), np.nan)

    # per word: every evaluated row of a word is drawn down to the smallest hits among these rows
    _rows_e = np.unique(np.concatenate([drift_ia, drift_ib]))
    _pos = np.full(n_rows, -1)
    _pos[_rows_e] = np.arange(len(_rows_e))
    _word_code = np.unique(row_word, return_inverse=True)[1]
    _m_word = np.full(_word_code.max() + 1, np.iinfo(np.int64).max)
    np.minimum.at(_m_word, _word_code[_rows_e], row_hits[_rows_e])
    _m_row = _m_word[_word_code[_rows_e]]
    _hsel = np.flatnonzero(_pos[hit_row] >= 0)
    _hgroup = _pos[hit_row[_hsel]]
    _d_word = np.full((DRIFT_N_DRAWS, _nq), np.nan)
    _sum_vectors = np.zeros(1)

    for _x in mo.status.progress_bar(range(DRIFT_N_DRAWS), title="Ziehungen"):
        _rank = drift_rank(_group, drift_pool_speech, _rng)
        _keep = _rank < _m_pair[drift_pool_pair]
        _X = drift_vectors(drift_pool_hit[_keep], _group[_keep], _rows_of)
        _d_pair[_x] = 1 - np.einsum("ij,ij->i", _X[0::2], _X[1::2])

        _rank = drift_rank(_hgroup, hit_speech[_hsel], _rng)
        _keep = _rank < _m_row[_hgroup]
        _X = drift_vectors(_hsel[_keep], _hgroup[_keep], _rows_e)
        _d_word[_x] = 1 - np.einsum("ij,ij->i", _X[_pos[drift_ia]], _X[_pos[drift_ib]])
        _sum_vectors = _sum_vectors + np.nan_to_num(_X)

    _X_mean = unit_rows(_sum_vectors)
    drift_sub = {
        "pair": np.nanmean(_d_pair, axis=0),
        "word_distances": np.nanmean(_d_word, axis=0),
        "word_vectors": 1 - np.einsum("ij,ij->i", _X_mean[_pos[drift_ia]], _X_mean[_pos[drift_ib]]),
    }
    # share of the hits of the evaluated rows that a variant uses
    drift_used = {
        "pair": float(2 * _m_pair.sum() / (drift_n_a + drift_n_b).sum()),
        "word": float(_m_row.sum() / row_hits[_rows_e].sum()),
    }
    return DRIFT_N_DRAWS, drift_sub, drift_used


@app.cell
def _(
    DRIFT_N_DRAWS,
    H_GRID,
    H_MAX,
    SEED,
    drift_n_a,
    drift_n_b,
    drift_null,
    drift_pool_hit,
    drift_pool_pair,
    drift_pool_side,
    drift_pool_speech,
    drift_pooled,
    drift_pooled_obs,
    drift_rank,
    drift_slice_a,
    drift_slice_b,
    mo,
    np,
):
    # saturation curve on the probe pairs (both rows with at least H_MAX hits); mean over the probe pairs per draw
    _rng = np.random.default_rng(SEED + 6000)
    _probes = np.flatnonzero(np.minimum(drift_n_a, drift_n_b) >= H_MAX)
    drift_probe_n = len(_probes)
    drift_rare_obs = np.full((DRIFT_N_DRAWS, len(H_GRID)), np.nan)
    drift_rare_null = np.full((DRIFT_N_DRAWS, len(H_GRID)), np.nan)
    drift_rare_full = (np.nan, np.nan)

    if drift_probe_n > 0:
        _pp = np.full(len(drift_n_a), -1)
        _pp[_probes] = np.arange(drift_probe_n)
        _sel = _pp[drift_pool_pair] >= 0
        _pair = _pp[drift_pool_pair[_sel]]
        _side = drift_pool_side[_sel]
        _hit = drift_pool_hit[_sel]
        _speech = drift_pool_speech[_sel]
        _target = 2 * _pair + _side
        # null: two disjoint samples from the pooled hits, the second one starting at half of the pool
        _start_b = ((drift_n_a[_probes] + drift_n_b[_probes]) // 2)[_pair]
        _sa, _sb = drift_slice_a[_probes], drift_slice_b[_probes]
        for _x in mo.status.progress_bar(range(DRIFT_N_DRAWS), title="Ziehungen"):
            _r_obs = drift_rank(_target, _speech, _rng)
            _r_null = drift_rank(_pair, _speech, _rng)
            for _g, _h in enumerate(H_GRID):
                _k = _r_obs < _h
                drift_rare_obs[_x, _g] = np.nanmean(drift_pooled(_hit[_k], _target[_k], _sa, _sb))
                _in_b = (_r_null >= _start_b) & (_r_null < _start_b + _h)
                _k = (_r_null < _h) | _in_b
                drift_rare_null[_x, _g] = np.nanmean(
                    drift_pooled(_hit[_k], (2 * _pair + _in_b)[_k], _sa, _sb)
                )
        # all hits: observed drift with shared marginals and mean drift under permutation (actual sizes)
        drift_rare_full = (
            float(np.nanmean(drift_pooled_obs[_probes])),
            float(np.nanmean(drift_null[:, _probes])),
        )
    return drift_probe_n, drift_rare_full, drift_rare_null, drift_rare_obs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Auswertung

    **Tabelle 4.1, Sättigungskurve:** je Trefferzahl h die mittlere Drift der Prüfpaare bei h Belegen je Zeile und die mittlere Drift ohne Unterschied zwischen den Zeitscheiben, gemittelt über die Ziehungen. Die letzte Zeile zeigt die Prüfpaare mit allen Belegen (ohne Unterschied: Mittel über die Permutationen).

    **Tabelle 4.2, Volumenabhängigkeit:** r innerhalb der Zielwörter je Variante mit 95-%-Intervall, die Differenz der Beträge gegenüber der Rohdrift mit 95-%-Intervall und die Einstufung nach der Entscheidungsregel. Die Rangkorrelation nach Spearman (1904) gibt an, wie weit die Variante die Reihenfolge der Paare nach ihrer Drift gegenüber der Rohdrift erhält (1 = gleiche Reihenfolge). Die beiden letzten Zeilen sind Zwischengrößen der Permutationsrechnung und gehen nicht in die Entscheidung ein.

    **Tabelle 4.3, Permutationstest:** je Gruppe nach der kleineren Trefferzahl eines Paars (Terzile) der Anteil der Paare mit p < 0,05. Ohne jeden Unterschied zwischen den Zeitscheiben wären etwa 5 % zu erwarten.
    """)
    return


@app.cell
def _(
    DRIFT_FLOOR,
    DRIFT_N_DRAWS,
    DRIFT_N_PERM,
    H_GRID,
    H_MAX,
    SEED,
    SVD_K,
    SVD_P,
    drift_ia,
    drift_n_a,
    drift_n_b,
    drift_null,
    drift_pooled_obs,
    drift_probe_n,
    drift_rare_full,
    drift_rare_null,
    drift_rare_obs,
    drift_raw,
    drift_sub,
    drift_used,
    fmt_int,
    fmt_num,
    fmt_pct,
    mo,
    np,
    pl,
    row_word,
    show_table,
):
    from scipy.stats import spearmanr as _spearmanr

    DRIFT_N_BOOT = 1000

    _null_mean = np.nanmean(drift_null, axis=0)
    _ref = "Rohdrift (Referenz)"
    # variant: (drift per pair, share of hits used, implementation)
    _variants = {
        _ref: (drift_raw, 1.0, "heutiger Stand von emb"),
        "Teilstichprobe je Paar": (
            drift_sub["pair"], drift_used["pair"],
            "liefert einen Abstand je Paar, aber keinen Vektor je Zeile; im Bundle nur als zusätzliches Feld je Paar "
            "darstellbar",
        ),
        "Teilstichprobe je Wort, Distanzen gemittelt": (
            drift_sub["word_distances"], drift_used["word"],
            "liefert einen Abstand je Paar, aber keinen Vektor je Zeile; im Bundle nur als zusätzliches Feld je Paar "
            "darstellbar",
        ),
        "Teilstichprobe je Wort, Vektoren gemittelt": (
            drift_sub["word_vectors"], drift_used["word"],
            "liefert einen Vektor je Zeile und ist als Schalter in emb umsetzbar, ohne das Bundle-Schema zu ändern",
        ),
        "Drift abzüglich Permutationserwartung": (
            drift_pooled_obs - _null_mean, 1.0,
            "verwirft keine Belege; liefert Abstand und p-Wert je Paar, die im Bundle ein zusätzliches Feld brauchen",
        ),
    }
    _info = {
        "Drift mit gemeinsamen Randsummen": drift_pooled_obs,
        "Permutationserwartung (ohne Unterschied)": _null_mean,
    }
    _all = {**{_n: _v[0] for _n, _v in _variants.items()}, **_info}

    # common pairs: finite in every variant, words with at least two such pairs
    _valid = np.all([np.isfinite(_x) for _x in _all.values()], axis=0)
    _word = np.unique(row_word[drift_ia], return_inverse=True)[1]
    _use = _valid & (np.bincount(_word[_valid], minlength=_word.max() + 1)[_word] >= 2)
    _w = np.unique(_word[_use], return_inverse=True)[1]
    _nw = int(_w.max()) + 1
    _log_n = np.log(np.minimum(drift_n_a, drift_n_b)[_use].astype(np.float64))

    def _demean(x):
        return x - (np.bincount(_w, x, minlength=_nw) / np.bincount(_w, minlength=_nw))[_w]

    # within-word sums per word: products, squares of drift and log hits (deviations from the word mean)
    _dy = _demean(_log_n)
    _sums = {}
    for _n, _x in _all.items():
        _dx = _demean(_x[_use].astype(np.float64))
        _sums[_n] = [np.bincount(_w, _v, minlength=_nw) for _v in (_dx * _dy, _dx * _dx, _dy * _dy)]

    def _r(sums, weights):
        xy, xx, yy = sums
        return (weights @ xy) / np.sqrt((weights @ xx) * (weights @ yy))

    # bootstrap over words: each row of _weights counts how often each word is drawn
    _weights = np.random.default_rng(SEED + 7000).multinomial(
        _nw, np.full(_nw, 1 / _nw), size=DRIFT_N_BOOT
    ).astype(np.float64)
    _r_full = {_n: float(_r(_s, np.ones(_nw))) for _n, _s in _sums.items()}
    _r_boot = {_n: _r(_s, _weights) for _n, _s in _sums.items()}

    def _ci(x):
        return tuple(float(_q) for _q in np.quantile(x, [0.025, 0.975]))

    def _fmt_ci(x):
        lo, hi = _ci(x)
        return f"{fmt_num(lo)} bis {fmt_num(hi)}"

    _delta = {_n: np.abs(_r_boot[_n]) - np.abs(_r_boot[_ref]) for _n in _variants if _n != _ref}
    _classes = {}
    for _n, _d in _delta.items():
        _lo, _hi = _ci(_d)
        _classes[_n] = "belastbar geringer" if _hi < 0 else "belastbar höher" if _lo > 0 else "kein belastbarer Unterschied"
    _rho = {_n: float(_spearmanr(_x[_use], drift_raw[_use])[0]) for _n, _x in _all.items()}

    # table 4.1: saturation curve
    _rare_obs, _rare_null = drift_rare_obs.mean(axis=0), drift_rare_null.mean(axis=0)
    _share = list(_rare_null / _rare_obs) + [drift_rare_full[1] / drift_rare_full[0]]
    _table_rare = pl.DataFrame({
        "Belege je Zeile": [str(_h) for _h in H_GRID] + [f"alle (≥ {H_MAX})"],
        "Drift beobachtet": [fmt_num(_x) for _x in _rare_obs] + [fmt_num(drift_rare_full[0])],
        "Drift ohne Unterschied": [fmt_num(_x) for _x in _rare_null] + [fmt_num(drift_rare_full[1])],
        "ohne Unterschied / beobachtet": [fmt_pct(_x) if np.isfinite(_x) else "–" for _x in _share],
    })

    # table 4.2: volume dependence
    _rows_vol = []
    for _n in _all:
        _is_variant = _n in _variants
        _rows_vol.append({
            "Variante": _n,
            "Belege verwendet": fmt_pct(_variants[_n][1]) if _is_variant else "alle",
            "r": fmt_num(_r_full[_n]),
            "r, 95-%-Intervall": _fmt_ci(_r_boot[_n]),
            "Δ|r| gegenüber Rohdrift": fmt_num(float(np.abs(_r_full[_n]) - np.abs(_r_full[_ref])), 3) if _n != _ref else "–",
            "Δ|r|, 95-%-Intervall": _fmt_ci(_delta[_n]) if _n in _delta else "–",
            "Einstufung": _classes.get(_n, "Referenz" if _n == _ref else "zur Information"),
            "Rangkorrelation mit Rohdrift": fmt_num(_rho[_n]),
        })

    # table 4.3: permutation test by terciles of the smaller hits of a pair
    _p = (1 + (drift_null >= drift_pooled_obs[None, :]).sum(axis=0)) / (1 + DRIFT_N_PERM)
    _low = np.minimum(drift_n_a, drift_n_b)
    _ok = np.isfinite(drift_pooled_obs)
    _idx = np.flatnonzero(_ok)
    _rank = np.empty(len(_idx), dtype=np.int64)
    _rank[np.argsort(_low[_idx], kind="stable")] = np.arange(len(_idx))
    _groups = {"alle Paare": _idx}
    for _g, _name in enumerate(["geringe Textmenge", "mittlere Textmenge", "hohe Textmenge"]):
        _m = _idx[_rank * 3 // len(_idx) == _g]
        if len(_m):
            _groups[f"{_name} ({_low[_m].min()}–{_low[_m].max()} Belege)"] = _m
    _table_perm = pl.DataFrame([
        {
            "Gruppe": _g,
            "Paare": fmt_int(len(_m)),
            "Anteil p < 0,05": fmt_pct(float((_p[_m] < 0.05).mean())),
            "Drift mit gemeinsamen Randsummen": fmt_num(float(np.mean(drift_pooled_obs[_m]))),
            "Permutationserwartung": fmt_num(float(np.nanmean(_null_mean[_m]))),
        }
        for _g, _m in _groups.items()
    ])

    # verdict
    _lo0, _hi0 = _ci(_r_boot[_ref])
    _text = [
        f"Ausgewertet: {fmt_int(_use.sum())} Paare aus {fmt_int(_nw)} Zielwörtern (beide Zeilen mindestens "
        f"{DRIFT_FLOOR} Belege, mindestens zwei Paare je Zielwort). SVD mit k = {SVD_K}, p = {SVD_P:g}; "
        f"{DRIFT_N_DRAWS} Ziehungen, {DRIFT_N_PERM} Permutationen, {DRIFT_N_BOOT} Bootstrap-Wiederholungen.",
        f"Sättigungskurve ({fmt_int(drift_probe_n)} Prüfpaare): Bei {H_GRID[0]} Beleg(en) je Zeile beträgt die Drift "
        f"ohne Unterschied {fmt_pct(_share[0]) if np.isfinite(_share[0]) else '–'} der beobachteten, bei {H_MAX} "
        f"Belegen {fmt_pct(_share[-2]) if np.isfinite(_share[-2]) else '–'}, mit allen Belegen "
        f"{fmt_pct(_share[-1]) if np.isfinite(_share[-1]) else '–'}.",
        f"Permutationserwartung: r = {fmt_num(_r_full['Permutationserwartung (ohne Unterschied)'])} "
        f"(95-%-Intervall {_fmt_ci(_r_boot['Permutationserwartung (ohne Unterschied)'])}). So stark hinge die Drift "
        "allein über den Stichprobenumfang von der Trefferzahl ab, wenn sich die Zeitscheiben nicht unterschieden.",
        f"Permutationstest: Bei {fmt_pct(float((_p[_ok] < 0.05).mean()))} der Paare liegt die beobachtete Drift über "
        "dem, was ohne Unterschied zu erwarten wäre (p < 0,05; ohne jeden Unterschied etwa 5 %).",
    ]
    if not (_hi0 < 0 or _lo0 > 0):
        _text.append(
            f"Rohdrift: r = {fmt_num(_r_full[_ref])} (95-%-Intervall {_fmt_ci(_r_boot[_ref])}). Das Intervall schließt 0 "
            "ein; eine Abhängigkeit der Drift von der Trefferzahl ist in diesen Daten nicht nachweisbar."
        )
        _kind = "success"
        _rec = "**Empfehlung:** kein Eingriff; die Drift wird wie bisher aus den Vektoren der vollständigen Daten berechnet."
    else:
        _text.append(
            f"Rohdrift: r = {fmt_num(_r_full[_ref])} (95-%-Intervall {_fmt_ci(_r_boot[_ref])}). "
            + (
                "Ein Wort zeigt zwischen schwach belegten Zeitscheiben belastbar eine größere Drift als zwischen gut "
                "belegten."
                if _hi0 < 0
                else "Ein Wort zeigt zwischen gut belegten Zeitscheiben belastbar eine größere Drift als zwischen "
                "schwach belegten; das entspricht nicht dem erwarteten Stichprobeneffekt (s. Einschränkungen)."
            )
        )
        _better = [_n for _n, _c in _classes.items() if _c == "belastbar geringer"]
        _worse = [_n for _n, _c in _classes.items() if _c == "belastbar höher"]
        if _worse:
            _text.append("Belastbar höhere Volumenabhängigkeit als die Rohdrift: " + "; ".join(_worse) + ".")
        if not _better:
            _kind = "warn"
            _rec = (
                "**Empfehlung:** Keine Variante verringert die Volumenabhängigkeit belastbar. Es bleibt bei der "
                "Rohdrift; die Trefferzahl `n_hits` je Zeile liegt im Bundle vor, die Anzeige regelt das Frontend."
            )
        else:
            _best = min(_better, key=lambda n: abs(_r_full[n]))
            _equivalent = []
            for _n in _better:
                if _n != _best:
                    _lo, _hi = _ci(np.abs(_r_boot[_n]) - np.abs(_r_boot[_best]))
                    if _lo <= 0 <= _hi:
                        _equivalent.append(_n)
            _kind = "success"
            _rec = (
                f"**Empfehlung: {_best}** (r = {fmt_num(_r_full[_best])} gegenüber {fmt_num(_r_full[_ref])} der "
                f"Rohdrift; verwendet {fmt_pct(_variants[_best][1])} der Belege; {_variants[_best][2]})."
            )
            if _equivalent:
                _rec += " Gleichwertig in |r|: " + "; ".join(
                    f"{_n} (r = {fmt_num(_r_full[_n])}; verwendet {fmt_pct(_variants[_n][1])} der Belege; "
                    f"{_variants[_n][2]})"
                    for _n in _equivalent
                ) + ". Zwischen gleichwertigen Varianten entscheiden Belegverlust und Umsetzbarkeit."

    mo.vstack([
        mo.md("**Tabelle 4.1: Sättigungskurve**"),
        show_table(_table_rare),
        mo.md("**Tabelle 4.2: Volumenabhängigkeit der Drift**"),
        show_table(pl.DataFrame(_rows_vol)),
        mo.md("**Tabelle 4.3: Permutationstest nach Textmenge**"),
        show_table(_table_perm),
        mo.md("\n\n".join(_text)),
        mo.callout(mo.md(_rec), kind=_kind),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Einschränkungen

    - Die Volumenabhängigkeit unterscheidet nicht zwischen einem Stichprobeneffekt und echtem Wandel, der mit der Trefferzahl einhergeht: Über ein Wort wird oft gerade dann viel geredet, wenn sich sein Kontext ändert (Methodology 1.18). Eine solche positive Abhängigkeit kann einen negativen Stichprobeneffekt in r der Rohdrift teilweise verdecken. r der Permutationserwartung enthält nur den Stichprobeneffekt.
    - r misst eine lineare Abhängigkeit von der logarithmierten Trefferzahl. Eine Abhängigkeit, die nur bei sehr kleinen Trefferzahlen auftritt, zeigt die Sättigungskurve deutlicher.
    - Ausgewertet werden nur benachbarte Zeitscheiben. Paare über Lücken hinweg und Abstände über mehrere Zeitscheiben werden nicht geprüft.
    - Die Untergrenze `DRIFT_FLOOR` ist eine Konvention. Die Paare darunter, bei denen der Stichprobeneffekt am größten ist, gehen nicht ein; für die Zeilen darunter gelten die Abschnitte 1 und 2.
    - Die Teilstichproben je Wort hängen von der schwächsten ausgewerteten Zeile eines Worts ab: Eine einzige schwach belegte Zeitscheibe bestimmt die Trefferzahl aller Zeitscheiben des Worts.
    - Neu berechnete Zeilen werden in den Raum der vollständigen Daten projiziert, an dessen SVD die vollständigen Zeilen beteiligt sind (wie in Abschnitt 1).
    - In der Permutationsrechnung erhalten beide Zeilen eines Paars dieselben Randsummen. Unterschiede zwischen den Zeitscheiben, die nur in den Randsummen liegen, also in den Kontexten aller übrigen Zielwörter, gehen in diese Rechnung nicht ein; in die Rohdrift gehen sie ein.
    - Die Ziehung nach Reden teilt höchstens eine Rede je Zeile bzw. je Paar auf.
    - Die p-Werte je Paar sind nicht für multiples Testen korrigiert. Der Anteil unter 0,05 ist mit den etwa 5 % zu vergleichen, die ohne jeden Unterschied zu erwarten sind; er sagt nicht, welche einzelnen Paare sich unterscheiden.
    - Das Ergebnis gilt für k, p, Raster und KWIC-Variante dieses Laufs und ist nach jeder Änderung neu zu bestimmen.

    ### Literatur

    - Bland, J. M. & Altman, D. G. (1995): Calculating correlation coefficients with repeated observations: Part 1 – correlation within subjects. *BMJ* 310(6977), 446.
    - Davison, A. C. & Hinkley, D. V. (1997): *Bootstrap Methods and their Application*. Cambridge University Press.
    - Dubossarsky, H., Weinshall, D. & Grossman, E. (2017): Outta control: Laws of semantic change and inherent biases in word representation models. In: *Proceedings of EMNLP 2017*, 1136–1145.
    - Efron, B. & Tibshirani, R. J. (1993): *An Introduction to the Bootstrap*. Chapman & Hall.
    - Good, P. (2005): *Permutation, Parametric and Bootstrap Tests of Hypotheses*. 3. Aufl. Springer.
    - Hamilton, W. L., Leskovec, J. & Jurafsky, D. (2016): Diachronic word embeddings reveal statistical laws of semantic change. In: *Proceedings of ACL 2016*, 1489–1501.
    - Hurlbert, S. H. (1971): The nonconcept of species diversity: a critique and alternative parameters. *Ecology* 52(4), 577–586.
    - Kutuzov, A., Øvrelid, L., Szymanski, T. & Velldal, E. (2018): Diachronic word embeddings and semantic shifts: a survey. In: *Proceedings of COLING 2018*, 1384–1397.
    - Phipson, B. & Smyth, G. K. (2010): Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology* 9(1), Art. 39.
    - Sanders, H. L. (1968): Marine benthic diversity: a comparative study. *The American Naturalist* 102(925), 243–282.
    - Spearman, C. (1904): The proof and measurement of association between two things. *The American Journal of Psychology* 15(1), 72–101.

    Weitere Angaben zum Messverfahren s. Abschnitt 1. Die bibliografischen Angaben sind vor einer Veröffentlichung an den Originalen zu prüfen.
    """)
    return


if __name__ == "__main__":
    app.run()
