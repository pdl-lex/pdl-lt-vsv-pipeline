import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import math

    import marimo as mo
    import numpy as np
    import polars as pl

    return math, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diagnose zu pipeline_post.py

    Dieses Notebook prüft Verfahrensentscheidungen der Postprocessing-Stufe, also die Auswahl der nächsten Nachbarn aus dem Vektorraum, an den Daten:

    1. Hubness der Nachbarlisten
    2. Hubness-Korrektur mit CSLS und Mutual Proximity
    3. Länge der Nachbarlisten für eine Filterung nach der Auswahl *(geplant)*
    4. Schnittmenge der Nachbarlisten benachbarter Zeitscheiben und Rauschboden *(geplant)*

    Jeder Abschnitt ist gleich aufgebaut: Verfahren, Fragestellung, Methode, erwartete Ergebnisse und ihre Deutung, Entscheidungsregel, Berechnung, automatische Auswertung mit Empfehlung. Die Auswertung leitet ihre Aussagen ausschließlich aus den Werten ab, die im selben Lauf berechnet werden. Ergebnisse früherer Messungen werden nicht übernommen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Daten

    Grundlage sind die Ausgaben von pipeline_emb_ppmi.py, also dieselben Eingaben, die pipeline_post.py verwendet:

    - die Zeilenvektoren (eine Zeile je Paar aus Zielwort und Zeitscheibe, z. B. „Sicherheit, 2022-Q2“),
    - die Kontextwort-Vektoren (Spaltenseite der SVD),
    - die Tabellen zu Zeilen und Spalten mit Wortart und Belegung,
    - die Liste der Rednernamen aus pipeline_pre_ppmi.py.

    Die Nachbarkandidaten werden mit denselben Filtern bestimmt wie in pipeline_post.py: Mindestbelegung, Wortart NOUN/PROPN/ADJ/VERB, keine Rednernamen. Die Parameter stehen in der folgenden Zelle und müssen mit pipeline_post.py übereinstimmen. Die Nachbarn einer Zeile sind die Kandidaten mit dem höchsten Kosinus zum Zeilenvektor.

    Begriffe: Eine **Zeile** ist ein Paar aus Zielwort und Zeitscheibe. Ein **Kandidat** ist ein Kontextwort, das die Filter besteht. Eine **Nachbarliste der Länge $k_{\mathrm{NN}}$** enthält die $k_{\mathrm{NN}}$ Kandidaten mit dem höchsten Kosinus zu einer Zeile. Die Listenlänge heißt hier $k_{\mathrm{NN}}$ (im Code `k_nn`), um sie vom Rang k der SVD in pipeline_emb_ppmi.py zu unterscheiden; in der Literatur zu Nachbarlisten heißt sie meist $k$.
    """)
    return


@app.cell
def _(mo, np, pl):
    # filter parameters, must match pipeline_post.py
    NEIGHBOR_POS_ALLOWED = ["NOUN", "PROPN", "ADJ", "VERB"]

    _emb = mo.notebook_dir() / ".." / "02_embedding" / "output"
    _pre = mo.notebook_dir() / ".." / "01_preprocessing" / "output"

    target_slice_vectors = np.load(_emb / "target_slice_vectors.npy")
    context_vectors = np.load(_emb / "context_vectors.npy")
    target_slices = pl.read_parquet(_emb / "target_slices.parquet")
    context_words = pl.read_parquet(_emb / "context_words.parquet")
    speaker_names = pl.read_csv(_pre / "speaker_names.csv")["name"]

    # vectors and labels are matched by position only
    assert (target_slices["target_slice_idx"].to_numpy() == np.arange(target_slice_vectors.shape[0])).all()
    assert (context_words["context_idx"].to_numpy() == np.arange(context_vectors.shape[0])).all()

    neighbor_n_target_slices_min = max(10, round(len(target_slices) * 0.02))

    candidates = context_words.filter(
        (pl.col("n_target_slices") >= neighbor_n_target_slices_min)
        & pl.col("pos").is_in(NEIGHBOR_POS_ALLOWED)
        & ~(
            (pl.col("pos") == "PROPN")
            & pl.col("context_lemma").is_in(speaker_names.implode())
        )
    )

    # cosine table: one row per matrix row, one column per candidate.
    # Column i belongs to row i of candidates; do not re-sort candidates.
    _target_slice_vectors_unit = target_slice_vectors / np.linalg.norm(target_slice_vectors, axis=1, keepdims=True)
    _cand = context_vectors[candidates["context_idx"].to_numpy()]
    _cand_unit = _cand / np.maximum(np.linalg.norm(_cand, axis=1, keepdims=True), 1e-12)
    cosine = _target_slice_vectors_unit @ _cand_unit.T

    # target word per row, as an index into the sorted list of target words
    target_words = sorted(target_slices["target_lemma"].unique().to_list())
    row_word = np.array([target_words.index(_w) for _w in target_slices["target_lemma"].to_list()])

    mo.md(
        f"Zeilen: {cosine.shape[0]} ({len(target_words)} Zielwörter), "
        f"Kandidaten: {cosine.shape[1]:,} von {len(context_words):,} Kontextwörtern "
        f"(Mindestbelegung {neighbor_n_target_slices_min})".replace(",", ".")
    )
    return candidates, cosine, row_word, target_slices, target_words


@app.cell
def _(math, mo, np):
    def show_table(df):
        """Table with all rows on one page."""
        return mo.ui.table(df, pagination=False, selection=None, show_column_summaries=False, show_data_types=False)

    def fmt_num(x, sign=False, digits=3):
        return (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace("-", "−").replace(".", ",")

    def fmt_pct(x):
        return f"{x * 100:.0f} %"

    def nn_membership(C, k_nn):
        """Boolean matrix (rows x candidates): True where the candidate is among the k_nn highest cosines of the row."""
        top = np.argpartition(-C, k_nn - 1, axis=1)[:, :k_nn]
        member = np.zeros(C.shape, dtype=bool)
        np.put_along_axis(member, top, True, axis=1)
        return member

    def skewness(x):
        """Third standardized moment, as used for the k-occurrence by Radovanović et al. (2010)."""
        x = np.asarray(x, dtype=float)
        return float(((x - x.mean()) ** 3).mean() / x.std() ** 3)

    def binomial_skewness(n, p):
        """Skewness of a binomial distribution B(n, p)."""
        return float((1 - 2 * p) / np.sqrt(n * p * (1 - p)))

    _erf = np.vectorize(math.erf)

    def normal_cdf(z):
        """Distribution function of the standard normal distribution, elementwise."""
        return 0.5 * (1.0 + _erf(np.asarray(z) / math.sqrt(2.0)))

    return (
        binomial_skewness,
        fmt_num,
        fmt_pct,
        nn_membership,
        normal_cdf,
        show_table,
        skewness,
    )


@app.cell
def _(
    binomial_skewness,
    candidates,
    nn_membership,
    np,
    pl,
    row_word,
    skewness,
    target_words,
):
    HUB_TOP = 20

    def hub_statistics(C, k_nn):
        """k-occurrence statistics of the neighbor lists of length k_nn under the score matrix C (rows x candidates).
        Returns the summary values, the HUB_TOP candidates with the highest k-occurrence and the membership matrix."""
        n, m = C.shape
        member = nn_membership(C, k_nn)
        nk = member.sum(axis=0)
        # breadth: number of target words with at least one row listing the candidate
        word_onehot = np.eye(len(target_words), dtype=int)[row_word]  # rows x target words
        breadth = ((word_onehot.T @ member.astype(int)) > 0).sum(axis=0)

        top = (
            candidates.select(["context_lemma", "pos", "n_target_slices"])
            .with_columns(
                pl.Series("N", nk),
                pl.Series("Anteil Zeilen", np.round(nk / n, 3)),
                pl.Series("Zielwörter", breadth),
            )
            .sort(["N", "context_lemma"], descending=[True, False])
            .head(HUB_TOP)
        )
        summary = {
            "mittlere N": round(float(nk.mean()), 2),
            "größte N": int(nk.max()),
            "Anteil N = 0": round(float((nk == 0).mean()), 3),
            "Schiefe": round(skewness(nk), 2),
            "Binomialreferenz": round(binomial_skewness(n, k_nn / m), 2),
            "zielwortübergreifend unter Top 20": int((top["Zielwörter"] == len(target_words)).sum()),
        }
        return summary, top, member

    def is_hubness(summary):
        """Decision rule of section 1: skewness above the binomial reference and more than half of the
        HUB_TOP most frequent candidates occur in the lists of all target words."""
        return (
            summary["Schiefe"] > summary["Binomialreferenz"]
            and summary["zielwortübergreifend unter Top 20"] > HUB_TOP / 2
        )

    return HUB_TOP, hub_statistics, is_hubness


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Hubness der Nachbarlisten

    ### Verfahren

    In hochdimensionalen Räumen kommen einzelne Punkte in den Nachbarlisten sehr vieler anderer Punkte vor, andere in keiner. Radovanović, Nanopoulos & Ivanović (2010) bezeichnen diese Eigenschaft als **Hubness**, die häufig genannten Punkte als **Hubs** und die nie genannten als **Antihubs**. Gemessen wird sie über die **k-Occurrence**: die Zahl der Nachbarlisten der Länge $k_{\mathrm{NN}}$, in denen ein Punkt $c$ vorkommt, hier geschrieben als $N(c)$. Als zusammenfassende Kennzahl dient die Schiefe der Verteilung von $N$ über alle Punkte (k-Skewness):

    $$
    S_N = \frac{\mathrm{E}\left[(N - \mu_N)^3\right]}{\sigma_N^3}
    $$

    Eine positive Schiefe bedeutet: Wenige Punkte kommen in sehr vielen Listen vor, die meisten in wenigen oder keiner. Hubs sind für Nachbarlisten problematisch, weil sie einer Abfrage nicht wegen ihrer Bedeutung nahe sind, sondern weil sie vielen Abfragen nahe sind. Dieselbe Erscheinung ist beim Abruf über zwei verschiedene Mengen beschrieben, also wie hier von Abfragen (Zeilen) auf Kandidaten (Kontextwörter) (Dinu, Lazaridou & Baroni 2015).

    ### Fragestellung

    Weisen die Nachbarlisten, wie pipeline_post.py sie bildet (roher Kosinus, Filter wie oben), Hubness auf? Hängt das von der Listenlänge $k_{\mathrm{NN}}$ ab? Geprüft werden die Listenlängen 10 (Länge der Textliste im Nachbarpanel; zugleich der Standardwert der Nachbarschaftsgröße bei CSLS, Conneau et al. 2018), 50, 150 (Länge der Nachbarlisten im Bundle) und 1000 (Länge der langen, ungefilterten Listen in pipeline_post.py; Gonen et al. 2020 verwenden diese Länge für den Vergleich von Nachbarlisten).

    ### Methode

    Die Abfragen und die Kandidaten sind hier verschiedene Mengen: wenige hundert Zeilen, mehrere tausend Kandidaten. Die Schiefe ist deshalb nicht mit Werten aus der Literatur vergleichbar, in der Abfragen und Kandidaten dieselbe Menge sind. Sie wird stattdessen mit einer Referenz verglichen und durch eine zweite Kennzahl ergänzt.

    1. **k-Occurrence und Schiefe.** Für jede Listenlänge $k_{\mathrm{NN}}$ wird $N(c)$ für alle Kandidaten gezählt und die Schiefe berechnet.
    2. **Binomialreferenz.** Wählte jede der $n$ Zeilen ihre $k_{\mathrm{NN}}$ Nachbarn unabhängig und zufällig aus den $m$ Kandidaten, wäre $N(c)$ binomialverteilt mit $n$ Versuchen und der Wahrscheinlichkeit $p = k_{\mathrm{NN}}/m$. Die Schiefe dieser Verteilung ist $(1 - 2p)/\sqrt{n\,p\,(1-p)}$ (Johnson, Kemp & Kotz 2005). Sie gibt an, welche Schiefe allein aus der Größe der beiden Mengen folgt.
    3. **Zielwortbreite.** Die Zeilen sind nicht unabhängig: Dasselbe Zielwort hat in vielen Zeitscheiben ähnliche Nachbarn. Ein Kandidat, der eng mit einem Zielwort verbunden ist, erreicht deshalb ein hohes $N$, ohne ein Hub zu sein. Für jeden Kandidaten wird darum zusätzlich gezählt, in den Listen wie vieler verschiedener Zielwörter er mindestens einmal vorkommt. Ein Kandidat, der in den Listen **aller** Zielwörter vorkommt, heißt im Folgenden **zielwortübergreifend**. Er ist Zeilen mit verschiedenen Zielwörtern nahe und entspricht damit der Beschreibung eines Hubs.
    4. Ausgewertet werden die 20 Kandidaten mit der höchsten k-Occurrence je Listenlänge.

    ### Erwartete Ergebnisse und ihre Deutung

    | Befund | Deutung |
    |---|---|
    | Schiefe nicht über der Binomialreferenz | Die Nachbarn verteilen sich so gleichmäßig, wie es die Mengengrößen zulassen: keine Hubness |
    | Schiefe über der Referenz, die häufigsten Kandidaten gehören jeweils zu einem oder wenigen Zielwörtern | Die Konzentration geht auf die Zielwörter zurück, die über viele Zeitscheiben dieselben Nachbarn haben: keine Hubness im Sinne der Fragestellung |
    | Schiefe über der Referenz, die häufigsten Kandidaten sind überwiegend zielwortübergreifend | Einzelne Kandidaten sind Zeilen unabhängig vom Zielwort nahe: Hubness |

    ### Entscheidungsregel

    Hubness liegt bei einer Listenlänge $k_{\mathrm{NN}}$ vor, wenn (a) die Schiefe von $N$ über der Binomialreferenz liegt und (b) mehr als die Hälfte der 20 Kandidaten mit der höchsten k-Occurrence zielwortübergreifend ist. Die Zahl 20 und die Schwelle „mehr als die Hälfte“ sind Festlegungen dieses Notebooks und nicht der Literatur entnommen. Für die Listenlängen, bei denen Hubness vorliegt, werden in Abschnitt 2 Verfahren zur Hubness-Korrektur geprüft.
    """)
    return


@app.cell
def _(cosine, hub_statistics, is_hubness, pl):
    HUB_K_NN = [10, 50, 150, 1000]

    hub_summary = []
    hub_tops = {}
    for _k_nn in HUB_K_NN:
        _summary, _top, _ = hub_statistics(cosine, _k_nn)
        hub_tops[_k_nn] = _top
        hub_summary.append({"k_NN": _k_nn, **_summary})
    hub_summary = pl.DataFrame(hub_summary)

    # list lengths with hubness under raw cosine; section 2 refers to these
    hub_k_nn = [_r["k_NN"] for _r in hub_summary.iter_rows(named=True) if is_hubness(_r)]
    return HUB_K_NN, hub_k_nn, hub_summary, hub_tops


@app.cell(hide_code=True)
def _(
    HUB_TOP,
    fmt_num,
    fmt_pct,
    hub_k_nn,
    hub_summary,
    hub_tops,
    mo,
    show_table,
    target_words,
):
    _n_words = len(target_words)
    _paragraphs = []
    for _row in hub_summary.iter_rows(named=True):
        _k_nn = _row["k_NN"]
        _above = _row["Schiefe"] > _row["Binomialreferenz"]
        _broad = _row["zielwortübergreifend unter Top 20"]
        _top_words = ", ".join(hub_tops[_k_nn]["context_lemma"].head(5).to_list())
        _paragraphs.append(
            f"**k_NN = {_k_nn}.** Schiefe {fmt_num(_row['Schiefe'], digits=2)}, Binomialreferenz "
            f"{fmt_num(_row['Binomialreferenz'], digits=2)}: "
            + ("über der Referenz. " if _above else "nicht über der Referenz. ")
            + f"Von den {HUB_TOP} Kandidaten mit der höchsten k-Occurrence kommen {_broad} in den Listen "
            f"aller {_n_words} Zielwörter vor. Die häufigsten: {_top_words}. "
            f"{fmt_pct(_row['Anteil N = 0'])} der Kandidaten stehen in keiner Liste. "
            + ("Nach der Entscheidungsregel liegt **Hubness** vor." if _k_nn in hub_k_nn else
               "Nach der Entscheidungsregel liegt keine Hubness vor.")
        )

    if hub_k_nn:
        _kind = "warn"
        _rec = (
            f"**Hubness bei k_NN = {', '.join(str(_x) for _x in hub_k_nn)}.** "
            "Für diese Listenlängen sind in Abschnitt 2 Verfahren zur Hubness-Korrektur zu prüfen. "
            "Die Hubs der Tabellen 1.2 sind außerdem Kandidaten für die Liste der Domänen-Stoppwörter; "
            "ob ein Hub ein Stoppwort ist, ist eine inhaltliche Entscheidung."
        )
    else:
        _kind = "success"
        _rec = "**Keine Hubness bei den geprüften Listenlängen.** Eine Hubness-Korrektur ist nicht erforderlich."

    _tops = []
    for _k_nn, _top in hub_tops.items():
        _tops += [mo.md(f"**Tabelle 1.2 (k_NN = {_k_nn}):** {HUB_TOP} Kandidaten mit der höchsten k-Occurrence"), show_table(_top)]

    mo.vstack(
        [
            mo.md("**Tabelle 1.1:** k-Occurrence je Listenlänge"),
            show_table(hub_summary),
            mo.accordion({"Tabellen 1.2: häufigste Kandidaten je Listenlänge": mo.vstack(_tops)}),
            mo.md("### Auswertung"),
            mo.md("\n\n".join(_paragraphs)),
            mo.callout(mo.md(_rec), kind=_kind),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Hubness-Korrektur: CSLS und Mutual Proximity

    ### Verfahren

    Verfahren zur Hubness-Korrektur ersetzen den rohen Kosinus durch einen Wert, der berücksichtigt, wie nahe die beiden Punkte allen übrigen Punkten sind. Schnitzer et al. (2012) unterscheiden **lokale** Verfahren, die nur die unmittelbare Nachbarschaft eines Punkts einbeziehen, und **globale** Verfahren, die die Ähnlichkeiten zu allen Punkten einbeziehen. Geprüft wird je ein etabliertes Verfahren beider Arten.

    **CSLS** (Cross-domain Similarity Local Scaling; Conneau et al. 2018), ein lokales Verfahren:

    $$
    \mathrm{CSLS}(x, y) = 2\cos(x, y) - r_Z(x) - r_K(y)
    $$

    Dabei ist $r_Z(x)$ der mittlere Kosinus der Zeile $x$ zu ihren $k_{\mathrm{CSLS}}$ nächsten Kandidaten und $r_K(y)$ der mittlere Kosinus des Kandidaten $y$ zu seinen $k_{\mathrm{CSLS}}$ nächsten Zeilen. Verwendet wird der Standardwert $k_{\mathrm{CSLS}} = 10$ der Autoren. Innerhalb einer Zeile ist $r_Z(x)$ konstant; die Rangfolge der Nachbarn hängt also nur vom Abzug $r_K(y)$ ab. Ein Kandidat, der seinen nächsten Zeilen sehr nahe ist, wird abgewertet.

    **Mutual Proximity** (Schnitzer et al. 2012), ein globales Verfahren, hier in der Variante mit Normalverteilung. Für die Zeile $x$ werden Mittelwert $\mu_x$ und Standardabweichung $\sigma_x$ ihrer Kosinuswerte zu allen Kandidaten berechnet, für den Kandidaten $y$ entsprechend $\mu_y$ und $\sigma_y$ seiner Kosinuswerte zu allen Zeilen:

    $$
    \mathrm{MP}(x, y) = \Phi\!\left(\frac{\cos(x, y) - \mu_x}{\sigma_x}\right) \cdot \Phi\!\left(\frac{\cos(x, y) - \mu_y}{\sigma_y}\right)
    $$

    $\Phi$ ist die Verteilungsfunktion der Standardnormalverteilung. MP ist die geschätzte Wahrscheinlichkeit, dass $\cos(x, y)$ sowohl für $x$ als auch für $y$ über einem zufällig gezogenen Kosinuswert liegt; das Produkt setzt die Unabhängigkeit beider Teile voraus. Ein Kandidat, der allen Zeilen ähnlich nahe ist, erreicht im zweiten Faktor nur mittlere Werte und wird dadurch abgewertet.

    Beide Verfahren wurden für Nachbarlisten innerhalb einer Menge (Mutual Proximity) bzw. für den Abruf zwischen zwei Vektorräumen (CSLS) entwickelt. Die Anwendung auf Zeilen und Kontextwörter derselben SVD ist eine Übertragung.

    ### Fragestellung

    Beseitigt eines der Verfahren die Hubness bei den Listenlängen, für die Abschnitt 1 sie festgestellt hat, ohne die kurzen Listen ($k_{\mathrm{NN}} = 10$) wesentlich zu verändern? Die kurzen Listen zeigten in Abschnitt 1 keine Hubness; eine Korrektur, die sie weitgehend ersetzt, griffe in Listen ein, die dafür keinen Anlass bieten.

    ### Methode

    1. **Kennwerte aus Abschnitt 1** für beide Verfahren und alle Listenlängen: Schiefe und Zahl der zielwortübergreifenden Kandidaten unter den 20 häufigsten.
    2. **Gezielte Wirkung:** Anteil der Listenplätze, die auf die Hubs aus Abschnitt 1 entfallen, also auf die zielwortübergreifenden unter den 20 häufigsten Kandidaten bei rohem Kosinus, vor und nach der Korrektur.
    3. **Erhalt der kurzen Listen:** Je Zeile der Anteil der Top 10 bei rohem Kosinus, der auch in den Top 10 nach der Korrektur steht; zusammengefasst als Median über alle Zeilen und als Anteil der Zeilen, bei denen weniger als die Hälfte erhalten bleibt.
    4. **Kontrollgröße Belegung:** Median der Belegung (n_target_slices) über alle Listeneinträge. Hubness-Korrekturen werten Hubs ab und heben dadurch andere Kandidaten an. Sinkt der Median deutlich, rücken seltenere Kontextwörter nach vorn, deren Vektoren auf weniger Daten beruhen.

    Alle Zusammenfassungen über Zeilen oder Listeneinträge verwenden den Median.

    ### Erwartete Ergebnisse und ihre Deutung

    | Befund | Deutung |
    |---|---|
    | weniger zielwortübergreifende Kandidaten und geringerer Anteil der Hubs, kurze Listen weitgehend erhalten | Das Verfahren wirkt gezielt auf die Hubs: geeignet |
    | weniger zielwortübergreifende Kandidaten, kurze Listen weitgehend ersetzt | Das Verfahren wirkt, verändert aber auch unauffällige Listen: Eingriff zu stark |
    | kaum Änderung der Kennwerte | Das Verfahren erfasst die Art von Hubness in diesen Daten nicht: nicht geeignet |

    ### Entscheidungsregel

    Ein Verfahren ist geeignet, wenn bei jeder Listenlänge mit Hubness aus Abschnitt 1 (a) nach der Korrektur höchstens die Hälfte der 20 häufigsten Kandidaten zielwortübergreifend ist, das Kriterium (b) aus Abschnitt 1 also nicht mehr erfüllt ist, und (b) die Top 10 je Zeile im Median zu mindestens der Hälfte erhalten bleiben. Sind beide Verfahren geeignet, wird das mit dem größeren Erhalt der Top 10 empfohlen, weil es weniger eingreift. Die Schwelle „mindestens die Hälfte“ ist eine Festlegung dieses Notebooks. Die Kontrollgröße Belegung geht nicht in die Regel ein, weil es für sie keine begründete Schwelle gibt; sie wird in der Auswertung genannt.
    """)
    return


@app.cell
def _(cosine, normal_cdf, np):
    K_CSLS = 10

    # CSLS: mean cosine of each row to its K_CSLS nearest candidates (r_Z)
    # and of each candidate to its K_CSLS nearest rows (r_K)
    _r_rows = -np.sort(-cosine, axis=1)[:, :K_CSLS].mean(axis=1)
    _r_cand = -np.sort(-cosine, axis=0)[:K_CSLS, :].mean(axis=0)
    score_csls = 2 * cosine - _r_rows[:, None] - _r_cand[None, :]

    # Mutual Proximity, Gaussian variant: z-scores of each cosine within its row
    # (over all candidates) and within its column (over all rows)
    _z_rows = (cosine - cosine.mean(axis=1, keepdims=True)) / cosine.std(axis=1, keepdims=True)
    _z_cand = (cosine - cosine.mean(axis=0, keepdims=True)) / cosine.std(axis=0, keepdims=True)
    score_mp = normal_cdf(_z_rows) * normal_cdf(_z_cand)
    return K_CSLS, score_csls, score_mp


@app.cell
def _(
    HUB_K_NN,
    candidates,
    cosine,
    hub_statistics,
    hub_tops,
    nn_membership,
    np,
    pl,
    score_csls,
    score_mp,
    target_words,
):
    SHORT_K_NN = 10

    correction_methods = {"roher Kosinus": cosine, "CSLS": score_csls, "Mutual Proximity": score_mp}

    _lemma = candidates["context_lemma"].to_numpy()
    _n_rows = candidates["n_target_slices"].to_numpy()
    _raw_short = nn_membership(cosine, SHORT_K_NN)

    # hubs of section 1: candidates occurring in the lists of all target words,
    # among the HUB_TOP most frequent ones under raw cosine, per list length
    _hub_mask = {
        _k_nn: np.isin(
            _lemma,
            hub_tops[_k_nn].filter(pl.col("Zielwörter") == len(target_words))["context_lemma"].to_list(),
        )
        for _k_nn in HUB_K_NN
    }

    correction_summary = []
    correction_short = []
    correction_tops = {}
    for _name, _score in correction_methods.items():
        for _k_nn in HUB_K_NN:
            _summary, _top, _member = hub_statistics(_score, _k_nn)
            correction_tops[(_name, _k_nn)] = _top
            correction_summary.append(
                {
                    "Verfahren": _name,
                    "k_NN": _k_nn,
                    "Schiefe": _summary["Schiefe"],
                    "zielwortübergreifend unter Top 20": _summary["zielwortübergreifend unter Top 20"],
                    "Anteil Listenplätze der Hubs aus Abschnitt 1": round(
                        float(_member[:, _hub_mask[_k_nn]].sum() / _member.sum()), 3
                    ),
                    "Median n_target_slices der Listeneinträge": float(np.median(_n_rows[np.nonzero(_member)[1]])),
                }
            )
        _kept = (nn_membership(_score, SHORT_K_NN) & _raw_short).sum(axis=1) / SHORT_K_NN
        correction_short.append(
            {
                "Verfahren": _name,
                "Median erhaltener Anteil der Top 10": float(np.median(_kept)),
                "Anteil Zeilen mit weniger als der Hälfte erhalten": round(float((_kept < 0.5).mean()), 3),
            }
        )
    correction_summary = pl.DataFrame(correction_summary)
    correction_short = pl.DataFrame(correction_short)
    return (
        SHORT_K_NN,
        correction_methods,
        correction_short,
        correction_summary,
        correction_tops,
    )


@app.cell(hide_code=True)
def _(
    HUB_TOP,
    K_CSLS,
    correction_short,
    correction_summary,
    correction_tops,
    fmt_num,
    fmt_pct,
    hub_k_nn,
    mo,
    pl,
    show_table,
):
    _paragraphs = []
    _suitable = {}
    _passed_k = {}
    _raw = correction_summary.filter(pl.col("Verfahren") == "roher Kosinus")
    for _name in ["CSLS", "Mutual Proximity"]:
        _rows = correction_summary.filter(pl.col("Verfahren") == _name)
        _short = correction_short.filter(pl.col("Verfahren") == _name).row(0, named=True)
        _kept = _short["Median erhaltener Anteil der Top 10"]

        _parts = []
        _crit_a = True
        _passed_k[_name] = []
        for _k_nn in hub_k_nn:
            _r = _rows.filter(pl.col("k_NN") == _k_nn).row(0, named=True)
            _r0 = _raw.filter(pl.col("k_NN") == _k_nn).row(0, named=True)
            _ok = _r["zielwortübergreifend unter Top 20"] <= HUB_TOP / 2
            _crit_a = _crit_a and _ok
            if _ok:
                _passed_k[_name].append(_k_nn)
            _parts.append(
                f"bei k_NN = {_k_nn} sind {_r['zielwortübergreifend unter Top 20']} der {HUB_TOP} häufigsten "
                f"Kandidaten zielwortübergreifend (roher Kosinus: {_r0['zielwortübergreifend unter Top 20']}), "
                f"auf die Hubs aus Abschnitt 1 entfallen {fmt_pct(_r['Anteil Listenplätze der Hubs aus Abschnitt 1'])} "
                f"der Listenplätze (roher Kosinus: {fmt_pct(_r0['Anteil Listenplätze der Hubs aus Abschnitt 1'])}), "
                f"Median der Belegung {fmt_num(_r['Median n_target_slices der Listeneinträge'], digits=0)} "
                f"(roher Kosinus: {fmt_num(_r0['Median n_target_slices der Listeneinträge'], digits=0)})"
            )
        _crit_b = _kept >= 0.5
        _suitable[_name] = _crit_a and _crit_b
        _paragraphs.append(
            f"**{_name}.** " + ("; ".join(_parts) + ". " if _parts else "")
            + f"Von den Top 10 je Zeile bleiben im Median {fmt_pct(_kept)} erhalten; bei "
            f"{fmt_pct(_short['Anteil Zeilen mit weniger als der Hälfte erhalten'])} der Zeilen weniger als die Hälfte. "
            + f"Kriterium (a) {'erfüllt' if _crit_a else 'nicht erfüllt'}, Kriterium (b) {'erfüllt' if _crit_b else 'nicht erfüllt'}: "
            + ("**geeignet**." if _suitable[_name] else "nicht geeignet.")
        )

    if not hub_k_nn:
        _kind, _rec = "success", "**Abschnitt 1 hat keine Hubness festgestellt.** Eine Korrektur ist nicht erforderlich."
    elif any(_suitable.values()):
        _cands = [_n for _n, _s in _suitable.items() if _s]
        _best = max(_cands, key=lambda _n: correction_short.filter(pl.col("Verfahren") == _n)["Median erhaltener Anteil der Top 10"][0])
        _kind, _rec = "success", (
            f"**Empfehlung: {_best}** für die Nachbarlisten. "
            + (f"Auch {', '.join(_n for _n in _cands if _n != _best)} ist geeignet, verändert die Top 10 aber stärker. " if len(_cands) > 1 else "")
            + "Ob die korrigierten Listen inhaltlich besser sind, misst dieser Abschnitt nicht; "
            "dafür sind die Beispiele unten durchzusehen."
        )
    else:
        _kind, _rec = "danger", (
            "**Keines der beiden Verfahren erfüllt die Entscheidungsregel.** "
            "Die Hubness bleibt bestehen oder die Korrektur verändert die kurzen Listen zu stark. "
            + "".join(
                f"{_n} erfüllt Kriterium (a) nur bei k_NN = {', '.join(str(_x) for _x in _ks)}. "
                for _n, _ks in _passed_k.items() if _ks
            )
            + "In Frage kommen Domänen-Stoppwörter für die Hubs, die eine inhaltliche Durchsicht als solche bestätigt, "
            "oder kürzere Listen."
        )

    _tops = []
    for (_name, _k_nn), _top in correction_tops.items():
        if _name != "roher Kosinus" and _k_nn in hub_k_nn:
            _tops += [mo.md(f"**Tabelle 2.3 ({_name}, k_NN = {_k_nn}):** {HUB_TOP} Kandidaten mit der höchsten k-Occurrence"), show_table(_top)]

    mo.vstack(
        [
            mo.md(f"**Tabelle 2.1:** Kennwerte je Verfahren und Listenlänge (k_CSLS = {K_CSLS})"),
            show_table(correction_summary),
            mo.md("**Tabelle 2.2:** Erhalt der Top 10 je Zeile gegenüber rohem Kosinus"),
            show_table(correction_short),
            mo.accordion({"Tabellen 2.3: häufigste Kandidaten nach der Korrektur": mo.vstack(_tops) if _tops else mo.md("keine")}),
            mo.md("### Auswertung"),
            mo.md("\n\n".join(_paragraphs)),
            mo.callout(mo.md(_rec), kind=_kind),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Beispiele zur inhaltlichen Durchsicht

    Die Kennwerte zeigen, ob ein Verfahren die Verteilung der Nachbarn verändert, nicht, ob die Nachbarn danach inhaltlich zutreffender sind. Dafür lassen sich hier die Listen einzelner Zeilen nebeneinander ansehen. Die Auswahl geht nicht in die Auswertung ein.
    """)
    return


@app.cell
def _(mo, target_slices):
    _labels = [
        f"{_w} | {_s}"
        for _w, _s in zip(target_slices["target_lemma"].to_list(), target_slices["slice"].to_list())
    ]
    example_row = mo.ui.dropdown(
        options={_l: _i for _i, _l in enumerate(_labels)},
        value=_labels[-1],
        label="Zeile",
        searchable=True,
    )
    example_row
    return (example_row,)


@app.cell
def _(candidates, correction_methods, example_row, np, pl, show_table):
    _i = example_row.value
    _lemma = candidates["context_lemma"].to_numpy()
    _n_show = 20
    show_table(
        pl.DataFrame(
            {"Rang": list(range(1, _n_show + 1))}
            | {
                _name: _lemma[np.argsort(-_score[_i])[:_n_show]].tolist()
                for _name, _score in correction_methods.items()
            }
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Einschränkungen

    - Abschnitt 1 prüft die Nachbarlisten mit den Filtern von pipeline_post.py. Andere Filter, etwa Domänen-Stoppwörter, ändern die Kandidatenmenge und damit die k-Occurrence.
    - Die Zielwörter stammen aus einem gemeinsamen Wortfeld. Ein zielwortübergreifender Kandidat kann deshalb auch eine Bedeutung haben, die allen Zielwörtern gemeinsam ist. Die Zielwortbreite trennt Hubs von solchen Kandidaten nicht; dafür ist eine inhaltliche Durchsicht der Tabellen 1.2 nötig.
    - Die Binomialreferenz nimmt an, dass die Zeilen unabhängig voneinander sind. Das trifft nicht zu, weil dasselbe Zielwort in vielen Zeitscheiben vorkommt. Eine Schiefe über der Referenz ist deshalb allein kein Nachweis von Hubness; die Entscheidungsregel verlangt zusätzlich die Zielwortbreite.
    - Die Zielwortbreite zählt einen Kandidaten schon dann für ein Zielwort, wenn er in einer einzigen Zeitscheibe in dessen Liste vorkommt. Bei großen Listenlängen wird sie dadurch großzügiger.
    - CSLS und Mutual Proximity sind für andere Anordnungen entwickelt worden (s. Abschnitt 2). Ihre Anwendung auf Zeilen und Kontextwörter derselben SVD ist eine Übertragung, die in den Originalarbeiten nicht evaluiert ist.
    - Bei CSLS gehören die $k_{\mathrm{CSLS}}$ nächsten Zeilen eines Kandidaten meist zu einem einzigen Zielwort, weil dasselbe Zielwort in vielen Zeitscheiben vorkommt. $r_K(y)$ misst dann die Nähe zu diesem Zielwort, nicht die Nähe zu Zeilen allgemein. $k_{\mathrm{CSLS}}$ ist nicht an diese Daten angepasst; verwendet wird der Standardwert der Autoren.
    - Die Variante von Mutual Proximity mit Normalverteilung setzt voraus, dass die Kosinuswerte je Zeile und je Kandidat näherungsweise normalverteilt und die beiden Faktoren unabhängig sind. Beides ist nicht geprüft.
    - Der Erhalt der Top 10 misst, wie stark ein Verfahren die kurzen Listen verändert, nicht, ob es sie verbessert oder verschlechtert. Für die Qualität ist eine inhaltliche Durchsicht nötig.

    ## Literatur

    - Conneau, A., Lample, G., Ranzato, M., Denoyer, L. & Jégou, H. (2018): Word translation without parallel data. *International Conference on Learning Representations (ICLR)*. arXiv:1710.04087.
    - Dinu, G., Lazaridou, A. & Baroni, M. (2015): Improving zero-shot learning by mitigating the hubness problem. *ICLR Workshop Track*. arXiv:1412.6568.
    - Gonen, H., Jawahar, G., Seddah, D. & Goldberg, Y. (2020): Simple, interpretable and stable method for detecting words with usage change across corpora. *Proceedings of ACL 2020*, 538–555.
    - Johnson, N. L., Kemp, A. W. & Kotz, S. (2005): *Univariate Discrete Distributions*. 3. Aufl. Wiley.
    - Radovanović, M., Nanopoulos, A. & Ivanović, M. (2010): Hubs in space: Popular nearest neighbors in high-dimensional data. *Journal of Machine Learning Research* 11, 2487–2531.
    - Schnitzer, D., Flexer, A., Schedl, M. & Widmer, G. (2012): Local and global scaling reduce hubs in space. *Journal of Machine Learning Research* 13, 2871–2902.

    Die bibliografischen Angaben sind vor einer Veröffentlichung an den Originalen zu prüfen.
    """)
    return


if __name__ == "__main__":
    app.run()
