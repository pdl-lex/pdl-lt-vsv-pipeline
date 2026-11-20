import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    import marimo as mo
    import numpy as np
    import polars as pl
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds

    return mo, np, pl, sp, svds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # AUC in der Split-half-Prüfung: a_zeit und a_wort

    Dieses Notebook veranschaulicht die Kennwerte a_zeit und a_wort aus `pipeline_emb_ppmi_diagnose.py` an den echten Daten: eine Teilung der Reden in die Hälften A und B, SVD an Hälfte A (k und p wie in emb), beide Hälften per *folding-in* auf dieselben Achsen projiziert.

    **Das Zuordnungsspiel.** Gegeben ist Hälfte A einer Zeile, z. B. *Sicherheit, 2022-H1*. Gesucht ist unter den B-Hälften aller Halbjahre von *Sicherheit* die eigene. Die eigene B-Hälfte ist der **positive Fall**, die B-Hälften der anderen Halbjahre sind die **negativen Fälle**. Als Maß dient die Kosinusähnlichkeit zu Hälfte A.

    - **ROC-Kurve:** Eine Kosinus-Schwelle wird von oben nach unten verschoben. Alles, was über der Schwelle liegt, gilt als „die eigene Hälfte“. Für jede Schwelle wird abgetragen, welcher Anteil der positiven Fälle darüber liegt (Trefferquote) und welcher Anteil der negativen (Fehlalarmquote).
    - **AUC** (Fläche unter der ROC-Kurve): die Wahrscheinlichkeit, dass die eigene Hälfte einen höheren Kosinus hat als ein zufällig gewähltes anderes Halbjahr. Mit nur einem positiven Fall ist das genau der Anteil der anderen Halbjahre, die unter dem eigenen Kosinus liegen: **a_zeit**. Gleiche Werte zählen zur Hälfte.
    - **a_wort** ist dasselbe Spiel mit anderen Gegnern: die anderen Zielwörter desselben Halbjahrs.

    Die Rechnung dauert beim ersten Start etwa eine halbe Minute (Einlesen, PPMI, SVD).
    """)
    return


@app.cell
def _(mo, np, pl, sp, svds):
    # same inputs and the same computation as pipeline_emb_ppmi_diagnose.py (split 0, SVD fitted on half A)
    import json

    PROJECT, KWIC_VARIANT, SLICE_GRID = "bundestag", "punkt", "halfyear"
    SEED = 42
    _run = f"{PROJECT}_{KWIC_VARIANT}_{SLICE_GRID}"
    _pre = mo.notebook_dir() / "input" / KWIC_VARIANT
    _emb = mo.notebook_dir() / "output" / KWIC_VARIANT

    _meta = json.loads((_emb / f"emb_meta_{_run}.json").read_text(encoding="utf-8"))
    SVD_K, SVD_P = int(_meta["k"]), float(_meta["eig_p"])

    _hits = (
        pl.read_parquet(_pre / f"kwic_hits_sliced_{_run}.parquet")
        .select(["id", "slice", "target_lemma", "context_lemmas"])
        .with_row_index("h")
    )
    _tokens = (
        _hits.select(["h", "context_lemmas"])
        .explode("context_lemmas", empty_as_null=False)
        .rename({"context_lemmas": "context_lemma"})
        .drop_nulls("context_lemma")
    )
    _keys = ["target_lemma", "slice"]
    rows = _hits.group_by(_keys).len().rename({"len": "n_hits"}).sort(_keys).with_row_index("i")
    _columns = _tokens.select("context_lemma").unique().sort("context_lemma").with_row_index("j")
    _n_rows = rows.height
    _hit_row = _hits.join(rows.select(["i", *_keys]), on=_keys, how="left").sort("h")["i"].to_numpy()

    row_word = rows["target_lemma"].to_numpy()
    slice_labels = sorted(rows["slice"].unique().to_list())
    row_slice = rows["slice"].replace_strict(slice_labels, list(range(len(slice_labels)))).to_numpy()

    _hc = _tokens.group_by(["h", "context_lemma"]).len().join(_columns, on="context_lemma")
    _hit_counts = sp.csr_matrix(
        (_hc["len"].to_numpy().astype(np.float64), (_hc["h"].to_numpy(), _hc["j"].to_numpy())),
        shape=(_hits.height, _columns.height),
    )

    # split 0 of the diagnosis: half of the speeches of each slice go to half A
    _rng = np.random.default_rng(SEED)
    _speeches = _hits.select(["id", "slice"]).unique().sort("id")
    _split = _speeches.with_columns(pl.Series("key", _rng.random(_speeches.height))).with_columns(
        (pl.col("key").rank("ordinal").over("slice") <= pl.len().over("slice") / 2).alias("in_a")
    )
    _in_a = _hits.select(["h", "id"]).join(_split.select(["id", "in_a"]), on="id", how="left").sort("h")["in_a"].to_numpy()

    def _counts(sel):
        S = sp.csr_matrix((np.ones(len(sel)), (_hit_row[sel], sel)), shape=(_n_rows, _hits.height))
        return (S @ _hit_counts).tocsr()

    def _ppmi(C):
        # PPMI as in emb: log2(observed / expected) with the marginals of the slice, clipped at 0
        S = sp.csr_matrix((np.ones(_n_rows), (row_slice, np.arange(_n_rows))), shape=(len(slice_labels), _n_rows))
        Z = np.asarray((S @ C).todense())
        T = Z.sum(axis=1)
        row_total = np.asarray(C.sum(axis=1)).ravel()
        coo = C.tocoo()
        s = row_slice[coo.row]
        with np.errstate(divide="ignore"):
            pmi = np.log2(coo.data * T[s] / (row_total[coo.row] * Z[s, coo.col]))
        M = sp.csr_matrix((np.maximum(pmi, 0), (coo.row, coo.col)), shape=C.shape)
        M.eliminate_zeros()
        return M

    def _unit(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(norms > 0, X / norms, np.nan)

    _sel = [np.flatnonzero(_in_a), np.flatnonzero(~_in_a)]
    _C = [_counts(s) for s in _sel]
    _M = [_ppmi(c) for c in _C]
    n_half = [np.bincount(_hit_row[s], minlength=_n_rows) for s in _sel]

    # SVD on half A, both halves folded in: x = m V_k S_k^(p-1)
    _, _s, _vt = svds(_M[0], k=SVD_K, rng=np.random.default_rng(SEED))
    _order = np.argsort(_s)[::-1]
    _V, _s = _vt[_order].T, _s[_order]
    XA = _unit(np.asarray(_M[0] @ _V) * _s ** (SVD_P - 1))
    XB = _unit(np.asarray(_M[1] @ _V) * _s ** (SVD_P - 1))

    row_index = {(w, sl): i for i, w, sl in rows.select(["i", "target_lemma", "slice"]).iter_rows()}
    mo.md(
        f"Teilung 0 (SEED {SEED}), SVD an Hälfte A mit k = {SVD_K}, p = {SVD_P:g}. "
        f"{rows['target_lemma'].n_unique()} Zielwörter × {len(slice_labels)} Halbjahre, {_n_rows:,} Zeilen.".replace(",", ".")
    )
    return XA, XB, n_half, row_index, row_slice, row_word, rows, slice_labels


@app.cell
def _(mo, np):
    # colours: own half = categorical slot 1, false alarms = slot 2, the rest recessive
    C_OWN, C_ALARM, C_REST = "#2a78d6", "#eb6834", "#8a8984"

    def fmt(x, d=3):
        return "–" if x is None or np.isnan(x) else f"{x:.{d}f}".replace(".", ",")

    def svg_chart(marks, xlim, ylim, xticks, yticks, xlabel, ylabel, width=560, height=340, xfmt=None, yfmt=None):
        """Minimal SVG chart. marks: list of functions (sx, sy) -> SVG string. Text uses currentColor
        so the chart follows marimo's light and dark theme."""
        ml, mr, mt, mb = 52, 24, 12, 44
        pw, ph = width - ml - mr, height - mt - mb

        def sx(x):
            return ml + (x - xlim[0]) / (xlim[1] - xlim[0]) * pw

        def sy(y):
            return mt + ph - (y - ylim[0]) / (ylim[1] - ylim[0]) * ph

        xfmt = xfmt or (lambda v: fmt(v, 1))
        yfmt = yfmt or (lambda v: fmt(v, 1))
        out = [
            f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;font:11px system-ui,sans-serif;color:inherit">'
        ]
        for t in yticks:
            out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="currentColor" stroke-opacity="0.12"/>')
            out.append(f'<text x="{ml - 6}" y="{sy(t) + 4:.1f}" text-anchor="end" fill="currentColor" fill-opacity="0.7">{yfmt(t)}</text>')
        for t in xticks:
            lab = t[1] if isinstance(t, tuple) else xfmt(t)
            v = t[0] if isinstance(t, tuple) else t
            out.append(f'<text x="{sx(v):.1f}" y="{mt + ph + 16}" text-anchor="middle" fill="currentColor" fill-opacity="0.7">{lab}</text>')
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}" stroke="currentColor" stroke-opacity="0.4"/>')
        out.append(f'<text x="{ml + pw / 2}" y="{height - 6}" text-anchor="middle" fill="currentColor">{xlabel}</text>')
        out.append(
            f'<text transform="translate(13,{mt + ph / 2}) rotate(-90)" text-anchor="middle" fill="currentColor">{ylabel}</text>'
        )
        out += [m(sx, sy) for m in marks]
        out.append("</svg>")
        return mo.Html("".join(out))

    def auc_from_scores(own, neg):
        """Share of negatives below the positive, ties count half."""
        neg = neg[~np.isnan(neg)]
        if np.isnan(own) or len(neg) == 0:
            return np.nan
        return ((neg < own).sum() + 0.5 * (neg == own).sum()) / len(neg)

    return C_ALARM, C_OWN, C_REST, auc_from_scores, fmt, svg_chart


@app.cell
def _(mo, rows):
    _words = sorted(rows["target_lemma"].unique().to_list())
    word_ui = mo.ui.dropdown(_words, value="Sicherheit" if "Sicherheit" in _words else _words[0], label="Zielwort", searchable=True)
    game_ui = mo.ui.radio(
        {"a_zeit: andere Halbjahre desselben Worts": "a_zeit", "a_wort: andere Wörter desselben Halbjahrs": "a_wort"},
        value="a_zeit: andere Halbjahre desselben Worts",
        label="Gegner im Zuordnungsspiel",
    )
    return game_ui, word_ui


@app.cell
def _(mo, rows, word_ui):
    _slices = rows.filter(rows["target_lemma"] == word_ui.value)["slice"].sort().to_list()
    slice_ui = mo.ui.dropdown(_slices, value="2022-H1" if "2022-H1" in _slices else _slices[-1], label="Halbjahr")
    return (slice_ui,)


@app.cell
def _(
    XA,
    XB,
    auc_from_scores,
    game_ui,
    n_half,
    np,
    row_index,
    row_slice,
    row_word,
    slice_labels,
    slice_ui,
    word_ui,
):
    q = row_index[(word_ui.value, slice_ui.value)]
    if game_ui.value == "a_zeit":
        cand = np.flatnonzero(row_word == row_word[q])  # all slices of the word, incl. the own one
        cand_label = [slice_labels[row_slice[c]] for c in cand]
    else:
        cand = np.flatnonzero(row_slice == row_slice[q])  # all words of the slice, incl. the own one
        cand_label = [str(row_word[c]) for c in cand]
    cos = XB[cand] @ XA[q]  # cosine of half A of the query to half B of every candidate
    is_own = cand == q
    own_cos = float(cos[is_own][0])
    neg_cos = cos[~is_own]
    neg_valid = neg_cos[~np.isnan(neg_cos)]
    auc = auc_from_scores(own_cos, neg_cos)
    q_hits = (int(n_half[0][q]), int(n_half[1][q]))
    return auc, cand, cand_label, cos, is_own, neg_valid, own_cos, q_hits


@app.cell
def _(mo, np, own_cos, cos):
    _lo = np.floor(np.nanmin(cos) * 50) / 50
    _hi = np.ceil(np.nanmax(cos) * 50) / 50 + 0.02
    _step = 0.002
    _val = _lo + np.floor((own_cos - _lo) / _step) * _step if not np.isnan(own_cos) else _lo
    threshold_ui = mo.ui.slider(
        start=float(_lo), stop=float(_hi), step=_step, value=float(round(_val, 3)),
        label="Kosinus-Schwelle", show_value=True, full_width=True,
    )
    return (threshold_ui,)


@app.cell
def _(auc, fmt, game_ui, mo, neg_valid, own_cos, q_hits, slice_ui, word_ui):
    _n_neg = len(neg_valid)
    _below = int((neg_valid < own_cos).sum())
    _ties = int((neg_valid == own_cos).sum())
    _who = "anderen Halbjahren" if game_ui.value == "a_zeit" else "anderen Zielwörtern"
    _tie_txt = f" und {_ties} gleich auf (zählen halb)" if _ties else ""
    mo.vstack([
        mo.hstack([word_ui, slice_ui], justify="start"),
        game_ui,
        mo.md(
            f"**Rechenbeispiel.** Hälfte A von *{word_ui.value}, {slice_ui.value}* ({q_hits[0]} Belege) hat zur eigenen "
            f"Hälfte B ({q_hits[1]} Belege) den Kosinus **{fmt(own_cos)}**. Von {_n_neg} {_who} liegen "
            f"**{_below}** darunter{_tie_txt}. Also **{game_ui.value} = "
            f"{_below}{' + ' + str(_ties) + '/2' if _ties else ''} / {_n_neg} = {fmt(auc)}**."
        ),
    ])
    return


@app.cell
def _(
    C_ALARM,
    C_OWN,
    C_REST,
    auc,
    cand,
    cand_label,
    cos,
    fmt,
    game_ui,
    is_own,
    mo,
    np,
    own_cos,
    slice_labels,
    svg_chart,
    threshold_ui,
):
    t = threshold_ui.value
    _ok = ~np.isnan(cos)
    _ylo = np.floor(np.nanmin(cos) * 10) / 10
    _yhi = min(1.0, np.ceil(np.nanmax(cos) * 10) / 10 + 0.05)
    _yt = np.round(np.arange(_ylo, _yhi + 1e-9, 0.1), 2)

    # left: candidates, x = time (a_zeit) or rank (a_wort)
    if game_ui.value == "a_zeit":
        _x = np.array([slice_labels.index(lab) for lab in cand_label], dtype=float)
        _xlim = (-0.8, len(slice_labels) - 0.2)
        _xt = [(i, slice_labels[i][:4]) for i in range(0, len(slice_labels), 4)]
        _xl = "Halbjahr der B-Hälfte"
    else:
        _rank = np.empty(len(cand))
        _rank[np.argsort(-np.nan_to_num(cos, nan=-9))] = np.arange(1, len(cand) + 1)
        _x = _rank
        _xlim = (0, len(cand) + 1)
        _xt = [1] + list(range(100, len(cand) + 1, 100))
        _xl = "Rang nach Kosinus"

    def _dots(sx, sy):
        out = []
        order = np.argsort(is_own)  # own one drawn last
        for i in order:
            if not _ok[i]:
                continue
            col = C_OWN if is_own[i] else (C_ALARM if cos[i] >= t else C_REST)
            r = 6 if is_own[i] else (4 if game_ui.value == "a_zeit" else 2.5)
            title = f"{cand_label[i]}: {fmt(cos[i])}" + (" (eigene Hälfte)" if is_own[i] else "")
            out.append(
                f'<circle cx="{sx(_x[i]):.1f}" cy="{sy(cos[i]):.1f}" r="{r}" fill="{col}" '
                f'stroke="var(--background, #fff)" stroke-width="1"><title>{title}</title></circle>'
            )
        out.append(
            f'<line x1="{sx(_xlim[0]):.1f}" x2="{sx(_xlim[1]):.1f}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" '
            f'stroke="currentColor" stroke-dasharray="5 4" stroke-width="1.5"/>'
            f'<text x="{sx(_xlim[0]) + 4:.1f}" y="{sy(t) - 5:.1f}" text-anchor="start" fill="currentColor">Schwelle {fmt(t)}</text>'
        )
        return "".join(out)

    _left = svg_chart([_dots], _xlim, (_ylo, _yhi), _xt, _yt, _xl, "Kosinus zu Hälfte A", xfmt=lambda v: str(int(v)))

    # right: ROC curve. With one positive it is a step: flat at 0 until the threshold passes the own cosine
    _neg = cos[~is_own & _ok]
    _scores = np.unique(np.concatenate([_neg, [own_cos]]))[::-1]
    _pts = [(0.0, 0.0)] + [(float((_neg >= s).mean()), float(own_cos >= s)) for s in _scores]
    _fpr_t, _tpr_t = float((_neg >= t).mean()), float(own_cos >= t)
    # trail: the part of the curve the threshold has already passed (scores >= t), ending at the current point
    _n_done = 1 + int((_scores >= t).sum())
    # the jump: from the last point before the own cosine to the point that includes it
    _k_own = 1 + int(np.flatnonzero(_scores == own_cos)[0])
    _jump_x = (_pts[_k_own - 1][0] + _pts[_k_own][0]) / 2

    def _roc(sx, sy):
        def path(p):
            return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in p)

        out = [
            f'<polygon points="{path(_pts + [(1, 0)])}" fill="{C_OWN}" fill-opacity="0.12"/>',
            f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="currentColor" stroke-opacity="0.35" stroke-dasharray="3 3"/>',
            f'<text x="{sx(0.55):.1f}" y="{sy(0.5) + 16:.1f}" fill="currentColor" fill-opacity="0.6" transform="rotate(-38 {sx(0.55):.1f} {sy(0.5) + 16:.1f})">Zufall (AUC 0,5)</text>',
            # whole curve, faint
            f'<polyline points="{path(_pts)}" fill="none" stroke="{C_OWN}" stroke-opacity="0.35" stroke-width="2.5" stroke-linejoin="round"/>',
            # trail, solid
            f'<polyline points="{path(_pts[:_n_done])}" fill="none" stroke="{C_OWN}" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>',
        ]
        # one vertex per threshold step (one per candidate)
        for n, (x, y) in enumerate(_pts[1:], start=1):
            done = n < _n_done
            out.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{C_OWN if done else "var(--background, #fff)"}" '
                f'stroke="{C_OWN}" stroke-width="1.5"><title>Schwelle {fmt(_scores[n - 1])}: '
                f"Fehlalarmquote {fmt(x, 2)}, Trefferquote {y:.0f}</title></circle>"
            )
        _anchor = "start" if _jump_x < 0.6 else "end"
        _dx = 8 if _anchor == "start" else -8
        out.append(
            f'<text x="{sx(_jump_x) + _dx:.1f}" y="{sy(0.55):.1f}" text-anchor="{_anchor}" fill="currentColor">'
            f"Sprung: eigene Hälfte</text>"
            f'<text x="{sx(_jump_x) + _dx:.1f}" y="{sy(0.55) + 14:.1f}" text-anchor="{_anchor}" fill="currentColor">'
            f"(Kosinus {fmt(own_cos)})</text>"
        )
        # current threshold: ring in text ink, like the dashed threshold line on the left
        out.append(
            f'<circle cx="{sx(_fpr_t):.1f}" cy="{sy(_tpr_t):.1f}" r="8" fill="none" stroke="currentColor" stroke-width="2.5">'
            f"<title>Schwelle {fmt(t)}: Fehlalarmquote {fmt(_fpr_t, 2)}, Trefferquote {_tpr_t:.0f}</title></circle>"
        )
        out.append(
            f'<text x="{sx(0.97):.1f}" y="{sy(0.25):.1f}" text-anchor="end" fill="currentColor" font-size="14">'
            f"Fläche = {fmt(auc)}</text>"
        )
        return "".join(out)

    _right = svg_chart(
        [_roc], (-0.05, 1.05), (-0.06, 1.06), [0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1],
        "Fehlalarmquote (negative Fälle über der Schwelle)", "Trefferquote", width=400,
        xfmt=lambda v: fmt(v, 2), yfmt=lambda v: fmt(v, 2),
    )

    _n_alarm = int((_neg >= t).sum())
    mo.vstack([
        threshold_ui,
        mo.hstack([_left, _right], widths=[1.4, 1], align="center"),
        mo.md(
            f"<span style='color:{C_OWN}'>●</span> eigene B-Hälfte (positiver Fall) · "
            f"<span style='color:{C_ALARM}'>●</span> negative Fälle über der Schwelle (Fehlalarme) · "
            f"<span style='color:{C_REST}'>●</span> negative Fälle darunter. "
            f"Bei Schwelle {fmt(t)}: Trefferquote {_tpr_t:.0f} von 1, Fehlalarmquote {_n_alarm} von {len(_neg)} "
            f"= {fmt(_fpr_t, 2)}. In der ROC-Kurve rechts markiert der Ring die aktuelle Schwelle (wie die gestrichelte "
            "Linie links), die kräftige Linie ist der Weg, den die Schwelle von oben kommend schon zurückgelegt hat. "
            "Jeder Punkt der Kurve ist ein Kandidat, an dem die Schwelle vorbeigeht. Schiebe die Schwelle nach unten: "
            "Solange sie über dem eigenen Kosinus liegt, steigen nur die Fehlalarme, und die Kurve läuft nach rechts am Boden. "
            "Sobald sie ihn unterschreitet, springt die Trefferquote auf 1 und die Kurve läuft oben weiter. Je früher "
            "der Sprung, desto größer die Fläche. Die Fläche unter der Kurve ist der Anteil der negativen Fälle, die "
            "*nicht* vor dem positiven kommen – also genau der Wert oben."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Warum ein Anteil statt des Kosinus?

    Die Kosinuswerte desselben Worts liegen dicht beieinander (bei *Sicherheit* fast alle Halbjahre zwischen 0,3 und 0,5), und ihr Niveau hängt von der Belegzahl ab: Viele Belege ergeben in beiden Hälften ein stabiles Profil und damit einen hohen Kosinus – aber auch die anderen Halbjahre desselben Worts liegen dann hoch. Ein fester Kosinus-Grenzwert würde deshalb bei jedem Wort und jeder Belegzahl etwas anderes bedeuten.

    Die AUC zählt nur die **Reihenfolge**: Liegt die eigene Hälfte vor den Gegnern? Damit ist sie für jede Zeile auf dieselbe Skala von 0 bis 1 gebracht (0,5 = Zufall), und eine einzige Schwelle von 0,7 lässt sich auf alle Zeilen anwenden.

    Die beiden Grafiken zeigen alle Zeilen dieser Teilung (gewähltes Wort hervorgehoben). Links: Der eigene Kosinus steigt mit der Belegzahl. Rechts: Derselbe eigene Kosinus kann zu sehr verschiedenen a_zeit-Werten führen – entscheidend ist, wie nah die anderen Halbjahre liegen.
    """)
    return


@app.cell
def _(XA, XB, auc_from_scores, n_half, np, row_word):
    # a_zeit (one-sided, half A against the B halves) for every row
    all_own = np.full(len(row_word), np.nan)
    all_azeit = np.full(len(row_word), np.nan)
    for _w in np.unique(row_word):
        _idx = np.flatnonzero(row_word == _w)
        _S = XA[_idx] @ XB[_idx].T
        for _a, _i in enumerate(_idx):
            all_own[_i] = _S[_a, _a]
            all_azeit[_i] = auc_from_scores(_S[_a, _a], np.delete(_S[_a], _a))
    all_hits = np.minimum(n_half[0], n_half[1])
    return all_azeit, all_hits, all_own


@app.cell
def _(
    C_OWN,
    C_REST,
    all_azeit,
    all_hits,
    all_own,
    fmt,
    mo,
    np,
    row_slice,
    row_word,
    slice_labels,
    svg_chart,
    word_ui,
):
    _ok = ~np.isnan(all_own) & ~np.isnan(all_azeit) & (all_hits > 0)
    _hl = row_word == word_ui.value
    with np.errstate(divide="ignore"):
        _lx = np.log10(all_hits)
    _xmax = np.ceil(_lx[_ok].max())

    def _cloud(xv, yv):
        def mark(sx, sy):
            out = []
            for i in np.flatnonzero(_ok & ~_hl):
                out.append(f'<circle cx="{sx(xv[i]):.1f}" cy="{sy(yv[i]):.1f}" r="1.6" fill="{C_REST}" fill-opacity="0.3"/>')
            for i in np.flatnonzero(_ok & _hl):
                out.append(
                    f'<circle cx="{sx(xv[i]):.1f}" cy="{sy(yv[i]):.1f}" r="4.5" fill="{C_OWN}" stroke="var(--background, #fff)" stroke-width="1">'
                    f"<title>{row_word[i]}, {slice_labels[row_slice[i]]}: {all_hits[i]} Belege, Kosinus {fmt(all_own[i])}, a_zeit {fmt(all_azeit[i])}</title></circle>"
                )
            return "".join(out)
        return mark

    def _level(sx, sy):
        return (
            f'<line x1="{sx(-1):.1f}" x2="{sx(1):.1f}" y1="{sy(0.7):.1f}" y2="{sy(0.7):.1f}" stroke="currentColor" stroke-dasharray="5 4"/>'
            f'<text x="{sx(-0.95):.1f}" y="{sy(0.7) - 5:.1f}" fill="currentColor">Niveau 0,7</text>'
        )

    _a = svg_chart(
        [_cloud(_lx, all_own)], (0, _xmax), (-0.2, 1), [(e, f"{10 ** e:g}") for e in range(int(_xmax) + 1)],
        [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1], "Belege in der kleineren Hälfte (log)", "eigener Kosinus A–B", width=460,
    )
    _b = svg_chart(
        [_cloud(all_own, all_azeit), _level], (-1, 1) if np.nanmin(all_own[_ok]) < 0 else (0, 1), (0, 1),
        [-1, -0.5, 0, 0.5, 1] if np.nanmin(all_own[_ok]) < 0 else [0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1],
        "eigener Kosinus A–B", "a_zeit", width=460, xfmt=lambda v: fmt(v, 2), yfmt=lambda v: fmt(v, 2),
    )
    mo.hstack([_a, _b], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unterschiede zur Diagnose

    Das Spiel oben ist die einfachste Form. `pipeline_emb_ppmi_diagnose.py` rechnet dasselbe Prinzip etwas robuster:

    - Die Ähnlichkeit zu einer Vergleichszeile ist dort der Mittelwert aus „A der Zeile mit B der Vergleichszeile“ und „B der Zeile mit A der Vergleichszeile“.
    - Die SVD wird in beiden Richtungen berechnet (an A, dann an B), die Kennwerte werden gemittelt.
    - Statt einer Teilung werden `N_SPLITS` = 20 Teilungen ausgewertet, und die Schwelle 0,7 wird auf den Mittelwert über die Prüfzeilen angewandt.

    Die Werte hier weichen deshalb leicht von denen der Diagnose ab; die Logik der AUC ist dieselbe.
    """)
    return


if __name__ == "__main__":
    app.run()
