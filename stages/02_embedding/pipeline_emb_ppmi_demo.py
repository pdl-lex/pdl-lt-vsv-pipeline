import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interaktive Demo für die Berechnung von PPMI
    In diesem Notebook wird die Funktionsweise von PPMI an einem interaktiven Beispiel erläutert. Damit lassen sich bestimme Verhaltensweisen besser nachvollziehen. Zudem werden ausgewählte Szenarien erläutert.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PPMI: Grundlagen und Beispiele

    Die Formel für PPMI (Positive Pointwise Mutual Information) ist mathematisch sehr einfach - aber die Effekte bei sich verändernden Werten und Randbedingungen sind intuitiv nicht leicht zu verstehen.


    ###Formel

    $$
    \text{{PPMI}} = \max\left(0,\ \log_2 \frac{{\text{{beobachtet}}}}{{\text{{erwartet}}}}\right)
    = \max\left(0,\ \log_2 \frac{{\text{{Anzahl}} \cdot \text{{Gesamt}}}}{{\text{{Randsumme}}_{{\text{{Ziel}}}} \cdot \text{{Randsumme}}_{{\text{{Kontext}}}}}}\right)
    $$

    Nachfolgend wird dies an einem einfachen Beispiel mit zwei Zielwörtern und drei Kontextwörtern interaktiv berechnet. Das gewünschte Ziel- und Kontextwort kann ausgewählt werden, und die Kookkuranzen können in der Tabelle frei eingegeben werden. Die ausgeblendeten Code-Zellen können ignoriert werden - sie enthalten die Logik im Hintergrund.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    target_words = ["Bundeswehr", "Frieden"]
    context_words = ["Europa", "Krieg", "der"]
    initial_counts = [[10, 25, 60], [20, 5, 45]]

    # one number input per co-occurrence cell (rows: targets, columns: contexts)
    cells = mo.ui.array([
        mo.ui.array([mo.ui.number(start=0, stop=100_000, value=v) for v in row])
        for row in initial_counts
    ])
    return cells, context_words, target_words


@app.cell(hide_code=True)
def _(cells):
    import numpy as np

    counts = np.array(cells.value)
    row_sums = counts.sum(axis=1)  # per target word
    col_sums = counts.sum(axis=0)  # per context word
    total = counts.sum()
    return col_sums, counts, np, row_sums, total


@app.cell(hide_code=True)
def _(
    col_sums,
    context_select,
    context_words,
    counts,
    row_sums,
    target_select,
    target_words,
):
    t_idx = target_words.index(target_select.value)
    c_idx = context_words.index(context_select.value)

    observed = counts[t_idx, c_idx]
    target_sum = row_sums[t_idx]
    context_sum = col_sums[c_idx]
    return c_idx, context_sum, observed, t_idx, target_sum


@app.cell(hide_code=True)
def _(context_sum, mo, target_sum, total):
    # stop here with a hint if the table is empty (avoids division by zero)
    mo.stop(total == 0, mo.md("*Die Tabelle enthält keine Kookkurrenzen, bitte mindestens einen Wert > 0 eintragen.*"))

    p_target = target_sum / total
    p_context = context_sum / total
    p_joint_expected = p_target * p_context  # probability under independence
    expected = p_joint_expected * total      # equals target_sum * context_sum / total
    return expected, p_context, p_joint_expected, p_target


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Interaktives Beispiel
    """)
    return


@app.cell(hide_code=True)
def _(context_words, mo, target_words):
    target_select = mo.ui.dropdown(options=target_words, value=target_words[0], label="Zielwort")
    context_select = mo.ui.dropdown(options=context_words, value=context_words[0], label="Kontextwort")

    mo.hstack([target_select, context_select], justify="start", gap=2)
    return context_select, target_select


@app.cell(hide_code=True)
def _(
    c_idx,
    cells,
    col_sums,
    context_words,
    mo,
    row_sums,
    t_idx,
    target_words,
    total,
):
    # scoped styling so other tables in the notebook are unaffected
    table_style = """
    <style>
      .cooc-table { border-collapse: collapse; margin: 1rem 0; }
      .cooc-table th, .cooc-table td { padding: 0.5rem 1rem; text-align: right; }
      .cooc-table th:first-child { text-align: left; }
      .cooc-table tr:last-child td, .cooc-table tr:last-child th { border-top: 1px solid #ccc; }
      .cooc-table .sel { background: rgba(200, 200, 200, 0.25); }
      .cooc-table td:last-child, .cooc-table th:last-child { border-left: 1px solid #ccc; }
    </style>
    """

    header = (
        "<tr><th></th>"
        + "".join(f"<th>{c}</th>" for c in context_words)
        + "<th>Randsumme</th></tr>"
    )

    def sel(selected):
        """Return a class attribute that highlights values used in the calculation."""
        return " class='sel'" if selected else ""

    body = "".join(
        f"<tr><th>{t}</th>"
        + "".join(
            f"<td{sel(i == t_idx and j == c_idx)}>{cells[i][j]}</td>"
            for j in range(len(context_words))
        )
        + f"<td{sel(i == t_idx)}><b>{row_sums[i]}</b></td></tr>"
        for i, t in enumerate(target_words)
    )

    footer = (
        "<tr><th>Randsumme</th>"
        + "".join(f"<td{sel(j == c_idx)}><b>{s}</b></td>" for j, s in enumerate(col_sums))
        + f"<td{sel(True)}><b>gesamt: {total}</b></td></tr>"
    )

    mo.Html(f"{table_style}<table class='cooc-table'>{header}{body}{footer}</table>")
    return


@app.cell(hide_code=True)
def _(
    context_select,
    context_sum,
    expected,
    mo,
    p_context,
    p_joint_expected,
    p_target,
    target_select,
    target_sum,
    total,
):
    def fmt(x, digits=3):
        """Format a number with a German decimal comma for LaTeX."""
        return f"{x:.{digits}f}".replace(".", "{,}")

    t, c = target_select.value, context_select.value

    mo.md(rf"""
    **Wahrscheinlichkeit des Zielworts**

    $$
    P(\text{{{t}}}) = \frac{{\text{{Randsumme}}_{{\text{{Ziel}}}}}}{{\text{{Gesamt}}}}
    = \frac{{{target_sum}}}{{{total}}} = {fmt(p_target)}
    $$

    **Wahrscheinlichkeit des Kontextworts**

    $$
    P(\text{{{c}}}) = \frac{{\text{{Randsumme}}_{{\text{{Kontext}}}}}}{{\text{{Gesamt}}}}
    = \frac{{{context_sum}}}{{{total}}} = {fmt(p_context)}
    $$

    **Erwartete Wahrscheinlichkeit für das gemeinsame Auftreten** (bei Unabhängigkeit)

    $$
    P(\text{{{t}}}) \cdot P(\text{{{c}}}) = {fmt(p_target)} \cdot {fmt(p_context)} = {fmt(p_joint_expected)}
    $$

    **Erwartete Anzahl**

    $$
    \text{{erwartet}} = \frac{{\text{{Randsumme}}_{{\text{{Ziel}}}} \cdot \text{{Randsumme}}_{{\text{{Kontext}}}}}}{{\text{{Gesamt}}}}
    = \frac{{{target_sum} \cdot {context_sum}}}{{{total}}} = {fmt(expected)}
    $$
    """)
    return c, fmt, t


@app.cell(hide_code=True)
def _(
    c,
    context_sum,
    expected,
    fmt,
    mo,
    np,
    observed,
    pmi,
    ppmi,
    ratio,
    t,
    target_sum,
    total,
):
    if expected == 0:
        concrete = r"\text{nicht definiert (erwartet = 0)}"
        interpretation = (
            f"*{t}* oder *{c}* kommt in dieser Zeitscheibe gar nicht vor. "
            "PMI ist damit nicht definiert, PPMI wird per Konvention auf 0 gesetzt."
        )
    else:
        pmi_str = r"-\infty" if np.isinf(pmi) else fmt(pmi)
        concrete = (
            rf"\max\left(0,\ \log_2 \frac{{{observed} \cdot {total}}}{{{target_sum} \cdot {context_sum}}}\right)"
            rf" = \max\left(0,\ \log_2 {fmt(ratio)}\right) = \max(0,\ {pmi_str}) = {fmt(ppmi)}"
        )
        if observed == 0:
            interpretation = (
                "Die beiden Wörter treten **nie gemeinsam** auf. "
                r"Da $\log_2 0 = -\infty$ ist, sorgt $\max(0, \cdot)$ für PPMI = 0."
            )
        elif ratio < 1:
            interpretation = (
                f"Die beiden Wörter treten **seltener als erwartet** auf (nur das ${fmt(ratio, 2)}$-Fache). "
                f"PMI ist negativ (${pmi_str}$) und wird auf 0 gekappt."
            )
        else:
            interpretation = (
                f"Die beiden Wörter treten **${fmt(ratio, 2)}$-mal so häufig** auf wie bei Unabhängigkeit erwartet. "
                "Jede Verdopplung dieses Faktors erhöht PPMI um 1."
            )

    mo.vstack([
        mo.md(rf"""
    **PPMI**

    $$
    \text{{PPMI}} = \max\left(0,\ \log_2 \frac{{\text{{beobachtet}}}}{{\text{{erwartet}}}}\right)
    = \max\left(0,\ \log_2 \frac{{\text{{Anzahl}} \cdot \text{{Gesamt}}}}{{\text{{Randsumme}}_{{\text{{Ziel}}}} \cdot \text{{Randsumme}}_{{\text{{Kontext}}}}}}\right)
    $$

    $$
    {concrete}
    $$
    """),
        mo.callout(mo.md(interpretation), kind="success" if ppmi > 0 else "neutral"),
    ])
    return


@app.cell(hide_code=True)
def _(c_idx, context_words, mo, ppmi_matrix, t_idx, target_words):
    heat_style = """
    <style>
      .ppmi-table { border-collapse: collapse; margin: 1rem 0; }
      .ppmi-table th, .ppmi-table td { padding: 0.5rem 1rem; text-align: right; }
      .ppmi-table th:first-child { text-align: left; }
    </style>
    """

    vmax = ppmi_matrix.max()

    def heat_cell(value, selected):
        """Render a cell whose background intensity reflects the PPMI value."""
        alpha = 0.6 * value / vmax if vmax > 0 else 0
        outline = "outline: 2px solid #e6a800; outline-offset: -2px;" if selected else ""
        label = f"{value:.3f}".replace(".", ",")
        return f"<td style='background: rgba(40, 120, 200, {alpha:.2f}); {outline}'>{label}</td>"

    heat_header = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in context_words) + "</tr>"

    heat_body = "".join(
        f"<tr><th>{t_word}</th>"
        + "".join(
            heat_cell(ppmi_matrix[i, j], i == t_idx and j == c_idx)
            for j in range(len(context_words))
        )
        + "</tr>"
        for i, t_word in enumerate(target_words)
    )

    mo.vstack([
        mo.md("**PPMI für alle Wortpaare**"),
        mo.Html(f"{heat_style}<table class='ppmi-table'>{heat_header}{heat_body}</table>"),
        mo.md("*Farbintensität relativ zum höchsten Wert der Tabelle; 0 = keine positive Assoziation.*"),
    ])
    return


@app.cell(hide_code=True)
def _(col_sums, counts, mo, np, row_sums, total):
    mo.stop(total == 0)

    expected_matrix = np.outer(row_sums, col_sums) / total

    # log2(0) and 0/0 produce warnings; those cells are set to 0 below anyway
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi_matrix = np.log2(counts / expected_matrix)

    ppmi_matrix = np.where(counts > 0, np.maximum(pmi_matrix, 0), 0.0)
    return (ppmi_matrix,)


@app.cell(hide_code=True)
def _(expected, np, observed):
    if observed == 0:
        ratio = 0.0
        pmi = -np.inf  # log2(0); undefined if expected is also 0
    else:
        ratio = observed / expected
        pmi = np.log2(ratio)

    ppmi = max(0.0, pmi)
    return pmi, ppmi, ratio


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Szenarien
    Der PPMI-Wert verhält sich nicht immer so, wie erwartet. Nachfolgend einige Szenarien, die bestimmte Konstellationen verdeutlichen.

    Alle Zahlen beziehen sich auf die Starttabelle des Notebooks (Bundeswehr: 10 / 25 / 60, Frieden: 20 / 5 / 45) und lassen sich direkt nachspielen.

    **Mehr Kookkurrenzen heißt nicht automatisch höherer PPMI**
    Erhöht man die Kookkurrenz Bundeswehr–Krieg, steigt nicht nur der beobachtete Wert. Auch die Randsummen beider Wörter und die Gesamtzahl wachsen, und damit steigt die erwartete Häufigkeit mit. Anfangs wächst der beobachtete Wert schneller, PPMI steigt: Bei 10 liegt er bei 0,32, bei 25 bei 0,53, und bei etwa 35 erreicht er mit 0,54 sein Maximum. Danach dominieren die beiden Wörter zunehmend die Tabelle, und ihr gemeinsames Auftreten wird immer „erwartbarer“: Bei 100 liegt PPMI bei 0,43, bei 1.000 nur noch bei 0,08. Im Grenzfall geht er gegen 0.

    **PPMI ändert sich, ohne dass sich am Wortpaar etwas ändert**
    Setzt man Frieden–der von 45 auf 1.000, bleiben Anzahl und Randsummen von Bundeswehr–Krieg exakt gleich. Trotzdem steigt ihr PPMI von 0,53 auf 3,30. Der Grund ist die gestiegene Gesamtzahl: Bundeswehr und Krieg machen nun einen viel kleineren Anteil aller Beobachtungen aus, ihr gemeinsames Auftreten ist also weniger erwartbar. PPMI ist immer relativ zum ganzen Korpus.

    **Ein Wort wird „geselliger“, und seine Assoziationen werden schwächer**
    Erhöht man Bundeswehr–der von 60 auf 200, sinkt der PPMI von Bundeswehr–Krieg von 0,53 auf 0,11, bei 1.000 sogar auf 0. Bundeswehr kommt jetzt insgesamt viel häufiger vor, vor allem mit einem Funktionswort. Die 25 gemeinsamen Vorkommen mit Krieg sind dadurch nichts Besonderes mehr. Häufige, unspezifische Kontexte verwässern also die Assoziationen eines Zielworts.

    **Ein einziges Vorkommen schlägt 25**
    Setzt man die Frieden-Zeile auf 0 / 1 / 0, kommt Frieden genau einmal vor, und zwar mit Krieg. Frieden–Krieg erreicht damit einen PPMI von 1,88. Bundeswehr–Krieg mit 25 gemeinsamen Vorkommen fällt dagegen auf 0. Seltene Wörter können mit sehr wenigen Belegen extreme PPMI-Werte erzeugen. Diese bekannte Schwäche von PMI ist der Grund, warum man in der Praxis Mindesthäufigkeiten ansetzt oder glättet.

    **PPMI = 0 kann Verschiedenes bedeuten**
    Frieden–Krieg hat in der Starttabelle 5 gemeinsame Vorkommen. Der PMI liegt bei −1,35, die beiden Wörter treten also deutlich seltener zusammen auf als erwartet, und PPMI kappt das auf 0. Setzt man die Zelle auf 0, ist PPMI ebenfalls 0. Ob zwei Wörter sich meiden, nie zusammen vorkommen oder schlicht unabhängig sind, ist am PPMI nicht mehr zu erkennen. Das ist eine bewusste Vereinfachung, weil negative PMI-Werte bei kleinen Zahlen sehr unzuverlässig sind.

    **Die Korpusgröße allein spielt keine Rolle**
    Multipliziert man alle Zellen mit 10, bleiben sämtliche PPMI-Werte exakt gleich, da sich beobachteter und erwarteter Wert im gleichen Maß ändern. Zeitscheiben unterschiedlicher Größe lassen sich deshalb grundsätzlich vergleichen. Wie verlässlich die Werte sind, hängt aber sehr wohl von der Größe ab: Eine Zeitscheibe mit 5 Belegen liefert denselben PPMI wie eine mit 5.000, aber mit viel größerer Zufallsschwankung.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PPMI vs log-likelihood bzw. G²
    Beide Metriken beruhen auf dem logarithmierten Verhältnis von beobachteter zu erwarteter Häufigkeit, log(O/E). PPMI betrachtet es nur für die eine Zelle und misst damit die **Stärke einer Assoziation**. G² (in der Korpuslinguistik meist einfach „Log-Likelihood“ genannt) summiert dagegen O · log(O/E) über alle vier Zellen der 2×2-Kontingenztafel, gewichtet also mit der Häufigkeit und misst damit, **wie gut belegt** die Assoziation ist.

    PPMI ist aber für die Vektorisierung das etabliertere Standard-Verfahren. Es ist unabhängig von der Korpusgröße, eignet sich damit besser für Zeitscheiben-Vergleiche, und ist weniger kontextabhängig.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tatsächliche Berechnung
    Die tatsächliche Berechnung erfolgt im Notebook *pipeline_emb_ppmi.py*.
    """)
    return


if __name__ == "__main__":
    app.run()
