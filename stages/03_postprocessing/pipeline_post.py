import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import json
    import numpy as np

    return json, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pipeline zur Erstellung von Bundle-Daten für VSV: Bundle-Generierung
    Dieses Notebook bietet einen vollständige, kommentierte Pipeline zur Generierung von Daten, ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Vorbereitung des Bundles
    Das Lexoterm-Tool VSV (Visualisierung Semantischer Vektoren) erwartet die Daten in einem definierten Format (siehe unten bzw. Readme Datenmodell.md)
    In den Folgenden Schritten werden aus den bisher erzeugten Daten die nötigen Daten für das Bundle erstellt.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Datenmodell

    ```
    bundle_xyz.json
    ├── meta            Metadaten
    ├── entities        Zielwörter
    ├── time_slices     Zeitscheiben
    ├── facets_schema   deklarierte Facetten-Achsen
    ├── records         nicht-aggregierte Zellen
    ├── aggregates      aggregierte Zelle
    ├── anchors         Ankerwörter
    └── neighbors       Nächste Nachbarn
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Metadaten (meta)**
    Beschreibung, Vektordimension, Projektion

    **Zielwörter (entities)**
    händisch erstellte Liste, in diesem Notebook hart-codiert in "Definition Suchwörter"

    **Zeitscheiben (time_slices)**
    Die Größe der Zeitscheiben kann gewählt werden: Quartale, Halbjahre, Jahre. Prinzipiell wären auch noch feinere (und gröbere) Auflösungen denkbar, aber dafür müssen ausreichend Daten vorhanden sein.
    Umfasst der Untersuchungszeitraum 10 Jahre, und man wählt die Auflösung Quartale, ergeben sich 10 x 4 = 40 Zeitscheiben.

    **deklarierte Facetten-Achsen (facets_schema)**
    Grundsätzlich lassen sich weitere Analyse-Dimensionen definieren. Bei der Analyse von Zeitungskorpora böte sich z.B. "Publikation" an.
    Die Anzahl der Facetten ist nicht beschränkt, das Frontend (VSV) kann momentan aber nur eine weitere Facette farblich unterscheidbar darstellen.

    **nicht-aggregierte Zellen (records)**
    Hier befinden sich die nicht aggregierten Messwerte. Die Anzahl der Zellen errechnet sich aus: Zielwörter x Zeitscheiben x Facetten.
    Bei 20 Zielwörtern, 40 Zeitscheiben und 3 Facetten-Achsen ergeben sich so 2.400 Zellen.

    **aggregierte Zellen (aggregates)**
    Wie zuvor, aber ohne Facetten-Auflösung, also aggregiert: Zielwörter x Zeitscheiben.

    **Ankerwörter**
    Händisch ausgewählte Wörter, die zusätzlich Clustern zugeordnet werden. Sie stammen aus einer parameterunabhängigen Maximalliste (input/anchors.csv); im Bundle landen die, die unter den Kontextwörtern des Laufs vorkommen. Ankerwörter bilden quasi das semantische Grundgerüst des 3D-Raums, in den die Trajektorien der semantischen Verschiebung projiziert werden. Ohne sie sähe man zwar die Verschiebungspfade, aber ohne jegliche Anhaltspunkte. Eine automatisch erzeugte Vorschlagsliste gibt es derzeit nicht (s. Abschnitt Anker).

    **Nächste Nachbarn (neighbors)**
    Vorberechnete nächste Nachbarn pro Zielwort/Zeitscheibe, gerankt nach rohem Kosinus im Raum nach SVD. Die Listenlänge (n_neighbors) ist zugleich die Obergrenze des Reglers im Frontend, der damit auch die Wortwolke um das Zielwort steuert. Für die 3D-Darstellung im Frontend wird danach eine weitere PCA-Reduktion auf 3 Dimensionen durchgeführt. Die Nähe im hochdimensionalen Raum und die Nähe in der 3D-Ansicht sind nicht direkt vergleichbar.

    **Suchraum für das Backend**
    Zusätzlich schreibt das Notebook neben das Bundle die Vektoren und eine Indextabelle mit allen Attributen, nach denen gefiltert werden kann (Wortart, Belegung, Rednername). Damit soll das Backend die Nachbarn künftig bei jeder Abfrage mit den im Frontend gewählten Filtern berechnen (s. Abschnitt am Ende).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Input
    Geladen werden die Daten aus stages/02_embedding/output/[kwic_variant].
    Pro KWIC-Variante und Slice existieren fünf Dateien:
    * Zielwörter
     * **target_slice_vectors_[kwic_variant]_[slice_grid].npy**: Zielwortvektoren pro Zeitscheibe (Zeile i gehört zu target_slice_idx in  target_slices.parquet)
     * **target_slices_[kwic_variant]_[slice_grid].parquet**: target_slice_idx, Zielwort, Zeitscheibe und die Trefferzahl (n_hits) je Zeile

    * Kontextwörter
     * **context_vectors_[kwic_variant]_[slice_grid].npy**: Kontextwort-Vektoren (Zeile j gehört zu context_idx in context_words.parquet)
     * **context_words_[kwic_variant]_[slice_grid].parquet**: context_idx, Kontextwort, Belegung (in wie vielen Zeilen PPMI > 0, Summe der Kookkurrenzen) und häufigste Wortart
    * Metadaten
     * **emb_meta_[kwic_variant]_[slice_grid].json**: Quelldatei, Zeitraster, k, p und der Frobenius-Anteil

    Möchte man verschiedene Outputs für verschiedene Varianten erzeugen, muss der embedding-Schritt mehrfach durchlaufen werden.

    In den folgenden Zellen werden die Daten einzeln geladen und zur besseren Veranschaulichung nochmals dargestellt.
    """)
    return


@app.cell
def _():
    kwic_variant="punkt"
    slice_grid="halfyear"
    return kwic_variant, slice_grid


@app.cell
def _(kwic_variant, np, slice_grid):
    # target_slice_vectors.npy (target words per slice)
    # one row per target word and slice, row i belongs to target_slice_idx in target_slices

    target_slice_vectors = np.load(f"../02_embedding/output/{kwic_variant}/target_slice_vectors_{kwic_variant}_{slice_grid}.npy")

    print(f"Die Vektor-Matrix der Zielwörter hat {target_slice_vectors.shape[1]} Spalten, definiert durch den Wert von k im Embedding-Schritt. Sie hat {target_slice_vectors.shape[0]} Zeilen. Das entspricht Zielwörter x Jahre (13) x Scheibengröße abzüglich nicht belegter Scheiben.")

    target_slice_vectors
    return (target_slice_vectors,)


@app.cell
def _(kwic_variant, np, slice_grid):
    # context_vectors.npy (context words)
    # one row per context word (column side of the SVD), row j belongs to context_idx in context_words_emb

    context_vectors = np.load(f"../02_embedding/output/{kwic_variant}/context_vectors_{kwic_variant}_{slice_grid}.npy")

    print(f"Die Vektor-Matrix der Kontextwörter hat {context_vectors.shape[1]} Spalten, definiert durch den Wert von k im Embedding-Schritt. Sie hat {context_vectors.shape[0]} Zeilen: eine je Kontextwort-Lemma, das in mindestens einer Zelle (Zielwort × Zeitscheibe) einen PPMI-Wert > 0 hat. Das entspricht den Spalten der PPMI-Matrix.")

    context_vectors
    return (context_vectors,)


@app.cell
def _(kwic_variant, pl, slice_grid):
    # vector row headers and data
    # target_slices
    # target_slice_idx, target_lemma, slice, n_hits

    target_slices = pl.read_parquet(f"../02_embedding/output/{kwic_variant}/target_slices_{kwic_variant}_{slice_grid}.parquet")

    print(f"Die Tabelle 'target_slices' hat {target_slices.shape[1]} Spalten: Index, Lemma, Slice und Frequenz. Sie hat {target_slices.shape[0]} Zeilen:  Das entspricht Zielwörter x Jahre (13) x Scheibengröße abzüglich nicht belegter Scheiben. Der Index 'row i' passt zum Index von vectors (gleiche Zeilenzahl).")


    target_slices
    return (target_slices,)


@app.cell
def _(kwic_variant, pl, slice_grid):
    # context_vectors column headers and data
    # context_words_emb
    # context_idx, context_lemma, n_target_slices, n_cooc, pos

    context_words_emb = pl.read_parquet(f"../02_embedding/output/{kwic_variant}/context_words_{kwic_variant}_{slice_grid}.parquet")

    print(f"Die Tabelle 'context_words_emb' hat {context_words_emb.shape[1]} Spalten: Index, Kontext-Lemma, Anzahl der belegten Zellen, Anzahl der Vorkommen in den belegten Zellen, und POS. Sie hat {context_words_emb.shape[0]} Zeilen: eine je Kontextwort-Lemma, das in mindestens einer Zelle (Zielwort × Zeitscheibe) einen PPMI-Wert > 0 hat. Der Index 'column j' passt zum Index von context_vectors (gleiche Zeilenzahl). \n\nDer PPMI-Wert ist nur dann >0, wenn das Vorkommen häufiger ist als erwartet. Ein Kontextwort kanna lso durchaus häufiger auftauchen, wird aber nur gewertet, wenn das Vorkommen wahrscheinlich ist als zu erwarten war.\n\nn_occurrences ist immer gleich oder größer als n_target_slices. Mehrfach gezählt wird ein Kontextwort in drei Fällen: 1) Das Wort steht in mehreren KWIC-Kontexten derselben Zelle. 2) Das Wort steht mehrmals im selben Kontext. 3) Ein Satz enthält zwei Zielwörter. Dann gibt es zwei KWIC-Zeilen, und der Kontext wird für beide Zielwörter gezählt." )

    context_words_emb
    return (context_words_emb,)


@app.cell
def _(json, kwic_variant, slice_grid):
    # Meta Data
    # emb_meta.json

    with open(f"../02_embedding/output/{kwic_variant}/emb_meta_{kwic_variant}_{slice_grid}.json", encoding="utf-8") as _f:
        emb_meta = json.load(_f)

    # _ to avoid double import
    _slice_grid = emb_meta["slice_grid"]  # year, halfyear, quarter

    print(emb_meta)
    return (emb_meta,)


@app.cell
def _(
    context_vectors,
    context_words_emb,
    np,
    target_slice_vectors,
    target_slices,
):
    # vectors and labels are matched by position only; this catches stale or mixed-up files
    assert (target_slices["target_slice_idx"].to_numpy() == np.arange(target_slice_vectors.shape[0])).all()
    assert (context_words_emb["context_idx"].to_numpy() == np.arange(context_vectors.shape[0])).all()

    # print(f"vectors: {vectors.shape}, context_vectors: {context_vectors.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kosinus-Tabelle
    Für die spätere Analyse ist die Kosinus-Distanz entscheidend, also die unterschiedlichen Größen der Innenwinkel. Die Formel für Kosinus lautet a·b / (|a|·|b|). Normiert man die Vektoren vorher zu Einheitsvektoren der Länge 1, dann wird der Nenner 1, und die Kosinus-Tabelle ergibt sich durch ein einfaches Skalarprodukt. Die Länge der Vektoren ist für den Winkel nicht entscheidend, deswegen hat diese Normierung an dieser Stelle keine negativen Auswirkungen.
    Der Vektorraum bleibt durch diese Operation unverändert - es entsteht nur die Tabelle der Kosinus-Winkel.
    """)
    return


@app.cell
def _(context_vectors, np, target_slice_vectors):
    # normalize both sides to length 1; the matrix product is then the complete
    # cosine table: one row per matrix row, one column per context word
    target_slice_vectors_unit = target_slice_vectors / np.maximum(
        np.linalg.norm(target_slice_vectors, axis=1, keepdims=True), 1e-12
    )
    context_vectors_unit = context_vectors / np.maximum(
        np.linalg.norm(context_vectors, axis=1, keepdims=True), 1e-12
    )
    target_slice_context_cosines = target_slice_vectors_unit @ context_vectors_unit.T

    print(f"Kosinus-Tabelle: {target_slice_context_cosines.shape} ({target_slice_context_cosines.nbytes / 1e6:.0f} MB)")
    return (
        context_vectors_unit,
        target_slice_context_cosines,
        target_slice_vectors_unit,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Erzeugung von IDs
    In den folgenden Zellen werden IDs erzeugt, die zur Verlinkung der verschiedenen Teil des Bundles dienen. Diese werden als *slugs* erstellt; eventuell wäre dieser Schritt auch verzichtbar, dann würden die Wörter selbst als IDs dienen (evtl. Risiko durch Sonderzeichen).
    """)
    return


@app.cell
def _(pl, target_slices):
    # The following function creates simple IDs for the target words. In principle, different words ("Maß", "Mass") could be mapped to the same ID by mistake, but since the target word list is small and known, the risk is low. The assert condition performs a simple check in this respect.

    def to_target_id(column):
        """Lemma -> entity_id: lowercase, umlauts transliterated."""
        return (
            column.str.to_lowercase()
            .str.replace_all("ä", "ae")
            .str.replace_all("ö", "oe")
            .str.replace_all("ü", "ue")
            .str.replace_all("ß", "ss")
            .str.replace_all("-", "_")
        )

    # target words
    targets = (
        target_slices.select("target_lemma")
        .unique()
        .sort("target_lemma")
        .with_columns(
            [
                to_target_id(pl.col("target_lemma")).alias("id"),
                pl.col("target_lemma").alias("label"),
            ]
        )
        .select(["id", "label"])
    )

    assert targets["id"].n_unique() == targets.height, (
        "to_target_id() collision: at least two lemmas were mapped to the same ID."
    )

    targets
    return targets, to_target_id


@app.cell
def _(pl, target_slices):
    # ids for time_slices # ids may be superfluous
    # values must be gapless and in order
    # due to the label form (e.g. "2013-Q4"), alphabetical order equals chronological order
    time_slices = (
        target_slices.select("slice")
        .unique()
        .sort("slice")
        .with_row_index("index")
        .with_columns(
            [
                pl.col("slice").alias("id"),
                pl.col("slice").alias("label"),
            ]
        )
        .select(["id", "label", "index"])
    )
    time_slices
    return (time_slices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3D-Koordinaten mittels PCA
    ### Zielwörter(-Aggregate)
    Zur visuellen Darstellung in einer 3D-Ansicht müssen die Wortvektoren aus dem hochdimensionalen Raum in den dreidimensionalen überführt werden. Dies geschieht mittels Principal Component Analysis, welche die Achsen der größten Varianz (Annahme: größte Erklärungskraft für Unterschiede) selektiert.

    Konzeptionell wichtig ist, dass alle Koordinaten in den gleichen Raum projiziert werden, und nicht etwa pro Zeitscheibe in verschieden Räume. Würde man die Zeitscheiben unabhängig voneinander einer PCA unterziehen, so wären die entstehenden Koordinatenräume nicht vergleichbar.

    Gefittet wird die PCA nur auf den Aggregaten (Zielwort × Zeitscheibe). Alle Kontextwörter, darunter die Anker, werden anschließend mit denselben Achsen in den Raum projiziert, ohne neuen Fit (Out-of-Sample-Projektion, s. Kontextwörter). So hängen die Achsen nicht davon ab, welche und wie viele Anker ein Lauf im Vokabular findet oder im Frontend eingeblendet werden.
    """)
    return


@app.cell
def _(target_slice_vectors_unit):
    # normalization to length 1, computed above with the cosine table
    # The vectors have different lengths (Euclidean norm, not dimensionality), resulting
    # from the amount of data in a cell (more speech volume => more text => more
    # cooccurrences => more non-zero cells and larger values => larger geometric length).
    # Unnormalized, this length difference is the largest spread in the data.
    # PCA would build its first axis from it instead of from the actual (semantically
    # interesting) differences in direction.

    # The similarity computation via cosine is length-invariant, but choosing the right axes is crucial.

    # The normalization here is not to be confused with the row normalization **before** the SVD, which was rejected.

    target_slice_vectors_unit.shape
    return


@app.cell
def _(np, target_slice_vectors_unit):
    # centering and SVD
    # Mathematically, PCA is the SVD of a column-wise centered matrix.
    # Difference to the earlier SVD: length normalization and centering

    # column mean over all aggregates, i.e. the "center of mass" of the point
    # cloud in the n-dimensional, length-normalized space. A vector with
    # n entries, n = number of dimensions after SVD
    pca_mean = target_slice_vectors_unit.mean(axis=0)

    # centering (the n-dim mean vector is subtracted from each row) of the point
    # cloud, so that its center of mass lies at the origin. Necessary because
    # SVD/PCA only finds the "right" axes on centered data.
    pca_centered = target_slice_vectors_unit - pca_mean

    # decomposition
    # conversely: pca_centered = U @ diag(S) @ Vt.
    # The rows of Vt are the principal axes, sorted by explained variance in
    # descending order; S the corresponding singular values (strength of each
    # axis); U the coordinates of the training points in this new axis system,
    # normalized to unit length.
    pca_U, pca_S, pca_Vt = np.linalg.svd(pca_centered, full_matrices=False)

    # cut to the three strongest axes
    pca_axes = pca_Vt[:3]

    # For the fitted points this is identical to U[:, :3] * S[:3] - but here as a
    # general formula that works the same way for new vectors:
    #   (new_vector_unit - pca_mean) @ pca_axes.T
    target_slice_coords = pca_centered @ pca_axes.T

    pca_explained = (pca_S[:3] ** 2).sum() / (pca_S**2).sum()
    print(f"erklärte Varianz (3 Komponenten): {pca_explained:.1%}")
    return pca_axes, pca_explained, pca_mean, target_slice_coords


@app.cell
def _(
    pl,
    target_slice_coords,
    target_slice_vectors_unit,
    target_slices,
    to_target_id,
):
    # combine vectors and 3D coordinates with existing data
    # order of target_slice_vectors_unit and target_slice_coords = order of target_slices, which has
    # not been re-sorted since loading
    bundle_aggregates = (
        target_slices.with_columns(
            [
                pl.Series("vector", target_slice_vectors_unit.tolist()),
                pl.Series("coords_3d", target_slice_coords.tolist()),
            ]
        )
        # rename to the schema names
        .with_columns(
            [
                to_target_id(pl.col("target_lemma")).alias("entity_id"),
                pl.col("slice").alias("time_slice_id"),
            ]
        )
        .select(
            ["entity_id", "time_slice_id", "vector", "coords_3d", pl.col("n_hits").alias("freq")]
        )
    )

    # every matrix row has KWIC hits; null would point to a broken join
    assert bundle_aggregates["freq"].null_count() == 0, (
        f"freq null in {bundle_aggregates['freq'].null_count()} Zeilen"
    )

    # only needs to be saved for analysis
    # bundle_aggregates.write_parquet("output/bundle_aggregates.parquet")
    bundle_aggregates
    return (bundle_aggregates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hinweis: Die Benennung "aggregates" bezieht sich darauf, dass die Facetten nicht aufgelöst, sondern aggregiert werden.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Kontextwörter
    In den folgenden Zellen werden die 3D-Koordinaten für die Kontextwörter erzeugt. Zusätzlich werden alle Begriffe, die identisch sind mit den Rednernamen (erzeugt im Preprocessing), als "is_speaker_name" markiert. Dies dient der späteren Filterbarkeit im Front-End. Tokenisierungsreste werden als "is_token_artifact" markiert. Auch die weiteren Werte, die der Filterung dienen, werden mitgeführt (pos, n_target_slices, n_cooc, is_speaker_name). Die Zeilenfolge entspricht context_idx, Zeile j gehört also zu Spalte j der Kosinus-Tabelle.
    """)
    return


@app.cell
def _(pl):
    # names of the persons speaking in the corpus, exported by pipeline_pre_ppmi.py.
    # More precise than NER and without another NLP run, because it is an exact
    # list of real persons. Does not cover names of persons who are not speakers.
    speaker_names = pl.read_csv("../01_preprocessing/output/speaker_names.csv")["name"]
    return (speaker_names,)


@app.cell
def _(
    context_vectors_unit,
    context_words_emb,
    pca_axes,
    pca_mean,
    pl,
    speaker_names,
):
    # coordinates for every context word: same normalization, pca_mean and
    # pca_axes as the PCA fit, no new fit
    context_coords = (context_vectors_unit - pca_mean) @ pca_axes.T

    # tokenization residues: no letter at all ("2", "000"), single characters
    # ("e", "§"), or not starting and ending with a letter or digit
    # ("-AG", "Dr.", "‑Prozent"). Keeps "CDU/CSU", "F-35", "87a", "90er".
    _L = pl.col("context_lemma")
    _is_token_artifact = (
        ~_L.str.contains(r"\p{L}")
        | (_L.str.len_chars() == 1)
        | ~_L.str.contains(r"^[\p{L}\d].*[\p{L}\d]$")
    )

    # recompile, flag speaker names and tokenization residues
    context_words = context_words_emb.with_columns(
        [
            (
                (pl.col("pos") == "PROPN")
                & pl.col("context_lemma").is_in(speaker_names.implode())
            ).alias("is_speaker_name"),
            _is_token_artifact.alias("is_token_artifact"),
            pl.Series("coords_3d", context_coords.tolist()),
        ]
    )

    print(
        f"Kontextwörter: {len(context_words):,}, davon "
        f"{context_words['is_speaker_name'].sum():,} als Rednername und "
        f"{context_words['is_token_artifact'].sum():,} als Tokenisierungsreste markiert"
    )
    return (context_words,)


@app.cell
def _(context_words):
    context_words
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Anker
    Anker sind thematisch besonders wichtige Wörter, die im VSV-Tool als beschriftete Punkte im Wortraum dargestellt werden. Sie bilden das semantische Grundgerüst, vor dem die Trajektorien lesbar werden. Ohne sie sähe man zwar die Verschiebungspfade, aber ohne jegliche Anhaltspunkte.

    **Ankerliste**
    input/anchors.csv ist eine **Maximalliste** aller sinnvollen Ankerwörter, unabhängig von den Parametern in pre und emb. Sie wird händisch gepflegt und vom Notebook nicht verändert. Spalten:

    * **lemma:** das Lemma, wie es unter den Kontextwörtern stehen kann. Mehrere Lemma-Varianten (etwa aus unterschiedlicher Lemmatisierung: Einsatzkräft, Einsatzkraft) dürfen dasselbe Label tragen.
    * **label:** Anzeigeform; leer bedeutet: Lemma unverändert anzeigen.
    * **cluster:** thematische Gruppe.

    Welche Anker im Bundle landen, ergibt sich aus den Kontextwörtern des jeweiligen Laufs. Lemmata, die nicht vorkommen, werden übergangen; das ist kein Fehler. Kommen mehrere Varianten desselben Labels vor, wird die mit der größten Belegung (n_target_slices) genommen, denn ein Label darf im Bundle nur einmal vorkommen. Die Koordinaten stammen aus der Kontextwort-Tabelle, die Anker bestimmen die PCA-Achsen nicht mit.

    Für die Darstellung sind ca. 50-100 Anker in maximal 6 Clustern empfehlenswert. Die Auswahl aus der Maximalliste erfolgt im Frontend.

    Eine automatisch erzeugte Vorschlagsliste gibt es nicht mehr. Die frühere Rangliste nach „Kontrast“ (Spitzen-Kosinus minus mittlerer Kosinus eines Kandidaten) war kein etabliertes Verfahren und wurde entfernt. Als Ersatz ist eine Keyness-Analyse vor und nach der Zäsur vorgesehen (Log-Likelihood nach Dunning 1993; Rayson & Garside 2000) auf Basis der Kookkurrenz-Zählungen aus dem Preprocessing.

    Das Front-End wird in Zukunft die Möglichkeit bieten, Zielwörter als Anker auszuschließen (denn die Anzeige von Trajektorien und Ankern zugleich ist verwirrend.)

    Die gesamte Ankerlogik funktioniert nicht wirklich bei einer großen Zahl von Zielwörtern (z.B. 500); allerdinsg funktioniert dann die 3D-Darstellung generell nicht vernünftig. Diese macht nur bei einer kleinen Zahl (<20) Sinn, also für die "Endauswahl".
    """)
    return


@app.cell
def _(context_vectors_unit, context_words, pl):
    # empty label means "show lemma unchanged". Polars reads empty fields as null;
    # coalesce takes the first non-null value, i.e. the lemma here.
    anchor_list = pl.read_csv("input/anchors.csv").with_columns(
        pl.coalesce([pl.col("label"), pl.col("lemma")]).alias("label")
    )

    # match against the context words of this run; missing lemmas are expected
    # a word that is not a context word cannot be an anchor
    anchor_matches = anchor_list.join(
        context_words.select(["context_lemma", "context_idx", "n_target_slices", "coords_3d"]),
        left_on="lemma",
        right_on="context_lemma",
        how="inner",
    )

    # one variant per label, the one with the largest occupancy;
    # lemma as tie-breaker keeps the choice deterministic
    # important if several words are mapped to the same label
    anchors = (
        anchor_matches.sort(["n_target_slices", "lemma"], descending=[True, False])
        .unique(subset="label", keep="first", maintain_order=True)
        .sort(["cluster", "label"])
    )

    # vector: the length-normalized context vector, as for the aggregates.
    # context_idx is read from the same (final) frame, so the order matches.
    # this is effectively a join, just by adding the column assuming the correct order
    anchors = anchors.with_columns(
        pl.Series("vector", context_vectors_unit[anchors["context_idx"].to_numpy()].tolist())
    ).select(["lemma", "label", "cluster", "vector", "coords_3d"])

    # only needs to be saved for analysis
    # anchors.write_parquet("output/bundle_anchors.parquet")

    # coverage per cluster: labels in the list vs. labels found in this run
    anchor_coverage = (
        anchor_list.group_by("cluster")
        .agg(pl.col("label").n_unique().alias("in_list"))
        .join(
            anchors.group_by("cluster").len(name="in_bundle"),
            on="cluster",
            how="left",
        )
        .with_columns(pl.col("in_bundle").fill_null(0))
        .sort("cluster")
    )
    anchors_missing = (
        anchor_list.join(anchors, on="label", how="anti")["label"].unique().sort()
    )

    print(
        f"Ankerliste: {anchor_list['label'].n_unique()} Labels "
        f"({len(anchor_list)} Lemmata) in {anchor_list['cluster'].n_unique()} "
        f"Clustern, davon unter den Kontextwörtern dieses Laufs: {len(anchors)}"
    )
    print(f"nicht unter den Kontextwörtern: {', '.join(anchors_missing.to_list()) or '-'}")
    anchor_coverage
    return (anchors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nearest Neighbours
    Anker und Nachbarn erfüllen verschiedene Zwecke. Anker sind dauerhaft beschriftete, sinntragende Wörter, die als Punkte im 3D-Raum angezeigt werden. Deswegen sind die Anker händisch kuratiert.

    Die Nachbarn werden aus allen Kontextwörtern ausgewählt, gerankt nach der rohen Kosinus-Distanz im k-dimensionalen Raum.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filter-Vorbereitung

    Die mitgelieferten neighbors im Bundle werden mit den untenstehenden Voreinstellungen gefiltert. Matrix und Suchraum bleiben aber vollständig, im Frontend lassen sich die Filter ändern oder deaktivieren.
    Es müssen alle Filterbedingungen erfüllt sein:

    * Stopwords
    * Mindestbelegung
    * Wortart
    * Rednername
    * Tokenisierungsreste
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Stopwords
    Der folgende Code liest die händisch erstellte Liste "input/stopwords.txt" ein, verarbeitet sie, und speichert sie als Liste für das Bundle.
    """)
    return


@app.cell
def _(context_words):
    # process stopword list provided in the repo: one lemma per line, "#" starts a comment line
    with open("input/stopwords.txt", encoding="utf-8") as _f:
        _lines = [_l.strip() for _l in _f]

    # dict.fromkeys removes duplicates and keeps the order of the file
    stopwords = list(dict.fromkeys(_l for _l in _lines if _l and not _l.startswith("#")))

    # find stopwords that are not among context_words; just for statistics
    _missing = sorted(set(stopwords) - set(context_words["context_lemma"].to_list()))

    print(
        f"Stoppwörter: {len(stopwords)}, davon unter den Kontextwörtern dieses Laufs: "
        f"{len(stopwords) - len(_missing)}"
    )
    print(f"Nicht unter den Kontextwörtern: {', '.join(_missing) or '-'}")
    return (stopwords,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Mindestbelegung
    Erfordert einen PPMI-Wert > 0 in mindestens **neighbor_n_target_slices_min** Matrixzeilen (2 % der Zeilen, mindestens 10).
    Hintergrund: PMI überschätzt seltene Ereignisse (Church & Hanks 1990; Levy, Goldberg & Dagan 2015). Im SVD-Raum kommt hinzu: Ist ein Kontextwort nur in einer Zeile i belegt, ist sein Vektor v_j = M_ij · S⁻¹ · u_i; alle solchen Wörter derselben Zeile haben dieselbe Richtung und denselben Kosinus. Identische Kosinuswerte in einer Nachbarliste zeigen eine zu niedrige Schwelle an.

    Vorsicht bei großer Zielwortliste: 1000 Zielwörtern × 26 Zeitscheiben ergeben rund 26.000 Zeilen. 2 % davon wären 520 belegte Zellen: Dieser Wert ist vermutlich zu hoch (ggf. Code anpassen; änderbar im Frontend)
    """)
    return


@app.cell
def _(target_slices):
    neighbor_n_target_slices_min = max(10, round(len(target_slices) * 0.02))
    neighbor_n_target_slices_min 
    return (neighbor_n_target_slices_min,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Wortart
    Voreinstellung der erlaubten Wörter: NOUN, PROPN, ADJ, VERB
    """)
    return


@app.cell
def _():
    neighbor_pos_allowed = ["NOUN", "PROPN", "ADJ", "VERB"]
    return (neighbor_pos_allowed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Rednername und Tokenisierungsreste
    Diese Eigenschaften wurden im Abschnitt "Kontextwörter" geflaggt. Im Folgenden werden sie dann nur noch ausgefiltert.
    """)
    return


@app.cell
def _(
    context_words,
    neighbor_n_target_slices_min,
    neighbor_pos_allowed,
    pl,
    stopwords,
    target_slices,
):
    # actual filter
    neighbor_candidates = context_words.filter(
        (pl.col("n_target_slices") >= neighbor_n_target_slices_min)
        & pl.col("pos").is_in(neighbor_pos_allowed)
        & ~pl.col("is_speaker_name")
        & ~pl.col("is_token_artifact")
        & ~pl.col("context_lemma").is_in(stopwords)
    )

    print(
        f"Nachbar-Kandidaten: {len(neighbor_candidates):,} von {len(context_words):,} "
        f"(Mindestbelegung {neighbor_n_target_slices_min} von {len(target_slices)} Zeilen, "
        f"Wortart, Rednernamen, Stoppwörter)"
    )
    return (neighbor_candidates,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Kosinus-Kreuz-Tabelle
    baseline_cosines enthält den Kosinus jeder Matrixzeile zu jedem Kandidaten. Daraus wird später der Grundpegel berechnet (wie ähnlich ist ein beliebiges Paar?), mit dem die Ähnlichkeitsprüfung am Listenende die Nachbarlisten vergleicht. Für das Ranking der Nachbarn wird diese Tabelle nicht verwendet.
    """)
    return


@app.cell
def _(neighbor_candidates, target_slice_context_cosines):
    # the filter only selects columns of the shared cosine table.
    # Column i of baseline_cosines belongs to row i of neighbor_candidates,
    # so do not re-sort neighbor_candidates afterwards.
    baseline_cosines = target_slice_context_cosines[:, neighbor_candidates["context_idx"].to_numpy()]
    return (baseline_cosines,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Voller Suchraum für die Filterung
    Backend bzw. Frontend des VSV-Tools können die Nachbarn und Anker je nach Filtereinstellungen neu berechnen. Dazu benötigen sie aber die die vollen Informationen zu Zielwörtern (bzw. deren Zeitscheiben-Aggregate) und Kontextwörtern. Diese werden nun erstellt.
    Die Daten zu den Zielwörter-Aggregaten sind auch in der Haupt-JSON-Datei enthalten. Diese Doppelung ist insofern gewollt, als das Tool auch nur mit dem JSON funktioniert. Fehlen die nachfolgend erzeugten Daten, funktioniert nur die Filterung im Frontend nicht.
    Die Numpy-Matrix enthält nur Zahlen; der Index dazu liefert die Beschriftung und die Metadaten (unterschiedliche Datentypen).

    * **.space_vectors.npy:** alle Vektoren als eine Matrix. Zuerst die Abfragevektoren (eine Zeile je Zielwort und Zeitscheibe, identisch mit aggregates.vector), danach die Kontextwortvektoren (eine Zeile je Kontextwort, ungefiltert).
    * **.space_index.parquet:** eine Zeile je Matrixzeile, in derselben Reihenfolge (Spalte row). Spalten: role ("query" oder "candidate"), slice, entity_id (nur Abfragen), lemma, coords_3d und alle Attribute, nach denen gefiltert werden kann (pos, n_target_slices, n_cooc, is_speaker_name, is_token_artifact).

    Das Format passt auch für SGNS. Dort hat jede Zeitscheibe einen eigenen Raum, und jeder Kandidat trägt seine Zeitscheibe. Bei PPMI teilen sich alle Zeitscheiben einen Raum: Die Zeitscheibe der Kandidaten ist leer und bedeutet „gilt für alle Zeitscheiben“. Eine Abfrage vergleicht den Abfragevektor einer Zeitscheibe mit den Kandidaten derselben Zeitscheibe bzw. mit den Kandidaten ohne Zeitscheibe. Abfrage- und Kandidatenvektoren müssen im selben Raum liegen; bei SGNS also entweder beide vor oder beide nach der Ausrichtung.

    meta.neighbor_space im Bundle nennt die beiden Dateien, die Art der Nachbarn (PPMI: Kontextwörter, also das Kollokationsprofil; SGNS: Wörter im selben Raum), die Spalten, die als Filter angeboten werden, mit den Voreinstellungen dieses Notebooks, und unter stopwords die mitgelieferte Stoppwortliste (Abgleich über das Lemma). Filtertypen: "categorical" (erlaubte Werte), "minimum" (Untergrenze), "exclude_if_true" (Einträge mit Wert true ausblenden).

    Die Bedeutung einer Abfrage (Filtertypen, Stoppwörter, Zeitscheiben-Regel, Sortierung) legt das Bundle-Schema der VSV-App fest (SCHEMA.md, Abschnitt meta.neighbor_space; Kopie im Wurzelverzeichnis dieses Repos); meta.neighbor_space.format nennt die Version dieses Formats.

    Der Suchraum steht vor der Erstellung der Nachbarn, weil diese aus genau diesen Vektoren gerechnet werden (s. unten). So ergeben die mitgelieferten neighbors und eine Backend-Abfrage mit den Voreinstellungen dieselben Listen.
    """)
    return


@app.cell
def _(
    context_vectors_unit,
    context_words,
    neighbor_n_target_slices_min,
    neighbor_pos_allowed,
    np,
    pl,
    slice_grid,
    stopwords,
    target_slice_coords,
    target_slice_vectors_unit,
    target_slices,
    to_target_id,
):
    # grid in the file name: several versions can coexist
    bundle_base = f"bundestag_zeitenwende_{slice_grid}"

    # queries first, then candidates; position in the matrix = column "row" of the index
    # converted to f32 to save disk space
    space_vectors = np.vstack([target_slice_vectors_unit, context_vectors_unit]).astype(np.float32)

    _queries = target_slices.select(
        pl.lit("query").alias("role"),
        pl.col("slice"),
        to_target_id(pl.col("target_lemma")).alias("entity_id"),
        pl.col("target_lemma").alias("lemma"),
    ).with_columns(pl.Series("coords_3d", target_slice_coords.tolist()))

    # PPMI: one space shared by all slices, so the candidates carry no slice
    _candidates = context_words.select(
        pl.lit("candidate").alias("role"),
        pl.lit(None, dtype=pl.Utf8).alias("slice"),
        pl.lit(None, dtype=pl.Utf8).alias("entity_id"),
        pl.col("context_lemma").alias("lemma"),
        "coords_3d",
        "pos",
        "n_target_slices",
        "n_cooc",
        "is_speaker_name",
        "is_token_artifact",
    )

    # concatenates the index rows of the queries (target slices) and the candidates (context words)
    # diagonal_relaxed: if a table doesn't have a column, it gets null in teh concatenaded table; data types may differ
    space_index = pl.concat([_queries, _candidates], how="diagonal_relaxed").with_row_index("row")

    # checks that length of the matrix and length of its index are identical
    assert len(space_index) == len(space_vectors)

    # add metadata and define filters
    neighbor_space_meta = {
        # version of the search space format, see SCHEMA.md of the VSV app
        "format": 1,
        "vectors": f"{bundle_base}.space_vectors.npy",
        "index": f"{bundle_base}.space_index.parquet",
        "neighbor_kind": "context",
        "similarity": "cosine",
        "filters": [
            {
                "column": "pos",
                "type": "categorical",
                "label": "Wortart",
                "default": neighbor_pos_allowed,
            },
            {
                "column": "n_target_slices",
                "type": "minimum",
                "label": "Mindestbelegung",
                "description": "In wie vielen belegten Zellen (Zielwort × Zeitscheibe) das Kontextwort einen PPMI-Wert > 0 hat",
                "default": neighbor_n_target_slices_min,
                # upper bound of the value, shown as "≥ 10 von 80"
                "out_of": len(target_slices),
            },
            {
                "column": "n_cooc",
                "type": "minimum",
                "label": "Mindestzahl Kookkurrenzen",
                "description": "Wie oft das Kontextwort in diesen Zellen insgesamt vorkommt (mindestens so groß wie die Belegung)",
                "default": None,
            },
            {
                "column": "is_speaker_name",
                "type": "exclude_if_true",
                "label": "Rednernamen ausblenden",
                "default": True,
            },
            {
                "column": "is_token_artifact",
                "type": "exclude_if_true",
                "label": "Tokenisierungsreste ausblenden",
                "description": "Zahlen, Einzelzeichen und Lemmata mit Satzzeichen am Rand",
                "default": True,
            },
        ],
        # shipped list, matched against the lemma; the frontend works on a copy
        "stopwords": stopwords,
        # filter column that sizes the words in the candidate cloud
        "frequency": "n_cooc",
    }

    #save the two files
    space_files = [f"output/{bundle_base}.space_vectors.npy", f"output/{bundle_base}.space_index.parquet"]
    np.save(space_files[0], space_vectors)
    space_index.write_parquet(space_files[1])

    print(
        f"Suchraum: {space_vectors.shape[0]:,} Vektoren × {space_vectors.shape[1]} Dimensionen "
        f"({(space_index['role'] == 'query').sum()} Abfragen, "
        f"{(space_index['role'] == 'candidate').sum():,} Kandidaten), "
        f"{space_vectors.nbytes / 1e6:.1f} MB"
    )
    return (
        bundle_base,
        neighbor_space_meta,
        space_files,
        space_index,
        space_vectors,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vorgefilterte Nachbarn
    Der im vorigen Schritt erzeugte vollständige Suchraum ist die Grundlage für die nun zu erstellenden vorgefilterten Nachbarn. Diese werden im Haupt-JSON mitgeliefert, welches den Kern des Datenbundles darstellt.

    Struktur laut Schema: neighbors[entity_id][time_slice_id] = Neighbor[]
    Die Array-Reihenfolge ist der Rang, das VSV-Tool sortiert nicht um. Gerankt wird nach roher Kosinus-Distanz.

    Gerechnet wird mit den Vektoren des Suchraums (gespeichert als float32, gerechnet in float64). Bei gleichem Kosinus entscheidet die Zeile im Suchraum (row aufsteigend). Gerundet wird erst nach dem Sortieren.

    n_neighbors: Es werden nicht nur die Nachbarn für die Top-10-Liste erzeugt, sondern auch die für die Wortwolke um das Zielwort. Deswegen ist der Wert mit 150 relativ hoch.
    """)
    return


@app.cell
def _(
    neighbor_candidates,
    np,
    pl,
    space_index,
    space_vectors,
    target_slices,
    to_target_id,
):
    n_neighbors = 150

    # Search space rows: the queries come first (same order as target_slices),
    # the candidates follow in context_words order, i.e. candidate context_idx sits at
    # row n_queries + context_idx. The asserts make sure this still holds.
    _n_queries = len(target_slices)
    _query_index = space_index[:_n_queries]

    # the search space starts with the queries, one per row of target_slices
    assert (_query_index["role"] == "query").all()

    # ... in the same order as target_slices
    assert (_query_index["slice"] == target_slices["slice"]).all()

    # position of each candidate in the search space: the candidates follow the
    # queries, so context_words position context_idx sits at row _n_queries + context_idx
    neighbor_candidate_rows = neighbor_candidates["context_idx"].to_numpy().astype(np.int64) + _n_queries

    # check: the computed rows really hold the expected lemmas
    assert (
        space_index[neighbor_candidate_rows]["lemma"] == neighbor_candidates["context_lemma"]
    ).all()

    # candidates in ascending row order; with a stable sort, equal cosines then
    # keep this order (tie rule: row ascending, as in the backend)
    assert (np.diff(neighbor_candidate_rows) > 0).all()

    # selection by raw cosine, as in the backend: dot product of the normalized
    # vectors, stored as float32 but computed in float64.
    # Stable sort, so equal cosines keep row order.
    _space64 = space_vectors.astype(np.float64)
    _cosine = _space64[:_n_queries] @ _space64[neighbor_candidate_rows].T
    neighbor_top = np.argsort(-_cosine, axis=1, kind="stable")[:, :n_neighbors]
    neighbor_scores = np.take_along_axis(_cosine, neighbor_top, axis=1)

    # combining all info into a long form table
    # np.repeat repeats each row entry n_neighbors times (matching .ravel()),
    # np.tile counts 1..n_neighbors within each row.
    # The rank is needed because the following join may change the row order.
    neighbor_target_ids = target_slices.with_columns(
        to_target_id(pl.col("target_lemma")).alias("entity_id")
    )["entity_id"].to_numpy()

    neighbor_ranks = pl.DataFrame(
        {
            "entity_id": np.repeat(neighbor_target_ids, n_neighbors),
            "time_slice_id": np.repeat(target_slices["slice"].to_numpy(), n_neighbors),
            "rank": np.tile(np.arange(1, n_neighbors + 1), len(target_slices)),
            "lemma": neighbor_candidates["context_lemma"].to_numpy()[neighbor_top.ravel()],
            "score": neighbor_scores.ravel(),
        }
    )

    neighbor_ranks
    return n_neighbors, neighbor_ranks, neighbor_scores


@app.cell
def _(anchors, neighbor_ranks, pl):
    # attach anchor labels and clusters
    # If the neighbor is one of the anchors in this bundle, it gets its display
    # text (Einsatzkräft -> Einsatzkräfte) and its cluster as group. Without this
    # step the frontend would not find the landmark, because it resolves neighbors via anchorByLabel. Only the variant chosen per label counts, other variants from the list stay ordinary neighbors.
    neighbor_anchor_info = anchors.select(
        [
            pl.col("lemma"),
            pl.col("label").alias("anchor_label"),
            pl.col("cluster").alias("anchor_cluster"),
        ]
    )

    neighbor_ranks_labeled = (
        neighbor_ranks.join(neighbor_anchor_info, on="lemma", how="left")
        .with_columns(
            [
                pl.coalesce([pl.col("anchor_label"), pl.col("lemma")]).alias("label"),
                pl.coalesce([pl.col("anchor_cluster"), pl.lit("KONTEXT")]).alias(
                    "group"
                )
            ]
        )
        # the join gives no order guarantee - restore rank
        .sort(["entity_id", "time_slice_id", "rank"])
    )

    # build nested dict; this is the structure the schema demands
    # benefit: neighbours are grouped to target words and slices: easy to find
    neighbors = {}

    for _row in neighbor_ranks_labeled.iter_rows(named=True):
        _per_entity = neighbors.setdefault(_row["entity_id"], {})
        _per_slice = _per_entity.setdefault(_row["time_slice_id"], [])
        _per_slice.append(
            {
                "label": _row["label"],
                "score": round(float(_row["score"]), 4),
                "group": _row["group"],
            }
        )
    return neighbor_ranks_labeled, neighbors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Export als CSV
    Für Auswertungen außerhalb des VSV-Tools wird das neighbors-Dict zusätzlich
    als flache CSV exportiert (neighbor_mentions, eine Zeile je Nennung) sowie als
    aggregierte CSV (neighbor_summary, eine Zeile je eindeutigem Nachbarn,
    mit Median-Score über alle Zielwörter/Zeitscheiben hinweg). Die Zielwörter
    selbst (entity_id) tauchen in neighbor_summary bewusst nicht mehr auf.

    Konkret wird diese Liste verwendet, um die finale Zielwort-Liste zu generieren.
    """)
    return


@app.cell
def _(neighbors, pl):
    # neighbor_mentions: each neighbor mention individually, with reference to the
    # target word (entity_id) and slice (time_slice_id)
    neighbor_mentions = pl.DataFrame(
        [
            {
                "entity_id": entity_id,
                "time_slice_id": time_slice_id,
                "label": entry["label"],
                "group": entry["group"],
                "score": entry["score"],
            }
            for entity_id, per_entity in neighbors.items()
            for time_slice_id, entries in per_entity.items()
            for entry in entries
        ]
    )

    # activate if needed
    # neighbor_mentions.write_csv("output/neighbor_mentions.csv")

    print(f"neighbor_mentions.csv: {len(neighbor_mentions):,} Zeilen")
    return (neighbor_mentions,)


@app.cell
def _(neighbor_mentions, pl):
    # neighbor_summary: a neighbor (label) appears only once, regardless of
    # how many target words/slices it belonged to.
    # score_median summarizes all individual scores of this neighbor.
    neighbor_summary = (
        neighbor_mentions.group_by(["label", "group"])
        .agg(
            [
                pl.col("score").median().round(4).alias("score_median"),
                pl.len().alias("n_lists"),
            ]
        )
        .sort("score_median", descending=True)
    )

    # activate if needed
    # neighbor_summary.write_csv("output/neighbor_summary.csv")

    print(
        f"neighbor_summary.csv: {len(neighbor_summary):,} eindeutige Nachbarn"
    )
    return (neighbor_summary,)


@app.cell
def _(neighbor_summary, pl):
    # neighbors that appear only once or twice across all top-150 lists are
    # often based on a single, unstable cosine measurement
    neighbor_n_lists_min = 3

    neighbor_summary_stable = neighbor_summary.filter(
        pl.col("n_lists") >= neighbor_n_lists_min
    )

    # activate if needed
    # neighbor_summary_stable.write_csv("output/neighbor_summary_stable.csv")

    print(
        f"neighbor_summary_stable.csv: {len(neighbor_summary_stable):,} von "
        f"{len(neighbor_summary):,} Nachbarn (Mindestvorkommen {neighbor_n_lists_min})"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ähnlichkeitsprüfung am Listenende
    Eine Liste von n_neighbors (=150) nächsten Nachbarn garantiert nicht, dass sich die Einträge (am Ende der Liste) wirklich ähnlich sind. Deswegen wird nun an verschiedenen Stichproben geprüft, wie der Kosinus ausfällt. Es sollte sich idealerweise/normalerweise eine abfallende Kurve zeigen, die nicht bei 0 endet, sondern deutlich darüber (über dem Grundpegel).

    Erreicht die Kurve schneller den Grundpegel, sollte n_neighbors niedriger angesetzt werden.
    """)
    return


@app.cell
def _(baseline_cosines, n_neighbors, neighbor_ranks, neighbor_scores, np):
    # baseline computation
    # how similar is an arbitrary word pair?

    baseline_mean = baseline_cosines.mean()
    baseline_median = np.median(baseline_cosines)

    print(
        f"Grundpegel (beliebiges Wortpaar): "
        f"Mittel {baseline_mean:+.3f}, Median {baseline_median:+.3f}"
    )

    # cosine per rank
    neighbor_cosine_per_rank = neighbor_scores.mean(axis=0)  # column mean = per rank

    print("Roher Kosinus im Mittel, je Rang:")
    for _r in [0, 7, 24, 49, 99, n_neighbors - 1]:
        if _r < n_neighbors:
            print(f"   Rang {_r + 1:4d}: {neighbor_cosine_per_rank[_r]:+.3f}")
    print(
        f"   Einträge mit Kosinus <= {baseline_mean}: {(neighbor_scores <= baseline_mean).mean():.1%}"
    )

    print(f"Verschiedene Nachbar-Lemmata insgesamt: {neighbor_ranks['lemma'].n_unique():,}")
    return


@app.cell
def _(neighbors):
    # sample: example target word in its latest slice
    neighbors["bundeswehr"][max(neighbors["bundeswehr"])][:10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Koordinaten der Nachbarn (neighbor_coords)
    Die Aggregate definieren den Koordinatenraum der PCA. Alle Kontextwörter, also Anker wie Nachbarn, werden nur hineinprojiziert; ihre Koordinaten stehen in der Kontextwort-Tabelle (context_words).

    neighbor_coords ist das Nachschlagewerk "Label -> Position" für alle Nachbarn, die keine Anker sind. Anker bringen ihre Koordinaten selbst mit. Ein Label darf nicht zugleich Anker und Eintrag in neighbor_coords sein, auch nicht über die Anzeigeform eines Ankers.
    """)
    return


@app.cell
def _(anchors, neighbor_candidates, neighbor_ranks, pl):
    # lemmas that occur in at least one neighbor list
    neighbor_lemmas_used = neighbor_ranks.select(pl.col("lemma").alias("context_lemma")).unique()

    # neighbor_candidates comes from context_words and already carries coords_3d
    # join neighbor_lemmas_used, semi: keep only those that occur on the right
    # join anchors, anti: but no anchors
    # label: also consider corrected display labels
    bundle_neighbor_coords = (
        neighbor_candidates.join(neighbor_lemmas_used, on="context_lemma", how="semi")
        .join(anchors, left_on="context_lemma", right_on="lemma", how="anti")
        .join(
            anchors.select(pl.col("label").alias("context_lemma")),
            on="context_lemma",
            how="anti",
        )
        .select(["context_lemma", "coords_3d"])
    )
    return (bundle_neighbor_coords,)


@app.cell
def _(anchors, bundle_neighbor_coords, neighbor_ranks_labeled, pl):
    # check: does every anchor and every neighbor have a coordinate?
    # whatever remains here would be a label the frontend cannot draw. Check via label, not lemma.

    covered_labels = pl.concat(
        [
            anchors.select("label"),
            bundle_neighbor_coords.select(pl.col("context_lemma").alias("label")),
        ]
    ).unique()

    without_coords = neighbor_ranks_labeled.join(covered_labels, on="label", how="anti")[
        "label"
    ].n_unique()

    print(
        f"neighbor_coords: {len(bundle_neighbor_coords):,} Koordinaten "
        f"(aus {neighbor_ranks_labeled['lemma'].n_unique():,} verschiedenen Nachbar-Lemmata, "
        f"Rest sind Anker)"
    )
    print(f"  Nachbar-Lemmata ohne Koordinate: {without_coords}  <- muss 0 sein")

    # bundle_neighbor_coords
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metadaten und JSON-Export
    """)
    return


@app.cell
def _(
    anchors,
    bundle_aggregates,
    bundle_base,
    bundle_neighbor_coords,
    emb_meta,
    json,
    kwic_variant,
    neighbor_space_meta,
    neighbors,
    pca_explained,
    slice_grid,
    targets,
    time_slices,
):
    import os

    # determine dimensions
    bundle_vector_dim = len(bundle_aggregates["vector"][0])

    # grid label
    slice_title = {
        "quarter": "Quartale",
        "halfyear": "Halbjahre",
        "year": "Jahre",
    }[slice_grid]

    # KWIC mode from pre, passed on by emb ("window40", "punkt", "somajo")
    # kwic_variant = emb_meta["kwic_variant"]
    if kwic_variant.startswith("window"):
        window_text = f"ein ±{kwic_variant.removeprefix('window')}-Wörter-Kontextfenster"
    elif kwic_variant in ("punkt", "somajo"):
        window_text = "satzbegrenzte Kontextfenster"
    else:
        window_text = f"Kontextfenster (Modus {kwic_variant})"

    # metadata
    bundle_meta = {
        "name": f"Zeitenwende im Bundestag ({slice_title})",
        "description": (
            f"{len(targets)} Zielwörter aus dem Zeitenwende-Wortfeld über "
            f"{len(time_slices)} Zeitscheiben ({slice_title.lower()}). "
            f"Vektoren sind per PPMI-SVD über {window_text} "
            f"KONSTRUIERT, nicht trainiert. Die 3D-Ansicht zeigt "
            f"nur {pca_explained:.0%} der Struktur - belastbare Aussagen kommen "
            f"aus den Modulen, die mit vector rechnen."
        ),
        # data provenance, shown as "Quelle" in the explorer
        "source": (
            "Plenarprotokolle des Deutschen Bundestages "
            "(CPP-BT, Fobbe 2026, CC BY-SA 4.0)"
        ),
        "vector_dim": bundle_vector_dim,
        # weighting of the row vectors U·S^p after the SVD
        "eig_p": float(emb_meta["eig_p"]),
        # short name for compact displays (status bar); details in "projection"
        "method": "PPMI-SVD",
        "projection": (
            f"PPMI-SVD (k={bundle_vector_dim}, Zeilenvektoren U·S^{emb_meta['eig_p']}, "
            f"Kontextwörter Vt ungewichtet), danach PCA (3 Komponenten) über "
            f"die Aggregate; Anker und übrige Kontextwörter per "
            f"Out-of-Sample-Projektion"
        ),
        "explained_variance_ratio_3d": round(float(pca_explained), 4),
        "explained_frobenius_ratio_k": round(float(emb_meta["retained_frobenius_sq"]), 4),
        # not in the schema yet, but meta is .passthrough(); see "Suchraum für das Backend"
        "neighbor_space": neighbor_space_meta,
    }

    # round numbers
    # json cannot serialize numpy types (float() converts them),
    # and full floating point precision inflates the file without any gain.
    def round_values(values, digits=5):
        return [round(float(x), digits) for x in values]

    bundle_aggregates_json = [
        {
            "entity_id": row["entity_id"],
            "time_slice_id": row["time_slice_id"],
            "facets": {},  # currently still empty due to too little data
            "vector": round_values(row["vector"]),
            "coords_3d": round_values(row["coords_3d"], 4),
            "freq": int(row["freq"]),
        }
        for row in bundle_aggregates.iter_rows(named=True)
    ]

    bundle_anchors_json = [
        {
            "label": row["label"],
            "cluster": row["cluster"],
            "vector": round_values(row["vector"]),
            "coords_3d": round_values(row["coords_3d"], 4),
            # not in the schema, but it is .passthrough() - and for traceability
            # the original lemma is worth its weight in gold
            "lemma": row["lemma"],
        }
        for row in anchors.iter_rows(named=True)
    ]

    bundle_neighbor_coords_json = [
        {
            "label": row["context_lemma"],
            "coords_3d": round_values(row["coords_3d"], 4),
        }
        for row in bundle_neighbor_coords.iter_rows(named=True)
    ]

    # assembly
    bundle = {
        "meta": bundle_meta,
        "facets_schema": [],  # comes with the records (e.g. parliamentary group)
        "entities": targets.to_dicts(),
        "time_slices": time_slices.to_dicts(),
        "anchors": bundle_anchors_json,
        "neighbor_coords": bundle_neighbor_coords_json,
        "records": [],  # deliberately empty
        "aggregates": bundle_aggregates_json,
        "neighbors": neighbors,
    }

    bundle_path = f"output/{bundle_base}.json"

    # ensure_ascii=False: umlauts stay umlauts
    with open(bundle_path, "w", encoding="utf-8") as _f:
        json.dump(bundle, _f, ensure_ascii=False, separators=(",", ":"))

    n_neighbor_entries = sum(
        len(entries)
        for per_entity in bundle["neighbors"].values()
        for entries in per_entity.values()
    )

    print(f"{bundle_path}: {os.path.getsize(bundle_path) / 1e6:.1f} MB")
    print(f"Raster {slice_grid}, vector_dim {bundle_vector_dim}")
    print(
        f"{len(bundle['aggregates'])} aggregates, "
        f"{len(bundle['anchors'])} anchors, "
        f"{len(bundle['neighbor_coords']):,} neighbor_coords, "
        f"{n_neighbor_entries:,} Nachbar-Einträge, "
        f"{len(bundle['entities'])} entities, "
        f"{len(bundle['time_slices'])} time_slices"
    )
    return (bundle_path,)


@app.cell
def _(mo):
    copy_switch = mo.ui.switch(
        value=True,
        label="Nach backend/bundles kopieren (sonst nur output/)",
    )
    copy_switch
    return (copy_switch,)


@app.cell
def _(bundle_path, copy_switch, space_files):
    # copy to the backend
    # output/ is the pipeline's workbench, backend/bundles/ of the pdl-lt-vsv repo
    # (next to this repo) is what the app actually loads.
    # The search space files belong to the bundle and are copied along.
    import shutil
    from pathlib import Path

    if copy_switch.value:
        _backend_dir = Path("../../../pdl-lt-vsv/backend/bundles")
        for _src in [bundle_path, *space_files]:
            shutil.copy2(_src, _backend_dir / Path(_src).name)
            print(f"kopiert nach {_backend_dir / Path(_src).name}")
    else:
        print("nicht kopiert (Schalter aus) - liegt nur in output/")
    return


if __name__ == "__main__":
    app.run()
