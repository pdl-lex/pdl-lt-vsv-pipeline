import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import re
    from pathlib import Path

    # character normalization shared with the data import (soft hyphens, U+2011, ligatures, ...)
    from pipeline_pre_dataimport import normalize_chars

    return Path, mo, normalize_chars, pl, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pipeline zur Erstellung von Bundle-Daten für VSV: Preprocessing
    Dieses Notebook bietet eine vollständige, kommentierte Pipeline zur Vorbereitung der Daten für ein nachfolgendes Embedding sowie die Bundle-Generierung für das Tool "Visualisierung Semantischer Vektoren".

    * **Vorbereitung der Daten / Preprocessing (pipeline_pre.py)**
    * Embedding bzw. Vektorisierung (pipeline_emb.py)
    * Nachbearbeitung und Generierung des Bundles (pipeline_post.py)

    Als Beispiel-Datensatz werden die Reden-Texte der Bundestagssitzungen zwischen 2013 und 2026 genutzt.

    Quelle: *Fobbe, S. (2026). Corpus der Plenarprotokolle des Deutschen Bundestages (CPP-BT) (Versions 2026-01-17) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.18177196*

    Die Datei "CPP-BT_2026-01-17_DE_CSV_Reden_Gesamt.csv" muss im Ordner "stages/01_preprocessing/input" abgelegt sein. Sie ist nicht Teil dieses Git-Repos. Grundsätzlich können auch andere Input-Daten verarbeitet werden.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Projekt
    Ein Projekt ist ein Korpus mit seinen Einstellungen (Quelldateien, Spalten, Facetten, Name und Quelle für das Bundle). Die Projekt-ID steht in allen Dateinamen der Pipeline, von den KWIC-Treffern bis zum Bundle (`kwic_hits_<projekt>_<variante>.csv`, …, `<projekt>_ppmi_<variante>_<raster>.json`). So überschreiben sich die Läufe verschiedener Korpora nicht. In emb und post wird dieselbe Projekt-ID in der ersten Einstellungszelle gesetzt.

    Die Projekte sind unter „Einlesen der Daten“ definiert. Ein neues Projekt braucht dort einen eigenen Zweig und hier einen Eintrag in der Auswahl.
    """)
    return


@app.cell
def _(mo):
    project_picker = mo.ui.dropdown(
        options={
            "bundestag: Bundestagsreden (Beispiel)": "bundestag",
            "zeitenwende: Zeitungen und Zeitschriften (pipeline_pre_dataimport.py)": "zeitenwende",
        },
        value="bundestag: Bundestagsreden (Beispiel)",
        label="Projekt",
    )
    mo.style(project_picker, width="560px")
    return (project_picker,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laden von Sprachmodellen
    Die in diesem Notebook verwendete Sprachmodelle für NLTK, Stanza und spaCy liegen im Ordner "assets" im Repo-Wurzelverzeichnis. Fehlen sie, lädt das Notebook sie beim ersten Lauf herunter (Internetzugang nötig); danach läuft die Pipeline auch offline.
    """)
    return


@app.cell
def _():
    # language models for NLTK, Stanza and spaCy live in assets/ (repo root), so
    # the pipeline also runs offline. Models are downloaded only if missing.
    # Each download goes to a temp folder first, so an aborted download does
    # not leave a half-filled model folder behind.
    import io as _io
    import shutil as _shutil
    import urllib.request as _urlreq
    import zipfile as _zipfile
    from pathlib import Path as _Path

    assets_dir = _Path(__file__).resolve().parents[2] / "assets"

    # NLTK: Punkt sentence tokenizer, German (English for the example cell)
    _punkt = assets_dir / "nltk_data" / "tokenizers" / "punkt_tab"
    if not (_punkt / "german").is_dir():
        import nltk as _nltk

        _tmp = assets_dir / "_tmp_nltk"
        _shutil.rmtree(_tmp, ignore_errors=True)
        _nltk.download("punkt_tab", download_dir=str(_tmp), quiet=True, raise_on_error=True)
        _punkt.mkdir(parents=True, exist_ok=True)
        for _name in ("german", "english", "README"):
            _shutil.rmtree(_punkt / _name, ignore_errors=True)
            _shutil.move(_tmp / "tokenizers" / "punkt_tab" / _name, _punkt / _name)
        _shutil.rmtree(_tmp)

    # Stanza: German models, only the processors used in this notebook
    _stanza_dir = assets_dir / "stanza_resources"
    if not (_stanza_dir / "resources.json").is_file():
        import stanza as _stanza

        _tmp = assets_dir / "_tmp_stanza"
        _shutil.rmtree(_tmp, ignore_errors=True)
        _stanza.download(
            "de",
            model_dir=str(_tmp),
            processors="tokenize,mwt,pos,lemma",
            logging_level="WARN",
        )
        _shutil.rmtree(_stanza_dir, ignore_errors=True)
        _tmp.rename(_stanza_dir)

    # spaCy: de_core_news_md, unpacked from the official wheel (not installed as package)
    _spacy_version = "3.8.0"
    _spacy_dir = assets_dir / "spacy" / "de_core_news_md"
    if not (_spacy_dir / "config.cfg").is_file():
        _url = (
            "https://github.com/explosion/spacy-models/releases/download/"
            f"de_core_news_md-{_spacy_version}/de_core_news_md-{_spacy_version}-py3-none-any.whl"
        )
        with _urlreq.urlopen(_url) as _resp:
            _whl = _zipfile.ZipFile(_io.BytesIO(_resp.read()))
        _prefix = f"de_core_news_md/de_core_news_md-{_spacy_version}/"
        _tmp = assets_dir / "_tmp_spacy"
        _shutil.rmtree(_tmp, ignore_errors=True)
        for _m in _whl.infolist():
            if _m.filename.startswith(_prefix) and not _m.is_dir():
                _target = _tmp / _m.filename[len(_prefix):]
                _target.parent.mkdir(parents=True, exist_ok=True)
                _target.write_bytes(_whl.read(_m))
        _spacy_dir.parent.mkdir(parents=True, exist_ok=True)
        _shutil.rmtree(_spacy_dir, ignore_errors=True)
        _tmp.rename(_spacy_dir)
    return (assets_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Einlesen der Daten für das Analyse-Korpus

    Hinweis zu den Beispieldaten: Die CSV-Dateien enthalten pro Zeile eine Rede, sowie weitere Angaben zur Rede (z.B. Datum) sowie zum Redner (z.B. Fraktion).

    Die Analyse-Pipeline und das Visualisierungstool VSV sind auf Zeitungsartikel, Reden oder andere Texte kurzer bis mittlerer Länge ausgelegt. Erwartet werden die Input-Spalten:
    * ID (optional)
    * Datum
    * Text
    * Facetten 1-3 (optional)

    Bei den "Facetten" handelt es sich um weitere Dimensionen, die später zur Filterung verwendet werden können. Beispiele: Name einer Partei oder einer Publikation.

    Wenn keine ID aus den Quelldaten übernommen wird, wird eine ergänzt.

    Das Korpus ergibt sich aus dem Projekt (Abschnitt „Projekt“ oben): die Bundestagsreden als Beispiel (`bundestag`) oder die Zeitungs- und Zeitschriften-Ausgaben, die `pipeline_pre_dataimport.py` erzeugt (`zeitenwende`; eine Datei oder eine pro Jahr bzw. Monat).

    **Große Korpora:** Polars hält eine CSV-Datei beim Einlesen vollständig im Speicher, auch wenn sie stückweise abgefragt wird (gemessen: rund 3 GB für eine 2,6-GB-Datei). Für die Bundestagsreden (382 MB) ist das kein Problem, für Zehntausende Zeitschriften-Ausgaben schon. Deshalb wird das Korpus hier nur „lazy“ beschrieben und nicht geladen. Die KWIC-Extraktion liest die Dateien selbst Zeile für Zeile, mit konstantem Speicherbedarf. Alle späteren Schritte arbeiten nur noch mit den Treffern.
    """)
    return


@app.cell
def _(Path, pl, project_picker, re):
    # per project:
    # - project: project ID, part of all file names of the pipeline
    # - project_meta: name and source of the corpus, passed on to the bundle (meta.name, meta.source)
    # - corpus_files: the source CSV files
    # - source_columns: source columns for id, date and text, read row by row by the KWIC extraction;
    #   id None = no id in the source, a running number from 1 is added
    # - prepare_corpus: selection and renaming for the lazy table (preview, checks, speaker names)
    # - speaker_facet: facet with the speaker names, None = corpus without speakers
    # - bundle_facets: facets that become facet axes in the bundle (facets_schema, records);
    #   "column" = name in the lazy table and in the hits, "source" = column in the source CSV,
    #   "id"/"label" = axis in the bundle. The speaker facet is deliberately not one of them
    #   (hundreds of values; it only feeds speaker_names.csv).
    # Missing facet values (empty, "NA") get MISSING_FACET_VALUE: they stay visible as a value of
    # their own in the bundle and are never filtered out.
    MISSING_FACET_VALUE = "nicht zugeordnet"
    project = project_picker.value
    if project == "bundestag":
        project_meta = {
            "name": "Zeitenwende im Bundestag",
            "source": "Plenarprotokolle des Deutschen Bundestages (CPP-BT, Fobbe 2026, CC BY-SA 4.0)",
        }
        corpus_files = ["input/CPP-BT_2026-01-17_DE_CSV_Reden_Gesamt.csv"]
        source_columns = {"id": "rede_id", "date": "sitzung_datum", "text": "rede_text"}
        speaker_facet = "facet_2"
        bundle_facets = [{"column": "facet_1", "source": "redner_fraktion", "id": "fraktion", "label": "Fraktion"}]

        def prepare_corpus(lf):
            # define colums of interest from original CSV
            columns_of_interest = [
                "rede_id",
                "sitzung_datum",
                "rede_text",
                "redner_vorname",
                "redner_nachname",
                "redner_fraktion",
            ]
            # provide final names for further processing
            columns_renamed = {
                "rede_id": "id",
                "sitzung_datum": "date",
                "redner_fraktion": "facet_1",
                "rede_text": "text",
                "redner": "facet_2",
            }
            return (
                lf.select(columns_of_interest)
                # concatenate speaker names into a new column and drop the two source columns
                .with_columns(
                    pl.concat_str(
                        [pl.col("redner_nachname"), pl.col("redner_vorname")],
                        separator="_",
                        ignore_nulls=True,
                    ).alias("redner")
                )
                .drop("redner_vorname", "redner_nachname")
                .rename(columns_renamed)
            )

    elif project == "zeitenwende":
        # name and source still to be finalized
        project_meta = {
            "name": "Zeitenwende in der Presse",
            "source": "Zeitungen und Zeitschriften (Quelle und Lizenz ergänzen)",
        }
        # output of pipeline_pre_dataimport.py: corpus_editions.csv, or one file per year / month
        # (corpus_editions_2025.csv, corpus_editions_2025-08.csv); the report file is left out
        corpus_files = sorted(
            str(_p)
            for _p in Path("output").glob("corpus_editions*.csv")
            if re.fullmatch(r"corpus_editions(_\d{4}(-\d{2})?)?\.csv", _p.name)
        )
        source_columns = {"id": None, "date": "date", "text": "text"}
        speaker_facet = None
        bundle_facets = [{"column": "facet_1", "source": "facet_1", "id": "publication", "label": "Publikation"}]

        def prepare_corpus(lf):
            return lf.select("date", "text", "facet_1")

    else:
        raise ValueError(f"Projekt {project!r} ist nicht definiert")

    # the ID goes into file names: letters, digits, "-" only ("_" separates the parts of a file name)
    assert re.fullmatch(r"[a-z0-9-]+", project), f"Projekt-ID {project!r}: nur a-z, 0-9 und -"
    assert corpus_files and all(Path(_f).is_file() for _f in corpus_files), (
        f"Korpusdateien nicht gefunden: {corpus_files or 'keine'}"
    )
    # column names in the hits; facet columns must not collide with the columns of the hits
    facet_columns = [_f["column"] for _f in bundle_facets]
    assert len(set(facet_columns)) == len(facet_columns) and all(_c.startswith("facet_") for _c in facet_columns)
    print(f"Projekt {project}: {project_meta['name']}")
    print(f"{len(corpus_files)} Korpusdatei(en): {', '.join(corpus_files)}")
    print(f"Facetten im Bundle: {', '.join(f'{_f['label']} ({_f['column']})' for _f in bundle_facets) or 'keine'}")
    return (
        MISSING_FACET_VALUE,
        bundle_facets,
        corpus_files,
        facet_columns,
        prepare_corpus,
        project,
        project_meta,
        source_columns,
        speaker_facet,
    )


@app.cell
def _(corpus_files, pl, prepare_corpus, source_columns):
    # lazy table: nothing is loaded here; .head() only reads the beginning of the file
    corpus = pl.scan_csv(corpus_files, null_values="NA").pipe(prepare_corpus)

    # add and populate id column if not existent
    # (the KWIC extraction numbers the rows in the same way, see read_corpus_rows)
    if source_columns["id"] is None:
        corpus = corpus.with_row_index("id", offset=1)

    corpus.head(5).collect()
    return (corpus,)


@app.cell
def _(corpus, pl, source_columns):
    # assert that the columns id, date and text exist, and that ids are unqiue

    # define a set of required columns, and subtract existing ones; the resulting set should be empty
    required = {"id", "date", "text"}
    missing = required - set(corpus.collect_schema().names())
    assert not missing, f"Fehlende Spalten: {missing}"

    # only ids from the source can be duplicated, generated ones are unique by construction
    # note: this reads the whole source file with polars (see above), fine for the Bundestag data
    if source_columns["id"] is not None:
        _ids = corpus.select(pl.len().alias("rows"), pl.col("id").n_unique().alias("unique")).collect()
        assert _ids["unique"][0] == _ids["rows"][0], "Werte in 'id' sind nicht eindeutig"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Speicherung der Rednernamen
    Die Namen der Redner werden hier in eine separate Liste gespeichert, und später im Post-Processing für die Filterung verwendet.
    """)
    return


@app.cell
def _(corpus, pl, project, speaker_facet):
    # speaker names as an exact list of real persons in the corpus (more precise than NER, no extra NLP run)
    # the speaker facet has the form "Nachname_Vorname"; multi-part names ("von der Leyen") are split into single
    # tokens, because lemmas are single tokens as well
    # exported for pipeline_post.py, where speaker names are removed from neighbor and anchor candidates;
    # the matrix itself stays unfiltered
    # the facet that holds the speaker names is set per corpus above (speaker_facet)
    if speaker_facet is None:
        # corpus without speakers: empty list (header only), pipeline_post.py then removes nothing
        speaker_names = pl.DataFrame(schema={"name": pl.Utf8})
    else:
        speaker_names = (
            corpus
            .select(pl.col(speaker_facet).str.split("_").alias("name"))
            .explode("name", empty_as_null=True)
            .select(pl.col("name").str.split(" "))
            .explode("name", empty_as_null=True)
            .drop_nulls()
            .filter(pl.col("name").str.len_chars() > 1)
            .unique()
            .sort("name")
            .collect()
        )

    # per project, independent of the KWIC variant
    speaker_names_path = f"output/speaker_names_{project}.csv"
    speaker_names.write_csv(speaker_names_path)
    print(f"{len(speaker_names):,} Namensbestandteile gespeichert")
    return (speaker_names_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Definition Zielwörter
    Die Zielwörter stehen in `input/target_forms.csv`, einer Zuordnung von Formen zu Lemmata (Spalten `form;lemma;source`, UTF-8, Trennzeichen `;`). Gesucht wird nach jeder Form als ganzem Wort; das Lemma des Treffers kommt direkt aus der Liste, nicht aus dem Lemmatizer.

    Die Zuordnung ist bewusst weit: Flexionsformen, Schreibvarianten (alte Rechtschreibung, Bindestrich), Transliterationen und gegenderte Formen führen auf ein Lemma (`Kyjiw` → `Kiew`, `Belarus` → `Weißrussland`, `Soldat*innen`, `SoldatInnen` → `Soldat`). Stanza allein leistet das nicht: es löst nur die Flexion auf und macht aus gegenderten Formen eigene Lemmata (`Soldat:in`, `Soldat/-inne`).

    Formen müssen einzelne Tokens sein, weil die Suche auf Leerzeichen-Tokens arbeitet; Mehrwortausdrücke („Kalter Krieg“) können nicht treffen. Erzeugt wird die Liste mit `build_target_forms.py` aus `input/target_list_nonfinal.csv`; die Zielwörter ohne Flexionsformen stehen in `input/target_lemmas.csv`.
    """)
    return


@app.cell(disabled=True)
def _():
    # data type is an immutable set, which is faster to process than a list

    # alternate larger set
    # target_forms = {"Bundeswehr", "Wehrpflicht", "Verteidigung", "Landesverteidigung", "Drohne", "Drohnen", "Hindukusch", "Rüstung", "Rüstungsindustrie", "Sicherheit", "Frieden", "Friedenssicherung"}

    #target_forms = {
        #"Bundeswehr",
        # "Wehrpflicht",
        # "Verteidigung",
        # "Rüstung",
        # "Sicherheit",
        # "Militär",
        # "Drohne",
        # "Aggressor",

    #}
    return


@app.cell
def _(normalize_chars, re):
    # target word list: form -> lemma (input/target_forms.csv, built by build_target_forms.py)
    # columns form;lemma;source with header, UTF-8 (with or without BOM), separator ";"
    import csv

    target_form_lemmas = {}  # form -> lemma
    with open("input/target_forms.csv", encoding="utf-8-sig", newline="") as _fh:
        for _row in csv.DictReader(_fh, delimiter=";"):
            # same character normalization as the corpus, otherwise e.g. a U+2011 hyphen never matches
            _form = normalize_chars(_row["form"].strip())
            _lemma = normalize_chars(_row["lemma"].strip())
            if not _form:
                continue
            assert not re.search(r"\s", _form), f"Zielwort-Form ist kein einzelnes Token: {_form!r}"
            assert target_form_lemmas.get(_form, _lemma) == _lemma, (
                f"Form {_form!r} hat zwei Lemmata: {target_form_lemmas[_form]!r}, {_lemma!r}"
            )
            target_form_lemmas[_form] = _lemma

    # data type is an immutable set, which is faster to process than a list
    target_forms = frozenset(target_form_lemmas)
    print(f"{len(target_forms):,} Formen, {len(set(target_form_lemmas.values())):,} Zielwort-Lemmata")
    return target_form_lemmas, target_forms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KWIC-Extraktion
    Für die Lokalisierung der Zielwörter im Text und Ausgabe eines Kontextfensters stehen zwei Modi zur Verfügung:
    * Das Kontextfenster wird auf den Satz begrenzt.
    * Das Kontextfenster wird auf eine wählbare Anzahl an Wörtern vor und nach dem Suchtreffer begrenzt.

    Für den Satzgrenzen-Modus stehen ggw. die Optionen NLTK und SoMaJo zur Verfügung.
    Ein Qualitätsvergleich steht noch aus; SoMaJo ist deutlich langsamer.
    Das SoMaJo-Modell arbeitet regelbasiert und ist für deutsche Texte optimiert. Das verwendete CMC-Modell (computer-mediated communication) ist eigentlich vorrangig für Social Media-Texte optimiert.
    """)
    return


@app.cell
def _(mo):
    # define widget elements
    kwic_mode = mo.ui.radio(
        options={
            "Festes Fenster": "window",
            "Satzgrenzen (Punkt)": "punkt",
            "Satzgrenzen (SoMaJo)": "somajo",
        },
        value="Satzgrenzen (Punkt)",
        label="KWIC-Modus",
        inline=True,
    )
    window_size = mo.ui.slider(
        10, 100, value=20, step=5, label="Fenster (Wörter)", show_value=True
    )
    run = mo.ui.run_button(label="Extraktion starten")

    def kwic_control_main(mode):
        return mo.hstack(
            [kwic_mode, window_size if mode == "window" else "", run],
            justify="start",
            gap=2,
        )

    def kwic_control_mode(mode):
        return mo.hstack([kwic_mode, window_size if mode == "window" else ""])

    return kwic_control_main, kwic_control_mode, kwic_mode, run, window_size


@app.cell
def _(kwic_control_main, kwic_mode):
    kwic_control_main(kwic_mode.value)
    return


@app.cell
def _(kwic_mode, window_size):
    match kwic_mode.value:
        case "window":
            kwic_variant = f"{kwic_mode.value}{window_size.value}"
        case "punkt":
            kwic_variant = kwic_mode.value
        case "somajo":
            kwic_variant = kwic_mode.value
        case _:
            kwic_variant = "unknown"

    print(f"Filename extension set to: {kwic_variant}")
    return (kwic_variant,)


@app.cell
def _(Path, kwic_variant, project):
    # all results of a run go to output/<kwic_variant>/ (one folder per method), and are read from
    # there; run_id = <project>_<kwic_variant> is part of every file name
    # (the slice grid is added where the file depends on it)
    out_dir = Path(f"output/{kwic_variant}")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{project}_{kwic_variant}"
    print(f"Ausgabeordner {out_dir}, Dateinamen *_{run_id}*")
    return out_dir, run_id


@app.cell
def _(re):
    # removes punctuation
    def strip_punctuation(word):
        return re.sub(r"^\W+|\W+$", "", word)

    # print(strip_punctuation("#Beispiel,"))
    return (strip_punctuation,)


@app.cell
def _(strip_punctuation, target_forms, window_size):
    # window extraction function
    def kwic_window(doc_id, text, window=window_size):
        words = text.split()
        hits = []
        for i, word_raw in enumerate(words):
            word = strip_punctuation(word_raw)
            if word in target_forms:
                hits.append(
                    {
                        "id": doc_id,
                        "target_form": word,
                        "position": i,
                        "context_before": " ".join(words[max(0, i - window) : i]),
                        "context_after": " ".join(words[i + 1 : i + 1 + window]),
                        "sentence_based": False,
                    }
                )
        return hits

    return (kwic_window,)


@app.cell
def _(assets_dir, re, strip_punctuation, target_forms):
    # sentence extraction function, with NLTK and SoMaJo
    import nltk
    from nltk.tokenize import PunktTokenizer

    # model data comes from assets/ (offline), no nltk.download()
    nltk.data.path.insert(0, str(assets_dir / "nltk_data"))
    from somajo import SoMaJo

    punkt = PunktTokenizer("german")  # load German model
    somajo = SoMaJo(
        "de_CMC", split_camel_case=False, character_offsets=True
    )  # load German CMC model (computer-mediated communication)
    from bisect import bisect_left, bisect_right

    TOKEN = re.compile(r"\S+")  # token = any sequence of non-whitespace characters
    collapse_whitespace = lambda s: re.sub(r"\s+", " ", s).strip()  # whitespace normalization

    # both following functions return start and end positions of detected sentences
    def punkt_spans(snip):
        return punkt.span_tokenize(snip)

    def somajo_spans(snip):
        for sent in somajo.tokenize_text([snip]):
            yield sent[0].character_offset[0], sent[-1].character_offset[1]

    def kwic_sentence(doc_id, text, spans=punkt_spans, pad=600, max_words=20):
        """
        Extract keyword-in-context (KWIC) entries for all target words in a text.

        For each whitespace-separated token whose normalized form is in `target_forms`,
        the surrounding sentence is determined with the given segmenter. The context is
        that sentence, clipped to at most `max_words` whitespace tokens on each side of
        the hit (DNB requirement: max. 20 words before and after the hit). Long sentences
        are therefore clipped, not discarded. A sentence border that only results from
        the border of the working window is ignored on that side (the clip applies).
        If the segmenter yields no sentence containing the hit, the fallback is
        `max_words` tokens on each side without sentence borders.
        Only whole tokens are taken; tokens crossing a sentence border are left out.

        Parameters
        ----------
        id : str
            Document identifier, passed through unchanged to every hit.
        text : str
            Full document text (never modified; all positions refer to it).
        spans : callable, default punkt_spans
            Sentence segmenter: takes a string and yields (start, end) character
            positions of sentences. Either `punkt_spans` or `somajo_spans`.
        pad : int, default 600
            Half-width (in characters) of the working window around a hit that is
            passed to the segmenter, instead of segmenting the whole text.
        max_words : int, default 20
            Maximum number of whitespace tokens on each side of the hit.

        Returns
        -------
        list of dict
            One dict per hit with the keys:
            - "id": document identifier
            - "target_form": normalized form of the hit
            - "position": token index in the document
            - "context_before": cleaned text before the hit (hit excluded)
            - "context_after": cleaned text after the hit (hit excluded)
            - "sentence_based": True if a sentence was found (possibly clipped),
              False if fallback

        Notes
        -----
        Requires the globals `target_forms` (set of target words) and `strip_punctuation()`.
        """
        toks = [m.span() for m in TOKEN.finditer(text)]  # (start, end) of all whitespace tokens
        starts = [s for s, _ in toks]
        ends = [e for _, e in toks]
        hits = []
        for i, (hs, he) in enumerate(toks):  # i = token index, hs/he = start/end of the token
            word = strip_punctuation(text[hs:he])  # remove punctuation at the edges
            if word not in target_forms:  # check if found word is one of the target forms
                continue
            lo, hi = (
                max(0, hs - pad),
                min(len(text), he + pad),
            )  # define a segmentation window around current word
            snip = text[lo:hi]  # save text in window to snip
            a, b = 0, len(text)  # sentence borders; default: none (fallback)
            sentence_based = False
            for s, e in spans(
                snip
            ):  # iterate over all sentences in current window until a match is found
                if s <= hs - lo < e:  # position of the found word must be within bounds
                    sentence_based = True
                    # a sentence border that coincides with the window border (and not the text border)
                    # is an artefact of the window: ignore it, the word limit applies on that side
                    if not (s == 0 and lo > 0):
                        a = lo + s
                    if not (e == len(snip) and hi < len(text)):
                        b = lo + e
                    break
            # whole tokens inside [a, b], at most max_words on each side of the hit
            j0 = max(bisect_left(starts, a), i - max_words)  # first context token before the hit
            j1 = min(bisect_right(ends, b) - 1, i + max_words)  # last context token after the hit
            hits.append(
                {
                    "id": doc_id,
                    "target_form": word,
                    "position": i,
                    "context_before": collapse_whitespace(text[starts[j0]:hs]) if j0 < i else "",
                    "context_after": collapse_whitespace(text[he:ends[j1]]) if j1 > i else "",
                    "sentence_based": sentence_based,
                }
            )
        return hits

    return kwic_sentence, nltk, punkt_spans, somajo_spans


@app.cell
def _(nltk):
    # Example for the NLTK punkt tokenizer, which learns via ML to distinguish between end of sentences and other "punkt" liek in abbreviations.
    sample_text = "Hello Mr. Smith! The temperature is 98.6 degrees. Are you ready?"
    sample_sentences = nltk.sent_tokenize(sample_text)

    print(sample_sentences)
    return


@app.cell
def _(
    kwic_mode,
    kwic_sentence,
    kwic_window,
    punkt_spans,
    somajo_spans,
    window_size,
):
    # extraction function as determined with radio buttons above
    if kwic_mode.value == "punkt":
        kwic_extract = lambda doc_id, text: kwic_sentence(doc_id, text, spans=punkt_spans)
    elif kwic_mode.value == "somajo":
        kwic_extract = lambda doc_id, text: kwic_sentence(doc_id, text, spans=somajo_spans)
    else:
        kwic_extract = lambda doc_id, text: kwic_window(doc_id, text, window=window_size.value)
    return (kwic_extract,)


@app.function
def read_corpus_rows(files, id_col, date_col, text_col, on_progress=None, normalize=None, facet_cols=()):
    """
    Yield (id, date, text, facets) for every row of the corpus CSV files, one row at a time.

    facets is a tuple with the values of the source columns facet_cols, in that order (None
    if missing); empty tuple without facet_cols.

    normalize(text) is applied to every non-empty text before it is yielded (character
    normalization, see normalize_chars in pipeline_pre_dataimport.py).

    Reads with the csv module instead of polars, so memory stays constant whatever the file
    size (polars holds a whole CSV file in memory while reading it, even in batches).
    Without id column (id_col None), rows are numbered from 1 across all files, in the same
    order as with_row_index("id", offset=1) on the lazy corpus. Empty fields and "NA" count
    as missing, as with null_values="NA" in polars.
    on_progress(n_bytes) is called with the number of bytes read since the last call.
    """
    import csv
    import io

    csv.field_size_limit(2**31 - 1)  # texts of whole editions exceed the default limit of 128 KB
    missing = {"", "NA"}
    next_id = 1
    for path in files:
        with open(path, "rb") as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            header = next(reader)
            i_date, i_text = header.index(date_col), header.index(text_col)
            i_id = header.index(id_col) if id_col else None
            i_facets = [header.index(c) for c in facet_cols]
            position = 0
            for row in reader:
                if i_id is None:
                    doc_id, next_id = next_id, next_id + 1
                else:
                    doc_id = None if row[i_id] in missing else row[i_id]
                date = None if row[i_date] in missing else row[i_date]
                text = None if row[i_text] in missing else row[i_text]
                facets = tuple(None if row[i] in missing else row[i] for i in i_facets)
                if text and normalize:
                    text = normalize(text)
                yield doc_id, date, text, facets
                if on_progress:
                    # position of the underlying byte stream; advances in blocks of the read buffer
                    new_position = raw.tell()
                    if new_position != position:
                        on_progress(new_position - position)
                        position = new_position


@app.function
def write_kwic_hits(rows, extract, out_path, batch_size=100_000, facet_names=()):
    """
    Run the KWIC extraction over (id, date, text, facets) rows and write all hits to one CSV file.

    The hits are written in batches of `batch_size`, so only one batch is held in memory.
    Each hit also gets the date and the facet values of its document (columns facet_names, in
    the order of the facets tuple); later steps then need no join with the corpus.
    The file is first written as *.part and renamed when complete, so an aborted run leaves no
    truncated CSV under the final name. Returns (number of documents, number of hits).
    """
    from pathlib import Path

    import polars as pl

    out_path = Path(out_path)
    part_path = out_path.with_name(out_path.name + ".part")
    n_docs, n_hits, hits, header = 0, 0, [], True
    with part_path.open("wb") as fh:
        for doc_id, date, text, facets in rows:
            n_docs += 1
            if not text:
                continue
            for hit in extract(doc_id, text):
                hit["date"] = date
                hit.update(zip(facet_names, facets))
                hits.append(hit)
            if len(hits) >= batch_size:
                pl.DataFrame(hits).write_csv(fh, include_header=header)
                n_hits, hits, header = n_hits + len(hits), [], False
        if hits:
            pl.DataFrame(hits).write_csv(fh, include_header=header)
            n_hits, header = n_hits + len(hits), False
        if header:  # no hits at all: header only
            columns = ["id", "target_form", "position", "context_before", "context_after", "sentence_based", "date"]
            fh.write((",".join([*columns, *facet_names]) + "\n").encode("utf-8"))
    part_path.replace(out_path)
    return n_docs, n_hits


@app.cell
def _(
    Path,
    bundle_facets,
    corpus_files,
    kwic_extract,
    kwic_variant,
    mo,
    normalize_chars,
    out_dir,
    pl,
    run,
    run_id,
    source_columns,
):
    # this cell only runs when the "Extraktion starten" button above is hit
    mo.stop(
        not run.value, mo.md("Einstellungen wählen und **Extraktion starten** klicken.")
    )

    # the corpus files are read row by row (read_corpus_rows), the hits written in batches
    # (write_kwic_hits); the progress bar counts megabytes of the corpus files
    _total_mb = sum(Path(_f).stat().st_size for _f in corpus_files) // 1_000_000 + 1
    _out = out_dir / f"kwic_hits_{run_id}.csv"
    with mo.status.progress_bar(total=_total_mb, title=f"KWIC-Extraktion ({kwic_variant}), MB") as _bar:
        _read = {"bytes": 0, "mb": 0}

        def _progress(n_bytes):
            _read["bytes"] += n_bytes
            _mb = min(_read["bytes"] // 1_000_000, _total_mb)
            if _mb > _read["mb"]:
                _bar.update(increment=_mb - _read["mb"])
                _read["mb"] = _mb

        _rows = read_corpus_rows(
            corpus_files,
            source_columns["id"],
            source_columns["date"],
            source_columns["text"],
            on_progress=_progress,
            # soft hyphens, non-breaking hyphens, invisible characters etc.; the Bundestag CSV
            # does not pass through pipeline_pre_dataimport.py, so the same step is applied here
            normalize=normalize_chars,
            # facet values of the document (e.g. parliamentary group, publication), written with every hit
            facet_cols=[_f["source"] for _f in bundle_facets],
        )
        # same file as read below (kwic_variant includes the window size); the cell below has to be
        # run again afterwards, it does not depend on this one
        _n_docs, _n_hits = write_kwic_hits(
            _rows, kwic_extract, _out, facet_names=[_f["column"] for _f in bundle_facets]
        )

    print(f"{_n_docs:,} Dokumente, {_n_hits:,} Treffer, gespeichert: {_out}")
    pl.read_csv(_out, n_rows=100, null_values="NA")
    return


@app.cell
def _(kwic_control_mode, kwic_mode):
    kwic_control_mode(kwic_mode.value)
    return


@app.cell
def _(out_dir, run_id):
    print(f"Will load the following pregenerated list: '{out_dir}/kwic_hits_{run_id}.csv'")
    return


@app.cell
def _(out_dir, pl, run_id):
    # Read pre-generated list (written by the extraction above, or by an earlier run)
    # default modes are "punkt", "somajo" and "window20"
    # Use control above to switch

    kwic_hits = pl.read_csv(out_dir / f"kwic_hits_{run_id}.csv", null_values="NA")
    print(f"Geladen: kwic_hits_{run_id}.csv")
    kwic_hits
    return (kwic_hits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lemmatisierung und POS-Bestimmung

    Die Text- bzw. Kontextwörter liegen noch in ihrer regulären, das heisst gemäß Grammatik flektierten Form vor. Um sie später vergleichen zu können, müssen sie auf ihre einheitliche Grundform (z.B. "Drohnen" > "Drohne") zurückgeführt werden. Im gleichen Zuge wird auch die Wortart mit bestimmt - diese kann später zur Filterung verwendet werden. Das Lemma der Zielwörter kommt nicht vom Lemmatizer, sondern aus der Zielwortliste (siehe „Definition Zielwörter“).

    Ggw. stehen zwei Lemmatizer zur Verfügung: spaCy und Stanza, wobei Stanza innerhalb der Pipeline von spaCy genutzt wird.
    Zwischenstand: spaCy ist schneller, Stanza besser.
    """)
    return


@app.cell(disabled=True)
def _(assets_dir):
    import spacy

    nlp = spacy.load(assets_dir / "spacy" / "de_core_news_md")
    return (nlp,)


@app.cell(disabled=True, hide_code=True)
def _(nlp, target_forms):
    # deactivated, because Stanza is preferred
    # feeds the sorted target_forms to the spacy nlp pipeline and returns the first token of the returned document as lemma.
    # Only works for single words, not for "Bundeswehr-Einsatz"

    target_lemmas_spacy = [doc[0].lemma_ for doc in nlp.pipe(sorted(target_forms))]

    ## zips original target forms and generated lemmas together
    target_lemma_map_spacy = dict(zip(sorted(target_forms), target_lemmas_spacy))
    target_lemma_map_spacy
    return (target_lemma_map_spacy,)


@app.cell(hide_code=True)
def _(kwic_hits, mo, nlp, out_dir, pl, run_id, target_lemma_map_spacy):
    # deactivated because Stanza is preferred
    # deactivated by default because it requires several minutes to run. instead, a pre-calculated list is loaded below.

    # join contexts before and after into a list
    contexts = (
        kwic_hits["context_before"] + " " + kwic_hits["context_after"]
    ).to_list()

    context_lemmas = []
    context_pos_tags = []

    # feeds all contexts to the spaCy pipeline, wrapped in a status bar
    # unwanted functionality of spaCy is disabled to increase speed
    # per context, returns a list of lemmas and a list of POS tags
    with mo.status.progress_bar(
        total=len(contexts), title="Lemmatisierung Kontext"
    ) as bar:
        for doc2 in nlp.pipe(contexts, disable=["parser", "ner", "senter"]):
            context_lemmas.append([t.lemma_ for t in doc2 if t.pos_ != "PUNCT"])
            context_pos_tags.append([t.pos_ for t in doc2 if t.pos_ != "PUNCT"])
            bar.update()

    # creates a new df with the lemmatized contexts and POS
    kwic_hits_lemmatized_spacy = kwic_hits.with_columns(
        [
            pl.Series("context_lemmas", context_lemmas),
            pl.Series("context_pos_tags", context_pos_tags),
        ]
    )

    # adds the target lemma, mapped from the target form via the dict generated above
    kwic_hits_lemmatized_spacy = kwic_hits_lemmatized_spacy.with_columns(
        pl.col("target_form").replace(target_lemma_map_spacy).alias("target_lemma")
    )

    # store results as parquet file
    kwic_hits_lemmatized_spacy.write_parquet(out_dir / f"kwic_hits_lemmatized_spacy_{run_id}.parquet")

    kwic_hits_lemmatized_spacy
    return


@app.cell
def _(assets_dir, target_form_lemmas, target_forms):
    import stanza

    # target lemma per form: from the target word list (target_forms.csv)
    # Before, Stanza lemmatized every form on its own (one-token sentence, no context). Tested on the
    # 2,615 forms of the list (2026-11-17): only 472 of 608 groups got a single lemma. Stanza resolves
    # inflection (Kriegen -> Krieg), but keeps gendered forms as own lemmas (Soldat*in, Soldat:in, broken
    # ones like Soldat/-inne), and does not merge spelling variants or transliterations (Kiew/Kyjiw,
    # Weißrussland -> "Weißrußland"). Kept as switch, default off.
    TARGET_LEMMAS_FROM_STANZA = False

    if TARGET_LEMMAS_FROM_STANZA:
        nlp_st = stanza.Pipeline(
            "de",
            processors="tokenize,pos,lemma",
            tokenize_pretokenized=True,
            dir=str(assets_dir / "stanza_resources"),
            download_method=None,  # offline: use only the models in assets/stanza_resources
            logging_level="WARN",
        )
        _words = sorted(target_forms)
        _doc = nlp_st([[_w] for _w in _words])  # every word = one "sentence" of one token
        target_lemma_map = dict(zip(_words, (_s.words[0].lemma for _s in _doc.sentences)))
    else:
        target_lemma_map = dict(target_form_lemmas)
    return stanza, target_lemma_map


@app.function
def make_context_mapper(form_lemmas, gender_rule=True):
    """
    Return a function that maps the words of a context text before it goes to Stanza.

    The returned function takes a text and returns (new_text, spans). Every whitespace token
    is split like strip_punctuation() into punctuation at the edges and a core:
    - core is a form in `form_lemmas` (the target word list): replaced by its lemma, and the
      lemma is fixed, i.e. it is used as is instead of Stanza's lemma ("Kyjiws" -> "Kiew",
      "Soldat*innen," -> "Soldat,"). spans lists (start, end, lemma) of these replacements in
      new_text.
    - gender_rule, core not in the list: a gendered form is written as the plain feminine
      form, which Stanza lemmatizes cleanly ("Bürger*innen", "Bürger:innen", "BürgerInnen" ->
      "Bürgerinnen" -> lemma "Bürgerin"; "Abgeordnete:r" -> "Abgeordnete"). Covered are the
      signs * : _ / /- · and the Binnen-I. The feminine lemma is then reduced to the masculine
      base form after Stanza, with the vocabulary of the whole run (feminine_base_map), so all
      ways of writing end up at "Bürger"; the stem alone cannot be used for that ("Kolleg*innen"
      -> "Kolleg", but the base form is "Kollege").

    Why before Stanza: Stanza's tokenizer splits gendered forms ("Soldat*innen" -> "Soldat",
    "*", "innen"; "Soldat:innen" -> "Soldat", ":innen"), so after Stanza they cannot be mapped,
    and the pieces end up as context words of their own.
    """
    import re

    token = re.compile(r"\S+")
    edges = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)  # same split as strip_punctuation()
    gender_sign = re.compile(r"^(\w*[a-zäöüß])(?:/-|[*:_/·])(in|innen)$")
    binnen_i = re.compile(r"^([A-ZÄÖÜ]\w*[a-zäöüß])(In|Innen)$")
    adj_sign = re.compile(r"^(\w*[a-zäöüß]e)[*:_/·]r$")

    def lookup(core):
        # -> (replacement, fixed lemma or None), or None if the core stays as it is
        if core in form_lemmas:
            return form_lemmas[core], form_lemmas[core]
        if gender_rule:
            for rule in (gender_sign, binnen_i):
                m = rule.match(core)
                if m:
                    return m[1] + m[2].lower(), None  # Bürger + innen
            m = adj_sign.match(core)
            if m:
                return m[1], None  # Abgeordnete
        return None

    def map_text(text):
        out, spans, pos, n = [], [], 0, 0  # n = length of the output so far
        for m in token.finditer(text):
            out.append(text[pos : m.start()])
            n += m.start() - pos
            pos = m.end()
            lead, core, trail = edges.match(m.group()).groups()
            found = lookup(core) if core else None
            if found is None:
                out.append(m.group())
                n += len(m.group())
                continue
            new, fixed = found
            out += [lead, new, trail]
            if fixed is not None:
                spans.append((n + len(lead), n + len(lead) + len(new), fixed))
            n += len(lead) + len(new) + len(trail)
        out.append(text[pos:])
        return "".join(out), spans

    return map_text


@app.function
def lemmatize_batch(batch, nlp, target_lemma_map, context_mapper=None, chunk=500, on_progress=None):
    """
    Lemmatize the contexts of a batch of KWIC hits with Stanza; returns the batch with
    the columns target_lemma, context_lemmas, context_pos_tags and context_gender added
    (context_gender: Stanza's grammatical gender Masc/Fem/Neut, null if none; used by
    feminine_base_map).

    Hit and contexts are lemmatized together as one text, so that Stanza sees the whole
    sentence; afterwards each word is assigned to "before" or "after" by its character
    position, the hit itself is left out. Punctuation is skipped.

    context_mapper (see make_context_mapper) maps the context words before Stanza; words
    it replaces with a fixed lemma get that lemma, even if Stanza splits them into several
    tokens. None = contexts go to Stanza unchanged.
    """
    import polars as pl
    import stanza
    from bisect import bisect_right

    # define the various parts
    # the 'b or ""' replaces null values with empty strings
    # texts recombines the three parts
    contexts_before = [(b or "").strip() for b in batch["context_before"].to_list()]
    target_hit_forms = [(t or "").strip() for t in batch["target_form"].to_list()]
    contexts_after = [(a or "").strip() for a in batch["context_after"].to_list()]

    # map the context words (target word list, gendered forms); fixed_spans holds per text the
    # (start, end, lemma) of replaced words, as positions in the combined text
    fixed_spans = [[] for _ in contexts_before]
    if context_mapper is not None:
        for i, (b, t, a) in enumerate(zip(contexts_before, target_hit_forms, contexts_after)):
            b, spans_b = context_mapper(b)
            a, spans_a = context_mapper(a)
            offset = len(b) + 1 + len(t) + 1  # start of context_after in the combined text
            contexts_before[i], contexts_after[i] = b, a
            fixed_spans[i] = spans_b + [(s + offset, e + offset, lem) for s, e, lem in spans_a]
    texts = [f"{b} {t} {a}" for b, t, a in zip(contexts_before, target_hit_forms, contexts_after)]

    # initialize empty result lists
    before_lemmas, before_pos_tags, before_genders = [], [], []
    after_lemmas, after_pos_tags, after_genders = [], [], []

    def gender(word):
        # grammatical gender from Stanza's features ("Case=Nom|Gender=Fem|Number=Sing"), or None
        for feat in (word.feats or "").split("|"):
            if feat.startswith("Gender="):
                return feat[7:]
        return None

    # inner loop: feed `chunk` texts to stanza at once. idx denotes line numbers in the batch
    for start in range(0, len(texts), chunk):
        idx = range(start, min(start + chunk, len(texts)))
        docs = nlp([stanza.Document([], text=texts[i]) for i in idx])

        # for every document, calculate the target word position
        for i, doc in zip(idx, docs):
            t_start = len(contexts_before[i]) + 1  # starts at context_before +1
            t_end = t_start + len(target_hit_forms[i])  # ends at context_before +1 + len
            bl, bp, bg, al, ap, ag = [], [], [], [], [], []
            spans = fixed_spans[i]
            span_starts = [s for s, _, _ in spans]
            spans_done = set()

            # process tokens in the document
            for sent in doc.sentences:
                for tok in sent.tokens:  # tokens are surface forms (e.g. "zum")
                    # word with a fixed lemma from the context mapping: the first token inside its
                    # span gets the lemma, further tokens of the same span are skipped
                    k = bisect_right(span_starts, tok.start_char) - 1
                    if k >= 0 and tok.start_char < spans[k][1]:
                        if k in spans_done:
                            continue
                        spans_done.add(k)
                        w0 = next((w for w in tok.words if w.upos != "PUNCT"), None)
                        words = [(spans[k][2], w0.upos if w0 else "X", gender(w0) if w0 else None)]
                    else:
                        # words are syntactical units (e.g. "zu" + "dem"); punctuation is skipped;
                        # if no lemma is available, use original text
                        words = [
                            (w.lemma or w.text, w.upos, gender(w)) for w in tok.words if w.upos != "PUNCT"
                        ]
                    for lemma, upos, gen in words:
                        if tok.end_char <= t_start:  # determine whether it belongs to before or after
                            bl.append(lemma)
                            bp.append(upos)
                            bg.append(gen)
                        elif tok.start_char >= t_end:
                            al.append(lemma)
                            ap.append(upos)
                            ag.append(gen)
                        # else: belongs to target word => ignore

            # append to lists
            before_lemmas.append(bl)
            before_pos_tags.append(bp)
            before_genders.append(bg)
            after_lemmas.append(al)
            after_pos_tags.append(ap)
            after_genders.append(ag)

        if on_progress:
            on_progress(len(idx))

    # list type given explicitly: a batch with only empty contexts would otherwise get List(Null)
    # and no longer match the other batches
    list_type = pl.List(pl.Utf8)
    return (
        # adds the lemmatized contexts and POS, split into before and after
        batch.with_columns(
            pl.Series("context_before_lemmas", before_lemmas, dtype=list_type),
            pl.Series("context_before_pos_tags", before_pos_tags, dtype=list_type),
            pl.Series("context_after_lemmas", after_lemmas, dtype=list_type),
            pl.Series("context_after_pos_tags", after_pos_tags, dtype=list_type),
            pl.Series("context_before_genders", before_genders, dtype=list_type),
            pl.Series("context_after_genders", after_genders, dtype=list_type),
        )
        # adds the target lemma, mapped from the target form via the dict generated above
        .with_columns(pl.col("target_form").replace(target_lemma_map).alias("target_lemma"))
        # we processed context_before and context_after separately, assuming this could be of use later
        # for the moment, we only need the joint columns, so before and after get concatenated here
        .with_columns(
            pl.concat_list("context_before_lemmas", "context_after_lemmas").alias("context_lemmas"),
            pl.concat_list("context_before_pos_tags", "context_after_pos_tags").alias("context_pos_tags"),
            pl.concat_list("context_before_genders", "context_after_genders").alias("context_gender"),
        )
        # also drop the separate columns
        # this way, the data structure is identical between spacy and stanza files
        # (except context_gender, which only Stanza writes)
        .drop(
            "context_before_lemmas",
            "context_before_pos_tags",
            "context_after_lemmas",
            "context_after_pos_tags",
            "context_before_genders",
            "context_after_genders",
        )
    )


@app.function
def lemmatize_kwic_hits(
    kwic_hits,
    nlp,
    target_lemma_map,
    out_path,
    context_mapper=None,
    settings=None,
    batch_size=20_000,
    on_progress=None,
):
    """
    Lemmatize all KWIC hits batch by batch and write the result to one parquet file.

    Every batch of `batch_size` hits is written as its own part file to <out_path stem>_parts/,
    so only one batch of lemma lists is held in memory. If a run is aborted, the next run
    skips the parts already written. The parts belong to exactly one input: a fingerprint
    of kwic_hits, the batch size, the target lemma map and `settings` (a JSON-serializable
    dict describing context_mapper) is stored with them, and if it differs, the parts are
    deleted and recomputed. When all parts are done, they are combined into out_path
    (streaming) and deleted. Returns the number of batches that were computed in this run.
    """
    import hashlib
    import json
    import shutil
    from pathlib import Path

    import polars as pl

    out_path = Path(out_path)
    parts_dir = out_path.with_name(out_path.stem + "_parts")
    meta_path = parts_dir / "meta.json"
    fingerprint = {
        "rows": kwic_hits.height,
        "batch_size": batch_size,
        "hash": hashlib.sha1(kwic_hits.hash_rows(seed=0).to_numpy().tobytes()).hexdigest(),
        "target_lemma_map": hashlib.sha1(
            json.dumps(sorted(target_lemma_map.items()), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "settings": settings or {},
    }
    if parts_dir.exists():
        try:
            same_input = json.loads(meta_path.read_text(encoding="utf-8")) == fingerprint
        except (OSError, ValueError):
            same_input = False
        if not same_input:
            shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(fingerprint), encoding="utf-8")

    parts, n_computed = [], 0
    for number, start in enumerate(range(0, kwic_hits.height, batch_size)):
        part = parts_dir / f"part_{number:06d}.parquet"
        parts.append(part)
        batch = kwic_hits.slice(start, batch_size)
        if part.exists():  # done in an earlier, aborted run
            if on_progress:
                on_progress(batch.height)
            continue
        lemmatized = lemmatize_batch(
            batch, nlp, target_lemma_map, context_mapper=context_mapper, on_progress=on_progress
        )
        tmp = part.with_name(part.stem + ".tmp.parquet")
        lemmatized.write_parquet(tmp)
        tmp.replace(part)
        n_computed += 1

    # combine the parts in their order, without loading them all at once
    tmp = out_path.with_name(out_path.stem + ".tmp.parquet")
    if parts:
        pl.scan_parquet(parts).sink_parquet(tmp)
    else:  # no hits: empty file with the input columns plus the new ones
        empty = pl.List(pl.Utf8)
        kwic_hits.with_columns(
            pl.lit(None, pl.Utf8).alias("target_lemma"),
            pl.lit(None, empty).alias("context_lemmas"),
            pl.lit(None, empty).alias("context_pos_tags"),
            pl.lit(None, empty).alias("context_gender"),
        ).write_parquet(tmp)
    tmp.replace(out_path)
    shutil.rmtree(parts_dir)
    return n_computed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Die Lemmatisierung des Kontexts läuft in Stapeln zu 20.000 Treffern, jeder Stapel wird sofort als Teildatei gespeichert (`output/<variante>/kwic_hits_lemmatized_stanza_<projekt>_<variante>_parts/`). Wird der Lauf abgebrochen, setzt der nächste Lauf beim ersten fehlenden Stapel fort. Das gilt nur, solange die KWIC-Treffer, die Zielwortliste und die Mapping-Einstellungen dieselben sind, sonst wird neu begonnen. Am Ende werden die Teile zu einer Datei zusammengeführt und gelöscht.

    **Mapping der Kontextwörter** (vor Stanza, beide Schalter standardmäßig an):
    * *Zielwortliste auf Kontextwörter anwenden*: Kontextwörter, die als Form in der Zielwortliste stehen, bekommen deren Lemma (`Kyjiw` → `Kiew`, `Soldat*innen` → `Soldat`). So ist ein Wort als Kontext dieselbe Dimension, unter welcher Schreibung es auch vorkommt.
    * *Gegenderte Formen außerhalb der Liste vereinheitlichen*: Formen mit Genderzeichen (`* : _ / /- ·`) und Binnen-I werden als einfache feminine Form geschrieben, bevor Stanza sie lemmatisiert (`Bürger*innen`, `BürgerInnen` → `Bürgerinnen`, Lemma `Bürgerin`; `Abgeordnete:r` → `Abgeordnete`). Auf die Grundform `Bürger` führt sie der Schritt „Movierte Feminina“ weiter unten.

    Das Mapping muss vor Stanza stattfinden, weil der Stanza-Tokenizer gegenderte Formen zerlegt (`Soldat*innen` → `Soldat` `*` `innen`, `Soldat:innen` → `Soldat` `:innen`); danach wären nur noch Bruchstücke zu sehen, die als eigene Kontextwörter in die Matrix eingingen. Die KWIC-Treffer selbst (`kwic_hits_*.csv`) bleiben unverändert.
    """)
    return


@app.cell
def _(mo):
    run_lemma = mo.ui.run_button(label="Lemmatisierung starten")
    map_contexts = mo.ui.checkbox(value=True, label="Zielwortliste auf Kontextwörter anwenden")
    gender_rule = mo.ui.checkbox(
        value=True, label="Gegenderte Formen außerhalb der Liste vereinheitlichen (Bürger*innen → Bürgerinnen)"
    )
    mo.vstack([map_contexts, gender_rule, run_lemma])
    return gender_rule, map_contexts, run_lemma


@app.cell
def _(
    assets_dir,
    gender_rule,
    kwic_hits,
    map_contexts,
    mo,
    out_dir,
    pl,
    run_id,
    run_lemma,
    stanza,
    target_form_lemmas,
    target_lemma_map,
):
    # stanza lemmatization pipeline
    # not 100% identical to the spacy pipeline, contains subtle improvements
    # this cell only runs when the "Lemmatisierung starten" button above is hit
    mo.stop(
        not run_lemma.value,
        mo.md("**Lemmatisierung starten** klicken, um die geladenen KWIC-Treffer zu lemmatisieren."),
    )

    nlp_st2 = stanza.Pipeline(
        "de",
        processors="tokenize,mwt,pos,lemma",  # four parts, mwt stands for multi-word-tokenaization, e.g. exapnding "am" to "an dem"
        tokenize_no_ssplit=True,  # every context remains "one sentence" and does not get split
        dir=str(assets_dir / "stanza_resources"),
        download_method=None,  # offline: use only the models in assets/stanza_resources
        logging_level="WARN",
    )

    # mapping of the context words before Stanza (see make_context_mapper); the settings go into
    # the fingerprint of the part files, so a changed setting starts a new run
    import hashlib as _hashlib

    _settings = {
        "map_contexts": map_contexts.value,
        "gender_rule": gender_rule.value,
        # the list used for the contexts (differs from target_lemma_map with TARGET_LEMMAS_FROM_STANZA)
        "context_list": _hashlib.sha1(repr(sorted(target_form_lemmas.items())).encode("utf-8")).hexdigest(),
    }
    _mapper = None
    if map_contexts.value or gender_rule.value:
        _mapper = make_context_mapper(
            target_form_lemmas if map_contexts.value else {}, gender_rule=gender_rule.value
        )

    # store results as parquet file, in the same folder and under the same name as read below
    # (the cell below has to be run again afterwards, it does not depend on this one)
    _out = out_dir / f"kwic_hits_lemmatized_stanza_{run_id}.parquet"
    with mo.status.progress_bar(total=kwic_hits.height, title="Lemmatisierung Kontext") as bar3:
        _n_computed = lemmatize_kwic_hits(
            kwic_hits,
            nlp_st2,
            target_lemma_map,
            _out,
            context_mapper=_mapper,
            settings=_settings,
            on_progress=lambda _n: bar3.update(increment=_n),
        )

    print(f"{_n_computed} Stapel in diesem Lauf berechnet, gespeichert: {_out}")
    pl.read_parquet(_out, n_rows=100)
    return


@app.cell
def _(kwic_control_mode, kwic_mode):
    kwic_control_mode(kwic_mode.value)
    return


@app.cell
def _(out_dir, run_id):
    print(f"Will load the following pregenerated list: '{out_dir}/kwic_hits_lemmatized_stanza_{run_id}.parquet'")
    return


@app.cell
def _(out_dir, pl, run_id):
    # loads a pre-calculated list (written by the lemmatization above, or by an earlier run).
    # Replace "stanza" with "spacy" to load the spacy lemmatization.
    # lazy: the following steps stream through the file instead of loading it at once
    kwic_hits_lemmatized = pl.scan_parquet(out_dir / f"kwic_hits_lemmatized_stanza_{run_id}.parquet")
    kwic_hits_lemmatized.head(100).collect()
    return (kwic_hits_lemmatized,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Alte Rechtschreibung (ß) in den Lemmata korrigieren
    Stanza gibt viele Lemmata in der Schreibung vor 1996 aus, auch wenn der Text sie neu schreibt: `dass` → `daß`, `Ausschuss` → `Ausschuß`, `bewusst` → `bewußt`, `Prozess` → `Prozeß` (im Bundestagslauf rund 150 Lemmata, 20.000 Tokens). Für die PPMI ist das harmlos, weil es einheitlich geschieht, aber die Nachbarwörter im Bundle stünden in alter Schreibung.

    Eine feste Regel ß → ss geht nicht: `beschließen`, `groß`, `Maßnahme`, `weiß` sind richtig, obwohl der Text `beschlossen` oder `größer` schreibt. Deshalb entscheidet der Wortschatz des Laufs:
    * Ein Lemma mit ß wird durch seine ss-Schreibung ersetzt, wenn diese in den Kontexten (Originaltext, Kleinschreibung egal) mindestens 30-mal so häufig als Wort vorkommt wie die ß-Schreibung (`Ausschuss` 1.539 : `Ausschuß` 0; `beschliessen` 0 : `beschließen` 986).
    * Der hohe Faktor schützt Paare, bei denen beide Schreibungen echte Wörter sind (`Buße`/`Busse`) und Namen (`Heß`, `Krauß`). Eine kleine Ausnahmeliste fängt Restfehler ab.
    * Lemmata, deren ss-Schreibung im Lauf nirgends als Wort vorkommt, bleiben unverändert (z. B. `mißglückt`, Bundestagslauf: wenige Dutzend Tokens).

    Die Zuordnung wird als `output/<variante>/spelling_map_<projekt>_<variante>.csv` gespeichert. Der Schritt läuft vor „Movierte Feminina“ und braucht keine neue Lemmatisierung. Wörter der Zielwortliste sind nicht betroffen, sie haben ihr Lemma bereits aus der Liste.
    """)
    return


@app.function
def spelling_map(kwic_hits_lemmatized, min_ratio=30, overrides=None):
    """
    Map context lemmas in old German orthography (ß after a short vowel, e.g. "Ausschuß", "daß")
    to the reformed spelling, using the vocabulary of the run; returns a DataFrame
    old, new, n_lemma, n_word_old, n_word_new (lazy in, eager out).

    For every lemma with "ß", the candidate is the lemma with all "ß" replaced by "ss". It is
    taken if the candidate occurs as a word in the original context texts at least `min_ratio`
    times as often as the lemma itself (case-insensitive), so correct ß lemmas whose ss form
    does not occur ("beschließen", "groß", "Maßnahme") and pairs of real words ("Buße"/"Busse")
    stay. overrides: {old: new or None}, applied last (None = keep).
    """
    import polars as pl

    lemmas = (
        kwic_hits_lemmatized.select(pl.col("context_lemmas").alias("lemma"))
        .explode("lemma")
        .filter(pl.col("lemma").str.contains("ß"))
        .group_by("lemma")
        .len()
        .collect()
    )
    # word counts of the original texts (target form included), only words with ß or ss
    text = pl.concat_str(
        [pl.col(c).fill_null("") for c in ("context_before", "target_form", "context_after")], separator=" "
    )
    words = (
        kwic_hits_lemmatized.select(text.str.to_lowercase().str.extract_all(r"\w+").alias("word"))
        .explode("word")
        .filter(pl.col("word").str.contains("ß|ss"))
        .group_by("word")
        .len()
        .collect()
    )
    n_word = dict(zip(words["word"], words["len"]))

    result = {}
    for lemma, count in zip(lemmas["lemma"], lemmas["len"]):
        new = lemma.replace("ß", "ss")
        n_old, n_new = n_word.get(lemma.lower(), 0), n_word.get(new.lower(), 0)
        if n_new > 0 and n_new >= min_ratio * n_old:
            result[lemma] = (new, count, n_old, n_new)
    n_lemma = dict(zip(lemmas["lemma"], lemmas["len"]))
    for old, new in (overrides or {}).items():
        if new is None:
            result.pop(old, None)
        elif old in n_lemma:
            result[old] = (new, n_lemma[old], n_word.get(old.lower(), 0), n_word.get(new.lower(), 0))
    schema = {"old": pl.Utf8, "new": pl.Utf8, "n_lemma": pl.Int64, "n_word_old": pl.Int64, "n_word_new": pl.Int64}
    return pl.DataFrame(
        [(o, *v) for o, v in result.items()], schema=schema, orient="row"
    ).sort("n_lemma", "old", descending=[True, False])


@app.cell
def _(mo):
    fix_spelling = mo.ui.checkbox(value=True, label="Alte Rechtschreibung (ß) in den Lemmata korrigieren")
    fix_spelling
    return (fix_spelling,)


@app.cell
def _(fix_spelling, kwic_hits_lemmatized, out_dir, pl, run_id):
    # residual errors of the vocabulary rule (checked on the Bundestag run, punkt, 2026-11-20)
    SPELLING_OVERRIDES = {
        "flößen": None,  # would be "flössen"
    }

    if not fix_spelling.value:
        kwic_hits_spelling = kwic_hits_lemmatized
        print("Alte Rechtschreibung: ausgeschaltet")
    else:
        spellings = spelling_map(kwic_hits_lemmatized, overrides=SPELLING_OVERRIDES)
        spellings.write_csv(out_dir / f"spelling_map_{run_id}.csv")
        _map = dict(zip(spellings["old"], spellings["new"]))
        kwic_hits_spelling = kwic_hits_lemmatized.with_columns(
            pl.col("context_lemmas").list.eval(pl.element().replace(_map))
        )
        print(
            f"Alte Rechtschreibung: {spellings.height:,} Lemmata, "
            f"{spellings['n_lemma'].sum():,} Tokens korrigiert"
        )
    return (kwic_hits_spelling,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Movierte Feminina auf die Grundform zurückführen
    Das Geschlecht wird nicht unterschieden: Personenbezeichnungen auf *-in* (movierte Feminina) werden unter den Kontextwörtern auf die maskuline Grundform zurückgeführt (`Bürgerin` → `Bürger`, `Kollegin` → `Kollege`, `Ärztin` → `Arzt`). Zusammen mit dem Mapping vor Stanza (`Bürger*innen`, `BürgerInnen` → `Bürgerinnen` → `Bürgerin`) führen damit alle Schreibungen auf eine Dimension.

    Eine feste Regel reicht dafür nicht (Umlaut: `Ärztin` → `Arzt`; *-e*: `Kollegin` → `Kollege`; Fehltreffer: `Stein`, `Termin`, `Benzin`). Deshalb wird die Grundform im Wortschatz des Laufs gesucht:
    * Kandidat ist ein Substantiv-Lemma auf *-in*, das Stanza überwiegend als feminin bestimmt (das schließt `Stein`, `Termin`, `Benzin` aus).
    * Grundform ist der Stamm oder Stamm + *e*, erst danach dieselben ohne Umlaut, jeweils der häufigste, der im Lauf als nicht-feminines Substantiv vorkommt. Bei Komposita ohne belegte Grundform hilft das Grundwort (`Bundesfamilienministerin` → `Ministerin` → `Bundesfamilienminister`). Sonst bleibt die feminine Form.
    * Eine kleine Ausnahmeliste fängt Restfehler ab (`Polin` → `Pole` statt `Pol`; `Engin` ist ein Vorname).

    Die Zuordnung wird als `output/<variante>/feminine_base_map_<projekt>_<variante>.csv` gespeichert und kann dort geprüft werden. Wörter der Zielwortliste sind nicht betroffen, sie haben ihr Lemma bereits aus der Liste. Läufe von vor 2026-11-17 haben keine Spalte `context_gender`; für sie entfällt der Schritt.
    """)
    return


@app.function
def feminine_base_map(kwic_hits_lemmatized, overrides=None):
    """
    Map feminine person nouns on -in among the context lemmas to their masculine base form,
    using the vocabulary of the run itself; returns a DataFrame feminine, base, n_feminine, n_base
    (lazy in, eager out).

    A lemma is a candidate if it is a NOUN ending in "in" and Stanza tagged it Gender=Fem in at
    least half of its occurrences. The base is searched in two groups, the first group that has
    one wins, and within the group the most frequent: (1) stem, stem + "e" (Bürger, Kollege),
    (2) the same with the last umlaut removed (Arzt, Franzose). A base must occur as a NOUN that
    is mostly not feminine. A compound without such a base is mapped via its longest mapped head
    (Parteikollegin -> head Kollegin -> Parteikollege), the base then need not occur in the run
    (n_base 0). overrides: {feminine: base or None}, applied last (None = keep).
    """
    import polars as pl

    nouns = (
        kwic_hits_lemmatized.select(
            pl.col("context_lemmas").alias("lemma"),
            pl.col("context_pos_tags").alias("upos"),
            pl.col("context_gender").alias("gender"),
        )
        .explode("lemma", "upos", "gender")
        .filter(pl.col("upos") == "NOUN")
        .group_by("lemma")
        .agg(pl.len().alias("n"), (pl.col("gender") == "Fem").fill_null(False).mean().alias("fem_share"))
        .collect()
    )
    n = dict(zip(nouns["lemma"], nouns["n"]))
    feminine = dict(zip(nouns["lemma"], nouns["fem_share"] >= 0.5))

    plain = str.maketrans("äöüÄÖÜ", "aouAOU")

    def candidates(stem):
        yield [stem, stem + "e"]
        # last umlaut removed ("Ärzt" -> "Arzt", "Bäuer" -> "Bauer", "Französ" -> "Franzos")
        i = max(stem.rfind(u) for u in "äöüÄÖÜ")
        if i >= 0:
            s = stem[:i] + stem[i].translate(plain) + stem[i + 1 :]
            yield [s, s + "e"]

    result, unmapped = {}, []
    for lemma, count in n.items():
        if not (lemma.endswith("in") and len(lemma) > 4 and lemma[0].isupper() and feminine[lemma]):
            continue
        for group in candidates(lemma[:-2]):
            found = [c for c in group if c in n and not feminine[c]]
            if found:
                base = max(found, key=lambda c: n[c])
                result[lemma] = (base, count, n[base])
                break
        else:
            unmapped.append(lemma)
    # compounds whose masculine form does not occur in the run: map via the longest mapped head
    # ("Bundesfamilienministerin" -> head "Ministerin" -> "Bundesfamilienminister")
    for lemma in unmapped:
        for i in range(1, len(lemma) - 4):
            head = lemma[i].upper() + lemma[i + 1 :]
            if head in result:
                base = lemma[:i] + result[head][0][0].lower() + result[head][0][1:]
                result[lemma] = (base, n[lemma], n.get(base, 0))
                break
    for fem, base in (overrides or {}).items():
        if base is None:
            result.pop(fem, None)
        elif fem in n:
            result[fem] = (base, n[fem], n.get(base, 0))
    schema = {"feminine": pl.Utf8, "base": pl.Utf8, "n_feminine": pl.Int64, "n_base": pl.Int64}
    return pl.DataFrame(
        [(f, b, nf, nb) for f, (b, nf, nb) in result.items()], schema=schema, orient="row"
    ).sort("n_feminine", "feminine", descending=[True, False])


@app.cell
def _(mo):
    demove_feminines = mo.ui.checkbox(value=True, label="Movierte Feminina auf die Grundform zurückführen")
    demove_feminines
    return (demove_feminines,)


@app.cell
def _(demove_feminines, kwic_hits_spelling, out_dir, pl, run_id):
    # input: the lemmatized hits after the spelling correction (kwic_hits_spelling)
    # residual errors of the vocabulary search (checked on the Bundestag run, punkt, 2026-11-17)
    FEMININE_OVERRIDES = {
        "Polin": "Pole",  # would be "Pol"
        "Engin": None,  # first name
    }

    if not demove_feminines.value:
        kwic_hits_contexts = kwic_hits_spelling
        print("Movierte Feminina: ausgeschaltet")
    elif "context_gender" not in kwic_hits_spelling.collect_schema().names():
        kwic_hits_contexts = kwic_hits_spelling
        print("Movierte Feminina: Lauf ohne Spalte context_gender, Lemmatisierung neu starten")
    else:
        feminine_bases = feminine_base_map(kwic_hits_spelling, FEMININE_OVERRIDES)
        feminine_bases.write_csv(out_dir / f"feminine_base_map_{run_id}.csv")
        _map = dict(zip(feminine_bases["feminine"], feminine_bases["base"]))
        kwic_hits_contexts = kwic_hits_spelling.with_columns(
            pl.col("context_lemmas").list.eval(pl.element().replace(_map))
        )
        print(
            f"Movierte Feminina: {feminine_bases.height:,} Lemmata, "
            f"{feminine_bases['n_feminine'].sum():,} Tokens zurückgeführt"
        )
    return (kwic_hits_contexts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wahl der Zeitscheiben-Größe
    In der weiteren Analyse werden die individuellen Texte (Reden, Artikel) nicht einzeln betrachtet (dazu böten sie viel zu wenig Kontext), sondern in Zeitscheiben zusammengefasst. Die Größe der Zeitscheibe kann weiter unten eingestellt werden. Zur Wahl stehen Quartale, Halbjahre und Jahre. Es werden entsprechende Kennungen (Q1-4, H1-2) generiert.
    """)
    return


@app.cell
def _(mo):
    slice_grid_picker = mo.ui.dropdown(
        options={
            "Quartale (4 pro Jahr)": "quarter",
            "Halbjahre (2 pro Jahr)": "halfyear",
            "Jahre (1 pro Jahr)": "year",
        },
        value="Halbjahre (2 pro Jahr)",
        label="Zeitscheiben-Raster",
    )

    mo.style(slice_grid_picker, width="360px")
    return (slice_grid_picker,)


@app.function
def slice_kwic_hits(kwic_hits_lemmatized, corpus, slice_grid, facet_columns=(), missing_value="nicht zugeordnet"):
    """
    Add the date (as date) and the time slice label to the lemmatized hits; lazy in, lazy out.

    Slice labels sort alphabetically in chronological order ("2018-Q1", "2018-H2", "2018").
    The facet columns (written by the KWIC extraction) are kept as strings; missing values
    (null, empty) become missing_value, so no hit drops out of the facet-resolved counts.
    """
    import polars as pl

    missing_cols = set(facet_columns) - set(kwic_hits_lemmatized.collect_schema().names())
    assert not missing_cols, f"Facettenspalten fehlen in den Treffern: {missing_cols} (KWIC-Extraktion neu starten)"
    kwic_hits_lemmatized = kwic_hits_lemmatized.with_columns(
        pl.col(c).cast(pl.Utf8).str.strip_chars().replace("", None).fill_null(missing_value)
        for c in facet_columns
    )

    if "date" in kwic_hits_lemmatized.collect_schema().names():
        # date written by the KWIC extraction; moved to the end, where the join below puts it too
        sliced = kwic_hits_lemmatized.select(pl.exclude("date"), pl.col("date").str.to_date())
    else:
        # hits from runs before the date was written with them: take it from the corpus via id
        sliced = kwic_hits_lemmatized.join(
            corpus.select("id", pl.col("date").str.to_date()),
            on="id",
            how="left",
            maintain_order="left",
        )

    # define expressions for years and quarters
    year = pl.col("date").dt.year().cast(pl.Utf8)
    quarter = pl.col("date").dt.quarter()

    if slice_grid == "quarter":
        slice_label = year + "-Q" + quarter.cast(pl.Utf8)
    elif slice_grid == "halfyear":
        # Q1,Q2 -> H1 and Q3,Q4 -> H2
        halfyear_number = ((quarter - 1) // 2 + 1).cast(pl.Utf8)
        slice_label = year + "-H" + halfyear_number
    else:
        slice_label = year

    return sliced.with_columns(slice_label.alias("slice"))


@app.cell
def _(
    MISSING_FACET_VALUE,
    corpus,
    facet_columns,
    kwic_hits_contexts,
    out_dir,
    pl,
    run_id,
    slice_grid_picker,
):
    # add Quartale, Halbjahre and Jahre to the lists, which up to now only have dates
    # lazy: the plan is written to parquet in streaming mode and also used for the counts below
    # (kwic_hits_contexts = lemmatized hits, old ß spellings corrected, feminine context lemmas
    # reduced to the base form)
    kwic_hits_sliced = slice_kwic_hits(
        kwic_hits_contexts, corpus, slice_grid_picker.value, facet_columns, MISSING_FACET_VALUE
    )

    _path = out_dir / f"kwic_hits_sliced_{run_id}_{slice_grid_picker.value}.parquet"
    _tmp = _path.with_name(_path.stem + ".tmp.parquet")
    kwic_hits_sliced.sink_parquet(_tmp)
    _tmp.replace(_path)

    pl.read_parquet(_path, n_rows=100)
    return (kwic_hits_sliced,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ko-Okkurenzen
    Für die Zählung der Ko-Okkurenzen, welche die Basis für die Vektorisierung (mittels PPMI) sind, wird keine Suche oder Schleife verwendet. Stattdessen werden zunächst alle Kontextwörter, die als Liste in den Zellen gespeichert sind,  "explodiert", also jedes in eine eigene Zeile geschrieben, wobei Zeitscheibe und Zielwort erhalten bleiben. Daraus ergibt sich zunächst eine sehr lange Liste.

    Im zweiten Schritt wird diese Liste, die jegliches Vorkommen von Zielwort und Kontextword in allen Kontexten enthält, wieder aggregiert, und zwar nach Zielwort, Kontextwort und Zeitscheibe. Im Ergebnis erhält man die Kookurrenz pro Paar pro Zeitscheibe.

    Beide Schritte laufen mit der Streaming-Engine von Polars: Die explodierte Liste wird stückweise gezählt und existiert nie vollständig im Speicher. Bei großen Korpora wären das Milliarden Zeilen.
    """)
    return


@app.function
def count_cooc(kwic_hits_sliced, facet_columns=()):
    """
    Cooccurrence counts per slice, facet values, target lemma and context lemma, from the lazy
    sliced hits. Without facet_columns: per slice, target lemma and context lemma.

    The streaming engine counts batch by batch, so the exploded list (one row per context
    word of every hit) never exists as a whole in memory.
    """
    # context_lemmas contains all words from the context
    # only the columns needed for the count are exploded; the POS tags (exploded in parallel
    # before) do not change the number of rows
    # empty contexts may occur (e.g. in case of one-word-headlines treated as one KWIC hit). empty_as_null=False excludes them from the list - they don't contribute to occurences anyway.
    keys = ["slice", *facet_columns, "target_lemma"]
    return (
        kwic_hits_sliced
        .select(*keys, "context_lemmas")
        .explode("context_lemmas", empty_as_null=False)
        # group by target word, context word and slice (and facet values)
        .group_by([*keys, "context_lemmas"])
        .len()
        .rename({"context_lemmas": "context_lemma", "len": "n_cooc"})
        .collect(engine="streaming")
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Facetten.** Gezählt wird zusätzlich je Facettenwert (z. B. Fraktion oder Publikation; welche Facetten das sind, steht in der Korpus-Konfiguration unter „Einlesen der Daten“). Die Zählung je Zielwort und Zeitscheibe ist die Summe über alle Facettenwerte. Treffer ohne Facettenwert zählen als eigener Wert („nicht zugeordnet“), damit beide Zählungen dieselben Treffer enthalten. Die facettenaufgelösten Zählungen gehen nicht in die SVD ein. Im Embedding werden aus ihnen die Vektoren der Facetten-Zellen (records) berechnet und in den Raum der Aggregate projiziert.
    """)
    return


@app.cell
def _(facet_columns, kwic_hits_sliced, pl):
    # counts per facet value; the counts per target word and slice are their sum (every hit has
    # exactly one value per facet, missing ones were set to MISSING_FACET_VALUE when slicing)
    cooc_counts_facets = count_cooc(kwic_hits_sliced, facet_columns)
    cooc_counts = (
        cooc_counts_facets
        .group_by(["slice", "target_lemma", "context_lemma"])
        .agg(pl.col("n_cooc").sum())
    )
    assert cooc_counts["n_cooc"].sum() == cooc_counts_facets["n_cooc"].sum()

    # facet values with their number of hits, for facets_schema in the bundle; most frequent first,
    # because the frontend assigns its colors in this order
    facet_values = {
        _c: (
            kwic_hits_sliced.group_by(_c).len().collect(engine="streaming")
            .sort(["len", _c], descending=[True, False])
            .rename({_c: "id", "len": "n_hits"})
            .to_dicts()
        )
        for _c in facet_columns
    }
    cooc_counts
    return cooc_counts, cooc_counts_facets, facet_values


@app.cell
def _(
    MISSING_FACET_VALUE,
    Path,
    bundle_facets,
    cooc_counts,
    cooc_counts_facets,
    facet_values,
    kwic_variant,
    out_dir,
    project,
    project_meta,
    run_id,
    slice_grid_picker,
):
    import json as _json

    _name = f"{run_id}_{slice_grid_picker.value}"
    _cooc_path = out_dir / f"cooc_counts_{_name}.csv"
    cooc_counts.write_csv(_cooc_path)
    print(f"Gespeichert: {_cooc_path}")

    # facet-resolved counts and the corpus metadata (project, facets), read by pipeline_emb_ppmi.py;
    # an empty facet list means: corpus without facets, no records in the bundle
    _facets_path = out_dir / f"cooc_counts_facets_{_name}.parquet"
    if bundle_facets:
        cooc_counts_facets.write_parquet(_facets_path)
        print(f"Gespeichert: {_facets_path}")
    else:
        _facets_path.unlink(missing_ok=True)

    # labels and further attributes of the facet values (optional): input/facet_values_<facet id>.csv,
    # separator ";", UTF-8, column "id" = value in the corpus, "label" = display name, any further
    # columns (e.g. "ausrichtung") are passed on to facets_schema; values without entry keep label = id
    import csv as _csv

    def _with_attributes(facet):
        _table = Path(f"input/facet_values_{facet['id']}.csv")
        _attrs = {}
        if _table.is_file():
            with open(_table, encoding="utf-8-sig", newline="") as _fh:
                for _row in _csv.DictReader(_fh, delimiter=";"):
                    _id = (_row.pop("id") or "").strip()
                    if _id:
                        _attrs[_id] = {_c: _v.strip() for _c, _v in _row.items() if _v and _v.strip()}
        _values = [{**_attrs.get(_v["id"], {}), **_v} for _v in facet_values[facet["column"]]]
        _found = {_v["id"] for _v in _values}
        _missing = sorted(_found - set(_attrs) - {MISSING_FACET_VALUE})
        _unused = sorted(set(_attrs) - _found)
        if _table.is_file():
            print(f"{_table}: {len(_attrs)} Einträge")
            if _missing:
                print(f"  im Korpus, nicht in der Tabelle (Bezeichnung = ID): {', '.join(_missing)}")
            if _unused:
                print(f"  in der Tabelle, ohne Treffer im Korpus: {', '.join(_unused)}")
        return {**facet, "values": _values}

    # corpus metadata for emb and post: project (file names, meta.name, meta.source of the bundle),
    # run parameters and the facet declaration
    _declaration = {
        "project": project,
        "name": project_meta["name"],
        "source": project_meta["source"],
        "kwic_variant": kwic_variant,
        "slice_grid": slice_grid_picker.value,
        "missing_value": MISSING_FACET_VALUE,
        "facets": [_with_attributes(_f) for _f in bundle_facets],
    }
    _meta_path = out_dir / f"corpus_meta_{_name}.json"
    with open(_meta_path, "w", encoding="utf-8") as _fh:
        _json.dump(_declaration, _fh, ensure_ascii=False, indent=2)
    print(f"Gespeichert: {_meta_path}")
    for _f in _declaration["facets"]:
        print(f"  {_f['label']}: " + ", ".join(f"{_v['id']} ({_v['n_hits']:,})" for _v in _f["values"]))

    # everything pipeline_emb_ppmi.py reads, for the copy switch at the end
    pre_output_files = [
        str(_cooc_path),
        str(out_dir / f"kwic_hits_sliced_{_name}.parquet"),
        str(_meta_path),
        *([str(_facets_path)] if bundle_facets else []),
    ]
    return (pre_output_files,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Filter
    Grundsätzlich könnte man vor Berechnung der Kookkurenzen die Liste auf unerwünschte Kontextwörter filtern, beispielsweise nach POS, Rednernamen und domänenspezifischen Stoppwörtern.
    Ein Vorteil der Filterung bereits im Pre-Processing wäre eine kleinere Matrix, und mithin weniger Speicherbedarf und Rechenzeit. Ein Nachteil könnte sein, dass die Zählung und SVD auf subtile Weise verschlechtert wird. Daher wurde entschieden, die Kookkurrenz-Zählung und Matrixzerlegung auf ungefilterten Daten durchzuführen. Im Postprocessing besteht dann die Möglichkeit, die anzuzeigenden Ankerwörter und nächsten Nachbarn zu filtern.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Output

    Als Ergebnis des Pre-Processing werden pro Projekt und KWIC-Extraktionsmethode (festes Fenster, NLTK Punkt und SoMaJo) die folgenden Dateien erzeugt. Alle liegen in `output/[kwic_variant]/` (ein Ordner je Methode) und werden von dort auch wieder gelesen; nur die Rednernamen hängen nicht von der Methode ab.

    * output/speaker_names_[projekt].csv: Extrahierte Redner-Namen, für spätere Filter-Vorbereitung. Bei Korpora ohne Redner (Zeitschriften) nur die Kopfzeile.
    * kwic_hits_[projekt]_[kwic_variant].csv: Treffer pro Zielwort, mit Kontext vor, Kontext nach, Zielwort, Datum und Facettenwerten (facet_1, …) des Dokuments. Ohne Facettenspalten bricht das Slicing ab: Extraktion neu starten.
    * kwic_hits_lemmatized_stanza_[projekt]_[kwic_variant].parquet: Lemmatisierte Version der KWIC-Treffer. Standardmäßg wird Stanza für die Lemmatisierung verwendet. Code für spaCy ist enthalten, aber deaktiviert.
    * feminine_base_map_[projekt]_[kwic_variant].csv: Zuordnung der movierten Feminina zur Grundform, zur Prüfung.
    * kwic_hits_sliced_[projekt]_[kwic_variant]_[slice_grid].parquet: Die lemmatisierten Treffer mit Zeitscheibe.
    * cooc_counts_[projekt]_[kwic_variant]_[slice_grid].csv: Kookkurenzen für Slice x Zielwort x Kontextwort.
    * cooc_counts_facets_[projekt]_[kwic_variant]_[slice_grid].parquet: dieselben Kookkurrenzen, zusätzlich aufgeteilt nach Facettenwert (Spalten facet_1, …). Nur bei Korpora mit Facetten.
    * corpus_meta_[projekt]_[kwic_variant]_[slice_grid].json: Projekt (ID, Name, Quelle), Laufparameter und Deklaration der Facetten für das Bundle (Spalte, ID, Bezeichnung, Werte mit Trefferzahl, Ersatzwert für fehlende Angaben). Leere Facettenliste = Korpus ohne Facetten.

    Der Kopierschalter unten legt die Dateien, die emb liest, nach `../02_embedding/input/[kwic_variant]/` und die Rednernamen nach `../03_postprocessing/input/`.
    """)
    return


@app.cell
def _(mo):
    copy_switch = mo.ui.switch(
        value=True,
        label="Nach 02_embedding/input kopieren, speaker_names_<projekt>.csv nach 03_postprocessing/input (sonst nur output/)",
    )
    copy_switch
    return (copy_switch,)


@app.cell
def _(Path, copy_switch, kwic_variant, pre_output_files, speaker_names_path):
    # copy to the input of the next stage
    # output/ is this stage's workbench, ../02_embedding/input/<kwic_variant>/ is what
    # pipeline_emb_ppmi.py works on. The files of the current run (kwic_variant,
    # slice_grid) are copied. speaker_names.csv is only read by pipeline_post.py and
    # goes straight to ../03_postprocessing/input/.
    import shutil

    if copy_switch.value:
        _next_dir = Path(f"../02_embedding/input/{kwic_variant}")
        _next_dir.mkdir(parents=True, exist_ok=True)
        _targets = [(_src, _next_dir) for _src in pre_output_files]
        _targets.append((speaker_names_path, Path("../03_postprocessing/input")))
        for _src, _dir in _targets:
            shutil.copy2(_src, _dir / Path(_src).name)
            print(f"kopiert nach {_dir / Path(_src).name}")
    else:
        print("nicht kopiert (Schalter aus) - liegt nur in output/")
    return


if __name__ == "__main__":
    app.run()
