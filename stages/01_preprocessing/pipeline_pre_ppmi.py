import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import re
    from pathlib import Path

    return Path, mo, pl, re


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

    Zur Wahl stehen zwei Korpora: die Bundestagsreden als Beispiel und die Zeitschriften-Ausgaben, die `pipeline_pre_dataimport.py` erzeugt (eine Datei oder eine pro Jahr bzw. Monat). Beide schreiben in dieselben Ausgabedateien.

    **Große Korpora:** Polars hält eine CSV-Datei beim Einlesen vollständig im Speicher, auch wenn sie stückweise abgefragt wird (gemessen: rund 3 GB für eine 2,6-GB-Datei). Für die Bundestagsreden (382 MB) ist das kein Problem, für Zehntausende Zeitschriften-Ausgaben schon. Deshalb wird das Korpus hier nur „lazy“ beschrieben und nicht geladen. Die KWIC-Extraktion liest die Dateien selbst Zeile für Zeile, mit konstantem Speicherbedarf. Alle späteren Schritte arbeiten nur noch mit den Treffern.
    """)
    return


@app.cell
def _(mo):
    corpus_source = mo.ui.dropdown(
        options={
            "Bundestagsreden (Beispiel)": "bundestag",
            "Zeitschriften-Ausgaben (pipeline_pre_dataimport.py)": "editions",
        },
        value="Bundestagsreden (Beispiel)",
        label="Korpus",
    )
    mo.style(corpus_source, width="480px")
    return (corpus_source,)


@app.cell
def _(Path, corpus_source, pl, re):
    # per corpus:
    # - corpus_files: the source CSV files
    # - source_columns: source columns for id, date and text, read row by row by the KWIC extraction;
    #   id None = no id in the source, a running number from 1 is added
    # - prepare_corpus: selection and renaming for the lazy table (preview, checks, speaker names)
    # - speaker_facet: facet with the speaker names, None = corpus without speakers
    if corpus_source.value == "bundestag":
        corpus_files = ["input/CPP-BT_2026-01-17_DE_CSV_Reden_Gesamt.csv"]
        source_columns = {"id": "rede_id", "date": "sitzung_datum", "text": "rede_text"}
        speaker_facet = "facet_2"

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

    else:
        # output of pipeline_pre_dataimport.py: corpus_editions.csv, or one file per year / month
        # (corpus_editions_2025.csv, corpus_editions_2025-08.csv); the report file is left out
        corpus_files = sorted(
            str(_p)
            for _p in Path("output").glob("corpus_editions*.csv")
            if re.fullmatch(r"corpus_editions(_\d{4}(-\d{2})?)?\.csv", _p.name)
        )
        source_columns = {"id": None, "date": "date", "text": "text"}
        speaker_facet = None

        def prepare_corpus(lf):
            return lf.select("date", "text", "facet_1")

    assert corpus_files and all(Path(_f).is_file() for _f in corpus_files), (
        f"Korpusdateien nicht gefunden: {corpus_files or 'keine'}"
    )
    print(f"{len(corpus_files)} Korpusdatei(en): {', '.join(corpus_files)}")
    return corpus_files, prepare_corpus, source_columns, speaker_facet


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
def _(corpus, pl, speaker_facet):
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

    speaker_names.write_csv("output/speaker_names.csv")
    print(f"{len(speaker_names):,} Namensbestandteile gespeichert")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Definition Zielwörter
    Als Zielwörter werden in diesem Beispiel nur Substantive gewählt, um Probleme mit der Flexion und Satzstellung zu vermeiden. "Drohne" und "Drohnen" sind Beispiele für einfache Flexionen. Gesucht werden nur ganze Wörter wie genannt, keine Komposita.
    Will man anderen Wortarten (vor allem Verben) einbeziehen, wird eine aufwändigere grammatikalische Behandlung nötig.
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
def _():
    # If you want to process a large list of target words, the code below reads them from a CSV file
    # The file must start with a header row (e.g. "target_form"): the first line is always skipped,
    # so a list without header silently loses its first target word.

    import csv
    with open("input/target_list.csv", encoding="utf-8", newline="") as f2:
        reader = csv.reader(f2)
        next(reader, None)  # skip header row (required, see above)
        target_forms = {row[0].strip() for row in reader if row and row[0].strip()}

    print(target_forms)
    return (target_forms,)


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
def _(Path, kwic_variant):
    # create output folder if necessary
    Path(f"output/{kwic_variant}").mkdir(parents=True, exist_ok=True)
    return


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
def read_corpus_rows(files, id_col, date_col, text_col, on_progress=None):
    """
    Yield (id, date, text) for every row of the corpus CSV files, one row at a time.

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
            position = 0
            for row in reader:
                if i_id is None:
                    doc_id, next_id = next_id, next_id + 1
                else:
                    doc_id = None if row[i_id] in missing else row[i_id]
                date = None if row[i_date] in missing else row[i_date]
                text = None if row[i_text] in missing else row[i_text]
                yield doc_id, date, text
                if on_progress:
                    # position of the underlying byte stream; advances in blocks of the read buffer
                    new_position = raw.tell()
                    if new_position != position:
                        on_progress(new_position - position)
                        position = new_position


@app.function
def write_kwic_hits(rows, extract, out_path, batch_size=100_000):
    """
    Run the KWIC extraction over (id, date, text) rows and write all hits to one CSV file.

    The hits are written in batches of `batch_size`, so only one batch is held in memory.
    Each hit also gets the date of its document; later steps then need no join with the corpus.
    The file is first written as *.part and renamed when complete, so an aborted run leaves no
    truncated CSV under the final name. Returns (number of documents, number of hits).
    """
    from pathlib import Path

    import polars as pl

    out_path = Path(out_path)
    part_path = out_path.with_name(out_path.name + ".part")
    n_docs, n_hits, hits, header = 0, 0, [], True
    with part_path.open("wb") as fh:
        for doc_id, date, text in rows:
            n_docs += 1
            if not text:
                continue
            for hit in extract(doc_id, text):
                hit["date"] = date
                hits.append(hit)
            if len(hits) >= batch_size:
                pl.DataFrame(hits).write_csv(fh, include_header=header)
                n_hits, hits, header = n_hits + len(hits), [], False
        if hits:
            pl.DataFrame(hits).write_csv(fh, include_header=header)
            n_hits, header = n_hits + len(hits), False
        if header:  # no hits at all: header only
            fh.write(b"id,target_form,position,context_before,context_after,sentence_based,date\n")
    part_path.replace(out_path)
    return n_docs, n_hits


@app.cell
def _(Path, corpus_files, kwic_extract, kwic_variant, mo, pl, run, source_columns):
    # this cell only runs when the "Extraktion starten" button above is hit
    mo.stop(
        not run.value, mo.md("Einstellungen wählen und **Extraktion starten** klicken.")
    )

    # the corpus files are read row by row (read_corpus_rows), the hits written in batches
    # (write_kwic_hits); the progress bar counts megabytes of the corpus files
    _total_mb = sum(Path(_f).stat().st_size for _f in corpus_files) // 1_000_000 + 1
    _out = Path(f"output/kwic_hits_{kwic_variant}.csv")
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
        )
        # same file name as read below (kwic_variant includes the window size)
        _n_docs, _n_hits = write_kwic_hits(_rows, kwic_extract, _out)

    print(f"{_n_docs:,} Dokumente, {_n_hits:,} Treffer, gespeichert: {_out}")
    pl.read_csv(_out, n_rows=100, null_values="NA")
    return


@app.cell
def _(kwic_control_mode, kwic_mode):
    kwic_control_mode(kwic_mode.value)
    return


@app.cell
def _(kwic_variant):
    print(
        f"Will load the following pregenerated list: 'output/{kwic_variant}/kwic_hits_{kwic_variant}.csv'"
    )
    return


@app.cell
def _(kwic_variant, pl):
    # Read pre-generated list
    # default modes are "punkt", "somajo" and "window20"
    # Use control above to switch

    kwic_hits = pl.read_csv(f"output/{kwic_variant}/kwic_hits_{kwic_variant}.csv", null_values="NA")
    print(f"Geladen: kwic_hits_{kwic_variant}.csv")
    kwic_hits
    return (kwic_hits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lemmatisierung und POS-Bestimmung

    Sowohl die Zielwörter als auch insbesondere die Text- bzw- Kontextwörter liegen noch in ihrer regulären, das heisst gemäß Grammatik flektierten Form vor. Um sie später vergleichen zu können, müssen sie auf ihre einheitliche Grundform (z.B. "Drohnen" > "Drohne") zurückgeführt werden. Im gleichen Zuge wird auch die Wortart mit bestimmt - diese kann später zur Filterung verwendet werden.

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
def _(kwic_hits, mo, nlp, pl, target_lemma_map_spacy):
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
    kwic_hits_lemmatized_spacy.write_parquet("output/kwic_hits_lemmatized_spacy.parquet")

    kwic_hits_lemmatized_spacy
    return


@app.cell
def _(assets_dir, target_forms):
    import stanza
    from stanza import DownloadMethod

    nlp_st = stanza.Pipeline(
        "de",
        processors="tokenize,pos,lemma",
        tokenize_pretokenized=True,
        dir=str(assets_dir / "stanza_resources"),
        download_method=None,  # offline: use only the models in assets/stanza_resources
        logging_level="WARN",
    )

    words = sorted(target_forms)
    doc = nlp_st([[w] for w in words])  # jedes Wort = ein "Satz" aus einem Token
    target_lemmas_stanza = [s.words[0].lemma for s in doc.sentences]
    target_lemma_map_stanza = dict(zip(sorted(target_forms), target_lemmas_stanza))
    target_lemma_map_stanza
    return stanza, target_lemma_map_stanza


@app.function
def lemmatize_batch(batch, nlp, target_lemma_map, chunk=500, on_progress=None):
    """
    Lemmatize the contexts of a batch of KWIC hits with Stanza; returns the batch with
    the columns target_lemma, context_lemmas and context_pos_tags added.

    Hit and contexts are lemmatized together as one text, so that Stanza sees the whole
    sentence; afterwards each word is assigned to "before" or "after" by its character
    position, the hit itself is left out. Punctuation is skipped.
    """
    import polars as pl
    import stanza

    # define the various parts
    # the 'b or ""' replaces null values with empty strings
    # texts recombines the three parts
    contexts_before = [(b or "").strip() for b in batch["context_before"].to_list()]
    target_hit_forms = [(t or "").strip() for t in batch["target_form"].to_list()]
    contexts_after = [(a or "").strip() for a in batch["context_after"].to_list()]
    texts = [f"{b} {t} {a}" for b, t, a in zip(contexts_before, target_hit_forms, contexts_after)]

    # initialize empty result lists
    before_lemmas, before_pos_tags = [], []
    after_lemmas, after_pos_tags = [], []

    # inner loop: feed `chunk` texts to stanza at once. idx denotes line numbers in the batch
    for start in range(0, len(texts), chunk):
        idx = range(start, min(start + chunk, len(texts)))
        docs = nlp([stanza.Document([], text=texts[i]) for i in idx])

        # for every document, calculate the target word position
        for i, doc in zip(idx, docs):
            t_start = len(contexts_before[i]) + 1  # starts at context_before +1
            t_end = t_start + len(target_hit_forms[i])  # ends at context_before +1 + len
            bl, bp, al, ap = [], [], [], []

            # process tokens in the document
            for sent in doc.sentences:
                for tok in sent.tokens:  # tokens are surface forms (e.g. "zum")
                    for w in tok.words:  # words are syntactical units (e.g. "zu" + "dem")
                        if w.upos == "PUNCT":  # skip punctuation
                            continue
                        lemma = w.lemma or w.text  # if no lemma is available, use original text
                        if tok.end_char <= t_start:  # determine whether it belongs to before or after
                            bl.append(lemma)
                            bp.append(w.upos)
                        elif tok.start_char >= t_end:
                            al.append(lemma)
                            ap.append(w.upos)
                        # else: belongs to target word => ignore

            # append to lists
            before_lemmas.append(bl)
            before_pos_tags.append(bp)
            after_lemmas.append(al)
            after_pos_tags.append(ap)

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
        )
        # adds the target lemma, mapped from the target form via the dict generated above
        .with_columns(pl.col("target_form").replace(target_lemma_map).alias("target_lemma"))
        # we processed context_before and context_after separately, assuming this could be of use later
        # for the moment, we only need the joint columns, so before and after get concatenated here
        .with_columns(
            pl.concat_list("context_before_lemmas", "context_after_lemmas").alias("context_lemmas"),
            pl.concat_list("context_before_pos_tags", "context_after_pos_tags").alias("context_pos_tags"),
        )
        # also drop the separate columns
        # this way, the data structure is identical between spacy and stanza files
        .drop(
            "context_before_lemmas",
            "context_before_pos_tags",
            "context_after_lemmas",
            "context_after_pos_tags",
        )
    )


@app.function
def lemmatize_kwic_hits(kwic_hits, nlp, target_lemma_map, out_path, batch_size=20_000, on_progress=None):
    """
    Lemmatize all KWIC hits batch by batch and write the result to one parquet file.

    Every batch of `batch_size` hits is written as its own part file to <out_path stem>_parts/,
    so only one batch of lemma lists is held in memory. If a run is aborted, the next run
    skips the parts already written. The parts belong to exactly one input: a fingerprint
    of kwic_hits and the batch size is stored with them, and if it differs, the parts are
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
        lemmatized = lemmatize_batch(batch, nlp, target_lemma_map, on_progress=on_progress)
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
        ).write_parquet(tmp)
    tmp.replace(out_path)
    shutil.rmtree(parts_dir)
    return n_computed


@app.cell
def _(mo):
    mo.md(r"""
    Die Lemmatisierung des Kontexts läuft in Stapeln zu 20.000 Treffern, jeder Stapel wird sofort als Teildatei gespeichert (`output/kwic_hits_lemmatized_stanza_<variante>_parts/`). Wird der Lauf abgebrochen, setzt der nächste Lauf beim ersten fehlenden Stapel fort. Das gilt nur, solange die KWIC-Treffer dieselben sind, sonst wird neu begonnen. Am Ende werden die Teile zu einer Datei zusammengeführt und gelöscht.
    """)
    return


@app.cell
def _(mo):
    run_lemma = mo.ui.run_button(label="Lemmatisierung starten")
    run_lemma
    return (run_lemma,)


@app.cell
def _(
    Path,
    assets_dir,
    kwic_hits,
    kwic_variant,
    mo,
    pl,
    run_lemma,
    stanza,
    target_lemma_map_stanza,
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

    # store results as parquet file
    _out = Path(f"output/kwic_hits_lemmatized_stanza_{kwic_variant}.parquet")
    with mo.status.progress_bar(total=kwic_hits.height, title="Lemmatisierung Kontext") as bar3:
        _n_computed = lemmatize_kwic_hits(
            kwic_hits,
            nlp_st2,
            target_lemma_map_stanza,
            _out,
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
def _(kwic_variant):
    print(
        f"Will load the following pregenerated list: 'output/{kwic_variant}/kwic_hits_lemmatized_stanza_{kwic_variant}.parquet'"
    )
    return


@app.cell
def _(kwic_variant, pl):
    # loads a pre-calculated list.
    # Replace "stanza" with "spacy" to load the spacy lemmatization.
    # lazy: the following steps stream through the file instead of loading it at once
    kwic_hits_lemmatized = pl.scan_parquet(
        f"output/{kwic_variant}/kwic_hits_lemmatized_stanza_{kwic_variant}.parquet"
    )
    kwic_hits_lemmatized.head(100).collect()
    return (kwic_hits_lemmatized,)


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
def slice_kwic_hits(kwic_hits_lemmatized, corpus, slice_grid):
    """
    Add the date (as date) and the time slice label to the lemmatized hits; lazy in, lazy out.

    Slice labels sort alphabetically in chronological order ("2018-Q1", "2018-H2", "2018").
    """
    import polars as pl

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
def _(Path, corpus, kwic_hits_lemmatized, kwic_variant, pl, slice_grid_picker):
    # add Quartale, Halbjahre and Jahre to the lists, which up to now only have dates
    # lazy: the plan is written to parquet in streaming mode and also used for the counts below
    kwic_hits_sliced = slice_kwic_hits(kwic_hits_lemmatized, corpus, slice_grid_picker.value)

    _path = Path(f"output/{kwic_variant}/kwic_hits_sliced_{kwic_variant}_{slice_grid_picker.value}.parquet")
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
def count_cooc(kwic_hits_sliced):
    """
    Cooccurrence counts per slice, target lemma and context lemma, from the lazy sliced hits.

    The streaming engine counts batch by batch, so the exploded list (one row per context
    word of every hit) never exists as a whole in memory.
    """
    # context_lemmas contains all words from the context
    # only the columns needed for the count are exploded; the POS tags (exploded in parallel
    # before) do not change the number of rows
    # empty contexts may occur (e.g. in case of one-word-headlines treated as one KWIC hit). empty_as_null=False excludes them from the list - they don't contribute to occurences anyway.
    return (
        kwic_hits_sliced
        .select("slice", "target_lemma", "context_lemmas")
        .explode("context_lemmas", empty_as_null=False)
        # group by target word, context word and slice
        .group_by(["slice", "target_lemma", "context_lemmas"])
        .len()
        .rename({"context_lemmas": "context_lemma", "len": "n_cooc"})
        .collect(engine="streaming")
    )


@app.cell
def _(kwic_hits_sliced):
    cooc_counts = count_cooc(kwic_hits_sliced)
    cooc_counts
    return (cooc_counts,)


@app.cell
def _(cooc_counts, kwic_variant, slice_grid_picker):
    cooc_counts.write_csv(f"output/{kwic_variant}/cooc_counts_{kwic_variant}_{slice_grid_picker.value}.csv")
    print(f"Gespeichert: output/{kwic_variant}/cooc_counts_{kwic_variant}_{slice_grid_picker.value}.csv")
    return


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

    Als Ergebnis des Pre-Processing werden pro KWIC-Extraktionsmethode (40-Zeichen-Fenster, NLTK Punkt und Somajo) die folgenden Dateien erzeugt:

    * speaker_names.csv: Extrahierte Redner-Namen, für spätere Filter-Vorbereitung: 1 Datei. Bei Korpora ohne Redner (Zeitschriften) nur die Kopfzeile.
    * kwic_hits_[kwic_variant]].csv: Treffer pro Zielwort, mit Kontext vor, Kontext nach, Zielwort und Datum des Dokuments. Für 3 KWIC-Varianten  bis zu 3 Dateien. (Ältere Dateien ohne Datumsspalte funktionieren weiter; das Datum kommt dann über die ID aus dem Korpus.)
    * kwic_hits_lemmatized_stanza_[kwic_variant].parquet: Lemmatisierte Version der kwic_hits_[kwic_variant]].csv. Standardmäßg wird NLTK Stanza für die Lemmatisierung verwendet. Code für spaCy ist enthalten, aber deaktiviert. Für 3 KWIC-Varianten  bis zu 3 Dateien.
    * kwic_hits_sliced_[kwic_variant]_[slice_grid].parquet: Pro Zeitscheibe aggregierte Version der vorigen Ausgabe: Größere Textmenge bietet mehr Kookkurenzen. Für 3 KWIC-Varianten und 3 Slice-Größen bis zu 9 Dateien.
    * cooc_counts_[kwic_variant]_[slice_grid]: Kookkurenzen für Slice x Zielwort x Kontextwort. Für 3 KWIC-Varianten und 3 Slice-Größen bis zu 9 Dateien.
    """)
    return


if __name__ == "__main__":
    app.run()
