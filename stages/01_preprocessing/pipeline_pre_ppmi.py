import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import re

    return mo, pl, re


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
    """)
    return


@app.cell
def _(pl):
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

    # scan_csv and collect means we use polars lazy execution, which is much faster for large datasets
    corpus = (
        pl.scan_csv("input/CPP-BT_2026-01-17_DE_CSV_Reden_Gesamt.csv", null_values="NA")
        .select(columns_of_interest)
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
        .collect()
    )

    # add and populate id column if not existent
    if "id" not in corpus.columns:
        corpus = corpus.with_row_index("id", offset=1)

    corpus
    return (corpus,)


@app.cell
def _(corpus):
    # assert that the columns id, date and text exist, and that ids are unqiue

    # define a set of required columns, and subtract existing ones; the resulting set should be empty
    required = {"id", "date", "text"}
    missing = required - set(corpus.columns)

    assert not missing, f"Fehlende Spalten: {missing}"
    assert corpus["id"].n_unique() == corpus.height, (
        "Werte in 'id' sind nicht eindeutig"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Speicherung der Rednernamen
    Die Namen der Redner werden hier in eine separate Liste gespeichert, und später im Post-Processing für die Filterung verwendet.
    """)
    return


@app.cell
def _(corpus, pl):
    # speaker names as an exact list of real persons in the corpus (more precise than NER, no extra NLP run)
    # the speaker facet has the form "Nachname_Vorname"; multi-part names ("von der Leyen") are split into single
    # tokens, because lemmas are single tokens as well
    # exported for pipeline_post.py, where speaker names are removed from neighbor and anchor candidates;
    # the matrix itself stays unfiltered
    # the facet that holds the speaker names in this corpus
    speaker_facet = "facet_2"

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

    import csv
    with open("input/target_list.csv", encoding="utf-8", newline="") as f2:
        reader = csv.reader(f2)
        next(reader, None)  # skip header – omit if not present
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
        10, 100, value=40, step=5, label="Fenster (Wörter)", show_value=True
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
def _(kwic_variant):
    from pathlib import Path

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
    TOKEN = re.compile(r"\S+")  # token = any sequence of non-whitespace characters
    collapse_whitespace = lambda s: re.sub(r"\s+", " ", s).strip()  # whitespace normalization

    # both following functions return start and end positions of detected sentences
    def punkt_spans(snip):
        return punkt.span_tokenize(snip)

    def somajo_spans(snip):
        for sent in somajo.tokenize_text([snip]):
            yield sent[0].character_offset[0], sent[-1].character_offset[1]

    def kwic_sentence(doc_id, text, spans=punkt_spans, pad=600, max_len=400, fb=250):
        """
        Extract keyword-in-context (KWIC) entries for all target words in a text.

        For each whitespace-separated token whose normalized form is in `target_forms`,
        the surrounding sentence is determined with the given segmenter. If no
        acceptable sentence is found (not found, truncated by the window border, or
        longer than `max_len`), a fixed character window around the hit is used instead.

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
        max_len : int, default 400
            Maximum accepted sentence length in characters.
        fb : int, default 250
            Half-width (in characters) of the fallback context window.

        Returns
        -------
        list of dict
            One dict per hit with the keys:
            - "id": document identifier
            - "target_form": normalized form of the hit
            - "position": token index in the document
            - "context_before": cleaned text before the hit (hit excluded)
            - "context_after": cleaned text after the hit (hit excluded)
            - "sentence_based": True if a full sentence was used, False if fallback

        Notes
        -----
        Requires the globals `target_forms` (set of target words) and `strip_punctuation()`.
        """
        hits = []
        for i, m in enumerate(
            TOKEN.finditer(text)
        ):  # iterate over whitespace-separated tokens; i = token index, m = match object
            word = strip_punctuation(m.group())  # remove punctuation at the edges
            if word not in target_forms:  # check if found word is one of the target forms
                continue
            hs, he = m.span()  # starting and end positions
            lo, hi = (
                max(0, hs - pad),
                min(len(text), he + pad),
            )  # define a segmentation window around current word
            snip = text[lo:hi]  # save text in window to snip
            span = None  # initialize with none for each hit, gets overwritten if a sentence is found
            for s, e in spans(
                snip
            ):  # iterate over all sentences in current window until a match is found
                if s <= hs - lo < e:  # position of the found word must be within bounds
                    cut = (
                        (s == 0 and lo > 0) or (e == len(snip) and hi < len(text))
                    )  # sentence touches the window border (but the window is not at the text border) -> possibly truncated
                    if (
                        not cut and e - s <= max_len
                    ):  # accepted if not cut and within max length
                        span = (lo + s, lo + e)
                    break
            a, b = (
                span or (max(0, hs - fb), min(len(text), he + fb))
            )  # if no acceptable sentence (not found, cut off, or too long): fixed window of +/- fb characters
            hits.append(
                {
                    "id": doc_id,
                    "target_form": word,
                    "position": i,
                    "context_before": collapse_whitespace(text[a:hs]),
                    "context_after": collapse_whitespace(text[he:b]),
                    "sentence_based": span is not None,
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


@app.cell
def _(corpus, kwic_extract, kwic_variant, mo, pl, run):
    # this cell only runs when the "Extraktion starten" button above is hit
    mo.stop(
        not run.value, mo.md("Einstellungen wählen und **Extraktion starten** klicken.")
    )



    _hit_dicts = []

    # call of the extraction function, wrapped in a progress bar
    for _doc_id, _text in mo.status.progress_bar(
        corpus.select(["id", "text"]).iter_rows(),
        total=len(corpus),
        title=f"KWIC-Extraktion ({kwic_variant})",
    ):
        _hit_dicts.extend(kwic_extract(_doc_id, _text))

    # same file name as read below (kwic_variant includes the window size)
    _kwic_hits = pl.DataFrame(_hit_dicts)
    _kwic_hits.write_csv(f"output/kwic_hits_{kwic_variant}.csv")
    _kwic_hits
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
    # default modes are "punkt", "somajo" and "window40"
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


@app.cell(disabled=True)
def _(assets_dir, kwic_hits, mo, stanza):
    # stanza lemmatization pipeline
    # not 100% identical to the spacy pipeline, contains subtle improvements

    nlp_st2 = stanza.Pipeline(
        "de",
        processors="tokenize,mwt,pos,lemma",  # four parts, mwt stands for multi-word-tokenaization, e.g. exapnding "am" to "an dem"
        tokenize_no_ssplit=True,  # every context remains "one sentence" and does not get split
        dir=str(assets_dir / "stanza_resources"),
        download_method=None,  # offline: use only the models in assets/stanza_resources
        logging_level="WARN",
    )

    # define the various parts
    # the 'b or ""' replaces null values with empty strings
    # texts recombines the three parts
    contexts_before = [(b or "").strip() for b in kwic_hits["context_before"].to_list()]
    target_hit_forms = [(t or "").strip() for t in kwic_hits["target_form"].to_list()]
    contexts_after = [(a or "").strip() for a in kwic_hits["context_after"].to_list()]
    texts = [f"{b} {t} {a}" for b, t, a in zip(contexts_before, target_hit_forms, contexts_after)]

    # initialize empty result lists
    before_lemmas, before_pos_tags = [], []
    after_lemmas, after_pos_tags = [], []

    # set the number of texts that stanza processes at once
    CHUNK = 500

    # outer function loop, wrapped in a progress bar, feeding CHUNK number of texts. idx denotes line numbers
    with mo.status.progress_bar(
        total=len(texts), title="Lemmatisierung Kontext"
    ) as bar3:
        for start in range(0, len(texts), CHUNK):
            idx = range(start, min(start + CHUNK, len(texts)))
            docs = nlp_st2([stanza.Document([], text=texts[i]) for i in idx])

            # for every document, calculate the target word position
            for i, doc3 in zip(idx, docs):
                t_start = len(contexts_before[i]) + 1  # starts at context_before +1
                t_end = t_start + len(target_hit_forms[i])  # ends at context_before +1 + len
                bl, bp, al, ap = [], [], [], []

                # process tokens in the document
                for sent in doc3.sentences:
                    for tok in sent.tokens:  # tokens are surface forms (e.g. "zum")
                        for w in (
                            tok.words
                        ):  # words are syntactical units (e.g. "zu" + "dem")
                            if w.upos == "PUNCT":  # skip punctuation
                                continue
                            lemma = (
                                w.lemma or w.text
                            )  # if no lemma is available, use original text
                            if (
                                tok.end_char <= t_start
                            ):  # determine whether it belongs to before or after
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

            bar3.update(increment=len(idx))
    return after_lemmas, after_pos_tags, before_lemmas, before_pos_tags


@app.cell(disabled=True)
def _(
    after_lemmas,
    after_pos_tags,
    before_lemmas,
    before_pos_tags,
    kwic_hits,
    kwic_variant,
    pl,
    target_lemma_map_stanza,
):
    # creates a new df with the lemmatized contexts and POS, split into before and after
    kwic_hits_lemmatized_stanza = kwic_hits.with_columns(
        [
            pl.Series("context_before_lemmas", before_lemmas),
            pl.Series("context_before_pos_tags", before_pos_tags),
            pl.Series("context_after_lemmas", after_lemmas),
            pl.Series("context_after_pos_tags", after_pos_tags),
        ]
    )

    # adds the target lemma, mapped from the target form via the dict generated above
    kwic_hits_lemmatized_stanza = kwic_hits_lemmatized_stanza.with_columns(
        pl.col("target_form").replace(target_lemma_map_stanza).alias("target_lemma")
    )

    # we processed context_before and context_after separately, assuming this could be of use later
    # for the moment, we only need the joint columns, so before and after get concatenated here
    kwic_hits_lemmatized_stanza = kwic_hits_lemmatized_stanza.with_columns(
        [
            pl.concat_list("context_before_lemmas", "context_after_lemmas").alias(
                "context_lemmas"
            ),
            pl.concat_list("context_before_pos_tags", "context_after_pos_tags").alias(
                "context_pos_tags"
            ),
        ]
    )

    # also drop the separate columns
    # this way, the data structure is identical between spcy and stanza files
    kwic_hits_lemmatized_stanza = kwic_hits_lemmatized_stanza.drop(
        "context_before_lemmas",
        "context_before_pos_tags",
        "context_after_lemmas",
        "context_after_pos_tags",
    )

    # store results as parquet file
    kwic_hits_lemmatized_stanza.write_parquet(
        f"output/kwic_hits_lemmatized_stanza_{kwic_variant}.parquet"
    )

    kwic_hits_lemmatized_stanza
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
    kwic_hits_lemmatized = pl.read_parquet(
        f"output/{kwic_variant}/kwic_hits_lemmatized_stanza_{kwic_variant}.parquet"
    )
    kwic_hits_lemmatized
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


@app.cell
def _(corpus, kwic_hits_lemmatized, pl, slice_grid_picker):
    # add Quartale, Halbjahre and Jahre to the lists, which up to now only have dates

    # slice tags are generated in a way so that alphabetical order and chronologial order are the same

    # add date in ISO date format
    kwic_hits_sliced = kwic_hits_lemmatized.join(
        corpus.select("id", pl.col("date").str.to_date()),
        on="id",
        how="left",
    )

    # define expressions for years and quarters
    year = pl.col("date").dt.year().cast(pl.Utf8)
    quarter = pl.col("date").dt.quarter()

    if slice_grid_picker.value == "quarter":
        slice_label = year + "-Q" + quarter.cast(pl.Utf8)

    elif slice_grid_picker.value == "halfyear":
        # Q1,Q2 -> H1 and Q3,Q4 -> H2
        halfyear_number = ((quarter - 1) // 2 + 1).cast(pl.Utf8)
        slice_label = year + "-H" + halfyear_number

    else:
        slice_label = year

    kwic_hits_sliced = kwic_hits_sliced.with_columns(slice_label.alias("slice"))

    kwic_hits_sliced
    return (kwic_hits_sliced,)


@app.cell
def _(kwic_hits_sliced, kwic_variant, slice_grid_picker):
    kwic_hits_sliced.write_parquet(
        f"output/{kwic_variant}/kwic_hits_sliced_{kwic_variant}_{slice_grid_picker.value}.parquet"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ko-Okkurenzen
    Für die Zählung der Ko-Okkurenzen, welche die Basis für die Vektorisierung (mittels PPMI) sind, wird keine Suche oder Schleife verwendet. Stattdessen werden zunächst alle Kontextwörter, die als Liste in den Zellen gespeichert sind,  "explodiert", also jedes in eine eigene Zeile geschrieben, wobei die restlichen Informationen erhalten bleiben. Die POS-Angaben werden gleichzeitig und parallel ebenfalls explodiert. Daraus ergibt sich zunächst eine sehr lange Liste.

    Im zweiten Schritt wird diese Liste, die jegliches Vorkommen von Zielwort und Kontextword in allen Kontexten enthält, wieder aggregiert, und zwar nach Zielwort, Kontextwort und Zeitscheibe. Im Ergebnis erhält man die Kookurrenz pro Paar pro Zeitscheibe.
    """)
    return


@app.cell
def _(kwic_hits_sliced):
    # context_lemmas contains all words from the context
    # context_pos_tags contains POS tags
    # .explode() with columns in the same call explodes them in parallel
    # empty contexts may occur (e.g. in case of one-word-headlines treated as one KWIC hit). empty_as_null=False excludes them from the list - they don't contribute to occurences anyway.
    cooc_exploded = kwic_hits_sliced.explode(
        ["context_lemmas", "context_pos_tags"], empty_as_null=False
    )

    cooc_exploded
    return (cooc_exploded,)


@app.cell
def _(cooc_exploded):
    # group by target word, context word and slice
    cooc_counts = (
        cooc_exploded
        .group_by(["slice", "target_lemma", "context_lemmas"])
        .len()
        .rename({"context_lemmas": "context_lemma", "len": "n_cooc"})
    )

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

    * speaker_names.csv: Extrahierte Redner-Namen, für spätere Filter-Vorbereitung: 1 Datei
    * kwic_hits_[kwic_variant]].csv: Treffer pro Zielwort, mit Kontext vor, Kontext nach und Zielwort. Für 3 KWIC-Varianten  bis zu 3 Dateien.
    * kwic_hits_lemmatized_stanza_[kwic_variant].parquet: Lemmatisierte Version der kwic_hits_[kwic_variant]].csv. Standardmäßg wird NLTK Stanza für die Lemmatisierung verwendet. Code für spaCy ist enthalten, aber deaktiviert. Für 3 KWIC-Varianten  bis zu 3 Dateien.
    * kwic_hits_sliced_[kwic_variant]_[slice_grid].parquet: Pro Zeitscheibe aggregierte Version der vorigen Ausgabe: Größere Textmenge bietet mehr Kookkurenzen. Für 3 KWIC-Varianten und 3 Slice-Größen bis zu 9 Dateien.
    * cooc_counts_[kwic_variant]_[slice_grid]: Kookkurenzen für Slice x Zielwort x Kontextwort. Für 3 KWIC-Varianten und 3 Slice-Größen bis zu 9 Dateien.
    """)
    return


if __name__ == "__main__":
    app.run()
