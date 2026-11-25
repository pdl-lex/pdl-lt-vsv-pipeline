import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import codecs
    import csv
    import hashlib
    import os
    import re
    import unicodedata
    from datetime import date
    from pathlib import Path

    import charset_normalizer
    import marimo as mo
    import polars as pl

    # paths are relative to this notebook, so it runs from any working directory
    BASE_DIR = Path(__file__).resolve().parent

    # ------------------------------------------------------------------
    # file name parsing: date patterns, tried in this order on the lower-cased file stem.
    # Add new variants here. Named groups: y (year), m (month), d (day),
    # mname (month name), n (issue number, or month, see two_part_mode).
    # (?<!\d) and (?!\d) make sure a pattern never cuts a longer digit run.
    # ------------------------------------------------------------------
    _SEP = r"[-_. ]"
    DATE_PATTERNS = [
        # spiegel_2025-08-23, spiegel_2025_08_23, spiegel 2025.08.23
        ("ymd", rf"(?<!\d)(?P<y>\d{{4}}){_SEP}(?P<m>\d{{1,2}}){_SEP}(?P<d>\d{{1,2}})(?!\d)"),
        # spiegel_23.08.2025, spiegel_23-08-2025
        ("dmy", rf"(?<!\d)(?P<d>\d{{1,2}}){_SEP}(?P<m>\d{{1,2}}){_SEP}(?P<y>\d{{4}})(?!\d)"),
        # spiegel_23_august_2025, spiegel_23. Aug 2025
        ("dmy_name", rf"(?<!\d)(?P<d>\d{{1,2}}){_SEP}*(?P<mname>[a-zäöü]+){_SEP}*(?P<y>\d{{4}})(?!\d)"),
        # spiegel_20250823, spiegel20250823
        ("ymd_compact", r"(?<!\d)(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})(?!\d)"),
        # spiegel_23082025
        ("dmy_compact", r"(?<!\d)(?P<d>\d{2})(?P<m>\d{2})(?P<y>\d{4})(?!\d)"),
        # spiegel_2025_37, spiegel_2025_nr_37 (year + issue number, as in the ExtracEval example)
        ("y_n", rf"(?<!\d)(?P<y>\d{{4}}){_SEP}(?:(?:nr|no|heft|ausgabe){_SEP}*)?(?P<n>\d{{1,2}})(?!\d)"),
        # spiegel_37_2025, spiegel_nr_37_2025
        ("n_y", rf"(?<!\d)(?P<n>\d{{1,2}}){_SEP}(?P<y>\d{{4}})(?!\d)"),
    ]
    # a match of these patterns is clearly meant as a date: if it is invalid (31.02.), the file name
    # counts as not recognized, instead of falling back to a weaker pattern ("02.2025" as issue 2)
    STRICT_KINDS = {"ymd", "dmy"}
    DATE_REGEXES = [(kind, re.compile(rx)) for kind, rx in DATE_PATTERNS]

    # plausible years; protects against e.g. "spiegel_1234_5678"
    MIN_YEAR, MAX_YEAR = 1900, 2100

    MONTHS = {
        "januar": 1, "jan": 1, "jänner": 1, "january": 1,
        "februar": 2, "feb": 2, "february": 2,
        "märz": 3, "maerz": 3, "mär": 3, "mrz": 3, "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "mai": 5, "may": 5,
        "juni": 6, "jun": 6, "june": 6,
        "juli": 7, "jul": 7, "july": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "oktober": 10, "okt": 10, "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "dezember": 12, "dez": 12, "december": 12, "dec": 12,
    }

    # ExtracEval appends "-<8 hex chars>" (content hash) to the document id
    HASH_SUFFIX = re.compile(r"-[0-9a-f]{8}$")
    # leftover words next to an issue number ("spiegel_nr_37_2025" -> "spiegel")
    ISSUE_WORDS = re.compile(r"(?:^|_)(?:nr|no|heft|ausgabe)$")

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------
    # UTF-32 must be checked before UTF-16: its little-endian BOM starts with the UTF-16 one
    BOMS = [
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ]
    # candidates for the automatic detection of files that are not UTF-8. Restricted to Western
    # European encodings: unrestricted, charset-normalizer often picks Central European cp1250
    # for German text, which decodes the umlauts correctly but turns e.g. "à", "è", "ñ" into "ŕ", "č", "ń".
    # Extend the list if the corpus contains other languages.
    DETECT_ENCODINGS = ["cp1252", "latin_1", "iso8859_15", "mac_roman", "cp850", "cp437"]

    # typical traces of UTF-8 that was decoded as cp1252/latin-1 ("Ã¼" instead of "ü", "â€ž" instead of "„")
    # second character: latin-1 range 0x80-0xBF or one of the cp1252 characters at 0x80-0x9F
    MOJIBAKE = re.compile("[\u00c3\u00c2\u00e2][\u0080-\u00bf\u20ac\u201a\u0192\u201e\u2026\u2020\u2021\u02c6\u2030\u0160\u2039\u0152\u017d\u2018\u2019\u201c\u201d\u2022\u2013\u2014\u02dc\u2122\u0161\u203a\u0153\u017e\u0178]")

    # ------------------------------------------------------------------
    # text cleaning
    # ------------------------------------------------------------------
    CHAR_MAP = str.maketrans(
        {
            # ligatures from PDF fonts (U+FB00-FB06); "<fi>nanziell" would never match "finanziell"
            "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
            "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
            # invisible characters (soft hyphens at line ends are handled before, see SOFT_HYPHEN)
            "\u00ad": None, "\u200b": None, "\u200c": None, "\u200d": None,
            "\u2060": None, "\ufeff": None,
            # hyphen (U+2010) and non-breaking hyphen (U+2011) -> hyphen-minus; otherwise "3<U+2011>Prozent"
            # is cut into "3" and "<U+2011>Prozent", and "G<U+2011>7" never matches "G-7"
            "\u2010": "-", "\u2011": "-",
            # special spaces (no-break, en/em/thin/hair, narrow no-break, ...) -> normal space
            **{chr(c): " " for c in [0x00A0, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000]},
            # page break (ExtracEval page separator) and vertical tab -> line break
            "\f": "\n", "\v": "\n",
            # other C0 control characters carry no text
            **{chr(c): None for c in [*range(0x00, 0x09), *range(0x0e, 0x20), 0x7f]},
        }
    )
    # soft hyphen = hyphenation point set by the layout: always join the word parts
    SOFT_HYPHEN = re.compile(r"\u00ad[ \t]*(?:\n[ \t]*)?")
    # hyphen at a line end. Followed by a lower-case letter, it is a hyphenation and goes:
    # "Bundes-\nwehr" -> "Bundeswehr". Otherwise the hyphen belongs to the word and stays, only the
    # line break goes: "CDU-\nPolitiker" -> "CDU-Politiker". Coordinations keep the space:
    # "Ein-\nund Ausfuhr" -> "Ein- und Ausfuhr" (see COORDINATION).
    COORDINATION = r"(?!(?:und|oder|bis|sowie|wie|als|noch|bzw|respektive|beziehungsweise)\b)"
    # (U+2010 is already mapped to "-" by CHAR_MAP, which runs before these)
    LINE_HYPHEN_SPLIT = re.compile(rf"(\w)-[ \t]*\n[ \t]*{COORDINATION}([a-zäöüß])")
    LINE_HYPHEN_WORD = re.compile(rf"(\w-)[ \t]*\n[ \t]*{COORDINATION}(?=\w)")
    WHITESPACE = re.compile(r"\s+")
    SPACES_IN_LINE = re.compile(r"[^\S\n]+")
    BLANK_LINES = re.compile(r"\n\s*\n+")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Pipeline zur Erstellung von Bundle-Daten für VSV: Datenimport
    Dieses Notebook bereitet Volltexte von Zeitschriften- und Zeitungsausgaben so auf, dass sie vom Preprocessing (`pipeline_pre_ppmi.py`) eingelesen werden können.

    * **Datenimport (pipeline_pre_dataimport.py)**
    * Vorbereitung der Daten / Preprocessing (pipeline_pre_ppmi.py)
    * Embedding bzw. Vektorisierung (pipeline_emb_ppmi.py)
    * Nachbearbeitung und Generierung des Bundles (pipeline_post.py)

    **Eingabe:** ein Ordner mit Textdateien, eine Datei pro Ausgabe, wie sie z. B. das Werkzeug [ExtracEval](https://github.com/fsimonjetz/extraceval) aus PDFs erzeugt (`runs/<name>/outputs/<tool>/*.txt`, Seiten durch Seitenvorschub `\f` getrennt). Name der Publikation und Datum stehen im Dateinamen, z. B. `spiegel_20250823.txt`.

    **Ausgabe:** eine CSV-Datei (UTF-8) mit den Spalten `date` (ISO-Format `JJJJ-MM-TT`), `text` und `facet_1` (Name der Publikation), bei großen Datenmengen aufgeteilt in mehrere Dateien. Eine `id`-Spalte wird nicht geschrieben; das Preprocessing ergänzt sie.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Einstellungen
    Die Eingabedateien werden im Ordner `input/editions` erwartet (auch in Unterordnern, z. B. ein Ordner pro Publikation). Dieser Ordner ist wie der gesamte Input nicht Teil des Git-Repos.

    `EXPECTED_FILES` ist die erwartete Gesamtzahl an Ausgaben. Sie dient nur der Hochrechnung der Dateigröße, solange erst ein Teil der Daten vorliegt.
    """)
    return


@app.cell
def _():
    # folder with one text file per edition; subfolders are included
    INPUT_DIR = BASE_DIR / "input" / "editions"
    FILE_GLOB = "*.txt"

    # output: OUTPUT_BASENAME.csv, or OUTPUT_BASENAME_<year>.csv / _<year-month>.csv when split
    OUTPUT_DIR = BASE_DIR / "output"
    OUTPUT_BASENAME = "corpus_editions"

    # expected number of editions in the final corpus, for the size projection
    EXPECTED_FILES = 96_000
    return EXPECTED_FILES, FILE_GLOB, INPUT_DIR, OUTPUT_BASENAME, OUTPUT_DIR


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Dateinamen: Publikation und Datum
    Aus dem Dateinamen (ohne Endung) werden das Datum und der Name der Publikation gelesen. Groß-/Kleinschreibung spielt keine Rolle, als Trennzeichen sind `_`, `-`, `.` und Leerzeichen erlaubt, und das Datum darf vor oder nach dem Namen stehen. Erkannt werden:

    | Beispiel | Format |
    |---|---|
    | `spiegel_20250823` | JJJJMMTT |
    | `spiegel_2025-08-23`, `spiegel_2025_08_23` | JJJJ-MM-TT |
    | `spiegel_23.08.2025`, `spiegel_23082025` | TT.MM.JJJJ, TTMMJJJJ |
    | `spiegel_23_august_2025`, `spiegel_23. Aug 2025` | Tag, Monatsname (deutsch/englisch), Jahr |
    | `spiegel_2025_37`, `spiegel_nr_37_2025` | Jahr und Heftnummer (siehe unten) |
    | `20250823_spiegel` | Datum vor dem Namen |
    | `SPIEGEL_2025_37-a1b2c3d4` | mit Prüfsummen-Anhang, wie ExtracEval ihn an die Dokument-ID hängt |

    Was nach Entfernen des Datums übrig bleibt, ist der Name der Publikation (klein geschrieben, Trennzeichen vereinheitlicht zu `_`). Weitere Varianten lassen sich in `DATE_PATTERNS` im Setup-Abschnitt am Anfang ergänzen.

    **Zweiteilige Angaben** wie `2025_37` sind mehrdeutig: Bei Wochentiteln ist die Zahl meist die Heftnummer, bei Monatstiteln der Monat. Mit der Einstellung „Heftnummer = Kalenderwoche“ wird Heft 37 dem Donnerstag der ISO-Kalenderwoche 37 zugeordnet (der Donnerstag liegt immer im richtigen Jahr). Das ist eine Näherung: Heftnummern und Kalenderwochen stimmen nicht immer überein, für Quartale und Halbjahre als Zeitscheiben reicht die Genauigkeit aber in aller Regel. Solche Daten sind in der Übersicht als `issue_week` bzw. `month` markiert.
    """)
    return


@app.function
def to_date(kind, groups, two_part_mode):
    """Date from the named groups of a DATE_PATTERNS match, or None if the values are invalid."""
    y = int(groups["y"])
    if not MIN_YEAR <= y <= MAX_YEAR:
        return None
    try:
        if groups.get("n") is not None:
            n = int(groups["n"])
            if two_part_mode == "month":
                return date(y, n, 1), "month"
            # ISO week n, Thursday: always lies in year y
            return date.fromisocalendar(y, n, 4), "issue_week"
        if groups.get("mname") is not None:
            m = MONTHS.get(groups["mname"])
            if m is None:
                return None
        else:
            m = int(groups["m"])
        return date(y, m, int(groups["d"])), "date"
    except ValueError:  # e.g. 31.02., month 13, week 54
        return None


@app.function
def parse_stem(stem, two_part_mode="week"):
    """
    Read publication and date from a file name without extension.

    Returns a dict with "magazine_key", "date" and "date_source", or None if no date is found.
    """
    candidates = [stem]
    hash_match = HASH_SUFFIX.search(stem)
    if hash_match:
        # try without the ExtracEval hash first; "spiegel-20250823" itself looks like a hash suffix,
        # so the full stem remains the fallback
        candidates.insert(0, stem[: hash_match.start()])
    for candidate in candidates:
        s = candidate.lower()
        for kind, regex in DATE_REGEXES:
            for match in regex.finditer(s):
                parsed = to_date(kind, match.groupdict(), two_part_mode)
                if parsed is None:
                    if kind in STRICT_KINDS:
                        return None
                    continue
                rest = f"{s[: match.start()]}_{s[match.end():]}"
                rest = re.sub(r"[-_. ]+", "_", rest).strip("_")
                rest = ISSUE_WORDS.sub("", rest).strip("_")
                return {"magazine_key": rest, "date": parsed[0], "date_source": parsed[1]}
    return None


@app.cell
def _():
    # examples; output in the cell below
    _examples = [
        "spiegeL_20250823",
        "SPIEGEL_2025-08-23",
        "Der Spiegel 23.08.2025",
        "die_zeit_23082025",
        "stern_23_august_2025",
        "focus_23. Aug 2025",
        "SPIEGEL_2026_37-a1b2c3d4",
        "spiegel_nr_37_2025",
        "20250823_spiegel",
        "spiegel-20250823",
        "readme",
    ]
    pl.DataFrame(
        [{"stem": _s, **(parse_stem(_s) or {})} for _s in _examples],
        strict=False,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Der Schlüssel wird als Facettenwert (`facet_1`) geschrieben und ist die ID der Publikation im Bundle. Kommt dieselbe Publikation unter verschiedenen Schreibweisen vor (`der_spiegel`, `spiegel`), bildet `MAGAZINE_NAMES` die Varianten auf eine ID ab. Schlüssel ohne Eintrag bleiben unverändert.

    Die Anzeigenamen stehen nicht hier, sondern in `input/facet_values_publication.csv` (Spalten `id;label`, weitere Spalten möglich). Das Preprocessing übernimmt sie in das Bundle. Die IDs dieser Tabelle sind das Ziel der Abbildung. Die Übersicht weiter unten zeigt alle gefundenen IDs und ob sie in der Tabelle stehen.
    """)
    return


@app.cell
def _():
    # lower-case key from the file name -> publication ID (facet value, column id of
    # input/facet_values_publication.csv); only needed where the key differs from the ID
    MAGAZINE_NAMES = {
        # "der_spiegel": "spiegel",
        # "die_zeit": "zeit",
        # "süddeutsche_zeitung": "sz",
    }
    return (MAGAZINE_NAMES,)


@app.cell
def _():
    two_part_mode = mo.ui.radio(
        options={"Heftnummer = Kalenderwoche": "week", "Zahl = Monat": "month"},
        value="Heftnummer = Kalenderwoche",
        label="Zweiteilige Angaben (z. B. 2025_37)",
        inline=True,
    )
    two_part_mode
    return (two_part_mode,)


@app.cell
def _(FILE_GLOB, INPUT_DIR, MAGAZINE_NAMES, two_part_mode):
    mo.stop(
        not INPUT_DIR.is_dir(),
        mo.md(f"Eingabeordner **{INPUT_DIR}** fehlt. Bitte anlegen und die Textdateien dort ablegen."),
    )

    # only names and sizes are read here, no file content
    _rows = []
    for _path in sorted(INPUT_DIR.rglob(FILE_GLOB)):
        if not _path.is_file():
            continue
        _parsed = parse_stem(_path.stem, two_part_mode.value) or {}
        _rows.append(
            {
                "file": str(_path.relative_to(INPUT_DIR)),
                "magazine_key": _parsed.get("magazine_key"),
                "date": _parsed.get("date"),
                "date_source": _parsed.get("date_source"),
                "bytes": _path.stat().st_size,
            }
        )

    files = pl.DataFrame(
        _rows,
        schema={
            "file": pl.Utf8,
            "magazine_key": pl.Utf8,
            "date": pl.Date,
            "date_source": pl.Utf8,
            "bytes": pl.Int64,
        },
    ).with_columns(pl.col("magazine_key").replace(MAGAZINE_NAMES).alias("magazine"))

    print(f"{files.height:,} Dateien gefunden, davon {files['date'].null_count():,} ohne erkanntes Datum")
    files
    return (files,)


@app.cell
def _(files):
    # file names without a date, or without a publication name: these files are left out
    unparsed = files.filter(pl.col("date").is_null() | (pl.col("magazine") == ""))
    mo.vstack(
        [
            mo.md(f"**Nicht verwertbare Dateinamen: {unparsed.height:,}** (werden übersprungen)"),
            unparsed.select("file", "magazine", "date"),
        ]
    ) if unparsed.height else mo.md("Alle Dateinamen wurden erkannt.")
    return


@app.cell
def _(files):
    # overview per publication: check for spelling variants that should be merged in MAGAZINE_NAMES,
    # and whether the ID has a display name in input/facet_values_publication.csv
    _table = BASE_DIR / "input" / "facet_values_publication.csv"
    _known = []
    if _table.is_file():
        with open(_table, encoding="utf-8-sig", newline="") as _fh:
            _known = [_row["id"].strip() for _row in csv.DictReader(_fh, delimiter=";") if _row.get("id")]
    (
        files.filter(pl.col("date").is_not_null())
        .group_by("magazine")
        .agg(
            pl.len().alias("files"),
            pl.col("date").min().alias("first"),
            pl.col("date").max().alias("last"),
            pl.col("date_source").unique().sort().str.join(", ").alias("date_source"),
            (pl.col("bytes").sum() / 1e6).round(1).alias("MB"),
        )
        .with_columns(pl.col("magazine").is_in(_known).alias("in_facet_values"))
        .sort("magazine")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Größenabschätzung
    Die CSV-Datei ist etwa so groß wie die Summe der Textdateien (UTF-8 ist für deutsche Texte kaum größer als ältere Kodierungen, die Bereinigung spart wenige Prozent). Liegt erst ein Teil der Daten vor, wird anhand der mittleren Dateigröße auf `EXPECTED_FILES` hochgerechnet.

    Zum Vergleich: Die Bundestagsreden umfassen 382 MB. Eine Ausgabe einer Wochenzeitschrift hat grob 0,3 bis 1 MB Text, 96.000 Ausgaben ergäben also **etwa 30 bis 100 GB**, mehr als 100-mal so viel wie die Reden.

    Für den Arbeitsspeicher spielt die Dateigröße keine Rolle: `pipeline_pre_ppmi.py` liest die Korpusdateien Zeile für Zeile und hält nie das ganze Korpus im Speicher. Die Aufteilung nach Jahr oder Monat ist trotzdem ab etwa 10 GB empfehlenswert, aus praktischen Gründen: Einzelne Jahrgänge lassen sich leichter prüfen, kopieren und ergänzen, und ein abgebrochener Import betrifft nur das laufende Jahr. Das Preprocessing liest alle Teile in ihrer Reihenfolge, als wären sie eine Datei.
    """)
    return


@app.function
def total_ram_bytes():
    """Physical memory of this machine in bytes, or None if it cannot be determined."""
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


@app.cell
def _(EXPECTED_FILES, files):
    _valid = files.filter(pl.col("date").is_not_null() & (pl.col("magazine") != ""))
    _n = _valid.height
    _total = int(_valid["bytes"].sum() or 0)
    _mean = _total / _n if _n else 0
    projected_bytes = max(_total, _mean * EXPECTED_FILES)
    _ram = total_ram_bytes()

    # pipeline_pre_ppmi.py reads the corpus row by row, so the RAM does not limit the file size;
    # above ~10 GB, yearly files are recommended for handling (checking, copying, adding years)
    recommended_split = "none" if projected_bytes <= 10e9 else "year"

    _gb = lambda b: f"{b / 1e9:,.2f} GB"
    mo.md(
        f"""
    | | |
    |---|---|
    | Dateien mit erkanntem Namen und Datum | {_n:,} |
    | Summe vorhanden | {_gb(_total)} |
    | mittlere Größe pro Ausgabe | {_mean / 1e6:,.2f} MB |
    | hochgerechnet auf {EXPECTED_FILES:,} Ausgaben | **{_gb(projected_bytes)}** |
    | Arbeitsspeicher dieses Rechners | {_gb(_ram) if _ram else "unbekannt"} |
    | Empfehlung | **{"eine Datei" if recommended_split == "none" else "eine Datei pro Jahr"}** |
    """
    )
    return (recommended_split,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Kodierung
    Jede Datei wird als Bytefolge gelesen und in drei Stufen dekodiert:

    1. **Byte Order Mark (BOM):** Beginnt die Datei mit einer BOM, ist die Kodierung eindeutig (UTF-8, UTF-16, UTF-32).
    2. **UTF-8:** Lässt sich die Datei fehlerfrei als UTF-8 lesen, wird das übernommen. Das ist der Normalfall (ExtracEval schreibt UTF-8). Andere Kodierungen wie Windows-1252 oder ISO-8859-1 sind mit Umlauten fast nie zufällig gültiges UTF-8, die Prüfung ist daher zuverlässig.
    3. **Automatische Erkennung** mit [charset-normalizer](https://github.com/jawah/charset_normalizer) für alle übrigen Dateien; schlägt auch das fehl, Windows-1252 mit Ersatzzeichen. Die Erkennung ist auf westeuropäische Kodierungen beschränkt (`DETECT_ENCODINGS` im Setup-Abschnitt). Uneingeschränkt wählt sie für deutsche Texte oft das mitteleuropäische cp1250, das Umlaute richtig, aber z. B. `à`, `è`, `ñ` falsch dekodiert.

    Danach wird auf **Mojibake** geprüft: UTF-8, das früher einmal fälschlich als Windows-1252 gelesen und wieder gespeichert wurde („MÃ¼nchen“ statt „München“). Ist die ganze Datei so verfälscht, wird das rückgängig gemacht. Mischfälle (teils korrekt, teils verfälscht) werden nicht repariert, aber im Bericht gezählt; dafür käme die Bibliothek `ftfy` in Frage.

    Die Ausgabe ist immer UTF-8 ohne BOM. Der Bericht am Ende zeigt die erkannten Kodierungen pro Datei.
    """)
    return


@app.function
def decode_bytes(raw):
    """Decode file content; returns (text, encoding name)."""
    for bom, encoding in BOMS:
        if raw.startswith(bom):
            return raw.decode(encoding, errors="replace"), encoding
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    best = charset_normalizer.from_bytes(raw, cp_isolation=DETECT_ENCODINGS).best()
    if best is not None:
        return str(best), best.encoding
    return raw.decode("cp1252", errors="replace"), "cp1252 (fallback)"


@app.function
def fix_mojibake(text):
    """
    Undo UTF-8 that was decoded as cp1252/latin-1 ("Ã¼" -> "ü").

    Only applied to the whole text and only if it reduces the traces; returns (text, fixed, remaining traces).
    """
    n = len(MOJIBAKE.findall(text))
    if n == 0:
        return text, False, 0
    for encoding in ("cp1252", "latin-1"):
        try:
            fixed = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        remaining = len(MOJIBAKE.findall(fixed))
        if remaining < n:
            return fixed, True, remaining
    return text, False, n


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Textbereinigung
    **Zum Seitenvorschub `\f`:** Technisch stört er das Preprocessing nicht, denn sowohl `str.split()` (Fenster-Modus) als auch die Regex `\s` (Satz-Modus) behandeln ihn als Leerraum. Er wird trotzdem ersetzt, weil er in CSV-Dateien und anderen Werkzeugen (Editoren, Excel, `grep`) unerwartet ist. Ein Seitenwechsel ist keine Satzgrenze: Artikel laufen oft über die Seite weiter. Er wird daher wie ein Zeilenumbruch behandelt.

    Weitere Schritte, die für die Suche nach Zielwörtern wichtig sind (ein Zielwort wird nur gefunden, wenn es Zeichen für Zeichen übereinstimmt):

    * **Unicode-Normalisierung (NFC):** PDF-Werkzeuge liefern Umlaute teils zerlegt (`u` + Trema-Zeichen). Das sieht gleich aus, ist aber eine andere Zeichenfolge als `ü`.
    * **Ligaturen** (`ﬁ`, `ﬂ`, `ﬀ` …) werden in Einzelbuchstaben aufgelöst.
    * **Weiche Trennstriche** (Soft Hyphens) und unsichtbare Zeichen werden entfernt, Sonderleerzeichen werden zu normalen Leerzeichen.
    * **Geschützte und typografische Bindestriche** (U+2011, U+2010) werden zum normalen Bindestrich. Sonst zerfällt `3‑Prozent` in `3` und `‑Prozent`, und `G‑7` stimmt nicht mit `G-7` überein.

    Diese Zeichen-Schritte (bis hierher) bilden die Funktion `normalize_chars`. Das Preprocessing (`pipeline_pre_ppmi.py`) wendet sie auf jedes Korpus an, auch auf solche, die nicht über diesen Import kommen (die Bundestagsreden).
    * **Silbentrennung am Zeilenende** (optional, Standard: an): `Bundes-⏎wehr` → `Bundeswehr`. Der Bindestrich fällt nur weg, wenn nach dem Umbruch ein Kleinbuchstabe folgt; sonst gehört er zum Wort: `CDU-⏎Politiker` → `CDU-Politiker`. Aufzählungen wie `Ein-⏎und Ausfuhr` bleiben unverändert. Ohne diesen Schritt würde jedes getrennte Zielwort übersehen.
    * **Leerraum zusammenfassen** (optional, Standard: an): Zeilenumbrüche, Tabulatoren und Mehrfach-Leerzeichen werden zu einem Leerzeichen; jede Ausgabe steht dann in einer einzigen CSV-Zeile. Das Preprocessing nutzt Zeilenumbrüche nicht. Ausgeschaltet bleiben Zeilen und Absätze erhalten (CSV-konform in Anführungszeichen).

    Nicht entfernt werden Kolumnentitel, Seitenzahlen, Bildunterschriften und Anzeigen. Sie bleiben als Rauschen in den Kontexten.

    Leere Dateien und Dateien mit identischem Text (z. B. doppelt abgelegte Ausgaben) werden übersprungen. Mehrere unterschiedliche Dateien zu Publikation und Datum (z. B. Regionalausgaben) bleiben erhalten, werden aber gemeldet.
    """)
    return


@app.function
def normalize_chars(text):
    """
    Character level only: line endings, NFC, soft hyphens, ligatures, invisible characters,
    special spaces and hyphens. Keeps line breaks and hyphenation; pipeline_pre_ppmi.py imports
    it and applies it to every corpus.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = SOFT_HYPHEN.sub("", text)
    return text.translate(CHAR_MAP)


@app.function
def clean_text(text, dehyphenate=True, collapse_whitespace=True):
    """Normalize a decoded text; see the markdown cell above for the single steps."""
    text = normalize_chars(text)
    if dehyphenate:
        text = LINE_HYPHEN_SPLIT.sub(r"\1\2", text)
        text = LINE_HYPHEN_WORD.sub(r"\1", text)
    if collapse_whitespace:
        return WHITESPACE.sub(" ", text).strip()
    text = SPACES_IN_LINE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return BLANK_LINES.sub("\n\n", text).strip()


@app.cell
def _():
    # example for the cleaning; "ü" is a decomposed "ü", "ﬁ" the "fi" ligature, "­" a soft hyphen
    _sample = (
        "Die Bundes-\nwehr pru\u0308ft den \ufb01nanziellen Rahmen.\f12  DER SPIEGEL\n"
        "Der CDU-\nPolitiker spricht über Ein-\nund Ausfuhr von Rüs\u00ad\ntungsgütern."
    )
    print(clean_text(_sample))
    return


@app.function
def output_key(day, split):
    """Part of the output file name for a date: None (one file), year or year-month."""
    if split == "year":
        return f"{day.year}"
    if split == "month":
        return f"{day:%Y-%m}"
    return None


@app.function
def write_corpus(entries, input_dir, output_dir, basename, split, dehyphenate, collapse_whitespace, on_progress=None, n_jobs=1, docs_per_task=10):
    """
    Decode, clean and write the editions as CSV (UTF-8, columns date, text, facet_1).

    entries: dicts with "file" (relative to input_dir), "date" and "magazine", sorted by date.
    Every output file is first written as *.part and renamed when complete, so an aborted run
    never leaves a truncated CSV that looks finished. Returns (written files, report rows).
    """
    from itertools import batched

    from joblib import Parallel, delayed

    def chunk_texts(chunk):  # runs in a worker process, one package of editions
        results = []
        for entry in chunk:
            raw = (input_dir / entry["file"]).read_bytes()
            text, encoding = decode_bytes(raw)
            text, mojibake_fixed, mojibake_left = fix_mojibake(text)
            replacement_chars = text.count("�")
            text = clean_text(text, dehyphenate, collapse_whitespace)
            results.append((entry, len(raw), text, encoding, mojibake_fixed, mojibake_left, replacement_chars))
        return results

    output_dir.mkdir(parents=True, exist_ok=True)
    # remove outputs of earlier runs with the same base name, otherwise e.g. a single file and
    # yearly files would be lying next to each other
    for old in [*output_dir.glob(f"{basename}.csv"), *output_dir.glob(f"{basename}_*.csv")]:
        if old.name != f"{basename}_report.csv":
            old.unlink()

    written, report, seen = [], [], {}
    current_key, handle, writer, part_path = object(), None, None, None

    def finish():
        if handle is not None:
            handle.close()
            final = part_path.with_suffix("")  # "x.csv.part" -> "x.csv"
            part_path.replace(final)
            written.append(final)

    try:
        jobs = (delayed(chunk_texts)(chunk) for chunk in batched(entries, docs_per_task))
        results = (r for chunk in Parallel(n_jobs=n_jobs, return_as="generator")(jobs) for r in chunk)  # input order
        for entry, bytes_in, text, encoding, mojibake_fixed, mojibake_left, replacement_chars in results:
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()

            if not text:
                status = "leer"
            elif digest in seen:
                status = f"Duplikat von {seen[digest]}"
            else:
                seen[digest] = entry["file"]
                status = "ok"
                key = output_key(entry["date"], split)
                if key != current_key:
                    finish()
                    name = basename if key is None else f"{basename}_{key}"
                    part_path = output_dir / f"{name}.csv.part"
                    handle = part_path.open("w", encoding="utf-8", newline="")
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerow(["date", "text", "facet_1"])
                    current_key = key
                writer.writerow([entry["date"].isoformat(), text, entry["magazine"]])

            report.append(
                {
                    "file": entry["file"],
                    "magazine": entry["magazine"],
                    "date": entry["date"],
                    "encoding": encoding,
                    "mojibake_fixed": mojibake_fixed,
                    "mojibake_left": mojibake_left,
                    "replacement_chars": replacement_chars,
                    "bytes_in": bytes_in,
                    "chars_out": len(text),
                    "status": status,
                }
            )
            if on_progress:
                on_progress()
        finish()
        handle = None
    finally:
        if handle is not None:  # aborted: close, but keep it as *.part
            handle.close()
    return written, report


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Import
    Die Aufteilung ist mit der Empfehlung aus der Größenabschätzung vorbelegt. Die Ausgabe ist nach Datum sortiert.
    """)
    return


@app.cell
def _(recommended_split):
    split = mo.ui.dropdown(
        options={"eine Datei": "none", "eine Datei pro Jahr": "year", "eine Datei pro Monat": "month"},
        value={"none": "eine Datei", "year": "eine Datei pro Jahr"}[recommended_split],
        label="Aufteilung",
    )
    dehyphenate = mo.ui.checkbox(value=True, label="Silbentrennung am Zeilenende auflösen")
    collapse_whitespace = mo.ui.checkbox(value=True, label="Leerraum zusammenfassen (eine Zeile pro Ausgabe)")
    run = mo.ui.run_button(label="Import starten")
    mo.vstack([mo.hstack([split, run], justify="start", gap=2), dehyphenate, collapse_whitespace])
    return collapse_whitespace, dehyphenate, run, split


@app.cell
def _(
    INPUT_DIR,
    OUTPUT_BASENAME,
    OUTPUT_DIR,
    collapse_whitespace,
    dehyphenate,
    files,
    run,
    split,
):
    # this cell only runs when the "Import starten" button above is hit
    mo.stop(not run.value, mo.md("Einstellungen wählen und **Import starten** klicken."))

    _entries = (
        files.filter(pl.col("date").is_not_null() & (pl.col("magazine") != ""))
        .sort("date", "magazine", "file")
        .select("file", "date", "magazine")
        .to_dicts()
    )
    with mo.status.progress_bar(total=len(_entries), title="Import") as _bar:
        written_files, _report_rows = write_corpus(
            _entries,
            INPUT_DIR,
            OUTPUT_DIR,
            OUTPUT_BASENAME,
            split.value,
            dehyphenate.value,
            collapse_whitespace.value,
            on_progress=_bar.update,
            n_jobs=8,
            docs_per_task=10,
        )

    import_report = pl.DataFrame(_report_rows, infer_schema_length=None)
    import_report.write_csv(OUTPUT_DIR / f"{OUTPUT_BASENAME}_report.csv")

    _sizes = sum(_p.stat().st_size for _p in written_files)
    print(f"{len(written_files)} Datei(en) geschrieben, zusammen {_sizes / 1e9:,.2f} GB:")
    for _p in written_files:
        print(f"  {_p}  ({_p.stat().st_size / 1e6:,.1f} MB)")
    return import_report, written_files


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Bericht
    Zusammenfassung über alle Dateien; die vollständige Liste steht in `output/corpus_editions_report.csv`.
    """)
    return


@app.cell
def _(import_report):
    _duplicates = import_report.filter(pl.col("status") == "ok").group_by("magazine", "date").len().filter(pl.col("len") > 1)
    _problems = import_report.filter(
        (pl.col("status") != "ok") | (pl.col("replacement_chars") > 0) | (pl.col("mojibake_left") > 0)
    )
    mo.vstack(
        [
            mo.md("**Status**"),
            import_report.group_by("status").len().sort("len", descending=True),
            mo.md("**Erkannte Kodierungen**"),
            import_report.group_by("encoding").agg(
                pl.len().alias("files"), pl.col("mojibake_fixed").sum().alias("mojibake_fixed")
            ).sort("files", descending=True),
            mo.md(
                f"**Dateien mit Auffälligkeiten: {_problems.height:,}** "
                "(übersprungen, Ersatzzeichen `�` oder verbliebenes Mojibake)"
            ),
            _problems,
            mo.md(f"**Mehrere Dateien zu gleicher Publikation und gleichem Datum: {_duplicates.height:,}**"),
            _duplicates.sort("magazine", "date"),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Kontrolle
    Die geschriebenen Dateien werden so eingelesen, wie `pipeline_pre_ppmi.py` es tut (`scan_csv`, `null_values="NA"`, Datum per `str.to_date()`), und auf Vollständigkeit geprüft.
    """)
    return


@app.cell
def _(import_report, written_files):
    _check = (
        pl.scan_csv(written_files, null_values="NA")
        .with_columns(pl.col("date").str.to_date())
        .select(
            pl.len().alias("rows"),
            pl.col("date").null_count().alias("dates_null"),
            pl.col("text").null_count().alias("texts_null"),
            pl.col("facet_1").null_count().alias("facets_null"),
            pl.col("date").min().alias("first"),
            pl.col("date").max().alias("last"),
            pl.col("facet_1").n_unique().alias("publications"),
            (pl.col("text").str.len_chars().mean() / 1e3).round(1).alias("mean_kchars"),
        )
        .collect()
    )
    _expected = import_report.filter(pl.col("status") == "ok").height
    assert _check["rows"][0] == _expected, f"{_check['rows'][0]} Zeilen gelesen, {_expected} erwartet"
    assert _check["dates_null"][0] == 0 and _check["texts_null"][0] == 0, "leere Werte in date oder text"
    _check
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Anschluss an das Preprocessing
    In `pipeline_pre_ppmi.py` unter „Einlesen der Daten“ das Korpus **„Zeitschriften-Ausgaben“** wählen. Es liest `output/corpus_editions.csv` bzw. alle Jahres- oder Monatsdateien (`corpus_editions_2025.csv`, …) in ihrer Reihenfolge. Die IDs werden dabei fortlaufend über alle Dateien vergeben. Rednernamen gibt es nicht, `speaker_names.csv` enthält dann nur die Kopfzeile.

    **Laufzeit:** Die KWIC-Extraktion durchläuft jedes Token in Python. Bei 30 bis 100 GB Text ist mit vielen Stunden zu rechnen. Die Lemmatisierung mit Stanza wächst mit der Zahl der Treffer, nicht mit der Textmenge, und kann nach einem Abbruch fortgesetzt werden.
    """)
    return


if __name__ == "__main__":
    app.run()
