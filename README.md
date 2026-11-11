# pdl-lt-vsv-pipeline

Datenpipeline für das Tool **„Visualisierung Semantischer Vektoren" (VSV)**
([`pdl-lt-vsv`](../pdl-lt-vsv)). Sie erzeugt aus einem Textkorpus ein
Vektor-Bundle, das VSV direkt laden kann. Beispielkorpus: Bundestagsreden
2013–2026 (CPP-BT, siehe Preprocessing-Notebook).

## Stufen

| Stufe | Notebook | Ergebnis |
|---|---|---|
| 1 Preprocessing | `stages/01_preprocessing/pipeline_pre_ppmi.py` | KWIC-Treffer, Kookkurrenzen |
| 2 Embedding | `stages/02_embedding/pipeline_emb_ppmi.py` | PPMI-SVD-Vektoren je Wort × Zeitscheibe |
| 3 Postprocessing | `stages/03_postprocessing/pipeline_post.py` | fertiges Bundle (JSON) |

Jede Stufe liest aus `output/` der vorherigen. `*_diagnose.py` und
`*_demo.py` sind begleitende Notebooks zur Prüfung von Verfahrensentscheidungen.
Die Korpusdatei (`stages/01_preprocessing/input/`) ist nicht im Repo.

## Notebooks starten

```bash
uv sync
cd stages/01_preprocessing
uv run marimo edit pipeline_pre_ppmi.py
```

Die Notebooks arbeiten mit relativen Pfaden (`../01_preprocessing/output/…`),
deshalb aus dem jeweiligen Stufen-Ordner starten.

## Installation auf einem Offline-Rechner

Für Analyserechner ohne Internet (z. B. an der DNB). Einmal als root ausführen,
solange der Rechner noch Netz hat:

```bash
curl -LsSf https://raw.githubusercontent.com/pdl-lex/pdl-lt-vsv-pipeline/main/scripts/setup_offline.sh \
  | sudo bash -s -- <gruppe> [/opt]
```

Das Skript klont das Repo nach `<prefix>/pdl-lt-vsv-pipeline`, installiert
Python 3.13 und die Umgebung (`uv sync --locked`), lädt die Sprachmodelle nach
`assets/` (`scripts/fetch_assets.py`) und prüft, ob sie offline laden. Danach
gehört das Repo der Gruppe `<gruppe>`, und `/etc/profile.d/pdl-lt-vsv-pipeline.sh`
schaltet uv dauerhaft offline. Es folgt denselben Konventionen wie
`zeitenwende-sgns/scripts/setup_offline.sh` (gleicher Prefix, gleiches Python
unter `<prefix>/uv/python`), beide Installationen vertragen sich.

Prüfen, ob die Modelle laden (geht auch offline):
`uv run python scripts/fetch_assets.py --check`

## Schema

[`SCHEMA.md`](SCHEMA.md) beschreibt das Bundle-Format – den Vertrag zwischen
Pipeline und VSV. Die Datei ist **nur eine Kopie**: Das Master-Schema liegt in
`pdl-lt-vsv/docs/SCHEMA.md`. Änderungen dort vornehmen und hierher kopieren.
