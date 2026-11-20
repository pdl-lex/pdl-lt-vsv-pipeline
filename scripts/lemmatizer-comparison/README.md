# Vergleich der Lemmatisierer: Stanza, spaCy, simplemma

Stand: 2026-11-20. Frage: Kann simplemma (oder spaCy) Stanza in `stages/01_preprocessing/pipeline_pre_ppmi.py` ersetzen, schneller und bei gleicher Qualität?

**Ergebnis:** Nein. Stanza liefert die besten Lemmata, sobald man seine alte Rechtschreibung (`Rußland`, `Ausschuß`) herausrechnet. Nur Stanza löst die movierten Feminina sauber auf (`Soldatinnen` → `Soldatin`). simplemma ist etwa 400-mal schneller, liefert aber keine Wortart und kein Genus. Die Pipeline braucht beides, für den Nachbarfilter (NOUN/PROPN/ADJ/VERB in `pipeline_post.py`) und für `feminine_base_map`. spaCy ist etwa 8-mal schneller, hat aber die schwächsten Lemmata (`Soldatinn`, `beschäftigter`, `grüne`).

## Versionen und Einstellungen

| System | Version | Einstellung |
|---|---|---|
| Stanza | 1.15.0 | wie `nlp_st2` in der Pipeline: `tokenize,mwt,pos,lemma`, `tokenize_no_ssplit=True` |
| spaCy | 3.8, `de_core_news_md` 3.8.0 | wie die (deaktivierte) spaCy-Zelle: ohne `parser`, `ner`, `senter`; trainierter Edit-Tree-Lemmatizer |
| simplemma | 2.0.0 | `lang="de"`, Standard (`greedy=False`); im Testset auch `greedy=True` |

Alles lief auf der CPU (16 Threads, keine GPU), spaCy in einem Prozess.

## Ergebnisse

### Geschwindigkeit
3.000 zufällige KWIC-Treffer aus `output/punkt/kwic_hits_bundestag_punkt.csv` (Seed 42, 440.000 Zeichen, 72.000 Wörter). Jedes System läuft komplett mit eigenem Tokenizer.

| System | Zeit | Wörter/s | Faktor zu Stanza |
|---|---|---|---|
| Stanza | 24,5 s | 2.950 | 1 |
| spaCy | 3,2 s | 22.800 | 7,7 |
| simplemma | 0,06 s | 1,3 Mio. | 438 |

Hochgerechnet auf 100.000 Treffer: Stanza etwa 14 Minuten, spaCy etwa 2 Minuten, simplemma etwa 2 Sekunden.

### Testset schwieriger Wörter (`testset.py`, 103 Wörter im Satz)

| Kategorie | n | Stanza | spaCy | simplemma | simplemma greedy |
|---|---|---|---|---|---|
| Movierte Feminina (`Soldatinnen`) | 15 | 15 | 7 | 14 | 14 |
| Genderzeichen roh (`Soldat*innen`) | 5 | 0 | 0 | 1 | 1 |
| Plural/Umlaut | 14 | 13 | 13 | 14 | 13 |
| Kontextabhängig (`Steuern`/`steuern`) | 20 | 18 | 17 | 18 | 18 |
| Verben | 14 | 14 | 12 | 13 | 13 |
| Adjektive | 7 | 7 | 7 | 7 | 7 |
| Nominalisiert (`Geflüchteten`) | 8 | 6 | 0 | 8 | 8 |
| Komposita | 12 | 11 | 8 | 9 | 11 |
| Eigennamen, Genitiv (`Putins`) | 8 | 5 | 6 | 3 | 3 |
| **Gesamt** | **103** | **89** | **70** | **87** | **88** |

- spaCy schneidet bei movierten Feminina falsch ab (`Soldatinn`, `Polizistinne`) und schreibt nominalisierte Adjektive klein (`abgeordneter`, `arbeitslose`). Auch bei Verben und Komposita erzeugt es Unformen (`wussen`, `Lieferkett`).
- simplemma lässt Genitive von Namen stehen (`Putins`, `Chinas`, `Selenskyjs`) und ebenso unbekannte Komposita (`Drohnenabwehrsysteme`).
- Rohe Genderzeichen kann kein System. Der Mapper der Pipeline (`make_context_mapper`) bleibt in jedem Fall nötig.

### Korpusstichprobe (dieselben 3.000 Treffer, Wort für Wort)
Alle drei Systeme bewerten dieselben Tokens (die Wörter von Stanza nach `mwt`). Satzzeichen sind ausgelassen, wie in der Pipeline.

| Übereinstimmung der Lemmata | alle Wörter | Inhaltswörter |
|---|---|---|
| Stanza = spaCy | 95,6 % | 93,3 % |
| Stanza = simplemma | 97,3 % | 94,4 % |
| spaCy = simplemma | 95,1 % | 92,1 % |
| alle drei gleich | 94,2 % | 90,4 % |

Wortarten Stanza = spaCy: 92,8 % (Inhaltswörter 87,8 %). Genus bei Nomen Stanza = spaCy: 92,3 %.

**Bewertung gegen Goldlemmata.** Die Grundlage sind die 200 häufigsten Inhaltswörter, bei denen die Systeme nicht alle übereinstimmen (`make_gold.py`). 8 Typen ließen sich ohne Kontext nicht entscheiden und sind ausgelassen. Bewertet werden 192 Typen mit 1.130 Vorkommen. Die Werte gelten nur für diese strittigen Fälle und sind deshalb niedrig.

| | Stanza | spaCy | simplemma |
|---|---|---|---|
| streng | 68,6 % | 69,2 % | 70,0 % |
| alte Rechtschreibung (ß) als richtig gezählt | **82,4 %** | 69,6 % | 72,5 % |

Fehlerarten (streng, Vorkommen):

| Fehlerart | Stanza | spaCy | simplemma |
|---|---|---|---|
| alte Rechtschreibung | 156 | 4 | 28 |
| Form unverändert gelassen | 64 | 35 | 122 |
| Nomen klein geschrieben | 6 | 135 | 2 |
| sonstiger Fehler | 129 | 174 | 187 |

Stanzas häufigster Fehler ist die alte Rechtschreibung (`Rußland`, `bewußt`, `Haß`, `Ausschuß`). Der Fehler ist systematisch, schadet der PPMI-Zählung deshalb nicht und ließe sich mit einer Ersetzungsliste nach Stanza beheben. In der Anzeige der Nachbarwörter im Bundle stehen diese Schreibungen aber. simplemma gibt Wörter, die nicht im Wörterbuch stehen, unverändert zurück (`Tornadoflugzeugen`, `weltpolitischen`). Bei seltenen Wörtern verteilen sich die Zählungen dann auf mehrere Formen.

## Grenzen
- Das Goldlemma stammt von einer Person (Claude) und wurde pro Typ vergeben, nicht pro Vorkommen. Kontextabhängige Typen wie `gehört` (gehören oder hören) sind ausgelassen.
- Bewertet sind nur strittige Fälle. Über Wörter, bei denen alle drei gleich (und gleich falsch) liegen, sagt der Test nichts.
- Die Texte gingen ohne den Kontext-Mapper der Pipeline in den Test. Mit Mapper würden z. B. `Soldatinnen` (Form der Zielwortliste) vorab auf `Soldat` gesetzt. Das betrifft vor allem die 71 Vorkommen von `Soldatinnen` in der Stichprobe, bei denen spaCy scheitert.

## Dateien

| Datei | Inhalt |
|---|---|
| `testset.py` | die 103 schwierigen Wörter mit Satz und zulässigen Lemmata |
| `compare.py` | führt alle drei Systeme aus und schreibt `results/` |
| `make_gold.py` | Goldlemmata der 200 häufigsten strittigen Typen, mit Annotationsregeln |
| `score.py` | bewertet die Systeme gegen die Goldlemmata |
| `results/hard_cases.csv` | Testset: Lemma und Wortart je System |
| `results/speed.json` | Laufzeiten |
| `results/agreement.json` | Übereinstimmung je Wortart, Wortart und Genus Stanza/spaCy |
| `results/diff_types.csv` | alle strittigen Inhaltswort-Typen mit Häufigkeit |
| `results/gold_corpus_types.csv` | die 200 Typen mit Goldlemma |
| `results/score.json` | Bewertung, streng und tolerant, mit Fehlerarten |

## Neu ausführen
simplemma ist keine Projektabhängigkeit. Es wird deshalb in einen eigenen Ordner installiert. Alle Befehle laufen im Repo-Wurzelverzeichnis:

```bash
uv pip install --python .venv/bin/python --target /tmp/simplemma-lib simplemma
PYTHONPATH=/tmp/simplemma-lib .venv/bin/python internal/lemmatizer-comparison/compare.py 3000
.venv/bin/python internal/lemmatizer-comparison/make_gold.py
.venv/bin/python internal/lemmatizer-comparison/score.py
```

`make_gold.py` ordnet die Goldlemmata der Reihenfolge in `diff_types.csv` zu. Ändert sich die Stichprobe oder die Version eines Modells, muss man die Goldliste prüfen und neu annotieren.
