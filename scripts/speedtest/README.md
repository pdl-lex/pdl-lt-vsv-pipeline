# Speedtest

Misst die rechenintensiven Schritte der Pipeline auf einem festen Datenauszug, um Rechner zu
vergleichen. Dauer auf dem Referenzrechner (`framework`, Ryzen AI Max+ 395, 32 Threads, nur CPU): **ca. 2:50 min**.

| Schritt | Was | Referenz |
|---|---|---|
| 1 KWIC, festes Fenster | `read_corpus_rows` + `normalize_chars` + `kwic_window`, jede 2. Rede (190 MB) | 16 s |
| 2 KWIC, Satzgrenzen | wie 1, mit `kwic_sentence` (NLTK Punkt); Differenz zu 1 = Kosten der Satzgrenzen | 33 s |
| 3 Stanza laden | `stanza.Pipeline` aus `assets/` | 1,5 s |
| 4 Lemmatisierung | `make_context_mapper` + `lemmatize_batch`, 12.000 zufällige Treffer | 101 s |
| 5 PPMI | PPMI aus den vollständigen Kookkurrenzen (punkt/halfyear), CSR-Matrix 12.280 × 120.882 | 0,5 s |
| 6 SVD | `svds`, k = 300 | 13 s |

## Aufruf (aus dem Repo-Wurzelverzeichnis)

```bash
uv run python internal/speedtest/speedtest.py              # Test
uv run python internal/speedtest/speedtest.py --label akku # mit Zusatz im Ergebnisnamen
uv run python internal/speedtest/speedtest.py --prepare    # Datenauszug neu erzeugen
```

Wichtig: Im Ordner data vorher die Datei corpus_UNZIPME.zip zu corpus.csv entpacken.

## Auf einem anderen Rechner

Nötig sind das Repo mit `uv sync` und den Modellen in `assets/` (`scripts/fetch_assets.py`) und der
Ordner `internal/speedtest/data/` (179 MB). Den Ordner einfach mitkopieren, die Korpusdatei
wird nicht gebraucht. `--prepare` geht nur dort, wo die vollständigen Pipeline-Daten liegen.

## Ergebnisse

- `results/<host>_<label>_<zeit>.json`: alles zu einem Lauf (Systeminfo, Zeiten, Hochrechnung)
- `results/results.csv`: eine Zeile pro Lauf (Trennzeichen `;`), zum Vergleich der Rechner

Die Ausgabe rechnet auf den vollen Bundestagslauf hoch (KWIC, Lemmatisierung). Auf dem
Referenzrechner passt das zu den gemessenen Werten in `internal/Dauer der Rechenschritte`
(1:11 min bzw. 1:35 h).

**Kontrollwerte:** 345.176 Treffer (Punkt), Matrix 12.280 × 120.882, `S[0] = 1204.8203`. Weichen sie ab,
haben die Rechner nicht dasselbe gerechnet (andere Daten oder Bibliotheksversionen), und die Zeiten sind nicht vergleichbar.

**Hinweise:**
- Stanza nutzt automatisch eine CUDA-GPU, wenn es eine findet; das zeigt die Zeile `GPU (Stanza)`.
  Die Lemmatisierung ist dann um ein Vielfaches schneller. Das entspricht dem echten Pipeline-Lauf.
- Ändert man `CORPUS_EVERY_NTH` oder `N_LEMMA_HITS`, muss `DATA_VERSION` hochgezählt und
  `--prepare` neu ausgeführt werden; alte Ergebnisse sind dann nicht mehr vergleichbar.
- `kwic_sentence`, `kwic_window` und die PPMI-Berechnung stehen in den Notebooks nur in Zellen und
  sind deshalb in `speedtest.py` kopiert (`KOPIE aus ...`). Ändern sie sich in den Notebooks, hier nachziehen.
