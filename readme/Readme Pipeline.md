# Pipeline: Analyse-Kette an echten Daten erproben

> **Stand 2026-11-20: Dieses Dokument beschreibt überwiegend die frühere
> Ein-Notebook-Phase** (September/Oktober 2026: `pipeline.py`,
> `pipeline-diagnose.py`, Quartale, 11 Zielwörter, k = 100, p = 1). Die dort
> genannten Notebooks (`pipeline.py`, `pipeline-diagnose.py`,
> `pipeline-sgns.py`) gibt es nicht mehr; ebenso gelten Angaben wie
> „`output/` ist versioniert“ nicht mehr. Die SGNS-Pipeline liegt heute im
> Repo `pdl-lex/zeitenwende-sgns`. Aktuell gilt:
>
> - **Aufbau:** drei Stufen unter `stages/` (pre, emb, post), s. `README.md`.
> - **Diagnose der Embedding-Stufe:**
>   `stages/02_embedding/pipeline_emb_ppmi_diagnose.py` mit vier Abschnitten:
>   (1) Mindesttrefferzahl h\* einer Zeile, (2) SVD ohne schwach belegte
>   Zeilen, (3) k und p mit Volumeneffekt, (4) Abhängigkeit der Drift von der
>   Trefferzahl. Letzter vollständiger Lauf 2026-11-19 (Export
>   `pipeline_emb_ppmi_diagnose.html`): h\* = 30, `MIN_HITS_FIT = None`,
>   k = 300, p = 0,5, Drift mit Teilstichprobe je Paar. Die frühere Fassung
>   (`pipeline_emb_ppmi_diagnose_alt.py`, Messreihe C: CDS, additive
>   Glättung, k und p) ist am 2026-11-20 gelöscht; ihre Ergebnisse stehen in
>   Methodology 2.5, 2.6 und 1.8.
> - **Methodische Entscheidungen und Messreihen:** `Readme Scientific
>   Methodology.md`; Kontrolle der Trefferzahl: `Readme Frequency
>   Control.md`; Begriffe: `Readme Glossary.md`; offene Umsetzungsschritte:
>   `nextsteps.md`.
>
> Der Rest bleibt als Protokoll der Entstehung stehen: Er erklärt, warum
> etwas so gebaut wurde. Zahlen daraus (z. B. r = 0,208, 40,9 %
> Frobenius-Anteil) gehören zur Messreihe A und sind mit den heutigen nicht
> vergleichbar.

Testet die für `pdl-lt-vsv` vorgesehenen Analyse-Ketten einmal an echten,
frei zugänglichen Daten, bevor die Workbench weitere Visualisierungs-Features
bekommt. Hintergrund und Alternativen-Abwägung:
`.claude/memory/corpus-decision.md` im Hauptrepo.

Es sind **zwei Verfahren**, die dasselbe Bundle-Schema füllen und sich dadurch
gegenseitig kontrollieren lassen:

- **PPMI-SVD** (KWIC → Lemma → Kookkurrenz → PPMI → SVD → Zeitscheiben) — das
  Hauptverfahren laut Projektplan. Vektoren werden *konstruiert*:
  deterministisch, ein gemeinsamer Raum, kein Alignment.
- **SGNS + Procrustes** (Volltext → Lemma → ein Modell je Zeitscheibe →
  orthogonale Ausrichtung) — das im Projektplan als Option vorgesehene
  Kontrollverfahren. Vektoren werden *trainiert*: stochastisch, je Scheibe ein
  eigener Raum, der nachträglich ausgerichtet wird.

Was beide teilen und worin sie sich unterscheiden, steht unter „Zwei Verfahren
im Vergleich". **Alles Weitere ist nach Pfad getrennt** — Ablauf, Stolpersteine
und offene Punkte stehen jeweils im Abschnitt des eigenen Verfahrens, und die
Überschrift sagt, welcher gemeint ist. Zahlen ohne Pfadangabe (Korpusgröße,
Zeitscheiben, Zielwortliste) gelten für beide.

Kein Batch-Skript, sondern **marimo-Notebooks** — reaktiv, git-freundlich als
reines `.py`. Start:

```
uv run marimo edit pipeline.py
```
## Setup

- `uv sync` installiert `marimo`, `polars`, `numpy`, `spacy`, `gensim` und das
  spaCy-Modell.
- **Python 3.13**, nicht 3.14 (`.python-version`). Grund ist allein `gensim`:
  Version 4.4.0 liefert Wheels bis `cp313`, für 3.14 gibt es nur ein sdist, das
  gegen Cython/C gebaut werden müsste. Der Rest des Stacks (polars, marimo,
  spaCy 3.8, numpy) läuft auf 3.13 unverändert. Der Downgrade am 2026-09-04 ist
  **gegen die dokumentierten Kennzahlen des PPMI-Pfads geprüft**: 544.678
  Kookkurrenz-Zeilen, Matrix 502 × 64.165, Frobenius-Anteil 40,9 % bei k=100 —
  alle drei identisch zu vorher. Nach dem Wechsel einmal `uv sync` (baut
  `.venv` neu, `uv.lock` ändert sich).
- Das Modell `de_core_news_md` ist in `pyproject.toml` per `[tool.uv.sources]`
  auf ein lokales Wheel in `assets/` gepinnt. `assets/` ist **nicht versioniert**
  (44 MB), das Wheel also nach einem frischen Klon einmalig beschaffen:
  `curl -L -o assets/de_core_news_md-3.8.0-py3-none-any.whl https://github.com/explosion/spacy-models/releases/download/de_core_news_md-3.8.0/de_core_news_md-3.8.0-py3-none-any.whl`
  Erst danach läuft `uv sync` durch.
- Quelldaten liegen in `source_data/` (nicht versioniert, ca. 2,9 GB), Herkunft
  in `source_data/quelle.txt`:
  **CPP-BT — Corpus der Plenarprotokolle des Deutschen Bundestages**,
  Fobbe, S. (2026), Version 2026-01-17, Zenodo,
  [10.5281/zenodo.18177196](https://doi.org/10.5281/zenodo.18177196),
  CC BY-SA-4.0. Vier CSVs (Reden/Protokolle × Metadaten/Volltext) + Codebook-PDF.
  Verwendet wird nur `CPP-BT_..._DE_CSV_Reden_Gesamt.csv` (Reden mit Volltext,
  35 Spalten inkl. `rede_text`) — Protokolle enthalten zu viel Sitzungs-Overhead,
  Reden reichen für Metadaten (Datum, Redner, Fraktion) + Volltext.
- `output/` **ist** versioniert: da die Quelldaten fehlen, wären die
  Zwischenstände sonst in einem Klon nicht rekonstruierbar. Inhalt s.u.
- `cache/` ist **nicht** versioniert und nimmt die Zwischenstände des
  SGNS-Pfads auf — gemessen: lemmatisierter Volltext 167 MB (14 Scheiben),
  Wortvektoren 14 MB je Modell × 36 Modelle (12 volle + 24 Hälften für die
  Auflösungsmessung) ≈ 480 MB, zusammen **rund 640 MB**. Anders als die
  KWIC-Artefakte in `output/` (17 MB) gehört das nicht in git.
  Rekonstruierbar ist es nur mit `source_data/`, also unter derselben
  Einschränkung wie die Quelldaten selbst. Faustregel für diesen Pfad: alles,
  was länger als ~2 Minuten rechnet, wird nach `cache/` geschrieben und beim
  nächsten Lauf übersprungen; alles darunter (die Kontrast-Tabellen brauchen
  0,2 s) wird jedes Mal frisch berechnet.

## Drei Notebooks: Kern-Pipeline, Diagnose, SGNS

> *Historisch (s. Hinweis oben); die hier beschriebenen Notebooks sind
> gelöscht.* Heute: Stufen-Notebooks unter `stages/`;
> Diagnose der Embedding-Stufe in
> `stages/02_embedding/pipeline_emb_ppmi_diagnose.py`.

- **`pipeline.py`** — die Kern-Pipeline von den Quelldaten bis zum
  JSON-Export (`output/bundestag_zeitenwende_*.json`). Enthält die beiden
  produktiv genutzten Experimentier-Schalter (Zeilennormierung vor der SVD,
  POS-/Stopword-Filter der Kontextwörter) sowie einen eigenen Abschnitt
  „Kurationswerkzeug", in dem die beiden Ranglisten stehen, mit denen
  `anchors.csv` bei Bedarf neu kuratiert wird — kein Bundle-Output, reines
  Kurationshilfsmittel.
- **`pipeline-diagnose.py`** — die Optimierungs-Diagnosen, die zur
  Modellierung geführt haben, aber im produktiven Durchlauf nichts zu suchen
  hatten: Sparsity-Effekt der Vokabularschwelle, der große
  Redevolumen-Effekt (die Messlatte hinter mehreren Funden im
  „Stolpersteine"-Abschnitt unten) und eine Anker-Kontrolle an bekannten
  Zeitenwende-Kontrollwörtern (Sondervermögen, Zeitenwende, Beschaffung, …).
  Läuft eigenständig: liest `output/kwic_lemmatisiert.parquet` direkt (spart
  den ~6:32-min-spaCy-Lauf) und baut Kookkurrenz/PPMI/SVD frisch mit eigenen
  Kopien der Schalter — inklusive der Vokabularschwelle, die in `pipeline.py`
  selbst fest deaktiviert ist. Default-Zeitscheiben-Raster ist hier
  **Quartale**, nicht Halbjahre wie im Hauptnotebook, damit die Zahlen mit
  den unten dokumentierten Messreihen vergleichbar bleiben. Erzeugt kein
  Bundle.

- **`pipeline-sgns.py`** — der zweite Verfahrensweg (SGNS + Procrustes),
  im Aufbau seit 2026-09-04. Teilt mit `pipeline.py` nur die Quelldatei und
  das Ziel-Schema, sonst nichts: er liest den **Volltext** aller Reden statt
  der KWIC-Fenster und braucht deshalb eine eigene, viel größere
  Vorverarbeitung. Warum das ein eigenes Notebook ist und kein Schalter in
  `pipeline.py`, steht unter „Warum zwei Notebooks und nicht ein Schalter".
  Stand und Stufenplan im Abschnitt „SGNS-Pfad" unten.

Start jeweils wie oben: `uv run marimo edit pipeline-diagnose.py`.

### Warum zwei Notebooks und nicht ein Schalter

Vier Gründe, in aufsteigender Härte:

1. **Der Input ist ein anderer.** Alles in `pipeline.py` oberhalb der
   Kookkurrenz — KWIC-Extraktion, Suchwortliste, Fenster-Lemmatisierung — ist
   für SGNS gegenstandslos. Ein Options-Schalter hieße: auf jedem Zweig ist
   ungefähr die Hälfte der Zellen tot.
2. **marimo bestraft es.** Jede Variable darf im Notebook nur einmal definiert
   sein; ein zweiter Verfahrenszweig erzwingt Präfixe auf allem — genau der
   Schmerz, aus dem in `pipeline-diagnose.py` schon die `diag_`-Präfixe
   entstanden sind. Und ein `if` *innerhalb* der Zellen nimmt dem reaktiven
   Graphen seinen Zweck: Zellen, die stillschweigend nichts tun.
3. **Messhygiene.** Die Stolpersteine und „Nächste Schritte" des PPMI-Pfads
   stehen auf einer dokumentierten Messreihe (40,9 % Frobenius-Anteil,
   r = 0,208, die fünf Kontrollränge). Ein Umbau von `pipeline.py` riskiert,
   dass diese Zahlen nicht mehr reproduzieren.
4. **Die Laufzeiten liegen um Größenordnungen auseinander.** Der PPMI-Pfad
   lebt davon, dass die teuren Zellen `disabled=True` sind und aus `output/`
   nachgeladen wird. Der SGNS-Pfad hat andere teure Stellen an anderen
   Positionen. Beides im selben reaktiven Graphen zu halten, ist eine
   verlässliche Quelle versehentlicher Stundenläufe.

## Zwei Verfahren im Vergleich

Beide arbeiten auf demselben Korpus, denselben Zeitscheiben und derselben
Zielwortliste, und beide füllen dasselbe Bundle nach `data/test/SCHEMA.md`.
Der Unterschied sitzt an genau einer Stelle: **wie ein Wortvektor entsteht.**

| | PPMI-SVD (`pipeline.py`) | SGNS + Procrustes (`pipeline-sgns.py`) |
|---|---|---|
| Vektor entsteht durch | Zählen + eine Matrixzerlegung | iteratives Training |
| Determinismus | vollständig | seed-abhängig |
| Räume | einer, alle Scheiben in einer Matrix | einer je Scheibe, nachträglich ausgerichtet |
| Alignment | keins nötig | orthogonales Procrustes, zwingend |
| Textbasis | ±40-Wörter-Fenster um 11 Zielwörter | Volltext aller Reden |
| Tokenmenge | 3,24 Mio. | 50,85 Mio. |
| Vektor eines Kontextworts | fällt als Spaltenseite ab (`Vt[:k].T`) | wird eigenständig geschätzt |
| Scheibenbreite realistisch | Quartal, Halbjahr, Jahr | nur Jahr (s. u.) |
| teuerste Stelle | SVD (3:16 min bei Quartalen) | Volltext-Lemmatisierung (einmalig) |

Die beiden liegen nicht auf einer Skala „grob/fein", sondern haben
**verschiedene blinde Flecken**: der PPMI-Pfad kann nur messen, was er gezählt
hat, und er zählt ausschließlich um die Zielwörter herum. Der SGNS-Pfad sieht
den ganzen Diskurs, zahlt das aber mit Stochastik und einem Ausrichtungsschritt,
der selbst eine Fehlerquelle ist. Genau deshalb ist der Vergleich informativ —
Übereinstimmung wäre bei zwei Verfahren mit *demselben* blinden Fleck kein
Argument.

### Warum der KWIC-Auszug für SGNS nicht reicht

Die naheliegende Abkürzung — SGNS einfach auf `output/kwic_lemmatisiert.parquet`
trainieren, die Lemmata liegen ja fertig vor — trägt aus vier Gründen nicht:

1. **PPMI-SVD schätzt 502 Zeilen, SGNS schätzt jedes Wort einzeln.** Die
   PPMI-Matrix hat 11 Zielwörter × ~50 Scheiben als Zeilen; die 64.165
   Kontextwörter bekommen ihren Vektor *geschenkt*, als Spaltenseite derselben
   Zerlegung (Ablauf-Schritt 18). Nichts daran wird eigenständig geschätzt.
   SGNS ist das Gegenteil: es schätzt für **jeden** Vokabeleintrag einen
   Parametervektor aus *dessen eigenen* Kontexten. Im KWIC-Material kommt
   „Haushalt" aber nur vor, wenn es zufällig innerhalb ±40 Wörter eines der 11
   Zielwörter steht. Das ist keine Stichprobe davon, wie „Haushalt" gebraucht
   wird, sondern davon, wie es *in der Nähe von Bundeswehr/Sicherheit/…*
   gebraucht wird. Jeder Vektor im Modell wäre auf die Zielwortliste
   konditioniert — und damit auch jede Nachbarschaftsliste, die man ihm
   entlockt.
2. **Procrustes braucht genau diese Wörter als Stützpunkte.** Die
   Rotationsmatrix wird aus dem geteilten Vokabular zweier Scheiben geschätzt
   (üblich: die häufigsten paar tausend Wörter). Deren Vektoren müssen *stabil*
   sein, sonst richtet man Rauschen an Rauschen aus. Schätzt man sie aus dem
   Militärdiskurs-Fenster, wird die Ausrichtung ausgerechnet an dem Vokabular
   gefittet, dessen Bewegung gemessen werden soll — das Kontrollverfahren wäre
   zirkulär.
3. **Volumen.** Der KWIC-Auszug hat 3,24 Mio. Kontext-Token über 50 Quartale,
   also ~250.000 je Jahr. Ein trainiertes Modell auf 250.000 Token ist Rauschen
   (Zahlen im nächsten Abschnitt).
4. **Die Fenster überlappen, und am Zielwort sind sie durchtrennt.** Die 42.252
   Treffer stammen aus nur 15.785 verschiedenen Reden — viele Fenster
   überlappen sich also und zählen dieselben Token mehrfach. Für eine
   Zähl-Matrix ist das eine dokumentierte Konvention (Ablauf-Schritt 8), für
   SGNS verzerrt es unsichtbar die Unigramm-Verteilung, aus der Negative
   Sampling und Subsampling ihre Wahrscheinlichkeiten ziehen. Dazu kommt:
   `kontext_davor` und `kontext_danach` enthalten das Zielwort selbst nicht.
   Man würde auf Text trainieren, in dem „Bundeswehr" kein einziges Mal
   vorkommt.

### Korpusgröße — sie entscheidet über die Zeitscheibenbreite

Zahlen aus `CPP-BT_..._Reden_Metadaten.csv` (Spalte `tokens`), Korpusversion
2026-01-17:

| | KWIC-Kontext (PPMI-Pfad) | Volltext (SGNS-Pfad) |
|---|---|---|
| Token gesamt | 3.236.709 | 50.852.653 |
| Reden | 15.785 | 78.188 |
| je Jahr | ~250.000 | ~4.400.000 |
| je Halbjahr | ~130.000 | ~2.200.000 |
| je Quartal | ~65.000 | ~1.100.000 |

Der KWIC-Auszug ist **6,4 %** der Tokenmasse des Korpus.

Für den **PPMI-Pfad** ist die Scheibenbreite frei wählbar — es liegen drei
Bundles nebeneinander (Quartal/Halbjahr/Jahr). Zählen funktioniert auch bei
dünner Besetzung, es wird nur unsicherer, und genau das transportiert `freq`
ins Frontend.

Für den **SGNS-Pfad** ist sie es nicht. Ein trainiertes Modell braucht je
Scheibe genug Text, damit die Gewichte konvergieren, und word2vec-
Nachbarschaftslisten sind bei kleinen Korpora schon zwischen zwei Seeds
instabil (Antoniak & Mimno 2018, *Evaluating the Stability of Embedding-based
Word Similarities*). Mit ~4,4 Mio. Token sind **Jahresscheiben** realistisch,
Halbjahre (2,2 Mio.) dünn, Quartale (1,1 Mio.) fallen aus. Zusammen mit dem
Randscheiben-Ausschluss — 2013 hat 176.792 Token, 2026 nur 138.451, beides
Korpus-Stichtagsartefakte — bleiben **2014–2025, zwölf Scheiben**.

Dieselbe Tabelle wirft die Downsampling-Frage auf: das dünnste volle Jahr ist
2017 mit 2,78 Mio. Token, das dichteste 2023 mit 5,02 Mio. — Faktor 1,8. Für
PPMI ist das verkraftbar (die Randsummen kürzen sich heraus), für ein
trainiertes Modell nicht ohne Weiteres, weil die Vektorqualität direkt an der
Trainingsmenge hängt. Der Projektplan sieht dafür den eigenen Schritt
„Downsampling und Kontrollset" vor; entschieden ist er noch nicht.

### Der Vor-Ort-Schnitt (DNB-Szenario)

Der Projektplan markiert zweimal dieselbe Stelle mit
`### Hier Mitnahme der KWIC aus der DNB, Ende des Vor-Ort-Einsatzes? ###` —
einmal hinter der KWIC-Extraktion, einmal hinter der Embedding-Generierung.
Das ist kein Widerspruch, sondern genau die Wahl, die die beiden Verfahren
voneinander trennt.

Rechtlich gilt: die Volltexte dürfen **vor Ort analysiert**, aber nicht
mitgenommen werden. Was das Haus verlässt, sind abgeleitete Daten — KWIC-
Auszüge im vereinbarten Umfang, Zählungen, Vektoren. Beide Verfahren sind
damit im DNB-Szenario grundsätzlich durchführbar, sie schneiden nur an
verschiedenen Stellen:

- **PPMI-Pfad — Schnitt früh**, direkt hinter der KWIC-Extraktion. Alles
  danach (Lemmatisierung, Kookkurrenz, PPMI, SVD, Bundle) läuft auf eigener
  Hardware und ist beliebig oft wiederholbar. Der Preis ist das
  ±40-Wörter-Fenster.
- **SGNS-Pfad — Schnitt spät**, hinter dem Training. Vorverarbeitung *und*
  Modelltraining müssen auf den Rechnern vor Ort laufen; mitgenommen werden
  die Vektoren plus die Häufigkeiten.

Die Folge ist keine rechtliche, sondern eine **organisatorische**: der
SGNS-Pfad verlegt die teuerste und iterationsbedürftigste Stufe an einen Ort,
an dem nicht beliebig oft nachgebessert werden kann. Hyperparameter
(`vector_size`, `window`, `min_count`, `epochs`, Seed), die
Vokabularentscheidung und das Downsampling müssen also **vor** dem
Vor-Ort-Einsatz feststehen. Genau dafür ist der Bundestags-Durchlauf da: ein
frei verfügbares Korpus mit derselben Struktur, an dem der Parametersatz und
der Rauschboden ermittelt werden, die man dann mitbringt.

Ein Detail, das daraus folgt: mitgenommen werden Vektoren, nicht Text — also
muss **alles, was das Bundle braucht, aus Vektoren plus Zählungen
rekonstruierbar sein**. Für `aggregates`, `anchors`, `neighbors`, `coords_3d`
und `freq` trifft das zu. Für alles, was auf den Rohtext zurückgreift
(Belegstellen, eine KWIC-Ansicht im Frontend), trifft es nicht zu — das
braucht den KWIC-Auszug zusätzlich, unabhängig vom gewählten Verfahren.

### Die Nachbarlisten der beiden Pfade sind nicht vergleichbar

**Gemessen 2026-09-04, Jahresraster, Zeitscheibe 2022.** Für `bundeswehr`,
`drohne` und `sicherheit` ist die Schnittmenge zwischen den `neighbors` des
PPMI-Bundles und den Top-10 des SGNS-Modells **leer** — nicht klein, leer.

| Wort (2022) | PPMI | SGNS |
|---|---|---|
| `bundeswehr` | Kameradschaft, Jahresbericht, Stab, Wehrbeauftragter, verrichten, vorgesetzter, Soldatenberuf, Spiegelbild | Streitkraft, Ausrüstung, Auslandseinsatz, Sondervermögen, Landesverteidigung, Einsatzbereitschaft, … |
| `drohne` | kostenfrei, Faso, Burkina, Felgentreu, billigen, Jürgen, zurücktreten, Terrorregime | bewaffnet, Bewaffnung, Tornado, Kampfhubschrauber, Kampfjet, Luftwaffe, … |
| `sicherheit` | sicherst, Fahrer, Horst, einreisen, Weiterbildung, Verbraucherin, Nährboden, Straßenverkehr | innerer, äußerer, Gewährleistung, Stabilität, Frieden, Zivilschutz, … |

Die naheliegende Erklärung — „PPMI liefert syntagmatische, SGNS
paradigmatische Nachbarn" — ist nur zur Hälfte richtig und wurde deshalb
gegengeprüft: von den zehn SGNS-Nachbarn von `Bundeswehr` stehen **sieben
tatsächlich häufig im Textfenster** daneben (`Sondervermögen` auf Fensterrang
4, `Soldatinn` 17, `Ausrüstung` 43, `Verteidigungsfähigkeit` 76,
`Streitkraft` 80). SGNS liefert also eine Mischung aus Quasi-Synonymen und
charakteristischen Begleitwörtern, nicht reine Paradigmatik.

Der wirkliche Unterschied ist ein anderer: **PPMI belohnt Exklusivität, nicht
Charakteristik.** `Sondervermögen` steht am vierthäufigsten neben
`Bundeswehr`, hat aber auch eine große Randsumme — PMI also mittel.
`Felgentreu` und `Burkina` kommen fast nur an einer Stelle vor und schlagen
die Decke durch. Das ist der PMI-Deckeneffekt (s. „Offen / später zu klären")
in seiner Auswirkung auf die Nachbarliste.

Zur Fairness gehört dazu: gemessen wurde am **Jahres**-Bundle mit k = 44 auf
132 Zeilen. Für genau diese Konfiguration sagt der PPMI-Pfad selbst voraus,
dass sich der Echo-Effekt auf der Spaltenseite verschärft (s. Kommentar zur
k-Wahl in `pipeline.py`). Der Vergleich läuft also gegen PPMI in seiner
schwächsten Auflösung — die aber die einzige ist, in der SGNS überhaupt
arbeiten kann. **„SGNS schlägt PPMI" ist daraus nicht ableitbar**, wohl aber:
die beiden Felder heißen gleich und bedeuten Verschiedenes.

Zwei Konsequenzen, beide für den Stufenplan bindend:

1. **Der Methodenvergleich darf nicht auf Nachbarlisten aufbauen.** Eine
   Jaccard-Überlappung zwischen den `neighbors` beider Bundles käme nahe null
   heraus, egal ob die Verfahren inhaltlich übereinstimmen oder nicht — sie
   misst dann die Definition, nicht die Daten. Der Vergleich muss auf den
   **Aggregatmaßen** ruhen: Driftkurven (Kosinus eines Worts zu sich selbst
   über die Zeit) und Kiyama-Matrizen. Die rechnen über alle Dimensionen und
   sind gegenüber der Nachbar-Definition unempfindlich.
2. **Im Frontend muss der Unterschied benannt sein.** Wer zwischen den Bundles
   umschaltet, sieht zwei völlig verschiedene Nachbarpanels für dasselbe Wort
   und dieselbe Scheibe. Ohne Hinweis liest sich das als Fehler.

### Was beide Pfade teilen

- **Das Ziel-Schema** `data/test/SCHEMA.md`. Beide exportieren dieselben
  Felder; unterschieden werden die Bundles über `meta.projection` und den
  Dateinamen. Nur so kann das Frontend zwischen ihnen umschalten und die
  bereits gebauten Module (Drift, Kiyama-Matrizen, Nachbarschafts-Fluss) den
  Vergleich übernehmen, ohne dass dafür Code entsteht.
- **Zielwortliste und Zäsur** — 12 Suchwörter, 11 Lemmata, `shift_point`.
- **Die Lemmatisierung**: dasselbe `de_core_news_md` und derselbe POS-Filter
  (`DET, ADP, PRON, AUX, CCONJ, SCONJ` raus, `PART` bleibt — dort steckt im
  Deutschen die Negation). Nicht aus Sparsamkeit, sondern damit die Vokabulare
  vergleichbar bleiben und `anchors.csv` in beiden Pfaden greift. Ein anderes
  Lemma-Set würde die Anker-Zuordnung still zerreißen, ohne Fehlermeldung.
- **`anchors.csv`** als kuratierte Landmarkenliste (`lemma,label,cluster`).
- **Der Randscheiben-Ausschluss.**
- **Die Bedeutung von `freq`**: auch im SGNS-Bundle bleibt `freq` die Zahl der
  Zielwort-Treffer in der Zelle — *nicht* die Trainings-Tokenmenge. Sonst
  bedeutet dasselbe Feld in zwei Bundles etwas Verschiedenes, und die
  Konfidenz-Kodierung des Frontends (Punktgröße ∝ `freq`) vergleicht Äpfel mit
  Birnen.

## Ablauf in `pipeline.py` (PPMI-Pfad)

1. **Quelldaten sichten** — Datei-Picker (`mo.ui.dropdown`) für die vier CSVs,
   Vorschau der ersten 100 Zeilen.
2. **Korpus laden** — `korpus`: nur die relevanten Spalten aus
   `Reden_Gesamt.csv` (`rede_id`, `sitzung_datum`, `redner_vorname/-nachname`,
   `redner_fraktion`, `redner_rolle_kurz`, `tokens`, `rede_text`).
3. **Suchwörter** — `suchwoerter`, ein `set` (nicht `list` — O(1) statt O(n)
   beim Mitgliedschaftstest), aktuell 12 unflektiert kuratierte Substantive
   rund ums Zeitenwende-Wortfeld (Bundeswehr, Sicherheit, Verteidigung, Frieden, …).
   Daraus werden **11** Zielwort-Lemmata: `Drohne` und `Drohnen` lemmatisieren
   beide zu `Drohne`. Kein stiller Ausfall.
4. **Normalisierung** — `normalisiere()` strippt Satzzeichen nur für den
   Match-Check (Regex `^\W+|\W+$`), der Kontext-Text bleibt unangetastet.
5. **KWIC-Extraktion** (Zelle mit `disabled=True`, ~1:20 Min Laufzeit) —
   `kwic_treffer()`: wortweiser Exact-Match (kein Substring-Regex → keine
   Fehltreffer in Komposita wie "Landesverteidigung"), ±40 Wörter Kontextfenster
   (grobe DNB-Vorgabe für erlaubte Extraktionsmenge), eine Ausgabezeile pro
   **Vorkommen**, nicht pro Rede. Ergebnis nach `output/kwic_treffer.csv`
   geschrieben; die Folgezelle liest von dort — Notebook läuft beim Ändern von
   Code nicht ungewollt 80 Sekunden erneut.
6. **Lemmatisierung** — spaCy `de_core_news_md`:
   - `nlp` in eigener Zelle laden (siehe Stolperstein unten).
   - Zielwort-Lemmata (`zielwort_lemmata`): nur die 12 Suchwörter selbst durch
     `nlp.pipe()` gejagt und als Dict gecacht — spart, pro Treffer neu zu
     lemmatisieren, für eine kleine geschlossene Liste unnötig.
   - Kontext-Lemmata (Zelle mit `disabled=True`, ~6:32 Min Laufzeit):
     `kontext_davor` + `kontext_danach` pro Treffer zusammen durch
     `nlp.pipe(..., disable=["parser","ner","senter"])` (nur Tagger+Lemmatizer
     nötig), Satzzeichen (`POS == PUNCT`) verworfen, POS sonst nur miterhoben
     (Nachnutzung laut Projektplan), nicht gefiltert. Ergebnis als
     `output/kwic_lemmatisiert.parquet` — **Parquet statt CSV**, weil die
     neuen Spalten `kontext_lemmata`/`kontext_pos` Listen sind und CSV
     verschachtelte Typen nur verlustbehaftet als Text serialisiert.
7. **Zeitscheiben** — `sitzung_datum` (Text "YYYY-MM-DD") aus `korpus`
   zurück in die KWIC-Tabelle gejoint, zu `pl.Date` geparst, zu Quartals-Label
   verdichtet (`"2022-Q1"`). Verteilung geprüft: 2013-Q4 bis 2026-Q1, 50
   Zeitscheiben. Auffällige Dellen in 2017-Q3/Q4 und 2021-Q3/Q4 — beides Mal
   Bundestagswahl + lange Regierungsbildung (Jamaika/GroKo-Verhandlungen 2017,
   Ampel-Vereidigung erst Dez. 2021), also plausibel echtes, kein defektes
   Signal — für jetzt akzeptiert, aber im Hinterkopf behalten (dünne
   Zeitscheiben → verrauschtere Kookkurrenzbasis).
8. **Kookkurrenz-Tabelle** — `kookkurrenz`: Kontext-Listen per `.explode()`
   entrollt, dann `group_by(["zeitscheibe","zielwort_lemma","kontext_lemma"])`
   gezählt → Langformat-Tabelle (Zeitscheibe, Zielwort, Kontextwort,
   `beobachtet`), 544.678 Zeilen, ~2,3 s. Das ist die rohe Kookkurrenzhäufigkeit
   je Tripel, **nicht** dedupliziert — ein Kontextwort, das zweimal im selben
   Fenster auftaucht, zählt zweimal. Spalte hieß ursprünglich `anzahl`, wurde
   auf `beobachtet` umbenannt, um sie im PPMI-Code klar von `erwartet`
   abzugrenzen (s. Schritt 10).
9. **Pivot-Versuch (Sackgasse, siehe unten)** — eine Zeitscheibe (2022-Q1) zu
   einer Zielwort×Kontextwort-Matrix pivotiert: 11 Zeilen, aber 7.533 Spalten,
   ~1,2 s — langsamer als die komplette Kookkurrenz-Tabelle über alle 50
   Quartale, weil `pivot`-Kosten mit der Spaltenzahl skalieren, nicht mit der
   Zeilenzahl der Quelldaten. Bei fast leerer (dünnbesetzter) Matrix und 50
   Quartalen wäre das strukturell teuer — **Ansatz verworfen**, s. nächste
   Schritte.
10. **PPMI auf der Langform, ohne Pivot** — drei Randsummen-Aggregationen, die
    sich nur darin unterscheiden, welche Spalte im `group_by` fehlt (das
    Fehlen bewirkt das Aufsummieren über genau diese Dimension): Randsumme je
    Zielwort (`group_by(["zeitscheibe","zielwort_lemma"])`), Randsumme je
    Kontextwort (`group_by(["zeitscheibe","kontext_lemma"])`), Gesamtsumme
    (`group_by(["zeitscheibe"])`) — je `sum(beobachtet)`. Per Left-Join an
    `kookkurrenz` zurückgehängt (→ `kookkurrenz_ppmi`; Check-Zelle bestätigt
    gleiche Zeilenzahl wie `kookkurrenz`). Danach `erwartet = randsumme_ziel ×
    randsumme_kontext / gesamt`, `pmi = log2(beobachtet/erwartet)`, `ppmi =
    pmi.clip(lower_bound=0)` (→ `kookkurrenz_ppmi_komplett`). Kein Pivot
    nötig — die Langform ist bereits das COO-Format einer Sparse-Matrix, nur
    mit Wort-Strings statt Integer-Indizes als „Koordinaten".
11. **Ergebnis-Ansicht** — interaktive marimo-Zelle: Dropdown für Zeitscheibe
    (`quartal_picker`) und Zielwort (`zielwort_picker`), Ergebnis
    `ppmi_gefiltert` absteigend nach `ppmi` sortiert, auf 3 Nachkommastellen
    gerundet, auf `kontext_lemma`/`beobachtet`/`ppmi` reduziert.
12. **Zwei Mindesthäufigkeit-Filter** — einzelne PPMI-Ranglisten erwiesen sich
    beim Durchklicken als Artefakte zu geringer Datenmenge (s. Stolperstein
    unten). `mindesthaeufigkeit_kontextwort_picker` filtert auf
    `randsumme_kontext >= X`. `mindesthaeufigkeit_zielwort_picker` filtert auf
    eine eigens berechnete `zielwort_haeufigkeit` — die echte Trefferzahl aus
    `kwic_liste_quartale` **vor** dem Explode
    (`group_by(["zeitscheibe","zielwort_lemma"]).len()`) — **nicht** auf
    `randsumme_ziel`, s. Stolperstein.
13. Erklärender Absatz im Notebook ergänzt („Wie viele Beobachtungen braucht
    PPMI, um verlässlich zu sein?") — die beiden Filter allein machen die
    Interpretation von PPMI-Werten nicht selbsterklärend.

14. **Zwei zusätzliche Assoziationsmaße** — in derselben Zelle wie PPMI:
    `local_mi = beobachtet × ppmi` (Evert) und
    `t_score = (beobachtet − erwartet) / √beobachtet` (Church & Hanks), dazu ein
    erklärender Notebook-Absatz je Maß. Grund: der PMI-Deckeneffekt (s. u.) macht
    exklusive Kontextwörter untereinander ununterscheidbar, weil sich
    `beobachtet` aus dem Bruch herauskürzt; beide Maße holen die Trefferzahl
    zurück in die Rechnung. `t_score` bewusst **nicht** bei 0 gekappt — negative
    Werte („seltener als erwartet") sind für die Anzeige informativ. Beide sind
    reine Diagnose-/Anzeigespalten; SVD-Input bleibt `ppmi` (s. Stolperstein).
15. **Gemeinsame SVD über alle Zeitscheiben** — Zeilen- und Spaltenindex per
    `.sort().with_row_index()` als eigene Tabellen (`zeilen_tabelle`,
    `spalten_tabelle`) vergeben und an die Langform gejoint, dann dichte
    numpy-Matrix befüllen und `np.linalg.svd(..., full_matrices=False)`.
    Ergebnis: **502 Zeilen × 64.165 Spalten**, 129 MB als `float32`, SVD ~3 s.
    Vektoren = `U[:, :k] * S[:k]` mit `k = 100` → `aggregates`-Tabelle mit
    Listen-Spalte `vector`. Erklärter Anteil der Frobenius-Masse bei k=100:
    **40,9 %**. Kein Pivot, kein scipy (s. Stolpersteine).
16. **Plausibilitätsprüfung** — Vektoren auf Länge 1 normieren, Skalarprodukt =
    Kosinus, für eine Beispielzeile die nächsten Nachbarn ausgeben. Erwartung:
    dasselbe Zielwort in anderen Quartalen. Trifft für `Sicherheit` in 50/50
    Quartalen zu, über alle Zeilen in **344/502 (68,5 %)** — die Abweichler sind
    semantisch benachbarte Zielwörter, kein Pipeline-Fehler (s. u.).

17. **Diagnose-Zelle Volumeneffekt** (seit 2026-09-03 in `pipeline-diagnose.py`,
    nicht mehr in `pipeline.py`) — reproduzierbare Messlatte, an der jede
    Fix-Variante mit denselben Zahlen bewertet wird (alle Namen `diag_`-
    präfigiert, weil marimo jede Variable nur einmal definiert haben will).
    Ergebnis: `diagnose` (eine Zeile je Matrixzeile, mit `randsumme_ziel`,
    `zielwort_haeufigkeit`, `belegte_spalten`, `zeilennorm`, `vektorlaenge`,
    `mittlere_aehnlichkeit`, `aehnlichkeit_global`, `rang`/`rang_von`) und
    `diagnose_korrelationen` (je Zielwort r gegen `log(Tokens)` für Ähnlichkeit,
    Belegung und Zeilennorm). `mittlere_aehnlichkeit` ist der Kosinus einer
    Zeile zu den *übrigen Quartalen desselben Zielworts*, Diagonale per
    `np.fill_diagonal(..., np.nan)` ausgeschlossen. Wichtig: nach dem Join des
    Volumens an `zeilen_tabelle` wieder `.sort("zeile_i")` — ein Join gibt keine
    Reihenfolgegarantie, die numpy-Kennzahlen werden aber per Position
    zugeordnet. Die Strukturkennzahlen werden bewusst auf der **rohen** `matrix`
    berechnet, nicht auf einem transformierten SVD-Input: sie sollen über alle
    Fix-Varianten vergleichbar bleiben.

18. **Ankervektoren aus der Spaltenseite** — `Vt` lag seit Schritt 15
    ungenutzt herum. `anker_vektoren = Vt[:k].T` gibt jedem der 63.851
    Kontextwörter einen Vektor im selben Raum, ohne dass irgendetwas neu
    gerechnet wird. Bewusst **ohne `* S`**: die Zeilenvektoren tragen S
    bereits (`U * S`); ein zweites Mal multipliziert steckt es im
    Skalarprodukt quadratisch und gewichtet die starken Achsen doppelt. Ohne S
    rekonstruiert `zielwort · anker` genau die PPMI-Zelle. Das ist keine bloße
    Skalenfrage — die Multiplikation mit der Diagonalmatrix S ändert die
    *Richtung* des Ankervektors. Kandidatenauswahl über `anker_belegung` (in
    wie vielen der 502 Matrixzeilen hat das Wort einen Eintrag mit
    `ppmi > 0`), Schwelle `anker_mindestzellen` — beides zwingend, s. die
    zwei Stolpersteine unten.

19. **Erster inhaltlicher Beleg, dass die Kette trägt.** Nächste
    Kontextwörter für `Bundeswehr`, Schwelle 50:

    | 2016-Q1 | 2022-Q1 |
    |---|---|
    | Mali, Mandat, Auslandseinsatz, Mission, Soldat, Einsatz, Parlament | einsatzbereit, Einsatzbereitschaft, Verteidigungshaushalt, Wehrbeauftragte, Attraktivität, Arbeitgeber, Abzug, Verteidigungsministerium |

    Historisch korrekt (Mandatsdebatten und Auslandseinsätze gegen
    Beschaffung, Haushalt und Personalgewinnung), deutlich unterschiedlich,
    und nirgends eingebaut — anders als bei den synthetischen Demo-Daten, wo
    der Befund per Konstruktion drinsteckt. Bis hierhin gab es nur
    Strukturkennzahlen (40,9 % erklärte Varianz, 68,5 % nächste Nachbarn);
    das ist die erste Aussage über Sprache. Die Kosinuswerte liegen mit
    0,34–0,49 deutlich unter denen der ungefilterten Liste (0,65–0,73) —
    das ist der Fortschritt, nicht der Preis: die hohen Werte kamen von
    Echos (s. u.), die naturgemäß nah an dem liegen, was sie spiegeln.

20. **Ankerkandidaten brauchen einen eigenen POS-Filter.** Die erste
    Kandidatenliste war voll von `immer`, `so`, `noch`, `dann`, `nur`, `auch` —
    alles **ADV**, und ADV steht nicht auf der Verbotsliste des
    Matrix-POS-Filters: unter den nach jenem Filter verbleibenden Tokens ist
    ADV mit **21,4 %** der zweitgrößte Block nach NOUN (39,9 %), vor VERB
    (18,0 %) und ADJ (10,6 %). Dazu `zu`/`am` als PART, das bewusst erhalten
    bleibt (dort sitzt die Negation). Für die Matrix war das folgenlos — PPMI
    erledigt Funktionswörter selbst —, für die Ankerauswahl nicht, weil dort
    nach Kosinus gerankt wird und breit gestreute Wörter durchschlagen.
    Deshalb `anker_pos_erlaubt = ["NOUN","PROPN","ADJ","VERB"]`, angewandt
    über die je Lemma **dominante** Wortart: ein Lemma trägt mehrere Tags
    („gut" ist 4.068× ADV und 3.584× ADJ). Wieder ein reiner Auswahlschritt,
    die Matrix bleibt unangetastet.

21. **Kontrast statt Reichweite** (`anker_rangliste_kontrast`). Der erste
    Versuch — jede der 502 Zellen nennt ihre 10 nächsten Kontextwörter, dann
    Nennungen zählen — ging schief: die Methode belohnt Wörter, die *vielen*
    Zellen mittelmäßig nah sind, und das sind die generischen (`gehen`,
    `sagen`, `finden`, `Deutschland`, jeweils 8 bis 10 der 11 Zielwörter). Sie
    sitzen im Schwerpunkt des Raums, und an einem Punkt, der von überall gleich
    weit weg ist, lässt sich keine Bewegung ablesen. Dieselbe Kosinus-Tabelle
    deshalb **je Spalte** statt je Zeile gelesen: `mittel` (Durchschnitt über
    alle 502 Zellen), `spitze` (Durchschnitt der 10 nächsten Zellen),
    `kontrast = spitze − mittel`. Dazu `bestes_zielwort`/`bestes_quartal` per
    `argmax` — die eigentliche Kurationshilfe, weil sie zeigt, *wofür* ein Wort
    steht, und die Cluster damit fast von selbst ergibt.

22. **Zweiter Durchgang: Kontrast über die Zeit** (`anker_rangliste_zeit`).
    Schritt 21 misst, was ein Zielwort von den anderen zehn trennt — deshalb
    fehlte das gesamte Zeitenwende-Vokabular (`Sondervermögen`, `Beschaffung`,
    `Aufrüstung`, `Einsatzbereitschaft`, `Fähigkeitslücke`) in den Top 200: es
    gehört zu `Bundeswehr` *und* `Verteidigung` *und* `Rüstung` gleichzeitig
    und hat in diesem Sinn wenig Kontrast, obwohl gerade solche Wörter die
    Zeitachse tragen. Die Metrik findet Wort-Unterschiede, nicht
    Zeit-Unterschiede. Der zweite Durchgang vergleicht deshalb die mittlere
    Nähe zu den Quartalen **nach** der Zäsur gegen die **davor**
    (`zeitsprung`), begrenzt auf die militärischen Zielwörter — sonst funkt der
    Energie- und Sozialdiskurs von 2022 dazwischen, der über `Sicherheit` mit
    im Raum sitzt (`Kurzarbeitergeld`, `Weiterbetrieb`, `kerntechnisch` haben
    alle ihr bestes Quartal in 2022-Q2). Gegenprobe ist das andere Ende
    derselben Liste: dort muss das alte Vokabular stehen (Afghanistan, Taliban,
    Mandat).

> **Revidiert 2026-10-05 (Stufen-Pipeline, post):** „Kontrast“ und „Echo“
> sind entfernt; „Kontrast“ war kein etabliertes Verfahren (Methodology 2.9).
> Ankerkandidaten künftig per Keyness vor/nach der Zäsur (Methodology 3.9).

Die Schritte 20–22 lieferten nur Kandidaten-Ranglisten für die **Landmarken**
(`anker_kandidaten.csv`, `anker_kandidaten_zeit.csv`) — kein Bundle-Feld. Ab
hier baut das Notebook das eigentliche Bundle zusammen, in der Reihenfolge, in
der die Zellen tatsächlich liegen (nicht mehr Diagnose, sondern Konstruktion):

23. **Anker einlesen und anwenden.** `anchors.csv` (händisch kuratiert aus den
    beiden Ranglisten oben, s. „Nächste Schritte" Punkt 2) wird eingelesen,
    leeres `label` fällt per `coalesce` auf `lemma` zurück, jedes `lemma` muss
    im Vokabular auflösen (harter Fehlerausdruck bei Tippfehlern oder nach
    einem Parameterwechsel — dieselbe Konvention wie bei den Suchwörtern in
    Schritt 3), die Vektoren kommen über `spalte_j` aus der Spaltenseite der
    SVD (Schritt 18). Ergebnis `anchors` (86 Zeilen: `lemma`, `label`,
    `cluster`, `vector`), geschrieben nach `output/anchors.parquet`.
    **Revidiert 2026-10-05:** Die Ankerliste ist eine parameterunabhängige
    Maximalliste (`stages/03_postprocessing/input/anchors.csv`); fehlende
    Lemmata sind kein Fehler, bei mehreren Varianten je Label gilt die mit der
    größten Belegung (Methodology 1.15).
24. **Grundgerüst, dann gemeinsame PCA.** `entities` (eine Zeile je Zielwort-
    Lemma, `zu_id()` transliteriert Umlaute zur `entity_id`, mit
    Kollisionsprüfung; **revidiert 2026-11-17:** `entity_id` ist das Lemma
    unverändert, ohne Umwandlung und damit ohne Kollisionsrisiko),
    `time_slices` (`index` aus der alphabetischen —bei
    `"20XX-QY"`-Labels zugleich chronologischen— Sortierung der Zeitscheiben)
    und `freq` (KWIC-Trefferzahl je Zelle, **vor** dem Explode gezählt, s. den
    `randsumme_ziel`-Stolperstein) entstehen zuerst, weil sie von der PCA nicht
    abhängen — umgekehrt schon (behobener Reihenfolgefehler vom 2026-09-03,
    s. `.claude/memory/project-state.md`). Danach die gemeinsame 3er-PCA:
    Aggregat- und Ankervektoren in **einen** Stapel (`np.vstack`), **eine**
    Zerlegung, per Indexgrenze wieder auf `aggregates.coords_3d` und
    `anchors.coords_3d` aufgeteilt — genau die Invariante „ein Raum, kein
    Alignment" aus `Readme Datenmodell.md` §4.
    **Revidiert 2026-10-05:** Gefittet wird nur auf den Aggregaten; Anker und
    alle übrigen Kontextwörter werden mit denselben Achsen projiziert, ohne
    neuen Fit. Die Invariante bleibt erhalten (Methodology 1.16, 2.12).
25. **`neighbors` (k = 150, Kontrast-Auswahl).** Dieselbe Kontrast-Metrik wie
    bei der Anker-Kuration (Rohwert minus Spaltenmittel, Schritt 21), aber
    nicht auf ~50 Anker-Kandidaten angewendet, sondern auf die volle
    belegungsgefilterte Kandidatenliste, und mit `neighbors_k = 150` statt der
    ursprünglich vorgesehenen 8 — begründet damit, dass dieselbe Liste seit
    2026-09-11 auch die 3D-Punktwolke um das Zielwort speist, nicht mehr nur
    die Top-10-Textanzeige (s. `.claude/memory/vocab-radius-feature.md`).
    Rang bleibt die Array-Position (kein Nachsortieren nach `score`); ist ein
    Nachbar zufällig ein Anker, übernimmt er dessen Anzeigelabel und Cluster
    als `group`, sonst das Fallback-Tag `KONTEXT`. Eine eigene Diagnosezelle
    prüft den **rohen** Kosinus je Rang (1/8/25/50/100/150) gegen den
    **Grundpegel** (mittlerer/medianer Kosinus eines beliebigen Wortpaars) —
    bleibt Rang 150 deutlich darüber, trägt k; fällt die Kurve schnell auf den
    Grundpegel, ist k zu hoch angesetzt.
    **Revidiert 2026-10-05/06:** Nachbarn nach rohem Kosinus, keine
    Kontrast-Auswahl; der Befund hinter „Echo“ wird als Mindestbelegung
    geführt (Methodology 1.10, 1.11, 2.9). Die Filter wählt seit 2026-10-06 der
    Nutzer im Frontend; die vorberechneten `neighbors` entsprechen den
    Voreinstellungen.
26. **`vocab_points` (Koordinaten für die übrigen Nachbarn).** Anker und
    Aggregate *definieren* den PCA-Raum (Schritt 24); alle anderen Wörter, die
    irgendwo in `neighbors` auftauchen, werden nur **hineinprojiziert**. Pool:
    jedes Nachbar-Lemma aus Schritt 25, abzüglich aller Anker — doppelt
    geprüft, sowohl gegen `anchors.lemma` als auch gegen `anchors.label`
    (verhindert die Text-Kollision, bei der ein Korpus-Lemma zufällig so
    heißt wie ein Anker-Anzeigename, s. den `marode`-Fund in
    `.claude/memory/vocab-radius-feature.md`). Dieselbe Normierung wie beim
    PCA-Fit, dieselben `pca_mittelwert`/`pca_achsen`, **keine neue PCA** — sonst
    verschöben sich alle bereits vergebenen Koordinaten. Eine Kontrollzelle
    prüft anschließend, ob wirklich jedes Nachbar-**Label** entweder unter den
    Ankern oder unter `vocab_points` eine Koordinate hat (muss 0 offene
    Labels ergeben).
27. **Metadaten, Bundle-Zusammenbau, JSON-Export.** `bundle_meta` trägt
    `vector_dim` (aus der tatsächlichen Vektorlänge, nie festgenagelt),
    `explained_variance_ratio_3d` und `explained_frobenius_ratio_k` (beide
    gerundet), eine automatisch generierte `description` mit dem erklärten
    Varianzanteil eingebettet — sie warnt sich selbst, wenn die 3D-Ansicht
    wenig Struktur zeigt. Vektoren/Koordinaten werden über eine kleine
    `runde()`-Hilfsfunktion auf 5 bzw. 4 Nachkommastellen gerundet (numpy-Typen
    sind nicht JSON-serialisierbar, volle Fließkommagenauigkeit bläht nur die
    Datei). `facets_schema` und `records` bleiben **bewusst leer** (s.
    „Nächste Schritte" Punkt 3). Export nach
    `output/bundestag_zeitenwende_{quartal,halbjahr,jahr}.json`
    (`{raster}` im Dateinamen, weil mehrere Auflösungen nebeneinander liegen);
    ein Schalter (Standard: an) kopiert danach zusätzlich nach
    `../backend/bundles/` — `output/` ist die Werkbank der Pipeline,
    `backend/bundles/` ist das, was die App tatsächlich lädt.

> **Neu 2026-10-05/06 (post, Suchraum für das Backend):** Neben das
> JSON-Bundle schreibt post den Suchraum: `{bundle}.vectors.npy` (alle
> Vektoren, float32, auf Länge 1 normiert; erst die Abfragen Zielwort ×
> Zeitscheibe, dann alle Kontextwörter) und `{bundle}.index.parquet` (eine
> Zeile je Vektor mit Rolle, Zeitscheibe, Lemma, 3D-Koordinaten und den
> Filterattributen Wortart, Belegung, Kookkurrenzen, Rednername).
> `meta.neighbor_space` nennt Formatversion (`format: 1`), Dateien, Art der
> Nachbarn, die Filter mit Voreinstellungen und die mitgelieferte
> Stoppwortliste (`stages/03_postprocessing/input/stopwords.txt`). Das Format
> ist für SGNS ausgelegt: Kandidaten mit Zeitscheibe = ein Raum je Scheibe,
> ohne Zeitscheibe = gemeinsamer Raum (PPMI).
> Die Bedeutung einer Abfrage (Filtertypen, Stoppwörter, Zeitscheiben-Regel,
> Sortierung mit Gleichstandsregel, Rechnung in float64) legt das Schema der
> App fest (`SCHEMA.md`, Kopie im Wurzelverzeichnis dieses Repos). post rechnet
> die vorberechneten `neighbors` nach genau dieser Regel aus dem Suchraum; sie
> sind der Rückfall ohne Backend, und ein Konformitätstest im Backend prüft,
> dass eine Abfrage mit den Voreinstellungen sie exakt reproduziert (Stand
> 2026-10-06: alle 287 Listen). Methodische Prüfungen zur Nachbarauswahl
> (Hubness; CSLS und Mutual Proximity) stehen in
> `stages/03_postprocessing/pipeline_post_diagnose.py`.

## Stolpersteine (PPMI-Pfad)

Die ersten drei Punkte sind **allgemein** und gelten für jedes Notebook hier;
alles ab „`randsumme_ziel` ist nicht die Zielwort-Trefferzahl" betrifft
ausschließlich die PPMI-Matrix. Die Stolpersteine des SGNS-Pfads stehen in
dessen eigenem Abschnitt weiter unten.

- **marimo-Zyklus**: `nlp = spacy.load(...)` darf nicht in derselben Zelle
  stehen wie Code, der `zielwort_lemmata` benutzt — sonst hängt die Zelle, die
  `nlp` erzeugt, gleichzeitig von einer Zelle ab, die selbst `nlp` braucht.
  Deshalb eigene, unabhängige Zelle nur für `import spacy` + `nlp = spacy.load(...)`.
- **CSV vs. Parquet**: sobald eine Spalte ein Listen-Typ ist, nicht mehr
  `write_csv`/`read_csv` verwenden, sondern `write_parquet`/`read_parquet`.
- Lange laufende Zellen (KWIC-Extraktion, Kontext-Lemmatisierung) auf
  `disabled=True` gesetzt und stattdessen aus `output/` nachgeladen, damit
  Notebook-Edits nicht versehentlich mehrminütige Neuberechnungen auslösen.
- **`randsumme_ziel` ist nicht die Zielwort-Trefferzahl**: `randsumme_ziel`
  ist die Summe von `beobachtet` über alle Kontextwörter, also die Anzahl der
  Kontext-**Tokens** — bei einem Fenster von bis zu 80 Wörtern pro Treffer
  (Median exakt 80, Std ≈ 8, Min 21 an Reden-/Zwischenruf-Rändern) ein
  Vielfaches der tatsächlichen Trefferzahl. Ein Zielwort mit nur 1 Treffer
  kann trotzdem `randsumme_ziel = 80` haben (verifiziert an "Hindukusch"/
  2022-Q1). Ein "Mindesthäufigkeit Zielwort"-Filter braucht deshalb eine
  eigene Aggregation auf `kwic_liste_quartale` **vor** dem Explode
  (`zielwort_haeufigkeit`, s. Ablauf-Schritt 12) — nicht `randsumme_ziel`.
  Auch keine Abkürzung über einen festen Multiplikator (z. B. `× 80`): die
  Fensterlänge streut real genug (21–85), dass das Randfälle systematisch
  falsch einordnet, und koppelt den Filter unsichtbar an den `fenster`-Wert
  in `kwic_treffer()`.

- **scipy/sparse ist für diese Matrix nicht nötig.** Die gemeinsame Matrix hat
  nur **502 Zeilen** (11 Zielwörter × ~50 Quartale). Der Aufwand einer SVD
  richtet sich nach der kleineren Dimension, deshalb reicht eine dichte
  numpy-Matrix: 129 MB als `float32`, `np.linalg.svd(full_matrices=False)` in
  ~3 s. scipy ist im venv gar nicht installiert und wird auch nicht gebraucht.
  Nebenbei: bei einem späteren Wechsel auf `scipy.sparse.coo_matrix` beachten,
  dass diese **doppelte Koordinaten aufsummiert**, während numpys Fancy
  Indexing sie überschreibt (hier egal — die `group_by`-Herkunft garantiert
  Eindeutigkeit, verifiziert: 434.640 Zeilen = 434.640 eindeutige Tripel).
- **`zeilen_tabelle` nach `.with_row_index()` nicht mehr umsortieren oder
  filtern.** Die Vektoren werden in Schritt 15 per `pl.Series` einfach
  danebengelegt; die Zuordnung hängt allein an der unveränderten Zeilen-
  reihenfolge. Ein zwischengeschobenes `.sort()` würde jeder Zeile den Vektor
  einer anderen geben — ohne Fehlermeldung. Genau das prüft Schritt 16.
- **`local_mi` ist als SVD-Input ungeeignet.** Alle drei Maße haben dasselbe
  Besetzungsmuster (`beobachtet − erwartet > 0 ⟺ pmi > 0`), die Wahl wäre also
  ein Einzeiler. Entscheidend ist die Verteilung der Frobenius-Masse, denn die
  bestimmt, was eine gekappte SVD überhaupt anpasst: in 2022-Q1 liegen bei
  `ppmi` 1,8 % der Masse in den Top-10-Zellen, bei `local_mi` aber **56,0 %**
  (dominiert von `der`, `wir`). PPMI bleibt SVD-Input — auch wegen der
  theoretischen Grundlage (Levy & Goldberg 2014) und weil `specs.md` auf
  „PPMI-SVD-joint" festgelegt ist. Ein späterer A/B-Kandidat wäre allenfalls
  der bei 0 gekappte t-Score (Top-10-Masse 4,2 %).
- **Hapax-Filterung nur global, nie je Zeitscheibe.** Kontextwörter mit genau
  einem Vorkommen sind 25,6 % der Zellen und tragen 41,8 % der Frobenius-Masse,
  ein Filter wäre also wirksam. *Pro Zeitscheibe* angewandt zerstört er aber
  genau das gesuchte Signal: `Zeitenwende` ist in 2014-Q2, 2016-Q3 und 2017-Q1
  Hapax und springt danach auf 30/107/59/…/101 — der Filter löschte die
  Vorgeschichte des Begriffs. 3.388 von 3.689 Wörtern mit globaler Häufigkeit
  ≥ 50 sind in mindestens einem Quartal Hapax. Ein **globaler** Schwellenwert
  über alle 50 Scheiben ist zwar strukturell verträglich (feste Spaltenindizierung), aber **nicht
  folgenlos**: schon `> 4` hebt die gepoolte Volumenkorrelation von 0,208 auf
  0,442 — die frühere Einschätzung „unbedenklich" ist damit widerlegt, s. den
  Filter-Stolperstein unten.

- **Zeilennormierung vor der SVD macht den Volumeneffekt drastisch schlimmer.**
  Getestet 2026-09-01: Matrixzeilen vor `np.linalg.svd` auf Länge 1 normiert,
  um zu verhindern, dass dichte Quartale den Fit dominieren. Ergebnis gepoolt
  über alle 502 Zeilen: r(mittlere Ähnlichkeit × log Tokens) **0,208 → 0,929**,
  r(globale Ähnlichkeit × log Tokens) **−0,138 → 0,967**, alle 11 Zielwörter
  positiv. Kontrollfall `Sicherheit`/2022-Q2 (echte Distanz) **Rang 49 → 33**,
  2022-Q1 **38 → 26**; erklärte Varianz bei k=100 **40,9 % → 32,8 %**.
  Der Kosinus ist skaleninvariant — im *vollen* Raum ist die Normierung ein
  No-Op, der gesamte Effekt entsteht also durch die Kürzung auf k=100.
  Erklärung: die große Norm der dichten Zeilen hat die Top-Achsen bestimmt,
  und das sind die Richtungen des *geteilten* Vokabulars; dünne Zeilen wurden
  auf diese gemeinsame Basis projiziert und dadurch geglättet. Mit gleichem
  Gewicht können dünne Zeilen eigene Achsen für ihr privates Hapax-Vokabular
  beanspruchen, stehen danach fast orthogonal zu allem, und die dichten Zeilen
  rücken im geteilten Rest zusammen. **Die Normdominanz war kein Defekt,
  sondern ein zufälliger Schutz.** (Der damals daraus gezogene Schluss, der
  Mechanismus sei die Sparsity des Spaltenraums, war voreilig — s. den
  übernächsten Eintrag.) Verworfen; der Schalter bleibt im Notebook.

- **Globaler Mindesthäufigkeits-Filter auf den Kontextwörtern: ebenfalls
  getestet und verworfen.** Idee war, dünnen Zeilen ihr vermeintlich privates
  Vokabular zu nehmen, indem nur global häufige Kontextwörter eine Matrixspalte
  bekommen. Die Annahme war falsch — der Filter schneidet fast ausschließlich
  bei den *dichten* Zeilen: von Schwelle 1 auf 100 fallen 96,4 % aller Spalten
  weg, dabei behält die dünnste Zeile 77 % ihrer Spalten (39 → 30), die
  dichteste nur 20 % (5.653 → 1.116). Per Zipf/Heaps sitzen die seltenen Wörter
  eben in den *großen* Stichproben, nicht in den kleinen.

  | Schwelle | Spalten | r gepoolt | `Sicherheit` | Rang 2022-Q1 | Rang 2022-Q2 | erkl. Varianz |
  |---|---|---|---|---|---|---|
  | **1 (aus)** | 64.165 | **0,208** | **0,336** | **38** | **49** | **40,9 %** |
  | 5 | 17.404 | 0,442 | 0,604 | 33 | 45 | — |
  | 50 | 3.689 | — | — | — | — | 46,6 % |
  | 100 | 2.285 | 0,878 | 0,835 | 11 | 34 | 50,0 % |

  Die Verschlechterung ist **monoton und setzt sofort ein** — es gibt keinen
  unbedenklichen Schwellenwert. Bei 100 landet 2022-Q1, das Zeitenwende-
  Quartal, auf Rang 11 von 50, ist also eines der unauffälligsten überhaupt.
  Der steigende Varianzanteil ist dabei kein Gegenargument: fallen 96 % der
  Spalten weg, verschwindet mit ihnen auch schwer anpassbare Frobenius-Masse,
  der Anstieg ist also teilweise reine Buchhaltung. Der Schalter
  (`vokabular_schwelle_picker`) lebt seit 2026-09-03 in `pipeline-diagnose.py`;
  in `pipeline.py` selbst ist der Filter fest deaktiviert (kein Schalter mehr,
  Vokabular ungefiltert — entspricht Wert 1 = aus).

- **POS-Filter der Kontextwörter: getestet, folgenlos, trotzdem behalten.**
  `kontext_pos` lag seit der Lemmatisierung ungenutzt in
  `kwic_lemmatisiert.parquet`, ein Filter war also ohne neuen spaCy-Lauf
  testbar. Verworfen wurden `DET, ADP, PRON, AUX, CCONJ, SCONJ` (`PART`
  bewusst **nicht** — dort steckt im Deutschen die Negation, und „nicht" im
  Fenster von `Bundeswehr` ist inhaltlich nicht nichts). Das entfernt
  **46,8 %** aller Kontext-Tokens (3.236.709 → 1.721.571) und ändert am
  Ergebnis so gut wie nichts:

  | | Baseline | POS-Filter |
  |---|---|---|
  | Kookkurrenz-Zeilen | 544.678 | 510.500 (−6,2 %) |
  | Matrixspalten | 64.165 | 63.851 (−0,49 %) |
  | erklärte Varianz | 40,9 % | 40,9 % |
  | r gepoolt | 0,208 | 0,239 |
  | Ränge Kontrollfälle | 50/47/42/38/49 | 50/47/41/37/49 |

  Kurz: **46,8 % der Tokenmasse → 0,49 % der Spalten → 0,0 Prozentpunkte
  erklärte Varianz.** Der Grund ist PPMI selbst — Funktionswörter haben eine
  riesige `randsumme_kontext`, also ein `erwartet` in derselben Größenordnung
  wie `beobachtet`, also PMI ≈ 0, also PPMI = 0. Der Filter hat nur nachgeholt,
  was die Formel längst getan hatte. Die Randsummen kürzen sich zudem sauber
  heraus: 2017-Q3 behält 51,0 % seiner Kontext-Tokens, 2022-Q2 53,1 % — der
  Funktionswortanteil dünner und dichter Scheiben unterscheidet sich nur um
  gut zwei Prozentpunkte.

  Der erwartete Nachteil ist messbar, aber winzig, und er ordnet sich sauber
  nach Datenreichtum (Δ `r_aehnlichkeit`, positiv = schlechter):
  Hindukusch (37 Scheiben) +0,139, Landesverteidigung (40) +0,182,
  Friedenssicherung (43) +0,169 · Rüstung/Rüstungsindustrie/Drohne (46–48)
  +0,03…+0,07 · Bundeswehr/Frieden/Sicherheit/Verteidigung (50) −0,03…0,00.
  **Behalten**, weil er nichts kostet und alles Stromabwärtige sauberer macht:
  Anzeige-Ranglisten, später `neighbors` und `anchors`, kleineres Bundle.

  > **Revidiert 2026-10-01 (Stufen-Pipeline):** Die Matrix wird nicht mehr
  > gefiltert. POS-Filter, Rednernamen und Domänen-Stoppwörter wirken nur noch
  > bei der Auswahl von Nachbarn und Ankern; seit 2026-10-06 sind sie Optionen
  > im Frontend (Seite „Nachbar-Filter“). Begründung: Methodology 2.4,
  > 1.12–1.14.

- **Das Prinzip hinter drei getesteten Eingriffen.** Ordnet man sie danach,
  *welches Ende der Häufigkeitsverteilung* sie angreifen, ergibt sich ein
  klares Bild (r gepoolt, Baseline 0,208):

  | Eingriff | greift an | r |
  |---|---|---|
  | Vokabularfilter (Schwelle 100) | seltener Schwanz | 0,878 |
  | Zeilennormierung | Zeilengewicht, das aus dem Schwanz kommt | 0,929 |
  | POS-Filter | häufiger Kopf | 0,239 |

  Beschreibt man eine Zeile **nur über häufige Wörter**, ist sie im
  Wesentlichen eine verrauschte Schätzung der Hintergrund-Worthäufigkeits-
  verteilung, und wie stark sie davon abweicht, bestimmt allein die
  Stichprobengröße. Ähnlichkeit misst dann Stichprobentiefe, also Volumen —
  genau das sagen 0,878 und 0,929. Was eine Zeile *thematisch* unterscheidbar
  macht, sitzt dagegen im seltenen Schwanz; das 2022er-Vokabular ist ja gerade
  neu. **Der seltene Schwanz ist nicht das Rauschen, er ist das Signal** — und
  im unveränderten Zustand ist er zugleich das, was den Volumeneffekt niedrig
  hält, weil er die großen Quartale distinkt macht. Das erklärt, warum in der
  Baseline 8 von 11 Zielwörtern eine *negative* Korrelation zeigen. Der Kopf
  dagegen trägt fast die halbe Tokenmasse und praktisch null Information.
  Konsequenz: die Matrixseite ist ausgereizt — Eingriffe am Schwanz sind
  zerstörerisch, Eingriffe am Kopf folgenlos.

- **Die Spaltenseite der SVD gibt Echos zurück, wenn man nicht filtert.**
  Fragt man für eine Zelle die nächsten Kontextwörter ab, stehen ungefiltert
  Wörter wie „Ouagadougou", „Euskirchen", „fachfremd" oben — und zwar mit
  **exakt identischen Kosinuswerten** (in `Bundeswehr | 2016-Q1` sieben Wörter
  mit 0,6539999842643738, in `2022-Q1` drei Paare). Der Median der belegten
  Zellen dieser Top-20 ist **1**. Grund: hat ein Kontextwort nur in einer
  einzigen Matrixzeile einen Eintrag, ist seine Spalte
  `v_j = M_ij · u_i · S⁻¹` — die Richtung hängt allein an der Zeile `i`, der
  PPMI-Wert ist bloß ein Längenfaktor, den der Kosinus wegnormiert. Alle
  exklusiven Wörter derselben Zelle bekommen dadurch buchstäblich denselben
  Vektor. Sie sind keine Nachbarn, sondern die Zelle selbst, zurückgespiegelt.
  Das ist der PMI-Deckeneffekt auf der Spaltenseite: für die *Zeilen* ist er
  entschärft (eine Achse braucht ein von mehreren Zeilen geteiltes Muster),
  die Spaltenvektoren der Einzelgänger existieren trotzdem und kommen zurück,
  sobald man sie abfragt. Gegenmittel ist eine Mindestbelegung bei der
  **Auswahl** der Anker — ausdrücklich kein Eingriff in die Matrix: SVD und
  Raum bleiben unverändert, anders als beim verworfenen Vokabularfilter oben.
  Kontrollzeichen, dass die Schwelle zu niedrig sitzt: identische
  Kosinuswerte in der Rangliste.

- **Die Mindestbelegung misst Zeitstabilität, nicht Qualität — Schwelle
  niedrig halten.** Erster Versuch war `anker_mindestzellen = 50` („ein
  Zehntel der 502 Zeilen"), ein hübsches Argument ohne Bezug zum Problem.
  Ergebnis: **`Sondervermögen` fiel heraus.** Es hat 47 Zellen mit `ppmi > 0`,
  und mehr kann es nicht haben — das Wort gibt es erst seit 2022, es kann
  höchstens 16 der 50 Quartale belegen. Der Filter fragt „war dieses Wort
  lange da?", und neue Wörter beantworten das per Definition mit nein. Damit
  trifft er genau die Wörter, wegen denen das Projekt existiert: dieselbe
  Falle wie beim Vokabularfilter, nur an der Nachgeschichte statt an der
  Vorgeschichte. Zum Vergleich `Zeitenwende` 59 Zellen (knapp drin),
  `Attraktivität` 50 (exakt an der Kante). Der Echo-Effekt sitzt dagegen bei
  `zellen == 1`, dagegen genügt **10**. Nicht verwechseln: `anker_belegung`
  zählt aus `svd_input`, also nur Zellen mit `ppmi > 0` — zählt man alle
  Kookkurrenzen, kommt `Sondervermögen` auf 90 Zellen und der Filter sähe
  harmlos aus. Verifiziert, dass das Wort sehr wohl in der Matrix steht:
  `Bundeswehr | 2022-Q1` beobachtet 104, erwartet 47,7, PMI +1,13.

- **Stanza führt gegenderte Formen und Schreibvarianten nicht zusammen
  (2026-11-17).** Getestet an den 2.615 Formen der Zielwortliste, jede Form
  ohne Kontext lemmatisiert, wie damals die Zielwort-Zelle in pre: nur 472 der
  608 Gruppen bekamen ein einziges Lemma (insgesamt 1.170 verschiedene).
  Flexion löst Stanza zuverlässig auf (`Kriegen` → `Krieg`, `Risiken` →
  `Risiko`). Gegenderte Formen bleiben aber eigene Lemmata (`Soldat*in`,
  `Soldat:in`, kaputt: `Soldat/-inne`). Varianten und Transliterationen führt
  es nicht zusammen (`Kiew`/`Kyjiw`/`Kyiv`, `Belarus`/`Weißrussland`).
  Uneinheitlich ist es auch: `Weißrussland` → `Weißrußland`, aber
  `Weißrusslands` → `Weißrussland`. **Im Kontext zerlegt der Tokenizer
  gegenderte Formen sogar:** `Soldat*innen` → `Soldat` `*` `innen`,
  `Soldat:innen` → `Soldat` `:innen`. Die Bruchstücke gingen als eigene
  Kontextwörter (`innen` als NOUN/ADV, `:innen` als VERB/PROPN) in die Matrix
  ein. Folge: Das Ziel-Lemma kommt jetzt aus der Liste
  (`input/target_forms.csv`, `form;lemma`, erzeugt mit
  `build_target_forms.py`). Stanza für die Zielwörter ist als Schalter
  `TARGET_LEMMAS_FROM_STANZA` erhalten, Default aus. Die Kontexte werden
  **vor** Stanza gemappt (`make_context_mapper`), mit zwei Schaltern, beide
  standardmäßig an: Zielwortliste auf Kontextwörter anwenden, und gegenderte
  Formen außerhalb der Liste vereinheitlichen (`Bürger*innen`, `BürgerInnen`
  → `Bürgerinnen`). Das Kürzen auf den Stamm wurde verworfen, weil es
  `Kolleg*innen` → `Kolleg` ergab. Restfehler: `LinkedIn` → `Linkedin`.

- **Movierte Feminina: Grundform aus dem Wortschatz des Laufs, nicht per
  Regel (2026-11-17).** Das Geschlecht wird nicht unterschieden, also
  `Bürgerin` → `Bürger`, `Ärztin` → `Arzt`. Einfach `-in` abzuschneiden geht
  nicht: Umlaut (`Ärztin`), `-e` (`Kollegin` → `Kollege`) und Fehltreffer.
  Ein Vorlauf auf dem alten punkt-Lauf, ohne Genus-Filter, fand 480 Paare mit
  rund 11.700 Tokens (`Kollegin` 3.223, `Soldatin` 1.826, `Bürgerin` 1.277),
  aber auch `Stein` → `Ste`, `Putin` → `Put`, `Polin` → `Pol` und
  `Gestein` → `Geste`. Umsetzung `feminine_base_map` nach der Lemmatisierung,
  Schalter in pre, Default an:
  - Kandidat ist ein Substantiv-Lemma auf `-in` mit Stanza-Genus überwiegend
    `Fem`. Dafür gibt es die neue Spalte `context_gender`. Das schließt
    `Stein`, `Termin` und `Benzin` aus.
  - Gesucht wird zuerst Stamm bzw. Stamm + `e`, dann dasselbe ohne Umlaut.
  - Komposita gehen über das Grundwort (`Parteikollegin` → `Parteikollege`).
  - Ausnahmen: `Polin` → `Pole`, `Engin` bleibt.
  - Die Zuordnung steht in `output/<variante>/feminine_base_map_*.csv`.

  Stichprobe aus 3.000 Treffern: 71 Paare ohne Fehltreffer. Folge: Auch
  `Bundeskanzlerin` geht in `Bundeskanzler` auf, der Unterschied zwischen den
  Kanzlerin- und den Kanzler-Jahren ist in den Kontexten nicht mehr sichtbar.

## Nächste Schritte (PPMI-Pfad)

**Stand 2026-09-18: dieser Abschnitt ist größtenteils erledigt.** Er ist als
Fahrplan entstanden (Stand damals: nur `aggregates.vector` war fertig) und
seither nicht mehr umbenannt worden, obwohl `pipeline.py` inzwischen alle vier
Bundles (`quartal`, `halbjahr`, `jahr`, dazu `pipeline-sgns.py` für `sgns`) bis
zum JSON-Export durchrechnet — Ablauf-Schritte 23–27 unten. Tabelle und
Begründungen bleiben stehen, weil sie erklären *warum* etwas so gebaut wurde,
nicht weil noch etwas davon aussteht — außer explizit als „weiterhin offen"
markiert.

| Bundle-Feld | Stand | Aufwand |
|---|---|---|
| `meta` | **fertig** | `vector_dim` dynamisch, `shift_point` aus der Zäsur, plus `explained_variance_ratio_3d`/`explained_frobenius_ratio_k` |
| `time_slices` | **fertig** | Quartal/Halbjahr/Jahr per Picker wählbar (s. Setup) |
| `entities` | **fertig** | 11 Zielwort-Lemmata |
| `facets_schema` | **weiterhin offen** | bewusst `[]` — hängt an der Fraktion-Entscheidung (s. 3.) |
| `aggregates` → `vector` | **fertig** | — |
| `aggregates` → `freq` | **fertig** | `zielwort_haeufigkeit` |
| `aggregates` → `coords_3d` | **fertig** | gemeinsame PCA über Aggregate + Anker (Ablauf-Schritt 24) |
| `anchors` → `vector` | **fertig** | Spaltenseite `Vt[:k].T`, 86 Wörter kuratiert in `anchors.csv` |
| `records` (Fraktion) | **weiterhin offen** | Bundle liefert `"records": []` bewusst leer (s. 3.) |
| `neighbors` | **fertig, aber k = 150 statt der hier geplanten 8** | Kontrast-Auswahl gegen die belegungsgefilterte Kandidatenliste, s. Ablauf-Schritt 25 |
| `vocab_points` | **fertig** (seit 2026-09-11, nicht Teil dieser Liste damals) | Out-of-Sample-Projektion der übrigen Nachbar-Lemmata, s. Ablauf-Schritt 26 |
| `sentiment` | geparkt | methodisch offen (specs §6) |

Reihenfolge — Schritt 1 galt lange als blockierend, ist es nach der Messreihe
vom 2026-09-01 aber nicht mehr (s. dort); ab 2 ist alles abarbeitbar:

1. **Volumen — was davon übrig ist.** Die ursprüngliche Diagnose („die
   Kosinus-Ähnlichkeit korreliert mit der Datenmenge des Quartals, r = 0,606;
   damit würde das Tool Redevolumen als Bedeutungswandel anzeigen") hat die
   Messung **nicht bestätigt**. Stand nach vier Messreihen:

   *Was gilt.* Die strukturelle Asymmetrie ist real und groß: `Sicherheit`
   reicht von 2017-Q3 (14 Treffer, 304 belegte Spalten) bis 2022-Q2 (629
   Treffer, 3.756 Spalten), und die Zahl der belegten Spalten korreliert je
   Zielwort mit r = 0,89–0,97 gegen `log(Tokens)`. Dünne Zellen sitzen in
   *jeder* getesteten Konfiguration am Ende der Ähnlichkeitsrangliste
   (2017-Q3: Rang 50 von 50).

   *Was nicht gilt.* Die Kennzahl r = 0,606 reproduziert nicht (gepoolt 0,208,
   `Sicherheit` 0,336), und sie hat vor allem **kein einheitliches Vorzeichen**:
   8 von 11 Zielwörtern sind negativ, viel Volumen geht dort mit *großer*
   Distanz einher. Sie ist konfundiert — über ein Wort wird genau dann viel
   geredet, wenn sich sein Kontext ändert. Die Aussage „Volumen tarnt sich als
   Bedeutungswandel" ist in dieser Pauschalität nicht belegt.

   *Was das Problem tatsächlich ist.* Nicht Verzerrung, sondern **unbezifferte
   Unsicherheit**. Eine Zelle mit 14 Treffern ist eine schlechte Schätzung —
   ihre Lage ist weder systematisch falsch noch vertrauenswürdig, sie ist
   unbekannt. Der Fehlermodus ist also nicht „das Tool zeigt Volumen als
   Bedeutung", sondern „das Tool zeigt einen selbstbewusst aussehenden Punkt
   für eine Zelle, die es gar nicht auflösen kann".

   *Folge für die Reihenfolge.* Dieser Punkt ist damit **nicht mehr
   blockierend**: in der Baseline gibt es keinen starken systematischen
   Volumeneffekt, den die PCA in `coords_3d` einbrennen könnte. Was bleibt, ist
   eine Anzeige- und Ehrlichkeitsfrage auf Bundle-Ebene, die `coords_3d` nicht
   aufhält.

   *Was zu tun bleibt*, in aufsteigendem Aufwand:

   (a) **Mindest-Trefferzahl je Aggregat** — Zellen unter N Treffern nicht
   ausliefern oder als unsicher markieren. Trivial, `zielwort_haeufigkeit`
   liegt vor; offen ist allein, wo N liegt.
   *Stand 2026-11-20:* N gemessen (Diagnose von emb, Abschnitt 1):
   h\* = 30 bei punkt/halfyear, k = 300, p = 0,5. Zellen darunter werden
   markiert, nicht entfernt; Übergabe als `meta.min_reliable_freq` ist offen
   (`nextsteps.md`, Methodology 3.12).

   (b) **Zeitscheibenbreite als Parameter** — Quartal / Halbjahr / Jahr
   umschaltbar. Das Einzige, was die Besetzung dünner Zellen tatsächlich
   anhebt, und es gehört ohnehin zum Datensatz statt zum Tool (`time_slices`
   ist ein Bundle-Feld). Preis: Zeitauflösung, und 2022-Q1 muss scharf bleiben.

   (c) **Split-Half-Rauschboden** — die Treffer einer Zelle zufällig halbieren,
   aus jeder Hälfte unabhängig eine Kookkurrenzzeile bauen, beide per
   `zeile @ V[:, :k]` in den *bestehenden* Raum projizieren (nie separat
   zerlegen — Invariante „Ein Raum, kein Alignment") und den Kosinus zwischen
   ihnen messen. Beide Hälften beschreiben dasselbe Wort im selben Quartal,
   jeder Unterschied ist also reines Stichprobenrauschen: der Wert ist die
   **Auflösungsgrenze dieser Zelle**. Über mehrere Ziehungen gemittelt ergibt
   das eine Unsicherheitsangabe je Zelle, mit der sich (a) begründen statt
   raten lässt und die das Frontend anzeigen kann. Weil die Hälften nur halb
   so groß sind wie die Zelle, ist der Wert eine konservative Untergrenze.
   *Stand 2026-11-20:* Umgesetzt als Split-half-Prüfung im Diagnose-Notebook
   von emb (Abschnitte 1–3) und als Permutationserwartung der Drift je Paar
   (Abschnitt 4). Ergebnis: Rauschen allein erklärt die Drift nicht (alle
   Paare ab 30 Belegen liegen darüber), aber der Abzug der Zufallserwartung
   überkorrigiert; die Drift wird stattdessen mit Teilstichprobe je Paar
   berechnet (Methodology 1.23).

2. **`anchors` — Weg entschieden am 2026-09-01: Spaltenseite (a).**
   Ankervektoren sind `Vt[:k].T`, kosten nichts, und der Raum bleibt
   unverändert (Ablauf-Schritt 18). Weg (b) — Ankerwörter als eigene
   Zielwörter mit neuem KWIC- und spaCy-Lauf — ist nicht nötig und wäre teuer
   erkauft: er ändert die Matrix und macht damit sämtliche Messreihen
   (40,9 %, r = 0,208, die Kontrollränge) unvergleichbar, die
   Aufwandsschätzung „~8 Min" ist zu optimistisch (die 12 Suchwörter ergeben
   42.252 Treffer, häufige Ankerwörter liegen eine Größenordnung darüber, die
   Lemmatisierung skaliert linear), und Anker bekämen eine Zeitachse, die das
   Datenmodell für sie ausdrücklich nicht vorsieht („feste Punkte, sie
   bewegen sich nicht"). Das frühere Argument, der Schema-Text deute auf (b),
   trägt nicht: der zitierte Satz steht so nicht in `data/test/SCHEMA.md`;
   dort heißt es zu Ankern nur „feste Punkte im Raum", der Satz übers
   Projizieren betrifft `records`.

   Entscheidend ist ohnehin, dass beide Wege **derselbe Mechanismus** sind:
   `e_j @ V[:, :k] = v_j`, also ist (a) nichts anderes als (b) mit der
   ärmstmöglichen Kookkurrenzzeile — dem Wort allein statt seinem Kontext. Ein
   späterer Wechsel bliebe folgenlos für Bundle, Schema und Raum; nur die
   Ankervektoren würden reichhaltiger. Falls das je gewünscht ist, wäre der
   billige Zwischenweg, die Ankerzeilen aus den **bereits lemmatisierten**
   Fenstern zu bauen (Kontextwort × Kontextwort innerhalb der vorhandenen
   KWIC-Treffer, `kontext_lemmata` liegt vor) und per `zeile @ V[:, :k]` zu
   projizieren — kein neuer spaCy-Lauf, keine Änderung an den Achsen.

   *Offen bleibt die Liste selbst*: Vereinigung der Top-k über alle 502 Zellen,
   auf ~50 gekürzt (Ablauf-Schritt 8 im Notebook), dazu die Lemma-Dubletten
   (`Soldat` 6.678 / `Soldatinn` 3.869 — spaCy-Fehler aus „Soldatinnen" /
   `Soldatin` 420 landen alle in derselben Top-10) und die `cluster`-Labels,
   für die es kein frei nachnutzbares Dornseiff-Äquivalent gibt. Kandidaten:
   k-Means über die Ankervektoren, GermaNet-Sachgruppen, Handkuration — oder
   gar keine, das Frontend verträgt das (zod-Default `"_"`), verliert dann
   aber die Cluster-Einfärbung.

   *Verfahren festgelegt (2026-09-01): hybrid.* Die Pipeline liefert die
   Kandidaten (Ablauf-Schritte 20–22), die Auswahl bleibt fachlich. Begründung:
   Anker sind keine Messergebnisse, sondern Interpretationshilfen — sie legen
   fest, in welchem Vokabular der Betrachter die Bewegung liest, und diese
   Setzung soll sichtbar von einem Menschen kommen. Bei einer Diskursanalyse
   ist das ein Vorzug, kein Mangel. Ergebnis ist die Datei
   `pipeline/anchors.csv` (`lemma,label,cluster`), **kuratiert auf 86 Wörter in
   6 Clustern** (die ursprüngliche Zielgröße von ~50 wurde beim Kuratieren
   überschritten, ohne dass die Lesbarkeits-Obergrenze unten neu bewertet
   wurde — im Auge behalten). Beide Zahlen sind Lesbarkeits-, keine
   Rechengrenzen: mehr als ~60 beschriftete Punkte überlagern sich in der
   3D-Szene, mehr als ~7 Cluster-Farben sind nicht mehr unterscheidbar (die
   Palette hätte 12).

   Alle übrigen Spalten der Ranglisten (`kontrast`, `spitze`, `zellen`,
   `bestes_zielwort` …) sind reine Kurationshilfe und werden verworfen;
   `vector` wird beim Bundle-Bau über `label` nachgeschlagen, `coords_3d`
   kommt aus der gemeinsamen PCA. Die Datei überlebt Pipeline-Neuläufe, weil
   sie nur Strings enthält — `lemma` muss deshalb **exakt** dem
   `kontext_lemma` entsprechen, und die Lookup-Zelle muss jedes nicht gefundene
   Lemma laut melden: kein stiller Ausfall, dieselbe Konvention wie bei den
   Suchwörtern in Ablauf-Schritt 3.

   *Warum drei Spalten und nicht zwei:* `lemma` ist der Schlüssel für den
   Vektor-Lookup, `label` der Anzeigetext — sie fallen auseinander, sobald
   spaCy ein Lemma verstümmelt hat (`Einsatzkräft`, `Bewerberinn`,
   `Zivilbeschäftigt`, `angehörige`, `Schwächsten`). Solche Wörter existieren
   im Raum genau einmal und korrekt; nur ihre Beschriftung ist kaputt, und die
   repariert man in `label` statt das Wort wegzuwerfen. Leeres `label` heißt
   „`lemma` unverändert anzeigen". Die Lemmatisierung selbst zu reparieren
   wäre der falsche Weg — neuer spaCy-Lauf, andere Matrix, alle Messreihen
   ungültig, und das für ein reines Anzeigeproblem.

   **Fallstrick:** das Frontend löst Nachbarn über `anchorByLabel` auf (exakt
   und ersatzweise klein geschrieben). Heißt der Anker `Einsatzkräfte`, gibt
   `neighbors` aber weiter `Einsatzkräft` aus, findet es die Landmarke nicht
   mehr und der Nachbar bleibt Text ohne Linie. Die Korrektur muss deshalb als
   **eine** Mapping-Tabelle `lemma → label` vorliegen, die beim Anker-Bau *und*
   beim `neighbors`-Bau angewandt wird.

   Davon zu trennen sind **echte Dubletten** mit mehreren Vektoren
   (`Soldat`/`Soldatin`/`Soldatinn`, `Wehrbeauftragter`/`Wehrbeauftragten`):
   verschiedene Zeilen in `spalten_tabelle`, da hilft kein Umbenennen. Eine
   Form wählen (die mit der größeren `haeufigkeit`/`zellen`), die anderen
   streichen. Die Vektoren zu mitteln wäre technisch möglich, ergäbe aber einen
   Punkt, der kein Wort mehr ist.

   *Die Kuration betrifft nur die Landmarken.* Der zweite Ankerzweck — die
   Auflösung der NN-Labels — braucht sie nicht: `neighbors` kann gegen das
   volle Kandidatenfeld gerechnet werden (8.277 nach Belegungs- und
   POS-Filter). Nachbarn, die nicht unter den 86 Ankern sind, erscheinen im
   Frontend als Text ohne Landmarke, was das Datenmodell ausdrücklich vorsieht —
   seit dem `vocab_points`-Umbau (Ablauf-Schritt 26) gilt das nur noch für
   Nachbarn, die überhaupt keinem der beiden Reservoire angehören, und das ist
   laut der Abdeckungsprüfung in Schritt 26 **niemand**.

   **Erledigt.** `anchors.csv` liegt vor, wird eingelesen und im Bundle-Bau
   angewandt (Ablauf-Schritt 23).
3. **`records` (Facette Fraktion) — weiterhin offen, unverändert seit
   2026-09-01.** Fraktion
   liegt im Korpus vor (`redner_fraktion`, über `rede_id` joinbar): CDU/CSU
   12.326 Treffer, SPD 9.281, Grüne 4.969, `NA` 4.813, Linke 4.121, AfD 3.591,
   FDP 2.854, fraktionslos 208, BSW 89. Wort × Quartal × Fraktion ergibt
   **2.188 Zellen, davon 59 % mit unter 10 Treffern**, Median 6. Das Schema
   sieht das ausdrücklich vor (`freq` transportiert die Dünne, das Frontend
   kodiert sie), trotzdem bewusst entscheiden: alle Zellen liefern, ab einer
   Mindestgröße abschneiden, oder Fraktionen zusammenfassen? Wichtig:
   Record-Vektoren werden **in denselben Raum projiziert** (`zeile @ V[:, :k]`),
   niemals separat zerlegt — Invariante (1) „Ein Raum, kein Alignment".

   *Umgesetzt wie vorgeschlagen:* der Punkt wurde **übersprungen**, alle vier
   ausgelieferten Bundles exportieren `"records": []` und `"facets_schema": []`
   bewusst leer (Ablauf-Schritt 27). Die gemeinsame PCA läuft also nur über
   Anker + Aggregate — kommt die Fraktions-Facette später dazu, ist die
   bewusst eingeplante Neuprojektion in diesem Absatz weiterhin die Blaupause.
   **Das ist der einzige Punkt dieses Abschnitts, der noch offen ist.**
4. **Gemeinsame PCA (3 Komponenten) über *alle* Vektoren** — Aggregate +
   Anker zusammen → `coords_3d`. Analog `Readme Datenmodell.md` §8, Schritt 7.
   Die SVD liefert den Ähnlichkeitsraum, die 3er-PCA nur die Renderkoordinaten.
   (Records fließen, s. Punkt 3, bislang nicht ein.)

   **Erledigt** — Ablauf-Schritt 24.
5. **`neighbors`** sowie `meta`, `entities`, `time_slices` — reine Fleißarbeit,
   `facets_schema` bleibt an Punkt 3 hängen.

   **Erledigt, anders als hier geplant.** Nicht „Top-k-Kosinus gegen Anker und
   andere Zielwörter", sondern Kontrast-Auswahl (Rohwert minus Zeilen-
   Durchschnitt, wie schon in Ablauf-Schritt 21) gegen die vollständige
   belegungsgefilterte Kandidatenliste, und **k = 150 statt 8** — Begründung
   und Zahlen in Ablauf-Schritt 25 und `.claude/memory/vocab-radius-feature.md`.
   Dazu neu, hier nicht vorhergesehen: `vocab_points` (Ablauf-Schritt 26), weil
   150 Nachbarn ohne Koordinaten für die meisten von ihnen nutzlos wären.
6. **JSON-Export und Laden im Frontend** — der eigentliche Beweis, dass die
   Kette trägt.

   **Erledigt.** Alle vier Bundles liegen in `output/` und (bei aktiviertem
   Kopier-Schalter) in `backend/bundles/`. Größen liegen deutlich über der
   hier geschätzten ~9 MB (z. B. Halbjahr-Bundle 3,0 MB trotz `neighbors_k=150`,
   weil `vector_dim` mit 90–100 kleiner ist als hier angenommen und `neighbors`
   nur Label+Score+Group statt volle Vektoren trägt) — „Parquet statt JSON"
   war bislang nicht nötig, bleibt aber als geparkte Option in
   `.claude/memory/key-decisions.md` vermerkt.
7. Danach: Kiyama-Ähnlichkeitsmatrizen / Hamilton-Drift zwischen den
   Zeitscheiben vergleichen, insbesondere um 2022-Q1 (Zeitenwende-Rede
   27.02.2022) und 2014 (Krim). **Im Frontend nutzbar** — die Module dafür
   existieren seit 2026-08-07 (s. `.claude/memory/zeitreihen-module.md`) und
   lesen jedes SCHEMA.md-konforme Bundle, unabhängig vom Pfad.

## SGNS-Pfad (`pipeline-sgns.py`)

**Stand 2026-09-04: im Aufbau.** Was hier steht, ist der verabredete Plan;
„Ablauf" und „Stolpersteine" werden gefüllt, während das Notebook entsteht —
so wie es der PPMI-Pfad vorgemacht hat. Verfahrensbegründung, Abgrenzung und
Korpuszahlen stehen oben unter „Zwei Verfahren im Vergleich"; hier nur das,
was allein diesen Pfad betrifft.

### Was SGNS + Procrustes ist

SGNS = **S**kip-**G**ram with **N**egative **S**ampling, das word2vec-Verfahren
von Mikolov et al. (2013); in `gensim` ist das `Word2Vec(sg=1, negative=5, …)`.
Es trainiert je Zeitscheibe ein eigenes Modell auf dem Text dieser Scheibe.

Zwei solche Modelle sind zunächst **nicht vergleichbar**: die Zielfunktion legt
die Lösung nur bis auf eine Rotation fest. Derselbe Text, ein anderer Seed —
und alle Vektoren stehen gedreht im Raum, ohne dass sich an einer einzigen
Ähnlichkeit *innerhalb* des Modells etwas geändert hätte. Kosinuswerte zwischen
zwei unausgerichteten Modellen sind daher bedeutungslos, egal wie plausibel sie
aussehen.

Deshalb der zweite Schritt, **orthogonales Procrustes** (Hamilton et al. 2016,
das Verfahren aus dem Paper in `data/zeitenwende/`): gesucht ist die
Rotationsmatrix `R`, die `‖W_a · R − W_b‖_F` minimiert, wobei `W_a` und `W_b`
dieselben Wörter in derselben Zeilenreihenfolge enthalten müssen. Die Lösung
ist geschlossen und billig: `R = U · Vᵀ` aus der SVD von `W_aᵀ · W_b`. Weil `R`
orthogonal ist, bleiben Längen und Winkel innerhalb jeder Scheibe unverändert —
die Ausrichtung *erfindet* keine Ähnlichkeit, sie legt nur die Räume
übereinander. Was danach an Unterschied übrig bleibt, ist die gemessene
Verschiebung.

Der Preis gegenüber PPMI-SVD ist damit klar benannt: eine geschätzte Rotation,
deren Güte an den Stützwörtern hängt, plus ein Trainingsprozess, der bei jedem
Lauf etwas anderes liefert. Beides gibt es im PPMI-Pfad nicht — und beides
muss beziffert werden, bevor irgendeine Drift-Zahl geglaubt wird.

### Stufenplan

Bewusst klein anfangen. Jede Stufe hat ein **Abbruchkriterium**, damit nicht
erst nach Stunden Rechenzeit klar wird, dass nichts herauskommt.

**Stufe 0 — Textbasis für *eine* Zeitscheibe.**
Jahr **2022**: Zäsurjahr, 4,94 Mio. Token, also zugleich die realistische
Scheibengröße. Aus `Reden_Gesamt.csv` alle Reden dieses Jahres, `rede_text`
durch spaCy (nur Tagger + Lemmatizer, `n_process`), eine Tokenliste **je
Rede** — die Rede ist die Dokumentgrenze, damit das SGNS-Fenster nicht über
Rednerwechsel hinwegläuft. PUNCT raus, derselbe POS-Filter wie im PPMI-Pfad.
Ergebnis nach `cache/`, nicht nach `output/`.
*Zu protokollieren:* Token, Typen, Wanduhrzeit, Dateigröße — daran wird der
Volllauf über alle zwölf Jahre hochgerechnet.

**Stufe 1 — ein Modell, und der billigste wichtige Test des ganzen Plans.**
`Word2Vec(sg=1, negative=5, vector_size=100, window=5, epochs=5, …)`.
`vector_size=100` bewusst gleich dem SVD-`k`, damit `vector_dim` in beiden
Bundles übereinstimmt und der Vergleich nicht an der Dimensionszahl hängt.
Plausibilitätsprüfung: nächste Nachbarn von `Bundeswehr` 2022 gegen die
PPMI-SVD-Nachbarn derselben Zelle.
Dann **derselbe Lauf mit einem anderen Seed**. Die Überlappung der beiden
Nachbarschaftslisten ist der **Rauschboden des Verfahrens bei dieser
Korpusgröße**. Auf der PPMI-Seite gibt es dazu kein Gegenstück — die ist
deterministisch, ihr Rauschboden ist null.
*Abbruchkriterium:* ist die Seed-zu-Seed-Überlappung schon niedrig, ist alles
Weitere Rauschen, und der Pfad endet hier mit einem dokumentierten Negativ-
Ergebnis. Das wäre kein verlorener Aufwand, sondern genau die Auskunft, die
man vor einem DNB-Vor-Ort-Einsatz haben will.

**Stufe 2 — zweite Scheibe, Procrustes, und der zweite Rauschboden.**
Zweite Scheibe **2016** (vor der Zäsur, 4,07 Mio. Token, vergleichbar groß).
Zum Vokabular: die erste Fassung dieses Plans sah vor, **ein** festes
Vokabular über beide Scheiben zu bauen. Das ist am 2026-09-04 verworfen
worden — es tritt dieselbe Falle wie die Mindestbelegung im PPMI-Pfad, s.
Stolperstein „Das gemeinsame Vokabular ist nicht die Stützwortmenge".
Richtig ist: jede Scheibe behält ihr eigenes Vokabular, und nur die
**Stützwörter der Rotation** kommen aus dem Schnitt beider.
Erste inhaltliche Zahl: cos(`Bundeswehr`₂₀₁₆, `Bundeswehr`₂₀₂₂) nach
Ausrichtung — plus Gegenprobe an Wörtern, die sich *nicht* bewegt haben
sollten.
Bewertet wird sie gegen zwei Böden: den Seed-Boden aus Stufe 1 und einen
**Split-Half innerhalb 2022** (Reden der Scheibe zufällig halbieren, beide
Hälften unabhängig trainieren, ausrichten, Kosinus messen). Beide Hälften
beschreiben dasselbe Jahr — jeder Unterschied ist reines Stichprobenrauschen.
Drift unterhalb dieses Bodens ist keine Drift. Das ist derselbe Gedanke wie
Punkt 1c unter „Nächste Schritte (PPMI-Pfad)", hier aber nicht optional,
sondern die Voraussetzung dafür, dass eine Zahl überhaupt gelesen werden darf.

**Stufe 3 — alle zwölf Scheiben in einem Raum.**
Nur Jahresscheiben, 2014–2025 (Begründung oben bei den Korpuszahlen). Alle
Scheiben auf **eine Referenzscheibe** ausrichten, nicht verkettet t→t+1: eine
Kette akkumuliert Rotationsfehler und liefert am Ende gerade *keinen*
gemeinsamen Raum, den die 3D-Ansicht aber braucht.

**Stufe 4 — zweites Bundle und der Vergleich im Frontend.**
`output/bundestag_zeitenwende_sgns_jahr.json`, gleiches Schema.
`meta.projection` benennt das Verfahren ausdrücklich samt Seed — das ehrliche
Gegenstück zum jetzigen „per PPMI-SVD … KONSTRUIERT, nicht trainiert". Dann
macht die App den Vergleich fast von selbst: Drift-Kurven, Kiyama-Matrizen und
Nachbarschafts-Fluss existieren, man schaltet nur das Bundle um.
Quantitativ dazu: **Spearman-Korrelation der beiden Driftkurven je
Zielwort**, gemessen gegen die Rauschböden aus Stufe 1 und 2. Die ursprünglich
hier vorgesehene Jaccard-Überlappung der Nachbarlisten ist am 2026-09-04
gestrichen worden — sie misst die Definition von `neighbors`, nicht die Daten
(Begründung unter „Die Nachbarlisten der beiden Pfade sind nicht
vergleichbar").

14. **Auflösung je (Zielwort, Scheibe)** — Split-Half in jeder der zwölf
    Scheiben, Kontrast gegen den Hintergrund. Referenzwerte: Median der 500
    häufigsten Wörter je Jahr **0,349**, 90 %-Quantil über alle Wörter 0,294,
    Median über alle Wörter nur 0,128 (von seltenen Wörtern getragen).

    | Wort | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|
    | **Bundeswehr** | 0,43 | 0,38 | 0,36 | 0,37 | 0,42 | 0,42 | 0,40 | 0,40 | 0,47 | 0,40 | 0,36 | 0,39 |
    | **Frieden** | 0,40 | 0,37 | 0,43 | 0,40 | 0,38 | 0,45 | 0,41 | 0,36 | 0,43 | 0,39 | 0,43 | 0,40 |
    | **Sicherheit** | 0,33 | 0,34 | 0,30 | 0,36 | 0,36 | 0,35 | 0,33 | 0,29 | 0,32 | 0,32 | 0,38 | 0,36 |
    | Verteidigung | 0,36 | 0,21 | 0,23 | 0,08 | 0,38 | 0,30 | 0,28 | 0,16 | 0,41 | 0,33 | 0,40 | 0,41 |
    | Drohne | 0,29 | 0,12 | 0,15 | −0,01 | 0,10 | 0,23 | 0,25 | −0,01 | 0,28 | 0,24 | 0,26 | 0,36 |
    | Rüstung | 0,17 | 0,14 | 0,21 | 0,20 | 0,19 | 0,17 | 0,12 | – | 0,27 | 0,12 | 0,14 | 0,27 |
    | Landesverteidigung | 0,10 | 0,14 | – | – | 0,14 | 0,11 | 0,16 | – | 0,26 | 0,19 | 0,16 | 0,14 |
    | Rüstungsindustrie | 0,13 | 0,04 | 0,11 | 0,09 | 0,14 | 0,25 | 0,10 | 0,11 | 0,15 | 0,16 | 0,14 | 0,05 |
    | Hindukusch | 0,14 | 0,13 | 0,14 | 0,12 | 0,09 | 0,03 | 0,09 | 0,15 | 0,15 | – | 0,16 | – |
    | Wehrpflicht | 0,13 | 0,14 | 0,11 | 0,09 | 0,07 | 0,07 | 0,19 | – | 0,07 | 0,08 | 0,09 | 0,21 |
    | Friedenssicherung | 0,08 | 0,13 | 0,12 | 0,08 | 0,09 | 0,10 | 0,12 | – | 0,09 | 0,15 | 0,05 | 0,10 |

    **Drei von elf Zielwörtern sind über alle zwölf Scheiben belastbar**
    (Bundeswehr, Frieden, Sicherheit), zwei haben brauchbare Jahre mit Löchern
    (Verteidigung, Drohne — beide brechen 2017 und 2021 ein, die dünnsten
    Jahre), sechs liegen durchweg im nicht individuierten Bereich.
    `Drohne 2017` und `Drohne 2021` stehen bei −0,01: das Wort ist dort seiner
    eigenen Fortsetzung *unähnlicher* als einem beliebigen anderen Wort.
    Das ist kein Scheitern des Verfahrens, sondern seine Auskunft — die
    Zielwortliste wurde für ein Zeitungskorpus entworfen, in dem diese Wörter
    häufig sind. Im Bundestag sind sie es nicht (s. Rohhäufigkeiten unter
    „Offene Entscheidungen", Punkt 10).

15. **Zwei Ausrichtungsregime, gemessen statt gewählt.** Die Verkettung zweier
    getrennt geschätzter Rotationen ist nicht die optimale Direktrotation. Wie
    groß der Unterschied ist, wurde über alle 66 Jahrespaare × 4 Zielwörter
    gemessen: |direkt − über Referenz| Median **0,013**, 90 %-Quantil 0,05–0,06,
    max **0,12**. Der Median ist harmlos, das Maximum liegt in der
    Größenordnung der gesuchten Effekte. Daraus folgt:

    - **Driftkurven und Kiyama-Matrizen: paarweise direkt ausrichten.** 66
      Rotationen kosten nichts und sind exakt.
    - **`coords_3d`: über die gemeinsame Referenz**, dort ist es unvermeidlich
      — ein gemeinsamer Raum ist die Voraussetzung der 3D-Ansicht. Der
      Median-Fehler von 0,013 ist verkraftbar für eine Darstellung, die
      ohnehin nur einen Bruchteil der Struktur zeigt.

    Die Referenzwahl selbst ist dabei fast belanglos: 2014/2019/2022/2025
    liegen bei Median 0,0142 / 0,0126 / 0,0148 / 0,0158. 2019 bleibt, aber das
    Argument „mittige Referenz" trägt weniger, als beim Aufschreiben vermutet.

16. **Jahr-zu-Jahr-Drift — der eigentliche Befund.** Kosinus t → t+1, direkt
    ausgerichtet, in Klammern das Perzentil im eigenen Häufigkeitsband
    (niedrig = mehr Bewegung als vergleichbar häufige Wörter). „Hintergrund"
    ist der Median der 500 häufigsten Wörter des Paares.

    | Übergang | Hintergrund | Bundeswehr | Frieden | Sicherheit | Verteidigung |
    |---|---|---|---|---|---|
    | 14→15 | 0,812 | 0,786 (41 %) | 0,858 (95 %) | 0,829 (71 %) | 0,851 (69 %) |
    | 15→16 | 0,815 | 0,843 (86 %) | 0,821 (75 %) | 0,813 (49 %) | 0,856 (78 %) |
    | 16→17 | 0,810 | 0,782 (40 %) | 0,838 (85 %) | 0,830 (65 %) | 0,827 (58 %) |
    | 17→18 | 0,792 | 0,796 (62 %) | 0,835 (90 %) | 0,791 (45 %) | 0,809 (59 %) |
    | 18→19 | 0,819 | 0,837 (78 %) | 0,846 (88 %) | 0,863 (89 %) | 0,755 (25 %) |
    | 19→20 | 0,809 | 0,797 (50 %) | 0,786 (53 %) | 0,772 (27 %) | 0,794 (47 %) |
    | 20→21 | 0,806 | 0,823 (78 %) | 0,810 (72 %) | 0,805 (55 %) | 0,775 (30 %) |
    | **21→22** | **0,793** | **0,728 (20 %)** | 0,826 (90 %) | 0,804 (61 %) | **0,708 (13 %)** |
    | 22→23 | 0,821 | 0,849 (80 %) | 0,862 (96 %) | 0,800 (31 %) | 0,814 (67 %) |
    | 23→24 | 0,824 | 0,810 (49 %) | 0,885 (99 %) | 0,787 (21 %) | 0,860 (91 %) |
    | 24→25 | 0,805 | 0,790 (45 %) | 0,876 (99 %) | 0,852 (72 %) | 0,853 (94 %) |

    **Der Übergang 2021→2022 ist für `Bundeswehr` und `Verteidigung` der
    stärkste Sprung der gesamten Reihe** — bei beiden zugleich der niedrigste
    Kosinus und das niedrigste Perzentil. Und das gegen einen Hintergrund, der
    in diesem Übergang selbst absackt (0,793, zweitniedrigster Wert): 2022
    bewegt sich das ganze Vokabular ungewöhnlich stark, diese beiden Wörter
    bewegen sich trotzdem deutlich mehr als ihresgleichen. `Frieden` zeigt
    nichts (durchweg 72–99 %, bewegt sich also *weniger* als vergleichbare
    Wörter), `Sicherheit` ebenfalls nicht (21→22 bei 61 %).

17. **Wie stark ist dieser Befund? — die Nullverteilung.** Die erste Fassung
    dieses Abschnitts schloss, der SGNS-Pfad „findet die Zäsur unabhängig".
    Das ist so nicht haltbar, und die Korrektur ist wichtiger als der Befund
    selbst: **bei elf Übergängen ist ein Minimum-Perzentil von 0,20 nicht
    auffällig.** Wären die Perzentile eines Wortes zufällig verteilt, läge die
    Wahrscheinlichkeit, irgendwo unter 0,20 zu landen, bei 1 − 0,8¹¹ = **91 %**
    (für `Verteidigung` mit 0,13: 78 %). Jedes Wort für sich sagt nichts.

    Gemessen statt gerechnet: für 1.471 Wörter mit Werten in allen elf
    Übergängen wurde ausgezählt, wo ihr Minimum liegt. Das Ergebnis ist
    **flach** — 8,6 % bis 9,7 % je Übergang, gegen 9,1 % Zufallserwartung, mit
    2021→2022 bei 8,8 %. Nach der Perzentil-Normierung ist also **kein**
    Übergang von sich aus auffällig; die Normierung rechnet den Hintergrund
    vollständig heraus. Das ist zugleich die beste Bestätigung, dass die
    Normierung richtig gebaut ist.

    Was daraus folgt:

    - Ein einzelnes Wort mit Minimum bei 2021→2022 wäre bedeutungslos.
    - `Bundeswehr` **und** `Verteidigung` haben beide ihr Minimum dort. Unter
      Unabhängigkeit p ≈ 0,088² = **0,008**. Die beiden teilen aber Kontexte,
      sind also nicht unabhängig — der wahre Wert liegt höher.
    - `Frieden` hat sein Minimum bei 2019→2020, und selbst dort steht es bei
      pz 0,53, also über dem Median; `Sicherheit` bei 2023→2024 (pz 0,21).
      Von den vier belastbaren Wörtern zeigen nur die beiden
      militärisch-institutionellen auf 2022.

    **Fazit, so weit es trägt:** Der SGNS-Pfad findet die Zäsur an der Stelle,
    an der die Projekthypothese sie erwartet, bei genau den Wörtern, die der
    Kontrast-Test vorher als belastbar ausgewiesen hatte, und mit einer
    Nullverteilung, gegen die sich das beziffern lässt. Er ruht dabei auf
    **zwei Wörtern und einem einzigen Seed** — ein starkes Indiz, kein Beweis.
    Was ihn absichern würde, ist die Seed-Mittelung aus der offenen
    Entscheidung 7: dieselbe Rechnung mit M = 5 Seeds je Scheibe, und die
    Frage lautet dann, ob die beiden Minima seedstabil auf 2021→2022 bleiben.
    Das ist der Unterschied zwischen einem Indiz und einem Ergebnis, und er
    kostet ~40 Minuten Rechenzeit.

    Dass der PPMI-Pfad diese Fehlerrechnung nicht hat, ist übrigens kein
    Vorteil, sondern eine offene Flanke: er ist reproduzierbar, aber seine
    Driftkurven stehen bisher ohne jede Nullverteilung da.

18. **Seed-Wiederholung (5 Seeds) — was vom Befund bleibt.** Fünf
    vollständige, unabhängige Wiederholungen der ganzen Kette (nicht eine
    Mittelung von Vektoren, s. Zelle 29), 60 Modelle, ~40 min. Gecacht werden
    nur die winzigen Ergebnistabellen `cache/pz_seed<n>.parquet`, nicht die
    Modelle. Die Tabelle ist **nur nach Kontrast sortiert lesbar**:

    | Wort | Kontrast Ø | Seeds mit Min. bei 21→22 | pz_min Ø | häufigstes Minimum |
    |---|---|---|---|---|
    | Frieden | 0,404 | **0/5** | 0,574 | 2016→2017 |
    | **Bundeswehr** | 0,400 | **3/5** | 0,201 | **2021→2022** |
    | Sicherheit | 0,337 | **0/5** | 0,161 | 2017→2018 |
    | **Verteidigung** | 0,296 | **4/5** | 0,162 | **2021→2022** |
    | *— Auflösungsgrenze —* | | | | |
    | Drohne | 0,188 | 1/5 | 0,003 | 2017→2018 |
    | Rüstung | ~0,18 | 5/5 | 0,036 | 2021→2022 |
    | Landesverteidigung | 0,157 | 1/5 | 0,023 | 2017→2018 |
    | Rüstungsindustrie | 0,124 | 3/5 | 0,024 | 2021→2022 |
    | Hindukusch | 0,119 | 0/5 | 0,196 | 2024→2025 |
    | Wehrpflicht | 0,114 | 0/5 | 0,003 | 2020→2021 |
    | Friedenssicherung | 0,101 | 1/5 | 0,458 | 2015→2016 |

    *Oberhalb der Grenze — eine saubere Dissoziation.* Vier auflösbare Wörter,
    und sie teilen sich so, wie die Hypothese es verlangt: `Bundeswehr` (3/5)
    und `Verteidigung` (4/5) treffen die Zäsur, `Sicherheit` und `Frieden`
    treffen sie in **keinem** Seed. `Frieden` bewegt sich überhaupt nicht —
    selbst sein stärkster Übergang liegt bei pz 0,574, über dem Median
    vergleichbarer Wörter. Gegen die Nullverteilung (p = 0,088 je Übergang):
    P(≥3 von 5) = **0,006**, P(≥4 von 5) = **0,0003**.

    *Der Befund ist flacher als beim Einzel-Seed.* `Bundeswehr` kommt auf 3/5,
    nicht 5/5, und sein Minimum liegt in jedem Seed bei pz ≈ 0,20, nie tiefer.
    Belastbar ist deshalb: **`Bundeswehr` und `Verteidigung` haben sich an der
    Zäsur stärker verändert als rund 80 % vergleichbar häufiger Wörter.**
    Nicht: sie sind ausgerissen. Der stärkste Teil des Befunds ist nicht die
    Höhe des Ausschlags, sondern die Dissoziation — die beiden
    militärisch-institutionellen Wörter reagieren, die beiden abstrakten
    nicht.

    *Unterhalb der Grenze — eine Fehlersignatur.* Alle sechs Wörter dort haben
    `pz_min` nahe null (0,003–0,036), sind also in ihrem Minimum-Übergang das
    am stärksten bewegte Wort ihres ganzen Häufigkeitsbands. Das ist keine
    Bedeutungsverschiebung, sondern die Signatur eines nicht auflösbaren
    Vektors, und das Minimum landet dort, wo das dünnste Jahr liegt:
    `Wehrpflicht` bei 2020→2021 (3 Vorkommen 2021), `Drohne` und
    `Landesverteidigung` bei 2017→2018 (Drohne hat 2017 Kontrast −0,01).

    **`Rüstung` ist die Falle — und der beste Beleg dafür, dass die
    Kontrast-Schranke ihr Geld wert war.** 5 von 5 Seeds, das nominell
    stärkste Ergebnis der Tabelle, bei Kontrast ~0,18 und **4 Vorkommen im
    Jahr 2021**. Der 2021er-Vektor ist Rauschen, jeder Übergang, der ihn
    berührt, explodiert, und ob das Minimum auf 2020→2021 oder 2021→2022
    fällt, ist Münzwurf. Ohne die Schranke wäre `Rüstung` als Hauptergebnis
    präsentiert worden.

    *Die methodische Lehre, allgemeiner als dieser Datensatz:*
    **Seed-Stabilität allein ist nicht interpretierbar.** Sie schützt gegen
    Zufallsrauschen, nicht gegen Artefakte — ein systematisch dünnes Jahr
    produziert einen Fehlbefund, der in allen Seeds gleich aussieht. Erst
    Kontrast-Schranke *und* Seed-Stabilität zusammen ergeben eine Aussage.

### Offene Entscheidungen im SGNS-Pfad

1. ~~Anker sind im Schema zeitlos, SGNS gibt jedem Wort einen Vektor pro
   Scheibe.~~ **Entschieden 2026-09-04, durch Messung: Schwerpunkt über die
   ausgerichteten Vektoren aller Scheiben, in denen das Wort belegt ist.**
   Im PPMI-Pfad fällt der Ankervektor als Spaltenseite an und existiert genau
   einmal (Ablauf-Schritt 18); im ausgerichteten SGNS-Raum hat „Soldat" zwölf.
   Die Alternative „Vektor der Referenzscheibe" ist ausgeschlossen, und zwar
   nicht aus Geschmack: von den 86 Lemmata in `anchors.csv` existieren **45 in
   allen zwölf Scheiben, 41 lückenhaft, keines gar nicht**. Die Lücken sind
   kein Defekt, sondern Inhalt — `Taurus`, `Kriegstüchtigkeit`,
   `Munitionsbeschaffung`, `F-35`, `Sonderschuld`, `Kriegsbeginn`,
   `Bundeswehrbeschaffungsbeschleunigungsgesetz`; **elf Anker existieren in
   genau 4 von 12 Scheiben, also 2022–2025**. In der Referenzscheibe 2019 gibt
   es sie nicht, die Referenzvariante würde also ausgerechnet die Wörter
   löschen, die den Wandel markieren.

   Technisch trägt der Schwerpunkt, weil die Rotation aus den **Stützwörtern**
   geschätzt wird und nicht aus dem Anker selbst: die 2022er-Fassung von
   `Taurus` lässt sich in den 2019er-Rahmen drehen, obwohl das Wort dort fehlt.

   *Das war das dritte Mal.* Erst `anker_mindestzellen = 50` (`Sondervermögen`
   fiel heraus), dann das „feste Vokabular über alle Scheiben" (`Zeitenwende`
   fiel heraus), jetzt die Referenzscheibe. **Jede Regel, die Zeitstabilität
   misst, trifft zuerst das Neue** — und in einem Projekt über
   Bedeutungswandel ist das Neue der Gegenstand. Bei jeder weiteren Schwelle
   zuerst prüfen, was sie mit dem Vokabular *nach* der Zäsur macht.

   Nebenbefund für die Kuration: die Cluster sind sehr ungleich vollständig
   (`Bundeswehr` 8/22, `Politiker` 3/8, `Hindukusch` 6/7). Bei `Politiker` ist
   das erwartbar — Personennamen kommen und gehen mit den Wahlperioden.
2. **Es bricht eine Invariante, die `specs.md` §2 ausdrücklich festhält**
   („Ein gemeinsamer Vektorraum, kein Alignment … deterministisch,
   alignmentfrei"). Als Kontrollverfahren ist das in Ordnung, aber der Satz
   braucht einen Zusatz — sonst hält der nächste Leser das SGNS-Bundle für
   einen Fehler.
3. **Normierung.** SGNS-Vektornormen tragen Häufigkeitsinformation; Procrustes
   rechnet man üblicherweise auf L2-normierten Zeilen. Ob auch *exportiert*
   wird, was normiert wurde, betrifft `coords_3d` und jedes Modul, das mit
   Kosinus rechnet. Einmal entscheiden und aufschreiben — der PPMI-Pfad hat zu
   Normierung bereits eine teuer bezahlte Lektion (s. Stolperstein
   „Zeilennormierung vor der SVD").
4. **Downsampling auf gleiche Tokenmenge je Scheibe** (Faktor 1,8 zwischen 2017
   und 2023). Für PPMI verkraftbar, für ein trainiertes Modell nicht
   selbstverständlich. Optionen: auf das dünnste Jahr herunterziehen, oder
   ungleich lassen und den Effekt ausweisen.
5. **Welche Referenzscheibe.** Letztes volles Jahr, Zäsurjahr oder Mitte des
   Zeitraums — die Wahl bestimmt, wo der Raum am schärfsten ist.
6. **`vector_size = 100` ist übernommen, nicht begründet** — wie `k = 100` auf
   der PPMI-Seite. Der Vergleichbarkeit wegen erst einmal gleich lassen, aber
   als Kalibrierungspaar mit `window`, `min_count` und `epochs` im Blick
   behalten.
7. **Seed-Mittelung, ja oder nein** (neu nach der Messung in Stufe 1). Bei
   ~50 % Seed-Abhängigkeit der Nachbarlisten ist die naheliegende Antwort,
   je Scheibe *mehrere* Modelle zu trainieren und zu aggregieren — bei 40 s je
   Modell kostet M = 5 über zwölf Scheiben 40 Minuten, also bezahlbar. Offen
   ist das Wie: Nachbarlisten über Seeds auszählen (robust, aber liefert keine
   Vektoren fürs Bundle) oder die Modelle erst paarweise per Procrustes
   ausrichten und dann die Vektoren mitteln (liefert Vektoren, fügt aber eine
   zweite Ausrichtungsebene hinzu). Hamilton et al. mitteln nicht, Antoniak &
   Mimno empfehlen es ausdrücklich. Entscheidung spätestens in Stufe 3.
8. **`min_count = 10` ist zu hoch — drei der elf Zielwörter fallen heraus.**
   In 2022 ergibt die Schwelle 13.607 Wörter, im Schnitt mit 2016 noch 9.404.
   Darin fehlen `Landesverteidigung`, `Friedenssicherung` und `Hindukusch`,
   also gut ein Viertel der Zielwortliste — und das sind genau die dünnsten
   Wörter, die schon im PPMI-Pfad die wenigsten belegten Zeitscheiben haben.
   Zu entscheiden: `min_count` senken (5 oder 3), die Zielwörter unabhängig
   von der Schwelle ins Vokabular zwingen, oder Lücken zulassen — das Schema
   sieht sie über `freq` ausdrücklich vor, und ein Wort ohne Vektor in einer
   Scheibe ist ehrlicher als ein Vektor aus neun Vorkommen. Betrifft nur das
   *Modell*-Vokabular, nicht die Stützwörter der Rotation.
9. **Wie die Drift ausgewiesen wird** — nach der Messung vom 2026-09-04 ist
   die Antwort weitgehend erzwungen: **Perzentil im eigenen Häufigkeitsband**
   plus **Kontrast** als Gütemaß, nicht der rohe Kosinus (s. Stolpersteine).
   Offen bleibt, wie das ins Schema kommt: `aggregates` hat kein Feld dafür.
   Kandidaten sind ein zusätzliches Feld neben `freq` (das Schema ist
   `.passthrough()`) oder `freq` selbst so zu füllen, dass das Frontend die
   Unsicherheit schon über die vorhandene Konfidenz-Kodierung zeigt. Betrifft
   auch den PPMI-Pfad: solange dessen Driftkurven roh sind, liegen die beiden
   Bundles nicht auf derselben Skala.
10. **Der Kontrast als Auslieferungsschranke statt `min_count`.** Die
    Zielwörter sind im Volltext selten — Minimum über die zwölf Jahre:
    `Hindukusch` 2, `Landesverteidigung` 2, `Wehrpflicht` 3, `Rüstung` 4,
    `Friedenssicherung` 6, `Rüstungsindustrie` 7, `Drohne` 10. Selbst
    `min_count = 3` verliert `Hindukusch` und `Landesverteidigung`; die
    Schwelle zu senken ersetzt „fehlt" nur durch „Zufallsvektor". Der
    PPMI-Pfad kann mit solchen Wörtern arbeiten, weil er nichts schätzt — er
    baut eine Zeile aus allen Kontexten, und die SVD leiht ihr Struktur aus
    den übrigen 501 Zeilen. SGNS hat diese Anleihe nicht. Vorschlag daher:
    `min_count` niedrig (3) für das Modellvokabular, und die Entscheidung, ob
    eine Zelle ins Bundle darf, am gemessenen **Kontrast** festmachen statt an
    einer Häufigkeit. Damit ist zugleich die Frage beantwortet, wie die
    Zielwortlücken behandelt werden: sie bleiben Lücken, und `freq` trägt es.

### Ablauf in `pipeline-sgns.py`

*(Schritte 1–13 sind Stand 2026-09-04/-07, ausführlich dokumentiert und seither
inhaltlich unverändert; Schritt 14 fasst den seither gelaufenen Bundle-Bau
zusammen, s. dort.)*

1. **Korpus laden** — `korpus`: drei Spalten (`rede_id`, `sitzung_datum`,
   `rede_text`) aus `Reden_Gesamt.csv`, **alle 78.188 Reden**. Das ist der
   erste sichtbare Unterschied zu `pipeline.py`, das dieselbe Datei liest,
   danach aber auf die 15.785 Reden mit Zielwort-Treffer einengt. 302 MB im
   Speicher, 0,2 s — die 382 MB auf der Platte schrumpfen, weil nur drei der
   35 Spalten gelesen werden.
2. **Zeitscheibe** — `korpus_jahre`: Jahr aus `sitzung_datum`, als String,
   damit die lexikografische Sortierung der chronologischen entspricht (wie im
   PPMI-Pfad). **Kein Raster-Umschalter**: für ein trainiertes Modell sind nur
   Jahresscheiben groß genug. Dazu `volumen`, eine Übersicht Reden/Wörter je
   Jahr — grobe Wortzahl über `str.count_matches(" ") + 1` statt über spaCy,
   was 0,6 s statt Minuten kostet und für die Größenordnung reicht. Die Zahlen
   liegen ~4 % unter der `tokens`-Spalte der Metadaten-CSV (2022: 4,75 gegen
   4,94 Mio.), weil ein echter Tokenizer Satzzeichen abtrennt.
3. **Lemmatisierung je Scheibe** (~1:30 min je Jahr, Zelle läuft mit — sie
   schützt sich über die Existenzprüfung je Datei, nicht über `disabled`) —
   spaCy `de_core_news_md`, `nlp.pipe(disable=["parser","ner",
   "senter"], n_process=8, batch_size=50)`, **eine Tokenliste je Rede**. Die
   Rede ist die Einheit, weil sie später die Dokumentgrenze fürs
   Trainingsfenster ist und die Ziehungseinheit für den Split-Half-Test in
   Stufe 2. Ergebnis je Scheibe nach `cache/lemmata_<jahr>.parquet`
   (Listenspalten → Parquet, nicht CSV), Folgezelle liest von dort. Die Zelle
   überspringt bereits vorhandene Dateien und ist damit wiederaufnehmbar.
   Gemessen an 2022: 8.196 Reden, **4.257.530 Token**, 16 MB, 1:30 min.

   *Durchsatz, gemessen an 400 Reden aus 2022:*

   | `n_process` | Wörter/s | Hochrechnung eine Scheibe |
   |---|---|---|
   | 1 | 12.540 | 6,3 min |
   | 8 | **60.862** | **1,3 min** |
   | 16 | 61.179 | 1,3 min |

   Ab 8 Prozessen sättigt es — was dann bremst, ist nicht die Analyse, sondern
   das Serialisieren der Texte zwischen den Prozessen. Also `n_process=8`,
   obwohl die Maschine 32 Kerne hat.

4. **POS und Lemma getrennt gespeichert, Filter erst stromabwärts.** In
   Schritt 3 fliegen nur `PUNCT` und `SPACE` raus; der eigentliche POS-Filter
   sitzt in einer eigenen, billigen Zelle. Grund ist eine Lektion aus dem
   PPMI-Pfad: dort lag `kontext_pos` ungenutzt in der Parquet-Datei, und als
   die Frage nach einem POS-Filter aufkam, war sie **ohne neuen spaCy-Lauf**
   testbar. Der Cache ist damit entscheidungsneutral — alles, was noch offen
   ist (POS-Filter, Vokabularschwelle, Downsampling, Fenster, Seed), hängt
   stromabwärts. Ein Neulauf wäre nur bei einer Änderung am Tokenizer oder an
   der Groß-/Kleinschreibung nötig.
5. **Keine Kleinschreibung.** Abweichung von der word2vec-Konvention, und
   zwar bewusst: der PPMI-Pfad schreibt ebenfalls nicht klein, `anchors.csv`
   enthält `Soldat`, `Einsatzkräft`, `Sondervermögen` groß, und die Anker
   werden über **exakte** Lemma-Strings nachgeschlagen. Kleinschreibung hier
   ließe die Anker im SGNS-Bundle stillschweigend ins Leere laufen — kein
   Fehler, nur leere Landmarken. Im Deutschen kostet die Großschreibung
   ohnehin wenig, weil sie Wortarten trennt statt Dubletten zu erzeugen.
6. **Cache-Übersicht** — `cache_uebersicht`: Reden, Token und Anteil am
   größten Jahr je Scheibe. Ergebnis über alle vierzehn Scheiben:
   **43.810.079 Token, 167 MB.** Die Randscheiben bestätigen sich mit Anteil
   0,04 (2013) und 0,03 (2026), also zwei Größenordnungen unter dem Rest —
   kein Grenzfall. Es bleiben **zwölf Scheiben, 2.408.626 (2017) bis
   4.312.059 Token (2023), Faktor 1,79**. Die drei dünnsten vollen Jahre
   (2017 · 0,56, 2021 · 0,66, 2025 · 0,69) sind Wahljahre mit langer
   Regierungsbildung — dieselben Dellen, die der PPMI-Pfad in den Quartalen
   sieht, also plausibel und kein Defekt.
7. **POS-Filter und Trainingssätze** — dieselbe Verbotsliste wie im PPMI-Pfad
   (`DET, ADP, PRON, AUX, CCONJ, SCONJ`; `PART` bleibt). Angewandt auf den
   Cache statt bei der Lemmatisierung, also ohne spaCy-Lauf, 0,8 s je Scheibe.
   Behält in 2022 **2.300.975 von 4.257.530 Token (54,0 %)** — deckt sich mit
   dem PPMI-Pfad (dort 53,2 % behalten). Ergebnis ist eine Liste von
   Tokenlisten, eine je Rede: genau das Format, das `gensim.Word2Vec` erwartet.
8. **Training je Scheibe** — `Word2Vec(sg=1, negative=5, vector_size=100,
   window=5, min_count=10, epochs=5, workers=1, seed=…)`. `sg=1` plus
   `negative=5` *ist* SGNS; `vector_size=100` bewusst gleich dem SVD-`k`.
   Vokabular in 2022: **13.607 Wörter**. Laufzeit **40 s** je Scheibe bei
   `workers=1` (6 s bei `workers=8` — warum trotzdem 1, s. Stolpersteine).
   Zwölf Scheiben also gut 8 Minuten.

   *Erster inhaltlicher Beleg.* Nächste Nachbarn von `Bundeswehr` in 2022:
   Streitkraft, Ausrüstung, Auslandseinsatz, Landesverteidigung,
   Einsatzbereitschaft, Bündnisverteidigung, Sondervermögen, einsatzfähig,
   Verteidigungsfähigkeit, einsatzbereit. Das ist das Zeitenwende-Vokabular,
   und es überschneidet sich an mehreren Stellen mit dem, was der PPMI-Pfad
   für `Bundeswehr | 2022-Q1` liefert (einsatzbereit, Einsatzbereitschaft,
   Verteidigungshaushalt, …) — zwei Verfahren mit verschiedenen blinden
   Flecken kommen auf verwandte Listen. `Sicherheit` zeigt dagegen prompt
   seine Polysemie (äußerer, innerer, Gewährleistung, Stabilität), passend zum
   entsprechenden Punkt unter „Offen".
9. **Rauschboden (Stufe 1, das Abbruchkriterium).** Zwei Modelle auf
   identischen Daten, Seed 1 gegen Seed 2, `workers=1`. Mittlere
   Top-10-Überlappung: **11 Zielwörter 0,622 · 500 häufigste 0,610 · 500
   zufällige 0,522 · 500 seltenste 0,413**. Rund die Hälfte einer Nachbarliste
   hängt am Seed.
   *Das Abbruchkriterium greift nicht* — und zwar aus einem Grund, den die
   Nebeneinander-Tabelle direkt zeigt: was wackelt, ist der Schwanz der
   Rangliste, nicht das thematische Zentrum (`Streitkraft`, `Ausrüstung`,
   `Einsatzbereitschaft`, `Sondervermögen` stehen in jedem Lauf). Der später
   tatsächlich ausgewertete Wert ist zudem der Kosinus über alle 100
   Dimensionen, also ein Aggregat und deutlich ruhiger als eine
   Top-10-Rangfolge. Der Pfad geht weiter — aber jede Drift-Zahl aus Stufe 2
   gehört neben diese Tabelle gestellt.
10. **Direktvergleich der Nachbarlisten beider Pfade** — Ergebnis und seine
    Folgen für den Stufenplan stehen im allgemeinen Abschnitt „Die
    Nachbarlisten der beiden Pfade sind nicht vergleichbar", weil es eine
    Aussage über das Verhältnis der Verfahren ist und nicht über diesen
    Pfad allein.
11. **Zweite Scheibe und Stützwörter** (Stufe 2). 2016 gegen 2022, je ein
    eigenes Modell mit eigenem Vokabular (13.155 bzw. 13.607 Wörter), Schnitt
    **9.404 Wörter (69 % von 2022)**. Stützwörter der Rotation sind die 5.000
    häufigsten daraus, gewichtet nach dem *Minimum* der Häufigkeit in beiden
    Scheiben — ein Wort, das nur in einer Scheibe häufig ist, ist kein guter
    Stützpunkt.
12. **Orthogonales Procrustes** — `R = U · Vᵀ` aus `SVD(A_normiertᵀ ·
    B_normiert)`, angewandt auf **alle** Vektoren der Quellscheibe, nicht nur
    auf die Stützwörter. Zwei Prüfungen, beide bestanden:
    `max|R·Rᵀ − I| = 3,6 · 10⁻⁷` (R ist orthogonal) und
    `max|cos_vorher − cos_nachher| = 6,6 · 10⁻⁷` für 300 zufällige Wörter
    *innerhalb* von 2016 — **die Ausrichtung erfindet keine Ähnlichkeit**,
    sie dreht nur. Die zweite Prüfung ist die wichtigere: ein Vorzeichen- oder
    Transpositionsfehler in `R` fällt in den Drift-Zahlen nicht auf, hier
    schon.
13. **Die drei Böden** (das Ergebnis von Stufe 2). Alle als Median des
    Kosinus über das jeweils gemeinsame Vokabular:

    | Vergleich | Median | `Bundeswehr` |
    |---|---|---|
    | Seed-Boden — identische Daten, anderer Seed | **0,953** | — |
    | Split-Half — halbe Daten, dieselbe Scheibe (2022) | **0,873** | 0,859 |
    | **2016 → 2022, echte Zeitdifferenz** | **0,764** | **0,765** |

    *Korrektur am Verfahren, 2026-09-07.* Die Diagnosezelle trainierte beide
    Hälften mit `seed=1`. Das ist falsch: in gensim 4.4 heißt gleicher Seed
    dieselbe Startmatrix (s. Stolperstein zur Initialisierung), und weil die
    Häufigkeitsränge oben im Vokabular zwischen zwei Hälften fast identisch
    sind, starten dort beide Modelle am selben Punkt — ausgerechnet bei den
    Wörtern, aus denen gleich die Rotation geschätzt wird. Die produktive
    Hälften-Zelle machte es schon richtig (Seeds 1 und 2), nur diese
    Diagnosezelle hing hinterher; sie ist jetzt nachgezogen.

    Nachgemessen, und das Ergebnis ist eine Warnung vor der eigenen Intuition:
    **der Median-Boden ändert sich nicht** — 0,873 vorher, 0,873 nachher, über
    dieselben 11.687 gemeinsamen Wörter. Die vermutete Verzerrung der Kennzahl
    gibt es nicht. Was sich bewegt, ist die Einzelwortspalte: `Bundeswehr` fällt
    von 0,889 auf 0,859. Das ist keine Verzerrung, sondern schlicht die
    Seed-Streuung eines einzelnen Wortes (Größenordnung ±0,03–0,05, s. u.) —
    und genau deshalb steht in diesem Dokument an mehreren Stellen, dass ein
    Einzelwort-Kosinus ohne Boden daneben nicht gelesen werden darf. Der Fix
    bleibt trotzdem richtig: zwei Hälften, die als unabhängig verkauft werden,
    müssen unabhängig gezogen sein.

    Der Zeitunterschied liegt klar unter beiden Böden: **das Verfahren misst
    etwas.** Der Split-Half-Wert ist dabei eine konservative Schätzung, weil
    seine Modelle nur die halbe Datenmenge sehen; der wahre Boden für zwei
    volle Scheiben liegt zwischen 0,873 und 0,953.

    Gegenprobe an den Extremen der Liste, und sie fällt richtig aus: am
    stärksten bewegt haben sich `Angriffskrieg`, `Ampel`, `Nord`, `Fracking`;
    am wenigsten `Verehrter`, `Geehrte`, `zuhörerinnen`, `brutto` und Zahlen —
    die Anredeformeln des Plenums stehen still, weil ihr Kontext sich nie
    ändert.

14. **Bundle-Bau (Stufe 3, seit 2026-09-07/-11 gelaufen).** Diese Nummerierung
    endet bei Stufe 2 (2026-09-04); was danach kam, ist an den beiden Stellen
    dokumentiert, an denen es eigene Abschnitte verdient hat, statt hier ein
    viertes Mal dieselben Mechanismen zu erklären: die Auflösungs-Gütemaße
    `kontrast`/`generik` (Split-Half-Reproduzierbarkeit bzw. Ähnlichkeit zum
    Vokabular-Durchschnitt) in „Was das SGNS-Bundle aushält" unten, `neighbors`
    (`nb_k = 150`, **rohe Kosinus-Auswahl statt Kontrast** — Begründung in
    `data/test/SCHEMA.md`) und `vocab_points` (dieselbe Out-of-Sample-
    Projektion wie im PPMI-Pfad, Ablauf-Schritt 26 dort) im Abschnitt „Was
    beide Pfade teilen" und in `.claude/memory/vocab-radius-feature.md`. Die
    gemeinsame PCA nutzt dieselben 86 kuratierten Anker wie der PPMI-Pfad
    (`anchors.csv` wird von beiden Notebooks gelesen). Export analog:
    `output/bundestag_zeitenwende_sgns.json`, per Schalter nach
    `backend/bundles/` kopierbar.

### Was gecacht wird und was nicht

Gemessen am 2026-09-04, je Jahresscheibe:

| Schritt | Laufzeit | gecacht als |
|---|---|---|
| Lemmatisierung (spaCy, `n_process=8`) | 1:30 min | `cache/lemmata_<jahr>.parquet` |
| Vollmodell (`min_count=3`) | 49 s | `cache/wv_<jahr>.kv` |
| zwei Halbmodelle (`min_count=2`) | 50 s | `cache/wv_<jahr>_h1.kv`, `_h2.kv` |
| POS-Filter auf einer Scheibe | 0,8 s | — |
| Kontrast-Tabelle zweier Modelle | 0,2 s | — |

Über zwölf Scheiben also ~18 min Lemmatisierung, ~10 min Vollmodelle, ~10 min
Halbmodelle. Alle drei Schleifen überspringen vorhandene Dateien und sind
damit wiederaufnehmbar.

**Gespeichert wird nur `model.wv`, nicht das ganze `Word2Vec`-Objekt.** Die
Ausgabegewichte (`syn1neg`) werden nach dem Training nie wieder gebraucht und
sind die Hälfte der Datei — 14 statt 27 MB. `KeyedVectors` behält alles
Nötige: Vektoren, `get_vecattr(w, "count")` und `index_to_key` in
Häufigkeitsreihenfolge (verifiziert). Preis ist eine Signaturfrage: die
Hilfsfunktionen nehmen deshalb ein `wv`-Objekt entgegen, nicht ein Modell —
bei einem frisch trainierten `Word2Vec` also `modell.wv` übergeben.

### Stolpersteine (SGNS-Pfad)

- **gensim gibt es nicht für Python 3.14.** Version 4.4.0 liefert Wheels bis
  `cp313`; für 3.14 existiert nur ein sdist, das gegen Cython/C gebaut werden
  müsste. Deshalb ist das gesamte Projekt auf 3.13 gepinnt (s. „Setup"). Wer
  `.python-version` wieder anhebt, bricht diesen Pfad — nicht den PPMI-Pfad,
  was den Fehler umso verwirrender macht.

- **`workers > 1` macht das Training auch bei festem `seed` nicht
  reproduzierbar.** gensim verteilt die Sätze auf Threads; die Reihenfolge, in
  der Gewichtsaktualisierungen eintreffen, hängt am Scheduler des
  Betriebssystems und nicht am Seed. Gemessen an 2022: zwei Läufe mit
  **demselben** Seed und `workers=8` teilen bei `Bundeswehr` nur 8 von 10
  nächsten Nachbarn. Mit `workers=1` sind zwei Läufe bei gleichem Seed
  **bitgleich** (`np.array_equal` über die Vektormatrix: True). Der Preis ist
  40 s statt 6 s je Scheibe, über zwölf Scheiben also 8 statt 1,2 Minuten —
  gemessen an dem, was Reproduzierbarkeit hier wert ist, ein Trinkgeld.
  Deshalb steht `workers=1` fest, nicht als Vorschlag. Nebeneffekt, der den
  Fehler tückisch macht: mit `workers=8` misst ein Seed-Vergleich zwei
  Rauschquellen auf einmal und kann sie nicht trennen.

- **Reproduzierbar ist nicht stabil — und der Unterschied ist der ganze
  Punkt.** Mit `workers=1` und festem Seed liefert das Notebook bei jedem Lauf
  exakt dieselben Zahlen. Das heißt aber nur, dass *dieselbe Ziehung*
  wiederholt wird, nicht dass die Ziehung viel bedeutet. Gemessen an 2022,
  Seed 1 gegen Seed 2, mittlere Überlappung der Top-10-Nachbarlisten:

  | Wortgruppe | Überlappung |
  |---|---|
  | 6 Zielwörter | **65 %** |
  | 500 häufigste Wörter | 61 % |
  | 500 zufällige Wörter | 52 % |
  | 500 seltenste Wörter | 41 % |

  Ungefähr die **Hälfte einer Top-10-Liste ist seedabhängig** — das ist der
  Befund von Antoniak & Mimno (2018), an diesen Daten reproduziert, und er ist
  kein Argument gegen das Verfahren, sondern eine Maßeinheit. Zwei Folgen:
  (1) eine einzeln trainierte `neighbors`-Liste im SGNS-Bundle wäre zur Hälfte
  Rauschen und braucht Seed-Mittelung oder einen sichtbaren Vorbehalt;
  (2) jede Drift-Zahl gehört neben diesen Boden gestellt, nie allein. Die
  Kernaussage bleibt davon unberührt: das *thematische* Zentrum ist über Seeds
  hinweg stabil (Streitkraft, Ausrüstung, Einsatzbereitschaft, Sondervermögen
  stehen in jedem Lauf), es wackelt der Ranglisten-Schwanz. Und der eigentlich
  gemessene Wert — der Kosinus über alle 100 Dimensionen — ist ein Aggregat und
  damit deutlich ruhiger als eine Top-10-Rangfolge.
  **Der PPMI-Pfad hat hier keinen Rauschboden, weil er keinen braucht: sein
  Ergebnis ist eine Rechnung, keine Ziehung.**

- **Das gemeinsame Vokabular ist nicht die Stützwortmenge.** Der erste Plan
  war, *ein* festes Vokabular über alle Scheiben zu bauen, damit `W_a` und
  `W_b` identische Zeilen haben. Das ist falsch, und zwar auf dieselbe Weise
  wie die zu hohe `anker_mindestzellen`-Schwelle im PPMI-Pfad: `Zeitenwende`
  ist in 2016 nicht im Vokabular und in 2022 sehr wohl — ein Schnitt über alle
  Scheiben löscht genau die Wörter, wegen denen das Projekt existiert.
  Richtig ist die Trennung: `R` ist eine 100×100-Rotation, geschätzt aus den
  **Stützwörtern** (häufig und in beiden Scheiben vorhanden), danach aber auf
  **alle** Vektoren der Scheibe angewandt. Scheibenspezifisches Vokabular
  behält damit seinen Platz im gemeinsamen Raum, ohne die Ausrichtung zu
  beeinflussen. Dieselbe Lehre, zweiter Pfad: eine Schwelle, die Zeitstabilität
  misst, trifft immer zuerst das Neue.

- **Hohe Selbstähnlichkeit seltener Wörter ist keine Auflösung, sondern
  fehlende Individuierung.** Der Split-Half-Kosinus *steigt*, je seltener ein
  Wort ist — n = 2–3 kommt auf 0,941, n ≥ 301 nur auf 0,839. Das sieht nach
  „seltene Wörter sind besonders stabil" aus und ist das Gegenteil. Stellt man
  daneben, wie ähnlich ein Wort **beliebigen anderen** Wörtern ist
  (`hintergrund`, Mittel des Kosinus gegen 800 zufällige Wörter), löst es sich
  auf:

  | n je Hälfte | selbst | Hintergrund | **Kontrast** |
  |---|---|---|---|
  | 2–3 | 0,941 | 0,829 | **0,113** |
  | 6–10 | 0,912 | 0,788 | **0,123** |
  | 11–20 | 0,895 | 0,723 | **0,167** |
  | 21–50 | 0,881 | 0,639 | **0,238** |
  | 51–100 | 0,862 | 0,574 | **0,288** |
  | 101–300 | 0,843 | 0,528 | **0,315** |
  | 301+ | 0,839 | 0,477 | **0,360** |

  Seltene Wörter stimmen mit sich selbst überein, weil sie mit **allem**
  übereinstimmen: sie liegen in einem generischen Bereich, den SGNS
  im Wesentlichen aus dem Negative Sampling formt, nicht aus ihren eigenen
  wenigen Kontexten. Der **Kontrast (selbst − Hintergrund)** steigt dagegen
  monoton mit der Häufigkeit und ist damit das brauchbare Auflösungsmaß.
  Geprüft und ausgeschlossen wurde vorher eine andere Erklärung: zwei Modelle
  mit gleichem Seed starten am selben Punkt. Auf das Ergebnis wirkt sich das
  **nicht** aus — dieselbe Messung mit Seed 1 gegen Seed 2 unterscheidet sich
  um ±0,003.

  *Korrektur am Mechanismus, 2026-09-07.* Die frühere Fassung dieses Absatzes
  schrieb, gensim initialisiere „aus `hash(wort) + seed`". Das stimmt für
  gensim 4.4.0 nicht mehr: der Pfad `seeded_vector`/`hashfxn` ist dort
  `@deprecated` und wird von `init_weights` nicht aufgerufen. Tatsächlich läuft
  es über `resize_vectors(seed)` → `prep_vectors`, und das befüllt die
  Vektormatrix aus **einem** `np.random.default_rng(seed)` **zeilenweise, nach
  Vokabularindex**. Der Unterschied ist nicht kosmetisch:

  - Gleicher Seed gibt derselben **Zeile** denselben Start, nicht demselben
    **Wort**. Identisch ist der Start also nur, wo der Häufigkeitsrang
    übereinstimmt — oben im Vokabular fast immer, unten fast nie.
  - Weil `hash()` nicht beteiligt ist, ist die Reproduzierbarkeit **stärker**
    als hier bisher behauptet: sie überlebt Prozessgrenzen und braucht kein
    `PYTHONHASHSEED=0`. Mit `hashfxn=hash` hätte sie das gebraucht — das wäre
    ein stiller Reproduzierbarkeitsverlust über Sitzungen hinweg gewesen.

  Die Schlussfolgerung des Absatzes bleibt, weil sie an der Messung hängt
  (±0,003) und nicht an der Begründung. Nur die Begründung war falsch.

  *Das beantwortet nebenbei eine offene Frage des PPMI-Pfads.* Dessen Punkt 1
  („Nächste Schritte", Teilpunkt c) schlägt einen Split-Half-Rauschboden vor
  und lässt offen, wo die Mindest-Trefferzahl N liegt. Die Antwort ist: **gar
  kein N** — der Split-Half-Kosinus allein ist als Kriterium untauglich, weil
  er bei dünnen Zellen nach oben zeigt. Erst gegen den Hintergrund gerechnet
  wird er zur Auflösungsangabe, und dann braucht es keine Häufigkeitsschwelle
  mehr, weil der Kontrast das Wort direkt bewertet.

- **Eine Drift-Zahl gehört gegen das eigene Häufigkeitsband gelesen, nicht
  gegen den Gesamtmedian.** Die frühere Fassung dieses Punktes verglich
  `Bundeswehr` (cos 0,765) mit dem Median über alle 9.404 gemeinsamen Wörter
  (0,764) und schloss daraus, das Wort bewege sich durchschnittlich. Der
  Vergleich ist ungültig, weil der Gesamtmedian von unindividuierten Wörtern
  getragen wird, deren Kosinus aus dem vorigen Grund künstlich hoch ist. Die
  Drift 2016 → 2022 ist über die Bänder **u-förmig** — 0,789 (n 10–20), 0,730
  (n 101–300), 0,793 (n ≥ 1001) —, ein Gesamtmedian mittelt also über
  Populationen mit ganz verschiedenem Hintergrundniveau.

  Richtig gerechnet, gegen Wörter vergleichbarer Häufigkeit (Perzentil im
  eigenen Band, **niedrig = mehr Bewegung als seinesgleichen**):

  | Zielwort | n | cos | Bandmedian | Perzentil | Kontrast |
  |---|---|---|---|---|---|
  | `Rüstungsindustrie` | 16 | 0,660 | 0,782 | **9 %** | **0,099** |
  | `Drohne` | 45 | 0,676 | 0,756 | **17 %** | 0,244 |
  | `Bundeswehr` | 693 | 0,765 | 0,748 | 61 % | 0,480 |
  | `Wehrpflicht` | 10 | 0,805 | 0,789 | 61 % | 0,246 |
  | `Verteidigung` | 71 | 0,765 | 0,746 | 62 % | 0,419 |
  | `Rüstung` | 26 | 0,806 | 0,769 | 71 % | 0,398 |
  | `Sicherheit` | 1421 | 0,830 | 0,771 | 92 % | 0,514 |
  | `Frieden` | 463 | 0,819 | 0,737 | 95 % | 0,505 |

  Zwei Zahlen sind nötig, nicht eine: das **Perzentil** sagt, ob sich ein Wort
  auffällig bewegt hat, der **Kontrast**, ob man der Auskunft trauen darf.
  `Rüstungsindustrie` steht auf Perzentil 9 und sieht damit nach dem stärksten
  Beweger aus — bei Kontrast 0,099 ist die Zahl aber Rauschen. Das einzige
  Zielwort mit beidem ist `Drohne`. `Bundeswehr`, `Sicherheit` und `Frieden`
  haben sich zwischen 2016 und 2022 **weniger** bewegt als vergleichbar
  häufige Wörter.

  *Nicht überlesen:* das ist **ein** Jahrespaar, **ein** Seed, und 2022
  enthält als Scheibe die Wochen vor dem 27.02. mit. Es ist kein Befund über
  die Zeitenwende, sondern der Grund, Stufe 3 sauber zu rechnen. Konsequenz
  fürs Bundle bleibt: Drift als Perzentil im Häufigkeitsband ausweisen, mit
  Kontrast als Gütemaß daneben — nie als rohen Kosinus allein.

- **Schleifenvariablen in marimo brauchen einen Unterstrich.** Eine
  `for`-Variable auf oberster Zellenebene gilt marimo als Definition dieser
  Zelle — drei Zellen mit `for jahr in JAHRE:` sind deshalb ein
  `multiple-definitions`-Fehler, und das Notebook läuft gar nicht erst.
  `_jahr` macht sie zellenprivat und löst es. Funktionsparameter sind davon
  nicht betroffen (`def modell_pfad(jahr, …)` bleibt, wie es ist).
  Nützlich dabei: **`uv run marimo check pipeline-sgns.py`** findet das
  statisch, ohne das Notebook zu starten — bei Zellen, die zwanzig Minuten
  rechnen, ist das der Unterschied zwischen Sekunden und einem verlorenen
  Nachmittag.

- **Das Überspringen vorhandener Cache-Dateien prüft den Dateinamen, nicht die
  Parameter.** Alle drei teuren Schleifen (Lemmatisierung, Vollmodelle,
  Hälften) sind wiederaufnehmbar gebaut: existiert die Zieldatei, wird
  übersprungen. Ändert man danach einen Trainingsparameter — `epochs`,
  `window`, `vector_size` —, läuft die Zelle zwar erneut an, findet die alten
  Dateien und überspringt sie. Man rechnet ab da mit Modellen der alten
  Parameter weiter, **ohne Fehlermeldung**. `min_count` ist zufällig gedeckt,
  weil volle und halbe Modelle verschiedene Dateinamen haben; die übrigen
  Parameter sind es nicht. Regel: **bei jeder Änderung an `trainiere_modell`
  die `cache/wv_*.kv` löschen.**

- **Ein Cache-Seiteneffekt braucht eine Datenflusskante, sonst kennt marimo die
  Reihenfolge nicht.** Zwei Zellen schreiben Dateien, andere lesen sie wieder
  ein — und zwischen beiden bestand lange *keine* Abhängigkeit, weil der
  Dateipfad nur als String-Literal in beiden stand. marimo ordnet Zellen
  ausschließlich nach Datenfluss; wo keine Kante ist, entscheidet der
  Tiebreak in `dataflow.topological_sort`, und der ist die
  Registrierungsreihenfolge — also die Position in der Datei. Das ist eine
  Heuristik, kein Vertrag.

  Auf warmem Cache fällt das nie auf. Es bricht, sobald man `cache/wv_*_h1.kv`
  löscht und die Ladezelle einzeln neu laufen lässt: marimo führt die
  Trainingszelle nicht nach, weil es die Abhängigkeit gar nicht kennt, und man
  bekommt einen `FileNotFoundError` an einer Stelle, die „schon immer lief".
  Genauso beim Umsortieren von Zellen im Editor.

  Behoben am 2026-09-07 auf zwei Wegen, beide ohne Trick:
  - `lemma_pfad(jahr)` wird in der Lemmatisierungszelle **definiert** und von
    allen Lesern benutzt. Wer den Pfad über die Funktion holt, bekommt die
    Kante geschenkt — und die vier Kopien desselben f-Strings verschwinden mit.
  - Die beiden Trainingsschleifen geben eine Quittung zurück
    (`vollmodelle_bereit`, `haelften_bereit`), die Ladezelle fordert sie an und
    prüft sie per `assert`. Die Werte werden nicht gebraucht; sie sind die
    Kante. Vorher hing die Ladezelle nur zufällig richtig, weil sie `JAHRE` und
    `modell_pfad` braucht, die in der Vollmodell-Zelle definiert sind — zöge
    man die beiden in eine eigene Zelle (ein völlig naheliegender Refactor),
    wäre die Kante weg.

- **`score` und `generik` sind über die Scheiben hinweg nicht streng
  vergleichbar.** Die Kandidatenmenge der Nachbarsuche ist das Vokabular der
  jeweiligen Scheibe mit `count ≥ 10`, und das schwankt: 9.952 Wörter (2017)
  bis 14.256 (2023), Faktor 1,43. In einem größeren Pool findet die Top-8 im
  Schnitt etwas nähere Nachbarn, und das Zeilenmittel (`generik`) mittelt über
  eine anders zusammengesetzte Menge.

  Gemessen gegen einen festen Pool (6.124 Wörter, in allen zwölf Scheiben mit
  `count ≥ 10`) trägt die Pool-Zusammensetzung +0,012 bis +0,027 zu `generik`
  bei — gegen eine echte Jahresspanne von 0,311 bis 0,383 bei `Bundeswehr`.
  Die Rangfolge überlebt also (2022 ist so oder so das aufgelösteste Jahr, 2017
  und 2021 die generischsten), und der absolute Schwellenwert 0,55 bleibt
  benutzbar. Aber: **dünne Jahre werden doppelt bestraft** — schlechtere
  Vektoren *und* ein kleinerer Pool zeigen in dieselbe Richtung. Wer `generik`
  je Jahr auf die zweite Nachkommastelle liest, liest zu genau.

- **Alle zwölf Scheiben teilen `seed=1` — geprüft, es leckt nicht.** Da
  `prep_vectors` die Startmatrix zeilenweise nach Vokabularindex füllt, haben
  alle zwölf Modelle *dieselbe* Startmatrix, und die Häufigkeitsränge sind über
  die Jahre stark korreliert. Der Verdacht liegt nahe, dass das die
  Selbstähnlichkeit über die Scheiben künstlich anhebt und damit jede
  Drift-Zahl schönt. Der 5-Seed-Lauf kann das nicht sehen: er benutzt innerhalb
  eines Laufs überall denselben Seed, der Effekt steckt in allen fünf gleich.

  Direkt gemessen (2014 zusätzlich mit Seed 2 trainiert, Drift 2014 → 2022
  einmal gegen Seed 1 und einmal gegen Seed 2, 19.557 gemeinsame Wörter):

  | Häufigkeitsband | gleicher Seed | verschiedene Seeds | Δ |
  |---|---|---|---|
  | Median gesamt | 0,8296 | 0,8296 | +0,0001 |
  | n 3–9 | 0,8692 | 0,8687 | +0,0005 |
  | n 10–50 | 0,8055 | 0,8058 | −0,0003 |
  | n 51–300 | 0,7654 | 0,7658 | −0,0005 |
  | n 301+ | 0,7771 | 0,7795 | −0,0024 |

  **Kein systematischer Effekt.** Fünf Epochen reichen offenbar, um die
  winzige Initialisierung (`(rng.random − 0,5)/100`, also ±0,005) auszuwaschen.

  Was dabei nebenbei abfällt, ist die wichtigere Zahl: **pro Wort** bewegt
  derselbe Seed-Wechsel einer *einzigen* Scheibe `Sicherheit` um 0,055 und
  `Bundeswehr` um 0,033. Das ist die Größenordnung des gesuchten Effekts
  (21→22: `Bundeswehr` 0,728 gegen Hintergrund 0,793). Ein Einzelwort-Kosinus
  aus einem Seed ist damit endgültig keine Zahl, die man ohne Boden daneben
  hinschreiben darf — was dieses Dokument an mehreren Stellen behauptet, ist
  hier direkt beziffert.

*(weitere folgen)*

### Was das SGNS-Bundle aushält

Gemessen am ausgelieferten `bundestag_zeitenwende_sgns.json`, Stand 2026-09-07.
Der PPMI-Pfad wurde über Schwellen, Grenzen und Ausschlüsse optimiert; hier gibt
es dafür wenig Spielraum — die Grenzen sind gemessen, nicht gewählt. Diese Liste
ist der Ersatz: was man aus diesem Bundle ablesen darf und was nicht.

*Das Bundle wurde seither erneut gebaut* (`neighbors_k` 8 → 150, größerer
`vocab_points`-Pool, s. Ablauf-Schritt 14 oben). `kontrast`/`generik` selbst
werden davon nicht berührt — beide sind Eigenschaften der Aggregat-Zelle, nicht
der Nachbarliste —, die Punkte 1, 2, 4, 5, 6 unten sollten also weiter gelten.
Punkt 3 (Nachbarlisten-Stabilität) ist explizit bei `Top-8` gemessen und in
seiner eigenen Fußnote als Stand-2026-09-07-Messung markiert.

**1. Vier von elf Zielwörtern sind auswertbar, nicht elf.**

| kontrast-Band | Zellen | Anteil |
|---|---|---|
| belastbar (≥ 0,35) | 34 | 26 % |
| brauchbar (0,29–0,35) | 9 | 7 % |
| grenzwertig (0,20–0,29) | 17 | 13 % |
| **Rauschen (< 0,20)** | **64** | **49 %** |
| keine Messung (`null`) | 6 | 5 % |

Belastbare Scheiben je Wort: `Bundeswehr` 12/12, `Frieden` 12/12, `Sicherheit`
11/12, `Verteidigung` 7/12, `Drohne` 1/12 — **alle übrigen 0/12**. Das ist kein
Schwellenproblem: der PPMI-Pfad kann mit `Hindukusch` arbeiten, weil er nichts
schätzt und die SVD der dünnen Zeile Struktur aus den übrigen 501 leiht. SGNS hat
diese Anleihe nicht. Dieselbe Zielwortliste, die dort trug, trägt hier zu zwei
Dritteln nicht.

**2. Die 3D-Ansicht zeigt das Rauschen als Bewegung — und das ist die
gefährlichste Eigenschaft dieses Bundles.**

| Wort | kontrast Ø | 3D-Weglänge (Anteil der Bilddiagonale) |
|---|---|---|
| Frieden | 0,40 | 43 % |
| **Bundeswehr** | **0,40** | **70 %** |
| Sicherheit | 0,34 | 44 % |
| Verteidigung | 0,30 | 121 % |
| Drohne | 0,19 | 184 % |
| **Rüstung** | **0,18** | **193 %** |
| Wehrpflicht | 0,11 | 128 % |

**Korrelation kontrast ↔ Weglänge: r = −0,65.** Ungegatet rasen genau die Wörter
quer durchs Bild, deren Vektor nichts aussagt, während der eigentliche Befund
fast stillsteht. Das Frontend dämpft deshalb seit 2026-09-07 Punkte, Labels und
Linien über `viz/resolution.ts` — ohne Messung im Bundle passiert nichts, ältere
Bundles sehen unverändert aus.

**3. Die Nachbarlisten sind zu einem Drittel bis zur Hälfte Seed.**
An den *ausgelieferten* Listen gemessen (2014/2019/2022 zusätzlich mit Seed 2
trainiert, Top-8 verglichen): Überlappung **0,68** bei kontrast ≥ 0,29, **0,52**
darunter, 0,58 insgesamt. Von acht Zeilen wären also zwei bis vier andere.
Zusätzlich haben **35 der 130 Zellen `generik` > 0,55**, sind also gar nicht
interpretierbar — konzentriert auf `Friedenssicherung` (9 Scheiben),
`Wehrpflicht` (8), `Hindukusch` (6). Das Nachbarpanel warnt dort jetzt.

**4. Die Driftkurve im Frontend ist nicht die Driftkurve dieses Dokuments.**
Punkt 15 oben entscheidet „paarweise direkt ausrichten" — das Bundle *kann* das
nicht liefern, es enthält nur die auf 2019 gedrehten Vektoren. Die Abweichung ist
systematisch und wächst mit dem Abstand zur Referenz:

| Abstand des Übergangs von 2019 | mittleres Δ (Frontend − direkt) | max \|Δ\| |
|---|---|---|
| 0 Jahre | ±0,0000 | 0,000 |
| 1 Jahr | −0,0038 | 0,019 |
| 3 Jahre | −0,0114 | 0,079 |
| 5 Jahre | −0,0118 | 0,038 |

Negativ heißt: **das Frontend zeigt mehr Drift, als direkt gemessen da ist**, und
umso mehr, je weiter der Übergang von 2019 entfernt liegt — die Kurve hat eine
eingebaute U-Form. Der Zäsurbefund überlebt das (21→22 verschiebt sich bei
`Bundeswehr` um −0,005, bei `Verteidigung` um −0,029). Aber `Bundeswehr` 22→23
steht im Frontend bei 0,770 statt 0,849 (**Δ −0,079**) — die Aussage „nach 2022
beruhigt es sich wieder" wäre reines Artefakt.

**5. Die Zäsur liegt *in* der Scheibe, nicht zwischen den Scheiben.**
**927 der 8.196 Reden aus 2022 (11 %) fallen vor den 24.02.2022.** Jahresscheiben
sind erzwungen, ein Quartalsraster wie im PPMI-Pfad gibt es hier nicht. Der
21→22-Sprung ist damit eher unter- als überschätzt.

**6. Anker sind zeitlos, aber über verschiedene Zeitfenster gemittelt.**
45 Anker decken alle zwölf Scheiben ab, **17 nur ≤ 4** (`Taurus`,
`Kriegstüchtigkeit`, `F-35`, `Sonderschuld`, `Ukrainekrieg` …). Geprüft, ob das
späte Trajektorienpunkte künstlich an späte Anker zieht: **tut es nicht** — späte
Punkte liegen sogar etwas weiter von den Spät-Ankern (−0,072) als frühe, und
derselbe Effekt zeigt sich bei den Voll-Ankern (−0,058), ist also kein
differenzieller Artefakt. Bleibt die triviale Lesart: „`Bundeswehr` 2014 liegt
weit von `Taurus`" ist keine Erkenntnis.

**7. Geprüft und unproblematisch.**
- **`freq` ist zwischen beiden Bundles direkt vergleichbar**: Median-Verhältnis
  SGNS/PPMI **1,00** über die 130 gemeinsamen Zellen, Spanne 0,71–1,26. Die
  Konfidenz-Kodierung liest sich in beiden gleich.
- **Der Echo-Effekt ist wirklich klein**: 551 der 689 Nachbarwörter kommen in
  genau einer der 130 Listen vor — der Verzicht auf die Kontrast-Auswahl war
  richtig. Reste gibt es: `Stabilität` 21×, `äußerer`/`innerer` je 12×, die
  beiden letzten als Folge des fehlenden POS-Filters bei den Nachbarn.
  **Gemessen bei `nb_k = 8`, vor dem Regler-Umbau vom 2026-09-11** — bei den
  jetzt ausgelieferten 150 Nachbarn je Zelle nicht nachgemessen; mit deutlich
  mehr Einträgen je Liste ist eine höhere Wiederholungsquote plausibel.
- **Die eingefrorene Wortwolke ist vertretbar**: die Punkte streuen im Mittel nur
  0,10 um ihren eigenen Schwerpunkt = 6 % der Bilddiagonale; die lange Weglänge
  (86 %) ist Zittern, keine Drift. Der ernstere Punkt bleibt die erklärte
  Varianz — **Stand damals (2026-09-07) 20,4 % statt 36,7 %; nach dem
  Vokabular-Umbau (7.207 `vocab_points` statt 5.052, s.
  `.claude/memory/vocab-radius-feature.md`) liefert das ausgelieferte Bundle
  jetzt `explained_variance_ratio_3d_vocab = 0,4474` gegen weiterhin
  `explained_variance_ratio_3d = 0,3668`** — die Out-of-Sample-Wolke bildet
  sich damit sogar *besser* ab als der In-Sample-Raum, nicht schlechter. Nicht
  neu bewertet, ob das die „eingefrorene Wortwolke ist vertretbar"-Einschätzung
  ändert.

**Kurzfassung.** Das Bundle ist rechnerisch sauber und liefert vier auswertbare
Wörter. Drei der Darstellungen (3D-Szene, Nachbarpanel, Driftkurve) zeigen ohne
Gating ausgerechnet die schwächsten Zellen am auffälligsten. Die Zahlen zum
Gaten liegen als `kontrast`, `generik` und `freq` in jeder Zelle.

## Offen / später zu klären

Sofern nicht anders vermerkt, beziehen sich die Punkte auf den **PPMI-Pfad**.
Die beiden Punkte zur Suchwortliste (`Sicherheit`, Überschneidung der 11
Zielwörter) gelten für **beide** Verfahren, weil beide dieselbe Liste
benutzen — sie sind Eigenschaften der Wortauswahl, nicht des Embeddings.
Die offenen Punkte des SGNS-Pfads stehen in dessen eigenem Abschnitt darüber.

- **PMI-Deckeneffekt bei exklusiven Kontextwörtern**: Kommt ein Kontextwort
  ausschließlich im Fenster eines einzigen Zielworts vor (nie bei den anderen
  11 Suchwörtern in dieser Zeitscheibe), erreicht sein PPMI unabhängig von der
  absoluten Häufigkeit denselben Maximalwert `log2(gesamt/randsumme_ziel)`
  (mathematisch: `PMI <= -log2 P(zielwort)`, Gleichheit bei
  `P(kontext|zielwort) = 1`). Verifiziert an "Hindukusch"/2022-Q1 (nur 1
  Zielwort-Treffer, mehrere Kontextwörter mit identischem PPMI trotz
  unterschiedlicher `beobachtet`-Werte). **Für die SVD entschärft** (Stand
  2026-09-01): Hapax-Spitzen können keine Achse tragen, weil eine Achse ein von
  mehreren Zeilen geteiltes Muster braucht — verifiziert an
  `Bundeswehr | 2016-Q1`, dessen stärkste PPMI-Kontextwörter („gelassen",
  „trennen", „2008", „worin") im SVD-Vektor keine Rolle mehr spielen und dessen
  dominante Richtung stattdessen „ausgeben/Milliarde/Verteidigung/Rüstung" ist.
  **Für die Anzeige-Ranglisten bleibt er bestehen**; dagegen helfen die Spalten
  `local_mi`/`t_score` aus Ablauf-Schritt 14. Die beiden Mindesthäufigkeit-
  Filter können ihn strukturell *nicht* lösen: sie filtern auf Häufigkeit, der
  Deckeneffekt ist aber eine Frage der Exklusivität — schlimmer noch,
  `randsumme_kontext >= X` zwingt überlebende exklusive Wörter auf
  `beobachtet >= X`. Verworfene bzw. offene Alternativen: PMI² (getestet,
  funktionswortdominiert → verworfen), Kontext-Distributions-Smoothing
  (Levy & Goldberg 2015, α=0,75; hebt die Decke nicht auf), strengerer
  `zielwort_haeufigkeit`-Mindestwert.
  **Gemessen 2026-10-01:** Context Distribution Smoothing und additive
  Glättung (add-λ) verworfen (Methodology 2.5, 2.6). Gegen den Deckeneffekt
  bleibt es bei `local_mi`/`t_score` für die Anzeige.
- ~~Stoppwort-Filterung der Kontextwörter~~ — **entschieden 2026-09-01**:
  POS-basierter Filter aktiv (`DET, ADP, PRON, AUX, CCONJ, SCONJ`), s.
  Stolperstein. Er deckt wie vermutet nur einen Teil des Deckeneffekts ab und
  behebt dessen Ursache bei sehr seltenen Zielwörtern nicht — bleibt also
  nebenwirkungsfrei nützlich, ist aber keine Lösung des Deckeneffekts.
  **Seit 2026-10-01 nicht mehr in der Matrix**, s. Stolperstein POS-Filter
  (Methodology 2.4).
- ~~Satz- vs. Fenster-basierter Kontext für Embeddings~~ — **erledigt sich
  pfadweise, 2026-09-04.** Im PPMI-Pfad ist der Kontext das ±40-Wörter-Fenster
  aus der KWIC-Extraktion, und das ist im DNB-Szenario keine Wahl, sondern die
  Extraktionsgrenze. Im SGNS-Pfad ist die **Rede** die Dokumentgrenze und das
  Fenster gensims eigenes (`window=5`) — eine Satzsegmentierung würde den
  Parser/`senter` kosten, der in beiden Pfaden bewusst abgeschaltet ist, und
  brächte gegenüber der Redegrenze nichts, was das Trainingsfenster nicht
  selbst regelt.
- `redner_rolle_kurz` als möglicher Filter, `tokens` für Korpus-Balance-QC —
  vorgeschlagen, nicht umgesetzt.
- `rede_zitate`/`rede_kommentare` als Rezeptions-/Sentiment-Signal — bewusst
  zurückgestellt.
- Abbildung des Endergebnisses auf das Bundle-Schema (`data/test/SCHEMA.md`
  im Hauptrepo), mit Fraktion als Facette — ausgearbeitet unter „Nächste
  Schritte", inklusive Feld-für-Feld-Stand.
- **Die 11 Suchwörter messen teilweise dasselbe.** Über alle 502 Zeilen ist nur
  in 344 Fällen (68,5 %) der nächste Nachbar dasselbe Zielwort; je Wort reicht
  die Spanne von `Sicherheit` 50/50 und `Bundeswehr` 47/50 bis
  `Landesverteidigung` 17/40, `Rüstungsindustrie` 18/47 und `Verteidigung`
  19/50. Die Abweichler sind durchweg plausibel (`Verteidigung|2017-Q3` →
  `Bundeswehr|2017-Q3`, cos 0,865), also kein Pipeline-Fehler, sondern echte
  Vokabularüberschneidung: diese Wörter kommen in denselben Reden vor und
  teilen fast ihr gesamtes Kontextfenster. Folge für die Visualisierung:
  mehrere Punkte werden nahezu deckungsgleich wandern. Zu entscheiden, ob die
  Suchwortliste entzerrt oder das im Frontend nur erklärt wird.
- **`Sicherheit` ist polysem und fängt den halben Bundestagsdiskurs ein.** Das
  Gegenstück zum vorigen Punkt: dort messen mehrere Suchwörter dasselbe, hier
  misst *ein* Suchwort zu viel Verschiedenes. In der Kontrast-Rangliste
  (Ablauf-Schritt 21) steht `Sicherheit` bei über der Hälfte der Top-200 als
  `bestes_zielwort` — zusammen mit `Mietpreisbremse`, `BAföG`, `Rentenniveau`,
  `Kurzarbeitergeld`, `Kinderpornografie`, `Endlager`, `Straßenverkehr`,
  `Patientin`, `Anleger`, `Verbraucherin`. Das ist soziale, innere,
  Verbraucher- und Rechtssicherheit, nicht Verteidigungspolitik. Mit 8.870
  Vorkommen ist `Sicherheit` das häufigste Suchwort und zieht damit einen
  erheblichen Teil des Raums in einen ganz anderen Diskurs. Zu entscheiden:
  Suchwort schärfen (`Sicherheitspolitik`), ersetzen, oder drin lassen und den
  Befund ausstellen. Für die Anker ist vorerst Letzteres vorgesehen — ein
  eigener Cluster für den nichtmilitärischen Sicherheitsbegriff, damit die
  3D-Ansicht *zeigt* statt versteckt, dass `Sicherheit` weit weg von
  `Bundeswehr` sitzt.
- **`Hindukusch` ist ein Zitatmarker, kein Diskursbegriff.** Die drei
  kontraststärksten Kandidaten überhaupt sind `Struck` (0,740), `Peter` (0,713)
  und `verteidigen` (0,693) — alle drei mit bester Zelle `Hindukusch`, alle
  drei Bestandteile von Peter Strucks Satz „Deutschlands Sicherheit wird auch
  am Hindukusch verteidigt", den die SVD als Einheit wiedererkannt hat. Schönes
  Nebenergebnis (niemand hat der Pipeline von diesem Zitat erzählt), aber es
  heißt auch: `Hindukusch` misst die Zitierhäufigkeit eines Satzes, nicht die
  Entwicklung eines Begriffs. Passt zu den Kennzahlen — es ist mit 37 belegten
  Zeitscheiben das dünnste Zielwort.
- ~~`k = 100` ist übernommen, nicht begründet~~ — **erledigt 2026-11-17:**
  k = 300, p = 0,5, bestimmt mit dem Diagnose-Notebook von emb
  (`pipeline_emb_ppmi_diagnose.py`, Abschnitt 3); Befunde und Begründung in
  Methodology 1.8. Ursprünglicher Eintrag: Das Singulärwertspektrum ist
  flach (Rang 1: 319,6 → Rang 100: 72,2 → Rang 400: 42,4), k=100 hält nur
  40,9 % der Frobenius-Masse, und die Nachbarschaftsprofile korrelieren mit
  denen des vollen Raums nur zu 0,69. Das ist nicht per se schlecht —
  Rauschentfernung ist ja der Zweck —, heißt aber, dass k empirisch bestimmt
  werden muss statt aus dem Spektrum abgelesen. Zweite Stellschraube derselben
  Zeile: der Eigenwert-Exponent in `U[:, :k] * S[:k]**p`, aktuell `p = 1`;
  Levy, Goldberg & Dagan (2015) fanden für Wortähnlichkeit oft `p = 0,5`
  besser. Bei diesen Daten ändert `p` die *Rangfolge* der Nachbarn kaum, wohl
  aber das Niveau der Kosinuswerte (p=0: 0,64 · p=0,5: 0,75 · p=1: 0,86 für
  denselben Nachbarn) — für absolute Drift-Vergleiche über die Zeit also
  relevant. Beides zusammen als Paar kalibrieren.
