# Datenmodell — was das Tool liest und was es damit rechnet

Dieses Dokument beschreibt das **Datenformat** eines Bundles und für **jede
Visualisierung**, welche Felder sie liest und wie sie daraus ihre Werte rechnet.
Referenz ist durchgehend der synthetische Zeitenwende-Datensatz
[`data/test/zeitenwende_demo.json`](data/test/zeitenwende_demo.json).

Verwandte Dokumente:
[`data/test/SCHEMA.md`](data/test/SCHEMA.md) ist der knappe Vertrag für
Datenlieferanten, [`specs.md`](specs.md) begründet die Designentscheidungen,
[`README.md`](README.md) beschreibt die Bedienung. Dieses Dokument ist die
ausführliche Fassung dazwischen: **Feld für Feld, Modul für Modul.**

---

## 1. Das Format auf einen Blick

Ein Bundle ist **eine einzelne JSON-Datei** (UTF-8) mit acht Top-Level-Blöcken,
plus einem optionalen neunten (`neighbor_coords`, seit 2026-09-11, bis 2026-10-15 `vocab_points`, s. §2.6). Es
gibt kein Nachladen, keine Datenbank, keine mehrteiligen Dateien — die App
lädt die Datei einmal, validiert sie und rendert alles daraus.

```
zeitenwende_demo.json  ~1,94 MB
├── meta            Beschreibung, Vektordimension, Projektion, Zäsur
├── facets_schema   deklarierte Facetten-Achsen (hier: publication)
├── entities        die 6 Zielwörter
├── time_slices     die 64 Quartale 2010 Q1 … 2025 Q4
├── anchors         52 Kontextwörter mit festem Ort im Raum
├── aggregates      384 Zellen  = 6 Wörter × 64 Quartale
├── records         2 304 Zellen = 6 Wörter × 64 Quartale × 6 Publikationen
└── neighbors       6 × 64 Listen à 8 Nachbarn
```

Der Referenz-Datensatz oben ist **älter als `neighbor_coords`, `kontrast` und
`generik`** (s. §2.1, §2.6, §2.7) und enthält sie deshalb nicht — er wurde vor
diesen Feldern generiert und nicht nachgezogen. Die drei Felder werden
trotzdem hier dokumentiert, weil sie zum aktuellen Vertrag gehören und die
echten Bundestags-Bundles (`backend/bundles/bundestag_zeitenwende_*.json`)
sie ausliefern; Beispiele stammen von dort bzw. aus `data/test/SCHEMA.md`.

Geprüft wird beim Laden mit einem zod-Schema
([`frontend/src/types/schema.ts`](frontend/src/types/schema.ts)), und zwar nach
zwei Regeln:

- **Permissiv nach außen.** Unbekannte Zusatzfelder werden durchgereicht, nicht
  verworfen (`.passthrough()`). Im Zeitenwende-Bundle sind das z.B.
  `meta.explained_variance_ratio_3d` oder `ausrichtung` an einem Facettenwert.
  Neue Felder erfordern also **keine** Codeänderung.
- **Streng nach innen.** Was der Renderer wirklich braucht — `coords_3d` als
  Tripel, IDs, `time_slices.index` — wird geprüft, und Verstöße werden als
  lesbare Fehlermeldung angezeigt statt still zu scheitern.

**Pflicht ist nur `time_slices` (mindestens eine Scheibe).** Alle anderen Blöcke
haben Defaults. Damit überhaupt etwas zu sehen ist, braucht es zusätzlich
`entities` und `aggregates`; alles Weitere schaltet einzelne Fähigkeiten frei
(→ [§7 Degradation](#7-was-passiert-wenn-etwas-fehlt)).

---

## 2. Die Blöcke im Einzelnen

### 2.1 `meta`

```json
{
  "name": "DNB-Zeitenwende (synthetisch)",
  "description": "Synthetischer Demo-Datensatz im generischen Tool-Schema. …",
  "vector_dim": 48,
  "projection": "joint PCA (3 Komponenten) über alle Vektoren",
  "shift_point": "2022Q1",
  "explained_variance_ratio_3d": 0.8177,
  "warnhinweis_sentiment": "sentiment ist synthetischer Platzhalter …"
}
```

| Feld | Typ | Pflicht | Wirkung im Tool |
|---|---|---|---|
| `name` | string | nein (Default „Unbenanntes Bundle") | Kopfzeile, Brotkrümel, Bundle-Übersicht |
| `description` | string | nein | Bundle-Übersicht |
| `vector_dim` | int > 0 | nein | **nur Anzeige** — die tatsächliche Länge wird nie festgenagelt |
| `projection` | string | nein | Anzeige (Konfigurator-Kopf, Statuszeile, Übersicht) |
| `shift_point` | string \| null | nein | **ID einer Zeitscheibe.** Wird zur Zäsur-Hilfslinie in Zeitachse, Matrix, Kurven und Fluss |
| `explained_variance_ratio_3d` | number 0–1 | nein | Projektions-Hinweiskasten im Wortraum: erklärte Varianz der 3D-PCA, **in-sample** (gefittet auf Anker + Aggregate) |
| `explained_variance_ratio_3d_vocab` | number 0–1 | nein | dieselbe Kennzahl, aber **out-of-sample** — gilt für die nur hineinprojizierten `neighbor_coords`. Fehlt sie, zeigt der Hinweiskasten nur die erste Zahl, obwohl gerade die Vokabelwolke schlechter abgebildet ist (SGNS-Bundle: 20 % gegen 37 %) |
| `explained_frobenius_ratio_k` | number 0–1 | nein | derselbe Hinweiskasten, Stufe 1 der Reduktion (Rohmatrix → k Dimensionen, unzentrierte SVD) — andere Rechnung als die beiden Varianz-Felder oben, keine PCA |

`shift_point` muss eine `time_slices[].id` sein; sonst wird er ignoriert (kein
Fehler). Im generischen Demo-Bundle ist er `null` — dann fehlen die Hilfslinien
ersatzlos, alles andere funktioniert.

### 2.2 `facets_schema`

Deklariert die Facetten-Achsen. **Hier steckt die Wiederverwendbarkeit**: ein
anderes Projekt tauscht nur diesen Block.

```json
[{"id": "publication", "label": "Publikation", "type": "categorical",
  "values": [
    {"id": "SZ",             "label": "Süddeutsche Zeitung",   "ausrichtung": "mitte-links"},
    {"id": "FAZ",            "label": "Frankfurter Allgemeine","ausrichtung": "liberal-konservativ"},
    {"id": "taz",            "label": "die tageszeitung",      "ausrichtung": "links"},
    {"id": "WELT",           "label": "Die Welt",              "ausrichtung": "konservativ"},
    {"id": "Junge_Freiheit", "label": "Junge Freiheit",        "ausrichtung": "rechts"},
    {"id": "Handelsblatt",   "label": "Handelsblatt",          "ausrichtung": "wirtschaftsliberal"}]}]
```

Mehrere Facetten gleichzeitig sind erlaubt (`facets: {publication: "SZ",
ressort: "Politik"}`). Die **erste** Facette im Array ist die *Primärfacette*:
sie trägt die Einfärbung im Facettiert-Modus. `ausrichtung` ist ein
Zusatzfeld — durchgereicht, aber vom Tool nicht ausgewertet.

### 2.3 `entities` — die Zielwörter

```json
{"id": "drohne", "label": "Drohne", "category": "Bedeutungswandel",
 "note": "Sense shift zivil/Technik -> militärisch; graduell"}
```

| Feld | Pflicht | Wirkung |
|---|---|---|
| `id` | **ja** | Schlüssel für `aggregates`, `records`, `neighbors` |
| `label` | nein (Fallback: `id`) | überall in der Oberfläche |
| `category` | nein | Spalte in der Bundle-Übersicht |
| `note` | nein | Tooltip an der Wortauswahl |

Die **Reihenfolge in `entities` bestimmt die Farbe**: Wort *i* bekommt Slot *i*
der validierten 12er-Palette. Umsortieren färbt also um; neue Wörter hinten
anhängen lässt die bestehenden Farben stabil.

Zeitenwende-Bundle: `bundeswehr`, `ruestung`, `wehrpflicht`, `drohne`,
`zeitenwende`, `sondervermoegen` — je einem Wandel-Typ zugeordnet
(Konnotations-, Bedeutungs-, Häufigkeitswandel).

### 2.4 `time_slices` — die Zeitachse

```json
{"id": "2022Q1", "label": "2022 Q1", "start": "2022-01-01",
 "year": 2022, "quarter": 1, "index": 48}
```

| Feld | Pflicht | Wirkung |
|---|---|---|
| `id` | **ja** | Referenz aus Zellen und `neighbors` |
| `index` | **ja** (int) | **bestimmt allein die Reihenfolge** |
| `label` | nein (Fallback: `id`) | jede Achsen- und Zeitbeschriftung |
| `start`, `year`, `quarter` | nein | **werden derzeit nicht ausgewertet** |

Wichtig: Das Tool sortiert **ausschließlich nach `index`** und behandelt die
Scheiben danach als gleichabständig. Es liest weder `start` noch `year` — eine
unregelmäßige Zeitachse (etwa Dekaden gemischt mit Jahren) würde also
gleichmäßig gezeichnet. Das ist bewusst so: die x-Achse ist eine
**Scheiben-Achse**, kein Kalender.

Zeitenwende-Bundle: 64 Quartale, `index` 0…63, `2022Q1` = Index 48.

### 2.5 `anchors` — Landmarken und NN-Kandidaten

```json
{"label": "Panne", "cluster": "SKANDAL",
 "vector": [ … 48 floats … ], "coords_3d": [0.007, 0.804, -0.340]}
```

52 Ankerwörter in 6 Clustern (`SKANDAL`, `WEHR`, `KRIEG`, `RUEST`, `TECH`,
`POLITIK`, je 8–10 Wörter). Anker sind **feste Punkte** — sie haben keine
Zeitachse und bewegen sich nicht.

Sie leisten zweierlei: (a) beschriftete Landmarken im 3D-Raum, an denen die
Wanderung eines Zielworts ablesbar wird, und (b) die Auflösung von
NN-Labels — steht ein Nachbar als Anker im Bundle, kann das Tool eine Linie zu
ihm ziehen. Zuordnung über `label`, exakt und ersatzweise klein geschrieben.

`anchors[].vector` wird vom Frontend derzeit **nicht** gelesen (nur
`coords_3d`); es liegt für die geplante serverseitige Re-Projektion bei.

### 2.6 `neighbor_coords` — Koordinaten-Nachschlagewerk für Nachbarn (optional)

```json
{"label": "einsatzbereit", "coords_3d": [0.412, -0.118, 0.803]}
```

Nur zwei Felder, kein `cluster`, kein `vector`. Eingeführt am 2026-09-11 als
Ersatz für die vorherige Radius-Wortumgebung (die im Bild statt im Vektorraum
gefiltert hatte, s. `.claude/memory/vocab-radius-feature.md`).

**Zweck:** Ein Nachbar aus `neighbors` wird im 3D-Bild nur gezeichnet, wenn
seine Position bekannt ist. Anker bringen ihre Position selbst mit (§2.5);
`neighbor_coords` liefert sie für alle **übrigen** Nachbar-Labels nach. Das Feld
ist **kein eigener Suchraum** — welche Wörter als Nachbarn überhaupt in Frage
kommen, entscheidet ausschließlich der Lieferant im vollen Vektorraum (§2.8);
`neighbor_coords` beantwortet nur „wo zeichne ich das hin".

**Herkunft:** Out-of-Sample-Projektion in denselben Raum wie `coords_3d`
(`(normiert − pca_mittelwert) @ pca_achsen.T`) — **keine neue PCA**, sonst
würden sich alle bestehenden Anker-/Aggregat-Koordinaten mitverschieben. Weil
die Punkte nicht mitgefittet wurden, bildet sie dieselbe PCA schlechter ab als
die Trainingspunkte; dafür gibt es `meta.explained_variance_ratio_3d_vocab`
(§2.1).

Zwei Bedingungen, die ein Lieferant einhalten muss:

- **Abdeckung.** Jedes `neighbors[].label`, das kein `anchors[].label` ist,
  braucht hier einen Punkt — aus demselben Kandidatenpool wie die
  Nachbarauswahl, insbesondere **ohne** zusätzlichen Wortart-Filter. Ein
  strengerer Filter als bei den Nachbarn selbst reißt Löcher ins Bild.
- **Eindeutigkeit.** Ein Label darf nicht gleichzeitig als Anker-Anzeigename
  und als `neighbor_coords`-Label auftauchen — das Frontend löst Nachbarn über
  den reinen Text auf (`pointByLabel` in §3), bei einer Kollision gewinnt der
  Anker.

Fehlt der Block (ältere Bundles, das synthetische Demo-Bundle), werden nur die
Nachbarn im Bild gezeichnet, die zugleich Anker sind — der Rest bleibt auf die
Textliste beschränkt. Kein Fehler, nur weniger sichtbar.

**Bei den echten Bundestags-Bundles:** ~5.000–8.000 Punkte, abhängig von
Zeitscheiben-Raster und Pfad (PPMI vs. SGNS) — Details und Messzahlen zum
09-11-Umbau in `.claude/memory/vocab-radius-feature.md` und
`pipeline/README.md`.

### 2.7 `aggregates` und `records` — die Zellen

Beide haben dieselbe Form. Der Unterschied ist einzig `facets`:

```json
// aggregates: über alle Facetten zusammengefasst
{"entity_id": "bundeswehr", "time_slice_id": "2010Q1", "facets": {},
 "vector": [ … 48 floats … ], "coords_3d": [-0.2582, 0.4925, -0.2955],
 "freq": 526, "sentiment": -0.478}

// records: facettenaufgelöst
{"entity_id": "bundeswehr", "time_slice_id": "2010Q1", "facets": {"publication": "SZ"},
 "vector": [ … 48 floats … ], "coords_3d": [-0.2639, 0.4981, -0.2865],
 "freq": 103, "sentiment": -0.475}
```

| Feld | Pflicht | Bedeutung / Wirkung |
|---|---|---|
| `entity_id` | **ja** | muss in `entities` vorkommen, sonst wird die Zelle ignoriert |
| `time_slice_id` | **ja** | muss in `time_slices` vorkommen, sonst wird die Zelle ignoriert |
| `facets` | nein (Default `{}`) | `{}` = Aggregat; sonst Facettenwert je Achse |
| `coords_3d` | **ja** | genau drei Zahlen — die vorprojizierte Position |
| `vector` | nein | voller Vektor im gemeinsamen Raum; **Grundlage aller Ähnlichkeitsrechnung** |
| `freq` | nein (Default **1**) | Stichprobengröße → Konfidenz; **`0` heißt „nicht belegt"** |
| `sentiment` | nein | optionaler Skalar; **wird derzeit von keiner Ansicht gezeigt** |
| `kontrast` | nein | optionales Gütemaß: **sagt der Vektor überhaupt etwas über sein Wort aus?** Steuert im Wortraum das Auflösungs-Gating (§5.2) |
| `generik` | nein | optionales Gütemaß: **ist die Nachbarliste dieser Zelle interpretierbar?** Ebenfalls Auflösungs-Gating (§5.2) |

Zahlen im Zeitenwende-Bundle: 384 Aggregate, 2 304 Records, `vector_dim` 48,
alle Vektoren **L2-normiert** (Norm 1,0) und auf 5 Nachkommastellen gerundet;
`freq` 0…3 301 (Aggregate) bzw. 0…848 (Records); `sentiment` −0,563…0,571.
`kontrast`/`generik` fehlen im Zeitenwende-Bundle ganz (kein Gating,
s. §5.2) — sie kommen bislang nur im SGNS-Bundle vor.

**Die wichtigste Eigenheit:** 43 Aggregat-Zellen haben `freq: 0` — alle beim
Wort *Zeitenwende*, das vor 2020 Q4 im Korpus nicht vorkommt. Diese Zellen
enthalten trotzdem einen `vector` und `coords_3d`. Das sind **Platzhalter ohne
Aussage**, und jede Ansicht muss sie aussparen (→ [§4](#4-die-vier-invarianten)).

`kontrast` und `generik` sind **nicht ineinander umrechenbar** und auch nicht
zu `freq` komplementär — eine Zelle kann viele Belege haben und trotzdem einen
nichtssagenden Vektor. `null` heißt „sollte gemessen werden, ging nicht";
fehlt das Feld ganz, heißt das „dieser Lieferant misst das nicht". Beide
Fälle bleiben **neutral**, nie „schlecht" — sonst sähen PPMI-, Demo- und
Fremdbundles ohne diese Felder plötzlich anders aus.

### 2.8 `neighbors` — vorberechnete nächste Nachbarn

```json
{"bundeswehr": {"2022Q3": [
  {"label": "Wehrpflicht",        "score": 0.9185, "group": "ZIELWORT"},
  {"label": "Zeitenwende",        "score": 0.8495, "group": "ZIELWORT"},
  {"label": "einsatzbereit",      "score": 0.8182, "group": "WEHR"},
  {"label": "verteidigungsfähig", "score": 0.8181, "group": "WEHR"},
  … insgesamt 8 …]}}
```

Struktur: `neighbors[entity_id][time_slice_id] = Neighbor[]`. Die
**Array-Reihenfolge ist der Rang** — das Tool sortiert nicht nach `score` um
(→ [§4](#4-die-vier-invarianten)). `group` ist frei wählbar; deckt sie sich mit
einem `anchors[].cluster`, bekommt sie dieselbe Farbe wie dieser Cluster.

Im Zeitenwende-Bundle sind die Nachbarn gegen **Anker + die übrigen Zielwörter**
gerechnet; Letztere tragen die Gruppe `ZIELWORT`, die kein Anker-Cluster ist,
und die Liste ist **8** Einträge lang.

**Listenlänge k legt der Lieferant fest** und ist zugleich die Obergrenze
eines Reglers im Frontend (`NeighborKControl`, §5.2): dieselbe Liste speist
Textanzeige und beschriftete 3D-Punkte, der Regler schneidet nur ab. Die
echten Bundestags-Bundles liefern seit 2026-09-11 **k = 150** statt 8 — kurze
Listen reichen für die Textanzeige, für das Bild braucht es mehr, weil dort
erst sichtbar wird, wie weit echte Nachbarn in der Projektion auseinander-
liegen. `idx.maxNeighborK` (§3) liest die tatsächliche Länge aus dem Bundle,
statt sie zu raten — ältere Bundles mit 8 Einträgen funktionieren unverändert.

> **Neu 2026-10-06:** Liefert ein Bundle einen Suchraum (`meta.neighbor_space`,
> s. `SCHEMA.md`), rechnet das Backend die Nachbarn bei jeder Filteränderung
> neu (k = 1000), mit den Filtern der Seite „Nachbar-Filter“. Das Ergebnis
> ersetzt die Listen in allen Ansichten; die mitgelieferten `neighbors` sind
> dann der Rückfall ohne Backend und entsprechen genau den Voreinstellungen.

---

## 3. Wie das Tool die Datei aufbereitet

Zwischen Datei und Visualisierung liegt genau ein Schritt: `indexBundle()` in
[`frontend/src/data/model.ts`](frontend/src/data/model.ts). Er läuft **einmal
beim Laden** und baut die Struktur, mit der danach alle Module arbeiten.

```
  Datei (JSON)
      │  validateBundle()      zod, permissiv nach außen / streng nach innen
      ▼
   Bundle
      │  indexBundle()
      ▼
 IndexedBundle
      ├── sortedSlices          nach index sortiert, + sliceIndexById
      ├── shiftSliceIndex       shift_point → Position auf der Achse (oder null)
      ├── tracks[]              je Entity: samples[] an der Zeitachse ausgerichtet
      │                         + segments[] (Läufe belegter Positionen)
      ├── facetTracks           je Entity: eine Spur je Facetten-Kombination
      ├── anchors, anchorByLabel, clusters
      ├── neighborCoords, pointByLabel, maxNeighborK   NN-Bildauflösung (§2.6, §2.8)
      ├── center, scale         gemeinsames Framing der 3D-Szene
      └── aggFreqMax, recordFreqMax   Bezugsgrößen der Konfidenz
```

Sieben Details, die das Verhalten erklären:

1. **Samples liegen an der Zeitachse, nicht in den Daten.** `samples` hat immer
   `sliceCount` Einträge; wo keine Zelle existiert, steht `undefined`. Fehlende
   Zellen und `freq: 0` sind damit gleich behandelbar.
2. **`freq ?? 1`** — fehlt das Feld, gilt die Zelle als voll belegt. Ein Bundle
   ohne Frequenzen verliert die Konfidenz-Kodierung, aber nichts sonst.
3. **Segmente brechen an Lücken.** `segments` enthält nur zusammenhängende Läufe
   mit `freq > 0` und mindestens zwei Punkten — daher zeichnet der Wortraum über
   eine Belegungslücke hinweg **keine** Linie.
4. **`vector` wird als Referenz übernommen**, nicht kopiert. Die
   Ähnlichkeitsrechnung greift also ohne Speicher-Verdopplung auf die
   Bundle-Arrays zu.
5. **Framing über alle Punkte** (Anker + Aggregate + Records) — **`neighbor_coords`
   zählt bewusst nicht mit.** Mittelpunkt und Skalierungsfaktor auf Zielradius 6
   würden sonst von Ausreißern in der unkuratierten Nachbar-Wolke verzerrt.
   Deshalb sitzt jeder Datensatz beim Öffnen passend im Bild, egal in welcher
   Größenordnung die Koordinaten liegen.
6. **Unauflösbare Referenzen werden still übersprungen** — eine Zelle mit
   unbekannter `time_slice_id` oder `entity_id` fällt heraus, ohne den Rest zu
   gefährden.
7. **`pointByLabel` löst Nachbar-Labels auf Koordinaten auf, Anker zuerst.**
   Anker und `neighbor_coords` werden über denselben Text-Schlüssel (`label`,
   exakt und ersatzweise klein geschrieben) in eine gemeinsame Map gelegt;
   steht ein Label an beiden Stellen, gewinnt der Anker (§2.6). `maxNeighborK`
   ist die längste `neighbors`-Liste im Bundle — die Obergrenze des Reglers
   aus §5.2, nicht geraten, sondern aus den Daten gelesen.

---

## 4. Die vier Invarianten

Diese vier Regeln gelten in **allen** Ansichten. Wer eigene Daten liefert, muss
vor allem die ersten beiden bedienen.

**(1) Ein Raum, kein Alignment.** Alle Vektoren — Anker, Aggregate, Records,
alle Zeitscheiben — stammen aus **einer** gemeinsamen Projektion. Nur deshalb
ist Bewegung zwischen zwei Zeitpunkten als Bedeutungswandel lesbar und nicht als
Artefakt zweier Koordinatensysteme. Wird pro Zeitscheibe oder pro Facette
separat projiziert, ist jede Zahl in diesem Tool wertlos. (Das Alignment-Problem,
das Kiyama et al. mit inkrementellem Training lösen, stellt sich im
Bundle-Schema deshalb gar nicht erst.)

> **Revidiert 2026-10-05:** Die PCA wird nur noch auf den Aggregaten gefittet;
> Anker und alle übrigen Kontextwörter werden mit denselben Achsen projiziert,
> ohne neuen Fit. Mit Ankern im Fit hingen die Achsen davon ab, wie viele Anker
> ein Lauf trifft. Die Invariante „ein Raum“ bleibt erhalten (Methodology 1.15,
> 1.16, 2.12).

**(2) `freq == 0` heißt „kein Beleg" und ist kein kleiner Wert.** Durchgesetzt
wird das an genau einer Stelle — `entityVectors()` in
[`frontend/src/viz/similarity.ts`](frontend/src/viz/similarity.ts) liefert für
solche Scheiben `null`. Die Folgen sind überall sichtbar und beabsichtigt:

| Ansicht | Verhalten bei `freq == 0` |
|---|---|
| Wortraum | Trajektorie setzt aus, Punkt verschwindet |
| Matrix | Zeile **und** Spalte bleiben schraffiert (Textur, keine Farbe) |
| Drift / Wort-gegen-Wort | `null` statt 0 → die Kurve bricht ab |
| Nachbarschafts-Fluss | Zeitscheibe wird schraffiert und ausgeblendet — **auch dann, wenn das Bundle dort Nachbarn mitliefert** |

**(3) Ähnlichkeit rechnet auf `vector`, nie auf `coords_3d`.** Die 3D-Projektion
ist verlustbehaftet (hier ~82 % erklärte Varianz); Kosinuswerte daraus wären
systematisch verzerrt. Fehlen volle Vektoren, fällt die Rechnung auf `coords_3d`
zurück — und die Oberfläche schreibt das über jeder betroffenen Grafik an.

**(4) Reihenfolge kommt aus den Daten, nicht aus Sortierung.**
`time_slices[].index` bestimmt die Zeitachse, die Position in `entities` die
Farbe, die Position in `neighbors[…][…]` den Rang. Das Tool sortiert nichts
davon nach Werten um — sonst behauptete es eine Ordnung, die der Lieferant nicht
gemeint hat.

---

## 5. Was jede Visualisierung liest und rechnet

Kurzübersicht, danach je ein Abschnitt pro Modul.

| Modul | Liest | Kernrechnung |
|---|---|---|
| Datenquelle | `meta`, `entities`, `time_slices`, `facets_schema`, `freq` | Abzählen, Belegungsquote |
| Wortraum (3D) | `coords_3d`, `freq`, `anchors`, `neighbor_coords`, `neighbors`, `records`, `kontrast`/`generik` | Interpolation zwischen Scheiben |
| Ähnlichkeitsmatrizen | `vector`, `freq` | `M[i][j] = cos(v_ti, v_tj)` |
| Drift-Kurven | `vector`, `freq` | `d(t) = 1 − cos(v_ref, v_t)` |
| Wort gegen Wort | `vector`, `freq` | `s(t) = cos(v_a^t, v_b^t)` |
| Nachbarschafts-Fluss | `neighbors`, `freq`, `anchors` | Rang über Zeit + Mengenvergleich |

### 5.1 Datenquelle

**Liest:** `meta.*`, `entities[]`, `time_slices[]`, `facets_schema[]`,
`anchors.length`, Zellenzahlen, und je Wort dessen `freq`-Werte.

**Rechnung:** Belegung je Zielwort = Anzahl Zeitscheiben mit `freq > 0`, als
Balken gegen `sliceCount`. Zusätzlich wird geprüft, ob irgendeine Zelle einen
nicht-leeren `vector` hat — daraus ergibt sich die Anzeige „Ähnlichkeit rechnet
auf: `vector` (voll)" oder der Rückfall-Hinweis.

**Zeitenwende-Bundle:** 5 von 6 Wörtern 64/64; *Zeitenwende* 21/64.

### 5.2 Wortraum (3D)

**Liest:** `tracks[].samples[].coords` und `.freq`, `tracks[].segments`,
`anchors[].coords_3d` + `.cluster`, `neighbor_coords`, `neighbors`, `kontrast` und
`generik` je Zelle, sowie im Facetten-Modus `facetTracks` (aus `records`) und
die aktiven Filter. **Nicht** `vector`.

**Rechnung:**

- *Position bei kontinuierlicher Zeit*: der Slider steht auf einem reellen `t`
  zwischen zwei Scheiben, die Position wird linear interpoliert:
  `coords = lerp(coords[i], coords[i+1], frac)` — analog `freq` und `sentiment`.
- *Weltkoordinaten*: `(coords − center) × scale` (gemeinsames Framing).
- *Konfidenz*: Punktgröße und Deckkraft ∝ `freq / aggFreqMax` (im
  Facetten-Layer gegen `recordFreqMax`, weil Record-Frequenzen viel kleiner
  sind — sonst wären alle facettierten Punkte blass).
- *NN-Overlay*: Nachbarn der **gerundeten** Scheibe (NN sind pro diskreter
  Scheibe vorberechnet), **gemeinsam gekappt vom Nachbar-Regler** (`NeighborKControl`,
  §2.8) — Standard 8, bis zu `idx.maxNeighborK`. Text und Bild teilen sich
  dieselbe Liste; gezeichnet wird ein Punkt nur, wenn `pointForLabel()` ihn
  über `anchors` **oder** `neighbor_coords` auflöst (§2.6, §3), eine Linie nur zu
  Ankern. Im Bild sind zusätzlich maximal 40 Nachbarn gerendert
  (`MAX_RENDERED`), auch bei größerem k — die Liste zeigt trotzdem alle. Bei
  `freq == 0` des aktiven Worts wird gar nichts gezeigt.
- *Auflösungs-Gating*: liefert eine Zelle `kontrast` und/oder `generik`
  (bislang nur das SGNS-Bundle), dämpft `frontend/src/viz/resolution.ts`
  Punktgröße, Label und Linien unterhalb fester Schwellen
  (`KONTRAST_SCHRANKE = 0,29`, `GENERIK_SCHRANKE = 0,55`) — **nie ausblenden,
  nur dämpfen.** Grund: bei trainierten (statt konstruierten) Vektoren zeigt
  die 3D-Weglänge ohne dieses Gating ausgerechnet die unaufgelösten Wörter am
  auffälligsten (SGNS-Bundle: Korrelation Kontrast↔Weglänge r = −0,65, s.
  `pipeline/README.md`). `undefined` **und** `null` gelten beide als „kein
  Gating" — Bundles ohne die Felder sehen unverändert aus.
- *Hinweiskarten unter dem Nachbarn-Panel*: `VolumeNote` zeigt `freq` der
  aktuellen Scheibe gegen den Median des Worts (Warnung bei starker
  Unterbelegung); `ProjectionNote` erklärt die zweistufige Reduktion SVD → PCA
  anhand `meta.explained_frobenius_ratio_k` und `meta.explained_variance_ratio_3d`
  (beide optional in der Anzeige).
- *Facettiert / Streuung*: eine Spur bzw. Punktwolke je Facetten-Kombination aus
  `records`, gefiltert per UND-Verknüpfung über die Facetten-Achsen.

### 5.3 Ähnlichkeitsmatrizen (Kiyama)

**Liest:** `vector` und `freq` der **Aggregat**-Zellen, `sliceCount`,
`time_slices[].label`, `shift_point`.

**Rechnung:** für jedes ausgewählte Wort eine T×T-Matrix

```
M[i][j] = cos(v_ti, v_tj)      falls Scheibe i und j belegt
M[i][j] = NaN                  sonst  →  Schraffur
```

Die Matrix ist symmetrisch, also wird nur die obere Hälfte gerechnet; die
Diagonale ist per Definition 1.

Zwei Dinge, die die Aussagekraft tragen:

- Die **Farbdomäne** spannt über die tatsächlichen Werte **ohne die Diagonale**
  (Demo: ~0,34…1,0). Auf `[-1, 1]` gespreizt wäre jede Matrix einfarbig; mit
  Diagonale gestaucht ebenso. Umschaltbar gemeinsam ↔ je Matrix.
- `cos()` liefert bei Nullvektor oder Dimensionskonflikt bewusst `NaN`, nie 0 —
  ein stiller 0-Wert wäre als „unähnlich" fehlinterpretierbar.

**Zeitenwende-Bundle:** *Bundeswehr* und *Sondervermögen* zeigen zwei scharfe
Blöcke mit Grenze auf der Zäsur, *Drohne* einen Gradienten von der Diagonale weg
(stetige Drift), *Zeitenwende* eine fast vollständig schraffierte Fläche.

### 5.4 Drift-Kurven (Hamilton)

**Liest:** dieselben Vektoren wie die Matrix.

**Rechnung:** `d(t) = 1 − cos(v_ref, v_t)` — geometrisch **eine Zeile der
Matrix, an der Achse gespiegelt**. Welche Zeile, bestimmt der Bezugspunkt:

| Modus | `v_ref` | Eigenschaft |
|---|---|---|
| Erstbeleg | erste belegte Scheibe **dieses Worts** | fair für spät auftauchende Wörter; Startpunkte liegen verschieden |
| Zeitpunkt | eine feste Scheibe für alle Wörter | streng vergleichbar; Wörter ohne Beleg dort können nicht gezeigt werden |
| Vorgänger | die Scheibe `t−1` | Änderungs**rate**: Spitzen zeigen *wann* sich etwas bewegt |

Regeln: Im Vorgänger-Modus wird **strikt** die Nachbarscheibe verglichen, nie
„die letzte belegte" — sonst überspannte ein einzelner Wert eine beliebig lange
Lücke und meinte je nach Stelle etwas anderes. Ein unbelegter fester Bezugspunkt
führt nicht zu einer Ersatzrechnung, sondern zu einem Hinweis mit Wortnamen.
Die y-Achse ist bei 0 verankert, weil Distanz 0 („identisch mit dem
Bezugspunkt") ein echter Nullpunkt ist.

**Zeitenwende-Bundle:** *Drohne* steigt ab ~2014 stetig auf ~0,66; alle anderen
bleiben flach und machen an der Zäsur eine Stufe. Im Vorgänger-Modus liegt der
Ausschlag auf 2021 Q4 / 2022 Q1.

### 5.5 Wort gegen Wort

**Liest:** die Vektoren **zweier** Wörter je Zeitscheibe.

**Rechnung:** `s(t) = cos(v_a^t, v_b^t)` — Kreuz-, nicht Selbstähnlichkeit. Ein
Wert fehlt, sobald **eines** der beiden Wörter zu dieser Zeit unbelegt ist.

Gezeigt wird **ein Bezugswort gegen alle anderen** (n−1 Kurven). Alle Paare
wären k·(k−1)/2 Linien — bei 6 Wörtern 15, die keine Palette mehr trennt. Die
Achse misst hier Nähe statt Distanz, weil „steigt = rücken zusammen" die
natürliche Lesart ist; sie spannt um die Daten, denn bei Kosinus-Ähnlichkeit ist
0 kein sinnvoller Anker.

**Zeitenwende-Bundle:** gegen *Bundeswehr* steigt *Wehrpflicht* von 0,696 auf
0,971 und *Zeitenwende* von 0,467 auf 0,856, während *Rüstung* von 0,803 auf
0,638 fällt — Konvergenz und Divergenz im selben Bild.

### 5.6 Nachbarschafts-Fluss

**Liest:** `neighbors[entity][slice]` (Reihenfolge, `label`, `score`, `group`),
die `freq` der Aggregat-Zellen als Gate, `anchors` nur zur Prüfung, ob ein
Nachbar eine Landmarke hat, und `clusters` für die Farbreihenfolge.

**Rechnung:**

- *Rang* = Position in der Liste + 1. **Kein Nachsortieren nach `score`** — bei
  fehlendem oder anders normiertem Score behauptete das stillschweigend eine
  andere Reihenfolge.
- *Läufe*: Vorkommen eines Nachbarn werden in zusammenhängende Abschnitte
  zerlegt. Fällt er aus den Top-k und kommt zurück, sind das zwei Segmente.
- *Umschlag* je Scheibe: `|A_t \ A_{t−1}| / |A_t|`, also der Anteil der Top-k,
  der gegenüber der Vorscheibe neu ist — `null`, wo eine der beiden Listen
  fehlt, damit kein Wert eine Lücke überspannt.
- *Farbe* = Gruppe, nicht Wort. Slot-Reihenfolge: Anker-Cluster zuerst (damit
  eine Gruppe denselben Farbton hat wie ihre Anker im 3D-Raum), danach
  Zusatzgruppen wie `ZIELWORT` alphabetisch. Bei ~20 Nachbarn trennt keine
  Palette 20 Serien — die Wortidentität trägt die Direktbeschriftung.

**Zeitenwende-Bundle:** *Bundeswehr* hat 20 verschiedene Nachbarn über 64
Quartale; der stärkste Umschlag liegt mit **63 % genau auf 2022 Q1**, wo das
SKANDAL-Umfeld (Sparzwang, marode, Beschaffungschaos) vom WEHR-Umfeld (Bündnis,
einsatzbereit, Abschreckung) abgelöst wird.

---

## 6. Feldnutzungs-Matrix

Welches Feld wird von welchem Modul gelesen? (● = zentral, ○ = am Rand)

| Feld | Daten­quelle | Wortraum | Matrizen | Drift | Paare | Fluss |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| `meta.name` / `.description` | ● | | | | | |
| `meta.vector_dim` / `.projection` | ● | ○ | | | | |
| `meta.shift_point` | ● | ● | ● | ● | ● | ● |
| `entities[].id` / `.label` | ● | ● | ● | ● | ● | ● |
| `entities[].category` / `.note` | ● | ○ | | | | |
| `time_slices[].index` | ● | ● | ● | ● | ● | ● |
| `time_slices[].label` | ● | ● | ● | ● | ● | ● |
| `facets_schema` | ● | ● | | | | |
| `anchors[].coords_3d` / `.label` / `.cluster` | ○ | ● | | | | ○ |
| `anchors[].vector` | | | | | | |
| `neighbor_coords[].coords_3d` / `.label` | | ● | | | | |
| `aggregates[].coords_3d` | | ● | | | | |
| `aggregates[].vector` | ○ | | ● | ● | ● | |
| `aggregates[].freq` | ● | ● | ● | ● | ● | ● |
| `aggregates[].kontrast` / `.generik` | | ● | | | | |
| `records[].*` | ○ | ● | | | | |
| `neighbors` | | ● | | | | ● |
| `meta.explained_variance_ratio_3d` / `_3d_vocab` / `explained_frobenius_ratio_k` | | ○ | | | | |
| `sentiment` | | | | | | |

Zwei Felder werden aktuell von **keiner** Ansicht gezeigt: `sentiment` (wird
geladen und interpoliert, aber nicht dargestellt — die Achsen-Projektionsmethode
ist methodisch offen, siehe `specs.md` §6) und `anchors[].vector` (liegt für die
geplante serverseitige Re-Projektion bereit). `neighbor_coords` und
`kontrast`/`generik` sind neuer als der Zeitenwende-Referenzdatensatz (§1) und
tauchen deshalb in dessen Beispielen nirgends auf.

---

## 7. Was passiert, wenn etwas fehlt

Das Tool ist so gebaut, dass ein unvollständiges Bundle Fähigkeiten verliert,
aber nicht abstürzt.

| fehlt | Folge |
|---|---|
| `vector` (alle Zellen) | Ähnlichkeit rechnet auf `coords_3d`; Warnhinweis über Matrix, Drift und Paaren |
| `freq` | jede Zelle gilt als belegt (Default 1); keine Konfidenz-Kodierung, keine Maskierung |
| `neighbors` | kein NN-Overlay/-Panel; das Fluss-Modul zeigt einen Hinweis statt eines Diagramms |
| `anchors` | keine Landmarken; NN-Labels bleiben unaufgelöst (grau, ohne Linie) |
| `neighbor_coords` | im Bild werden nur Nachbarn gezeichnet, die zugleich Anker sind; die Textliste bleibt vollständig |
| `kontrast` / `generik` | kein Auflösungs-Gating — die 3D-Szene sieht aus wie ohne diese Felder, nichts wird gedämpft |
| `records` / `facets_schema` | Facettiert- und Streuungs-Modus sind leer; Aggregiert funktioniert |
| `shift_point` | keine Zäsur-Hilfslinien (so im generischen Demo-Bundle) |
| `label` (Entity/Scheibe/Facette) | die `id` wird angezeigt |
| einzelne Zellen | die Zeitscheibe ist an dieser Stelle unbelegt — behandelt wie `freq == 0` |

**Minimalbundle**, das etwas Sichtbares ergibt:

```json
{"time_slices": [{"id": "t1", "index": 0}, {"id": "t2", "index": 1}],
 "entities":    [{"id": "w1"}],
 "aggregates":  [{"entity_id": "w1", "time_slice_id": "t1", "coords_3d": [0,0,0]},
                 {"entity_id": "w1", "time_slice_id": "t2", "coords_3d": [1,0,0]}]}
```

Das rendert den Wortraum. Für Matrix, Drift und Paare braucht es zusätzlich
`vector` je Zelle, für den Fluss den `neighbors`-Block.

---

## 8. Wie die synthetischen Zeitenwende-Daten entstehen

Erzeugt von [`data/test/generate_dataset.py`](data/test/generate_dataset.py)
(deterministisch über einen festen Seed). Der Weg ist bewusst
**modellhaft-konstruiert**, damit jede Ansicht einen bekannten Befund hat, gegen
den man sie prüfen kann:

1. **Cluster-Schwerpunkte.** Je Cluster ein zufälliger Einheitsvektor in 48
   Dimensionen, gegen die bereits vergebenen leicht dekorreliert.
2. **Anker** = Schwerpunkt + 12 % Rauschen, renormiert. Die Ankerlabels selbst
   sind **reale Dornseiff-Lemmata** der einschlägigen Sachgruppen.
3. **Zielwort-Trajektorien** als designte *Cluster-Mischungen* über der Zeit,
   gesteuert von logistischen Kurven um die Zäsur (2022 Q1). *Bundeswehr* etwa
   fährt den SKANDAL-Anteil herunter und den WEHR-Anteil hoch; *Drohne* mischt
   TECH → KRIEG mit Zentrum ~2016, also **unabhängig von der Zäsur**;
   *Zeitenwende* ist vor 2022 praktisch inaktiv.
4. **Publikationen** verschieben denselben Verlauf: `dt` verschiebt den
   Wendepunkt (WELT und Junge Freiheit früher, taz später), `amp` skaliert die
   Bewegung, `sent` verschiebt das Sentiment. Daraus entstehen die `records`.
5. **Frequenzen** als Poisson-Ziehung um eine wortspezifische Basiskurve mal
   Publikationsgröße — daher die realistisch dünnen und leeren Frühzellen.
6. **Aggregate** = frequenzgewichteter Mittelwert der Publikationsvektoren,
   renormiert; `freq` = Summe, `sentiment` = gewichtetes Mittel.
7. **Gemeinsame PCA** (3 Komponenten) über **alle** Vektoren — Anker,
   Aggregate, Records zusammen. Ergebnis: `coords_3d` und
   `explained_variance_ratio_3d` = 0,8177.
8. **Nachbarn** als Top-8 nach Kosinus gegen Anker + die übrigen Zielwörter, je
   Wort und Scheibe auf Aggregat-Ebene.

**Was diese Daten nicht sind:** Die Vektoren sind synthetische Stand-ins — weder
trainierte Embeddings noch die später vorgesehene PPMI-SVD-Konstruktion. Sie
bilden nur die Raum-Eigenschaften nach, an denen die Visualisierung hängt.
`sentiment` ist ein reiner Platzhalter für die noch offene
Achsen-Projektionsmethode. Die Befunde („Bundeswehr wandert an der Zäsur ins
Wehrhaftigkeits-Umfeld") sind also **eingebaut, nicht gemessen** — sie taugen
zur Prüfung des Werkzeugs, nicht als Aussage über Sprache.

---

## 9. Checkliste für eigene Datensätze

1. **Eine gemeinsame Projektion** über alle Vektoren — Anker, Aggregate,
   facettenaufgelöste Zellen, alle Zeitscheiben. Niemals pro Scheibe oder Facette
   getrennt.
2. **`vector` mitliefern**, wenn Matrix, Drift und Wort-gegen-Wort belastbar sein
   sollen. Die Länge muss innerhalb des Bundles gleich sein, ist aber sonst frei.
3. **`freq` ehrlich setzen**, insbesondere `0` für „im Korpus nicht belegt". Das
   ist der einzige Weg, dem Tool Nichtwissen mitzuteilen.
4. **`time_slices[].index` lückenlos und in Reihenfolge** vergeben; das Tool
   zeichnet die Scheiben gleichabständig.
5. **IDs müssen auflösen**: `entity_id` → `entities[].id`, `time_slice_id` →
   `time_slices[].id`, `shift_point` → eine Scheiben-ID, NN-`label` →
   `anchors[].label` (sonst bleibt der Nachbar Text ohne Landmarke).
6. **Facetten deklarieren**, bevor sie in Zellen auftauchen — die erste Facette
   im `facets_schema` trägt die Einfärbung.
7. Bei feiner Facettierung die **Sparsity im Blick behalten**: Wort × Quartal ×
   Publikation × Ressort kann auf wenige Sätze je Zelle fallen. Das Tool macht
   das über `freq` sichtbar, statt es zu kaschieren — aber es kann es nicht
   heilen.
8. **Lange `neighbors`-Listen (k > Anker-Anzahl) brauchen `neighbor_coords`**,
   sonst bleiben die überzähligen Nachbarn im Bild unsichtbar (Textliste zeigt
   sie trotzdem). Derselbe Kandidatenpool wie bei der Nachbarauswahl, kein
   zusätzlicher Wortart-Filter, keine Label-Kollision mit `anchors` (§2.6).
9. **`kontrast`/`generik` nur liefern, wenn tatsächlich gemessen wurde** — beide
   Felder gehen direkt in ein Dämpfungs-Gating der 3D-Ansicht ein (§5.2). Ein
   fehlendes Feld ist neutral, ein geratener Wert wäre falsches Vertrauen.
