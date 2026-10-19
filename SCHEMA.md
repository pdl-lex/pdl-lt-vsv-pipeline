> **Kopie.** Original: `pdl-lt-vsv/docs/SCHEMA.md` (VSV-App), Stand 2026-10-15,
> `meta.neighbor_space` Formatversion 1. Änderungen nur im Original vornehmen
> und dann hierher kopieren.

# Vektor-Bundle-Schema (v0.1)

Der Vertrag zwischen Datenproduktion (eure Pipeline / jeder andere Lieferant)
und dem Visualisierungstool. Das Tool kennt **keine** Domäne – es kennt nur
Entities, Zeitscheiben, Facetten und Vektoren. Jedes Projekt liefert ein
Bundle in dieser Form; das Frontend rendert es identisch.

Zwei Beispiel-Bundles liegen bei:
- `zeitenwende_demo.json` – realistischer Fall, Facette `publication`
- `generic_demo.json` – fachfremd, Facette `korpus` (belegt Domänenunabhängigkeit)

## Top-Level

| Feld | Typ | Zweck |
|---|---|---|
| `meta` | object | Name, Beschreibung, optional `source` (Herkunft: Korpus, Zitation, Lizenz — im Explorer als „Quelle" angezeigt), `vector_dim`, optional `eig_p` (PPMI-SVD: Gewichtung der Zeilenvektoren U·S^p), optional `method` (Kurzname des Verfahrens, z.B. „PPMI-SVD“; Statuszeile), Projektionsverfahren, `shift_point`, erklärte Varianz (s.u.), optional `neighbor_space` (Suchraum für Nachbarabfragen, s.u.) |
| `facets_schema` | array | Deklariert die Facetten-Achsen (Publikation, Region, Sprecher …) |
| `entities` | array | Die Zielwörter (das, was sich über die Zeit bewegt) |
| `time_slices` | array | Zeitscheiben in Reihenfolge (`index`) |
| `anchors` | array | Kontext-/Landmarkenwörter: feste Punkte im Raum für NN + Orientierung |
| `neighbor_coords` | array | Optional. Koordinaten-Tabelle (Label + `coords_3d`) für Nachbarn, die keine `anchors` sind — damit das Frontend sie im 3D-Bild zeichnen kann |
| `records` | array | Facettenaufgelöste Zellen (Wort × Scheibe × Facettenwert) |
| `aggregates` | array | Über Facetten aggregierte Zellen (`facets: {}`) |
| `neighbors` | object | Vorberechnete Nearest-Neighbour-Listen je Wort/Scheibe |

### Erklärte Varianz in `meta` – zwei Zahlen, nicht eine

- `explained_variance_ratio_3d` – **in-sample**: gilt für die Punkte, auf denen
  die PCA gefittet wurde (im PPMI-Pfad seit 2026-10-05 nur die Aggregate;
  vorher Aggregate + Anker).
- `explained_variance_ratio_3d_vocab` – optional, **out-of-sample**: dieselben
  drei Achsen, angewandt auf die `neighbor_coords`. Die wurden nur hineinprojiziert,
  nicht mitgefittet, und werden deshalb schlechter abgebildet — im SGNS-Bundle
  **20 % gegen 37 %**. Da genau diese Wolke der Radius-Regler zeigt, wäre der
  In-Sample-Wert dort fast doppelt zu optimistisch. Fehlt das Feld, zeigt das
  Frontend nur die erste Zahl.

## `facets_schema` – der Kern der Wiederverwendbarkeit

Facetten sind **nicht** hart codiert. Jede Facette ist eine benannte Achse mit
kategorialen Werten. Ein zweites Projekt tauscht nur diesen Block aus:

```json
{"id": "publication", "label": "Publikation", "type": "categorical",
 "values": [{"id": "SZ", "label": "Süddeutsche Zeitung", "ausrichtung": "mitte-links"}, ...]}
```

Mehrere Facetten gleichzeitig sind erlaubt (`facets: {publication: "SZ", ressort: "Politik"}`).
Zusätzliche Facetten = weitere Keys, kein Code- oder Schemaänderung nötig. Grenze
ist nicht die Architektur, sondern die Zell-Sparsity (siehe unten).

## `records` / `aggregates`

```json
{
  "entity_id": "bundeswehr",
  "time_slice_id": "2022Q3",
  "facets": {"publication": "SZ"},   // {} bei aggregates
  "vector": [ ... vector_dim floats ... ],
  "coords_3d": [x, y, z],            // vorprojiziert (gemeinsame PCA)
  "freq": 313,                       // Stichprobengröße der Zelle -> Konfidenz
  "sentiment": 0.14,                 // optionaler Skalar-Overlay
  "kontrast": 0.47,                  // optional: sagt der Vektor etwas aus?
  "generik": 0.31                    // optional: ist die NN-Liste lesbar?
}
```

- `vector`: der volle Vektor im **gemeinsamen** Raum. Ermöglicht Re-Projektion
  auf gefilterte Wortauswahlen im Frontend/Backend.
- `coords_3d`: vom Lieferanten vorberechnete gemeinsame PCA – **ein** Basisraum
  für alle Punkte (Aggregate, Records, Anker, übrige Kontextwörter).
  Alignment-frei, weil ein einziger Raum. Gefittet wird auf einer festen
  Teilmenge (im PPMI-Pfad: die Aggregate), alles andere wird mit denselben
  Achsen hineinprojiziert, ohne neuen Fit. So hängen die Achsen nicht davon ab,
  welche Anker ein Lauf findet. Für schnelles Erst-Rendering ohne
  Server-Roundtrip.
- `freq`: pro Zelle. Frontend sollte Punktgröße/Deckkraft ∝ freq koppeln, damit
  dünn besetzte Zellen (kleine n) nicht optisch überinterpretiert werden.
- `sentiment`: optional; im Zeitenwende-Bundle ein **synthetischer Platzhalter**
  für die noch offene Achsen-Projektionsmethode ("ins Positive").
- `kontrast` / `generik`: optionale **Gütemaße** des Lieferanten. Sie beantworten
  zwei verschiedene Fragen, die `freq` beide nicht beantwortet — eine Zelle kann
  viele Belege haben und trotzdem einen nichtssagenden Vektor:
  - `kontrast` — **sagt der Vektor etwas über sein Wort aus?** Reproduzierbarkeit
    über zwei Datenhälften, abzüglich der Ähnlichkeit zu zufälligen anderen
    Wörtern. Hoch = individuiert, niedrig = im generischen Bereich.
  - `generik` — **ist die Nachbarliste interpretierbar?** Wie ähnlich der Vektor
    dem Vokabular im Allgemeinen ist. Hoch = alles ist ungefähr gleich nah, "am
    nächsten" zeichnet nichts mehr aus.

  **Nicht ineinander umrechenbar** — insbesondere ist `kontrast + generik` *nicht*
  der Selbst-Kosinus; die beiden stammen typischerweise aus verschiedenen
  Modellen und Vergleichsmengen. `null` heißt "sollte gemessen werden, ging
  nicht"; das Feld ganz wegzulassen heißt "misst dieser Lieferant nicht".
  **Beides führt zu keinerlei Gating** — Bundles ohne die Felder sehen aus wie
  bisher. Auswertung im Frontend ausschließlich über `viz/resolution.ts`.

  Warum das im Vertrag steht und nicht nur im Lieferanten: bei trainierten
  (statt konstruierten) Vektoren zeigt die 3D-Ansicht ohne dieses Gating
  ausgerechnet die *unaufgelösten* Wörter am auffälligsten — am SGNS-Bundle
  gemessen korreliert die 3D-Weglänge mit dem Kontrast zu **r = −0,65**.

## `anchors`

Kuratierte Kontextwörter mit Cluster-Label und Koordinaten: beschriftete
Landmarken im 3D-Raum, an denen man die Wanderung eines Zielworts ablesen kann.
Nachbarn werden nicht nur unter den Ankern gesucht, sondern unter allen
Kandidaten (s. `neighbors`); ist ein Nachbar zugleich Anker, trägt er dessen
Label und Cluster.

```json
{"label": "Einsatzkräfte", "cluster": "MILITÄR", "lemma": "Einsatzkräft",
 "vector": [...], "coords_3d": [x, y, z]}
```

- `label`: Anzeigeform, im Bundle eindeutig.
- `lemma`: optional, das Lemma im Vokabular, falls es von `label` abweicht
  (etwa durch Lemmatisierung). **Mit `meta.neighbor_space` maßgeblich:** Die
  Zuordnung eines Nachbarn zu einem Anker läuft über `lemma` (s. „Auswertung
  einer Abfrage“, Schritt 5). Ein Anker ohne `lemma` wird dort keinem Nachbarn
  zugeordnet.

Im Demo-Bundle sind es reale Dornseiff-Lemmata der einschlägigen Sachgruppen;
im PPMI-Pfad stammen sie aus einer kuratierten, parameterunabhängigen
Maximalliste, von der ein Lauf die im Vokabular vorhandenen übernimmt.

## `neighbor_coords`

Optionales Feld, nur `{"label": ..., "coords_3d": [x,y,z]}` — kein `cluster`,
kein `vector`. Per Out-of-Sample-Projektion in denselben Raum gelegt (keine neue
PCA, damit sich die Koordinaten von `anchors`/`aggregates` nicht verschieben).

**Zweck: Koordinaten-Nachschlagewerk zu den `neighbors`-Listen.** Ein Nachbar
wird im Bild nur gezeichnet, wenn seine Position bekannt ist; Anker bringen sie
mit, alle anderen Nachbarn finden sie hier. Das Feld ist *kein* eigener
Suchraum — ausgewählt wird ausschließlich im vollen Vektorraum (s. `neighbors`).
Mit `meta.neighbor_space` kommen die Koordinaten aller Kandidaten aus der
Indextabelle; `neighbor_coords` deckt dann nur noch die vorberechneten Listen ab
(Rückfall ohne Backend).

Zwei Bedingungen, beide von der Frontend-Seite her zwingend:

- **Abdeckung.** Jedes Label aus `neighbors`, das kein `anchors`-Eintrag ist,
  braucht hier einen Punkt. Praktisch heißt das: `neighbor_coords` aus demselben
  Kandidatenpool ziehen wie die Nachbarn, insbesondere ohne zusätzlichen
  Wortart-Filter. Ein strengerer Filter als bei der Nachbarauswahl hinterlässt
  Löcher im Bild.
- **Eindeutigkeit.** Ein Label darf nicht gleichzeitig in `anchors` und in
  `neighbor_coords` stehen — auch nicht über die *Anzeigeform* eines Ankers
  (`anchors[].label` kann von seinem Ausgangslemma abweichen). Das Frontend löst
  Nachbarn über den Text auf; derselbe Text an zwei Positionen ist mehrdeutig.
  Es gibt dann dem Anker den Vorrang, aber der Vertrag sollte den Fall gar nicht
  erst liefern.

Fehlt das Feld (ältere Bundles), werden nur diejenigen Nachbarn im Raum
gezeichnet, die zugleich Anker sind — der Rest bleibt auf die Liste beschränkt.

## `neighbors`

```json
{"bundeswehr": {"2022Q3": [{"label": "Wehrhaftigkeit", "score": 0.89, "group": "WEHR"}, ...]}}
```

Vorberechnet auf Aggregat-Ebene gegen die gefilterte Kandidatenliste (alle
Kontextwörter, nicht nur `anchors`). Welche Filter gelten, legt der Lieferant
fest; mit `meta.neighbor_space` sind es genau die Voreinstellungen der dort
deklarierten Filter und Stoppwörter. **Die Array-Reihenfolge IST der Rang; das
Tool sortiert nicht um.** Angezeigt wird `score` als roher Kosinus.

**Listenlänge k legt der Lieferant fest**, und sie ist gleichzeitig die
Obergrenze eines Reglers im Frontend: dieselbe Liste speist die Textanzeige und
die beschrifteten Punkte im 3D-Bild, der Regler schneidet sie nur ab. Kurze
Listen (8) reichen für die Textanzeige; für das Bild sind längere (50–150)
sinnvoll (Abfragen über `meta.neighbor_space` liefern bis 1000), weil dort erst sichtbar wird, wie weit echte Nachbarn in der
Projektion auseinanderliegen. Wer k erhöht, sollte prüfen, ob der rohe Kosinus
am letzten Rang noch deutlich über dem Grundpegel liegt (dem mittleren Kosinus
einer Zelle gegen beliebige Kandidaten) — sonst rankt das Verfahren nur noch
Rauschen. Die Positionen dazu kommen aus `anchors` bzw. `neighbor_coords`.

Ohne `meta.neighbor_space` **legt der Lieferant das Auswahlverfahren fest**;
mit Suchraum ist es durch die Auswertungsregel unten festgelegt (roher Kosinus).
Beide Pfade dieses Projekts ranken nach **rohem Kosinus**:
- *Konstruierte* Vektoren (PPMI-SVD): seit 2026-10-05. Die frühere Auswahl per
  Kosinus-„Kontrast“ (Rohwert minus Durchschnitt des Kandidaten über alle
  Zellen) ist verworfen; sie war kein etabliertes Verfahren, und die geprüften
  Hubness-Korrekturen (CSLS, Mutual Proximity) brachten keinen Vorteil. Der
  Effekt, dass Kontextwörter mit nur einer belegten Zeile deren Richtung
  übernehmen, wird über die Mindestbelegung gefiltert.
- *Trainierte* Vektoren (SGNS): Dort richtet ein Kontrast aktiv Schaden an: bei
  einer unaufgelösten Zelle ist der Kosinus zu allem etwa gleich groß, „Kosinus
  minus Spaltenmittel“ wird dann der Rangfolge nach zum negativen Spaltenmittel
  — gewählt würden die *unähnlichsten* Wörter. Solche Zellen markiert man über
  `generik`, statt sie umzusortieren.

Ist der Nachbar zufällig ein kuratierter Anker, übernimmt er
dessen Anzeigelabel + `group` (Cluster); sonst ist `group` ein generisches
Fallback-Tag. Für facettenspezifische NN könnte das Backend on-demand
nachrechnen (Vektoren liegen ja im Bundle).

**Mit `meta.neighbor_space`** gilt zusätzlich eine Konsistenzbedingung: Die
`neighbors` müssen genau das Ergebnis einer Nachbarabfrage mit den
Voreinstellungen sein (s. `meta.neighbor_space`, „Konsistenz mit
`neighbors`“). Sie dienen dann als Rückfall, wenn kein Backend erreichbar ist.

## `meta.neighbor_space` – Suchraum für Nachbarabfragen (Format 1)

Optional. Ein Bundle mit diesem Feld erlaubt es, die Nachbarlisten bei jeder
Abfrage neu zu berechnen, mit vom Nutzer gewählten Filtern und Stoppwörtern.
Der Abschnitt ist **normativ**: Er legt nicht nur die Daten fest, sondern auch,
wie eine Abfrage über diese Daten auszuwerten ist. Jede Umsetzung (Backend,
Lieferant für die vorberechneten `neighbors`) muss bei gleichen Eingaben
dieselben Listen liefern.

Aufgabenteilung: **Was** gefiltert werden kann und mit welchen
Voreinstellungen, deklariert der Lieferant in den Daten. **Wie** die Filter
wirken, legt dieser Abschnitt fest. Ein neues Filterattribut mit bekanntem
Filtertyp braucht deshalb keine Änderung an App oder Backend; ein neuer
Filtertyp ist eine Formatänderung (neue Formatversion).

### Felder

```json
"neighbor_space": {
  "format": 1,
  "vectors": "bundestag_zeitenwende_quarter.space_vectors.npy",
  "index": "bundestag_zeitenwende_quarter.space_index.parquet",
  "neighbor_kind": "context",
  "similarity": "cosine",
  "filters": [
    {"column": "pos", "type": "categorical", "label": "Wortart",
     "default": ["NOUN", "PROPN", "ADJ", "VERB"]},
    {"column": "n_rows", "type": "minimum", "label": "Mindestbelegung",
     "description": "…", "default": 10},
    {"column": "n_cooccurrences", "type": "minimum",
     "label": "Mindestzahl Kookkurrenzen", "default": null},
    {"column": "is_speaker_name", "type": "exclude_if_true",
     "label": "Rednernamen ausblenden", "default": true}
  ],
  "stopwords": ["Präsident", "Präsidentin", "…"]
}
```

| Feld | Pflicht | Bedeutung |
|---|---|---|
| `format` | ja | Formatversion, hier `1`. Eine Umsetzung, die die Version nicht kennt, darf den Suchraum nicht verwenden und fällt auf `neighbors` zurück. |
| `vectors` | ja | Dateiname der Vektormatrix (s. u.). Nur ein nackter Dateiname, kein Pfad; die Datei liegt im selben Verzeichnis wie das Bundle-JSON. |
| `index` | ja | Dateiname der Indextabelle, gleiche Regel. |
| `neighbor_kind` | ja | Was ein Nachbar ist: `context` = Kontextwörter, die Liste beschreibt das Kollokationsprofil (PPMI-SVD); `word` = Wörter im selben Vektorraum (SGNS). Wirkt nur auf die Beschriftung im Frontend, nicht auf die Rechnung. |
| `similarity` | ja | In Format 1 nur `cosine`. |
| `filters` | ja | Deklaration der angebotenen Filter, Reihenfolge = Anzeigereihenfolge. Kann leer sein. |
| `stopwords` | ja | Mitgelieferte Stoppwortliste (Lemmata). Kann leer sein. |
| `frequency` | nein | Spalte (eine der Filterspalten vom Typ `minimum`), die die Häufigkeit eines Kandidaten angibt. Bestimmt die Schriftgröße in der Wortwolke der Filterseite; fehlt das Feld, sind alle Wörter gleich groß. |

Ein Filtereintrag hat `column` (Spalte der Indextabelle), `type` (s. u.),
`label` (Anzeigetext), optional `description` und `default` (Voreinstellung,
Typ je Filtertyp). Weitere Felder sind erlaubt und werden ignoriert.

### Dateien

**Vektormatrix** (`.npy`, NumPy-Format): float32, Form (Anzahl Zeilen ×
Dimension). Jede Zeile ist auf Länge 1 normiert (im Rahmen der
float32-Genauigkeit). Umsetzungen **normieren nicht erneut**, sonst weichen
die Scores in den letzten Stellen ab.

**Indextabelle** (`.parquet`): eine Zeile je Zeile der Vektormatrix.

| Spalte | Typ | Bedeutung |
|---|---|---|
| `row` | Ganzzahl | Zeilennummer in der Vektormatrix; lückenlos 0 … n−1 in Tabellenreihenfolge |
| `role` | String | `query` (Abfragevektor) oder `candidate` (möglicher Nachbar) |
| `slice` | String / null | Zeitscheibe (`time_slices[].id`). Bei `query` Pflicht. Bei `candidate`: Kandidat gehört nur zu dieser Zeitscheibe; `null` = gilt für alle Zeitscheiben |
| `entity_id` | String / null | Bei `query` Pflicht (`entities[].id`), bei `candidate` null |
| `lemma` | String | Lemma; bei Kandidaten Grundlage für Stoppwörter und Anker-Zuordnung |
| `coords_3d` | Liste von 3 Zahlen | Position im 3D-Bild, im selben Raum wie `aggregates[].coords_3d` |
| *Filterattribute* | je Filtertyp | Jede in `filters` genannte Spalte; nur bei Kandidaten belegt |

Je Kombination aus `entity_id` und `slice` gibt es höchstens eine
`query`-Zeile. Die Abfragevektoren entsprechen den Aggregaten
(`aggregates[].vector`, normiert). Abfrage- und Kandidatenvektoren einer
Zeitscheibe müssen im selben Raum liegen (bei SGNS also beide vor oder beide
nach der Ausrichtung).

PPMI: ein gemeinsamer Raum, alle Kandidaten haben `slice = null`. SGNS: ein
Raum je Zeitscheibe, jeder Kandidat trägt seine Zeitscheibe.

### Filtertypen

Die Filterwerte einer Abfrage kommen vom Nutzer; fehlt ein Wert, gilt
`default`. Ein Wert `null` bedeutet bei jedem Typ: Filter nicht aktiv.

| Typ | Wert | Kandidat bleibt, wenn … | Spaltentyp |
|---|---|---|---|
| `categorical` | Liste von Strings | sein Wert in der Liste steht. Leere Liste: kein Kandidat bleibt. | String |
| `minimum` | Zahl | sein Wert ≥ dem Filterwert ist | Zahl |
| `exclude_if_true` | Boolean | `true`: sein Wert nicht `true` ist; `false`: Filter nicht aktiv | Boolean |

Optional bei `minimum`: `out_of` (Zahl), die Obergrenze des Werts. Die
Oberfläche zeigt den Filter dann als „≥ 10 von 80“ (PPMI: Zahl der belegten
Zellen Zielwort × Zeitscheibe, also der Zeilen der PPMI-Matrix).

**Fehlende Attributwerte** (`null` in der Spalte): Der Kandidat bleibt drin,
unabhängig vom Filter. (Vorläufige Festlegung, 2026-10-06.)

Mehrere Filter wirken zusammen (UND). Filter auf Spalten, die nicht in
`filters` deklariert sind, sind unzulässig.

### Stoppwörter

Die Abfrage enthält eine Stoppwortliste (Voreinstellung: `stopwords`). Ein
Kandidat fällt heraus, wenn sein `lemma` **exakt** (mit Groß-/Kleinschreibung)
in der Liste steht. Abgeglichen wird das Lemma, nicht das Anzeigelabel.
Stoppwörter, die nicht im Index vorkommen, sind kein Fehler.

### Auswertung einer Abfrage

Eingaben: `k`, Filterwerte, Stoppwortliste. Für jede `query`-Zeile q:

1. **Kandidaten:** alle `candidate`-Zeilen mit `slice = null` oder
   `slice = q.slice`, die alle Filter und die Stoppwortliste passieren.
2. **Ähnlichkeit:** Skalarprodukt aus dem Vektor von q und jedem
   Kandidatenvektor, **gerechnet in float64** (die float32-Werte werden vorher
   nach float64 umgewandelt). Begründung: In float32 hängt das Ergebnis vom
   Rechenweg ab (Matrix-Matrix- gegen Matrix-Vektor-Produkt, gemessen bis
   ca. 4·10⁻⁷), und das vertauscht fast gleiche Nachbarn. In float64 liegt der
   Unterschied bei ca. 10⁻¹⁶.
3. **Rang:** absteigend nach Ähnlichkeit; bei exakt gleicher Ähnlichkeit
   entscheidet `row` aufsteigend. Exakte Gleichstände sind häufig (Kandidaten
   mit identischem Vektor, etwa Kontextwörter mit nur einer belegten Zeile).
4. **Länge:** die ersten `k`; gibt es weniger Kandidaten, ist die Liste kürzer.
5. **Einträge:** `{label, score, group}` wie in `neighbors`, zusätzlich darf
   `lemma` mitgeliefert werden.
   - `label` und `group`: Gibt es einen Anker mit `anchors[].lemma` gleich dem
     Kandidaten-Lemma, dessen `label` und `cluster`; sonst das Lemma und
     `KONTEXT`.
   - `score`: die Ähnlichkeit aus Schritt 2, **erst nach dem Sortieren**
     auf 4 Nachkommastellen gerundet, und zwar korrekt gerundet auf dem
     float64-Wert wie Pythons `round(x, 4)` (nicht über `x · 10⁴`, das
     weicht an Rundungsgrenzen ab).

Ergebnis je `entity_id` und `slice`, aufgebaut wie `neighbors`.

### Konsistenz mit `neighbors`

Enthält ein Bundle `neighbor_space`, dann gilt: Eine Abfrage mit allen
Voreinstellungen (`default` jedes Filters, `stopwords`) und k = Länge der
mitgelieferten Listen ergibt **exakt** `neighbors` (Labels, Reihenfolge,
Scores). Das Backend prüft das in einem Konformitätstest für jedes
ausgelieferte Bundle.

Damit hat jeder mögliche Nachbar eine Position: Die Koordinaten aller
Kandidaten stehen im Index (`coords_3d`). `neighbor_coords` deckt nur die
Nachbarn der vorberechneten Listen ab.

## Konventionen / Fallstricke

- **Ein Raum, kein Alignment.** Alle Vektoren stammen aus derselben Projektion.
  Niemals pro Facette separat projizieren – sonst driften Koordinatensysteme
  auseinander und Bewegung ist nicht mehr als Bedeutungswandel lesbar.
- **Sparsity ist eine Datenfrage, keine Toolgrenze.** Bei feiner Facettierung
  (Wort × Quartal × Publikation × Ressort) können Zellen auf wenige Sätze
  fallen. `freq` transportiert das; das Tool macht es sichtbar statt es zu
  kaschieren. (Zeitenwende-Bundle enthält bewusst leere/dünne Frühzellen, z.B.
  "Zeitenwende" vor 2022.)
- **Terminologie:** die realen Vektoren sind später *konstruiert* (PPMI-SVD),
  nicht *trainiert*. Diese Demo-Vektoren sind weder – sie sind synthetische
  Stand-ins, die nur die Raum-Eigenschaften nachbilden.
