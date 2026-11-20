# Readme Frequency Control

Stand: 2026-11-20

Dieses Dokument fasst zusammen, was zur Kontrolle der Trefferzahl (Zahl der
KWIC-Belege je Zielwort und Zeitscheibe) geprüft und gemessen wurde und was
daraus für die Pipeline folgt. Die Messungen stehen in
`stages/02_embedding/pipeline_emb_ppmi_diagnose.py` (Abschnitte 1–4).
Methodische Einordnung: `Readme Scientific Methodology.md` (1.8, 1.17, 1.18,
1.23, 2.15, 3.12); Begriffe: `Readme Glossary.md`; Umsetzungsschritte:
`nextsteps.md`. Übertragbarkeit auf die SGNS-Pipeline: Abschnitt 8.

## 1. Ausgangslage

Ein Teil des gemessenen Bedeutungswandels ist ein Messartefakt der
Häufigkeit: Ein Kontextprofil aus wenigen Belegen ist eine ungenaue Schätzung,
und die Zufallsschwankung der Stichprobe erscheint im Vektorraum als Abstand.
Dubossarsky, Weinshall & Grossman (2017) zeigen, dass mehrere berichtete
Zusammenhänge zwischen Häufigkeit und Wandel (u. a. das *law of conformity*
bei Hamilton, Leskovec & Jurafsky 2016b) unter einer Kontrollbedingung
weitgehend verschwinden.

Zwei Unterscheidungen gelten für alle Entscheidungen:

- **Häufigkeitswandel ist selbst ein Befund.** Er bleibt als `freq` bzw.
  `n_hits` im Bundle. Jede Kontrolle betrifft nur das semantische Maß.
- **Gleiche Belegzahlen gleichen das Rauschen an, nicht den Inhalt.** Eine
  Zufallsstichprobe erhält im Erwartungswert die Zusammensetzung der Kontexte.
  Ein neuer Diskurs in einer Zeitscheibe bleibt auch nach einer Angleichung
  sichtbar.

Die Prüfung liegt in der Embedding-Stufe: Wie stark sich eine Zeile mit h
Belegen scheinbar verschiebt, lässt sich erst nach PPMI, SVD und *folding-in*
messen; eine einmalige Ziehung schon in pre ersetzte eine bekannte Verzerrung
durch eine unbekannte.

Die Diagnose stellt vier verschiedene Fragen:

| Abschnitt | Frage | Kennwert |
|---|---|---|
| 1 | Ab wie vielen Belegen ist eine **einzelne Zeile** wiedererkennbar? | a_zeit (AUC) |
| 2 | Soll die SVD ohne schwach belegte Zeilen berechnet werden? | Δ a_zeit, Δ a_wort |
| 3 | Welche k und p? Wie stark hängt die Ähnlichkeit von der Textmenge ab? | a_zeit, a_wort, Volumeneffekt r |
| 4 | Hängt der **Abstand zwischen zwei Zeilen** (Drift) von ihrer Trefferzahl ab, und welche Korrektur hilft? | r innerhalb der Wörter |

## 2. Daten und Ungleichheit der Trefferzahlen

Bundestag, KWIC-Variante `punkt`, Raster `halfyear`: 692.136 Belege aus
67.062 Reden, 596 Zielwörter, 26 Halbjahre, 12.280 Zeilen
(Zielwort × Zeitscheibe), 120.882 Kontextwörter. k = 300, p = 0,5.

Trefferzahl je Zeile: Median 9, oberes Quartil 31, 90-%-Quantil 114,
Maximum 5.363.

| Zeilen unter … Belegen | 3 | 5 | 10 | 30 | 50 |
|---|---|---|---|---|---|
| Anteil Zeilen | 22 % | 34 % | 52 % | 75 % | 82 % |
| Anteil Belege | 1 % | 1 % | 3 % | 10 % | 15 % |

Paare benachbarter Halbjahre desselben Zielworts:

| | Paare | Zielwörter | Verhältnis größere/kleinere Trefferzahl, Median | 90-%-Quantil | Anteil ≥ 2 |
|---|---|---|---|---|---|
| alle | 10.537 | 587 | 2,00 | 8,00 | 53 % |
| beide Zeilen ≥ 30 Belege | 2.275 | 203 | 1,43 | 3,07 | 23 % |

Drei Viertel der Zeilen liegen unter der Schwelle h\* = 30 (Abschnitt 5.1),
tragen aber nur 10 % der Belege. Nur 22 % der Paare und 203 der 596
Zielwörter haben Paare, deren beide Zeilen die Schwelle erreichen.

## 3. Messverfahren (kurz)

Alle Abschnitte verwenden die Split-half-Prüfung (Methodology 1.17): Die Reden
jeder Zeitscheibe werden zufällig halbiert (20 Teilungen, Seed 42), die SVD an
einer Hälfte berechnet, die andere per *folding-in* projiziert. Teilstichproben
werden nach Reden gezogen. „Belastbar“ heißt in den Abschnitten 1–3: gleiches
Vorzeichen in allen 20 Teilungen.

**Drift** (Abschnitt 4) ist die Kosinusdistanz 1 − cos zwischen den
Zeilenvektoren eines Zielworts in zwei benachbarten Zeitscheiben (Hamilton,
Leskovec & Jurafsky 2016b; Kutuzov et al. 2018). Ausgewertet werden Paare, deren
beide Zeilen mindestens `DRIFT_FLOOR` = h\* Belege haben.

- **Kennwert:** Pearson-r zwischen Drift und log der kleineren Trefferzahl des
  Paars, **innerhalb der Zielwörter** (Abweichungen vom Wortmittel; Bland &
  Altman 1995). Ziel r = 0; negativ heißt: mehr Drift zwischen schwach belegten
  Zeitscheiben.
- **Unsicherheit:** Resampling der Zielwörter mit Zurücklegen (in der Literatur
  *Bootstrap*; Efron & Tibshirani 1993), jedes Wort mit allen seinen Paaren
  (Davison & Hinkley 1997), 1.000 Wiederholungen, 95-%-Perzentilintervall.
  „Belastbar“ heißt hier: Intervall schließt 0 aus.
- **Varianten:** Rohdrift; Teilstichprobe je Paar (stärkere Zeile auf die
  Trefferzahl der schwächeren, 20 Ziehungen, Distanz gemittelt); Teilstichprobe
  je Wort (alle Zeilen eines Worts auf seine kleinste Trefferzahl), mit
  gemittelten Distanzen oder gemittelten Vektoren; Drift minus
  Permutationserwartung.
- **Permutationserwartung:** Reden beider Zeitscheiben eines Paars 99-mal
  zufällig auf zwei Zeilen mit den echten Trefferzahlen verteilt (Good 2005;
  Kontrollbedingung nach Dubossarsky et al. 2017); beide Zeilen mit den
  gemeinsamen Randsummen. p-Werte nach Phipson & Smyth (2010).
- **Sättigungskurve** (Rarefaktion; Sanders 1968; Hurlbert 1971): Für Paare
  mit ≥ 50 Belegen in beiden Zeilen (1.628 Prüfpaare) je Zeile genau h Belege
  gegen zwei Stichproben von je h Belegen aus den zusammengelegten Belegen
  (Drift ohne Unterschied).

## 4. Ergebnisse (Lauf 2026-11-19, alle Abschnitte)

### 4.1 Abschnitt 1: Reliabilität einer Zeile

| Belege je Hälfte | 1 | 5 | 7 | 10 | 20 | 30 | 50 | alle (≥ 50) |
|---|---|---|---|---|---|---|---|---|
| a_zeit, Teilstichproben | 0,485 | 0,482 | 0,506 | 0,545 | 0,675 | **0,762** | 0,854 | 0,935 |
| Teilungen ≥ 0,70 | 0 | 0 | 0 | 0 | 0 | **20** | 20 | 20 |
| a_wort | 0,729 | 0,912 | 0,947 | 0,974 | 0,996 | 0,999 | 1,000 | 1,000 |

- **h\* = 30** (a_zeit ≥ 0,70 in allen Teilungen); Zufallsgrenze **h₀ = 10**
  (a_zeit > 0,5 in allen Teilungen).
- Gegenprobe (natürliche Zeilen nach Trefferzahl der kleineren Hälfte):
  Niveau in allen Teilungen ab der Gruppe 10–14 Belege (0,763). Sie liegt
  **unter** h\*; die Gruppen vergleichen aber verschiedene Wörter und sind mit
  der Trefferzahl vermischt. Es gilt h\* = 30.
- a_wort taugt nicht als Kriterium: Schon ein Beleg unterscheidet ein Zielwort
  von den anderen (0,73). Das Problem dünner Zeilen ist die Abgrenzung von den
  anderen Zeitscheiben desselben Worts.

### 4.2 Abschnitt 2: SVD ohne schwach belegte Zeilen

| N | Zeilen unter N | Δ a_zeit | Δ a_wort | Einstufung | Δ a_zeit der nicht gefitteten Zeilen |
|---|---|---|---|---|---|
| 3 | 22 % | +0,004 | +0,003 | Verbesserung | −0,006 |
| 5 | 34 % | +0,004 | +0,002 | Verbesserung | −0,004 |
| 7 | 43 % | +0,004 | +0,001 | Verbesserung | −0,001 |
| 10 | 52 % | +0,002 | 0,000 | kein belastbarer Unterschied | −0,001 |
| 30 | 75 % | −0,011 | −0,000 | nicht berechenbar | −0,009 |

- Bei kleinem N sehr kleine, aber belastbare Gewinne für die gut belegten
  Zeilen; die ausgeschlossenen Zeilen verlieren dabei an a_zeit.
- Bei N = h\* = 30 sinkt a_zeit in allen Teilungen (−0,011). „Nicht
  berechenbar“ ist ein Artefakt der Regel: Das untere Terzil der
  Bewertungszeilen (30–55 Belege) ist leer, weil die Terzile nicht aus den
  Bewertungszeilen gebildet wurden (im Notebook seit 2026-11-20 behoben,
  noch nicht neu gerechnet).
- **Entscheidung:** `MIN_HITS_FIT = None` bleibt. Die Gewinne bei kleinem N
  (+0,004 AUC) rechtfertigen nicht, dass ein Fünftel bis zwei Fünftel der
  Zeilen schlechter dargestellt werden; bei der Deutungsschwelle 30 schadet
  der Ausschluss.

### 4.3 Abschnitt 3: k, p und Volumeneffekt

| k (p = 0,5) | 10 | 25 | 100 | **300** | 1.000 | 1.500 |
|---|---|---|---|---|---|---|
| a_zeit | 0,692 | 0,700 | 0,685 | **0,682** | 0,685 | 0,687 |
| a_wort | 0,924 | 0,938 | 0,925 | **0,908** | 0,879 | 0,868 |
| Volumeneffekt r | 0,581 | 0,688 | 0,693 | **0,527** | 0,286 | 0,188 |
| Anteil quadrierte Frobenius-Norm | 5 % | 6 % | 10 % | **18 %** | 38 % | 50 % |

- p wirkt auf a_zeit und a_wort kaum (≤ 0,02), stark aber auf den
  Volumeneffekt: Je kleiner p, desto kleiner r (k = 300: p = 1 0,672,
  p = 0,5 0,527, p = 0 0,450).
- Alle 35 Kombinationen sind gegenüber k = 300, p = 0,5 ein Zielkonflikt;
  Empfehlung des Notebooks: Referenz beibehalten.
- Der Volumeneffekt (gepoolt, zwischen Wörtern) ist bei der Referenz deutlich
  (r = 0,53): Gut belegte Zeilen sind den übrigen Zeitscheiben ihres Worts
  ähnlicher. Auf null bringt ihn nur k = 1.500, p = 0 (0,026), um den Preis von
  a_wort (0,849).

### 4.4 Abschnitt 4: Sättigungskurve

| Belege je Zeile | 1 | 5 | 10 | 20 | 30 | 50 | alle (≥ 50) |
|---|---|---|---|---|---|---|---|
| Drift beobachtet | 0,851 | 0,752 | 0,689 | 0,622 | 0,589 | 0,560 | 0,574 |
| Drift ohne Unterschied | 0,841 | 0,707 | 0,605 | 0,472 | 0,389 | 0,290 | 0,119 |
| Verhältnis | 99 % | 94 % | 88 % | 76 % | 66 % | 52 % | 21 % |

Die beobachtete Drift bleibt bei rund 0,57 stehen (echter Unterschied
benachbarter Halbjahre, Kosinus ≈ 0,43); die Drift ohne Unterschied fällt
weiter. Keine Sättigung im Raster, kein Knick.

### 4.5 Abschnitt 4: Volumenabhängigkeit der Drift (2.275 Paare, 203 Zielwörter)

| Variante | Belege | r | 95-%-Intervall | Δ\|r\| (Intervall) | Einstufung | Rangkorr. |
|---|---|---|---|---|---|---|
| Rohdrift (Referenz) | 100 % | −0,260 | −0,330 bis −0,192 | – | Referenz | 1,000 |
| **Teilstichprobe je Paar** | **73 %** | **−0,094** | −0,208 bis 0,004 | −0,166 (−0,222 bis −0,103) | belastbar geringer | 0,977 |
| Teilstichprobe je Wort, Distanzen | 19 % | −0,120 | −0,185 bis −0,058 | −0,139 (−0,177 bis −0,103) | belastbar geringer | 0,900 |
| Teilstichprobe je Wort, Vektoren | 19 % | −0,390 | −0,467 bis −0,318 | +0,131 (0,092 bis 0,173) | belastbar höher | 0,846 |
| Drift − Permutationserwartung | 100 % | +0,295 | 0,216 bis 0,357 | +0,035 (−0,101 bis 0,151) | kein belastbarer Unterschied | 0,807 |
| *Drift mit gemeinsamen Randsummen* | alle | −0,204 | −0,282 bis −0,133 | – | zur Information | 0,994 |
| *Permutationserwartung* | alle | −0,822 | −0,841 bis −0,797 | – | zur Information | 0,033 |

Empfehlung des Notebooks: **Teilstichprobe je Paar**; gleichwertig in |r| ist
die Teilstichprobe je Wort mit gemittelten Distanzen. Zwischen beiden
entscheidet der Belegverlust (73 % gegen 19 % verwendete Belege).

### 4.6 Abschnitt 4: Permutationstest

| Gruppe (kleinere Trefferzahl) | Paare | Anteil p < 0,05 | Drift (gem. Randsummen) | Permutationserwartung |
|---|---|---|---|---|
| alle | 2.275 | 100 % | 0,566 | 0,153 |
| 30–54 Belege | 759 | 100 % | 0,548 | 0,235 |
| 54–138 Belege | 758 | 100 % | 0,540 | 0,157 |
| 138–3.690 Belege | 758 | 100 % | 0,611 | 0,068 |

### 4.7 Vergleich mit dem Testlauf (2026-11-19, nur Abschnitt 4, `DRIFT_FLOOR` = 10)

Mit 4.327 Paaren ab 10 Belegen lag r der Rohdrift bei −0,41; die
Teilstichprobe je Paar senkte es nur auf −0,35, die Teilstichprobe je Wort
(Distanzen) auf −0,15 bei 13 % verwendeten Belegen, und diese war die
Empfehlung. Ab 30 Belegen kehrt sich das um. Die Untergrenze der Paare
bestimmt also, welche Korrektur genügt (Deutung in 5.3).

## 5. Deutung

1. **h\* = 30 heißt „wiedererkennbar“, nicht „rauschfrei“.** Bei 30 Belegen
   macht die Drift ohne Unterschied noch zwei Drittel der beobachteten aus.
   Der Abstand zweier Zeilen braucht mehr Belege als das Wiedererkennen einer
   Zeile.
2. **Eine höhere Schwelle ist keine Lösung.** Die Sättigungskurve hat keinen
   Knick; auch bei 50 Belegen ist das Rauschen halb so groß wie die
   beobachtete Drift, und schon über 30 liegen nur 25 % der Zeilen.
3. **Ungleich belegte Paare sind das Hauptproblem, sobald die Belegung
   reicht.** Die Angleichung innerhalb eines Paars genügt ab 30 Belegen
   (r −0,26 → −0,09), ab 10 Belegen nicht (−0,41 → −0,35). *Mögliche
   Erklärung, nicht geprüft:* Schwach belegte Zeilen zeigen systematisch in
   eine andere Richtung (Volumeneffekt, 4.3). Bei gleicher Belegung tragen
   beide Zeilen eines Paars dieselbe Verzerrung, und sie hebt sich im Kosinus
   weitgehend auf. Unter 30 Belegen kommt das absolute Rauschen hinzu, das
   sich je Paar nicht angleichen lässt.
4. **Gemittelte Vektoren führen die Ungleichheit wieder ein.** Bei gut
   belegten Zeilen glättet das Mittel über die Ziehungen das Rauschen, bei der
   schwächsten Zeile eines Worts sind alle Ziehungen gleich. Die einzige
   Variante, die in das heutige Bundle passt, ist damit schlechter als nichts.
5. **Rauschen und echter Unterschied addieren sich nicht.** Die beobachtete
   Drift hängt viel schwächer von der Belegung ab (−0,20) als reines Rauschen
   (−0,82). Gibt es einen echten Unterschied, kommt nur ein Teil des
   Rauschens hinzu; der Abzug der vollen Permutationserwartung überkorrigiert
   (r kippt auf +0,30). Aus demselben Grund ist das Verhältnis in der
   Sättigungskurve kein additiver Anteil.
6. **Über dem Rauschen ≠ Bedeutungswandel.** Alle 2.275 Paare liegen über der
   Permutationserwartung. Benachbarte Halbjahre unterscheiden sich im
   Bundestag fast immer (Debatten, Anlässe, Redner); Halbjahre sind keine
   austauschbaren Stichproben. Als Filter trennt der p-Wert hier nichts; er
   bestätigt nur, dass die Drift ab 30 Belegen keine reine Zufallsschwankung
   ist. Ob Themenwechsel oder Bedeutungswandel, beantwortet keine der
   Messungen.
7. **Die Rangfolge der Paare bleibt stabil.** Die Teilstichprobe je Paar
   erhält die Reihenfolge fast (0,977); das Rauschniveau eines Paars sagt fast
   nichts über seinen Rang in der beobachteten Drift (0,033). Die Korrektur
   entfernt vor allem das Artefakt bei ungleich belegten Paaren.
8. **Ein kleiner Rest bleibt.** r = −0,09 ist nicht von 0 zu unterscheiden,
   aber das Intervall berührt 0 nur knapp. „Nicht nachweisbar“ heißt nicht
   „nicht vorhanden“.

## 6. Folgerungen für Datenauswahl und Pipeline

Was **bleibt**:

- SVD an allen Zeilen (`MIN_HITS_FIT = None`).
- k = 300, p = 0,5 (Zielkonflikt überall, keine eindeutig bessere
  Kombination).
- Keine Schwelle bei 50; kein Abzug der Permutationserwartung; keine
  gemittelten Vektoren; kein p-Wert als Filter.

Was **zu ändern** ist (Umsetzung in `nextsteps.md`):

1. **Zeilen unter h\* markieren, nicht entfernen.** h\* = 30 als
   `meta.min_reliable_freq` ins Bundle; die App zeigt solche Zeilen
   abgeschwächt, sie bleiben als Nachbarn und für den Häufigkeitswandel im
   Raum.
2. **Drift nur für zuverlässige Paare, mit Teilstichprobe je Paar.**
   Zuverlässig ist ein Paar, wenn min(freq_A, freq_B) ≥ h\*. Die Drift wird in
   emb je Paar berechnet (20 Ziehungen) und als eigenes Feld je Paar ins
   Bundle übergeben; Positionen und Nachbarn bleiben unverändert. Paare
   darunter erhalten keinen oder einen ausdrücklich markierten Wert.
   Optional als Genauigkeitsangabe: min(freq_A, freq_B) oder die
   Permutationserwartung des Paars.
3. **Zielwörter nach Abdeckung auswählen.** Wörter mit kaum Zeilen über h\*
   liefern keine deutbare Entwicklung (nur 203 von 596 haben überhaupt ein
   zuverlässiges Paar). Kriterium z. B. mindestens n zuverlässige Paare.
4. **Zeitraster vergröbern, wo die Belege nicht reichen** (eigener
   Diagnoselauf). Ein Jahr verdoppelt grob die Belege je Zeile (Median
   9 → ~18, noch unter 30). h\* hängt am Raster und muss je Raster neu
   bestimmt werden.
5. **Facetten:** Records je Fraktion oder Publikation sind noch dünner belegt;
   dieselbe Schwelle gilt.

Rangfolge: 1–3 zusammen bringen am meisten und ändern nicht die Daten, sondern
was als deutbar ausgewiesen wird. 4 braucht einen eigenen Lauf.

## 7. Übertragbarkeit auf andere Korpora

Alle Zahlen gelten für dieses Korpus, dieses Raster, diese Zielwörter und
k = 300, p = 0,5. Auf dem DNB-Korpus (andere Textsorte, andere
Belegverteilung, Ziehen nach Werken statt nach Reden) sind **neu zu messen**:
h\*, k und p, die Stärke des Volumeneffekts und die beste Driftkorrektur.
Der Vergleich mit dem Testlauf (4.7) zeigt, dass schon die Untergrenze der
Paare die Empfehlung umdrehen kann.

**Übertragbar** sind die Mechanismen: Rauschen fällt mit der Belegzahl;
schwach belegte Zeilen haben eine systematisch andere Richtung; Rauschen und
Unterschied addieren sich nicht (Abzug überkorrigiert); gemittelte Vektoren
bringen die Ungleichheit zurück; „signifikant“ heißt nicht „Bedeutungswandel“.

Die Entscheidungsregeln des Notebooks stehen vor den Ergebnissen fest; nur die
Zahlen werden neu berechnet. Deshalb wird h\* ins Bundle übergeben und nicht
im Frontend fest eingebaut.

## 8. Übertragbarkeit auf die SGNS-Pipeline

Grundlage: Lektüre des Repos `pdl-lex/zeitenwende-sgns` (Stand Commit
`3e65c1b`, 2026-11-10); nichts davon ist dort gemessen. Die Pipeline trainiert
je Quelle und Zeitscheibe ein SGNS-Modell (gensim, 300 Dimensionen, Fenster 4,
`min_count` 5, `sample` 1e-4, 10 Epochen) auf dem laufenden Text und richtet
die Zeitscheiben nacheinander per orthogonalem Procrustes aus (Hamilton,
Leskovec & Jurafsky 2016b). Raster derzeit `quarter`.

**Kurzfassung:** Die Mechanismen (Abschnitt 7) und die Entscheidungsregeln
gelten auch für SGNS; die Zahlen und der größte Teil der Messtechnik nicht,
weil die Diagnose auf KWIC-Zeilen und *folding-in* beruht.

### 8.1 Was in SGNS anders ist

- **Keine Zeilen, sondern ein Modell je Zeitscheibe.** Ein Vektor entsteht
  aus dem Training auf dem ganzen Text der Scheibe. Einen Vektor aus h Belegen
  gibt es nur durch neues Training; jede Messung kostet Trainingsläufe statt
  Projektionen.
- **Zweite Rauschquelle: das Training.** Zufallsinitialisierung und Negative
  Sampling erzeugen schon auf demselben Korpus eine Drift (Hellrich & Hahn
  2016; Antoniak & Mimno 2018); Läufe mit mehreren Workern sind trotz Seed
  nicht reproduzierbar. Die Drift ohne Unterschied fällt deshalb auch bei sehr
  vielen Belegen nicht auf null.
- **Alignment statt gemeinsamem Raum.** Jede Scheibe wird auf die vorherige
  gedreht, Anker ist das ganze gemeinsame Vokabular einschließlich seltener,
  verrauschter Wörter. Das Alignment trägt eigenes, von der Scheibengröße
  abhängiges Rauschen bei.
- **Andere Kosinusskala.** Mittlere Ähnlichkeit zufälliger Wortpaare laut
  README ≈ 0,65. Driftwerte aus SGNS und PPMI sind nicht direkt vergleichbar;
  rangbasierte Kennwerte (AUC, Spearman) eher, r innerhalb der Wörter nur
  eingeschränkt.
- **Vermutlich höheres h\*.** SGNS braucht bei kleinen Daten und seltenen
  Wörtern eher mehr Belege als zählbasierte Modelle (Sahlgren & Lenci 2016).
- **Ziehungseinheit fehlt in den Korpusdateien.** `split.py` schreibt nur
  Sätze ohne `doc_id`. Halbierung, Permutation und Teilstichproben brauchen
  eine Variante, die auf den vorverarbeiteten Shards (mit `doc_id`) aufsetzt.
  Ziehen nach Sätzen wäre falsch, weil die Sätze einer Rede zusammenhängen.
- **Subsampling** (`sample` = 1e-4) dünnt häufige Wörter beim Training aus
  und gleicht so die Belegung häufiger Zielwörter teilweise schon an, seltener
  nicht. *Vermutung, nicht geprüft.*

### 8.2 Übertragbarkeit im Einzelnen

| Element | Übertragbar? | Umsetzung in SGNS |
|---|---|---|
| **Erkenntnisse** | | |
| Rauschen fällt mit der Belegzahl, ohne Knick | Mechanismus ja | neu messen; Untergrenze > 0 durch Trainingsrauschen |
| Volumeneffekt (schwach belegte Wörter mit systematisch anderer Richtung) | sehr wahrscheinlich | Häufigkeit ist in SGNS-Vektoren kodiert (Mu & Viswanath 2018) |
| Nichtadditivität; Abzug überkorrigiert | Mechanismus ja | gleiche Geometrie (Kosinus verrauschter Vektoren) |
| Über dem Rauschen ≠ Bedeutungswandel | ja | modellunabhängig |
| h\* = 30, k/p, beste Driftkorrektur | nein | neu messen |
| **Diagnosemethoden** | | |
| Abschnitt 1: a_zeit, a_wort | mit Aufwand | Reden einer Scheibe halbieren, zwei Modelle trainieren und ausrichten, AUC wie gehabt |
| Abschnitt 2: SVD ohne schwach belegte Zeilen | entfällt | nächstes Gegenstück `min_count` (Wort fällt ganz heraus, keine Projektion) |
| Abschnitt 3: k, p | entfällt | Gegenstück `vector_size`, `window`, `sample`, `epochs` |
| Abschnitt 3: Volumeneffekt | ja, ohne Neutraining | aus vorhandenen ausgerichteten Modellen und Häufigkeiten |
| Abschnitt 4: r innerhalb der Wörter, Resampling mit Zurücklegen, Spearman | ja, ohne Neutraining | Rohdrift benachbarter Scheiben = 1 − Nebendiagonale von `similarity_matrix` |
| Abschnitt 4: Permutationstest | ja, teuer | Original nach Dubossarsky et al. (2017), dort auch mit SGNS; Reden zweier Scheiben mischen, neu trainieren; ein Lauf liefert alle Zielwörter; eher 10–20 statt 99 Permutationen |
| Abschnitt 4: Sättigungskurve | ja, teuer | wie Permutation, zusätzlich Zielwörter auf h Belege ausdünnen |
| zusätzlich: Seed-Kontrolle | neu nötig | dieselbe Scheibe mit zwei Seeds trainieren → Drift allein durch Training |
| **Korrekturen** | | |
| Zeilen unter h\* markieren statt entfernen | ja | eigenes h\* |
| Drift nur für zuverlässige Paare | ja | min(freq_A, freq_B) ≥ h\*, freq = Vorkommen in der Scheibe |
| Teilstichprobe je Paar | möglich, teuer | überzählige Vorkommen des Zielworts in der stärkeren Scheibe durch Platzhalter ersetzen (Kontext bleibt); je Paar und Ziehung zwei Modelle |
| Teilstichprobe je Wort, Distanzen | günstiger | eine ausgedünnte Fassung aller Scheiben je Ziehung; hoher Belegverlust |
| Abzug der Permutationserwartung; gemittelte Vektoren | nicht einsetzen | Mechanismen sprechen dagegen, solange nicht anders gemessen |
| Zielwörter nach Abdeckung; Raster vergröbern | ja | Raster `quarter` ist dünner als unsere Halbjahre; `analyze_aligned` nutzt `year` |

### 8.3 Mögliche Reihenfolge

1. **Ohne Neutraining:** Häufigkeit je Zielwort und Scheibe, Rohdrift gegen
   log der kleineren Häufigkeit (r innerhalb der Wörter mit Intervall),
   Volumeneffekt. Zeigt schnell, ob dasselbe Muster auftritt.
2. **Seed-Kontrolle:** wenige Läufe; trennt Trainings- von Stichprobenrauschen.
3. **Split-half für h\* und Permutationen**, beide nach `doc_id` gezogen.
4. **Korrektur über Teilstichproben**, nur wenn Schritt 1 einen deutlichen
   Effekt zeigt.

Aufwand: Ein Quartal hat laut README etwa 750.000 Tokens; einzelne Trainings
sind also eher kurz. Laufzeiten sind nicht gemessen.

## 9. Einschränkungen der Messung

- r unterscheidet nicht zwischen Stichprobeneffekt und echtem Wandel, der mit
  der Trefferzahl einhergeht (über ein Wort wird oft gerade dann viel geredet,
  wenn sich sein Kontext ändert; Methodology 1.18). Nur r der
  Permutationserwartung enthält allein den Stichprobeneffekt.
- r misst eine lineare Abhängigkeit vom Logarithmus der Trefferzahl.
- Nur benachbarte Zeitscheiben; Abstände über mehrere Zeitscheiben und
  Matrizen aller Kombinationen sind nicht geprüft.
- Paare unter h\*, bei denen der Effekt am größten ist, gehen nicht ein; ihr
  Verhalten ist aus Abschnitt 1, der Sättigungskurve und dem Testlauf
  erschlossen, nicht im Permutationstest gemessen.
- Neu berechnete Zeilen werden in den Raum der vollständigen Daten projiziert,
  an dessen SVD die vollständigen Zeilen beteiligt sind.
- Gemeinsame Randsummen in der Permutationsrechnung: Unterschiede, die nur in
  den Randsummen liegen, gehen dort nicht ein, in die Rohdrift schon.
- p-Werte nicht für multiples Testen korrigiert.
- Die Erklärung über das Ungleichgewicht (5.3) ist nicht geprüft. Prüfbar
  wäre sie mit r der Rohdrift gegen log(größere/kleinere Trefferzahl).

## 10. Reproduktion

`stages/02_embedding/pipeline_emb_ppmi_diagnose.py`, Export
`pipeline_emb_ppmi_diagnose.html` (Lauf 2026-11-19). Parameter Abschnitt 4:
`DRIFT_FLOOR` = h\* (mindestens `DRIFT_FLOOR_MIN = 10`), `DRIFT_N_DRAWS = 20`,
`DRIFT_N_PERM = 99`, `DRIFT_N_BOOT = 1000`, Zufallszahlen aus `SEED`.
Abschnitt 4 verwendet den Raum der vollständigen Daten aus Abschnitt 2
(`full_counts`, `full_ppmi`, `full_V`, `full_s`).

## Literatur

- Antoniak, M. & Mimno, D. (2018): Evaluating the stability of embedding-based word similarities. *Transactions of the Association for Computational Linguistics* 6, 107–119.
- Bland, J. M. & Altman, D. G. (1995): Calculating correlation coefficients with repeated observations: Part 1 – correlation within subjects. *BMJ* 310(6977), 446.
- Davison, A. C. & Hinkley, D. V. (1997): *Bootstrap Methods and their Application*. Cambridge University Press.
- Dubossarsky, H., Weinshall, D. & Grossman, E. (2017): Outta control: Laws of semantic change and inherent biases in word representation models. In: *Proceedings of EMNLP 2017*, 1136–1145.
- Efron, B. & Tibshirani, R. J. (1993): *An Introduction to the Bootstrap*. Chapman & Hall.
- Good, P. (2005): *Permutation, Parametric and Bootstrap Tests of Hypotheses*. 3. Aufl. Springer.
- Hamilton, W. L., Leskovec, J. & Jurafsky, D. (2016b): Diachronic word embeddings reveal statistical laws of semantic change. In: *Proceedings of ACL 2016*, 1489–1501.
- Hellrich, J. & Hahn, U. (2016): Bad company – neighborhoods in neural embedding spaces considered harmful. In: *Proceedings of COLING 2016*, 2785–2796.
- Hosmer, D. W. & Lemeshow, S. (2000): *Applied Logistic Regression*. 2. Aufl. Wiley.
- Hurlbert, S. H. (1971): The nonconcept of species diversity: a critique and alternative parameters. *Ecology* 52(4), 577–586.
- Kutuzov, A., Øvrelid, L., Szymanski, T. & Velldal, E. (2018): Diachronic word embeddings and semantic shifts: a survey. In: *Proceedings of COLING 2018*, 1384–1397.
- Mahoney, M. W. & Drineas, P. (2009): CUR matrix decompositions for improved data analysis. *Proceedings of the National Academy of Sciences* 106(3), 697–702.
- Mu, J. & Viswanath, P. (2018): All-but-the-top: Simple and effective postprocessing for word representations. In: *Proceedings of ICLR 2018*.
- Phipson, B. & Smyth, G. K. (2010): Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology* 9(1), Art. 39.
- Sahlgren, M. & Lenci, A. (2016): The effects of data size and frequency range on distributional semantic models. In: *Proceedings of EMNLP 2016*, 975–980.
- Sanders, H. L. (1968): Marine benthic diversity: a comparative study. *The American Naturalist* 102(925), 243–282.
- Spearman, C. (1904): The proof and measurement of association between two things. *The American Journal of Psychology* 15(1), 72–101.

Die bibliografischen Angaben sind vor einer Veröffentlichung an den Originalen zu prüfen.
