# Readme Scientific Methodology

Stand: 2026-11-20

Dieses Dokument sammelt alle methodischen Entscheidungen der Pipeline an einem
Ort: welche Verfahren verwendet werden, welche getestet und verworfen wurden und
welche bekannt, aber nicht getestet sind. Es ersetzt
`literatur-belege-für-methoden.md` und `standardverfahren_gegen_hub-wörter.md`
und übernimmt die wissenschaftlichen Teile aus `nextsteps.md`. Technische
Umsetzungsschritte stehen weiterhin in `nextsteps.md`, Implementierungsdetails
und Stolpersteine im Code in `Readme Pipeline.md`.

## Aufbau

Jedes Verfahren hat denselben Aufbau:

- **Zweck**: ein Satz, was das Verfahren leisten soll.
- **Beschreibung**: wie es funktioniert und wie es bei uns umgesetzt ist.
- **Testergebnisse**: soweit vorhanden, mit Datum und Messreihe.
- **Fazit**: ob und mit welchen Parametern es verwendet wird, und warum.
- **Ausblick**: ob das Verfahren für das deutlich größere Zeitenwende-Korpus
  erneut geprüft werden sollte.
- **Literatur**: Kurzverweise. Die vollständigen Angaben stehen in der
  Bibliografie am Ende.

## Messreihen und Kennwerte

Die Testergebnisse stammen aus fünf Messreihen. Ihre Zahlen sind nicht
untereinander vergleichbar, weil sich Kontextdefinition, Lemmatisierung und
Parameter unterscheiden.

| Messreihe | Zeitraum | Werkzeug | Datengrundlage |
|---|---|---|---|
| **A** | 2026-09-01 bis 09-04 | `internal/pipeline.py`, `internal/pipeline-diagnose.py` (beide gelöscht) | ±40-Wörter-Fenster, spaCy, POS-Filter in der Matrix, k = 100, p = 1, Quartale (50 Zeitscheiben, 502 Zeilen) |
| **B** | 2026-10-01/02 | `internal/pipeline-diagnose.py`, Abschnitt „Gewichtungsvarianten“ (gelöscht) | Stufen-Pipeline, Quartale, Matrix ungefiltert |
| **C** | 2026-10-06 | `stages/02_embedding/pipeline_emb_ppmi_diagnose.py` (später `pipeline_emb_ppmi_diagnose_alt.py`, am 2026-11-20 gelöscht) | Stufen-Pipeline (Satzkontext, Stanza), Quartale, 287 Zeilen × 28.486 Kontextwörter, 40.211 Belege aus 15.505 Reden; Split-half-Prüfung mit 20 Teilungen |
| **D** | 2026-11-17 | `stages/02_embedding/pipeline_emb_ppmi_diagnose.py`, Abschnitte 1–3 | Ausgaben von pre vom 2026-11-11 (Satzkontext `punkt`, Stanza, vor der Begrenzung auf 20 Wörter), Halbjahre, 2.169 Zeilen × 66.661 Kontextwörter, 152.847 Belege aus 42.576 Reden; k = 100, p = 1 (Abschnitt 3: Raster aus k und p, Referenz k = 100, p = 1); Split-half-Prüfung mit 20 Teilungen |
| **E** | 2026-11-19 | `stages/02_embedding/pipeline_emb_ppmi_diagnose.py`, Abschnitte 1–4 (Export `pipeline_emb_ppmi_diagnose.html`) | Ausgaben von pre mit der vollen Zielwortliste, Satzkontext `punkt`, Halbjahre, 12.280 Zeilen × 120.882 Kontextwörter, 692.136 Belege aus 67.062 Reden, 596 Zielwörter; k = 300, p = 0,5 (Referenz in Abschnitt 3); Split-half-Prüfung mit 20 Teilungen. Ergebnisse zusammengefasst in `Readme Frequency Control.md` |

Die Messreihe C wiederholt die Prüfungen von B mit einem besseren
Messverfahren (s. 1.17). Bei CDS bestätigt sie B. Bei add-λ und bei k
weicht sie ab (s. 2.6, 1.8). Wo B und C sich widersprechen, ist das jeweils
vermerkt. E wiederholt D auf dem rund sechsmal größeren Datensatz und ergänzt
Abschnitt 4 (Drift); mehrere Befunde aus D bestätigen sich dort nicht
(s. 1.8, 1.18, 2.15, 3.12).

Häufig verwendete Kennwerte:

- **Volumeneffekt (r gepoolt)**: Korrelation zwischen der mittleren Ähnlichkeit
  einer Zeile zu den übrigen Zeitscheiben desselben Zielworts und dem
  Logarithmus der Textmenge dieser Zeile, über alle Zeilen. Ein hoher Wert
  bedeutet, dass Ähnlichkeit vor allem die Textmenge misst. Ein Betrag nahe 0
  ist besser; auch deutlich negative Werte kommen vor (s. 1.18).
- **Kontrollränge**: Rang bekannter Referenzzellen in der Ähnlichkeitsrangliste
  ihres Zielworts, z. B. `Sicherheit` 2022-Q1 und 2022-Q2 (echte Distanz
  erwartet, also hinterer Rang erwartet).
- **a_zeit, a_wort**: Unterscheidbarkeit der Zeitscheiben eines Worts bzw. der
  Zielwörter einer Zeitscheibe in der Split-half-Prüfung (s. 1.17).
- **Kontrast (SGNS-Pfad)**: Split-half-Ähnlichkeit eines Worts mit sich selbst
  minus seine mittlere Ähnlichkeit zu beliebigen anderen Wörtern (s. 1.20).

## Übergreifende Befunde

Vier Befunde ziehen sich durch viele Einzeltests. Sie sind hier vorangestellt,
weil sie bei jeder neuen Methodenentscheidung zu prüfen sind.

1. **Der seltene Schwanz der Häufigkeitsverteilung ist Signal, nicht Rauschen.**
   Eingriffe, die seltene Kontextwörter entfernen oder abwerten
   (Vokabularschwelle, Zeilennormierung, CDS), haben den Volumeneffekt jedes Mal
   stark erhöht. Eingriffe am häufigen Kopf (POS-Filter) waren folgenlos. Was
   eine Zeitscheibe thematisch unterscheidbar macht, sitzt im seltenen
   Vokabular. Beschreibt man eine Zeile nur über häufige Wörter, misst
   Ähnlichkeit vor allem die Stichprobengröße (Messreihe A, s. 2.1–2.3).
2. **Jede Regel, die Zeitstabilität misst, trifft zuerst das Neue.** Eine
   Mindestbelegung von 50 Zeilen entfernte `Sondervermögen`, ein festes
   Vokabular über alle Zeitscheiben entfernte `Zeitenwende`, eine
   Referenzscheibe für SGNS-Anker hätte `Taurus` und `Kriegstüchtigkeit`
   entfernt. In einem Projekt über Bedeutungswandel ist das Neue der Gegenstand.
   Bei jeder Schwelle zuerst prüfen, was sie mit dem Vokabular nach der Zäsur
   macht.
3. **Raum unverändert, Auswahl filtern.** Seit 2026-10-01 wird die Matrix nicht
   mehr gefiltert. Filter wirken nur bei der Auswahl von Nachbarn und Ankern,
   und seit 2026-10-05 sind sie Optionen im Frontend. Die Pipeline liefert die
   Attribute, der Nutzer filtert.
4. **Zufallserwartung nicht abziehen, Belegung angleichen.** Rauschen und
   echter Unterschied addieren sich bei Kosinusabständen nicht. Wer die volle
   Zufallserwartung abzieht, zieht bei schwach belegten Einheiten zu viel ab,
   und die Abhängigkeit von der Belegung kehrt sich um (Messreihe E, 1.23;
   dasselbe Muster im Wandel-Ranking, 1.21). Wirksam ist, die verglichenen
   Einheiten vor dem Vergleich auf gleiche Belegung zu ziehen. Und: „über der
   Zufallserwartung“ heißt nicht „Bedeutungswandel“ (1.23).

Zur Benennung: Für die beiden Arbeiten von Hamilton, Leskovec & Jurafsky aus
2016 folgt dieses Dokument Schlechtweg et al. (2019): **2016a** ist *Cultural
Shift or Linguistic Drift?* (EMNLP), **2016b** ist *Diachronic Word Embeddings
Reveal Statistical Laws of Semantic Change* (ACL). Die frühere Recherche zu
Hub-Wörtern hatte die Buchstaben vertauscht.

---

# 1. Verwendete Verfahren

## Vorverarbeitung

### 1.1 KWIC-Extraktion mit Satzkontext

**Zweck:** Für jedes Vorkommen eines Zielworts den umgebenden Text als Kontext
gewinnen, aus dem die Kookkurrenzen gezählt werden.

**Beschreibung:** Die Zielwörter werden wortweise exakt gesucht (kein
Teilstring, also keine Treffer in Komposita wie „Landesverteidigung“). Je
Vorkommen entsteht eine Zeile. Drei Modi stehen in
`stages/01_preprocessing/pipeline_pre_ppmi.py` zur Wahl:

- **Satzgrenzen mit NLTK Punkt** (Voreinstellung, `punkt`): Der Kontext ist der
  Satz, in dem das Zielwort steht. Punkt lernt per maschinellem Lernen,
  Satzenden von Abkürzungspunkten zu unterscheiden.
- **Satzgrenzen mit SoMaJo** (`somajo`): regelbasiert, für Deutsch optimiert,
  deutlich langsamer. Das verwendete CMC-Modell ist für Social-Media-Texte
  ausgelegt.
- **Festes Fenster** (`window40`): eine wählbare Zahl Wörter vor und nach dem
  Treffer. ±40 Wörter entsprechen der groben DNB-Vorgabe für die erlaubte
  Extraktionsmenge.

Messreihe A beruht auf dem ±40-Fenster, die Stufen-Pipeline (Messreihen B, C)
auf Satzkontext mit Punkt. Überlappende Fenster werden nicht dedupliziert.

**Testergebnisse:** Ein Qualitätsvergleich der drei Modi steht aus. Die Wahl
zwischen Satz und Fenster ist im PPMI-Pfad zugleich eine Frage der
DNB-Extraktionsgrenze, nicht nur der Methode.

**Fazit:** Verwendet wird Satzkontext mit NLTK Punkt. Die anderen Modi bleiben
als Option erhalten.

**Ausblick:** **Re-evaluieren.** Die Kontextdefinition beeinflusst alle
folgenden Schritte, und sie ist nie systematisch verglichen worden. Levy,
Goldberg & Dagan (2015) und Bullinaria & Levy (2012) finden für PPMI kleine
Fenster (±2) am besten, Schlechtweg et al. (2019) finden je nach Datensatz
Fenster von 2 bis 10 am besten. Für das Zeitenwende-Korpus ist außerdem zu
klären, welche Extraktionsgrenze die DNB tatsächlich zulässt.

**Literatur:** Levy, Goldberg & Dagan 2015; Bullinaria & Levy 2012;
Schlechtweg et al. 2019.

### 1.2 Lemmatisierung und POS-Tagging

**Zweck:** Flektierte Formen auf ihre Grundform zurückführen („Drohnen“ →
„Drohne“), damit Häufigkeiten nicht zersplittern, und die Wortart für spätere
Filter bestimmen.

**Beschreibung:** Zwei Werkzeuge stehen zur Verfügung: spaCy
(`de_core_news_md`) und Stanza (Verarbeitungsschritte tokenize, mwt, pos,
lemma; `mwt` zerlegt Verschmelzungen wie „am“ in „an dem“). Die Stufen-Pipeline
verwendet Stanza. Je Lemma wird die häufigste Wortart als dominante Wortart
geführt, weil ein Lemma mehrere Tags tragen kann („gut“ ist 4.068× ADV und
3.584× ADJ). Groß- und Kleinschreibung bleibt erhalten, weil die Anker über
exakte Lemma-Strings nachgeschlagen werden.

**Testergebnisse:** Kein systematischer Vergleich. Zwischenstand: spaCy ist
schneller, Stanza liefert bessere Lemmata. Bekannte spaCy-Fehler aus Messreihe
A: `Soldatinn` (aus „Soldatinnen“), `Einsatzkräft`, `Bewerberinn`,
`Zivilbeschäftigt`. Auch in der aktuellen Lemmaliste gibt es mehrdeutige oder
veraltete Lemmata (`treffen|triefen`, `angehören|anhören`, `Schluß`,
`Außetzung`).

**Fazit:** Verwendet wird Stanza. Lemmatisierung ist für das Deutsche wegen der
Flexionsvielfalt Standard (Turney & Pantel 2010).

**Ausblick:** **Stichprobenhaft re-evaluieren.** Ein größeres Korpus ändert an
der Wahl nichts Grundsätzliches. Die Fehlerquote sollte aber an einer
Stichprobe beziffert werden, weil Lemmafehler bei seltenen Wörtern (die
Signal tragen, s. Übergreifende Befunde) mehr Gewicht haben.

**Literatur:** Turney & Pantel 2010.

### 1.3 Zeitscheiben

**Zweck:** Die Belege in zeitliche Einheiten gliedern, zwischen denen Wandel
gemessen wird.

**Beschreibung:** Quartal (Voreinstellung), Halbjahr oder Jahr, aus dem
Sitzungsdatum gebildet. Die Labels („2022-Q1“) sind so gebaut, dass die
alphabetische Ordnung der chronologischen entspricht. Im Bundestagskorpus
ergeben Quartale 50 Zeitscheiben (2013-Q4 bis 2026-Q1).

**Testergebnisse:** Die dünnsten Quartale (2017-Q3/Q4, 2021-Q3/Q4) fallen auf
Bundestagswahlen mit langer Regierungsbildung. Das ist echtes Signal, kein
Defekt. Dünne Zeitscheiben landen in jeder getesteten Konfiguration am Ende der
Ähnlichkeitsrangliste (`Sicherheit` 2017-Q3 mit 14 Treffern: Rang 50 von 50,
Messreihe A). Gegen diese Textarmut hilft keine Gewichtung (Messreihe B). Der
SGNS-Pfad braucht mindestens Jahresscheiben (s. 1.20).

**Fazit:** Quartale als Voreinstellung, Halbjahr und Jahr als Alternativen. Die
Zeitscheibenbreite ist das einzige Mittel, das die Besetzung dünner Zellen
tatsächlich anhebt.

**Ausblick:** **Re-evaluieren.** Bei einem größeren Korpus sind feinere Raster
denkbar (Monate) oder Quartale werden durchgehend gut besetzt. Die Wahl hängt
an der Belegzahl je Zelle.

**Literatur:** –

### 1.4 Kookkurrenzzählung

**Zweck:** Für jedes Paar aus Zielwort und Kontextwort je Zeitscheibe zählen,
wie oft beide gemeinsam vorkommen.

**Beschreibung:** Die Kontextlisten werden zeilenweise entrollt und nach
Zeitscheibe, Zielwort und Kontextwort gezählt. Ein Kontextwort, das zweimal im
selben Kontext steht, zählt zweimal. Das Ergebnis ist eine Langformtabelle,
die bereits das Koordinatenformat einer dünn besetzten Matrix hat.

**Testergebnisse:** –

**Fazit:** Verwendet wie beschrieben.

**Ausblick:** Nicht re-evaluieren.

**Literatur:** Turney & Pantel 2010.

## Gewichtung

### 1.5 PPMI (Positive Pointwise Mutual Information)

**Zweck:** Rohzählungen in Assoziationsstärken umrechnen: Wie viel häufiger
treten Zielwort und Kontextwort gemeinsam auf, als bei Unabhängigkeit zu
erwarten wäre?

**Beschreibung:**

$$
\text{PPMI} = \max\left(0,\ \log_2 \frac{\text{beobachtet}}{\text{erwartet}}\right),
\qquad
\text{erwartet} = \frac{\text{Randsumme Zielwort} \cdot \text{Randsumme Kontextwort}}{\text{Gesamtsumme}}
$$

Alle Randsummen werden je Zeitscheibe berechnet. Werte unter 0 werden auf 0
gesetzt, weil negative PMI-Werte bei kleinen Zahlen unzuverlässig sind. Ob zwei
Wörter sich meiden, nie gemeinsam vorkommen oder unabhängig sind, ist danach
nicht mehr zu unterscheiden. Wir verwenden weder Context Distribution Smoothing
(α = 1, s. 2.5) noch einen Shift (k = 1, s. 3.1).

Eigenschaften, die das interaktive Beispiel in
`stages/02_embedding/pipeline_emb_ppmi.py` vorführt:

- Multipliziert man alle Zählungen mit demselben Faktor, bleibt PPMI gleich.
  Zeitscheiben unterschiedlicher Größe sind deshalb grundsätzlich vergleichbar,
  aber unterschiedlich verlässlich.
- Mehr gemeinsame Vorkommen heißt nicht automatisch höherer PPMI, weil auch die
  Randsummen wachsen.
- Seltene Wörter erzeugen mit wenigen Belegen extreme Werte (Church & Hanks
  1990; Turney & Pantel 2010).
- **PMI-Deckeneffekt:** Kommt ein Kontextwort in einer Zeitscheibe nur bei
  einem einzigen Zielwort vor, kürzt sich `beobachtet` aus dem Bruch heraus. Der
  Wert erreicht dann unabhängig von der Häufigkeit das Maximum
  `log2(Gesamtsumme / Randsumme Zielwort)`. 12 und 7 Treffer sind nicht mehr
  unterscheidbar.

PPMI misst die Stärke einer Assoziation. G² („Log-Likelihood“ in der
Korpuslinguistik) misst dagegen, wie gut belegt sie ist, weil es über alle vier
Zellen der Kontingenztafel mit der Häufigkeit gewichtet. Für die Vektorisierung
ist PPMI das etablierte Standardverfahren.

**Testergebnisse:**

- Der Deckeneffekt wirkt sich auf die SVD kaum aus. Hapax-Spitzen können keine
  Achse tragen, weil eine Achse ein Muster braucht, das mehrere Zeilen teilen.
  Belegt an `Bundeswehr` 2016-Q1: Die stärksten PPMI-Kontextwörter („gelassen“,
  „trennen“, „2008“, „worin“) spielen im SVD-Vektor keine Rolle, die dominante
  Richtung ist „ausgeben/Milliarde/Verteidigung/Rüstung“ (Messreihe A). Für die
  Anzeige-Ranglisten bleibt der Effekt bestehen; dagegen helfen `local_mi` und
  `t_score` (s. 1.6).
- PPMI im Vergleich zu `local_mi` als SVD-Eingabe: s. 2.7.

**Fazit:** Verwendet als Eingabe der SVD, mit log2, Randsummen je Zeitscheibe,
ohne Glättung und ohne Shift. Begründung: etabliertes Standardverfahren, als
Verhältnis unabhängig von der Größe der Zeitscheibe, theoretisch eng mit SGNS
verwandt (Levy & Goldberg 2014).

**Ausblick:** Das Verfahren selbst nicht re-evaluieren. Re-evaluieren sind
aber die Stellschrauben, die in 2.5, 2.6 und 3.1 verworfen bzw. nicht getestet
wurden: Ihre Wirkung hängt an der Dünne der Daten.

**Literatur:** Church & Hanks 1990; Turney & Pantel 2010; Levy & Goldberg 2014;
Levy, Goldberg & Dagan 2015; Evert 2005.

### 1.6 Local MI und t-Score als Anzeigespalten

**Zweck:** Die Trefferzahl wieder in die Bewertung bringen, damit exklusive
Kontextwörter in den Anzeige-Ranglisten unterscheidbar bleiben
(PMI-Deckeneffekt, s. 1.5).

**Beschreibung:**

- `local_mi = beobachtet × ppmi` (Evert 2005)
- `t_score = (beobachtet − erwartet) / √beobachtet`. √beobachtet ist der
  Standardfehler einer poissonverteilten Zählung. Der Wert wird bewusst nicht
  bei 0 gekappt, weil negative Werte („seltener als erwartet“) für die Anzeige
  informativ sind.

Beide sind reine Anzeige- und Diagnosespalten. Die SVD rechnet weiter auf PPMI.

**Testergebnisse:** s. 2.7 (Eignung als SVD-Eingabe).

**Fazit:** Verwendet für die Anzeige, nicht für die Vektoren.

**Ausblick:** Nicht re-evaluieren.

**Literatur:** Evert 2005; Church et al. 1991 für den t-Score. (Die früheren
Notizen nennen „Church & Hanks“; üblich zitiert wird für den t-Score Church,
Gale, Hanks & Hindle 1991. Vor einer Veröffentlichung prüfen.)

## Vektorraum

### 1.7 Gemeinsame Matrix für alle Zeitscheiben (ein Raum, kein Alignment)

**Zweck:** Alle Zeitscheiben im selben Koordinatensystem abbilden, damit ihre
Vektoren direkt vergleichbar sind.

**Beschreibung:** Die Zeilen der Matrix sind die Paare aus Zielwort und
Zeitscheibe (z. B. 11 Zielwörter × 50 Quartale), die Spalten die
Kontextwörter, gemeinsam über alle Zeitscheiben. Daraus wird **eine** SVD
berechnet. Ein nachträgliches Ausrichten der Räume entfällt. Die Kontextwörter
erhalten ihre Vektoren als Spaltenseite derselben Zerlegung (s. 1.9).

Das entspricht im Kern *Word Injection* (WI; Ferrari et al. 2017, bei
Schlechtweg et al. 2019) bzw. *Temporal Referencing* (Dubossarsky et al. 2019):
Das Zielwort bekommt je Zeitraum eine eigene Zeile, alles andere ist gemeinsam.
Die Alternative ist eine eigene SVD je Zeitscheibe mit anschließender
Ausrichtung (s. 3.3).

**Testergebnisse:** Kein eigener Test. Literatur:

- Schlechtweg et al. (2019), Tabelle 4: Im Mittel über SVD, RI und SGNS liegt
  WI (DURel ρ = 0,468, SURel 0,492) unter getrennten Räumen mit Orthogonal
  Procrustes (0,618 bzw. 0,590). Mean-Centering verbessert WI um 3 %.
- Dubossarsky et al. (2019) finden das Gegenteil: Der gemeinsame Raum ist
  besser. Ihre Ergebnisse beruhen allerdings auf synthetischen Daten.
- Ein Nachteil des gemeinsamen Raums: Zeitscheiben mit viel Text prägen die
  Achsen stärker. Das ist im Kern unser Volumeneffekt.

**Fazit:** Verwendet. Gründe: Bei 50 Zeitscheiben wären 49 Ausrichtungen nötig,
viele Quartale wären für eine eigene SVD zu dünn, und der gemeinsame Raum ist
deterministisch und alignmentfrei, wie `specs.md` §2 es festlegt. Dünne Zeilen
leihen sich in der gemeinsamen SVD Struktur von den übrigen Zeilen. Das ist der
Grund, warum der PPMI-Pfad mit Wörtern wie `Hindukusch` arbeiten kann, an
denen SGNS scheitert (s. 1.20).

**Ausblick:** **Re-evaluieren, gemeinsam mit 3.3 und 3.5.** Wird das Korpus so
groß, dass einzelne Zeitscheiben eine eigene SVD tragen, ist der Vergleich mit
getrennten Räumen und Orthogonal Procrustes möglich und nach Schlechtweg et al.
(2019) geboten.

**Literatur:** Schlechtweg et al. 2019; Ferrari et al. 2017; Dubossarsky et al.
2019; Hamilton et al. 2016b.

### 1.8 Abgeschnittene SVD mit Eigenwertgewichtung (k = 300, p = 0,5)

**Zweck:** Die sehr breite, dünn besetzte PPMI-Matrix auf wenige Dimensionen
verdichten und dabei Rauschen entfernen.

**Beschreibung:** Die PPMI-Matrix $M$ wird zerlegt in $M = U \Sigma V^\top$.
Behalten werden nur die $k$ größten Singulärwerte (Deerwester et al. 1990).
Nach Eckart & Young (1936) ist das die beste Approximation mit Rang $k$. Der
Zeilenvektor ist

$$
x = U_k\, \Sigma_k^{\,p}.
$$

Der Exponent $p$ (*eigenvalue weighting*) bestimmt, wie stark die starken
Dimensionen gewichtet werden: $p = 1$ mit dem Singulärwert, $p = 0{,}5$ mit
seiner Wurzel, $p = 0$ alle gleich. Der Anteil der erhaltenen Frobenius-Masse
wird in `svd_meta.json` mitgeschrieben. Er ist kein „erklärter Varianzanteil“,
weil die PPMI-Matrix nicht zentriert ist.

*Berechnung (seit 2026-11-17):* Die Matrix wird dünn besetzt gespeichert (CSR),
und es werden nur die $k$ größten Singulärwerte und -vektoren berechnet
(abgeschnittene SVD mit ARPACK, implizit neu gestartetes Lanczos-Verfahren;
Lehoucq, Sorensen & Yang 1998; `scipy.sparse.linalg.svds`). Vorher wurde die
vollständige SVD der dichten Matrix berechnet und auf $k$ gekürzt; beides
liefert dieselbe Rang-$k$-Approximation. Der Startvektor ist fest, das
Vorzeichen jeder Achse einheitlich gewählt (betragsgrößter Eintrag in $V^\top$
positiv), damit Läufe reproduzierbar sind. Die Frobenius-Masse im Nenner wird
aus den Einträgen der Matrix berechnet. Grund ist der Speicherbedarf: Die
dichte Matrix hat Zeilen × Kontextwörter Einträge (Bundestag, punkt/halfyear:
2.169 × 66.661, 578 MB bei 0,55 % besetzten Zellen; dünn besetzt 9 MB) und
passt beim Zeitungskorpus nicht mehr in den Arbeitsspeicher.

**Testergebnisse:**

- **Vergleich abgeschnittene gegen vollständige SVD** (2026-11-17, Bundestag,
  punkt/halfyear, k = 300, p = 0,5): erhaltene Frobenius-Masse identisch
  (39,4 %); größte Abweichung der Kosinuswerte Zielwort × Kontextwort
  2,7 · 10⁻⁷, Zielwort × Zielwort 2,2 · 10⁻⁷ (Größenordnung der
  float32-Rundung); Top-150-Nachbarlisten zu 97,2 % in identischer Reihenfolge,
  mittlere Überlappung 99,98 % (die Abweichungen sind Vertauschungen bei
  praktisch gleichem Kosinus); 139 von 300 Achsen mit umgekehrtem Vorzeichen
  (ohne Einfluss auf Kosinus und Nachbarn). Laufzeit von emb insgesamt rund
  8 Minuten gegenüber über 40 Minuten ohne Abschluss mit der vollständigen SVD
  (gleicher Rechner, gleiche Daten).
- **Messreihe A:** Das Singulärwertspektrum ist flach (Rang 1: 319,6, Rang 100:
  72,2, Rang 400: 42,4). k = 100 hielt nur 40,9 % der Frobenius-Masse, und die
  Nachbarschaftsprofile korrelierten mit denen des vollen Raums nur zu 0,69. p
  ändert die Rangfolge der Nachbarn kaum, wohl aber das Niveau der Kosinuswerte
  (für denselben Nachbarn p = 0: 0,64; p = 0,5: 0,75; p = 1: 0,86).
- **Messreihe B (2026-10-01/02):** k von 10 bis 100 und p ∈ {1; 0,5; 0},
  geprüft an Volumeneffekt, zeitlicher Ordnung (sind benachbarte Quartale
  ähnlicher als ferne?), Trennschärfe (ist der nächste Nachbar ein Quartal
  desselben Worts?) und Referenzfällen (`Sicherheit` 2017-Q3, 2022-Q2). Von
  k = 100 abwärts verbessern sich alle vier Kriterien. Unterhalb von k ≈ 20
  kehrt sich das um: Bei k = 10 geht die zeitliche Ordnung verloren, und das
  Vorzeichen des Volumeneffekts je Wort kehrt sich um. p = 0,5 senkt den
  Volumeneffekt bei kleinem k zusätzlich.

  | | k = 100, p = 1 | k = 25, p = 0,5 |
  |---|---|---|
  | r gepoolt | 0,409 | 0,236 |
  | r je Wort | −0,380 | −0,094 |
  | zeitliche Ordnung | −0,141 | −0,219 |
  | Trennschärfe | 0,868 | 0,934 |
  | Ähnlichkeitsniveau | ≈ 0,37 | ≈ 0,63 |

  `Sicherheit` 2022-Q2 bleibt weit hinten, 2017-Q3 in jeder Variante ganz
  hinten.
- **Messreihe C (2026-10-06):** Split-half-Prüfung, k von 5 bis 250,
  p ∈ {1; 0,5; 0}. Die SVD wird an einer Hälfte berechnet, die andere per
  *folding-in* projiziert. Referenz ist der damalige Stand von emb, k = 95
  (Regel: ein Drittel der Zeilenzahl, 30 bis 100) mit p = 1.

  | k | a_zeit p = 1 | a_zeit p = 0,5 | a_zeit p = 0 | a_wort p = 1 | a_wort p = 0,5 | a_wort p = 0 | Frobenius-Anteil |
  |---|---|---|---|---|---|---|---|
  | 5 | 0,707 | 0,708 | 0,707 | 0,980 | 0,981 | 0,981 | 13 % |
  | 10 | 0,710 | 0,715 | 0,716 | 0,972 | 0,974 | 0,974 | 17 % |
  | 25 | 0,689 | **0,704** | 0,713 | 0,944 | **0,953** | 0,955 | 26 % |
  | 50 | 0,680 | 0,712 | 0,730 | 0,919 | 0,935 | 0,937 | – |
  | 95 (Ref.) | **0,704** | 0,747 | 0,764 | **0,898** | 0,920 | 0,918 | 59 % |
  | 150 | 0,739 | 0,778 | 0,788 | 0,888 | 0,909 | 0,900 | 77 % |
  | 200 | 0,763 | **0,802** | 0,811 | 0,886 | **0,902** | 0,886 | 89 % |
  | 250 | 0,796 | 0,816 | 0,813 | 0,892 | 0,895 | 0,868 | 97 % |
  | PPMI ohne SVD | 0,799 | | | 0,880 | | | 100 % |

  Befunde:
  - **a_wort fällt mit wachsendem k durchgehend, a_zeit fällt bis k ≈ 30–50
    und steigt danach deutlich.** Kleines k erhält also die Identität der
    Zielwörter, zieht aber ihre Zeitscheiben zusammen. Das ist genau das
    Muster, das das Notebook als „Zielkonflikt“ vorhersagt.
  - **p < 1 ist bei fast jedem k besser für a_zeit**, p = 0,5 auch für a_wort.
  - **Unsere Einstellung k = 25, p = 0,5** ist gegenüber der Referenz ein
    Zielkonflikt: a_zeit gleich (0,704), a_wort höher (0,953 gegen 0,898).
  - **Empfehlung des Notebooks: k = 200, p = 0,5** (einzige Kombination mit
    „Verbesserung“ und größtem a_zeit-Anstieg: +0,097 über alle Zeilen,
    +0,167 bei hoher Textmenge, +0,085 bei mittlerer). k = 200 erreicht
    a_zeit der PPMI-Vektoren ohne SVD (0,799); den höchsten Wert hat
    k = 250, p = 0,5 mit 0,816, eingestuft als Zielkonflikt. Von 41 Varianten
    sind 24 eine Verbesserung, 15 ein Zielkonflikt.

- **Messreihe D (2026-11-17):** Diagnose-Notebook von emb, Abschnitt 3,
  punkt/halfyear (2.169 Zeilen). Split-half-Prüfung mit 20 Teilungen wie in C,
  dazu der Volumeneffekt (s. 1.18) je Teilung an der Hälfte, an der die SVD
  berechnet wird. Die SVD wird über die Gram-Matrix berechnet (alle k aus
  einer Eigenzerlegung); eine Kontrollzelle prüft die Übereinstimmung mit
  der SVD von emb. Referenz ist der damalige Stand von emb, k = 100, p = 1.
  Rang einer Hälfte mindestens 1.960, alle k bis 1.500 berechenbar.

  (Die Gesamttabelle dieses Laufs ist hier nicht erhalten; die Kernwerte
  stehen in den Befunden. r: Volumeneffekt, gepoolt, mit Vorzeichen.)

  Befunde:
  - **a_zeit steigt mit k bis k ≈ 700 und fällt danach**, a_wort ist bei
    k ≈ 50 am höchsten und fällt danach durchgehend. p = 0,5 ist für a_zeit
    bei jedem k ab 50 am besten, für a_wort bei jedem k bis 500. Das
    bestätigt C: kleines k trennt die Zielwörter, großes k die Zeitscheiben.
  - **Der Volumeneffekt hängt stark an k.** Bei der Referenz ist r = 0,30,
    bei k = 25 sogar 0,65. Mit wachsendem k sinkt er, geht bei k ≈ 200–300
    (p = 0,5) durch null und wird bei größerem k und p < 1 deutlich negativ
    (bis −0,44). Auch ein negativer Wert ist eine Abhängigkeit von der
    Textmenge; bewertet wird deshalb der Betrag (s. 1.18). Ohne SVD ist r =
    0,68: Die Kürzung baut den Volumeneffekt ab.
  - **Ab k ≈ 150 übertrifft die SVD die PPMI-Vektoren ohne Kürzung** in
    a_zeit (0,727) und bei jedem k in a_wort (0,868).
  - **Keine Kombination ist eine Pareto-Verbesserung** gegenüber der
    Referenz (34 von 35 Zielkonflikt, eine Verschlechterung). Der Grund ist
    durchgängig: Jede Kombination mit p = 0,5 senkt a_zeit im mittleren
    Terzil der Textmenge (Zeilen mit mittlerer Trefferzahl) um 0,013 bis
    0,020, in allen 20 Teilungen, während a_zeit im unteren und oberen
    Terzil steigt (oben um 0,03 bis 0,09).
  - Gegenüber der Referenz (Mittel über alle Zeilen; Volumeneffekt als
    Betrag):

    | | Δ a_zeit | Δ a_wort | \|r\| | Δ a_zeit mittleres Terzil |
    |---|---|---|---|---|
    | k = 100, p = 0,5 | +0,009 | +0,003 | 0,28 | −0,018 |
    | k = 200, p = 0,5 | +0,017 | −0,001 | 0,06 | −0,020 |
    | k = 300, p = 0,5 | +0,024 | −0,005 | 0,06 | −0,017 |
    | k = 700, p = 0,5 | +0,034 | −0,021 | 0,25 | −0,013 |

- **Messreihe E (2026-11-19):** Abschnitt 3 auf 12.280 Zeilen, Referenz
  k = 300, p = 0,5 (Auszug, p = 0,5; volle Tabellen im Export):

  | k | 10 | 25 | 100 | 300 | 1.000 | 1.500 |
  |---|---|---|---|---|---|---|
  | a_zeit | 0,692 | 0,700 | 0,685 | 0,682 | 0,685 | 0,687 |
  | a_wort | 0,924 | 0,938 | 0,925 | 0,908 | 0,879 | 0,868 |
  | Volumeneffekt r | 0,581 | 0,688 | 0,693 | 0,527 | 0,286 | 0,188 |
  | Anteil quadrierte Frobenius-Norm | 5 % | 6 % | 10 % | 18 % | 38 % | 50 % |

  Befunde:
  - **a_zeit ist über k nahezu flach** (0,68–0,70), anders als in D, wo es
    mit k bis ≈ 700 stieg. a_wort fällt wie in C und D mit wachsendem k.
  - **p wirkt auf a_zeit und a_wort kaum** (Unterschiede ≤ 0,02), stark aber
    auf den Volumeneffekt: Bei k = 300 r = 0,672 (p = 1), 0,527 (p = 0,5),
    0,450 (p = 0).
  - **Der Volumeneffekt ist durchgehend positiv** (0,026 bis 0,771). Die
    negativen Werte aus D treten nicht auf (s. 1.18).
  - **Alle 35 Kombinationen sind ein Zielkonflikt** gegenüber der Referenz;
    Empfehlung des Notebooks: Referenz beibehalten. Pareto-Front-Beispiele:
    k = 1.500, p = 0 senkt den Volumeneffekt auf 0,026, a_wort fällt auf
    0,849; k = 25, p = 0,5 hebt a_zeit auf 0,700 und a_wort auf 0,938, der
    Volumeneffekt steigt auf 0,688.
  - Der Anteil der quadrierten Frobenius-Norm ist bei k = 300 mit 18 %
    niedriger als in D (39,4 % bei 2.169 Zeilen), weil die Matrix viel größer
    ist. Er ist kein Gütemaß.

- **Widerspruch zwischen B und C.** B fand, dass die zeitliche Ordnung mit
  sinkendem k besser wird; C findet, dass die Unterscheidbarkeit der
  Zeitscheiben mit sinkendem k schlechter wird. Beides kann zugleich stimmen,
  weil die Kennwerte Verschiedenes messen: „zeitliche Ordnung“ fragt, ob
  benachbarte Quartale ähnlicher sind als ferne, a_zeit fragt, ob die
  Eigenart eines Quartals in unabhängigen Hälften reproduzierbar ist. Eine
  starke Kürzung kann die Zeitscheiben eines Worts glätten und dadurch die
  Ordnung verbessern, aber ihre Eigenart verwischen. Die Trennschärfe aus B
  entspricht a_wort, dort stimmen B und C überein. Den Volumeneffekt misst C
  nicht; ob er bei k = 200 wieder steigt, ist offen (in B lag er bei k = 100
  bei 0,409 gegen 0,236 bei k = 25). B arbeitet außerdem ohne getrennte
  Hälften, die Achsen sind also an dieselben Daten angepasst, an denen
  bewertet wird.

**Fazit:** Verwendet: **k = 300, p = 0,5** für punkt/halfyear (Entscheidung
2026-11-17 auf Grundlage von D; emb stand zuvor auf k = 100, p = 1). Da keine
Kombination eine Pareto-Verbesserung ist, ist die Wahl eine Abwägung. Für ein
Werkzeug über Bedeutungswandel hat a_zeit Vorrang. k = 300, p = 0,5 erhöht
a_zeit deutlich (+0,024), kostet a_wort nur wenig (−0,005) und senkt den
Volumeneffekt fast auf null (r von 0,30 auf −0,06). Größeres k erhöht a_zeit
kaum noch, senkt a_wort spürbar und kehrt den Volumeneffekt ins Negative.
k = 200, p = 0,5 wäre die vorsichtigere Wahl (a_wort unverändert, a_zeit
+0,017). In Kauf genommen ist der Rückgang von a_zeit im mittleren Terzil der
Textmenge (−0,017); er tritt bei jeder Kombination mit p = 0,5 auf und ist
nicht geklärt. p = 0,5 ist in B, C und D bestätigt und entspricht der
Empfehlung von Levy, Goldberg & Dagan (2015).
**Messreihe E bestätigt die Wahl** (keine Kombination besser in allen
Kennwerten), begründet sie aber anders: a_zeit hängt in E kaum an k und p;
k = 300, p = 0,5 ist ein Kompromiss zwischen a_wort (kleines k besser) und
Volumeneffekt (großes k, kleines p besser). Die Aussage „senkt den
Volumeneffekt fast auf null“ gilt nur für D; in E liegt er bei 0,53.
Zum Vergleich: Schlechtweg et al. (2019) verwenden d = 300 und p = 0, übernommen
aus der Literatur und nicht getestet.

**Ausblick:** **Re-evaluieren** bei jedem neuen Lauf von pre, anderem Raster,
anderer KWIC-Variante und neuem Korpus (Abschnitt 3 des Diagnose-Notebooks,
etwa 17 Minuten für alle Abschnitte). Der günstige Rang hängt an der
Zeilenzahl und an der Datenmenge je Zeile. D beruht auf den Ausgaben von pre
vom 2026-11-11, also vor der Begrenzung langer Sätze auf ±20 Tokens
(2026-11-17); nach einem neuen Lauf von pre wiederholen. Offen: warum p < 1
a_zeit im mittleren Terzil der Textmenge senkt.

**Literatur:** Deerwester et al. 1990; Eckart & Young 1936; Caron 2001;
Bullinaria & Levy 2012; Levy, Goldberg & Dagan 2015; Österlund, Ödling &
Sahlgren 2015; Schlechtweg et al. 2019.

### 1.9 Kontextwortvektoren aus der Spaltenseite der SVD

**Zweck:** Jedem Kontextwort einen Vektor im selben Raum geben wie den Zeilen,
ohne etwas neu zu rechnen. Daraus entstehen Anker und Nachbarn.

**Beschreibung:** Die Kontextwortvektoren sind `Vt[:k].T`, bewusst ohne
Gewichtung mit Σ. Bei p = 1 rekonstruiert das Skalarprodukt aus Zeilen- und
Kontextvektor genau die PPMI-Zelle. Ein zweites Σ würde die starken Achsen im
Skalarprodukt quadratisch gewichten und die Richtung der Kontextvektoren
ändern.

Diese Vektoren sind dasselbe wie eine Projektion der Einheitszeile des Worts in
den Raum: `e_j @ V[:, :k] = v_j`. Ein Anker als „eigenes Zielwort“ mit eigener
Kookkurrenzzeile wäre also derselbe Mechanismus mit reichhaltigerer Zeile.

**Testergebnisse:** Hat ein Kontextwort nur in einer einzigen Zeile i einen
Eintrag, ist sein Vektor `v_j = M_ij · S⁻¹ · u_i`. Seine Richtung hängt allein
an Zeile i. Alle solchen Wörter derselben Zeile haben denselben Vektor und
denselben Kosinus (in `Bundeswehr` 2016-Q1 sieben Wörter mit exakt
0,6539999842643738, Messreihe A). Sie sind keine Nachbarn, sondern die Zeile
selbst, zurückgespiegelt. Gegenmittel ist die Mindestbelegung (s. 1.11).

**Fazit:** Verwendet, ungewichtet.

**Offene Frage:** Seit p = 0,5 rekonstruiert das Produkt aus Zeilen- und
Kontextvektor nicht mehr die PPMI-Zelle. Soll die Kontextseite ebenfalls mit
S^p gewichtet werden (`Vt[:k].T * S[:k] ** eig_p`)? Das betrifft den Kosinus
zwischen Zeilen und Kontextwörtern und damit die Nachbarn. Nicht getestet. Der
Code in `pipeline_emb_ppmi.py` verweist auf diese Frage.

**Ausblick:** **Re-evaluieren**, und zwar zuerst die offene Frage, unabhängig
vom Korpus.

**Literatur:** Levy, Goldberg & Dagan 2015 (Gewichtung von Wort- und
Kontextvektoren).

## Nachbarn, Anker, Darstellung

### 1.10 Nachbarn nach rohem Kosinus

**Zweck:** Für jede Zeile (Zielwort × Zeitscheibe) die Kontextwörter finden,
die ihr im Raum am nächsten sind.

**Beschreibung:** Zeilen- und Kontextvektoren werden auf Länge 1 normiert, das
Skalarprodukt ist dann der Kosinus. Die Nachbarn werden nach rohem Kosinus
gerankt, ohne Korrektur. Im Bundle stehen je Zeile 150 vorberechnete Nachbarn
mit den Voreinstellungen der Filter (s. 1.11–1.14). Das Backend berechnet die
Nachbarn künftig bei jeder Abfrage aus den Vektoren, mit k_NN = 1000.

Die Nachbarn des PPMI-Pfads sind ein **Kollokationsprofil**: Kontextwörter,
die an derselben Stelle im Raum liegen wie die Zeile. Die Nachbarn des
SGNS-Pfads sind **Wortnachbarn**: Wörter mit ähnlichem Gebrauch. Die beiden
Listen sind nicht vergleichbar (s. 2.13).

**Testergebnisse:**

- **Hubness-Diagnose** (`pipeline_post_diagnose.py`, Abschnitt 1, s. 1.19):
  Bei kurzen Listen (k_NN = 10) keine Hubness, bei langen Listen ja.
- **CSLS und Mutual Proximity** beseitigen die Hubs nicht bzw. liefern
  gleichwertige oder etwas schlechtere Listen (inhaltliche Durchsicht
  2026-10-05; s. 2.10, 2.11).
- Eine simulierte Backend-Abfrage mit den Voreinstellungen reproduziert die
  vorberechneten Nachbarn (285 von 287 Listen identisch, 2 nur in der
  Reihenfolge fast gleicher Werte).

**Fazit:** Verwendet: roher Kosinus, keine automatische Hubness-Korrektur.

**Ausblick:** **Re-evaluieren.** Hubness hängt von der Dimensionalität und von
der Zahl der Abfragen und Kandidaten ab. Beides ändert sich mit dem Korpus. Die
Diagnose aus 1.19 erneut laufen lassen.

**Literatur:** Radovanović, Nanopoulos & Ivanović 2010; Conneau et al. 2018;
Schnitzer et al. 2012; Gonen et al. 2020.

### 1.11 Mindestbelegung der Nachbar- und Ankerkandidaten

**Zweck:** Kontextwörter ausschließen, deren Vektor nur auf einer oder sehr
wenigen Zeilen beruht und deshalb die Zeile spiegelt (s. 1.9).

**Beschreibung:** Ein Kontextwort muss in mindestens `neighbor_min_rows` Zeilen
einen PPMI-Wert > 0 haben. Voreinstellung: max(10; 2 % der Zeilenzahl). Gezählt
werden nur Zellen mit PPMI > 0, nicht alle Kookkurrenzen. Die Matrix bleibt
unverändert. Die Schwelle ist im Frontend änderbar.

**Testergebnisse (Messreihe A):** Der erste Versuch mit 50 Zellen („ein Zehntel
der 502 Zeilen“) entfernte `Sondervermögen`: Das Wort gibt es erst seit 2022,
es kann höchstens 16 von 50 Quartalen belegen und hat 47 Zellen mit PPMI > 0.
`Zeitenwende` hatte 59 Zellen (knapp drin), `Attraktivität` exakt 50. Der
Spiegeleffekt sitzt dagegen bei genau einer Zelle; dagegen genügen 10.
Kontrollzeichen einer zu niedrigen Schwelle sind identische Kosinuswerte in
einer Nachbarliste.

**Fazit:** Verwendet, mit niedriger Schwelle. Die Mindestbelegung misst
Zeitstabilität, nicht Qualität (s. Übergreifende Befunde, Punkt 2).

**Ausblick:** **Re-evaluieren.** Die Schwelle skaliert mit der Zeilenzahl. Mit
mehr Zeitscheiben oder Zielwörtern prüfen, dass Wörter, die erst nach der Zäsur
auftreten, nicht herausfallen.

**Literatur:** Church & Hanks 1990; Levy, Goldberg & Dagan 2015 (PMI
überschätzt seltene Ereignisse).

### 1.12 Wortart-Filter der Kandidaten

**Zweck:** Nur Inhaltswörter als Nachbarn und Anker anbieten.

**Beschreibung:** Zugelassen sind NOUN, PROPN, ADJ und VERB, bezogen auf die
dominante Wortart je Lemma. Die Matrix bleibt ungefiltert (s. 2.4).

**Testergebnisse (Messreihe A):** Ohne Filter war die erste Ankerliste voll
von `immer`, `so`, `noch`, `dann`, `nur`, `auch`. ADV ist nach dem damaligen
Matrix-Filter mit 21,4 % der zweitgrößte Block nach NOUN (39,9 %), vor VERB
(18,0 %) und ADJ (10,6 %). Für die Matrix ist das folgenlos, für eine Auswahl
nach Kosinus nicht, weil breit gestreute Wörter durchschlagen.

**Fazit:** Verwendet als Voreinstellung, im Frontend änderbar.

**Ausblick:** Nicht re-evaluieren.

**Literatur:** Turney & Pantel 2010 (POS-Tagging und Stoppwortfilter als
übliche Vorverarbeitung, ohne konkrete Vorschrift). Das DWDS-Wortprofil
(Didakowski & Geyken 2014) und die Sketch Engine (Kilgarriff et al. 2004)
arbeiten mit syntaktischen Relationen zwischen Inhaltswörtern; Funktionswörter
fallen dort durch die Konstruktion heraus, nicht durch einen Filter.

### 1.13 Rednernamen

**Zweck:** Namen von Rednerinnen und Rednern aus Nachbarn und Ankern
ausblenden, weil sie Sprecher markieren, nicht Bedeutung.

**Beschreibung:** Exakte Liste der Personen aus den Korpusmetadaten
(`speaker_names.csv` aus pre), mehrteilige Namen in einzelne Tokens zerlegt,
angewandt nur auf Lemmata mit Wortart PROPN. Genauer als NER und ohne weiteren
NLP-Lauf.

**Testergebnisse:** –. Ein Beispiel für das Problem aus Messreihe A:
`Hindukusch` wird von `Struck` und `Peter` dominiert (Zitat „Deutschlands
Sicherheit wird auch am Hindukusch verteidigt“).

**Fazit:** Verwendet als Voreinstellung, im Frontend abschaltbar.

**Ausblick:** Nicht re-evaluieren, aber für das Zeitenwende-Korpus prüfen,
welche Metadaten eine solche Liste liefern. Bei Zeitungstexten gibt es keine
Rednerliste; dann wären Autorennamen oder NER zu erwägen.

**Literatur:** Keine direkte Quelle; die Maßnahme ist aus dem Korpus begründet.
Am nächsten kommen Hamilton et al. (2016), die Eigennamen filtern (welche
der beiden Arbeiten, ist in den Notizen nicht festgehalten). Gonen et
al. (2020) halten Namen dagegen für eine Quelle von Artefakten in
ausgerichteten Räumen.

### 1.14 Domänen-Stoppwörter

**Zweck:** Anreden, parlamentarische Formeln und Allerweltsverben ohne
spezifische Bedeutung aus Nachbarn und Ankern ausblenden.

**Beschreibung:** Mitgelieferte Liste
`stages/03_postprocessing/input/stopwords.txt` (vorläufig 16 Anreden und
Formeln, 10 häufige Verben; „Frau“ bewusst nicht). Im Frontend bearbeitet der
Nutzer eine Kopie. Allgemeine Stoppwortlisten (NLTK) werden nicht verwendet.

**Testergebnisse:** Kein Test der Liste selbst. Zur Wirkung allgemeiner
Stoppwörter auf die Matrix s. 2.4.

**Fazit:** Verwendet als Voreinstellung. Die Liste wird überarbeitet.

**Ausblick:** **Neu erstellen.** Die Liste ist korpusspezifisch. Für ein
Zeitungskorpus sind die Parlamentsformeln gegenstandslos, dafür kommen andere
Formeln hinzu (Datelines, Agenturkürzel, Bildunterschriften).

**Literatur:** Che Abdul Rahman et al. (Stoppwortliste für den malaysischen
Hansard mit Anreden und Vermerken wie „Lachen“ und „Beifall“; aus Korpuslinguistik
und Text-Mining, nicht aus Distributionssemantik); arXiv:2012.02294
(domänenspezifische Stoppwortlisten allgemein); Gonen et al. 2020 (entfernen
Stoppwörter und häufigste Wörter auf Ebene der Nachbarkandidaten); Bullinaria &
Levy 2012 (s. 2.4).

### 1.15 Anker als kuratierte Maximalliste

**Zweck:** Feste, beschriftete Orientierungspunkte im 3D-Raum, an denen der
Betrachter die Bewegung der Zielwörter liest.

**Beschreibung:** `stages/03_postprocessing/input/anchors.csv`
(`lemma,label,cluster`), unabhängig von den Parametern. Mehrere
Lemma-Varianten dürfen dasselbe Label tragen; gewählt wird die mit der größten
Belegung. `label` repariert Anzeigefehler der Lemmatisierung, ohne das Wort zu
verwerfen. Anker sind keine Messergebnisse, sondern Interpretationshilfen: Sie
legen fest, in welchem Vokabular die Bewegung gelesen wird. Diese Setzung soll
sichtbar von einem Menschen kommen (Verfahren „hybrid“, 2026-09-01: Die
Pipeline liefert Kandidaten, die Auswahl ist fachlich).

**Testergebnisse (Messreihe A):** Die Kandidaten-Ranglisten über „Reichweite“
und „Kontrast“ sind verworfen (s. 2.9). Ein Anker als eigenes Zielwort mit
eigenem KWIC-Lauf (Weg b) ist nicht nötig: Er ist derselbe Mechanismus wie die
Spaltenseite (s. 1.9), würde aber die Matrix ändern und alle Messreihen
unvergleichbar machen.

**Fazit:** Verwendet. 86 Anker in 6 Clustern; mehr als ~60 beschriftete
Punkte überlagern sich, mehr als ~7 Cluster-Farben sind nicht unterscheidbar.

**Ausblick:** **Neu kuratieren.** Die Liste ist korpus- und
fragestellungsspezifisch. Für die Kandidaten ist Keyness vorgesehen (s. 3.9).

**Literatur:** –

### 1.16 PCA auf drei Dimensionen

**Zweck:** Renderkoordinaten für die 3D-Ansicht. Der Ähnlichkeitsraum ist die
SVD, die PCA liefert nur die Darstellung.

**Beschreibung:** Die Vektoren werden auf Länge 1 normiert, sonst würde die
erste Achse die Längenunterschiede abbilden, die aus der Textmenge stammen.
Danach werden sie zentriert und zerlegt. Gefittet wird nur auf den Aggregaten
(Zielwort × Zeitscheibe). Anker und alle übrigen Kontextwörter werden mit
denselben Achsen projiziert, ohne neuen Fit. Diese Längennormierung ist nicht
zu verwechseln mit der verworfenen Zeilennormierung vor der SVD (s. 2.1).

**Testergebnisse:** Bis 2026-10-05 bestimmten die Anker die PCA-Achsen mit.
Dann hingen die Achsen davon ab, wie viele Anker ein Lauf trifft (s. 2.12). Der
3D-Raum erklärt nur einen Teil der Struktur (SGNS-Bundle: 36,7 % in-sample);
Nähe in 3D und Nähe im vollen Raum sind nicht direkt vergleichbar.

**Fazit:** Verwendet, Fit nur auf den Aggregaten.

**Ausblick:** Nicht re-evaluieren. Den erklärten Varianzanteil im Bundle
weiter mitschreiben.

**Literatur:** –

## Evaluation und Diagnose

### 1.17 Split-half-Prüfung (a_zeit, a_wort)

**Zweck:** Varianten des Verfahrens daran messen, wie gut sie Zeilen
unabhängig von der zufälligen Auswahl der Texte reproduzieren und zugleich
voneinander unterscheiden.

**Beschreibung:** Die Reden jeder Zeitscheibe werden zufällig in zwei Hälften
geteilt (nach Reden, nicht nach Belegen, weil Belege derselben Rede nicht
unabhängig sind). Aus jeder Hälfte entsteht eine eigene Matrix. Für jede Zeile
wird die Ähnlichkeit ihrer Hälfte A zu ihrer Hälfte B mit der Ähnlichkeit zu
Vergleichszeilen verglichen:

- **a_zeit**: Vergleichszeilen sind die anderen Zeitscheiben desselben Worts.
- **a_wort**: Vergleichszeilen sind die anderen Zielwörter derselben
  Zeitscheibe.

Beide Werte sind AUC-Werte (0,5 = Zufall, 1 = perfekt) und beruhen auf Rängen,
nicht auf Kosinusdifferenzen, weil Varianten die Verteilung der Kosinuswerte
insgesamt verschieben. 20 Teilungen, Seed 42. Eine Änderung gilt als
belastbar, wenn sie in allen Teilungen dasselbe Vorzeichen hat. Empfohlen wird
nur eine Pareto-Verbesserung. Für die SVD wird die SVD an einer Hälfte
berechnet und die andere per *folding-in* projiziert, weil eine SVD der
vollständigen Daten die Hälften künstlich ähnlich machen würde.

**Testergebnisse:** Messreihe C (2026-10-06), Ergebnisse in 1.8, 2.5 und
2.6. Laufzeit des ganzen Notebooks: unter einer Minute. Messreihen D und E
mit dem Diagnose-Notebook von emb (1.8, 2.15, 3.12; E zusätzlich 1.23).

**Grenzen:** Gemessen wird Reliabilität, nicht Validität. a_zeit erfasst jede
reproduzierbare Eigenschaft einer Zeitscheibe, also auch Themen und
Textmenge. Wechselwirkungen zwischen Glättung und SVD sind nicht geprüft.

**Fazit:** Verwendet als Standard-Messverfahren der Embedding-Stufe. Löst die
Messlatte aus 1.18 ab.

**Ausblick:** **Auf dem neuen Korpus als Erstes laufen lassen.** Das Verfahren
selbst nicht ändern, aber alle Entscheidungen der Embedding-Stufe damit neu
prüfen.

**Literatur:** Spearman 1910; Brown 1910; Antoniak & Mimno 2018; Hanley &
McNeil 1982; Mann & Whitney 1947; Haxby et al. 2001; Hastie, Tibshirani &
Friedman 2009; Deerwester et al. 1990; Berry, Dumais & O'Brien 1995.

### 1.18 Volumeneffekt-Messlatte und Kontrollränge

**Zweck:** Prüfen, ob die Ähnlichkeit einer Zeile vor allem ihre Textmenge
abbildet statt ihre Bedeutung.

**Beschreibung:** Je Zeile die mittlere Ähnlichkeit zu den übrigen
Zeitscheiben desselben Worts, korreliert mit log(Tokens), gepoolt und je Wort.
Dazu die Ränge bekannter Referenzzellen. Strukturkennzahlen werden auf der
rohen Matrix berechnet, damit sie über Varianten vergleichbar bleiben.

**Testergebnisse:**

- Die ursprüngliche Diagnose r = 0,606 reproduzierte nicht (Messreihe A:
  gepoolt 0,208, `Sicherheit` 0,336). 8 von 11 Zielwörtern zeigten eine
  negative Korrelation: Viel Volumen geht dort mit großer Distanz einher. Über
  ein Wort wird genau dann viel geredet, wenn sich sein Kontext ändert.
- Die Zahl der belegten Spalten korreliert je Wort mit r = 0,89–0,97 mit
  log(Tokens). Die Asymmetrie ist real; das Problem ist unbezifferte
  Unsicherheit dünner Zellen, nicht systematische Verzerrung.
- Auf den Daten der Stufen-Pipeline liegt der Volumeneffekt höher (0,40 statt
  0,21). Warum, ist nicht geklärt.

- **Messreihe D (2026-11-17, s. 1.8):** Gepoolt, je Teilung an der Hälfte
  berechnet, an der die SVD berechnet wird. Der Wert hängt stark an k (k = 25:
  0,65; k = 100: 0,30; k = 300, p = 0,5: −0,06) und wird bei großem k mit
  p < 1 deutlich negativ (bis −0,44). Ein negativer Wert ist ebenso eine
  Abhängigkeit von der Textmenge (gut belegte Zeilen sind den übrigen
  Zeitscheiben ihres Worts allein deswegen unähnlicher). Bewertet wird
  deshalb seit D der Betrag |r|; „niedriger ist besser“ galt nur, solange
  alle Werte positiv waren.
- **Messreihe E (2026-11-19, s. 1.8):** Auf 12.280 Zeilen ist der
  Volumeneffekt bei allen 36 Kombinationen positiv (0,026 bis 0,771), bei der
  Referenz k = 300, p = 0,5 r = 0,527. Er sinkt mit wachsendem k und kleinerem
  p. Die negativen Werte aus D wiederholen sich nicht; das Vorzeichen hängt
  also an den Daten. Dass gut belegte Zeilen den übrigen Zeitscheiben ihres
  Worts ähnlicher sind, betrifft vor allem den Vergleich *zwischen* Wörtern
  (gepoolt); innerhalb der Wörter misst 1.23 die Abhängigkeit der Drift.

**Fazit:** Verwendet in Messreihe A und B, in D als Kontrollgröße neben der
Split-half-Prüfung (1.17), bewertet als Betrag.

**Offene Frage:** Dünne Zellen (z. B. `Sicherheit` 2017-Q3) lassen sich über
Gewichtung nicht retten. Vorgesehen ist eine Markierung der Zeilen unter
h\* im Frontend (s. 3.12) und für die Drift eine Teilstichprobe je Paar
(s. 1.23).

**Ausblick:** **Re-evaluieren.** Der Volumeneffekt hängt direkt an der
Verteilung der Textmengen über die Zeitscheiben.

**Literatur:** Dubossarsky, Weinshall & Grossman 2017 (Frequenzeffekte als
Artefakt).

### 1.19 Hubness-Diagnose

**Zweck:** Prüfen, ob einzelne Kandidaten in auffällig vielen Nachbarlisten
vorkommen, nicht wegen ihrer Bedeutung, sondern weil sie vielen Abfragen nahe
sind.

**Beschreibung:** Für jede Listenlänge k_NN ∈ {10, 50, 150, 1000} wird die
k-Occurrence N(c) jedes Kandidaten gezählt und ihre Schiefe berechnet. Weil
Abfragen (Zeilen) und Kandidaten (Kontextwörter) verschiedene Mengen sind, wird
die Schiefe mit einer Binomialreferenz verglichen. Zusätzlich wird die
**Zielwortbreite** gezählt: in den Listen wie vieler verschiedener Zielwörter
ein Kandidat vorkommt. Hubness liegt vor, wenn die Schiefe über der Referenz
liegt **und** mehr als die Hälfte der 20 häufigsten Kandidaten in den Listen
aller Zielwörter vorkommt. Die Schwellen 20 und „mehr als die Hälfte“ sind
eigene Festlegungen.

**Testergebnisse:** s. 1.10. Grenzen: Die Zielwörter stammen aus einem
Wortfeld, ein zielwortübergreifender Kandidat kann also auch eine gemeinsame
Bedeutung tragen. Das trennt erst die inhaltliche Durchsicht.

**Fazit:** Verwendet als Diagnose (`pipeline_post_diagnose.py`, Abschnitt 1).

**Ausblick:** **Erneut laufen lassen** (s. 1.10).

**Literatur:** Radovanović, Nanopoulos & Ivanović 2010; Dinu, Lazaridou &
Baroni 2015; Johnson, Kemp & Kotz 2005; Feldbauer & Flexer 2019.

## Kontrollverfahren

### 1.20 SGNS mit Orthogonal Procrustes (Kontrollpfad)

**Zweck:** Ein zweites, unabhängiges Verfahren mit anderen blinden Flecken,
um die Befunde des PPMI-Pfads zu kontrollieren.

**Beschreibung:** Skip-Gram with Negative Sampling (word2vec; gensim
`Word2Vec(sg=1, negative=5, vector_size=100, window=5, epochs=5, workers=1)`)
auf dem Volltext aller Reden, ein Modell je Jahr (2014–2025). Die Modelle sind
nur bis auf eine Rotation bestimmt und werden per Orthogonal Procrustes
ausgerichtet: $R = UV^\top$ aus der SVD von $A^\top B$, geschätzt aus den 5.000
häufigsten gemeinsamen Wörtern (Stützwörter), angewandt auf alle Vektoren. Die
Rotation erfindet keine Ähnlichkeit, sie dreht nur (geprüft:
`max|cos_vorher − cos_nachher| = 6,6 · 10⁻⁷`). Den Pfad baut Fabian weiter;
Details in `Readme Pipeline.md`, Abschnitt „SGNS-Pfad“.

**Testergebnisse (`internal/pipeline-sgns.py`, 2026-09-04 bis 09-11):**

- **Korpusgröße:** Der KWIC-Auszug (3,24 Mio. Token, 6,4 % des Korpus) reicht
  für SGNS nicht. Volltext: ~4,4 Mio. Token je Jahr. Quartale (1,1 Mio.) fallen
  aus, nur Jahresscheiben sind realistisch.
- **Rauschböden:** Seed gegen Seed bei identischen Daten: Median-Kosinus 0,953;
  Split-half innerhalb 2022: 0,873; 2016 → 2022: 0,764. Das Verfahren misst
  also etwas. Die Hälfte einer Top-10-Nachbarliste hängt am Seed (Überlappung
  0,41–0,65, je nach Häufigkeit), wie bei Antoniak & Mimno (2018).
- **Kontrast statt Selbstähnlichkeit:** Seltene Wörter sind sich selbst
  ähnlich, weil sie allem ähnlich sind (Split-half-Kosinus n = 2–3: 0,941;
  n ≥ 301: 0,839). Erst Kontrast = Selbstähnlichkeit − Ähnlichkeit zu
  beliebigen Wörtern steigt monoton mit der Häufigkeit (0,113 → 0,360) und
  taugt als Auflösungsmaß. Eine Mindesttrefferzahl wird dadurch überflüssig.
- **Drift gegen das eigene Häufigkeitsband:** Drift ist über die
  Häufigkeitsbänder u-förmig. Sie ist deshalb als Perzentil im eigenen
  Häufigkeitsband zu lesen, mit Kontrast als Gütemaß daneben.
- **Zäsurbefund:** Über 5 Seeds hat `Verteidigung` in 4 und `Bundeswehr` in 3
  Seeds den stärksten Übergang bei 2021 → 2022, `Frieden` und `Sicherheit` in
  keinem (P(≥3 von 5) = 0,006; P(≥4 von 5) = 0,0003 gegen eine gemessene, flache
  Nullverteilung). Die beiden Wörter haben sich an der Zäsur stärker verändert
  als ~80 % vergleichbar häufiger Wörter.
- **Nur 4 von 11 Zielwörtern sind auflösbar** (`Bundeswehr`, `Frieden`,
  `Sicherheit`, `Verteidigung`). 49 % der Zellen haben Kontrast < 0,20.
  `Rüstung` ist trotz 5/5 Seeds ein Artefakt des dünnen Jahres 2021 (4
  Vorkommen). Seed-Stabilität allein schützt nicht gegen Artefakte; erst
  Kontrast-Schranke und Seed-Stabilität zusammen ergeben eine Aussage.
- **Ausrichtungsregime:** Paarweise direkte Ausrichtung für Driftkurven (Fehler
  der Verkettung über eine Referenz: Median 0,013, max 0,12). Für `coords_3d`
  ist die gemeinsame Referenz (2019) unvermeidlich; die Driftkurve im Frontend
  hat dadurch eine eingebaute U-Form.
- **Nachbarlisten beider Pfade:** s. 2.13.

**Fazit:** Verwendet als Kontrollverfahren, nicht als Hauptverfahren. Drift
wird als Perzentil im Häufigkeitsband mit Kontrast als Gütemaß ausgewiesen,
nie als roher Kosinus allein. `workers=1` für Reproduzierbarkeit.

**Offene Fragen:** Seed-Mittelung (Antoniak & Mimno 2018 empfehlen sie,
Hamilton et al. 2016b mitteln nicht); Downsampling auf gleiche Tokenmenge je
Jahr (Faktor 1,8 zwischen 2017 und 2023); `min_count`; `vector_size = 100` ist
übernommen, nicht begründet.

**Ausblick:** **Re-evaluieren, mit Vorrang.** Das Zeitenwende-Korpus ist genau
der Fall, für den SGNS gebaut ist: viel Text je Zeitscheibe. Feinere
Zeitscheiben könnten möglich werden, und mehr Zielwörter könnten auflösbar
sein. Alle Parameter, Downsampling und Seed-Mittelung müssen vor einem
DNB-Vor-Ort-Einsatz feststehen, weil Training dort nicht beliebig wiederholt
werden kann.

**Literatur:** Mikolov et al. 2013a, 2013b; Hamilton et al. 2016b; Antoniak &
Mimno 2018; Schlechtweg et al. 2019; Levy & Goldberg 2014.

## Wandelmaß

### 1.21 Wandel-Ranking: Nachbarüberlappung und Kosinus mit Permutations-Baseline

**Zweck:** Unter vielen Zielwörtern (geplant 500–1000) diejenigen finden, deren
Umfeld oder Lage im Raum sich zwischen zwei Zeiträumen A und B stärker
verändert hat, als die Schwankung zwischen den Zeitscheiben erwarten lässt. Das Ranking dient der
Auswahl; *wie* sich ein Wort verändert hat, zeigen die übrigen Ansichten
(Wortraum, Drift, Matrizen) für die Auswahl.

**Beschreibung:** Umgesetzt im Frontend (`pdl-lt-vsv`,
`frontend/src/viz/overlap.ts`, Modul „Wandel-Ranking“), nicht in post, weil
die Nachbarlisten erst nach dem Nachbar-Filter des Nutzers feststehen
(Übergreifender Befund 3).

- **Maß:** Schnittmenge der Top-k-Nachbarlisten nach Gonen et al. (2020),
  `overlap@k = |NN_k(A) ∩ NN_k(B)| / k`, Voreinstellung k = 100. Geteilt wird
  durch die kürzere Liste, falls eine Liste kürzer als k ist. Kein Vergleich
  von Vektoren über Zeitscheiben hinweg, daher ohne Ausrichtung anwendbar und
  im PPMI- wie im SGNS-Pfad gleich zu lesen (bei PPMI Kollokationsprofil, bei
  SGNS Wortnachbarn).
- **Zusammenlegen je Zeitraum:** Gonen et al. trainieren je Korpus ein Modell;
  hier liegen Listen je Zeitscheibe vor. Sie werden per Rangpunkten (Borda)
  zu einer Liste je Zeitraum zusammengelegt: jede Liste auf k gekürzt, Rang r
  (ab 0) erhält k − r Punkte, summiert über die belegten Scheiben. Jede Scheibe
  hat gleiches Gewicht.
- **Permutations-Baseline:** Die belegten Scheiben von A und B werden ohne
  Rücksicht auf ihren Zeitraum auf zwei Gruppen derselben Größe wie A und B
  verteilt, beide zusammengelegt und verglichen; alle Aufteilungen, falls es
  höchstens 64 sind, sonst 64 feste Zufallsaufteilungen. Der Mittelwert ist
  die Überlappung, die ohne Unterschied zwischen A und B bei derselben Tiefe
  des Zusammenlegens zu erwarten ist.
- **Sortierung nach dem Abstand** `Baseline − overlap@k`, absteigend, nicht
  nach der rohen Überlappung (seit 2026-11-17). Die rohe Überlappung hängt von
  der Belegung ab: dünn belegte Wörter haben schon innerhalb eines Zeitraums
  instabile Listen und stünden oben, ohne dass sich etwas verschoben hätte.
  Die Baseline rechnet das je Wort heraus. Das ist die Absicherung gegen die
  Belegung, die das Ranking für eine große Zielwortliste braucht.
- **p-Wert:** Anteil der Aufteilungen mit höchstens so viel Überlappung wie
  A ↔ B, als (b + 1)/(m + 1) (Phipson & Smyth 2010). Die kleinste erreichbare
  Zahl hängt von der Zahl der Aufteilungen ab (64: 1/65; 3 + 3 Scheiben:
  C(6,3) = 20, also 1/21). Keine Korrektur für multiples Testen: bei 600
  Zielwörtern sind bei α = 0,05 rund 30 zufällig kleine p-Werte zu erwarten.
  Der p-Wert ist deshalb Zusatzinformation, der Abstand der Sortierschlüssel.
- **Zweites Maß: Kosinus-Abstand der Mittelvektoren** (seit 2026-11-17),
  `1 − cos(v̄_A, v̄_B)`, das klassische Maß für Wandel im Raum (Hamilton et
  al. 2016b), hier für zwei Zeiträume statt zwei Zeitscheiben. Jeder
  Scheibenvektor wird vor dem Mitteln auf Länge 1 gebracht, damit jede
  belegte Scheibe gleiches Gewicht hat, wie beim Zusammenlegen der Listen.
  Gerechnet auf `aggregates.vector`, nur ohne volle Vektoren auf `coords_3d`
  (dann in der Ansicht vermerkt). Baseline, Abstand (`1 − cos` minus
  Baseline) und p wie bei der Überlappung, mit denselben Aufteilungen, aber
  umgekehrter Richtung (p = Anteil der Aufteilungen mit mindestens so großem
  Abstand). Sortiert wird wahlweise nach dem Abstand der Überlappung oder des
  Kosinus; beide Ränge stehen in jeder Zeile.
- **Einseitig belegte Wörter:** Wörter ohne Beleg in A oder B haben keine
  Überlappung. Sie stehen getrennt, nach Belegen sortiert, neu aufgekommene
  (nur in B) zuerst (Übergreifender Befund 2).
- **Binnen-Überlappung** (Split-Half innerhalb eines Zeitraums) nur
  beschreibend; als Referenz für A ↔ B ungeeignet, weil die Hälften nur halb
  so viele Scheiben zusammenlegen und deshalb systematisch tiefer liegen.
- **Ausgabe:** Tabelle mit Nachbar-Diff je Wort (gemeinsame, weggefallene,
  hinzugekommene Nachbarn) und CSV-Export (`rank`, `rank_overlap`,
  `rank_cos`, `overlap`, `baseline`, `gap`, `p`, `cos_distance`,
  `cos_baseline`, `cos_gap`, `cos_p`, Belegung und Belege je Zeitraum) zur
  Eingrenzung der Zielwortliste.

**Testergebnisse:** Probelauf 2026-11-17 auf `bundestag_sgns_year.json`
(SGNS-Pfad, 14 Zielwörter), A = 2014–2019, B = 2020–2025, k = 100, ohne
Nachbar-Filter:

- Nach roher Überlappung stand `Wehrpflicht` auf Rang 2 (0,07). Die Baseline
  liegt bei 0,05 (Abstand −0,02, p = 0,92): kein Wandel über die Schwankung
  hinaus. Nach dem Abstand Rang 12 von 12.
- `Krieg` und `Bundeswehr` rücken von Rang 10 und 11 auf 3 und 6 (Abstand
  0,13 bzw. 0,11, p = 0,031): hohe Überlappung, aber stabile Listen, daher
  deutlicher Abstand.
- `Zeitenwende` hat den größten Abstand (0,22), aber p = 0,33: in A nur 10
  Belege in wenigen Jahren, daher wenige Aufteilungen. Faktisch ein in B neu
  aufgekommenes Wort.
- `Wehrhaftigkeit`, `Kriegstüchtigkeit`: nur in B belegt.
- Kosinus: dieselben ersten zwei (`Zeitenwende`, `Sondervermögen`);
  `Wehrpflicht` Rang 10 von 12 (Abstand 0,025, p = 0,15).

Probelauf 2026-11-17 auf `bundestag_zeitenwende_halfyear.json` (PPMI-Pfad,
104 Zielwörter, 100 in beiden Zeiträumen belegt), A = 2013-H2–2021-H2,
B = 2022-H1–2026-H1, k = 100, ohne Nachbar-Filter:

- **Beide Maße stimmen weitgehend überein:** Spearman ρ = 0,80 zwischen den
  Rängen; gemeinsam unter den ersten 10: 9, den ersten 20: 14, den ersten 30:
  20. Oben stehen bei beiden `Zeitenwende`, `Held`, `Angriffskrieg`,
  `Sondervermögen`, `Bündnis`, `Gas`, `Armee`.
- **Abweichungen** vor allem bei häufigen, allgemeinen Wörtern, die der
  Kosinus höher einstuft als die Überlappung: `Bundeswehr` (Kosinus 18,
  Überlappung 82), `Ausweitung` (38 / 77), `Gewalt` (43 / 76), `Opfer`
  (48 / 74), `Milliarde` (15 / 49). Umgekehrt `Atombombe` (96 / 40),
  `Front` (49 / 10), `Ukraine-Krise` (61 / 16). Hamilton et al. (2016a)
  erwarten getrennte Ausschläge (Lage gegen Umfeld); hier nicht weiter geprüft.
- **p trennt beim Kosinus kaum:** 66 von 100 Wörtern mit p < 0,05 (Überlappung:
  41), viele am kleinsten erreichbaren Wert 1/65. Die Baseline des Kosinus ist
  leicht zu übertreffen; für die Rangfolge zählt der Abstand.
- **Der Abstand hängt mit der Belegung zusammen:** Spearman zwischen Abstand
  und log der kleineren Belegzahl (min(freq_A, freq_B)) 0,42 (Kosinus) bzw.
  0,23 (Überlappung); die Rohwerte dagegen −0,08 bzw. −0,06. Gut belegte
  Wörter haben eine niedrige Baseline und damit mehr Raum für einen Abstand.
  Der Abstand misst also *nachweisbaren* Wandel; bei schwach belegten Wörtern
  kann ein echter Wandel im Rauschen untergehen. Für die Eingrenzung heißt
  das: schwach belegte Wörter mit großem Rohwert gesondert ansehen, nicht nur
  nach dem Abstand abschneiden.

**Grenzen:** Gegen Annotationen nicht validiert (s. 3.13). Gonen et al.
berichten auf DURel ρ = 0,59 (Procrustes + Kosinus 0,814), auf SURel 0,859
(0,800). Die Baseline mischt Zeitscheiben, nicht Sätze; sie ist damit keine
Kontrollbedingung im Sinne von 3.10 und schützt nicht gegen Effekte, die alle
Scheiben eines Zeitraums gemeinsam betreffen (etwa eine geänderte
Häufigkeit). Gonen et al. entfernen häufige und seltene Wörter aus den
Listen; hier übernimmt das der Nachbar-Filter des Nutzers. Die Wahl von k
hängt von der Listenlänge im Bundle ab (150; über das Backend bis 1000).
Beim Kosinus gelten zusätzlich die Vorbehalte des gemeinsamen Raums: der
Volumeneffekt (1.18, PC1 folgt der Belegung) und im SGNS-Bundle die
Ausrichtung aller Jahre auf die Referenz 2019 (1.20), die Abstände um 2019
verkleinert. Die Baseline fängt den Volumeneffekt nur je Wort ab, nicht
zwischen Wörtern (s. Testergebnisse). **Der Abzug der Baseline überkorrigiert
vermutlich:** In der Diagnose der Drift (1.23) kehrt der Abzug der
Zufallserwartung die Abhängigkeit von der Belegung um (r von −0,26 auf
+0,30), weil sich Rauschen und Unterschied nicht addieren. Der positive
Zusammenhang zwischen Abstand und Belegung im Probelauf (0,42 bzw. 0,23) passt
zu diesem Muster. Für das Ranking nicht eigens geprüft.

**Fazit:** Verwendet als Rangliste zur Auswahl von Zielwörtern, mit zwei
Maßen nebeneinander. Voreinstellung ist die Sortierung nach der Überlappung
(nicht abhängig von Ausrichtung und Volumeneffekt im Raum); der Kosinus dient
als Gegenprobe. Wörter, die nach beiden Maßen oben stehen, sind die
belastbarsten Kandidaten.

**Ausblick:** **Auf dem neuen Korpus mit der vollen Zielwortliste laufen
lassen** und die Liste anhand des CSV eingrenzen. Den Einfluss von k und
der Nachbar-Filter auf die Rangfolge prüfen. Die Abweichungen zwischen den
Maßen (häufige, allgemeine Wörter höher beim Kosinus) an einigen Wörtern im
Nachbar-Diff und in den Driftkurven ansehen. Ein Kriterium für schwach
belegte Wörter mit großem Rohwert finden (s. Testergebnisse). Prüfen, ob
ein Angleichen der Belegung beider Zeiträume vor dem Vergleich den Abzug
ersetzen kann (s. 1.23, Übergreifender Befund 4).

**Literatur:** Gonen et al. 2020; Hamilton et al. 2016a, 2016b; Phipson &
Smyth 2010; Dwork et al. 2001; Schlechtweg et al. 2019.

## Facetten

### 1.22 Facetten-Zellen (records) per *folding-in*, Randsummen je Teilkorpus

**Zweck:** Zielwörter je Facettenwert (Fraktion, Publikation) im selben Raum
wie die Aggregate abbilden, damit sich die Verläufe verschiedener Fraktionen
oder Publikationen vergleichen lassen.

**Beschreibung:** Eine Facetten-Zelle ist ein Zielwort in einer Zeitscheibe
bei einem Facettenwert. Die Kookkurrenzen werden in pre zusätzlich je
Facettenwert gezählt; die Zählung je Aggregat ist ihre Summe. Treffer ohne
Facettenwert bilden den eigenen Wert „nicht zugeordnet“, damit beide
Zählungen dieselben Treffer enthalten.

- *Raum:* Die Facetten-Zellen gehen nicht in die SVD ein. Ihre PPMI-Zeilen
  werden per *folding-in* projiziert, $m V_k \Sigma_k^{p-1}$, mit demselben
  k und p wie die Aggregate (vgl. 2.15). Für eine Zeile der Zerlegung ist das
  gleich $U_k \Sigma_k^p$; emb prüft das an den Aggregaten. Achsen, PCA und
  Koordinaten der Aggregate ändern sich dadurch nicht (1.7, 1.16). Eine
  eigene Zerlegung je Facettenwert würde Räume erzeugen, die erst ausgerichtet
  werden müssten (vgl. 3.3).
- *Randsummen der PPMI:* Voreinstellung `RECORD_MARGINALS = "facet"`.
  Spaltensummen und Gesamtsumme werden je Zeitscheibe und Facettenwert
  berechnet, also im Teilkorpus etwa einer Publikation in einem Halbjahr. Das
  überträgt die bestehende Regel, nach der jede Zeitscheibe ihre eigenen
  Randsummen hat, auf die Facette; Hamilton et al. (2016b) berechnen PPMI
  ebenso je Zeitraum getrennt. Ein Kontextwort, das eine Publikation
  durchgehend häufig verwendet, erhält dort keinen hohen PPMI-Wert allein
  wegen seiner Häufigkeit. Unterschiede zwischen Facettenwerten betreffen
  dann die Verwendung des Zielworts, nicht das Gesamtvokabular des
  Teilkorpus. Gewählt, weil die Fragestellung Unterschiede im Wandel zwischen
  den Publikationen ist (Entscheidung 2026-11-17). Alternative `"slice"`:
  Randsummen der ganzen Zeitscheibe, nur die Zeilensumme stammt aus der
  Facetten-Zelle; ein record misst dann die Abweichung vom Gesamtbild der
  Zeitscheibe, einschließlich des Hausvokabulars.
- *Vokabular:* Spalten sind die Kontextwörter der Aggregat-Matrix.
  Kontextwörter ohne positiven PPMI-Wert in einem Aggregat fallen bei den
  records weg. Facetten-Zellen ohne positive PPMI-Spalte haben keinen Vektor;
  ihre Zahl steht in `emb_meta` (`n_records_without_vector`).
- *Dünne Zellen:* Alle Zellen werden geliefert, `freq` (= `n_hits`) trägt die
  Belegung (vgl. 3.12). Mit `"facet"` beruhen auch die Randsummen auf dem
  kleineren Teilkorpus; schwach belegte Teilkorpora ergeben daher unsichere
  PPMI-Werte (vgl. 1.5).
- *Bundle:* records mit `coords_3d`, `freq` und `facets`, ohne `vector`
  (Bundle-Größe); `facets_schema` aus der Deklaration in pre.

**Testergebnisse:** Kein Methodenvergleich. Technische Prüfung (Bundestag,
punkt/halfyear, Fraktion als Facette, 2026-11-17): Die Summe der
Facetten-Zählungen ist identisch mit der Aggregat-Zählung; *folding-in* der
Aggregat-Zeilen reproduziert die SVD-Vektoren (größte relative Abweichung
1,5 · 10⁻¹⁰). 9.923 records aus 9 Facettenwerten (darunter „nicht zugeordnet“
mit 14.042 Treffern), alle mit Vektor; die Summe ihrer `freq` ist gleich der
der Aggregate (152.847 Treffer).

**Fazit:** Verwendet, Randsummen je Teilkorpus (`"facet"`), `"slice"` als
Schalter erhalten.

**Ausblick:** **Je Korpus prüfen.** Ob Unterschiede zwischen Facettenwerten
über dem Rauschen liegen, ist offen: analog zur Split-half-Prüfung (1.17)
die Treffer je Facetten-Zelle teilen und die Abweichung zwischen den Hälften
mit der Abweichung zwischen Facettenwerten vergleichen; dabei beide
Randsummen-Varianten und den Einfluss der Belegung (1.18) messen.

**Literatur:** Deerwester et al. 1990; Berry, Dumais & O'Brien 1995;
Hamilton et al. 2016b; Levy, Goldberg & Dagan 2015.

## Frequenzkontrolle der Drift

### 1.23 Drift benachbarter Zeitscheiben mit Teilstichprobe je Paar

**Zweck:** Die Drift eines Zielworts zwischen zwei benachbarten Zeitscheiben
so messen, dass sie nicht von der Belegung der beiden Zeilen abhängt.

**Beschreibung:** Drift ist die Kosinusdistanz 1 − cos zwischen den
Zeilenvektoren eines Worts in zwei benachbarten Zeitscheiben (Hamilton et al.
2016b; Kutuzov et al. 2018). Geprüft in `pipeline_emb_ppmi_diagnose.py`,
Abschnitt 4, an Paaren mit mindestens h\* Belegen in beiden Zeilen.
Kennwert: Pearson-r zwischen Drift und log der kleineren Trefferzahl,
innerhalb der Zielwörter (Bland & Altman 1995), Ziel 0; Unsicherheit durch
Resampling der Zielwörter mit Zurücklegen (Bootstrap; Efron & Tibshirani
1993; ganze Gruppen: Davison & Hinkley 1997). Verglichen werden Rohdrift,
Teilstichprobe je Paar (stärkere Zeile auf die Trefferzahl der schwächeren,
20 Ziehungen nach Dokumenten), Teilstichprobe je Wort (gemittelte Distanzen
oder gemittelte Vektoren) und Drift minus Permutationserwartung
(Permutationstest nach Reden, Good 2005; Phipson & Smyth 2010). Dazu eine
Sättigungskurve (Rarefaktion; Sanders 1968; Hurlbert 1971). Verfahren im
Einzelnen: `Readme Frequency Control.md`.

**Testergebnisse (Messreihe E, 2026-11-19, 2.275 Paare aus 203 Zielwörtern):**

| Variante | Belege | r | 95-%-Intervall | Einstufung |
|---|---|---|---|---|
| Rohdrift | 100 % | −0,260 | −0,330 bis −0,192 | Referenz |
| Teilstichprobe je Paar | 73 % | −0,094 | −0,208 bis 0,004 | belastbar geringer |
| Teilstichprobe je Wort, Distanzen | 19 % | −0,120 | −0,185 bis −0,058 | belastbar geringer |
| Teilstichprobe je Wort, Vektoren | 19 % | −0,390 | −0,467 bis −0,318 | belastbar höher |
| Drift − Permutationserwartung | 100 % | +0,295 | 0,216 bis 0,357 | kein belastbarer Unterschied |

- Teilstichprobe je Paar und je Wort (Distanzen) sind in |r| gleichwertig;
  je Paar verwendet viermal so viele Belege und erhält die Reihenfolge der
  Paare fast (Rangkorrelation 0,977).
- Im Testlauf mit Untergrenze 10 statt 30 (4.327 Paare) genügte die
  Teilstichprobe je Paar nicht (r −0,41 → −0,35); dort war je Wort besser.
- Gemittelte Vektoren verschlechtern: Bei der schwächsten Zeile eines Worts
  sind alle Ziehungen gleich, bei gut belegten glättet das Mittel.
- Der Abzug der Permutationserwartung überkorrigiert: Reines Rauschen hängt
  viel stärker von der Belegung ab (r = −0,82) als die beobachtete Drift
  (−0,20, gleiche Randsummen); Rauschen und Unterschied addieren sich nicht.
- Sättigungskurve: Bei 30 Belegen je Zeile beträgt die Drift ohne
  Unterschied 66 % der beobachteten, bei 50 Belegen 52 %, mit allen Belegen
  21 %. Keine Sättigung im Raster.
- Permutationstest: 100 % der Paare mit p < 0,05, in allen drei Terzilen.
  Benachbarte Halbjahre unterscheiden sich fast immer (Debatten, Anlässe);
  der Test zeigt einen Unterschied der Kontexte, nicht Bedeutungswandel.

**Grenzen:** r unterscheidet nicht zwischen Stichprobeneffekt und echtem
Wandel, der mit der Belegung einhergeht. Nur benachbarte Zeitscheiben. Paare
unter h\* gehen nicht ein. Die Erklärung, warum die Angleichung je Paar
genügt (gleiche Verzerrung beider Zeilen bei gleicher Belegung, s.
Volumeneffekt 1.18), ist nicht geprüft.

**Fazit:** Verwendet (Umsetzung offen, `nextsteps.md`): Drift nur für Paare
mit beiden Zeilen ≥ h\*, berechnet mit Teilstichprobe je Paar, als eigenes
Feld je Paar im Bundle; die Zeilenvektoren bleiben unverändert. Verworfen:
Abzug der Permutationserwartung, gemittelte Vektoren, p-Wert als Filter.
Ersetzt die bisherige Driftkurve aus den Zeilenvektoren (keine Nullverteilung,
s. 3.10).

**Ausblick:** **Je Korpus prüfen.** Die Untergrenze der Paare entscheidet mit
darüber, welche Variante genügt. Abschnitt 4 nach jedem neuen Lauf von
Abschnitt 1 wiederholen. Offen bleibt die Unterscheidung von Themenwechsel
und Bedeutungswandel (z. B. über Kontrollwörter mit vermutlich stabiler
Bedeutung).

**Literatur:** Hamilton et al. 2016b; Kutuzov et al. 2018; Dubossarsky,
Weinshall & Grossman 2017; Bland & Altman 1995; Efron & Tibshirani 1993;
Davison & Hinkley 1997; Good 2005; Phipson & Smyth 2010; Sanders 1968;
Hurlbert 1971; Spearman 1904.

---

# 2. Getestete, verworfene Verfahren

### 2.1 Zeilennormierung vor der SVD

**Zweck:** Verhindern, dass dichte Zeitscheiben mit großer Vektorlänge die
SVD-Achsen dominieren.

**Beschreibung:** Matrixzeilen vor der SVD auf Länge 1 normieren.

**Testergebnisse (Messreihe A, 2026-09-01):** Volumeneffekt gepoolt 0,208 →
**0,929**, globale Ähnlichkeit −0,138 → 0,967, alle 11 Zielwörter positiv.
`Sicherheit` 2022-Q2 Rang 49 → 33, 2022-Q1 Rang 38 → 26. Frobenius-Anteil bei
k = 100: 40,9 % → 32,8 %. Im vollen Raum wäre die Normierung wirkungslos (der
Kosinus ist skaleninvariant), der Effekt entsteht durch die Kürzung auf k.
Erklärung: Die großen Normen der dichten Zeilen bestimmen die Hauptachsen, und
das sind die Richtungen des geteilten Vokabulars. Dünne Zeilen werden auf diese
Basis projiziert und dadurch geglättet. Mit gleichem Gewicht beanspruchen sie
eigene Achsen für ihr Hapax-Vokabular und stehen danach fast orthogonal zu
allem.

**Fazit:** Verworfen. Die Normdominanz war kein Defekt, sondern ein Schutz.

**Ausblick:** Nicht re-evaluieren, es sei denn, die Textmengen je Zeitscheibe
sind im neuen Korpus annähernd gleich.

**Literatur:** –

### 2.2 Globaler Mindesthäufigkeitsfilter auf den Kontextwörtern

**Zweck:** Dünnen Zeilen ihr vermeintlich privates, seltenes Vokabular nehmen.

**Beschreibung:** Nur Kontextwörter mit einer Gesamthäufigkeit über einer
Schwelle bekommen eine Matrixspalte.

**Testergebnisse (Messreihe A):** Die Annahme war falsch. Der Filter schneidet
fast nur bei den dichten Zeilen: Von Schwelle 1 auf 100 fallen 96,4 % der
Spalten weg; die dünnste Zeile behält 77 % ihrer Spalten, die dichteste nur
20 %. Seltene Wörter sitzen nach Zipf und Heaps in den großen Stichproben.

| Schwelle | Spalten | r gepoolt | `Sicherheit` | Rang 2022-Q1 | Rang 2022-Q2 | Frobenius-Anteil |
|---|---|---|---|---|---|---|
| 1 (aus) | 64.165 | 0,208 | 0,336 | 38 | 49 | 40,9 % |
| 5 | 17.404 | 0,442 | 0,604 | 33 | 45 | – |
| 50 | 3.689 | – | – | – | – | 46,6 % |
| 100 | 2.285 | 0,878 | 0,835 | 11 | 34 | 50,0 % |

Die Verschlechterung setzt sofort ein und ist monoton. Der steigende
Frobenius-Anteil ist kein Gegenargument: Mit den Spalten verschwindet auch
schwer anpassbare Masse.

**Fazit:** Verworfen. Es gibt keinen unbedenklichen Schwellenwert.

**Ausblick:** Nicht re-evaluieren. Der Mechanismus (seltene Wörter tragen das
Signal) ist nicht an die Korpusgröße gebunden.

**Literatur:** Gonen et al. 2020 und Hamilton et al. 2016a filtern nach
Häufigkeit, aber auf Ebene der Nachbarkandidaten bzw. des Vokabulars für ein
Wandelmaß, nicht in einer PPMI-Matrix.

### 2.3 Hapax-Filter je Zeitscheibe

**Zweck:** Kontextwörter mit genau einem Vorkommen entfernen, weil sie viel
Frobenius-Masse tragen.

**Beschreibung:** Hapax-Kontexte sind 25,6 % der Zellen und tragen 41,8 % der
Frobenius-Masse (Messreihe A).

**Testergebnisse:** Je Zeitscheibe angewandt zerstört der Filter das gesuchte
Signal: `Zeitenwende` ist in 2014-Q2, 2016-Q3 und 2017-Q1 Hapax und springt
danach auf 30, 107, 59, … 101 Vorkommen. Der Filter löscht die Vorgeschichte
des Begriffs. 3.388 von 3.689 Wörtern mit Gesamthäufigkeit ≥ 50 sind in
mindestens einem Quartal Hapax. Global angewandt ist es der Filter aus 2.2.

**Fazit:** Verworfen.

**Ausblick:** Nicht re-evaluieren.

**Literatur:** –

### 2.4 POS- und Stoppwortfilter in der Matrix

**Zweck:** Funktionswörter vor der Zählung entfernen.

**Beschreibung:** Entfernt wurden DET, ADP, PRON, AUX, CCONJ und SCONJ. PART
blieb, weil dort im Deutschen die Negation steckt.

**Testergebnisse (Messreihe A):** Der Filter entfernt 46,8 % der Kontext-Tokens
und ändert am Ergebnis fast nichts:

| | ohne Filter | mit POS-Filter |
|---|---|---|
| Kookkurrenz-Zeilen | 544.678 | 510.500 |
| Matrixspalten | 64.165 | 63.851 (−0,49 %) |
| Frobenius-Anteil | 40,9 % | 40,9 % |
| r gepoolt | 0,208 | 0,239 |
| Ränge der Kontrollfälle | 50/47/42/38/49 | 50/47/41/37/49 |

Grund ist PPMI selbst: Funktionswörter haben eine riesige Randsumme, ihr
Erwartungswert liegt in der Größenordnung der Beobachtung, also PMI ≈ 0 und
PPMI = 0. Der Nachteil ist messbar, aber klein, und trifft die dünnsten
Zielwörter (Δ r bis +0,18 bei `Landesverteidigung`).

Bullinaria & Levy (2012) testen Stoppwortlisten für PPMI-Räume systematisch.
Nach früherer Erinnerung brachten sie für PPMI wenig, weil PPMI frequente,
gleichmäßig verteilte Kontexte selbst abwertet. Dieser Befund ist **nicht
nachgeprüft**; vor dem Zitieren im Paper lesen. Wenn Funktionswörter über die
Zielwörter einer Zeitscheibe ungefähr proportional verteilt sind, schrumpfen
beim Entfernen Gesamtsumme und Zeilensummen etwa im gleichen Maß, und die
PPMI-Werte der übrigen Wörter ändern sich kaum.

**Fazit:** Zuerst (2026-09-01) beibehalten, weil er nichts kostete. Seit
2026-10-01 **verworfen als Matrixfilter**: Die Matrix bleibt ungefiltert,
POS-Filter, Rednernamen und Domänen-Stoppwörter wirken nur noch bei der
Auswahl von Nachbarn und Ankern (s. 1.12–1.14). Das entspricht der Logik von
Gonen et al. (2020), die auf dem Vokabular filtern, nicht in der Gewichtung.

**Ausblick:** Nicht re-evaluieren.

**Literatur:** Bullinaria & Levy 2012; Gonen et al. 2020; Turney & Pantel 2010.

### 2.5 Context Distribution Smoothing (CDS, α = 0,75)

**Zweck:** Die Überschätzung seltener Kontextwörter durch PMI dämpfen.

**Beschreibung:** Die Kontextwahrscheinlichkeit wird geglättet:

$$
P_\alpha(c) = \frac{\#(c)^\alpha}{\sum_{c'} \#(c')^\alpha}, \qquad \alpha = 0{,}75
$$

Mit α < 1 steigt die geschätzte Wahrscheinlichkeit seltener Kontextwörter, ihr
PPMI sinkt; bei häufigen ist es umgekehrt. 0,75 stammt aus dem Negative
Sampling von word2vec (Mikolov et al. 2013b). Levy, Goldberg & Dagan (2015)
berichten, dass CDS PPMI stärker hilft als anderen Verfahren; Schlechtweg et
al. (2019) setzen α = 0,75 fest, ohne es zu testen. In der Recherche zu
Hub-Wörtern wurde CDS als Standardschritt empfohlen.

**Testergebnisse:**

- Messreihe A: CDS hebt den PMI-Deckeneffekt nicht auf.
- Messreihe B (2026-10-01): Der Volumeneffekt verdoppelt sich (r gepoolt
  0,395 → 0,80). Gerade über seltene Kontextwörter unterscheiden sich textarme
  Zeilen von den übrigen; CDS nimmt ihnen diese Unterscheidung.
- Messreihe C (2026-10-06): **Verschlechterung**, beide Kennwerte sinken in
  allen 20 Teilungen. a_zeit 0,799 → 0,729, a_wort 0,880 → 0,821. Die
  Vorprüfung bestätigt, dass CDS wirkt wie beschrieben: Der Anteil der
  Hapax-Kontexte an der quadrierten Vektorlänge sinkt im Mittel von 59 % auf
  39 %, bei hoher Textmenge von 72 % auf 26 %. Gerade dort, wo der erwartete
  Nutzen am größten sein sollte, sinkt a_zeit am stärksten (−0,140).

**Fazit:** Verworfen (α = 1), in B und C übereinstimmend. Der Befund passt zu
Übergreifender Befund 1.

**Ausblick:** **Re-evaluieren.** Der Schaden entsteht über die Dünne textarmer
Zeilen. Bei durchgehend gut besetzten Zeitscheiben kann CDS seinen
dokumentierten Nutzen entfalten.

**Literatur:** Levy, Goldberg & Dagan 2015; Mikolov et al. 2013b; Turney &
Pantel 2010; Schlechtweg et al. 2019.

### 2.6 Additive Glättung (add-λ)

**Zweck:** Zufallsschwankungen kleiner Zählungen dämpfen.

**Beschreibung:** Vor der PPMI-Berechnung wird zu jeder Zählung einer
Zeitscheibe λ addiert, auch zu nie beobachteten Paaren (Lidstone-Glättung, bei
λ = 1 Laplace-Glättung). Geglättet werden nur Zeilen mit Beleg und
Kontextwörter, die in der Zeitscheibe vorkommen.

**Testergebnisse:**

- Messreihe B (2026-10-01): Schon bei λ = 0,01 entfallen 31 % der positiven
  Matrixzellen auf Paare ohne beobachtete Kookkurrenz (λ = 0,1: 60 %). Die
  Ränge der Referenzfälle verschieben sich stark (`Sicherheit` 2021-Q3: Rang
  48 → 4).
- Messreihe C (2026-10-06), gegenüber „ohne Glättung“ (a_zeit 0,799,
  a_wort 0,880):

  | λ | a_zeit | a_wort | Anteil nie beobachteter Paare | Einstufung |
  |---|---|---|---|---|
  | 0,01 | 0,813 | 0,865 | 7 % | kein belastbarer Unterschied |
  | 0,1 | 0,869 | 0,896 | 26 % | Zielkonflikt |
  | 1 | 0,882 | 0,936 | 36 % | Zielkonflikt |

  Die Gewinne entstehen fast nur bei geringer Textmenge (1–15 Belege; bei
  λ = 1 a_zeit +0,330, a_wort +0,214). Bei hoher Textmenge sinkt a_zeit
  (λ = 1: −0,049), bei mittlerer sinkt a_wort (λ = 0,1: −0,075). Die Anteile
  nie beobachteter Paare liegen unter denen aus B (31 % bzw. 60 %), weil C nur
  Zeilen mit Beleg und nur Kontextwörter der jeweiligen Zeitscheibe glättet.

- **Deutung (Vermutung, nicht geprüft):** Geglättet wird über die Kontextwörter
  der ganzen Zeitscheibe. Bei dünnen Zeilen besteht der Vektor danach zu einem
  großen Teil aus dem Vokabular des Quartals, nicht aus den Belegen des
  Zielworts. Beide Hälften teilen dieses Vokabular, und es unterscheidet sich
  von anderen Quartalen. Das hebt a_zeit, misst aber das Quartal, nicht die
  Bedeutung des Zielworts. Das Notebook nennt diese Grenze von a_zeit
  ausdrücklich (s. 1.17).

**Fazit:** Verworfen. In C nicht als Verbesserung eingestuft, und die
Gewinne bei dünnen Zeilen beruhen auf nie beobachteten Paaren. Gegen den
Deckeneffekt bleibt es bei `local_mi` und `t_score` für die Anzeige.

**Ausblick:** **Nur die Deutung prüfen**, nicht das Verfahren: Mit einem Maß,
das Quartalsthemen herausrechnet, ließe sich klären, ob der Gewinn bei dünnen
Zeilen echt ist. Der Anteil erfundener Zellen wächst mit der Spaltenzahl,
also mit dem Korpus.

**Literatur:** Manning & Schütze 1999, Kap. 6.2; Turney & Pantel 2010.

### 2.7 Local MI als SVD-Eingabe

**Zweck:** Den Deckeneffekt schon in den Vektoren beheben.

**Beschreibung:** `local_mi` (s. 1.6) statt PPMI als Matrixwert.

**Testergebnisse (Messreihe A):** Alle drei Maße haben dasselbe
Besetzungsmuster. Entscheidend ist die Verteilung der Frobenius-Masse, weil sie
bestimmt, was eine abgeschnittene SVD anpasst: In 2022-Q1 liegen bei PPMI 1,8 %
der Masse in den Top-10-Zellen, bei `local_mi` 56,0 % (dominiert von `der`,
`wir`), beim bei 0 gekappten t-Score 4,2 %.

**Fazit:** Verworfen. Der gekappte t-Score wäre allenfalls ein späterer
Vergleichskandidat.

**Ausblick:** Nicht re-evaluieren.

**Literatur:** Evert 2005; Levy & Goldberg 2014.

### 2.8 PMI²

**Zweck:** Den Deckeneffekt beheben, indem `beobachtet` quadriert eingeht.

**Testergebnisse (Messreihe A):** Funktionswortdominiert.

**Fazit:** Verworfen.

**Ausblick:** Nicht re-evaluieren.

**Literatur:** Daille 1994 (nicht nachgeprüft; vor dem Zitieren prüfen).

### 2.9 Auswahl nach Reichweite, Kontrast und Echo

**Zweck:** Generische Kontextwörter („gehen“, „sagen“, „Deutschland“) aus
Anker- und Nachbarlisten heraushalten.

**Beschreibung:** Drei eigene Verfahren aus Messreihe A:

- **Reichweite:** Jede Zeile nennt ihre 10 nächsten Kontextwörter, gezählt wird,
  wie oft ein Wort genannt wird. Belohnt Wörter, die vielen Zeilen mittelmäßig
  nah sind, also die generischen im Schwerpunkt des Raums.
- **Kontrast:** Kosinus minus Spaltenmittel (bzw. `spitze − mittel` je
  Kontextwort). Findet Wort-Unterschiede, nicht Zeit-Unterschiede: Das ganze
  Zeitenwende-Vokabular (`Sondervermögen`, `Beschaffung`, `Aufrüstung`) fehlte
  in den Top 200. Ein zweiter Durchgang „Kontrast über die Zeit“ verglich die
  Nähe vor und nach der Zäsur.
- **Echo:** Bezeichnung für den Spiegeleffekt der Kontextwörter mit nur einer
  belegten Zeile (s. 1.9).

**Testergebnisse:** Die Recherche zu Hub-Wörtern ordnete Echo der
k-Occurrence/Hubness-Diagnose (Radovanović et al. 2010) und Kontrast der
Keyness bzw. CSLS zu und empfahl, beide gegen die etablierten Verfahren zu
prüfen. Das ist geschehen (s. 2.10, 2.11, 1.19).

**Fazit (2026-10-05):** Verworfen. „Kontrast“ war kein etabliertes Verfahren
und wurde durch den Test der etablierten Hubness-Korrekturen ersetzt. Der
Befund hinter „Echo“ bleibt gültig, wird aber ohne eigenen Begriff als
Mindestbelegung geführt (s. 1.11). Ankerkandidaten kommen künftig aus der
Keyness (s. 3.9).

**Ausblick:** Nicht re-evaluieren.

**Literatur:** Radovanović, Nanopoulos & Ivanović 2010; Conneau et al. 2018;
Dubossarsky, Weinshall & Grossman 2017.

### 2.10 CSLS (Cross-domain Similarity Local Scaling)

**Zweck:** Hubs in den Nachbarlisten abwerten.

**Beschreibung:**

$$
\mathrm{CSLS}(x, y) = 2\cos(x, y) - r_Z(x) - r_K(y)
$$

$r_Z(x)$ ist der mittlere Kosinus der Zeile zu ihren 10 nächsten Kandidaten,
$r_K(y)$ der mittlere Kosinus des Kandidaten zu seinen 10 nächsten Zeilen.
Innerhalb einer Zeile entscheidet nur $r_K(y)$ über die Rangfolge. CSLS ist das
meistgenutzte Verfahren für den Abruf zwischen zwei Räumen (Übersetzung). Die
Anwendung auf Zeilen und Kontextwörter derselben SVD ist eine Übertragung.

**Testergebnisse (`pipeline_post_diagnose.py`, Abschnitt 2, und inhaltliche
Durchsicht 2026-10-05):** CSLS beseitigt die Hubs nicht. Bei unseren Daten
gehören die 10 nächsten Zeilen eines Kandidaten meist zu einem einzigen
Zielwort; $r_K(y)$ misst dann die Nähe zu diesem Zielwort, nicht die Nähe zu
Zeilen allgemein.

**Fazit:** Verworfen.

**Ausblick:** **Re-evaluieren, falls 1.19 auf dem neuen Korpus Hubness zeigt.**
Mit mehr Zielwörtern verliert der Einwand an Gewicht, dass die nächsten Zeilen
eines Kandidaten zu einem Zielwort gehören.

**Literatur:** Conneau et al. 2018; Dinu, Lazaridou & Baroni 2015.

### 2.11 Mutual Proximity

**Zweck:** Hubs abwerten, indem Nachbarschaft symmetrisch gemacht wird.

**Beschreibung:** Variante mit Normalverteilung:

$$
\mathrm{MP}(x, y) = \Phi\!\left(\frac{\cos(x, y) - \mu_x}{\sigma_x}\right) \cdot \Phi\!\left(\frac{\cos(x, y) - \mu_y}{\sigma_y}\right)
$$

Ein Kandidat, der allen Zeilen ähnlich nahe ist, erreicht im zweiten Faktor
nur mittlere Werte. Mutual Proximity und Local Scaling schnitten im Vergleich
von Feldbauer & Flexer (2019) am besten ab.

**Testergebnisse (wie 2.10):** Gleichwertige bzw. etwas schlechtere Listen als
roher Kosinus. Die Annahmen (Normalverteilung, Unabhängigkeit der Faktoren)
sind nicht geprüft.

**Fazit:** Verworfen.

**Ausblick:** Wie 2.10.

**Literatur:** Schnitzer et al. 2012; Feldbauer & Flexer 2019; Feldbauer,
Rattei & Flexer 2020 (scikit-hubness).

### 2.12 Anker im PCA-Fit

**Zweck:** Die 3D-Achsen so wählen, dass Anker und Zielwörter gemeinsam gut
abgebildet werden.

**Beschreibung:** Aggregate und Anker in einem Stapel, eine Zerlegung.

**Testergebnisse:** Die Achsen hingen davon ab, wie viele Anker ein Lauf trifft
und welche eingeblendet sind.

**Fazit (2026-10-05):** Verworfen. Fit nur auf den Aggregaten (s. 1.16).

**Ausblick:** Nicht re-evaluieren.

**Literatur:** –

### 2.13 Nachbarlisten als Grundlage des Methodenvergleichs

**Zweck:** PPMI- und SGNS-Pfad über die Überlappung ihrer Nachbarlisten
vergleichen.

**Testergebnisse (2026-09-04, Jahr 2022):** Für `Bundeswehr`, `Drohne` und
`Sicherheit` ist die Schnittmenge leer. Die Erklärung „PPMI syntagmatisch, SGNS
paradigmatisch“ trifft nur halb zu: 7 der 10 SGNS-Nachbarn von `Bundeswehr`
stehen auch häufig im Kontextfenster. Der eigentliche Unterschied: PPMI belohnt
Exklusivität, nicht Charakteristik (Deckeneffekt).

**Fazit:** Verworfen. Der Vergleich der Verfahren ruht auf den Aggregaten
(Driftkurven, Ähnlichkeitsmatrizen), nicht auf Nachbarlisten. Im Frontend muss
benannt sein, dass die Nachbarpanels Verschiedenes zeigen.

**Ausblick:** Nicht re-evaluieren.

**Literatur:** –

### 2.14 Festes Vokabular und Referenzscheiben-Anker im SGNS-Pfad

**Zweck:** Identische Zeilen für die Procrustes-Ausrichtung bzw. ein Vektor je
Anker.

**Testergebnisse (2026-09-04):** Ein Schnitt über alle Zeitscheiben löscht
`Zeitenwende` (2016 nicht im Vokabular). Die Referenzscheibe 2019 für Anker
hätte 11 Anker gelöscht, die nur 2022–2025 existieren (`Taurus`,
`Kriegstüchtigkeit`, `F-35`, …).

**Fazit:** Verworfen. Stützwörter (Schnitt, häufig) und Vokabular (je Scheibe)
werden getrennt; Ankervektor ist der Schwerpunkt über alle Scheiben, in denen
das Wort belegt ist.

**Ausblick:** Nicht re-evaluieren.

**Literatur:** Hamilton et al. 2016b.

### 2.15 SVD ohne schwach belegte Zeilen (*folding-in*)

**Zweck:** Verhindern, dass Zeilen mit wenigen Treffern die Achsen der SVD
mitbestimmen.

**Beschreibung:** Die SVD wird nur an den Zeilen mit mindestens N Treffern
(`n_hits`) berechnet. Die übrigen Zeilen werden per *folding-in*
($m V_k \Sigma_k^{p-1}$) in denselben Raum projiziert; sie bleiben erhalten und
sind markiert (`is_fitted`). Die PPMI-Werte werden an der vollständigen Matrix
berechnet. Umgesetzt als Schalter `MIN_HITS_FIT` in emb, Voreinstellung `None`
(alle Zeilen). Geprüft in `pipeline_emb_ppmi_diagnose.py`, Abschnitt 2:

- Vorprüfung an der vollständigen Matrix: Anteil der Zeilen unter N an der
  quadrierten Norm und an den Hebelwerten (*statistical leverage scores*) der
  ersten k Achsen (Mahoney & Drineas 2009).
- Split-half-Vergleich (1.17): SVD an allen Zeilen der Hälfte (Referenz) gegen
  SVD an den Zeilen mit mindestens N Belegen in der Hälfte. Bewertet werden die
  Zeilen mit mindestens N Belegen in beiden Hälften, Entscheidungsregel wie in
  Messreihe C (gleiches Vorzeichen in allen Teilungen, Pareto-Verbesserung über
  a_zeit, a_wort und drei Gruppen nach Textmenge).

**Testergebnisse (Messreihe D, 2026-11-17):**

| N | Anteil Zeilen | Anteil quadrierte Norm | Anteil Hebelwerte | Δ a_zeit | Δ a_wort | Einstufung |
|---|---|---|---|---|---|---|
| 3 | 17,7 % | 3,8 % | 0,5 % | −0,005 | −0,001 | Verschlechterung |
| 5 | 26,5 % | 7,2 % | 1,0 % | −0,009 | −0,002 | Verschlechterung |
| 7 | 33,0 % | 10,5 % | 1,6 % | −0,013 | −0,002 | Verschlechterung |
| 10 | 40,7 % | 15,4 % | 2,8 % | −0,019 | −0,002 | Verschlechterung |

- Die Rückgänge sind klein, aber in allen 20 Teilungen negativ und wachsen
  mit N.
- Auch die nicht gefitteten Zeilen selbst verlieren (Δ a_zeit −0,013 bei
  N = 3 bis −0,029 bei N = 10).
- Die schwach belegten Zeilen haben kaum Gewicht in der SVD: Ihre PPMI-Werte
  sind einzeln hoch, verteilen sich aber auf wenige Kontexte, die Zeilen
  bleiben kurz. Die Achsen werden ohnehin von den dichten Zeilen bestimmt
  (vgl. 2.1). Dass ihr Ausschluss dennoch schadet, deutet darauf hin, dass sie
  gemeinsam zur Schätzung der Kontextstruktur beitragen.
- Die Tabelle beruht auf k = 100, p = 1. **Wiederholung mit k = 300, p = 0,5**
  (2026-11-17, gleiche Daten und Teilungen): wieder bei allen N (3, 5, 7, 10)
  eine Verschlechterung. Die Einzelwerte stehen in der Auswertung des
  Notebooks.

**Testergebnisse (Messreihe E, 2026-11-19, k = 300, p = 0,5):**

| N | Anteil Zeilen | Anteil quadrierte Norm | Anteil Hebelwerte | Δ a_zeit | Δ a_wort | Einstufung | Δ a_zeit nicht gefittet |
|---|---|---|---|---|---|---|---|
| 3 | 22 % | 5 % | 2 % | +0,004 | +0,003 | Verbesserung | −0,006 |
| 5 | 34 % | 10 % | 4 % | +0,004 | +0,002 | Verbesserung | −0,004 |
| 7 | 43 % | 15 % | 6 % | +0,004 | +0,001 | Verbesserung | −0,001 |
| 10 | 52 % | 20 % | 8 % | +0,002 | 0,000 | kein belastbarer Unterschied | −0,001 |
| 30 (= h\*) | 75 % | 43 % | 19 % | −0,011 | −0,000 | nicht berechenbar | −0,009 |

- Anders als in D bei kleinem N eine Pareto-Verbesserung, aber sehr klein
  (+0,004 AUC); die ausgeschlossenen Zeilen verlieren dabei.
- Bei N = h\* = 30 sinkt a_zeit in allen Teilungen. „Nicht berechenbar“ ist
  ein Artefakt: Das untere Terzil der Bewertungszeilen ist leer, weil die
  Terzile nicht aus den Bewertungszeilen gebildet wurden (im Notebook seit
  2026-11-20 behoben, noch nicht neu gerechnet).

**Fazit:** Für diese Daten verworfen (D und E). In E ist der Gewinn bei
kleinem N zu klein, um die schlechtere Darstellung von 22–43 % der Zeilen zu
rechtfertigen, und bei der Deutungsschwelle h\* schadet der Ausschluss. Der Schalter bleibt als Option in emb,
Voreinstellung `None`. Schwach belegte Zeilen werden markiert statt
ausgeschlossen (s. 3.12).

**Ausblick:** **Je Korpus prüfen.** Abschnitt 2 des Diagnose-Notebooks für das
neue Korpus und nach der Entscheidung über k erneut laufen lassen; den
Schalter nur bei der Einstufung „Verbesserung“ setzen.

**Literatur:** Deerwester et al. 1990; Berry, Dumais & O'Brien 1995; Eckart &
Young 1936; Levy, Goldberg & Dagan 2015; Mahoney & Drineas 2009.

---

# 3. Nicht getestete und nicht verwendete Verfahren

### 3.1 Shifted PPMI (−log k)

**Zweck:** Schwache, zufällige Assoziationen abschneiden.

**Beschreibung:**

$$
\text{SPPMI} = \max\left(\log\frac{O}{E} - \log k,\ 0\right) = \max\left(\log\frac{O}{k \cdot E},\ 0\right)
$$

Zwei Wirkungen: Die Schwelle steigt von O > E auf O > k·E (bei k = 5 muss ein
Paar fünfmal so oft vorkommen wie erwartet), und alle verbleibenden Werte
sinken um den festen Betrag log k. Das ist eine Subtraktion, keine Skalierung,
und ändert deshalb Kosinuswerte und SVD. Herkunft: SGNS mit k Negativ-Samples
zerlegt implizit eine Matrix aus PMI − log k (Levy & Goldberg 2014).

**Literaturbefunde:** Levy, Goldberg & Dagan (2015): für PPMI je nach Aufgabe
nützlich, für SVD-Vektoren durchgehend schädlich. Schlechtweg et al. (2019),
Fußnote 8: PPMI mit k = 1 im Mittel besser (ρ = 0,549) als mit k = 5 (0,439).

**Fazit:** Nicht verwendet, nicht getestet. Die Recherche zu Hub-Wörtern hatte
einen Test mit k = 5 empfohlen. Da wir nach PPMI eine SVD rechnen, sprechen
beide Quellen dagegen.

**Ausblick:** Nur testen, falls auf dem neuen Korpus eine Variante ohne SVD
(reine PPMI-Vektoren) in Frage kommt.

**Literatur:** Levy & Goldberg 2014; Levy, Goldberg & Dagan 2015; Schlechtweg
et al. 2019.

### 3.2 Random Indexing (RI) und Shared Random Vectors (SRV)

**Zweck:** Dimensionsreduktion ohne Zerlegung.

**Beschreibung:** Jedes Kontextwort bekommt einen zufälligen, fast leeren
Kurzvektor (d = 300, s = 2 Einträge +1 oder −1). Der Vektor eines Zielworts ist
die mit den Kookkurrenzen gewichtete Summe dieser Zufallsvektoren (M · R). Nach
dem Johnson-Lindenstrauss-Lemma bleiben Abstände dabei ungefähr erhalten.
RI ist billig und inkrementell. Mit derselben Zufallsmatrix für alle Zeiträume
(SRV) liegen alle im selben Raum, analog zu unserer gemeinsamen Matrix.

**Literaturbefunde:** Schlechtweg et al. (2019), Tabelle 3: RI best/Mittel
DURel 0,601/0,374, SURel 0,622/0,299; SVD 0,728/0,498 bzw. 0,639/0,300.

**Fazit:** Nicht verwendet. Die SVD ist bei unserer Matrixgröße in Sekunden
gerechnet, Inkrementalität wird nicht gebraucht, und die Achsen von RI sind
Zufall, während wir sinnvolle Achsen für Projektion und Frobenius-Anteil
brauchen.

**Ausblick:** Nur prüfen, falls die SVD auf dem neuen Korpus an Speicher oder
Laufzeit scheitert.

**Literatur:** Sahlgren 2004; Basile, Caputo & Semeraro 2015; Johnson &
Lindenstrauss 1984; Schlechtweg et al. 2019.

### 3.3 Getrennte Räume je Zeitscheibe mit Orthogonal Procrustes (für PPMI-SVD)

**Zweck:** Jede Zeitscheibe bekommt eigene Achsen; Zeitscheiben mit viel Text
können die Achsen der anderen nicht prägen.

**Beschreibung:** Je Zeitscheibe eine eigene SVD, danach Ausrichtung per
Orthogonal Procrustes. Varianten: OP mit Längennormierung und Mean-Centering,
OP− ohne Mean-Centering (Hamilton et al. 2016b), OP+ mit weiteren Schritten
nach Artetxe et al. (2018).

**Literaturbefunde:** In Schlechtweg et al. (2019) ist SGNS + OP + Kosinus auf
beiden Datensätzen das beste Modell (DURel ρ = 0,866, SURel 0,851). Im Mittel
über SVD, RI und SGNS schlägt OP den gemeinsamen Raum (WI). Mean-Centering ist
dabei wichtig; OP+ ist auf SURel wegen der nicht orthogonalen Whitening-
Transformation um ~20 % schlechter. Die Datensätze haben nur je 22 Zielwörter.

**Fazit:** Nicht verwendet (Begründung s. 1.7). Im SGNS-Pfad ist OP Pflicht und
umgesetzt.

**Ausblick:** **Testen.** Wenn einzelne Zeitscheiben des neuen Korpus eine
eigene SVD tragen, ist der direkte Vergleich mit 1.7 möglich.

**Literatur:** Hamilton et al. 2016b; Schlechtweg et al. 2019; Artetxe, Labaka
& Agirre 2017, 2018.

### 3.4 Vector Initialization (VI)

**Zweck:** Ausrichtung ohne Procrustes: Das SGNS-Modell der späteren Zeitscheibe
startet mit den Vektoren der früheren.

**Literaturbefunde:** Schlechtweg et al. (2019): unerwartet schlecht (DURel
mittleres ρ = −0,017, SURel 0,082), vermutlich frequenzgetrieben (häufige Wörter
werden stärker aktualisiert).

**Fazit:** Nicht verwendet.

**Ausblick:** Nicht testen.

**Literatur:** Kim et al. 2014; Schlechtweg et al. 2019.

### 3.5 Mean-Centering und All-but-the-Top nach der SVD

**Zweck:** Den gemeinsamen Mittelwertvektor und die obersten
Hauptkomponenten entfernen, weil sie Wortfrequenz kodieren und Hubs erzeugen.

**Beschreibung:** Mittelwertvektor abziehen, danach die obersten D
Hauptkomponenten entfernen (Faustregel D ≈ d/100). Nur für dichte Vektoren,
also nach der SVD; auf roher PPMI zerstört Zentrieren die Dünnbesetzung.
Punkte nahe am Mittelwert werden zu Hubs (Radovanović et al. 2010), Zentrieren
wirkt also auch gegen Hubness.

**Literaturbefunde:** Mu & Viswanath (2018): im Mittel +1,7 % bei
Wortähnlichkeit. Schlechtweg et al. (2019): Mean-Centering verbessert WI um 3 %
und ist ein wichtiger Teil von OP.

**Fazit:** Nicht getestet. Unsere SVD-Zeilenvektoren werden nicht zentriert.
Die PCA für 3D zentriert, aber nur für die Darstellung.

**Ausblick:** **Testen**, und zwar schon auf den jetzigen Daten mit der
Split-half-Prüfung (1.17). Es ist die nächstliegende Erweiterung unseres
gemeinsamen Raums.

**Literatur:** Mu & Viswanath 2018; Radovanović, Nanopoulos & Ivanović 2010;
Schnabel et al. 2015; Schlechtweg et al. 2019.

### 3.6 Subsampling häufiger Wörter

**Zweck:** Den Einfluss sehr häufiger Wörter verringern, indem Vorkommen
zufällig verworfen werden.

**Literaturbefunde:** Levy, Goldberg & Dagan (2015): hilft bei Ähnlichkeit,
schadet bei Analogien. Schlechtweg et al. (2019): bei SGNS ohne Nutzen
(ρ = 0,506 mit, 0,517 ohne), bei RI deutlich (0,413 gegen 0,285).

**Fazit:** Nicht verwendet. Für PPMI ist es durch die Randsummen weitgehend
überflüssig (s. 2.4).

**Ausblick:** Im SGNS-Pfad mitprüfen.

**Literatur:** Mikolov et al. 2013b; Levy, Goldberg & Dagan 2015; Schlechtweg
et al. 2019.

### 3.7 Wandelmaße: Local Neighborhood Distance und Dispersionsmaße

**Zweck:** Bedeutungswandel als Zahl je Wort und Zeitscheibenpaar ausdrücken.

**Beschreibung:**

- **Local Neighborhood Distance (LND):** Für ein Wort wird die Vereinigung
  seiner k nächsten Nachbarn aus beiden Zeitscheiben gebildet. Je Zeitscheibe
  entsteht ein Vektor der Kosinuswerte zu diesen Nachbarn, die beiden Vektoren
  werden per Kosinus verglichen. Braucht keine Ausrichtung. Erfasst nach
  Hamilton et al. (2016a) vor allem kulturelle Verschiebungen bei Nomen,
  während der globale Kosinus regelmäßigen Drift bei Verben erfasst.
- **Dispersionsmaße:** Frequenzdifferenz (FD), Typendifferenz (TD),
  Entropiedifferenz (HD).

**Literaturbefunde:** Schlechtweg et al. (2019): Kosinusdistanz schlägt LND,
wo ausgerichtet werden kann (SGNS + OP: 0,723 gegen 0,620); LND ist die
robuste Alternative ohne Ausrichtung. Dispersionsmaße schneiden schlecht ab,
sind stark frequenzabhängig und zeigen auf SURel sogar starke negative
Korrelationen (bis ρ = −0,79).

**Fazit:** LND nicht verwendet. Unsere Wandelmaße sind der Kosinus im
gemeinsamen Raum (Driftkurven, Ähnlichkeitsmatrizen, Wandel-Ranking im
Frontend, 1.21) und die Schnittmenge der Nachbarlisten (ebenfalls 1.21), die
wie LND ohne Ausrichtung auskommt.

**Ausblick:** LND als zweites Maß testen, wenn der Methodenvergleich mit SGNS
quantitativ wird. Dispersionsmaße nicht testen.

**Literatur:** Hamilton et al. 2016a; Schlechtweg et al. 2017, 2019; Santus et
al. 2014.

### 3.8 Schnittmenge der Nachbarlisten (Gonen et al. 2020)

Umgesetzt, s. 1.21 (Wandel-Ranking mit Permutations-Baseline) und die
Metrik „Überlappung“ der Ähnlichkeitsmatrizen und Driftkurven im Frontend.
Die Abschnittsnummer bleibt als Verweis erhalten.

### 3.9 Keyness für Ankerkandidaten

**Zweck:** Ankerkandidaten finden, die vor bzw. nach der Zäsur typisch sind.

**Beschreibung:** Log-Likelihood (G²) auf den Kookkurrenz-Zählungen aus pre,
Zeitraum vor gegen nach der Zäsur, mit Log Ratio als Effektgröße. Ergänzend
logDice (14 + log₂(2·f_xy / (f_x + f_y))), das weder extrem seltene noch extrem
häufige Partner bevorzugt.

**Fazit:** Nicht umgesetzt, geplant. Ersetzt die verworfene Kontrast-Rangliste
(s. 2.9).

**Ausblick:** **Testen.** Keyness profitiert direkt von größeren Zählungen.

**Literatur:** Dunning 1993; Rayson & Garside 2000; Hardie 2014 (nicht
nachgeprüft); Rychlý 2008; Evert 2005.

### 3.10 Kontrollbedingung durch Mischen der Zeitscheiben

**Zweck:** Prüfen, ob ein gemessener Wandel auch dort auftritt, wo kein Wandel
möglich ist.

**Beschreibung:** Die Sätze eines Zielworts werden über die Zeitscheiben
gemischt und mit gleicher Häufigkeit neu verteilt. Ein Maß, das Wandel misst,
sollte danach deutlich weniger Wandel zeigen.

**Literaturbefunde:** Dubossarsky, Weinshall & Grossman (2017) reproduzierten
angebliche Gesetze des Bedeutungswandels auch im gemischten Korpus, für Count,
PPMI und SVD. Mit PPMI + SVD erklärte Frequenz allein 67 % der Varianz in den
echten und 56 % in den gemischten Daten. Schlechtweg et al. (2019), Tabelle 5:
Auf DURel fällt ρ nach dem Mischen von 0,816 auf 0,180, auf SURel bleibt es
stabil (0,767 gegen 0,763) und sinkt erst nach Angleichen der Häufigkeiten
(0,576).

**Fazit:** Nicht umgesetzt in der Form des Mischens über alle Zeitscheiben.
Für Paare benachbarter Zeitscheiben gibt es seit Messreihe E eine
Kontrollbedingung im Diagnose-Notebook (1.23): Die Reden beider Zeitscheiben
werden gemischt und neu verteilt. Ergebnis: Alle Paare ab h\* Belegen liegen
über dieser Zufallserwartung; sie trennt Bedeutungswandel also nicht von
Themenwechsel, und ihr Abzug überkorrigiert. Die Driftkurven der App stehen
bisher ohne Nullverteilung da; der SGNS-Pfad hat mit Perzentil im Häufigkeitsband und
Seed-Wiederholung einen Ersatz (s. 1.20). Das Wandel-Ranking (1.21) hat eine
Nullverteilung aus dem Mischen ganzer Zeitscheiben; das ersetzt das Mischen
der Sätze nicht, weil es Effekte, die einen ganzen Zeitraum betreffen, nicht
erfasst.

**Ausblick:** **Testen.** Ohne Kontrollbedingung ist eine Driftzahl auf dem
neuen Korpus nicht interpretierbar.

**Literatur:** Dubossarsky, Weinshall & Grossman 2017; Schlechtweg et al. 2019;
Tahmasebi, Borin & Jatowt 2021.

### 3.11 Weitere Hubness-Korrekturen und Ähnlichkeitsmaße

**Zweck:** Alternativen zu rohem Kosinus, CSLS und Mutual Proximity.

**Beschreibung:**

- **Local Scaling** (Schnitzer et al. 2012), **Globally Corrected Retrieval**
  (Dinu et al. 2015), **Inverted Softmax** (Smith et al. 2017),
  **gegenseitige Nachbarn** (Mutual k-NN: y zählt nur, wenn auch x unter den k
  Nachbarn von y liegt).
- **APSyn** (Santus et al. 2016): Ähnlichkeit über die Rangsumme der gemeinsamen
  Top-N-Kontexte zweier Wörter.
- **Lin (1998)** und **Weeds & Weir (2005)**: Ähnlichkeit über geteilte
  Kontexte bzw. als Precision und Recall geteilter Kontexte. Lin ist die Basis
  des Sketch-Engine-Thesaurus.
- **Ähnlichkeit zweiter Ordnung** gegenüber einer Ankermenge: glättet Rauschen,
  verstärkt Hubs aber eher, wenn die Ankermenge nicht frequenzgefiltert ist.

**Fazit:** Nicht getestet. Nach dem Befund zu CSLS und Mutual Proximity
(2.10, 2.11) besteht derzeit kein Anlass.

**Ausblick:** Nur testen, wenn 1.19 auf dem neuen Korpus Hubness zeigt.

**Literatur:** Schnitzer et al. 2012; Dinu, Lazaridou & Baroni 2015; Lazaridou,
Dinu & Baroni 2015; Smith et al. 2017; Santus et al. 2016, 2018; Lin 1998;
Weeds & Weir 2005.

### 3.12 Automatische Häufigkeitsfilter der Nachbarn und Mindesttrefferzahl je Zelle

**Zweck:** Generische Hochfrequenzwörter und dünn belegte Zellen automatisch
ausschließen.

**Beschreibung:**

- **Frequenzfilter der Kandidaten** wie bei Gonen et al. (2020) (Top-200 und
  Stoppwörter raus, Nachbarn nur mit Rohfrequenz > 100) oder eine
  **Frequenzband-Beschränkung** um das Zielwort (Schnabel et al. 2015: Nachbarn
  sind nach Frequenz geordnet).
- **Mindesttrefferzahl N je Zelle**: Zellen unter N Treffern nicht ausliefern
  oder als unsicher markieren.

**Testergebnisse zur Mindesttrefferzahl (Messreihe D, 2026-11-17):**
Bestimmt wurde, ab welcher Trefferzahl eine Zeile reliabel ist
(`pipeline_emb_ppmi_diagnose.py`, Abschnitt 1). Verfahren: Split-half-Prüfung
wie in 1.17, SVD an einer Hälfte, *folding-in* der anderen. Aus den gut
belegten Zeilen (mindestens 50 Belege in jeder Hälfte, rund 360 Zeilen aus
30–32 Zielwörtern) werden je Hälfte Teilstichproben von genau h Belegen
gezogen, nach Reden und geschachtelt; die Randsummen sind die der Hälfte, in
der die Zeile durch ihre Teilstichprobe ersetzt ist. Kriterium ist a_zeit,
Niveau 0,7 (übliche Grenze für akzeptable Trennschärfe einer AUC, Hosmer &
Lemeshow 2000), in allen Teilungen und für jedes größere h. Gegenprobe:
natürliche Zeilen, gruppiert nach ihrer Trefferzahl in der kleineren Hälfte.

| Belege je Hälfte | 1 | 2 | 3 | 5 | 7 | 10 | 20 | 50 | alle |
|---|---|---|---|---|---|---|---|---|---|
| a_zeit, Teilstichproben | 0,591 | 0,679 | 0,730 | 0,779 | 0,804 | 0,825 | 0,849 | 0,852 | 0,821 |
| a_zeit, natürliche Zeilen | 0,498 | 0,588 | 0,639 | 0,690 | 0,751 | 0,794 | 0,834 | 0,821 | – |
| a_wort, Teilstichproben | 0,750 | 0,870 | 0,926 | 0,969 | 0,985 | 0,993 | 0,999 | 1,000 | 1,000 |

(Natürliche Zeilen: Gruppe ab dem jeweiligen Wert, z. B. 5–6 Belege.)

- Schwelle aus den Teilstichproben: h\* = 3. Die natürlichen Zeilen erreichen
  das Niveau in allen Teilungen erst ab 10 Belegen. Die tatsächlich schwach
  belegten Zeilen sind schlechter wiederzuerkennen, als ihre Trefferzahl
  allein erwarten lässt; h\* ist eine Untergrenze.
- a_wort taugt nicht als Kriterium: Schon ein einziger Beleg unterscheidet ein
  Zielwort von den anderen Zielwörtern derselben Zeitscheibe (0,75). Das
  Problem dünner Zeilen ist die Abgrenzung von den anderen Zeitscheiben
  desselben Worts.
- Unter 3 Belegen liegen 17,7 % der Zeilen mit 0,35 % der Belege, unter 10
  Belegen 40,7 % der Zeilen mit 2,2 % der Belege.
- Ob die SVD ohne diese Zeilen berechnet werden sollte, prüft 2.15: nein.
- Die Tabelle beruht auf k = 100, p = 1. **Wiederholung mit k = 300, p = 0,5**
  (2026-11-17, gleiche Daten und Teilungen): h\* = 7, Gegenprobe weiterhin 10.
  Mit dem größeren k braucht eine Zeile mehr Belege, um das Niveau zu
  erreichen; die vorsichtigere Schwelle aus der Gegenprobe bleibt gleich. Die
  Einzelwerte stehen in der Auswertung des Notebooks.

**Testergebnisse (Messreihe E, 2026-11-19, k = 300, p = 0,5, 12.280 Zeilen):**

| Belege je Hälfte | 1 | 5 | 7 | 10 | 20 | 30 | 50 | alle (≥ 50) |
|---|---|---|---|---|---|---|---|---|
| a_zeit, Teilstichproben | 0,485 | 0,482 | 0,506 | 0,545 | 0,675 | 0,762 | 0,854 | 0,935 |
| a_wort, Teilstichproben | 0,729 | 0,912 | 0,947 | 0,974 | 0,996 | 0,999 | 1,000 | 1,000 |

- **h\* = 30** (Niveau 0,70 in allen Teilungen); Zufallsgrenze h₀ = 10
  (a_zeit > 0,5 in allen Teilungen). Deutlich höher als in D (h\* = 7 bei
  k = 300). Warum, ist nicht geprüft; E beruht auf einer anderen
  Zielwortliste und vermutlich auf den Ausgaben von pre nach der Begrenzung
  langer Sätze auf ±20 Tokens, also weniger Kontext je Beleg.
- Gegenprobe: Natürliche Zeilen erreichen das Niveau ab der Gruppe 10–14
  Belege, also **unter** h\* (in D lag sie darüber). Die Gruppen vergleichen
  verschiedene Wörter; es gilt h\* = 30.
- Unter 30 Belegen liegen 75 % der Zeilen mit 10 % der Belege, unter 10
  Belegen 52 % mit 3 %.
- Die offene Frage aus D (a_zeit mit allen Belegen niedriger als mit 50
  gezogenen) wiederholt sich nicht: 0,935 gegen 0,854.
- h\* ist eine Schwelle für die Wiedererkennung einer Zeile. Für die Drift
  zwischen zwei Zeilen reicht sie ohne Korrektur nicht (1.23).

**Offene Frage (Messreihe D):** Bei den Prüfzeilen liegt a_zeit mit allen Belegen der Hälfte
niedriger (0,821) als mit 50 gezogenen Belegen (0,852). Mehr Daten sollten die
Wiedererkennung nicht verschlechtern. Ein Fehler in den Randsummen der
Teilstichproben ist ausgeschlossen (eine Teilstichprobe aus allen Belegen
ergibt exakt die Zeile der Hälfte). Mögliche Ursachen: ungleich große Hälften,
einzelne Reden mit vielen Belegen. Die Schwelle berührt das nicht.

**Fazit:** Ein automatischer Ausschluss wird weiterhin nicht verwendet.
Grundsatz „Raum unverändert, Auswahl filtern“: Die Pipeline liefert Attribute
(Belegung, Kookkurrenzen, Wortart, Rednername), gefiltert wird im Frontend.
Für dünne Zellen ist statt einer Schwelle eine Konfidenzanzeige vorgesehen;
die gemessene Schwelle (h\* bzw. der Wert der Gegenprobe) ist der dokumentierte
Richtwert dafür. Der SGNS-Pfad zeigt, dass ein Kontrastmaß eine
Häufigkeitsschwelle ersetzen kann (s. 1.20).

Entscheidung nach Messreihe E: h\* = 30 wird als `meta.min_reliable_freq`
ins Bundle übergeben; das Frontend markiert Zeilen darunter (Umsetzung offen,
`nextsteps.md`).

**Ausblick:** Abschnitt 1 des Diagnose-Notebooks für das neue Korpus und nach
jeder Änderung von k, p, Raster oder KWIC-Variante erneut laufen lassen.

**Literatur:** Gonen et al. 2020; Schnabel et al. 2015; Hamilton et al. 2016a;
Hosmer & Lemeshow 2000.

### 3.13 Externe Validierung

**Zweck:** Prüfen, ob die Vektoren Bedeutung zutreffend abbilden (Validität
statt Reliabilität).

**Beschreibung:** Abgleich der Nachbarn mit dem DWDS-Wortprofil (logDice über
syntaktische Relationen) oder GermaNet; für Wandelmaße die deutschen
Goldstandards DURel, SURel und SemEval-2020 Task 1.

**Literaturbefunde:** DURel und SURel haben je nur 22 Zielwörter; Rangfolgen
von Verfahren darauf sind entsprechend unsicher. In SemEval-2020 schnitten
statische Einbettungen besser ab als kontextualisierte.

**Fazit:** Nicht umgesetzt.

**Ausblick:** **Testen.** Alle bisherigen Messungen prüfen Reliabilität. Für
eine Publikation über das Zeitenwende-Korpus braucht es mindestens einen
externen Abgleich.

**Literatur:** Didakowski & Geyken 2014; Schlechtweg, Schulte im Walde &
Eckmann 2018; Hätty, Schlechtweg & Schulte im Walde 2019; Schlechtweg et al.
2020.

---

# Bibliografie

Die bibliografischen Angaben sind vor einer Veröffentlichung an den Originalen
zu prüfen. Ausdrücklich nicht nachgeprüft sind: der Befund von Bullinaria &
Levy (2012) zu Stoppwortlisten, Levy & Goldberg (2014), Hardie (2014), Daille
(1994), Kaiser et al. (2021). Für Dubossarsky, Weinshall & Grossman (2017)
weichen die Seitenangaben ab (ACL Anthology: 1136–1145; bei Schlechtweg et al.
2019: 1147–1156).

## Methoden und Studien

- Antoniak, M. & Mimno, D. (2018): Evaluating the stability of embedding-based word similarities. *Transactions of the ACL* 6, 107–119.
- Artetxe, M., Labaka, G. & Agirre, E. (2017): Learning bilingual word embeddings with (almost) no bilingual data. *Proceedings of ACL 2017*, 451–462.
- Artetxe, M., Labaka, G. & Agirre, E. (2018): Generalizing and improving bilingual word embedding mappings with a multi-step framework of linear transformations. *Proceedings of AAAI 2018*, 5012–5019.
- Basile, P., Caputo, A. & Semeraro, G. (2015): Temporal random indexing: A system for analysing word meaning over time. *Italian Journal of Computational Linguistics* 1, 55–68.
- Berry, M. W., Dumais, S. T. & O'Brien, G. W. (1995): Using linear algebra for intelligent information retrieval. *SIAM Review* 37(4), 573–595.
- Bland, J. M. & Altman, D. G. (1995): Calculating correlation coefficients with repeated observations: Part 1 – correlation within subjects. *BMJ* 310(6977), 446.
- Brown, W. (1910): Some experimental results in the correlation of mental abilities. *British Journal of Psychology* 3(3), 296–322.
- Bullinaria, J. A. & Levy, J. P. (2012): Extracting semantic representations from word co-occurrence statistics: Stop-lists, stemming, and SVD. *Behavior Research Methods* 44(3), 890–907. https://doi.org/10.3758/s13428-011-0183-8
- Caron, J. (2001): Experiments with LSA scoring: Optimal rank and basis. In: Berry, M. W. (Hrsg.): *Computational Information Retrieval*. SIAM, 157–169.
- Che Abdul Rahman, Ho Abdullah, Zainudin, Tiun & Jaludin: Domain-specific stop words in Malaysian parliamentary debates 1959–2018. *GEMA Online Journal of Language Studies*. https://ejournal.ukm.my/gema/article/view/45125
- Church, K. W. & Hanks, P. (1990): Word association norms, mutual information, and lexicography. *Computational Linguistics* 16(1), 22–29.
- Church, K. W., Gale, W., Hanks, P. & Hindle, D. (1991): Using statistics in lexical analysis. In: Zernik, U. (Hrsg.): *Lexical Acquisition*. Erlbaum, 115–164.
- Conneau, A., Lample, G., Ranzato, M., Denoyer, L. & Jégou, H. (2018): Word translation without parallel data. *ICLR 2018*. arXiv:1710.04087.
- Daille, B. (1994): *Approche mixte pour l'extraction automatique de terminologie*. Dissertation, Université Paris 7.
- Davison, A. C. & Hinkley, D. V. (1997): *Bootstrap Methods and their Application*. Cambridge University Press.
- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. & Harshman, R. (1990): Indexing by latent semantic analysis. *Journal of the American Society for Information Science* 41(6), 391–407.
- Didakowski, J. & Geyken, A. (2014): From DWDS corpora to a German word profile – methodological problems and solutions. *OPAL – Online publizierte Arbeiten zur Linguistik* 2/2014, 39–47. https://www.dwds.de/d/wortprofil
- Dinu, G., Lazaridou, A. & Baroni, M. (2015): Improving zero-shot learning by mitigating the hubness problem. *ICLR Workshop Track*. arXiv:1412.6568.
- Dubossarsky, H., Weinshall, D. & Grossman, E. (2017): Outta control: Laws of semantic change and inherent biases in word representation models. *Proceedings of EMNLP 2017*, 1136–1145. https://aclanthology.org/D17-1118/
- Dubossarsky, H., Hengchen, S., Tahmasebi, N. & Schlechtweg, D. (2019): Time-Out: Temporal referencing for robust modeling of lexical semantic change. *Proceedings of ACL 2019*.
- Dwork, C., Kumar, R., Naor, M. & Sivakumar, D. (2001): Rank aggregation methods for the web. *Proceedings of WWW 2001*, 613–622. https://doi.org/10.1145/371920.372165
- Dunning, T. (1993): Accurate methods for the statistics of surprise and coincidence. *Computational Linguistics* 19(1), 61–74.
- Eckart, C. & Young, G. (1936): The approximation of one matrix by another of lower rank. *Psychometrika* 1(3), 211–218.
- Efron, B. & Tibshirani, R. J. (1993): *An Introduction to the Bootstrap*. Chapman & Hall.
- Evert, S. (2005): *The Statistics of Word Cooccurrences: Word Pairs and Collocations*. Dissertation, Universität Stuttgart. https://elib.uni-stuttgart.de/items/c9a37ea0-002f-40ac-b66e-6257b2e03e23
- Feldbauer, R. & Flexer, A. (2019): A comprehensive empirical comparison of hubness reduction in high-dimensional spaces. *Knowledge and Information Systems*. https://doi.org/10.1007/s10115-018-1205-y
- Ferrari, A., Donati, B. & Gnesi, S. (2017): Detecting domain-specific ambiguities: An NLP approach based on Wikipedia crawling and word embeddings. *Proceedings of the IEEE 25th International Requirements Engineering Conference Workshops*, 393–399.
- Gonen, H., Jawahar, G., Seddah, D. & Goldberg, Y. (2020): Simple, interpretable and stable method for detecting words with usage change across corpora. *Proceedings of ACL 2020*, 538–555. https://doi.org/10.18653/v1/2020.acl-main.51
- Good, P. (2005): *Permutation, Parametric and Bootstrap Tests of Hypotheses*. 3. Aufl. Springer.
- Hamilton, W. L., Leskovec, J. & Jurafsky, D. (2016a): Cultural shift or linguistic drift? Comparing two computational measures of semantic change. *Proceedings of EMNLP 2016*, 2116–2121. https://doi.org/10.18653/v1/D16-1229
- Hamilton, W. L., Leskovec, J. & Jurafsky, D. (2016b): Diachronic word embeddings reveal statistical laws of semantic change. *Proceedings of ACL 2016*, 1489–1501. https://doi.org/10.18653/v1/P16-1141
- Hanley, J. A. & McNeil, B. J. (1982): The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 143(1), 29–36.
- Hardie, A. (2014): Log Ratio – an informal introduction. ESRC Centre for Corpus Approaches to Social Science (CASS), Lancaster.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009): *The Elements of Statistical Learning*. 2. Aufl. Springer.
- Hätty, A., Schlechtweg, D. & Schulte im Walde, S. (2019): SURel: A gold standard for incorporating meaning shifts into term extraction. *Proceedings of *SEM 2019*.
- Haxby, J. V., Gobbini, M. I., Furey, M. L., Ishai, A., Schouten, J. L. & Pietrini, P. (2001): Distributed and overlapping representations of faces and objects in ventral temporal cortex. *Science* 293(5539), 2425–2430.
- Hosmer, D. W. & Lemeshow, S. (2000): *Applied Logistic Regression*. 2. Aufl. Wiley.
- Hurlbert, S. H. (1971): The nonconcept of species diversity: a critique and alternative parameters. *Ecology* 52(4), 577–586.
- Johnson, N. L., Kemp, A. W. & Kotz, S. (2005): *Univariate Discrete Distributions*. 3. Aufl. Wiley.
- Johnson, W. B. & Lindenstrauss, J. (1984): Extensions of Lipschitz mappings into a Hilbert space. *Contemporary Mathematics* 26.
- Kaiser, J., Kurtyigit, S., Kotchourko, S. & Schlechtweg, D. (2021): Effects of pre- and post-processing on type-based embeddings in lexical semantic change detection. *Proceedings of EACL 2021*.
- Kilgarriff, A., Rychlý, P., Smrž, P. & Tugwell, D. (2004): The Sketch Engine. *Proceedings of EURALEX 2004*.
- Kim, Y., Chiu, Y.-I., Hanaki, K., Hegde, D. & Petrov, S. (2014): Temporal analysis of language through neural language models. *Proceedings of the ACL Workshop on Language Technologies and Computational Social Science*, 61–65.
- Kutuzov, A., Øvrelid, L., Szymanski, T. & Velldal, E. (2018): Diachronic word embeddings and semantic shifts: A survey. *Proceedings of COLING 2018*, 1384–1397. https://aclanthology.org/C18-1117/
- Lazaridou, A., Dinu, G. & Baroni, M. (2015): Hubness and pollution: Delving into cross-space mapping for zero-shot learning. *Proceedings of ACL-IJCNLP 2015*, 270–280. https://aclanthology.org/P15-1027/
- Lehoucq, R. B., Sorensen, D. C. & Yang, C. (1998): *ARPACK Users' Guide: Solution of Large-Scale Eigenvalue Problems with Implicitly Restarted Arnoldi Methods*. Philadelphia: SIAM. https://doi.org/10.1137/1.9780898719628
- Levy, O. & Goldberg, Y. (2014): Neural word embedding as implicit matrix factorization. *Advances in Neural Information Processing Systems 27*, 2177–2185.
- Levy, O., Goldberg, Y. & Dagan, I. (2015): Improving distributional similarity with lessons learned from word embeddings. *Transactions of the ACL* 3, 211–225. https://doi.org/10.1162/tacl_a_00134
- Lin, D. (1998): Automatic retrieval and clustering of similar words. *Proceedings of COLING-ACL 1998*, Bd. 2, 768–774. https://doi.org/10.3115/980691.980696
- Mahoney, M. W. & Drineas, P. (2009): CUR matrix decompositions for improved data analysis. *Proceedings of the National Academy of Sciences* 106(3), 697–702.
- Mann, H. B. & Whitney, D. R. (1947): On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics* 18(1), 50–60.
- Manning, C. D. & Schütze, H. (1999): *Foundations of Statistical Natural Language Processing*. MIT Press.
- Mikolov, T., Chen, K., Corrado, G. & Dean, J. (2013a): Efficient estimation of word representations in vector space. arXiv:1301.3781.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. (2013b): Distributed representations of words and phrases and their compositionality. *Advances in Neural Information Processing Systems 26*, 3111–3119.
- Mu, J. & Viswanath, P. (2018): All-but-the-top: Simple and effective postprocessing for word representations. *ICLR 2018*. arXiv:1702.01417.
- Österlund, A., Ödling, D. & Sahlgren, M. (2015): Factorization of latent variables in distributional semantic models. *Proceedings of EMNLP 2015*, 227–231.
- Phipson, B. & Smyth, G. K. (2010): Permutation P-values should never be zero: Calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology* 9(1), Article 39. https://doi.org/10.2202/1544-6115.1585
- Radovanović, M., Nanopoulos, A. & Ivanović, M. (2010): Hubs in space: Popular nearest neighbors in high-dimensional data. *Journal of Machine Learning Research* 11, 2487–2531.
- Rayson, P. & Garside, R. (2000): Comparing corpora using frequency profiling. *Proceedings of the Workshop on Comparing Corpora*, 1–6.
- Rychlý, P. (2008): A lexicographer-friendly association score. *Proceedings of RASLAN 2008*, 6–9.
- Sahlgren, M. (2004): An introduction to random indexing. *Language*, 1–9.
- Sanders, H. L. (1968): Marine benthic diversity: a comparative study. *The American Naturalist* 102(925), 243–282.
- Santus, E., Lenci, A., Lu, Q. & Schulte im Walde, S. (2014): Chasing hypernyms in vector spaces with entropy. *Proceedings of EACL 2014*, 38–42.
- Santus, E., Chiu, T.-S., Lu, Q., Lenci, A. & Huang, C.-R. (2016): Unsupervised measure of word similarity: How to outperform co-occurrence and vector cosine in VSMs. *Proceedings of AAAI 2016*, 4260–4261. arXiv:1603.09054.
- Santus, E. et al. (2018): A rank-based similarity metric for word embeddings (APSynP). *Proceedings of ACL 2018*. arXiv:1805.01923.
- Schlechtweg, D., Eckmann, S., Santus, E., Schulte im Walde, S. & Hole, D. (2017): German in flux: Detecting metaphoric change via word entropy. *Proceedings of CoNLL 2017*, 354–367.
- Schlechtweg, D., Schulte im Walde, S. & Eckmann, S. (2018): Diachronic Usage Relatedness (DURel): A framework for the annotation of lexical semantic change. *Proceedings of NAACL-HLT 2018*, 169–174.
- Schlechtweg, D., Hätty, A., del Tredici, M. & Schulte im Walde, S. (2019): A wind of change: Detecting and evaluating lexical semantic change across times and domains. *Proceedings of ACL 2019*, 732–746. https://doi.org/10.18653/v1/P19-1072
- Schlechtweg, D., McGillivray, B., Hengchen, S., Dubossarsky, H. & Tahmasebi, N. (2020): SemEval-2020 Task 1: Unsupervised lexical semantic change detection. *Proceedings of SemEval 2020*, 1–23. https://doi.org/10.18653/v1/2020.semeval-1.1
- Schnabel, T., Labutov, I., Mimno, D. & Joachims, T. (2015): Evaluation methods for unsupervised word embeddings. *Proceedings of EMNLP 2015*, 298–307. https://doi.org/10.18653/v1/D15-1036
- Schnitzer, D., Flexer, A., Schedl, M. & Widmer, G. (2012): Local and global scaling reduce hubs in space. *Journal of Machine Learning Research* 13, 2871–2902.
- Smith, S. L., Turban, D. H. P., Hamblin, S. & Hammerla, N. Y. (2017): Offline bilingual word vectors, orthogonal transformations and the inverted softmax. *ICLR 2017*. arXiv:1702.03859.
- Spearman, C. (1904): The proof and measurement of association between two things. *The American Journal of Psychology* 15(1), 72–101.
- Spearman, C. (1910): Correlation calculated from faulty data. *British Journal of Psychology* 3(3), 271–295.
- Tahmasebi, N., Borin, L. & Jatowt, A. (2021): Survey of computational approaches to lexical semantic change detection. In: Tahmasebi, N. et al. (Hrsg.): *Computational Approaches to Semantic Change*. Language Science Press, Kap. 1. https://doi.org/10.5281/zenodo.5040302
- Turney, P. D. & Pantel, P. (2010): From frequency to meaning: Vector space models of semantics. *Journal of Artificial Intelligence Research* 37, 141–188.
- Weeds, J. & Weir, D. (2005): Co-occurrence retrieval: A flexible framework for lexical distributional similarity. *Computational Linguistics* 31(4), 439–475. https://doi.org/10.1162/089120105775299122
- Wendlandt, L., Kummerfeld, J. K. & Mihalcea, R. (2018): Factors influencing the surprising instability of word embeddings. *Proceedings of NAACL-HLT 2018*, 2092–2102.
- arXiv:2012.02294: Accelerating text mining using domain-specific stop word lists.

## Software und Referenzimplementierungen

Stand der Recherche zu Hub-Wörtern (2026-10-01). Für die Produktion werden die
kurzen Formeln selbst in NumPy implementiert; die Pakete dienen als Referenz.

- **scikit-hubness** (Feldbauer, Rattei & Flexer 2020, *JOSS* 5(45), 1957, https://doi.org/10.21105/joss.01957; https://github.com/VarIr/scikit-hubness, BSD-3): Hubness-Diagnose, Mutual Proximity, Local Scaling, DisSimLocal. Nur schwach gepflegt.
- **MUSE** (https://github.com/facebookresearch/MUSE): Referenzimplementierung von CSLS. Archiviert seit 2023-10-31, CC BY-NC 4.0.
- **hyperwords** (Omer Levy; Kopien unter https://github.com/williamleif/histwords/tree/master/sgns/hyperwords und https://github.com/jfilter/hyperwords): PPMI und SVD mit den Parametern cds, neg, eig, sub. Python 2.7.
- **HistWords** (https://github.com/williamleif/histwords, Apache 2.0): Referenz für Hamilton et al. 2016b. Python 2, faktisch ungepflegt.
- **LSCDetection** (https://github.com/Garrafao/LSCDetection, GPL-3.0): Code zu Schlechtweg et al. 2019. Nachfolger: **LSCDBenchmark** (https://github.com/Garrafao/LSCDBenchmark, arXiv:2404.00176).
- **usage_change** (https://github.com/gonenhila/usage_change, GPL-3.0): Code zu Gonen et al. 2020.
- **gensim** 4.4.0 (LGPL-2.1): `Word2Vec` für den SGNS-Pfad; im Wartungsmodus, keine Hubness-Korrektur.
- **DWDS-Wortprofil** (https://github.com/zentrum-lexikographie/wordprofile): logDice, Mindestfrequenz 5.
- **APSyn** (https://github.com/esantus/APSyn): Einzelskript ohne Lizenz.
- **DISSECT** (https://github.com/composes-toolkit/dissect): PPMI, SVD und Ähnlichkeitsmaße; Python 2, seit 2015 ungepflegt.
