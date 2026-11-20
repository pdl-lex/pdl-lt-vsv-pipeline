# Readme Glossary

Stand: 2026-11-20

Begriffe der Embedding-Stufe und des Diagnose-Notebooks
(`stages/02_embedding/pipeline_emb_ppmi_diagnose.py`), gesammelt in der
Reihenfolge, in der sie besprochen werden.

Je Begriff: kurze Erklärung, englischer Fachbegriff und Status:

- **etabliert**: in der Literatur unter diesem Namen üblich
- **eigener Sprachgebrauch**: bei uns geprägt; der übliche Ausdruck steht
  dabei

Wo Literatur genannt wird, steht unter **Belegt durch** jeweils, wofür die
Quelle steht (Einführung des Begriffs, Nachweis einer Eigenschaft, Verwendung
im Fach). Die vollständigen Angaben stehen am Ende. Begriffe ohne Quelle sind
allgemeiner Fachwortschatz, der keinen Einzelbeleg braucht.

Die Zahlenbeispiele stammen, wenn nicht anders vermerkt, aus dem Datensatz
Bundestag, Variante `punkt`, Raster `halfyear` (Readme Frequency Control,
Abschnitt 3). Beispiele, die als *Rechenbeispiel* gekennzeichnet sind, dienen
nur der Veranschaulichung und sind keine Messwerte.

---

## 1. Daten und Matrix

### Beleg

Ein einzelner KWIC-Treffer: eine Stelle in einer Rede, an der das Zielwort
vorkommt, zusammen mit den Lemmata seines Kontexts (bei `punkt`: der Satz).

- **Englisch:** *KWIC hit*, *concordance line*, *occurrence*
  (KWIC = *keyword in context*)
- **Status:** etabliert (KWIC); „Beleg“ ist der übliche Ausdruck der
  Korpuslinguistik
- **Belegt durch:** Luhn (1960) führt das KWIC-Verfahren ein (Index, der jedes
  Stichwort mit seinem umgebenden Text zeigt).
- **Beispiel:** 692.136 Belege aus 67.062 Reden

### Zielwort

Ein Wort, dessen Verwendung über die Zeit untersucht wird.

- **Englisch:** *target word*
- **Status:** etabliert
- **Beispiel:** 596 Zielwörter, z. B. *Sicherheit*

### Zeitscheibe

Ein Zeitabschnitt, nach dem die Belege gruppiert werden.

- **Englisch:** *time slice*, *time bin*, *time period*
- **Status:** etabliert
- **Beispiel:** 26 Halbjahre (Raster `halfyear`), z. B. 2022-H1

### Kontextwort

Ein Lemma, das in einem Beleg in der Nähe des Zielworts vorkommt. Jedes
Kontextwort ist eine Spalte der Matrix.

- **Englisch:** *context word*, *context feature*
- **Status:** etabliert
- **Beispiel:** 120.882 Kontextwörter (Lauf des Diagnose-Notebooks vom
  2026-11-19)

### Kookkurrenzmatrix

Die Tabelle aus Zeilen (Zielwort × Zeitscheibe) und Spalten (Kontextwörter).
Eine Zelle gibt an, wie oft das Kontextwort in den Belegen dieser Zeile
vorkommt. Jedes Vorkommen eines Kontextlemmas in einem Beleg zählt einmal.

- **Englisch:** *co-occurrence matrix*, *word–context matrix*
- **Status:** etabliert

### Zeile

Ein Paar aus Zielwort und Zeitscheibe, z. B. *Sicherheit, 2022-H1*. Nicht jede
Kombination ist belegt.

- **Englisch:** in der Literatur meist *the vector of word w at time t*
- **Status:** eigener Sprachgebrauch
- **Beispiel:** 12.280 Zeilen (von 596 × 26 = 15.496 möglichen)

### Trefferzahl

Die Zahl der Belege einer Zeile (`n_hits` in emb).

- **Englisch:** *frequency*, *number of occurrences*, *sample size*
- **Status:** eigener Sprachgebrauch; in der Literatur meist einfach
  *frequency*
- **Beispiel:** Median 9, oberes Quartil 31, 90-%-Quantil 114, Maximum 5.363
  Belege je Zeile. Die Hälfte aller Zeilen beruht also auf 9 oder weniger
  Belegen.

---

## 2. Gewichtung

### PMI (Pointwise Mutual Information)

Misst, ob ein Kontextwort bei einem Zielwort häufiger vorkommt, als nach
seiner allgemeinen Häufigkeit zu erwarten ist:

PMI = log₂ (beobachtete Häufigkeit / erwartete Häufigkeit)

Die erwartete Häufigkeit ergibt sich aus den Randsummen (s. u.). Positiv:
häufiger als erwartet; 0: genau wie erwartet; negativ: seltener als erwartet.

- **Englisch:** *pointwise mutual information*
- **Status:** etabliert
- **Belegt durch:** Church & Hanks (1990) führen PMI als Assoziationsmaß
  zwischen Wörtern in die Lexikografie und Korpuslinguistik ein.
- **Rechenbeispiel:** Macht *Bundeswehr* in einer Zeitscheibe 1 % aller
  Kontexttokens aus, in der Zeile *Sicherheit* aber 4 %, ist das Verhältnis 4
  und PMI = log₂ 4 = 2.

### PPMI (Positive PMI)

PMI, bei dem alle negativen Werte auf 0 gesetzt werden. Übrig bleibt nur, was
überzufällig häufig ist. Die PPMI-Matrix ist die Eingabe der SVD.

- **Englisch:** *positive pointwise mutual information*
- **Status:** etabliert
- **Belegt durch:**
  - Bullinaria & Levy (2007): Im systematischen Vergleich von Gewichtungen
    für Wortvektoren schneidet PPMI am besten ab.
  - Levy, Goldberg & Dagan (2015): PPMI mit SVD ist ein starkes Verfahren,
    wenn es gut eingestellt ist. Außerdem: PMI überschätzt seltene Ereignisse.
    Sie zeigen das für seltene **Kontextwörter** (Spalten) und begründen damit
    die Glättung der Kontextverteilung (CDS, Methodology 2.5).
- **Schwäche:** Bei einer Zeile mit wenigen Belegen erhält jedes zufällig
  vorhandene Kontextwort einen hohen Wert. Das ist der Ausgangspunkt der
  Abschnitte 1, 2 und 4 des Diagnose-Notebooks.
  *Belegstatus:* Für seltene **Zeilen** ist das eine Übertragung, kein
  direkter Beleg. Sie folgt aus der Formel, die Zeile und Spalte symmetrisch
  behandelt: Ist die Zeilensumme klein, ist auch die erwartete Häufigkeit
  klein, und schon ein einziges Vorkommen liegt weit darüber.

### Randsummen

Die Summen einer Matrix über Zeilen oder Spalten. Aus ihnen wird die erwartete
Häufigkeit berechnet: Zeilensumme × Spaltensumme / Gesamtsumme. In emb werden
sie **je Zeitscheibe** gebildet. Die Erwartung für *Sicherheit, 2022-H1*
richtet sich also nach allen Belegen aus 2022-H1, nicht nach dem ganzen Korpus.

- **Englisch:** *marginals*, *marginal totals* (Oberbegriff für beide Arten)
- **Status:** etabliert

Die einzelnen Summen:

| Deutsch | Englisch | in der Kookkurrenzmatrix |
|---|---|---|
| Zeilensumme | *row total*, *row marginal*, *row sum* | alle Kontexttokens einer Zeile (z. B. *Sicherheit, 2022-H1*) |
| Spaltensumme | *column total*, *column marginal*, *column sum* | alle Vorkommen eines Kontextworts in einer Zeitscheibe, über alle Zielwörter |
| Gesamtsumme | *grand total* | alle Kontexttokens einer Zeitscheibe |

Teilt man die Summen durch die Gesamtsumme, erhält man Anteile. In der
PMI-Formel heißen sie **Randwahrscheinlichkeiten** (*marginal probabilities*),
P(w) für das Zielwort und P(c) für das Kontextwort.

---

## 3. Vektorraum

### SVD (Singulärwertzerlegung)

Zerlegt die PPMI-Matrix in Achsen, sortiert nach ihrem Singulärwert (s. u.). Die erste Achse bildet das stärkste gemeinsame Muster der Zeilen ab,
die zweite das stärkste verbleibende usw. Statt mit 120.882
Kontextwort-Spalten wird jede Zeile durch ihre Werte auf den Achsen
beschrieben.

- **Englisch:** *singular value decomposition*
- **Status:** etabliert
- **Belegt durch:** Deerwester et al. (1990) wenden die SVD als *Latent
  Semantic Analysis* (LSA) auf Wort-Dokument-Matrizen an und begründen damit
  ihren Einsatz in der distributionellen Semantik.

### Singulärwert

Die Zahl, nach der die Achsen der SVD sortiert werden. Ihr Quadrat gibt an,
welchen Teil der **Summe aller quadrierten PPMI-Werte** die Achse abbildet.
Die Summe der quadrierten Singulärwerte aller Achsen ist genau diese Summe.

**Berechnung, anschaulich:** Man sucht die Richtung im Raum der
Kontextwörter, auf die die Zeilen zusammen am stärksten fallen. Für eine
Richtung v (Länge 1) projiziert man jede Zeile auf v, quadriert die
Projektionen und summiert über alle Zeilen. Die Richtung, bei der diese Summe
am größten ist, ist die erste Achse; die Summe ist ihr quadrierter
Singulärwert σ₁². Die zweite Achse ist die beste Richtung senkrecht zur
ersten usw. Formal: σ₁ = max ‖M v‖ über alle v mit ‖v‖ = 1.

*Rechenbeispiel:* Zeilen (3, 0) und (0, 1). Auf die Richtung (1, 0) fallen
3² + 0² = 9, auf (0, 1) fallen 0² + 1² = 1. Singulärwerte: σ₁ = 3, σ₂ = 1.

**Berechnung, praktisch:** Die Singulärwerte sind die Quadratwurzeln der
Eigenwerte von M Mᵀ. emb und die Abschnitte 1 und 2 des Notebooks berechnen
nur die k größten mit einem iterativen Verfahren (`scipy.sparse.linalg.svds`,
Lanczos-Verfahren); Abschnitt 3 berechnet alle auf einmal über die
Eigenzerlegung von M Mᵀ (Gram-Matrix).

**Nicht Varianz:** Varianz misst die Streuung um den Mittelwert. Das gilt nur,
wenn man von jeder Spalte zuerst ihren Mittelwert abzieht (Zentrierung); dann
ist die SVD dasselbe wie die Hauptkomponentenanalyse (PCA), und die
quadrierten Singulärwerte sind die erklärte Varianz. Die PPMI-Matrix wird in
emb nicht zentriert. Die erste Achse bildet deshalb vor allem ab, was allen
Zeilen gemeinsam ist (ihr allgemeines Niveau), nicht ihre Unterschiede. Das
Diagnose-Notebook spricht deshalb vom „Anteil an der quadrierten
Frobenius-Norm“ und ausdrücklich nicht vom erklärten Varianzanteil
(Abschnitt 3, Vorprüfung).

- **Englisch:** *singular value*; *eigenvalue*; *Gram matrix*;
  *Lanczos method*; Summe der quadrierten Werte einer Matrix:
  *squared Frobenius norm*; *centering*; *explained variance*;
  *principal component analysis (PCA)*
- **Status:** etabliert
- **Belegt durch:** Golub & Van Loan (2013) für die Eigenschaften der SVD
  (Lehrbuch der numerischen linearen Algebra); Jolliffe & Cadima (2016) für
  den Zusammenhang zwischen SVD, Zentrierung und PCA.

### Zentrierung

Von jeder Spalte wird ihr Mittelwert über alle Zeilen abgezogen. Danach
beschreiben die Werte nur noch die Abweichung vom Durchschnitt, und eine SVD
der zentrierten Matrix ist eine PCA.

**Vor der SVD (auf der PPMI-Matrix):** in emb nicht gemacht.

- Die PPMI-Matrix ist dünn besetzt: Fast alle Zellen sind 0 und werden nicht
  gespeichert. Nach dem Zentrieren ist jede Zelle von null verschieden.
  Größenordnung: 12.280 Zeilen × 120.882 Spalten, rund 1,48 Mrd. Zellen,
  etwa 12 GB bei 8 Byte je Wert. (Rechentechnisch lässt sich das umgehen,
  indem man die Zentrierung nur implizit in die SVD einrechnet.)
- Die 0 verliert ihre Bedeutung: In PPMI heißt 0 „nicht über der Erwartung“.
  Nach dem Zentrieren wird daraus „unter dem Durchschnitt“, ein negativer Wert,
  der für jedes nicht vorkommende Kontextwort einer Zeile in die Ähnlichkeit
  eingeht.
- Distributionelle Semantik mit SVD (LSA, PPMI-SVD) arbeitet herkömmlich
  unzentriert.

**Nach der SVD (auf den k-dimensionalen Vektoren):** billig und in der
Literatur empfohlen, in emb noch nicht getestet (Methodology 3.5, Ausblick
„Testen“). Mittelwertvektor abziehen, optional zusätzlich die obersten
Hauptkomponenten entfernen (*all-but-the-top*).

- **Englisch:** *mean-centering*, *centering*; *all-but-the-top*
- **Status:** etabliert
- **Belegt durch:**
  - Mu & Viswanath (2018): Mittelwert abziehen und die obersten
    Hauptkomponenten entfernen verbessert Wortvektoren; die obersten
    Komponenten kodieren vor allem Worthäufigkeit.
  - Schlechtweg et al. (2019): Mean-Centering verbessert die Messung von
    Bedeutungswandel.
  - Radovanović, Nanopoulos & Ivanović (2010): Punkte nahe am Mittelwert werden
    zu *Hubs* (Nachbarn sehr vieler anderer Punkte); Zentrieren wirkt dem
    entgegen.
- **Verhältnis zu p:** p < 1 dämpft die starken Achsen, Zentrieren entfernt
  den gemeinsamen Anteil. Beides zielt auf dasselbe Problem; ob Zentrieren
  zusätzlich hilft, ist für unsere Daten offen.

### Abgeschnittene SVD, Rang k

Nur die k wichtigsten Achsen werden behalten, der Rest wird verworfen. Jede
Zeile wird dann durch k Zahlen beschrieben, ihren **Vektor**.

- **Englisch:** *truncated SVD*, *rank k*, *low-rank approximation*
- **Status:** etabliert
- **Belegt durch:** Eckart & Young (1936) beweisen, dass die abgeschnittene
  SVD unter allen Matrizen mit Rang k die beste Annäherung an die
  Ausgangsmatrix ist (kleinste quadrierte Abweichung).
- **Beispiel:** emb verwendet k = 300 (letzter Lauf).

### Frobenius-Norm, erhaltener Anteil

**Quadrierte Frobenius-Norm:** die Summe aller quadrierten Einträge einer
Matrix, ‖M‖²_F = Σ m²ᵢⱼ. Sie ist zugleich die Summe der quadrierten
Singulärwerte. Schreibt man alle Einträge als einen langen Vektor, ist die
Frobenius-Norm dessen euklidische Länge (L₂-Norm).

*Nicht verwechseln:* Bei Matrizen bezeichnet „2-Norm“ meist die
**Spektralnorm** ‖M‖₂ = σ₁, den größten Singulärwert. Deshalb sagt man bei
Matrizen ausdrücklich „Frobenius“.

**Erhaltener Anteil bei Rang k:** (σ²₁ + … + σ²ₖ) / ‖M‖²_F. Er gibt an,
welcher Teil der Summe der quadrierten PPMI-Werte die abgeschnittene SVD
nachbildet. Ist die Summe 1000 und erhält Rang k davon 500, ist der Anteil
50 %.

**Warum quadriert, nicht wegen der Positivität:** Auf senkrecht zueinander
stehenden Achsen addieren sich die Quadrate (Pythagoras). Nur so lässt sich
die Summe auf die Achsen aufteilen. Außerdem minimiert die abgeschnittene SVD
die quadrierte Abweichung (Eckart & Young 1936); der erhaltene Anteil ist 1
minus deren relativer Wert.

**Kein Gütemaß:**
- Hohe PPMI-Werte entstehen vor allem bei seltenen Ereignissen und zählen
  quadriert besonders stark. Ein großer Teil der Norm ist deshalb Eigenheit
  einzelner Zeilen und Grundrauschen, das keine gemeinsame Achse abbilden
  kann.
- Ein niedriger Anteil heißt deshalb nicht, dass Wesentliches fehlt. Ob es
  fehlt, zeigen a_zeit und a_wort.

- **Englisch:** *(squared) Frobenius norm*, *spectral norm*, *retained
  energy*, *reconstruction error*
- **Status:** etabliert
- **Belegt durch:** Golub & Van Loan (2013) für die Definitionen beider
  Normen und ihren Zusammenhang mit den Singulärwerten.
- **Beispiel:** k = 300 erhält 18 %, k = 1.500 erhält 50 %. Trotzdem steigt
  a_zeit von 300 auf 1.500 nur von 0,682 auf 0,687, und a_wort sinkt von
  0,908 auf 0,868 (p = 0,5). Was über die ersten 300 Achsen hinausgeht, ist
  also für diese Messung weitgehend nicht reproduzierbar. Möglich bleibt,
  dass seltene, aber echte Nuancen darin liegen: a_zeit und a_wort messen
  Reliabilität, nicht Validität.

### Achse (latente Dimension)

Eine der k Richtungen, die die SVD findet. Eine Achse ist **kein
ausgewähltes Kontextwort**, sondern eine gewichtete Mischung **aller**
Kontextwörter (eine Spalte von V_k). Die Koordinate einer Zeile auf einer
Achse ist eine reelle Zahl: die PPMI-Werte der Zeile, mit den Gewichten der
Achse multipliziert und aufsummiert.

*Rechenbeispiel (erfunden):* Achse 7 = 0,12 · *Bundeswehr* + 0,10 ·
*Streitkräfte* + 0,08 · *Soldat* − 0,05 · *Rente* + … (alle übrigen
Kontextwörter mit kleinen Gewichten). Eine Zeile mit hohen PPMI-Werten bei
*Bundeswehr* und *Soldat* erhält auf dieser Achse einen hohen Wert.

Nach der SVD hat die Matrix also so viele Zeilen wie vorher, aber statt der
Kontextwort-Spalten k Achsen-Spalten mit reellen Zahlen. Die Zählungen und
die einzelnen Kontextwörter sind darin nicht mehr direkt zu sehen. Welche
Kontextwörter eine Achse prägen, lässt sich an ihren größten Gewichten
ablesen (vgl. Methodology 1.9, Kontextwortvektoren).

- **Englisch:** *latent dimension*, *latent factor*, *singular vector*
  (die Gewichte: *right singular vector*, Spalte von V)
- **Status:** etabliert
- **Belegt durch:** Deerwester et al. (1990), die die Achsen als „latente“
  Bedeutungsdimensionen deuten (daher *Latent* Semantic Analysis).
**Jedes Kontextwort auf allen Achsen:** Ein Kontextwort hat auf jeder der k
Achsen ein Gewicht, in der Regel überall von null verschieden, auf einigen
groß, auf den meisten klein. *Bundeswehr* kann z. B. auf einer Achse
„Militär“ und auf einer Achse „Haushalt“ zugleich hoch liegen. Die k Gewichte
eines Kontextworts bilden seinen Kontextwortvektor (Methodology 1.9).

Was nicht doppelt vorkommt, ist das **Muster**: Die Achsen stehen senkrecht
aufeinander (*orthogonal*). Jede Achse bildet nur ab, was die vorherigen noch
nicht abgebildet haben. Ein Wort kann also auf mehreren Achsen zählen, aber
jedes Mal mit einem anderen Anteil seines Verhaltens.

- **Häufiges Missverständnis:** Die SVD wählt nicht die k wichtigsten
  Kontextwörter aus. Das wäre ein Spaltenfilter (vgl. Methodology 2.2,
  verworfen).

### Eigenwertgewichtung, Exponent p

Legt fest, wie stark die einzelnen Achsen im Vektor einer Zeile zählen. Der
Vektor ist x = U_k · Σ_k^p: die Koordinaten der Zeile auf den k Achsen (U_k),
jede Achse multipliziert mit ihrem Singulärwert hoch p.

**Schreibweise:** σ_j ist der Singulärwert der Achse j, eine einzelne Zahl je
Achse, für alle Zeilen dieselbe. Σ_k (großes Sigma) ist die Diagonalmatrix
mit σ₁ … σ_k auf der Diagonalen. U_k · Σ_k^p bedeutet: Koordinate j jeder
Zeile mal σ_j^p. σ ist also nicht der Vektor, sondern das Gewicht je Achse.

*Rechenbeispiel (erfunden):* σ₁ = 400, σ₂ = 100, p = 0,5, also Gewichte
√400 = 20 und √100 = 10. Eine Zeile mit den Koordinaten (0,02; 0,05) in U_k
erhält den Vektor (0,02 · 20; 0,05 · 10) = (0,4; 0,5).

| p | Gewicht einer Achse | Wirkung |
|---|---|---|
| 1 | Singulärwert σ | die tatsächlichen Koordinaten der Zeile; die ersten Achsen dominieren |
| 0,5 | √σ | Mittelweg |
| 0 | 1 | alle k Achsen zählen gleich |

*Rechenbeispiel:* zwei Achsen mit σ₁ = 3 und σ₂ = 1. Verhältnis der
Gewichte bei p = 1: 3 : 1, bei p = 0,5: 1,73 : 1, bei p = 0: 1 : 1.

**Warum man die starken Achsen dämpft:**

1. *Gemeinsames statt Unterscheidendes.* Die ersten Achsen bilden in einer
   nicht zentrierten Matrix vor allem ab, was allen Zeilen gemeinsam ist
   (s. Singulärwert). Für die Frage „Worin unterscheiden sich zwei Zeilen?“
   tragen sie wenig bei.
2. *Der Kosinus folgt den großen Koordinaten.* Ist eine Achse 20-mal so stark
   gewichtet wie die übrigen, bestimmt sie den Kosinus fast allein. Alle
   Vektoren zeigen dann ungefähr in dieselbe Richtung, und die Unterschiede
   auf den späteren Achsen gehen unter.
3. *Empirisch besser.* Mit p < 1 schneiden Wortvektoren in mehreren
   Untersuchungen besser ab (s. Belege).

**Gegenrichtung:** Bei p = 0 zählen auch die schwachen Achsen voll. Sie tragen
eher Zufallsschwankungen. p ist deshalb ein Kompromiss; welcher Wert für diese
Daten am besten ist, prüft Abschnitt 3 des Diagnose-Notebooks.

- **Englisch:** *eigenvalue weighting*, *eigenvalue weighting exponent p*
  (auch *singular value weighting*)
- **Status:** etabliert
- **Belegt durch:**
  - Caron (2001) führt den Exponenten ein und zeigt an Retrieval-Aufgaben,
    dass Werte unter 1 besser abschneiden.
  - Bullinaria & Levy (2012) bestätigen das für Wortvektoren: Das Weglassen
    bzw. Abschwächen der ersten Achsen verbessert die Ergebnisse.
  - Levy, Goldberg & Dagan (2015) empfehlen für PPMI-SVD p = 0 oder p = 0,5
    statt p = 1.
- **Beispiel:** emb verwendet p = 0,5 (letzter Lauf). Welches p für diese Daten
  am besten ist, prüft Abschnitt 3 des Diagnose-Notebooks.
- **Befund Abschnitt 3 (letzter Lauf):** Auf a_zeit wirkt p kaum. Alle
  Kombinationen liegen zwischen 0,677 und 0,705. Die Grenze setzen die
  wenigen Belege, nicht der Raum. Stark wirkt p auf den **Volumeneffekt**
  (s. dort): Je kleiner p, desto kleiner ist er.

### Vektor

Die k Zahlen, die eine Zeile nach der SVD beschreiben. Grundlage aller
späteren Schritte (Nachbarn, Drift, Darstellung).

- **Englisch:** *vector*, *word vector*, *embedding*
- **Status:** etabliert
- **Beispiel:** 12.280 Zeilen mit je 300 Zahlen

### Kosinusähnlichkeit

Ähnlichkeit zweier Vektoren nach ihrer **Richtung**, unabhängig von ihrer
Länge: der Kosinus des Winkels zwischen ihnen. 1 = gleiche Richtung,
0 = senkrecht (nichts gemeinsam), −1 = entgegengesetzt. Berechnet als
Skalarprodukt der beiden Vektoren, geteilt durch das Produkt ihrer Längen.

Die Länge wird ausgeblendet, weil sie vor allem davon abhängt, wie viele und
wie starke PPMI-Werte eine Zeile hat, nicht davon, in welchen Kontexten das
Wort steht.

- **Englisch:** *cosine similarity*
- **Status:** etabliert
- **Belegt durch:** Turney & Pantel (2010), Überblick über Vektorraummodelle
  der Semantik; dort als das übliche Ähnlichkeitsmaß beschrieben.
- **Rechenbeispiel:** (3, 0) und (0, 1): Kosinus 0. (3, 0) und (1, 0):
  Kosinus 1, obwohl die Längen verschieden sind.

**Skala:**

| Wert | Winkel | Bedeutung hier                                |
|------|--------|-----------------------------------------------|
| 1    | 0°     | gleiche Richtung: identisches Kontextprofil   |
| 0    | 90°    | senkrecht: keine gemeinsamen Kontexte         |
| −1   | 180°   | entgegengesetzt                               |

- **Auf PPMI-Zeilen direkt:** nur 0 bis 1. Alle Einträge sind ≥ 0, nichts
  kann entgegengesetzt sein.
- **Nach der SVD** (k = 300, p = 0,5): Einzelne Komponenten können negativ
  werden, rechnerisch ist −1 bis 1 möglich. Praktisch sind die Werte fast
  immer positiv, weil die Matrix nicht zentriert ist und alle Zeilen eine
  gemeinsame Grundrichtung teilen (s. „Zentrierung“). Hoch sind sie damit
  nicht unbedingt: Benachbarte Halbjahre desselben Worts haben bei vollen
  Belegzahlen eine Drift von etwa 0,57, also einen Kosinus von etwa 0,43
  (Diagnose-Notebook, Tabelle 4.1).

### Kosinusdistanz, Drift

1 − Kosinusähnlichkeit. Im Projekt heißt die Kosinusdistanz zwischen den
Vektoren eines Zielworts in zwei benachbarten Zeitscheiben **Drift**
(Abschnitt 4 des Diagnose-Notebooks).

| Kosinus | Drift | Bedeutung                                   |
|---------|-------|---------------------------------------------|
| 1       | 0     | keine Verschiebung                          |
| 0       | 1     | Kontexte haben nichts mehr gemeinsam        |
| −1      | 2     | nur rechnerisch, kommt praktisch nicht vor  |

Praktisch heißt das: 0 = unverändert, größer = stärker verschoben.

Was man daraus **nicht** ablesen kann:

1. *Absolute Werte sind nicht direkt deutbar.* Auch ohne echten Wandel ist die
   Drift nicht 0, weil zwei Zeitscheiben verschiedene Stichproben von Reden
   sind. Wie groß dieses Grundrauschen ist, hängt von der Belegung ab: wenige
   Treffer, mehr Zufall, höhere Drift. Deshalb gibt es die Zufallserwartung
   und h\* = 30 (s. dort).
2. *Die Länge fällt heraus, der Volumeneffekt nicht ganz.* Der Kosinus
   ignoriert die Vektorlänge, schwach belegte Zeilen haben aber trotzdem eine
   systematisch andere Richtung (s. „Volumeneffekt“).
3. *Keine Distanz im mathematischen Sinn:* Die Dreiecksungleichung gilt nicht.
   Der Name ist trotzdem üblich; für den Vergleich zweier Zeitscheiben spielt
   das keine Rolle.

- **Englisch:** *cosine distance*; Drift: *semantic drift*, *semantic change
  score*
- **Status:** etabliert (Standardmaß der Forschung zum Bedeutungswandel,
  *semantic change detection*)
- **Belegt durch:** Hamilton, Leskovec & Jurafsky (2016b) messen Wandel als
  Kosinusdistanz zwischen den Vektoren eines Worts in zwei Zeiträumen;
  Kutuzov et al. (2018) beschreiben das in ihrem Überblick als übliches Maß.

---

## 4. Messverfahren

### Split-half-Prüfung

Die Reden jeder Zeitscheibe werden zufällig in zwei Hälften A und B geteilt.
Aus jeder Hälfte wird eine eigene PPMI-Matrix berechnet, mit den Randsummen
dieser Hälfte. Beide Hälften beschreiben dasselbe Wort in derselben
Zeitscheibe; jeder Unterschied zwischen ihnen ist deshalb Zufall der
Stichprobe, kein Bedeutungswandel. Eine Zeile ist reliabel, wenn sie in beiden
Hälften ähnlich herauskommt **und sich zugleich von anderen Zeilen
unterscheidet** (s. Reliabilität).

**Gleich groß** sind die Hälften nur in der Zahl der **Reden** je
Zeitscheibe, nicht in der Zahl der Belege einer Zeile. Eine Zeile kann in A
12 und in B 7 Belege haben, je nachdem, in welchen Reden ihre Belege stehen.
Wo das Notebook eine Zeile nach ihrer Trefferzahl einordnet, nimmt es deshalb
die **kleinere der beiden Hälften**.

Geteilt wird nach **Reden**, nicht nach Belegen: Belege derselben Rede
(derselbe Redner, dasselbe Thema, wiederholte Formulierungen) würden die
Hälften künstlich ähnlich machen.

- **Englisch:** *split-half reliability*, *split-half test*
- **Status:** etabliert
- **Belegt durch:**
  - Spearman (1910) und Brown (1910) führen die Teilung eines Tests in zwei
    Hälften als Schätzung der Reliabilität ein (Psychometrie).
  - Antoniak & Mimno (2018) prüfen die Stabilität von Wortvektoren, indem sie
    die Dokumente eines Korpus neu ziehen; sie zeigen, dass Nachbarn und
    Ähnlichkeiten bei kleinen Korpora stark vom Zufall der Stichprobe abhängen.
- **Beispiel:** 20 Teilungen (`N_SPLITS`), Zufallszahlen aus `SEED = 42`.

### Teilung

Eine zufällige Aufteilung der Reden in die Hälften A und B. Alle Abschnitte
des Notebooks verwenden dieselben 20 Teilungen.

- **Englisch:** *split*
- **Status:** etabliert

### Folding-in

Eine Zeile, die nicht in die SVD eingegangen ist, wird nachträglich auf deren
Achsen abgebildet. Die SVD zerlegt die PPMI-Matrix in drei Teile:

| Teil | Dimension | Inhalt |
|---|---|---|
| U_k | Zeilen × k | Koordinaten der Zeilen, an denen die SVD berechnet wurde |
| Σ_k | k × k (Diagonale) | die k Singulärwerte |
| V_k | Kontextwörter × k | Gewicht jedes Kontextworts auf jeder Achse; legt die Achsen fest |

V_k ist die „Übersetzungstabelle“ von Kontextwörtern in Achsen. Eine neue
Zeile m (ein PPMI-Wert je Kontextwort) wird mit V_k multipliziert und mit den
Singulärwerten gewichtet: x = m · V_k · Σ_k^(p−1). Jede der k Koordinaten ist
ein Skalarprodukt der Zeile mit einer Achse. Das ist eine feste Rechnung
(Projektion), keine Auswahl. Voraussetzung: dieselben Kontextwörter in
derselben Reihenfolge als Spalten; fehlt eines, steht dort 0. Für die Zeilen,
an denen die SVD berechnet wurde, ergibt die Formel genau U_k Σ_k^p wie in emb.

**Verwendung im Notebook:** SVD an Hälfte A, Zeilen von B per folding-in
abgebildet, dann umgekehrt; die Kennwerte werden gemittelt. B bestimmt die
Achsen also nicht mit, sondern muss sich in einem Raum bewähren, den es nicht
mitgestaltet hat (wie Trainings- und Testteil im maschinellen Lernen). Eine
SVD über beide Hälften würde die Achsen auf beide zuschneiden und sie
künstlich ähnlich machen. Ebenso eingefaltet werden neu berechnete Zeilen
(Teilstichproben in Abschnitt 1 und 4, Permutationen in Abschnitt 4).

- **Englisch:** *folding-in*, *projection*, *out-of-sample extension*
- **Status:** etabliert
- **Belegt durch:** Deerwester et al. (1990) und Berry, Dumais & O'Brien (1995)
  beschreiben folding-in für LSA: neue Dokumente oder Terme werden über die
  vorhandenen singulären Vektoren in den Raum abgebildet, ohne die SVD neu zu
  berechnen.
- **Rechenbeispiel:** n Zeilen von B (n × 120.882) mal V_k (120.882 × 300)
  ergibt n × 300; danach jede der 300 Spalten mal σ_j^(p−1).

### Reliabilität und Validität

**Reliabilität:** Kommt bei Wiederholung der Messung dasselbe heraus, und
zwar so, dass die gemessenen Einheiten unterscheidbar bleiben? Beides gehört
zusammen: Ein Verfahren, das allen Zeilen fast denselben Vektor gibt, wäre
perfekt wiederholbar, aber nutzlos. In der Psychometrie ist Reliabilität
deshalb der Anteil der Unterschiede *zwischen* den Einheiten an der gesamten
Streuung; die Schwankung *innerhalb* einer Einheit (zwischen zwei Messungen)
ist der Fehleranteil. Im Notebook: Eine Zeile ist reliabel, wenn ihre beiden
Hälften einander ähnlicher sind als den Vergleichszeilen (a_wort, a_zeit).

**Validität:** Misst das Verfahren das, was es messen soll (hier: Bedeutung)?
Die Split-half-Prüfung misst nur Reliabilität. Eine Zeile kann reproduzierbar
sein und trotzdem vor allem das Thema eines Halbjahrs abbilden statt der
Bedeutung des Zielworts.

- **Englisch:** *reliability*, *validity*; Unterschiede zwischen den
  Einheiten: *between-subject variance*; Schwankung innerhalb:
  *within-subject variance*, *measurement error*
- **Status:** etabliert (Grundbegriffe der Messtheorie)
- **Belegt durch:** Lord & Novick (1968), Standardwerk der klassischen
  Testtheorie; dort Reliabilität als Anteil der „wahren“ Varianz an der
  Gesamtvarianz.

### a_wort und a_zeit

Zwei Kennwerte der Reliabilität einer Zeile. Beide fragen: Ist die Zeile ihrer
eigenen zweiten Hälfte ähnlicher als Vergleichszeilen? Sie unterscheiden sich
nur in den Vergleichszeilen:

- **a_wort:** die anderen Zielwörter **derselben Zeitscheibe**. Hoch, wenn das
  Profil der Zeile reproduzierbar ist und sie sich von anderen Wörtern
  unterscheiden lässt.
- **a_zeit:** **dasselbe Zielwort** in den anderen Zeitscheiben. Hoch, wenn
  sich das, was eine Zeitscheibe von den anderen Zeitscheiben desselben Worts
  unterscheidet, in beiden Hälften zeigt. Die strengere Frage, weil sich Zeilen
  desselben Worts meist ähneln; und die für Drift entscheidende.

Wert = Anteil der Vergleichszeilen, die weniger ähnlich sind als die eigene
zweite Hälfte (Gleichstände zur Hälfte). Verglichen wird immer über die
Hälften hinweg: eigene Ähnlichkeit = A gegen B derselben Zeile;
Vergleichsähnlichkeit = A gegen B der Vergleichszeile und umgekehrt, gemittelt.
Beide Seiten tragen so dasselbe Teilungsrauschen.

- **0,5** = Zufallserwartung: Die eigene Hälfte liegt mitten unter den
  Vergleichszeilen, der Vektor ist Grundrauschen.
- **1** = ähnlicher als alle Vergleichszeilen. Nicht immer erreichbar und nicht
  nötig: Bedeutungsnahe Wörter (z. B. *Krieg*, *Angriffskrieg*) dürfen sich
  ähnlicher sein als zwei verrauschte Hälften desselben Worts.

**Bezug zur AUC.** Man kann a_zeit als Ergebnis eines Zuordnungsspiels lesen.
Gegeben ist Hälfte A von *Sicherheit, 2022-H1*. Gesucht ist unter den
B-Hälften von *Sicherheit* aller 26 Halbjahre die eigene. Die Regel lautet
„je höher der Kosinus zu A, desto eher die eigene“. Die eigene B-Hälfte ist
der **positive Fall**, die 25 anderen sind **negative Fälle**.

- **ROC-Kurve:** Man verschiebt eine Kosinus-Schwelle von oben nach unten und
  erklärt alles darüber zur „eigenen Hälfte“. Für jede Schwelle trägt man ab,
  welcher Anteil der positiven Fälle schon darüber liegt (Trefferquote) und
  welcher Anteil der negativen (Fehlalarmquote). Das ergibt die Kurve.
- **AUC:** die Fläche unter dieser Kurve. Sie ist gleich der
  Wahrscheinlichkeit, dass der positive Fall einen höheren Kosinus hat als
  ein zufällig gezogener negativer. Bei nur einem positiven Fall ist das
  genau der Anteil der Vergleichszeilen mit kleinerem Kosinus, also a_zeit.

*Rechenbeispiel:* Die eigene Hälfte hat den Kosinus 0,62, und 20 der 25
anderen Halbjahre liegen darunter. Dann ist a_zeit = 20/25 = 0,8. In 80 %
der Paarvergleiche erkennt die Regel die eigene Hälfte richtig.

**Warum ein Anteil und kein Kosinus?** Kosinuswerte zwischen Zeilen desselben
Worts sind insgesamt hoch und hängen von der Belegzahl ab. Ein Kosinus von
0,62 kann deshalb gut oder schlecht sein, je nachdem, wie ähnlich sich die
übrigen Halbjahre sind. Die AUC zählt nur die **Reihenfolge**. Sie ist für
jedes Wort und jede Trefferzahl auf dieselbe Skala gebracht: 0,5 heißt
Raten, 1 heißt immer richtig. Deshalb lässt sich eine einzige Schwelle
(0,7) über alle Zeilen anwenden. a_wort ist dasselbe Spiel mit den
B-Hälften der anderen Zielwörter desselben Halbjahrs als negativen Fällen.

- **Englisch:** *AUC* (*area under the ROC curve*), *ROC* (*receiver
  operating characteristic*), *probability of superiority*, normierte
  *Mann–Whitney U*; a_wort und a_zeit selbst: eigener Sprachgebrauch
- **Status:** AUC etabliert; die Anwendung auf Hälften (eigene vs. andere
  Einheiten) etabliert; die Namen a_wort/a_zeit sind eigener Sprachgebrauch
- **Belegt durch:**
  - Hanley & McNeil (1982): AUC als Wahrscheinlichkeit, dass ein positiver Fall
    höher eingestuft wird als ein negativer.
  - Mann & Whitney (1947): die gleichwertige Rangstatistik U.
  - Haxby et al. (2001): Vergleich der Ähnlichkeit derselben Einheit über zwei
    Datenhälften mit der Ähnlichkeit zu anderen Einheiten (fMRT, Split-half).
  - Hosmer & Lemeshow (2000): 0,7 als übliche Grenze akzeptabler Trennschärfe
    einer AUC (Konvention).
- **Beispiel:** Die Schwelle h\* richtet sich nach **a_zeit**: kleinste
  Trefferzahl in `H_GRID`, bei der der Mittelwert von a_zeit über die
  Prüfzeilen in **allen** 20 Teilungen mindestens 0,7 (`AUC_LEVEL`) beträgt,
  ebenso bei jedem größeren h. Letzter Lauf: h\* = 30.

### Grundrauschen, Zufallserwartung (früher: Rauschboden)

**Grundrauschen:** die Unterschiede, die **ohne jeden echten Effekt** allein
durch den Zufall der Stichprobe entstehen (welche Belege zufällig vorliegen).
Bei der PPMI-SVD umso stärker, je weniger Belege eine Zeile hat.

**Zufallserwartung:** die gemessene Größe dazu, also wie groß ein Unterschied
im Mittel ist, wenn es keinen echten Effekt gibt. Erst was deutlich darüber
liegt, ist ein Befund.

Im Diagnose-Notebook kommen beide Wörter nicht vor. Die Sache steckt in der
Split-half-Prüfung (zwei Hälften derselben Zeile) und in Abschnitt 4 als
„Drift ohne Unterschied zwischen den Zeitscheiben“ bzw.
Permutationserwartung.

*Zum Wort „Rauschboden“:* `Readme Pipeline.md` verwendet es zusätzlich für
den SGNS-Pfad (Rauschen des Trainingsverfahrens durch den Zufallsstartwert).
Das ist etwas anderes und wird hier nicht behandelt. „Rauschboden“ wird in
diesem Glossar nicht mehr verwendet.

- **Englisch:** *background noise* (Grundrauschen); *noise floor* (Pegel des
  Grundrauschens, Messtechnik). In der Literatur zu Bedeutungswandel für die
  gemessene Größe meist *null baseline*, *control condition* oder
  *permutation baseline*.
- **Status:** „Grundrauschen“ ist allgemeinsprachlich und anschaulich, aber
  kein Fachbegriff der Statistik; geeignet für Erklärungen. Für die gemessene
  Größe in Veröffentlichungen „Zufallserwartung“ oder „Kontrollbedingung“.
- **Belegt durch:** Dubossarsky, Weinshall & Grossman (2017) führen für die
  Messung von Bedeutungswandel eine Kontrollbedingung ein (Korpus, in dem die
  Zeitscheiben zufällig durchmischt sind) und zeigen, dass ein großer Teil des
  scheinbaren Wandels auch dort auftritt.

### Gepaarter Vergleich

Alle Abschnitte verwenden **dieselben** 20 Teilungen. Werden zwei Varianten
verglichen (z. B. verschiedene Trefferzahlen h in Abschnitt 1, R und F(N) in
Abschnitt 2), dann unterscheiden sie sich in jeder Teilung nur in der
Variante, nicht im Zufall der Teilung. Verglichen wird die **Differenz je
Teilung**. Das macht kleine Unterschiede sichtbar, die sonst im
Teilungsrauschen untergingen.

- **Englisch:** *paired comparison*, *paired design*; in Simulationen
  *common random numbers*
- **Status:** etabliert

### Belastbar

Eine Änderung gilt im Notebook als belastbar, wenn sie in **allen 20
Teilungen** dasselbe Vorzeichen hat. Das ist eine strenge, einfache Regel
anstelle eines Signifikanztests.

Zwei Einschränkungen:

- **Kein p-Wert:** Die 20 Teilungen beruhen auf denselben Daten und sind
  nicht unabhängig. Die Wahrscheinlichkeit 2 · 0,5²⁰ ≈ 0,000002, die man bei
  unabhängigen Teilungen für zufällig gleiche Vorzeichen erhielte, gilt
  deshalb nicht.
- **Nichts über die Größe:** „Belastbar“ sagt nur, dass die Richtung stabil
  ist, nicht, dass der Unterschied groß ist.

- **Englisch:** sinngemäß *consistent across all splits*; verwandt:
  *sign test*
- **Status:** eigener Sprachgebrauch (Festlegung des Notebooks)
- **Beispiel:** Abschnitt 2, N = 5: a_zeit steigt belastbar, aber nur um
  0,0044.

### Hauptmessung und Gegenprobe (Abschnitt 1)

Beide fragen, wie a_zeit und a_wort von der Trefferzahl h abhängen. Sie tun
es auf zwei Wegen.

**Hauptmessung (kontrollierte Teilstichproben):**

- Genommen werden nur gut belegte Zeilen, mit mindestens 50 Belegen
  (`H_MAX`) in **jeder** Hälfte.
- Jede dieser Zeilen wird künstlich auf h Belege je Hälfte verkleinert. Die
  Belege werden dabei redenweise gezogen und neu berechnet, dann per
  folding-in abgebildet.
- Weil jede Prüfzeile alle h aus `H_GRID` erreichen kann, wird jedes h an
  **denselben Zeilen** gemessen. Ein Unterschied zwischen h = 5 und h = 30
  liegt dann nur an der Belegzahl, nicht an den Wörtern. Es ist ein
  kontrolliertes Experiment.

**Gegenprobe (natürliche Zeilen):**

- Alle Zeilen werden so genommen, wie sie sind, und nach ihrer tatsächlichen
  Trefferzahl gruppiert (s. Gruppierung nach Trefferzahl).
- Das ist realistisch, aber **konfundiert**: Selten belegte Zeilen sind auch
  andere Wörter (Fachwörter, Komposita, Randzeiten eines Worts). Ein
  Unterschied zwischen den Gruppen kann also an der Belegzahl liegen oder an
  den Wörtern.

**Deutung:**

| Gegenprobe gegenüber Hauptmessung | Bedeutung für h\* |
|---|---|
| schlechter | h\* ist eine Untergrenze. |
| besser | h\* ist vorsichtig gewählt. |

- **Englisch:** Hauptmessung: *subsampling*, *downsampling*,
  *within-item design*; Gegenprobe: *observational comparison*; das Problem:
  *confounding*, *confounder*
- **Status:** die Verfahren sind etabliert; die Namen „Hauptmessung“ und
  „Gegenprobe“ sind eigener Sprachgebrauch

### Gruppierung nach Trefferzahl

In der Gegenprobe kommt jede Zeile in eine Gruppe nach der Belegzahl ihrer
**kleineren** Hälfte. Die Gruppengrenzen sind die Werte aus `H_GRID`:

| Gruppe | 1 | 2 | 3 | 5 | 7 | 10 | 15 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| Belege | 1 | 2 | 3–4 | 5–6 | 7–9 | 10–14 | 15–19 | 20–29 | 30–49 | ≥ 50 |

- **Breite:** Die Gruppen werden nach oben breiter, weil ein Beleg mehr bei
  kleinen Zahlen viel ausmacht und bei großen kaum.
- **Vergleichbarkeit:** Die Gruppen passen zu den h der Hauptmessung, und
  beide Kurven lassen sich direkt vergleichen.
- **Mindestgröße:** Gruppen mit im Mittel weniger als 10 Zeilen
  (`MIN_ROWS_GROUP`) zählen für die Entscheidung nicht.
- **Wechsel:** Je nach Teilung kann dieselbe Zeile in verschiedenen Gruppen
  landen.

- **Englisch:** *binning*, *frequency bins*
- **Status:** etabliert

### Mindesttrefferzahl h\* und Zufallsgrenze h₀

**h\*:** die kleinste Trefferzahl in `H_GRID`, ab der a_zeit verlässlich gut
genug ist. Dazu muss der Mittelwert von a_zeit über die Prüfzeilen in
**allen** 20 Teilungen mindestens 0,7 (`AUC_LEVEL`) betragen, und zwar bei
diesem h und bei jedem größeren h. Grundlage ist die Hauptmessung.

**h₀:** dieselbe Regel mit dem Kriterium „über 0,5“. Ab h₀ ist eine Zeile
besser als Raten, aber noch nicht verlässlich.

h\* ist eine **Schwelle für die Deutung**: Zeilen darunter sind unsicher und
sollten markiert werden. Ob sie auch aus der SVD herausgenommen werden
sollen, ist eine andere Frage; sie wird in Abschnitt 2 beantwortet (s.
Fit-Ausschluss und Markierung).

Die Schwelle gilt nur für dieses Korpus, dieses Raster und diese k und p.

- **Englisch:** *minimum frequency threshold*, *frequency cut-off*
- **Status:** Mindesthäufigkeiten sind etabliert; die Bestimmung über
  Split-half und AUC ist eigene Festlegung
- **Beispiel:** letzter Lauf h\* = 30. Das sind 75 % der Zeilen unter der
  Schwelle (Median der Trefferzahl: 9).

---

## 5. Schwach belegte Zeilen und SVD (Abschnitt 2)

### Fit-Ausschluss und Markierung

Mit schwach belegten Zeilen kann man auf zwei Arten umgehen:

| | Fit-Ausschluss (`MIN_HITS_FIT` in emb) | Markierung |
|---|---|---|
| SVD berechnet an | nur Zeilen mit mindestens N Belegen | allen Zeilen |
| übrige Zeilen | per folding-in projiziert, `is_fitted = false` | wie alle anderen |
| wirkt auf | die Achsen des Raums, also alle Zeilen | nur die Anzeige |

- **Fit-Ausschluss:** Er lohnt sich nur, wenn die schwachen Zeilen den Raum
  für die übrigen verschlechtern. Das prüft Abschnitt 2, indem er die
  Referenz **R** (SVD an allen Zeilen) mit **F(N)** (SVD nur an Zeilen ab N
  Belegen) vergleicht.
- **Markierung:** Sie erfolgt heute nur über `freq` (= `n_hits`) im Bundle.
  Die Schwelle selbst wird nicht weitergegeben (s. nextsteps).

- **Englisch:** Fit-Ausschluss: *fitting on a subset*, *out-of-sample
  projection* der übrigen; Markierung: *flagging*
- **Status:** eigener Sprachgebrauch
- **Beispiel:** Letzter Lauf: Bei N = h\* = 30 sinkt a_zeit belastbar
  (−0,011). Die SVD bleibt deshalb an allen Zeilen (`MIN_HITS_FIT = None`).

### Hebelwert

Der Hebelwert misst, wie stark eine Zeile die k Achsen der SVD mitbestimmt.
Er ist die quadrierte Länge ihrer Zeile in U_k.

- **Summe:** Summiert über alle Zeilen ergibt er genau k.
- **Gleichverteilung:** Bestimmten alle Zeilen die Achsen gleich stark, hätte
  jede k / Zeilenzahl.
- **Vorprüfung in Abschnitt 2:** Verglichen wird der Anteil der schwachen
  Zeilen an den Hebelwerten mit ihrem Anteil an den Zeilen. Ist der Anteil an
  den Hebelwerten viel größer, bestimmen sie die Achsen überproportional mit.

- **Englisch:** *statistical leverage score*, *leverage*
- **Status:** etabliert (aus der Regressionsdiagnostik; für die SVD:
  Mahoney & Drineas 2009)
- **Belegt durch:** Mahoney & Drineas (2009) definieren den Hebelwert einer
  Zeile über die singulären Vektoren und nutzen ihn, um die für eine
  Zerlegung wichtigsten Zeilen zu bestimmen.
- **Beispiel:** Zeilen unter 10 Belegen machen 52 % der Zeilen aus, aber nur
  8 % der Hebelwerte (Verhältnis 0,16). Sie bestimmen die Achsen also
  **unterproportional**.

### Pareto-Verbesserung

Eine Variante ist besser, wenn mindestens ein Kennwert belastbar steigt und
keiner belastbar sinkt, auch in keiner Teilgruppe. In Abschnitt 2 heißt das:
a_zeit oder a_wort steigt über alle Bewertungszeilen, und in keinem Terzil
der Textmenge sinkt einer.

- **Ergebnis, wenn beides auftritt** (etwas steigt, etwas sinkt):
  **Zielkonflikt**.
- **Englisch:** *Pareto improvement*; Zielkonflikt: *trade-off*
- **Status:** etabliert (Ökonomie, Optimierung); die Übertragung auf die
  Kennwerte ist eigene Festlegung
- **Beispiel:** Abschnitt 2, N = 3, 5, 7: Pareto-Verbesserung, aber mit
  Anstiegen von höchstens 0,016.

### Bewertungszeilen und Terzile

**Bewertungszeilen:** die Zeilen mit mindestens N Belegen in **beiden**
Hälften. Nur sie gehen in beiden Richtungen in die SVD ein, und nur an ihnen
werden R und F(N) verglichen.

**Terzile:** Die Bewertungszeilen jeder Teilung werden nach ihrer
Trefferzahl in drei gleich große Gruppen geteilt (geringe, mittlere, hohe
Textmenge). So zeigt sich, ob der Eingriff einer Gruppe hilft und einer
anderen schadet.

*Auffälligkeit im letzten Lauf:* Bei N = 30 umfasst das untere Terzil Zeilen
mit 30–55 Belegen. Für eine Bewertungszeile bräuchte eine Zeile aber
mindestens 30 Belege in beiden Hälften, also mindestens 60 insgesamt. Die
Gruppe ist daher leer, und die Regel liefert „nicht berechenbar“. Ohne diese
Gruppe ergäbe sich „Verschlechterung“. Seit 2026-11-20 bildet das Notebook die
Terzile deshalb aus den Bewertungszeilen jeder Teilung (noch nicht neu
gerechnet).

- **Englisch:** *evaluation set*; *tertiles*
- **Status:** Terzile etabliert; „Bewertungszeilen“ eigener Sprachgebrauch

---

## 6. Rang k und Exponent p (Abschnitt 3)

### Pearson-Korrelation r

Misst, wie stark zwei Größen linear zusammenhängen. r liegt zwischen −1 und
1. Bei 0 gibt es keinen linearen Zusammenhang. r² ist der Anteil der
Streuung der einen Größe, der rechnerisch mit der anderen einhergeht.

**Andere Skala als a_zeit und a_wort:**

| | Skala | „kein Effekt“ | gut |
|---|---|---|---|
| a_zeit, a_wort (AUC) | 0 bis 1 | 0,5 = Raten | möglichst hoch |
| Volumeneffekt (r) | −1 bis 1 | 0 = keine Abhängigkeit | möglichst nah an 0 |

- **Englisch:** *Pearson correlation coefficient*; r²: *coefficient of
  determination*
- **Status:** etabliert
- **Beispiel:** r = 0,527 ergibt r² ≈ 0,28.

### Volumeneffekt

Misst, ob die Lage eines Vektors von der Textmenge seiner Zeile abhängt
statt nur von der Bedeutung. Gemessen wird an der Hälfte, an der die SVD
berechnet wird. Für jede Zeile gibt es zwei Zahlen:

- **x:** der Logarithmus der Zahl ihrer Kontexttokens, also ihre Textmenge.
  Logarithmiert, weil der Schritt von 10 auf 100 so viel zählen soll wie der
  von 100 auf 1.000.
- **y:** ihr mittlerer Kosinus zu den übrigen Zeitscheiben desselben Worts,
  also wie typisch sie für ihr Wort ist.

Der Volumeneffekt ist Pearsons r zwischen x und y über alle Zeilen aller
Wörter („gepoolt“). Bewertet wird der Betrag |r|; kleiner ist besser.

**Warum r positiv ist:**

1. **Rauschen:** Schwach belegte Zeilen haben verrauschte Vektoren, die
   allem weniger ähneln. Dieser Teil ist statistisch unvermeidlich; auch
   ein idealer Raum zeigt ihn.
2. **Gemeinsamer Anteil:** Die starken ersten Achsen bilden ab, was allen
   Zeilen gemeinsam ist. Gut belegte Zeilen liegen nah an diesem Profil.
   - Großes p (die starken Achsen zählen viel) erhöht r.
   - Großes k (mehr Achsen ohne diesen Anteil) senkt r.

**Ziel ist 0, aber nicht um jeden Preis:**
- **Echter Wandel:** Über ein Wort wird oft gerade dann viel geredet, wenn
  sich sein Gebrauch ändert. Ein Teil von r kann deshalb inhaltlich
  begründet sein.
- **Zweideutiges r ≈ 0:** Es kann heißen, dass der Raum nicht mehr von der
  Textmenge abhängt. Es kann aber auch heißen, dass er alle Zeilen einander
  gleich unähnlich macht. Ein gleichzeitig fallendes a_wort spricht für das
  Zweite.

**Gepoolt und innerhalb der Wörter:** Abschnitt 3 poolt über alle Wörter,
sodass auch Unterschiede zwischen häufigen und seltenen Wörtern in r
eingehen. Abschnitt 4 rechnet die Volumenabhängigkeit der Drift dagegen
**innerhalb** der Zielwörter. Die beiden r sind nicht direkt vergleichbar.

- **Englisch:** *frequency effect*, *frequency bias*; gepoolt: *pooled
  correlation*; innerhalb: *within-subject correlation*
- **Status:** „Volumeneffekt“ eigener Sprachgebrauch; Häufigkeitseffekte als
  Artefakt bei der Messung von Bedeutungswandel sind etabliert
- **Belegt durch:**
  - Dubossarsky, Weinshall & Grossman (2017) zeigen, dass gemessener
    Bedeutungswandel zum Teil von der Worthäufigkeit herrührt und auch in
    einer Kontrollbedingung ohne echten Wandel auftritt.
  - Bland & Altman (1995) beschreiben die Korrelation innerhalb von
    Personen (hier: Zielwörtern), die Abschnitt 4 verwendet.
- **Beispiel (letzter Lauf):**

  | | p = 1,0 | p = 0,5 | p = 0,0 |
  |---|---|---|---|
  | k = 10 | 0,623 | 0,581 | 0,551 |
  | k = 300 | 0,672 | 0,527 (Referenz) | 0,450 |
  | k = 1.500 | 0,495 | 0,188 | 0,026 |

### Pareto-Front

Die Kombinationen aus k und p, die von keiner anderen in **allen** Kennwerten
zugleich übertroffen werden: höheres oder gleiches a_zeit, höheres oder
gleiches a_wort und kleineres oder gleiches |Volumeneffekt|. Unter ihnen
gibt es keine eindeutig beste, nur eine Abwägung (s. Pareto-Verbesserung,
Zielkonflikt).

- **Englisch:** *Pareto front*, *Pareto frontier*, *non-dominated set*
- **Status:** etabliert
- **Beispiel (letzter Lauf):** Alle 35 Kombinationen sind gegenüber der
  Referenz ein Zielkonflikt.
  - k = 1.500, p = 0 senkt den Volumeneffekt auf 0,026, aber a_wort fällt
    von 0,908 auf 0,849.
  - k = 25, p = 0,5 hebt a_zeit auf 0,700 und a_wort auf 0,938, aber der
    Volumeneffekt steigt auf 0,688.

## 7. Drift und Textmenge (Abschnitt 4)

### Prüfpaare, Paare

**Paar:** zwei benachbarte Zeitscheiben desselben Zielworts, z. B.
Sicherheit 2015-H1 → 2015-H2. Ausgewertet werden Paare, deren beide Zeilen
mindestens h\* = 30 Belege haben (`DRIFT_FLOOR`); im letzten Lauf 2.275.

**Prüfpaare:** die Paare, deren beide Zeilen mindestens 50 Belege haben
(`H_MAX`). Nur aus ihnen lassen sich Teilstichproben bis 50 Belege ziehen;
sie tragen die Sättigungskurve.

- **Englisch:** *pair of adjacent time slices*

### Korrelation innerhalb der Wörter

Pearson-Korrelation r (s. dort), bei der von beiden Größen je Zielwort zuerst
der Mittelwert über seine Paare abgezogen wird. Damit zählt nur, wie die
Drift **eines** Worts mit der Belegung **seiner** Paare schwankt, nicht, ob
häufige Wörter insgesamt anders driften als seltene. Ziel ist r = 0.
Negativ heißt: Ein Wort driftet zwischen schwach belegten Zeitscheiben
stärker als zwischen gut belegten.

- **Englisch:** *within-subject correlation*, *within-group correlation*
- **Status:** etabliert
- **Belegt durch:** Bland & Altman (1995), Korrelation bei wiederholten
  Messungen an denselben Einheiten.
- **Beispiel:** Rohdrift r = −0,26 (Tabelle 4.2).

### Resampling mit Zurücklegen (statt „Bootstrap“)

Verfahren, um abzuschätzen, wie stark ein Ergebnis bei einer anderen
Auswahl schwanken würde, ohne eine zweite Auswahl zu haben. Aus den
vorhandenen Zielwörtern wird eine neue Liste gleicher Länge **mit
Zurücklegen** gezogen: Manche Wörter kommen doppelt vor, andere fehlen. An
dieser Liste wird r neu berechnet; das 1.000-mal (`DRIFT_N_BOOT`).

**Ganze Wörter, nicht einzelne Paare:** Die Paare eines Worts hängen
zusammen (gleiches Wort, ähnliche Belegung). Zöge man Paare einzeln, wäre
das Intervall zu schmal. Gezogen wird deshalb jedes Wort mit allen seinen
Paaren (Resampling ganzer Gruppen).

**95-%-Intervall:** Die 1.000 Werte werden sortiert, die unteren und oberen
2,5 % abgeschnitten (Perzentilintervall). Liegt es ganz auf einer Seite der
0, gilt der Wert als **belastbar** von 0 verschieden. Bei Δ|r| werden
Variante und Rohdrift in jeder Ziehung an denselben Wörtern berechnet
(gepaarter Vergleich).

| | Teilungen (Abschnitt 1–3) | Resampling (Abschnitt 4) |
|---|---|---|
| neu gezogen | Reden, auf zwei Hälften verteilt | Zielwörter, mit Zurücklegen |
| wie oft | 20 | 1.000 |
| „belastbar“ | gleiches Vorzeichen in allen Teilungen | Intervall schließt 0 aus |
| Frage | stabil gegenüber dem Zufall der Belege? | stabil gegenüber der Auswahl der Wörter? |

*Zum Wort:* Im Notebook und in der Literatur heißt das Verfahren
**Bootstrap**. Der Name ist wenig anschaulich; in diesem Glossar wird
„Resampling mit Zurücklegen“ bevorzugt.

- **Englisch:** *resampling with replacement* (bevorzugt); *bootstrap*;
  ganze Gruppen: *cluster bootstrap*, *block bootstrap*; *percentile
  interval*
- **Status:** etabliert, Standardverfahren der Statistik
- **Belegt durch:** Efron & Tibshirani (1993), Grundlagenwerk; Davison &
  Hinkley (1997) zum Ziehen ganzer Gruppen.
- **Beispiel:** Rohdrift r = −0,260, 95-%-Intervall −0,330 bis −0,192.

### Sättigungskurve

Für die Prüfpaare werden aus jeder Zeile genau h Belege gezogen (h = 1, 2,
3, 5, … 50) und die Drift berechnet (**beobachtet**). Ebenso werden aus den
zusammengelegten Belegen beider Zeilen zwei Stichproben von je h Belegen
gezogen; ihr Abstand ist die Drift **ohne Unterschied** zwischen den
Zeitscheiben, also reines Grundrauschen. Das Verhältnis zeigt, wie viel der
beobachteten Drift auch ohne Unterschied schon da wäre.

| Belege je Zeile | beobachtet | ohne Unterschied | Verhältnis |
|---|---|---|---|
| 1 | 0,851 | 0,841 | 99 % |
| 10 | 0,689 | 0,605 | 88 % |
| 30 (= h\*) | 0,589 | 0,389 | 66 % |
| 50 | 0,560 | 0,290 | 52 % |
| alle (≥ 50) | 0,574 | 0,119 | 21 % |

Lesart:
- Unter etwa 10 Belegen ist die Drift fast nur Rauschen.
- **h\* = 30 heißt nicht „rauschfrei“:** Wiedererkennen einer Zeile
  (Abschnitt 1) ist leichter als ein verlässlicher Abstand zwischen zwei
  Zeilen.
- Die Kurve flacht ab, erreicht aber im Raster keine Sättigung; eine
  natürliche Schwelle gibt es nicht. Eine höhere Mindestbelegung ist deshalb
  keine Lösung, sondern ein Vergleich mit der Zufallserwartung.
- Das Verhältnis ist **kein additiver Anteil** (s. Nichtadditivität).

- **Englisch:** *saturation curve*; Verfahren: *rarefaction* (Stichproben
  gleichen Umfangs vergleichen)
- **Status:** Rarefaktion etabliert (Ökologie); „Sättigungskurve“ ist die
  übliche deutsche Bezeichnung für solche Kurven.
- **Belegt durch:** Sanders (1968), Hurlbert (1971).

### Permutationstest, Permutationserwartung, p-Wert

**Permutationstest:** Die Reden beider Zeitscheiben eines Paars werden
zusammengeworfen und 99-mal (`DRIFT_N_PERM`) zufällig auf zwei Zeilen mit
den echten Belegzahlen verteilt. Jede Verteilung ergibt eine Drift, wie sie
ohne Unterschied zwischen den Zeitscheiben entstünde.

**Permutationserwartung:** der Mittelwert dieser Driftwerte, also die
Zufallserwartung für genau dieses Paar.

**p-Wert:** Anteil der Verteilungen mit mindestens so großer Drift wie
beobachtet, berechnet als (1 + Anzahl) / (1 + 99); der kleinste mögliche
Wert ist 0,01. p < 0,05: Die beobachtete Drift ist größer als fast alles,
was der Zufall erzeugt. Ohne jeden Unterschied wären etwa 5 % der Paare so.

**Gemeinsame Randsummen:** Damit beobachtete und permutierte Drift gleich
berechnet werden, bekommen beide Zeilen eines Paars dieselben Randsummen
(beider Zeitscheiben zusammen). Die so berechnete beobachtete Drift weicht
etwas von der Rohdrift ab (Rangkorrelation 0,994).

- **Englisch:** *permutation test*, *permutation baseline*, *null
  expectation*, *p-value*
- **Status:** etabliert
- **Belegt durch:** Good (2005), Permutationstests; Phipson & Smyth (2010),
  die Formel (1 + Anzahl) / (1 + N); Dubossarsky, Weinshall & Grossman (2017),
  Kontrollbedingung bei der Messung von Bedeutungswandel.
- **Beispiel (Tabelle 4.3):** In allen drei Terzilen haben 100 % der Paare
  p < 0,05. Die Permutationserwartung fällt mit der Textmenge: 0,235 (30–54
  Belege), 0,157 (54–138), 0,068 (138–3.690).

### Über dem Rauschen ≠ Bedeutungswandel

Der Permutationstest fragt nur, ob sich die Kontexte zweier Zeitscheiben
stärker unterscheiden als zwei zufällige Hälften desselben Topfs. Im
Bundestag ist das fast immer so: Andere Debatten, Anlässe, Redner und
Gesetzesvorhaben verändern die Kontexte jedes Worts, auch ohne
Bedeutungswandel. Halbjahre sind keine austauschbaren Stichproben.

Folgen:
- Wenn alle Paare signifikant sind, trennt der p-Wert nichts; als Filter
  taugt er in diesem Korpus nicht.
- Er bestätigt aber: Ab h\* = 30 ist die Drift keine reine Zufallsschwankung.
- „Themenwechsel oder Bedeutungswandel?“ beantwortet keiner der Tests.
  Dafür bräuchte es eine andere Vergleichsgröße, etwa Kontrollwörter mit
  vermutlich stabiler Bedeutung.

- **Englisch:** *statistical significance vs. semantic change*; Problem der
  *topical variation* bzw. des *confounding by topic*
- **Status:** Die Unterscheidung ist in der Literatur zum Bedeutungswandel
  etabliert; die deutsche Kurzformel ist eine Formulierung dieses Glossars.
- **Belegt durch:** Dubossarsky, Weinshall & Grossman (2017): Ein großer Teil
  des scheinbaren Wandels tritt auch in durchmischten Kontrollkorpora auf.
- **Beispiel:** 2.275 von 2.275 Paaren mit p < 0,05 (Tabelle 4.3).

### Varianten der Driftkorrektur

Verfahren, die die Abhängigkeit der Drift von der Belegung verringern sollen.
Alle an denselben Paaren berechnet (Tabelle 4.2):

| Variante | was passiert | Belege | r | Rang­korrelation |
|---|---|---|---|---|
| Rohdrift (Referenz) | Vektoren der vollständigen Daten | 100 % | −0,260 | 1,000 |
| **Teilstichprobe je Paar** | stärkere Zeile eines Paars auf die Belegzahl der schwächeren gezogen, 20 Ziehungen gemittelt | 73 % | **−0,094** | 0,977 |
| Teilstichprobe je Wort, Distanzen gemittelt | alle Zeilen eines Worts auf seine kleinste Belegzahl gezogen | 19 % | −0,120 | 0,900 |
| Teilstichprobe je Wort, Vektoren gemittelt | wie zuvor, aber Vektoren über die Ziehungen gemittelt | 19 % | −0,390 | 0,846 |
| Drift abzüglich Permutationserwartung | beobachtete minus Zufallsdrift | 100 % | +0,295 | 0,807 |

Beispiel „Sicherheit“, Teilstichprobe je Paar: 2015-H1 → 2015-H2 mit 40 /
200 Belegen wird als 40 gegen 40 gerechnet, 2015-H2 → 2016-H1 mit 200 / 180
als 180 gegen 180. „Je Wort“ brächte beide auf 40.

Lesart:
- **Teilstichprobe je Paar** ist die Empfehlung: belastbar geringer als die
  Rohdrift, Intervall (−0,208 bis 0,004) berührt die 0, Reihenfolge der Paare
  fast unverändert. In |r| gleichwertig ist die Teilstichprobe je Wort mit
  gemittelten Distanzen; den Ausschlag gibt der Belegverlust (73 % gegen
  19 % verwendete Belege). Erwartet war, dass die Variante je Paar wenig
  bringt; im Testlauf mit Untergrenze 10 statt 30 war das auch so.
- *Mögliche Erklärung (nicht geprüft):* Das Hauptproblem ist das
  **Ungleichgewicht** der Belegung innerhalb eines Paars. Schwach belegte
  Zeilen zeigen systematisch in eine andere Richtung (Volumeneffekt); bei
  gleicher Belegung tragen beide Zeilen dieselbe Verzerrung, und sie hebt
  sich im Kosinus weitgehend auf.
- **Vektoren gemittelt** ist schlechter als die Rohdrift: Bei gut belegten
  Zeilen glättet das Mittel über die Ziehungen das Rauschen, bei der
  schwächsten nicht.
- **Abzug der Permutationserwartung** überkorrigiert, das Vorzeichen kippt
  (s. Nichtadditivität).
- *Umsetzung:* Die Teilstichprobe je Paar liefert einen Abstand je Paar,
  keinen Vektor je Zeile. Die Drift müsste in emb je Paar berechnet und als
  eigenes Feld ins Bundle übergeben werden; Positionen und Nachbarn bleiben
  unverändert.

- **Englisch:** *subsampling*, *downsampling* (auf gleiche Belegzahl)
- **Status:** Downsampling auf gleiche Tokenmenge ist in der Literatur
  üblich; die Aufteilung in diese Varianten ist eine Konstruktion des
  Notebooks.

### Nichtadditivität von Rauschen und Unterschied

Rauschen und echter Unterschied addieren sich bei Winkeln nicht einfach.
Gibt es einen echten Unterschied, kommt nur ein Teil des Rauschens zur Drift
hinzu. Zieht man das volle Rauschen ab, zieht man bei schwach belegten Paaren
zu viel ab.

- **Beleg in den Daten (Tabelle 4.2):** Die beobachtete Drift hängt viel
  schwächer von der Belegzahl ab (r = −0,20, gemeinsame Randsummen) als das
  reine Rauschen (r = −0,82, Permutationserwartung). Die Differenz hat
  deshalb r = +0,30: Die Abhängigkeit dreht sich um.
- **Folge für die Sättigungskurve:** „66 % bei h = 30“ heißt nicht, dass 66 %
  der Drift Rauschen und 34 % Signal sind; nur, dass man ohne Unterschied
  schon auf zwei Drittel des Werts käme.
- **Englisch:** *non-additivity* (of noise and signal in cosine distance)
- **Status:** Die geometrische Tatsache ist elementar; als Begriff in dieser
  Form eine Formulierung dieses Glossars.

### Rangkorrelation nach Spearman

Wie Pearson-r, aber auf den **Rangplätzen** statt auf den Werten: Misst, ob
zwei Größen dieselbe Reihenfolge ergeben. 1 = gleiche Reihenfolge, 0 = kein
Zusammenhang. In Tabelle 4.2: Wie weit erhält eine Variante die Reihenfolge
der Paare nach ihrer Drift gegenüber der Rohdrift?

- **Englisch:** *Spearman's rank correlation*, *Spearman's ρ*
- **Status:** etabliert
- **Belegt durch:** Spearman (1904).
- **Beispiel:** Teilstichprobe je Paar 0,977 (Reihenfolge fast gleich);
  Permutationserwartung 0,033 (das Rauschniveau eines Paars sagt fast nichts
  über seinen Rang in der beobachteten Drift).

### Pilot statt endgültiger Werte

Alle Zahlen dieses Glossars gelten für das Bundestagskorpus mit punkt/halfyear,
k = 300, p = 0,5. Auf dem DNB-Korpus (andere Textsorte, andere Belegverteilung,
Ziehen nach Werken statt nach Reden) müssen h\*, k, p, der Volumeneffekt und
die beste Driftkorrektur neu gemessen werden. Übertragbar sind die
Mechanismen (Rauschen fällt mit der Belegzahl, Volumeneffekt,
Nichtadditivität, Problem gemittelter Vektoren). Die Entscheidungsregeln des
Notebooks stehen vor den Ergebnissen fest; nur die Zahlen werden neu
berechnet.

- **Englisch:** *pilot study*; Festlegen der Regeln vorab: *pre-specified
  decision rules*
- **Status:** etabliert

---

## Literatur

- Antoniak, M. & Mimno, D. (2018): Evaluating the stability of embedding-based word similarities. *Transactions of the Association for Computational Linguistics* 6, 107–119.
- Berry, M. W., Dumais, S. T. & O'Brien, G. W. (1995): Using linear algebra for intelligent information retrieval. *SIAM Review* 37(4), 573–595.
- Bland, J. M. & Altman, D. G. (1995): Calculating correlation coefficients with repeated observations: Part 1 – correlation within subjects. *BMJ* 310(6977), 446.
- Brown, W. (1910): Some experimental results in the correlation of mental abilities. *British Journal of Psychology* 3(3), 296–322.
- Bullinaria, J. A. & Levy, J. P. (2007): Extracting semantic representations from word co-occurrence statistics: a computational study. *Behavior Research Methods* 39(3), 510–526.
- Bullinaria, J. A. & Levy, J. P. (2012): Extracting semantic representations from word co-occurrence statistics: stop-lists, stemming, and SVD. *Behavior Research Methods* 44(3), 890–907.
- Caron, J. (2001): Experiments with LSA scoring: optimal rank and basis. In: Berry, M. W. (Hg.): *Computational Information Retrieval*. SIAM, 157–169.
- Church, K. W. & Hanks, P. (1990): Word association norms, mutual information, and lexicography. *Computational Linguistics* 16(1), 22–29.
- Davison, A. C. & Hinkley, D. V. (1997): *Bootstrap Methods and their Application*. Cambridge University Press.
- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. & Harshman, R. (1990): Indexing by latent semantic analysis. *Journal of the American Society for Information Science* 41(6), 391–407.
- Dubossarsky, H., Weinshall, D. & Grossman, E. (2017): Outta control: Laws of semantic change and inherent biases in word representation models. In: *Proceedings of EMNLP 2017*, 1136–1145.
- Eckart, C. & Young, G. (1936): The approximation of one matrix by another of lower rank. *Psychometrika* 1(3), 211–218.
- Efron, B. & Tibshirani, R. J. (1993): *An Introduction to the Bootstrap*. Chapman & Hall.
- Good, P. (2005): *Permutation, Parametric and Bootstrap Tests of Hypotheses*. 3. Aufl. Springer.
- Golub, G. H. & Van Loan, C. F. (2013): *Matrix Computations*. 4. Aufl. Johns Hopkins University Press.
- Hanley, J. A. & McNeil, B. J. (1982): The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 143(1), 29–36.
- Hamilton, W. L., Leskovec, J. & Jurafsky, D. (2016b): Diachronic word embeddings reveal statistical laws of semantic change. *Proceedings of ACL 2016*, 1489–1501.
- Haxby, J. V., Gobbini, M. I., Furey, M. L., Ishai, A., Schouten, J. L. & Pietrini, P. (2001): Distributed and overlapping representations of faces and objects in ventral temporal cortex. *Science* 293(5539), 2425–2430.
- Hosmer, D. W. & Lemeshow, S. (2000): *Applied Logistic Regression*. 2. Aufl. Wiley.
- Hurlbert, S. H. (1971): The nonconcept of species diversity: a critique and alternative parameters. *Ecology* 52(4), 577–586.
- Jolliffe, I. T. & Cadima, J. (2016): Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A* 374(2065), 20150202.
- Kutuzov, A., Øvrelid, L., Szymanski, T. & Velldal, E. (2018): Diachronic word embeddings and semantic shifts: A survey. *Proceedings of COLING 2018*, 1384–1397.
- Levy, O., Goldberg, Y. & Dagan, I. (2015): Improving distributional similarity with lessons learned from word embeddings. *Transactions of the Association for Computational Linguistics* 3, 211–225.
- Lord, F. M. & Novick, M. R. (1968): *Statistical Theories of Mental Test Scores*. Addison-Wesley.
- Luhn, H. P. (1960): Keyword-in-context index for technical literature (KWIC index). *American Documentation* 11(4), 288–295.
- Mahoney, M. W. & Drineas, P. (2009): CUR matrix decompositions for improved data analysis. *Proceedings of the National Academy of Sciences* 106(3), 697–702.
- Mann, H. B. & Whitney, D. R. (1947): On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics* 18(1), 50–60.
- Mu, J. & Viswanath, P. (2018): All-but-the-top: Simple and effective postprocessing for word representations. *ICLR 2018*. arXiv:1702.01417.
- Phipson, B. & Smyth, G. K. (2010): Permutation P-values should never be zero: Calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology* 9(1), Article 39.
- Radovanović, M., Nanopoulos, A. & Ivanović, M. (2010): Hubs in space: Popular nearest neighbors in high-dimensional data. *Journal of Machine Learning Research* 11, 2487–2531.
- Sanders, H. L. (1968): Marine benthic diversity: a comparative study. *The American Naturalist* 102(925), 243–282.
- Schlechtweg, D., Hätty, A., del Tredici, M. & Schulte im Walde, S. (2019): A wind of change: Detecting and evaluating lexical semantic change across times and domains. *Proceedings of ACL 2019*, 732–746.
- Spearman, C. (1904): The proof and measurement of association between two things. *The American Journal of Psychology* 15(1), 72–101.
- Spearman, C. (1910): Correlation calculated from faulty data. *British Journal of Psychology* 3(3), 271–295.
- Turney, P. D. & Pantel, P. (2010): From frequency to meaning: Vector space models of semantics. *Journal of Artificial Intelligence Research* 37, 141–188.

Die bibliografischen Angaben sind vor einer Veröffentlichung an den Originalen zu prüfen.
