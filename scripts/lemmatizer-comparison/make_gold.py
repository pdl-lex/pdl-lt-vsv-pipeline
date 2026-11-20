"""
Hand-made gold lemmas for the 200 most frequent content-word types in results/diff_types.csv
(types where Stanza, spaCy and simplemma do not all agree). Writes results/gold_corpus_types.csv.

Annotation (2026-11-20), judged by type (form + Stanza POS), without looking at every context:
- several acceptable lemmas separated by "|"; "?" = not decidable by type, left out of the scoring
- equivalent conventions are all accepted: ordinals/superlatives "letzt|letzter", nominalized
  adjectives "Beschäftigte|Beschäftigter", participle adjectives "organisiert|organisieren"
  (but lexicalized adjectives only as adjective: "gezielt", not "zielen")
- nouns need a capitalized lemma ("Vertrauen", not "vertrauen"; "Grüne", not "grüne")
- modern orthography ("Russland", "Ausschuss"); score.py also reports an ß-tolerant score
"""
from pathlib import Path

import polars as pl

GOLD = {
    0: "Russland", 1: "Soldatin", 2: "letzt|letzter", 3: "?", 4: "erneuerbar",
    5: "nah|nächst|nächster", 6: "vereint|Vereinte", 7: "deutsch|Deutsche", 8: "Grüne", 9: "Russland",
    10: "lieb", 11: "erst|erster", 12: "Linke", 13: "lieb", 14: "Vertrauen",
    15: "Putin", 16: "Vorgehen", 17: "erst|erster", 18: "Fakt|Faktum", 19: "deutsch|Deutsche",
    20: "völkerrechtswidrig", 21: "bewusst", 22: "gezielt", 23: "Beschäftigte|Beschäftigter", 24: "IS",
    25: "Verbündete|Verbündeter", 26: "Hass", 27: "?", 28: "einig", 29: "weit|weiter",
    30: "Linke", 31: "lassen", 32: "denken", 33: "dringend", 34: "gerecht",
    35: "Wehrbeauftragte|Wehrbeauftragter", 36: "Auslandseinsatz", 37: "Ausschuss", 38: "zweit|zweiter", 39: "Volumen",
    40: "besonder|besonderer", 41: "verlieren", 42: "Geflüchtete|Geflüchteter", 43: "inner|innerer", 44: "nah|Nahe|nahe",
    45: "entscheidend", 46: "zweit|zweiter", 47: "sogenannt|sogenannter", 48: "Geflüchtete|Geflüchteter", 49: "Fall",
    50: "gezielt", 51: "Grüne", 52: "vereinigt|Vereinigte", 53: "fallen", 54: "Polizistin",
    55: "Anlass", 56: "sogenannt|sogenannter", 57: "Schuld|Schulden", 58: "Missbrauch", 59: "vernetzt|vernetzen",
    60: "organisiert|organisieren", 61: "Linke", 62: "menschengemacht", 63: "Verhalten", 64: "völkerrechtswidrig",
    65: "Ausschuss", 66: "Angehörige|Angehöriger", 67: "recht|rechter", 68: "wiederholen", 69: "hybrid",
    70: "Jugendliche|Jugendlicher", 71: "Beschluss", 72: "verehrt|verehren", 73: "früh|früher", 74: "Dank",
    75: "hybrid", 76: "vorliegend", 77: "sexualisiert|sexualisieren", 78: "geplant|planen", 79: "entscheiden",
    80: "Beschäftigte|Beschäftigter", 81: "Deutsche|Deutscher", 82: "verabschieden", 83: "Studierende|Studierender",
    84: "Abgeordnete|Abgeordneter", 85: "überzeugt|überzeugen", 86: "angemessen", 87: "reichen", 88: "fürchten",
    89: "eskalieren", 90: "verüben", 91: "jährig", 92: "einplanen", 93: "Windschatten",
    94: "Wehrbeauftragte|Wehrbeauftragter", 95: "malisch", 96: "Einzelne|Einzelner", 97: "Ukrainerin", 98: "Weißrussland",
    99: "angespannt|anspannen", 100: "lieben", 101: "Klimafonds", 102: "?", 103: "robust",
    104: "Drohne", 105: "klar", 106: "Handeln", 107: "CO2", 108: "besonder|besonderer",
    109: "Denken", 110: "recht|rechter", 111: "Roma", 112: "Licht", 113: "nah|nächst|nächster",
    114: "bekannt", 115: "umgekehrt", 116: "ein", 117: "Erste|Erster", 118: "bewaffnet|bewaffnen",
    119: "Beamte|Beamter", 120: "Wissenschaftlerin", 121: "Bewusstsein", 122: "Eingreifen", 123: "erst|erster",
    124: "Kompromiss", 125: "Wohnen", 126: "zweit|zweiter", 127: "Moskau", 128: "NGO",
    129: "Gute|Gutes", 130: "Schluss", 131: "Erachten", 132: "Betriebskosten", 133: "jung|jüngst",
    134: "besonder|besonderer", 135: "Zivilistin", 136: "demografisch", 137: "Studierende|Studierender", 138: "gewiss",
    139: "Den", 140: "abwägen", 141: "umfassend", 142: "aufarbeiten", 143: "motiviert|motivieren",
    144: "angekündigt|ankündigen", 145: "Ganze|Ganzes", 146: "kurz", 147: "Patrouille", 148: "Helferin",
    149: "letzt|letzter", 150: "ausgebildet|ausbilden", 151: "Migrant", 152: "hinbekommen", 153: "mittler|mittlerer",
    154: "Rüstungsvorhaben", 155: "inner|innerer", 156: "Betroffene|Betroffener", 157: "lieb", 158: "Niederlande",
    159: "Grundakte", 160: "Rentnerin", 161: "instrumentalisieren", 162: "?", 163: "gewiss",
    164: "?", 165: "zurückdrängen", 166: "Dividende", 167: "interessiert|interessieren", 168: "?",
    169: "wahr", 170: "Jugendliche|Jugendlicher", 171: "Kita", 172: "verleugnen", 173: "bestimmt",
    174: "verehrt|verehren", 175: "mittler|mittlerer", 176: "herbeiführen", 177: "stehen", 178: "Lieferkette",
    179: "schicken", 180: "Überschuss", 181: "wissen", 182: "recht", 183: "sexualisiert|sexualisieren",
    184: "?", 185: "Rechtsausschuss", 186: "?", 187: "ausgerechnet", 188: "militärisch|Militärische",
    189: "landgestützt", 190: "Mittelstreckenrakete", 191: "groß", 192: "zweit|zweiter", 193: "bündeln",
    194: "nah|Nahe|nahe", 195: "mittler|Mittlere", 196: "fünft|fünfter", 197: "gestiegen|steigen", 198: "al",
    199: "zweitgrößt|zweitgrößter",
}

if __name__ == "__main__":
    res = Path(__file__).resolve().parent / "results"
    types = pl.read_csv(res / "diff_types.csv").head(len(GOLD))
    assert sorted(GOLD) == list(range(types.height))
    types.with_columns(pl.Series("gold", [GOLD[i] for i in range(types.height)])).write_csv(
        res / "gold_corpus_types.csv"
    )
    print("written:", res / "gold_corpus_types.csv")
