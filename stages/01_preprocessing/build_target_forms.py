"""
Build the target word mapping for pipeline_pre_ppmi.py from the hand-made source list.

Input:  input/target_list_nonfinal.csv  (one row per target word: head;form 1;form 2;...)
Output: input/target_forms.csv   form;lemma;source  - one row per form, read by the pipeline
        input/target_lemmas.csv  lemma;original_heads;n_forms - the target words without
                                 inflected forms, with the source heads merged into each lemma

The mapping is broad: spelling variants, transliterations and gendered forms all map to one
lemma (Belarus -> Weißrussland, Soldat*innen -> Soldat). `source` says where a form comes from:
"liste" (source list), "ergänzt" (added by hand below), "gender" (generated gendered form),
"variante" (generated spelling variant). Multiword forms are dropped: the KWIC search works on
whitespace tokens, they can never match.

Run from stages/01_preprocessing:  python build_target_forms.py
Once the output is maintained by hand, this script is no longer needed.
"""

import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline_pre_dataimport import normalize_chars  # noqa: E402

SOURCE = HERE / "input" / "target_list_nonfinal.csv"
OUT_FORMS = HERE / "input" / "target_forms.csv"
OUT_LEMMAS = HERE / "input" / "target_lemmas.csv"

# --- 1. hand-made corrections ------------------------------------------------------------------
# canonical lemma per source head (merges and renames)
MERGE = {
    # variants / duplicates
    "2‑Prozent-Ziel": "2-Prozent-Ziel",  # U+2011 non-breaking hyphen
    "Belarus": "Weißrussland",
    "Ukrainekrieg": "Ukraine-Krieg",
    "Ukrainekonflikt": "Ukraine-Konflikt",
    "Wehrbeauftragten": "Wehrbeauftragte",
    "Wehrbeauftragter": "Wehrbeauftragte",
    "Russisch": "russisch",  # language noun, not separable from the adjective at sentence start
    "Kalter Krieg": "Kalter-Krieg",  # lemma must be one token as well; only hyphenated forms match
    # gender: feminine head -> common head
    "Pazifistin": "Pazifist",
    "Verteidigungsministerin": "Verteidigungsminister",
    # head in the plural / odd form -> singular lemma (nominalised adjectives: -e, as in the Duden)
    "Fakten": "Fakt",
    "Einsatzkräfte": "Einsatzkraft",
    "Militärausgaben": "Militärausgabe",
    "Migrationsströme": "Migrationsstrom",
    "NATO-Kriterien": "NATO-Kriterium",
    "Rüstungsexporte": "Rüstungsexport",
    "Rüstungsgüter": "Rüstungsgut",
    "Wirtschaftssanktionen": "Wirtschaftssanktion",
    "Verbündeter": "Verbündete",
    "NATO-Verbündet": "NATO-Verbündete",
}
# forms removed: generation artefacts, and forms ending in an apostrophe (strip_punctuation in
# the KWIC search cuts the "'", they never match). Ambiguities (Patriot, Leopard) are accepted.
DROP = {"Ukraine-R", "Ukraine-Rs", "Ukraine-Rn", "Russischs",
        "Belarus'", "Donbas'", "Donbass'", "Ukraine'"}
# forms missing in the source
ADD = {
    "Verteidigungsminister": ["Verteidigungsminister", "Verteidigungsministers", "Verteidigungsministern"],
    "NATO": ["Nato", "NATOs", "Natos"],
}

# --- 2. gendered forms -------------------------------------------------------------------------
# lemma -> masculine stems (all spelling variants of the nominative singular)
GENDER_STEMS = {
    "Aggressor": ["Aggressor"], "Angreifer": ["Angreifer"], "Attentäter": ["Attentäter"],
    "Berufssoldat": ["Berufssoldat"], "Blauhelmsoldat": ["Blauhelmsoldat", "Blauhelm-Soldat"],
    "Bundeswehrsoldat": ["Bundeswehrsoldat", "Bundeswehr-Soldat"],
    "Bündnispartner": ["Bündnispartner"], "Despot": ["Despot"], "Diktator": ["Diktator"],
    "Faschist": ["Faschist"], "Feind": ["Feind"], "Generalinspekteur": ["Generalinspekteur"],
    "Held": ["Held"], "Kamerad": ["Kamerad"], "Kämpfer": ["Kämpfer"], "Kommandeur": ["Kommandeur"],
    "Kriegsgegner": ["Kriegsgegner"], "Kriegspräsident": ["Kriegspräsident"],
    "Kriegstreiber": ["Kriegstreiber"], "Kriegsverbrecher": ["Kriegsverbrecher"],
    "Machthaber": ["Machthaber"], "NATO-Partner": ["NATO-Partner", "Natopartner"],
    "Patriot": ["Patriot"], "Pazifist": ["Pazifist"],
    "Putin-Versteher": ["Putin-Versteher", "Putinversteher"],
    "Reservist": ["Reservist"], "Russe": ["Russ"], "Soldat": ["Soldat"], "Söldner": ["Söldner"],
    "Terrorist": ["Terrorist"], "Ukrainer": ["Ukrainer"], "Verbrecher": ["Verbrecher"],
    "Verteidiger": ["Verteidiger"], "Verteidigungsminister": ["Verteidigungsminister"],
    "Zeitsoldat": ["Zeitsoldat"], "Zivilist": ["Zivilist"],
}
# nominalised adjectives: lemma -> stems (Verbündete*r, Wehrbeauftragte:r, ...)
ADJ_NOUNS = {
    "Alliierte": ["Alliierte"],
    "Bundeswehrangehörige": ["Bundeswehrangehörige", "Bundeswehr-Angehörige"],
    "Geflüchtete": ["Geflüchtete"], "NATO-Verbündete": ["NATO-Verbündete", "Natoverbündete"],
    "Verbündete": ["Verbündete"], "Wehrbeauftragte": ["Wehrbeauftragte"],
    "Wehrdienstleistende": ["Wehrdienstleistende"], "Wehrpflichtige": ["Wehrpflichtige"],
}
# gender signs: star, colon, underscore, slash, slash-hyphen, middle dot (U+00B7)
SIGNS = ["*", ":", "_", "/", "/-", "·"]


def gender_forms(stem):
    yield from (stem + "in", stem + "innen")  # Soldatin, Soldatinnen
    yield from (stem + "In", stem + "Innen")  # Binnen-I: SoldatIn, SoldatInnen
    for s in SIGNS:
        yield from (stem + s + "in", stem + s + "innen")  # Soldat*in, Soldat*innen, ...


def adj_gender_forms(stem):
    for s in SIGNS:
        if s != "/-":
            yield stem + s + "r"  # Verbündete*r


# --- 3. spelling variants ------------------------------------------------------------------------
GENDERED = re.compile(r"[*:_/·]|[a-zäöüß]Innen$|[a-zäöüß]In$")


def spelling_variants(f):
    # "Nato-" for "NATO-" (Nato-Bündnis)
    if f.startswith("NATO-"):
        yield "Nato-" + f[5:]
    # hyphenated compound written closed: "Drohnen-Krieg" -> "Drohnenkrieg"
    # (exactly one hyphen, first part a normal word; "NATO-", "Kalter-", "Russland-Ukraine-" stay)
    m = re.fullmatch(r"([A-ZÄÖÜ][a-zäöüß]+)-([A-ZÄÖÜ])([a-zäöüß][^-]*)", f)
    if m and not m[1].startswith("Kalt"):
        yield m[1] + m[2].lower() + m[3]
    # Swiss spelling / capitals: ß -> ss ("Großmacht" -> "Grossmacht")
    if "ß" in f:
        yield f.replace("ß", "ss")
    # spelling before 1996: ss -> ß after a vowel at the word end or before a consonant
    # ("Russland" -> "Rußland", "Ausschuss" -> "Ausschuß"); not before "ch" (Aus|schuss), and not
    # in gendered forms (they do not occur in the old spelling; "Ruß*innen" would be wrong)
    if "ss" in f and not GENDERED.search(f):
        yield re.sub(r"(?<=[aeiouäöü])ss(?=$|[^aeiouäöüyc])", "ß", f)


# --- 4. build ----------------------------------------------------------------------------------
def build():
    with SOURCE.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=";"))[1:]

    forms = defaultdict(dict)  # lemma -> {form: source}, in insertion order
    merged_from = defaultdict(list)  # lemma -> source heads
    n_multiword = 0

    def add(lemma, form, source):
        forms[lemma].setdefault(form, source)

    for r in rows:
        head = r[0].strip()
        lemma = MERGE.get(head, head)
        merged_from[lemma].append(head)
        for f in [head] + [x for x in r[1:] if x.strip()]:
            f = normalize_chars(unicodedata.normalize("NFC", f.strip()))
            if f in DROP:
                continue
            if re.search(r"\s", f):
                n_multiword += 1
                continue
            add(lemma, f, "liste")
    for lemma, fs in ADD.items():
        for f in fs:
            add(lemma, f, "ergänzt")
    for lemma, stems in GENDER_STEMS.items():
        assert lemma in forms, f"GENDER_STEMS: unknown lemma {lemma}"
        for f in (g for st in stems for g in gender_forms(st)):
            add(lemma, f, "gender")
    for lemma, stems in ADJ_NOUNS.items():
        assert lemma in forms, f"ADJ_NOUNS: unknown lemma {lemma}"
        for f in (g for st in stems for g in adj_gender_forms(st)):
            add(lemma, f, "gender")
    # repeated until nothing new comes up, so variants combine ("NATO-Russland-Akte" ->
    # "Nato-Russland-Akte" -> "Nato-Rußland-Akte")
    for lemma, fs in forms.items():
        todo = list(fs)
        while todo:
            new = [v for f in todo for v in spelling_variants(f) if v not in fs]
            for v in new:
                fs.setdefault(v, "variante")
            todo = new
    # the lemma itself comes first among its forms
    for lemma in forms:
        forms[lemma] = {lemma: forms[lemma].pop(lemma, "liste"), **forms[lemma]}

    # checks: one lemma per form, single tokens, nothing the KWIC search would change
    owner = defaultdict(set)
    for lemma, fs in forms.items():
        for f in fs:
            owner[f].add(lemma)
    clash = {f: sorted(ls) for f, ls in owner.items() if len(ls) > 1}
    assert not clash, f"forms with more than one lemma: {clash}"
    assert not [f for f in owner if re.search(r"\s", f)], "multiword forms left"
    dead = [f for f in owner if re.sub(r"^\W+|\W+$", "", f) != f or normalize_chars(f) != f]
    assert not dead, f"forms changed by strip_punctuation / normalize_chars: {dead}"

    lemmas = sorted(forms, key=lambda s: (s.casefold(), s))
    with OUT_FORMS.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";", lineterminator="\r\n")
        w.writerow(["form", "lemma", "source"])
        for lemma in lemmas:
            w.writerows([f, lemma, src] for f, src in forms[lemma].items())
    with OUT_LEMMAS.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";", lineterminator="\r\n")
        w.writerow(["lemma", "original_heads", "n_forms"])
        w.writerows([lemma, " | ".join(merged_from[lemma]), len(forms[lemma])] for lemma in lemmas)

    by_source = defaultdict(int)
    for fs in forms.values():
        for src in fs.values():
            by_source[src] += 1
    print(f"{len(rows)} source heads -> {len(forms)} lemmas, {len(owner)} forms {dict(by_source)}")
    print(f"{n_multiword} multiword forms dropped")
    print(f"written: {OUT_FORMS.name}, {OUT_LEMMAS.name}")


if __name__ == "__main__":
    build()
