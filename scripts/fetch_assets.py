"""Download the language models into assets/ and check that they load offline.

    uv run python scripts/fetch_assets.py           # download missing models
    uv run python scripts/fetch_assets.py --check   # only load them, no network

Same logic and versions as the "Laden von Sprachmodellen" cell in
stages/01_preprocessing/pipeline_pre_ppmi.py. If you change a model or version
there, change it here too.
"""

import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SPACY_VERSION = "3.8.0"

PUNKT = ASSETS / "nltk_data" / "tokenizers" / "punkt_tab"
STANZA = ASSETS / "stanza_resources"
SPACY = ASSETS / "spacy" / "de_core_news_md"


def fetch() -> None:
    # Each download goes to a temp folder first, so an aborted download does
    # not leave a half-filled model folder behind.
    if not (PUNKT / "german").is_dir():
        import nltk

        tmp = ASSETS / "_tmp_nltk"
        shutil.rmtree(tmp, ignore_errors=True)
        nltk.download("punkt_tab", download_dir=str(tmp), quiet=True, raise_on_error=True)
        PUNKT.mkdir(parents=True, exist_ok=True)
        for name in ("german", "english", "README"):
            shutil.rmtree(PUNKT / name, ignore_errors=True)
            shutil.move(tmp / "tokenizers" / "punkt_tab" / name, PUNKT / name)
        shutil.rmtree(tmp)

    if not (STANZA / "resources.json").is_file():
        import stanza

        tmp = ASSETS / "_tmp_stanza"
        shutil.rmtree(tmp, ignore_errors=True)
        stanza.download(
            "de",
            model_dir=str(tmp),
            processors="tokenize,mwt,pos,lemma",
            logging_level="WARN",
        )
        shutil.rmtree(STANZA, ignore_errors=True)
        tmp.rename(STANZA)

    if not (SPACY / "config.cfg").is_file():
        url = (
            "https://github.com/explosion/spacy-models/releases/download/"
            f"de_core_news_md-{SPACY_VERSION}/de_core_news_md-{SPACY_VERSION}-py3-none-any.whl"
        )
        with urllib.request.urlopen(url) as resp:
            whl = zipfile.ZipFile(io.BytesIO(resp.read()))
        prefix = f"de_core_news_md/de_core_news_md-{SPACY_VERSION}/"
        tmp = ASSETS / "_tmp_spacy"
        shutil.rmtree(tmp, ignore_errors=True)
        for m in whl.infolist():
            if m.filename.startswith(prefix) and not m.is_dir():
                target = tmp / m.filename[len(prefix):]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(whl.read(m))
        SPACY.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(SPACY, ignore_errors=True)
        tmp.rename(SPACY)


def check() -> None:
    # Load every model the way the notebooks do, without any download.
    import nltk
    from nltk.tokenize import PunktTokenizer

    nltk.data.path.insert(0, str(ASSETS / "nltk_data"))
    PunktTokenizer("german")
    print("ok  nltk punkt_tab (german)")

    from somajo import SoMaJo

    SoMaJo("de_CMC", split_camel_case=False, character_offsets=True)
    print("ok  somajo de_CMC")

    import spacy

    spacy.load(SPACY)
    print("ok  spacy de_core_news_md")

    import stanza

    stanza.Pipeline(
        "de",
        processors="tokenize,mwt,pos,lemma",
        dir=str(STANZA),
        download_method=None,
        logging_level="WARN",
    )
    print("ok  stanza de (tokenize,mwt,pos,lemma)")


if __name__ == "__main__":
    if "--check" not in sys.argv[1:]:
        fetch()
    check()
