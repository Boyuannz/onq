"""Check IFEval's English tokenizer data before sending generation requests."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def check_ifeval_resources():
    import nltk

    try:
        # Exercise both APIs used by the pinned IFEval scorer, not just imports.
        if len(nltk.sent_tokenize("First sentence. Second sentence.")) != 2:
            raise RuntimeError("Unexpected English sentence tokenizer behavior")
        if not nltk.word_tokenize("A short sentence."):
            raise RuntimeError("English word tokenizer returned no tokens")
        root = nltk.data.find("tokenizers/punkt_tab/english/")
        hashes = {}
        for name in ("collocations.tab", "sent_starters.txt",
                     "abbrev_types.txt", "ortho_context.tab"):
            with root.join(name).open() as stream:
                hashes[name] = hashlib.sha256(stream.read()).hexdigest()
    except (LookupError, OSError) as exc:
        raise RuntimeError(
            "IFEval requires the English punkt_tab resource. Run this environment's "
            "python -m OnQ.evaluation.evalscope.resources --download before evaluation. "
            "Offline installations can provision tokenizers/punkt_tab/english/ under "
            "NLTK_DATA, then run the command without --download."
        ) from exc
    return {"status": "PASS", "nltk": nltk.__version__,
            "resource": "punkt_tab/english", "sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="Explicitly provision punkt_tab in this Python environment")
    args = parser.parse_args()
    if args.download:
        import nltk
        destination = Path(sys.prefix) / "nltk_data"
        destination.mkdir(parents=True, exist_ok=True)
        if not nltk.download("punkt_tab", download_dir=str(destination),
                             quiet=True, raise_on_error=True):
            raise RuntimeError("punkt_tab download failed")
        nltk.data.path.insert(0, str(destination))
    print(json.dumps(check_ifeval_resources(), indent=2))


if __name__ == "__main__":
    main()
