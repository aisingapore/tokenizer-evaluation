"""
Fertility and token-count ratio vs. English on FLORES+ dev (997 parallel sentences).

For each tokenizer and each language in `parallel_data` (config/config.yaml):
  - fertility              = total tokens / total words
  - token_ratio_vs_english = total tokens / the same tokenizer's English total tokens

All text is NFC-normalized before counting words and tokens. Special tokens are
excluded: no BOS/EOS/template tokens are added, and literal special-token strings
in the text (e.g. "<bos>") are tokenized as plain text rather than as special IDs.

Words are counted with --word-segmenter:
  - whitespace (default): str.split(), same as evaluate.py. Not meaningful across
    languages for Thai, Lao, Khmer, Burmese and Chinese, which don't space words.
  - icu: ICU word boundaries (UAX #29, dictionary-based for Thai/Lao/Khmer/Burmese/
    Chinese), applied to all languages; punctuation and spaces are not counted.
    Requires PyICU:  PKG_CONFIG_PATH="$(brew --prefix icu4c)/lib/pkgconfig" pip install pyicu

Usage:
    python evaluate_flores.py
    python evaluate_flores.py --word-segmenter icu
    python evaluate_flores.py --tokenizers google/gemma-4-31B-it openai/gpt-oss-120b
"""
import argparse
import csv
import json
import os
import unicodedata

import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

DEFAULT_TOKENIZERS = [
    "google/gemma-4-31B-it",                        # Gemma 4
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",   # Nemotron 3
    "openai/gpt-oss-120b",                          # GPT-OSS
]

FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_SPLIT = "dev"
FLORES_DEV_SIZE = 997
BASELINE_LANG = "English"


# --- Data ---

def flores_code(path):
    """'openlanguagedata/flores_plus/ban_Latn.dev' -> 'ban_Latn'"""
    return os.path.splitext(os.path.basename(path))[0]


def load_flores_dev(languages, cache_dir, hf_token):
    """
    Downloads FLORES+ dev for each language and returns {lang: [NFC text, ...]},
    with every language aligned to the same sentence IDs, plus per-language NFC stats.
    """
    by_lang = {}
    for lang, code in languages.items():
        path = hf_hub_download(FLORES_REPO, f"{FLORES_SPLIT}/{code}.jsonl",
                               repo_type="dataset", cache_dir=cache_dir, token=hf_token)
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        id_to_text = {r["id"]: r["text"] for r in rows}
        if len(rows) != FLORES_DEV_SIZE or len(id_to_text) != FLORES_DEV_SIZE:
            raise ValueError(f"{lang} ({code}): expected {FLORES_DEV_SIZE} unique sentences, "
                             f"got {len(rows)} rows / {len(id_to_text)} unique IDs")
        by_lang[lang] = id_to_text

    ids = sorted(by_lang[BASELINE_LANG])
    for lang, id_to_text in by_lang.items():
        if sorted(id_to_text) != ids:
            raise ValueError(f"{lang}: sentence IDs do not match {BASELINE_LANG}")

    texts, nfc_changed = {}, {}
    for lang, id_to_text in by_lang.items():
        raw = [id_to_text[i] for i in ids]
        texts[lang] = [unicodedata.normalize("NFC", t) for t in raw]
        nfc_changed[lang] = sum(a != b for a, b in zip(raw, texts[lang]))
    return texts, nfc_changed


# --- Word counting ---

def make_word_counter(segmenter):
    """Returns (count_words(text) -> int, description)."""
    if segmenter == "whitespace":
        return (lambda text: len(text.split())), "whitespace split (str.split)"

    import icu  # PyICU; imported here so the whitespace path doesn't need it

    # Root locale: ICU selects its dictionary break engines by script, so per-language
    # locales give identical counts.
    bi = icu.BreakIterator.createWordInstance(icu.Locale.getRoot())

    def count_words(text):
        bi.setText(text)
        # Rule status below NONE_LIMIT marks spaces/punctuation/symbols, not words.
        return sum(1 for _ in bi if bi.getRuleStatus() >= icu.UWordBreak.NONE_LIMIT)

    return count_words, f"ICU {icu.ICU_VERSION} word break (root locale), punctuation/spaces excluded"


# --- Tokenization ---

def special_token_ids(tok):
    ids = set(tok.all_special_ids)
    ids.update(i for i, added in tok.added_tokens_decoder.items() if added.special)
    return ids


def count_tokens(tok, texts, special_ids):
    encoded = tok(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]
    total = sum(len(ids) for ids in encoded)
    leaked = sum(1 for ids in encoded for i in ids if i in special_ids)
    if leaked:
        raise RuntimeError(f"{leaked} special token(s) found in encoded output")
    return total


# --- Main ---

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_hf_token():
    token_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_token.txt")
    if os.path.exists(token_path):
        with open(token_path) as f:
            return f.read().strip()
    return os.environ.get("HF_TOKEN")


def print_table(results, languages):
    models = list(results)
    short = [m.split("/")[-1][:22] for m in models]
    for metric, fmt in (("total_tokens", "{:>24,d}"), ("fertility", "{:>24.3f}"),
                        ("token_ratio_vs_english", "{:>24.3f}")):
        print(f"\n{metric}")
        print(f"{'Language':<12}" + "".join(f"{s:>24}" for s in short))
        for lang in languages:
            print(f"{lang:<12}" + "".join(fmt.format(results[m][metric][lang]) for m in models))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizers", nargs="+", default=DEFAULT_TOKENIZERS,
                        help="HF model IDs or local paths. Default: Gemma 4, Nemotron 3, GPT-OSS.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml"))
    parser.add_argument("--word-segmenter", choices=["whitespace", "icu"], default="whitespace",
                        help="How words are counted for fertility. Default: whitespace.")
    parser.add_argument("--output-dir", default=None,
                        help="Default: <config output_dir>/flores_dev/<word-segmenter>")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code=True when loading tokenizers.")
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir = config.get("cache_dir")
    output_dir = args.output_dir or os.path.join(config["output_dir"], "flores_dev", args.word_segmenter)
    os.makedirs(output_dir, exist_ok=True)
    hf_token = load_hf_token()
    count_words, word_definition = make_word_counter(args.word_segmenter)

    languages = {lang: flores_code(p) for lang, p in config["parallel_data"].items()}
    if BASELINE_LANG not in languages:
        raise ValueError(f"'{BASELINE_LANG}' must be in parallel_data to compute ratios")

    print(f"Loading FLORES+ {FLORES_SPLIT} for {len(languages)} languages...")
    texts, nfc_changed = load_flores_dev(languages, cache_dir, hf_token)
    words = {lang: sum(count_words(t) for t in ts) for lang, ts in texts.items()}

    results = {}
    for model in args.tokenizers:
        print(f"Evaluating {model}...")
        tok = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, token=hf_token,
                                            trust_remote_code=args.trust_remote_code)
        special_ids = special_token_ids(tok)
        tokens = {lang: count_tokens(tok, ts, special_ids) for lang, ts in texts.items()}
        results[model] = {
            "fertility": {lang: tokens[lang] / words[lang] for lang in languages},
            "token_ratio_vs_english": {lang: tokens[lang] / tokens[BASELINE_LANG] for lang in languages},
            "total_tokens": tokens,
        }

    metadata = {
        "dataset": f"{FLORES_REPO} ({FLORES_SPLIT})",
        "sentences_per_language": FLORES_DEV_SIZE,
        "normalization": "NFC",
        "special_tokens": "excluded (add_special_tokens=False, split_special_tokens=True)",
        "word_definition": word_definition,
        "flores_codes": languages,
        "total_words": words,
        "sentences_changed_by_nfc": nfc_changed,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tokenizer", "language", "flores_code", "sentences", "words",
                    "tokens", "fertility", "token_ratio_vs_english"])
        for model, r in results.items():
            for lang, code in languages.items():
                w.writerow([model, lang, code, FLORES_DEV_SIZE, words[lang], r["total_tokens"][lang],
                            f"{r['fertility'][lang]:.4f}", f"{r['token_ratio_vs_english'][lang]:.4f}"])

    print_table(results, languages)
    print(f"\nSaved results.json, results.csv, metadata.json to {output_dir}")


if __name__ == "__main__":
    main()
