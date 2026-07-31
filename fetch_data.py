import os
import sys
import random
import argparse
import yaml
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Local directories
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MONO_DIR = os.path.join(DATA_DIR, "monolingual")

# Remote Files (Hugging Face Hub)
# Fill this in with the datasets to download, e.g.:
#
# REMOTE_FILES = {
#     "English": {
#         "repo_id": "HuggingFaceFW/fineweb-edu",
#         "repo_type": "dataset",
#         "files": ["sample/10BT/000_00000.parquet"],
#     },
#     "German": {
#         "repo_id": "HuggingFaceFW/fineweb-2",
#         "repo_type": "dataset",
#         "files": ["data/deu_Latn/train/001_00015.parquet"],
#     },
# }
REMOTE_FILES = {
    "Balinese": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/ban_Latn/train/000_00000.parquet"],
    },
    "Chinese": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/cmn_Hani/train/000_00050.parquet"],
    },
    "English": {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "repo_type": "dataset",
        "files": ["sample/10BT/003_00000.parquet"],
    },
    "Filipino": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/fil_Latn/train/000_00000.parquet"],
    },
    "Indonesian": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/ind_Latn/train/001_00000.parquet"],
    },
    "Javanese": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/jav_Latn/train/000_00000.parquet"],
    },
    "Khmer": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/khm_Khmr/train/000_00000.parquet"],
    },
    "Laos": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/lao_Laoo/train/000_00000.parquet"],
    },
    "Burmese": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/mya_Mymr/train/000_00000.parquet"],
    },
    "Sudanese": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/sun_Latn/train/000_00000.parquet"],
    },
    "Tamil": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/tam_Taml/train/001_00000.parquet"],
    },
    "Thai": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/tha_Thai/train/002_00000.parquet"],
    },
    "Vietnamese": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/vie_Latn/train/001_00005.parquet"],
    },
    "Malay": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "repo_type": "dataset",
        "files": ["data/zsm_Latn/train/001_00000.parquet"],
    },
}

REQUIRED_COLUMN = "text"


def verify_parquet_schema(file_path, required_column=REQUIRED_COLUMN):
    """Checks (without loading the full file) that the parquet file has the required column."""
    try:
        schema = pq.ParquetFile(file_path).schema_arrow
    except Exception as e:
        print(f"    FAILED to read parquet schema: {e}")
        return False

    if required_column not in schema.names:
        print(f"    FAILED: '{required_column}' column not found. Available columns: {schema.names}")
        return False

    print(f"    Verified: '{required_column}' column present.")
    return True


def print_sample_rows(file_path, column=REQUIRED_COLUMN, n=3):
    """Prints n randomly chosen values from the given column as a sanity check."""
    try:
        texts = pq.read_table(file_path, columns=[column]).column(column).to_pylist()
    except Exception as e:
        print(f"    FAILED to read samples: {e}")
        return

    if not texts:
        print("    No rows available to sample.")
        return

    sample = random.sample(texts, min(n, len(texts)))
    print(f"    Sample rows ({len(sample)} of {len(texts)}):")
    for i, text in enumerate(sample, 1):
        preview = (text or "").replace("\n", " ")[:200]
        print(f"      [{i}] {preview}")


def download_file(repo_id, filename, local_path, repo_type="dataset", revision=None):
    print(f"--> Downloading {filename} from {repo_id}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            local_dir=os.path.dirname(local_path),
        )
        # hf_hub_download preserves the remote sub-directory structure under local_dir,
        # so move/rename to the flat destination path we expect.
        if os.path.abspath(downloaded_path) != os.path.abspath(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            os.replace(downloaded_path, local_path)
        print("    Done.")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch parquet data files from Hugging Face Hub.")
    parser.add_argument("--languages", nargs="+", help="Specific languages to download (e.g. English German). Default: all.")
    args = parser.parse_args()

    if not REMOTE_FILES:
        print("REMOTE_FILES is empty. Please populate it in fetch_data.py with the datasets to download.")
        return

    # Filter files based on user input
    target_files = REMOTE_FILES
    if args.languages:
        lower_targets = set(l.lower() for l in args.languages)
        target_files = {k: v for k, v in REMOTE_FILES.items() if k.lower() in lower_targets}

        if not target_files:
            print(f"No matching languages found for: {args.languages}")
            print(f"Available: {list(REMOTE_FILES.keys())}")
            return

    os.makedirs(MONO_DIR, exist_ok=True)

    # Track local paths for config update
    local_data_map = {}

    for lang, spec in target_files.items():
        repo_id = spec["repo_id"]
        repo_type = spec.get("repo_type", "dataset")
        revision = spec.get("revision")
        files = spec["files"]

        lang_dir = os.path.join(MONO_DIR, lang.lower())
        os.makedirs(lang_dir, exist_ok=True)

        local_paths = []
        for remote_filename in files:
            filename = os.path.basename(remote_filename)
            l_path = os.path.join(lang_dir, filename)

            if os.path.exists(l_path):
                print(f"Skipping {lang} (already exists): {l_path}")
            elif not download_file(repo_id, remote_filename, l_path, repo_type=repo_type, revision=revision):
                continue

            if verify_parquet_schema(l_path):
                print_sample_rows(l_path)
                local_paths.append(os.path.relpath(l_path, REPO_ROOT))
            else:
                print(f"    Discarding {l_path} (missing '{REQUIRED_COLUMN}' column).")
                os.remove(l_path)

        if local_paths:
            local_data_map[lang] = local_paths

    # Update Config
    config_path = os.path.join(REPO_ROOT, "config", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'monolingual_data' not in config:
            config['monolingual_data'] = {}

        for lang, paths in local_data_map.items():
            # Use globs for local paths to match folder content
            dir_path = os.path.dirname(paths[0])
            config['monolingual_data'][lang] = [os.path.join(dir_path, "*.parquet")]

        print("--> Updating config.yaml with local paths...")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        print("    Done.")
    else:
        print("Warning: config/config.yaml not found.")

    print(f"\nDownload complete for: {list(target_files.keys())}")


if __name__ == "__main__":
    main()
