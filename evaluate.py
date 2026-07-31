import yaml
import json
import os
import argparse
import random
import glob
import multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from datasets import load_dataset

# --- Metrics Functions (Eq 1-4 from Paper) ---

def calc_compression(tokenizer, samples):
    """Eq 1: Bytes / Tokens (Higher is better)"""
    total_bytes = sum(len(t.encode('utf-8')) for t in samples)
    total_tokens = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in samples)
    return total_bytes / total_tokens if total_tokens > 0 else 0

def calc_fertility(tokenizer, samples):
    """Eq 2: Tokens / Words (Lower is better)"""
    total_tokens = 0
    total_words = 0
    for t in samples:
        words = t.split() # Simple whitespace pre-tokenizer approximation
        if not words: continue
        tokens = tokenizer.encode(t, add_special_tokens=False)
        total_tokens += len(tokens)
        total_words += len(words)
    return total_tokens / total_words if total_words > 0 else 0

def calc_vocab_util(tokenizer, all_samples):
    """Eq 3: % of vocab used"""
    unique_ids = set()
    for t in all_samples:
        unique_ids.update(tokenizer.encode(t, add_special_tokens=False))
    return len(unique_ids) / tokenizer.vocab_size

def calc_token_count(tokenizer, samples):
    """Calculates the total number of tokens for the given samples."""
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in samples)

def calc_gini(costs):
    """Eq 4: Gini Coefficient on Tokenization Cost (Lower is better)"""
    if not costs: return 0
    costs = sorted(costs)
    n = len(costs)
    numerator = sum((n + 1 - (i + 1)) * c for i, c in enumerate(costs))
    denominator = sum(costs)
    return (1 / n) * (n + 1 - (2 * numerator / denominator))

# --- Data Loading ---

def stream_parquet_samples_random(file_paths, col_name, target_mb, seed=42):
    """
    Randomly sample text from parquet files until ~target_mb is reached.
    Uses PyArrow to sample row groups randomly, avoiding full dataset scans.
    """
    target_bytes = target_mb * 1024 * 1024
    samples = []
    total_bytes = 0

    # Normalize file_paths to a list of specific files
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Expand globs if necessary
    expanded_paths = []
    for p in file_paths:
        if "*" in p:
            expanded_paths.extend(glob.glob(p))
        else:
            expanded_paths.append(p)

    if not expanded_paths:
        print(f"Warning: No files found for paths: {file_paths}")
        return []

    # 1. Collect all available row groups across all files
    # Store as (file_path, row_group_index)
    all_row_groups = []

    try:
        for p in expanded_paths:
            try:
                pf = pq.ParquetFile(p)
                for i in range(pf.num_row_groups):
                    all_row_groups.append((p, i))
            except Exception as e:
                print(f"Warning: Could not read metadata for {p}: {e}")
    except Exception as e:
        print(f"Error initializing parquet files: {e}")
        return []

    # 2. Shuffle the row groups to process them in random order
    rng = random.Random(seed)
    rng.shuffle(all_row_groups)

    # 3. Iterate and collect text until target size is reached
    for p, rg_idx in all_row_groups:
        if total_bytes >= target_bytes:
            break

        try:
            # Read specific row group
            pf = pq.ParquetFile(p)
            table = pf.read_row_group(rg_idx, columns=[col_name])

            # Convert to python list (shuffling rows within the group for extra randomness)
            texts = table[col_name].to_pylist()
            rng.shuffle(texts)

            for text in texts:
                if not isinstance(text, str) or not text.strip():
                    continue

                text_bytes = len(text.encode('utf-8'))

                # If this single row group is massive, we might overshoot slightly,
                # but checking per-row keeps us close to target.
                samples.append(text)
                total_bytes += text_bytes

                if total_bytes >= target_bytes:
                    break

        except Exception as e:
            print(f"Error reading row group {rg_idx} from {p}: {e}")
            continue

    return samples


def stream_parquet_samples(file_paths, col_name, target_mb):
    samples = []
    curr_bytes = 0
    target_bytes = target_mb * 1024 * 1024

    # Handle single string path or list of paths
    if isinstance(file_paths, str): file_paths = [file_paths]

    try:
        # datasets can handle a list of glob patterns
        ds = load_dataset("parquet", data_files={'train': file_paths}, split='train', streaming=True)
        for row in ds:
            text = row.get(col_name, "")
            if not isinstance(text, str) or not text.strip(): continue

            samples.append(text)
            curr_bytes += len(text.encode('utf-8'))
            if curr_bytes >= target_bytes: break
    except Exception as e:
        print(f"Error streaming {file_paths}: {e}")

    return samples

def load_parallel_file(path):
    # Assumes text files for parallel data (like FLORES dev sets)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except:
        return []

# --- Parallel Tokenizer Evaluation ---
#
# Each tokenizer is loaded and evaluated in its own worker process. The (potentially
# large) monolingual/parallel sample data is sent to each worker once via the Pool
# initializer rather than per-task, to avoid re-pickling it for every tokenizer.

_worker_state = {}

def _init_worker(mono_data, parallel_data, data_stats, cache_dir, hf_token, load_mono):
    _worker_state.update(
        mono_data=mono_data,
        parallel_data=parallel_data,
        data_stats=data_stats,
        cache_dir=cache_dir,
        hf_token=hf_token,
        load_mono=load_mono,
    )


def _evaluate_tokenizer(task):
    """Runs in a worker process. Returns (model_name, res, granular_res); res is None on load failure."""
    model_name, selected_metrics = task
    mono_data = _worker_state['mono_data']
    parallel_data = _worker_state['parallel_data']
    data_stats = _worker_state['data_stats']
    cache_dir = _worker_state['cache_dir']
    hf_token = _worker_state['hf_token']
    load_mono = _worker_state['load_mono']

    print(f"Evaluating {model_name}...")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    except Exception as e:
        print(f"  Skipping {model_name} (load fail: {e})")
        return model_name, None, None

    res = {}
    granular_res = {'data_size': {}, 'token_counts': {}}

    if load_mono:
        for lang, stats in data_stats.items():
            granular_res['data_size'][lang] = stats['size_mb']

        want_fertility = selected_metrics is None or 'fertility' in selected_metrics
        want_compression = selected_metrics is None or 'compression' in selected_metrics
        want_vocab_per_lang = selected_metrics is None or 'vocab_utilization_per_lang' in selected_metrics
        if want_fertility: res['fertility'] = {}
        if want_compression: res['compression'] = {}
        if want_vocab_per_lang: res['vocab_utilization_per_lang'] = {}

        all_text = []
        for lang, samples in mono_data.items():
            if want_fertility: res['fertility'][lang] = calc_fertility(tok, samples)
            if want_compression: res['compression'][lang] = calc_compression(tok, samples)
            if want_vocab_per_lang: res['vocab_utilization_per_lang'][lang] = calc_vocab_util(tok, samples)

            # Always calculate token counts for granular stats
            granular_res['token_counts'][lang] = calc_token_count(tok, samples)
            all_text.extend(samples)

        # Global Vocab Utilization
        if selected_metrics is None or 'vocab_utilization' in selected_metrics:
            res['vocab_utilization'] = calc_vocab_util(tok, all_text)

    # Fairness (Gini)
    if selected_metrics is None or 'gini' in selected_metrics:
        costs = []
        for lang, samples in parallel_data.items():
            # We calculate compression on parallel data to get comparable costs
            comp = calc_compression(tok, samples)
            if comp > 0: costs.append(1 / comp)
        res['gini'] = calc_gini(costs)

    print(f"  Finished {model_name}")
    return model_name, res, granular_res


def _tokenizers_to_evaluate(model_names, results, selected_metrics, skip_existing):
    to_eval = []
    for model_name in model_names:
        needs_eval = not skip_existing or model_name not in results
        if not needs_eval and selected_metrics:
            needs_eval = any(m not in results[model_name] for m in selected_metrics)

        if needs_eval:
            to_eval.append(model_name)
        else:
            print(f"Skipping {model_name} (already evaluated)")
    return to_eval


def evaluate_tokenizers(model_names, mono_data, parallel_data, data_stats, selected_metrics,
                         cache_dir, hf_token, load_mono, results, granular_results,
                         num_workers=None, on_result=None):
    """
    Evaluates model_names in parallel worker processes, merging each tokenizer's
    results into `results`/`granular_results` (in place) as it completes.
    `on_result(results, granular_results)`, if given, is called after each merge
    (e.g. to save incrementally).
    """
    if not model_names:
        return results, granular_results

    num_workers = max(1, min(num_workers or (os.cpu_count() or 1), len(model_names)))
    print(f"Evaluating {len(model_names)} tokenizer(s) using {num_workers} worker process(es)...")

    tasks = [(model_name, selected_metrics) for model_name in model_names]

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(mono_data, parallel_data, data_stats, cache_dir, hf_token, load_mono),
    ) as pool:
        for model_name, new_res, new_granular in pool.imap_unordered(_evaluate_tokenizer, tasks):
            if new_res is None:
                continue

            merged_res = results.get(model_name, {}).copy()
            merged_res.update(new_res)
            results[model_name] = merged_res

            merged_granular = granular_results.get(model_name, {'data_size': {}, 'token_counts': {}}).copy()
            if load_mono:
                merged_granular['data_size'] = new_granular['data_size']
                merged_granular['token_counts'] = new_granular['token_counts']
            granular_results[model_name] = merged_granular

            if on_result:
                on_result(results, granular_results)

    return results, granular_results


# --- Shared Helpers ---

def _load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    if not os.path.exists(config_path):
        config_path = "config.yaml"  # Fallback to current directory
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def _load_hf_token():
    token_path = os.path.join(os.path.dirname(__file__), "hf_token.txt")
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = f.read().strip()
        print("Loaded Hugging Face token from hf_token.txt")
        return token
    return None


# --- Main Runners ---

def run_eval(selected_metrics=None, skip_existing=True, num_workers=None):
    config = _load_config()

    os.makedirs(config['output_dir'], exist_ok=True)
    cache_dir = config.get("cache_dir", None)

    out_file = os.path.join(config['output_dir'], "results.json")
    granular_out_file = os.path.join(config['output_dir'], "granular_results.json")

    results = _load_json(out_file)
    granular_results = _load_json(granular_out_file)
    if results:
        print(f"Loaded existing results for {len(results)} tokenizers")

    hf_token = _load_hf_token()

    # 1. Load Data
    # Only load monolingual data if we need efficiency metrics
    load_mono = not (selected_metrics and set(selected_metrics) == {'gini'})

    mono_data = {}
    data_stats = {}  # Stats per language

    if load_mono:
        print("Loading Monolingual Data...")
        for lang, paths in config['monolingual_data'].items():
            print(f"  Streaming {lang}...")
            samples = stream_parquet_samples(paths, config['text_column_name'], config['sample_size_mb'])
            mono_data[lang] = samples

            total_bytes = sum(len(t.encode('utf-8')) for t in samples)
            data_stats[lang] = {
                "samples": len(samples),
                "size_mb": total_bytes / (1024 * 1024),
            }

    print("Loading Parallel Data...")
    parallel_data = {lang: load_parallel_file(path) for lang, path in config['parallel_data'].items()}

    # 2. Evaluate Tokenizers (in parallel)
    to_eval = _tokenizers_to_evaluate(config['tokenizers'], results, selected_metrics, skip_existing)
    evaluate_tokenizers(
        to_eval, mono_data, parallel_data, data_stats, selected_metrics,
        cache_dir, hf_token, load_mono, results, granular_results,
        num_workers=num_workers,
    )

    # 3. Save
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

    with open(granular_out_file, 'w') as f:
        json.dump(granular_results, f, indent=2)
    print(f"Granular results (size & tokens) saved to {granular_out_file}")


def run_eval_random(selected_metrics=None, skip_existing=True, random_mode=False, seed=42, num_workers=None):
    config = _load_config()

    # Create unique output directory based on seed
    output_dir = os.path.join(config['output_dir'], f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    cache_dir = config.get("cache_dir", None)

    out_file = os.path.join(output_dir, "results.json")
    granular_out_file = os.path.join(output_dir, "granular_results.json")

    results = _load_json(out_file)
    granular_results = _load_json(granular_out_file)
    if results:
        print(f"Loaded existing results for {len(results)} tokenizers")

    hf_token = _load_hf_token()

    # 1. Load Data (Randomly Sampled)
    load_mono = not (selected_metrics and set(selected_metrics) == {'gini'})

    mono_data = {}
    data_stats = {}  # Stats per language

    if load_mono:
        print(f"Loading Monolingual Data (Random Seed {seed})...")
        for lang, paths in config['monolingual_data'].items():
            print(f"  Streaming {lang}...")
            samples = stream_parquet_samples_random(
                paths,
                config['text_column_name'],
                config['sample_size_mb'],
                seed=seed,
            )
            mono_data[lang] = samples

            total_bytes = sum(len(t.encode('utf-8')) for t in samples)
            data_stats[lang] = {
                "samples": len(samples),
                "size_mb": total_bytes / (1024 * 1024),
            }

    # Save metadata about the data used
    metadata = {
        "seed": seed,
        "target_sample_size_mb": config['sample_size_mb'],
        "data_stats": data_stats,
        "monolingual_data_paths": config['monolingual_data'],
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Data metadata saved to {os.path.join(output_dir, 'metadata.json')}")

    print("Loading Parallel Data...")
    parallel_data = {lang: load_parallel_file(path) for lang, path in config['parallel_data'].items()}

    # 2. Evaluate Tokenizers (in parallel), saving after each one completes
    def _save(results, granular_results):
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        with open(granular_out_file, 'w') as f:
            json.dump(granular_results, f, indent=2)

    to_eval = _tokenizers_to_evaluate(config['tokenizers'], results, selected_metrics, skip_existing)
    evaluate_tokenizers(
        to_eval, mono_data, parallel_data, data_stats, selected_metrics,
        cache_dir, hf_token, load_mono, results, granular_results,
        num_workers=num_workers, on_result=_save,
    )

    _save(results, granular_results)

    print(f"Results saved to {out_file}")
    print(f"Granular results saved to {granular_out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tokenizers.")
    parser.add_argument("--random", action="store_true", default=False,
                        help="Evaluate tokenizers in random order. Default: False.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling. Default: 42.")
    parser.add_argument("--metrics", nargs="+",
                        choices=['fertility', 'compression', 'vocab_utilization', 'vocab_utilization_per_lang', 'gini'],
                        help="Specific metrics to run. Default: all.")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip tokenizers that already have results. Default: True.")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-evaluate all tokenizers, even if they have existing results.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of tokenizers to evaluate in parallel. Default: min(CPU count, number of tokenizers).")
    args = parser.parse_args()

    #run_eval(args.metrics, args.skip_existing, num_workers=args.workers)
    run_eval_random(args.metrics, args.skip_existing, args.random, args.seed, num_workers=args.workers)
