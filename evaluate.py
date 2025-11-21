import yaml
import json
import os
import argparse
import numpy as np
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

# --- Main Runner ---

def run_eval(selected_metrics=None, skip_existing=True):
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    if not os.path.exists(config_path):
        config_path = "config.yaml" # Fallback to current directory

    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    cache_dir = config.get("cache_dir", None)
    
    # Load existing results if they exist
    out_file = os.path.join(config['output_dir'], "results.json")
    granular_out_file = os.path.join(config['output_dir'], "granular_results.json")
    
    existing_results = {}
    existing_granular_results = {}
    
    # If skipping existing, load them first
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            existing_results = json.load(f)
            print(f"Loaded existing results for {len(existing_results)} tokenizers")
    
    if os.path.exists(granular_out_file):
        with open(granular_out_file, 'r') as f:
            existing_granular_results = json.load(f)
    
    # Load HF Token if available
    hf_token = None
    token_path = os.path.join(os.path.dirname(__file__), "hf_token.txt")
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            hf_token = f.read().strip()
        print("Loaded Hugging Face token from hf_token.txt")
    
    # 1. Load Data
    # Only load monolingual data if we need efficiency metrics
    load_mono = True
    if selected_metrics and 'gini' in selected_metrics and len(selected_metrics) == 1:
        load_mono = False
    
    mono_data = {}
    data_stats = {} # Stats per language
    
    if load_mono:
        print("Loading Monolingual Data...")
        for lang, paths in config['monolingual_data'].items():
            print(f"  Streaming {lang}...")
            samples = stream_parquet_samples(paths, config['text_column_name'], config['sample_size_mb'])
            mono_data[lang] = samples
            
            # Calculate size in MB
            total_bytes = sum(len(t.encode('utf-8')) for t in samples)
            data_stats[lang] = {
                "samples": len(samples),
                "size_mb": total_bytes / (1024 * 1024)
            }
        
    print("Loading Parallel Data...")
    parallel_data = {}
    for lang, path in config['parallel_data'].items():
        parallel_data[lang] = load_parallel_file(path)

    # 2. Evaluate Tokenizers
    results = existing_results.copy()  # Start with existing results
    granular_results = existing_granular_results.copy()  # Start with existing granular results
    
    for model_name in config['tokenizers']:
        # Determine if we need to evaluate this tokenizer
        needs_eval = False
        
        if not skip_existing:
            needs_eval = True
        elif model_name not in results:
            needs_eval = True
        else:
            # Check if any requested metric is missing
            # If no specific metrics requested, and we have results, we assume it's done (unless skip_existing=False)
            # But if specific metrics are requested (like 'gini'), we check if they exist.
            if selected_metrics:
                for m in selected_metrics:
                    if m not in results[model_name]:
                        needs_eval = True
                        break
        
        if not needs_eval:
            print(f"Skipping {model_name} (already evaluated)")
            continue
            
        print(f"Evaluating {model_name}...")
        try:
            tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
        except Exception as e:
            print(f"  Skipping {model_name} (load fail: {e})")
            continue
            
        # Initialize result container, preserving existing if we are just adding metrics
        if model_name in results:
            res = results[model_name]
        else:
            res = {}
            
        if model_name in granular_results:
            granular_res = granular_results[model_name]
        else:
            granular_res = {'data_size': {}, 'token_counts': {}}

        # Copy data stats to granular results if we have them
        if load_mono:
            for lang, stats in data_stats.items():
                granular_res['data_size'][lang] = stats['size_mb']

        # Efficiency Metrics & Vocab Utilization per Language
        if load_mono:
            all_text = []
            
            if selected_metrics is None or 'fertility' in selected_metrics:
                if 'fertility' not in res: res['fertility'] = {}
            if selected_metrics is None or 'compression' in selected_metrics:
                if 'compression' not in res: res['compression'] = {}
            if selected_metrics is None or 'vocab_utilization_per_lang' in selected_metrics:
                if 'vocab_utilization_per_lang' not in res: res['vocab_utilization_per_lang'] = {}
                
            for lang, samples in mono_data.items():
                if selected_metrics is None or 'fertility' in selected_metrics:
                    res['fertility'][lang] = calc_fertility(tok, samples)
                if selected_metrics is None or 'compression' in selected_metrics:
                    res['compression'][lang] = calc_compression(tok, samples)
                
                if selected_metrics is None or 'vocab_utilization_per_lang' in selected_metrics:
                     res['vocab_utilization_per_lang'][lang] = calc_vocab_util(tok, samples)

                # Always calculate token counts for granular stats
                token_count = calc_token_count(tok, samples)
                granular_res['token_counts'][lang] = token_count

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
                if comp > 0: costs.append(1/comp)
            res['gini'] = calc_gini(costs)

        results[model_name] = res
        granular_results[model_name] = granular_res

    # 3. Save
    out_file = os.path.join(config['output_dir'], "results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

    granular_out_file = os.path.join(config['output_dir'], "granular_results.json")
    with open(granular_out_file, 'w') as f:
        json.dump(granular_results, f, indent=2)
    print(f"Granular results (size & tokens) saved to {granular_out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tokenizers.")
    parser.add_argument("--metrics", nargs="+", 
                        choices=['fertility', 'compression', 'vocab_utilization', 'vocab_utilization_per_lang', 'gini'],
                        help="Specific metrics to run. Default: all.")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip tokenizers that already have results. Default: True.")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-evaluate all tokenizers, even if they have existing results.")
    args = parser.parse_args()
    
    run_eval(args.metrics, args.skip_existing)
