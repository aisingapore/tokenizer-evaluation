# Tokenizer Evaluation Framework

A comprehensive framework for evaluating Large Language Model (LLM) tokenizers across multiple languages and domains, with a focus on efficiency, fairness, and vocabulary utilization.

## Overview

This repository provides a systematic approach to benchmarking tokenizers used in modern LLMs. It evaluates tokenizers across multiple dimensions:

- **Efficiency**: Compression ratio and fertility metrics
- **Fairness**: Gini coefficient for multilingual equity
- **Vocabulary Utilization**: Global and per-language vocabulary usage
- **Language Coverage**: Support for Nordic languages (Icelandic, Faroese, Swedish, Danish, Norwegian), Germanic languages (German, Dutch, English), and code

The framework follows evaluation methodologies from recent research, including approaches used by Apertus and other multilingual tokenizer studies.

## Features

- 📊 **Six core metrics**: Compression ratio, fertility, Gini coefficient, vocabulary utilization (global & per-language), and token counts
- 🌍 **Multilingual support**: Evaluate tokenizers across 8+ languages with special focus on under-represented Nordic languages
- 💻 **Code tokenization**: Includes programming language evaluation
- 📈 **Automated visualization**: Generate publication-ready plots for all metrics
- ⚙️ **Flexible configuration**: Easy YAML-based setup for tokenizers, data sources, and evaluation parameters
- 🔄 **Incremental evaluation**: Skip already-evaluated tokenizers or re-run specific metrics
- 📝 **Detailed reporting**: Auto-generated comprehensive evaluation reports

## Installation

### Prerequisites

- Python 3.8+
- Hugging Face account (for accessing gated models like Llama)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd wp4-tokenizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up Hugging Face authentication for gated models:
```bash
echo "your_hf_token_here" > hf_token.txt
```

## Quick Start

### 1. Download Tokenizers

Pre-download all tokenizers specified in the configuration:

```bash
python download.py
```

This caches tokenizers locally in `./hf_cache/` for faster evaluation.

### 2. Prepare Data

The framework expects:
- **Monolingual data**: Parquet files in `data/monolingual/{language}/` directories
- **Parallel data**: FLORES+ dev sets in `openlanguagedata/flores_plus/`

Use the provided scripts to fetch data:
```bash
python fetch_data.py      # Download monolingual data from HuggingFace datasets
python fetch_flores.py    # Download FLORES+ parallel data
```

### 3. Run Evaluation

Evaluate all tokenizers across all metrics:

```bash
python evaluate.py
```

**Options:**
- `--metrics [metric1 metric2 ...]`: Run specific metrics only (choices: `fertility`, `compression`, `vocab_utilization`, `vocab_utilization_per_lang`, `gini`)
- `--skip-existing`: Skip tokenizers already evaluated (default: True)
- `--no-skip-existing`: Re-evaluate all tokenizers

**Examples:**
```bash
# Evaluate only fairness metric
python evaluate.py --metrics gini

# Re-evaluate all tokenizers and metrics
python evaluate.py --no-skip-existing

# Evaluate specific metrics
python evaluate.py --metrics compression fertility
```

### 4. Generate Visualizations

Create plots for all metrics:

```bash
python visualize.py
```

Plots are saved to `eval_results/plots/`:
- `compression_ratio.png` - Efficiency per language (higher is better)
- `fertility.png` - Tokens per word per language (lower is better)
- `gini_coefficient.png` - Fairness across languages (lower is better)
- `vocab_utilization.png` - Global vocabulary usage (higher is better)
- `vocab_utilization_per_language.png` - Language-specific vocabulary usage
- `total_token_count.png` - Absolute token counts per language
- `data_size_mb.png` - Input data sizes

## Configuration

Edit `config/config.yaml` to customize the evaluation:

```yaml
# Cache and output directories
cache_dir: "./hf_cache"
output_dir: "./eval_results"

# Tokenizers to evaluate
tokenizers:
  - "mistralai/Mistral-Nemo-Base-2407"
  - "meta-llama/Meta-Llama-3.1-8B"
  - "Qwen/Qwen2.5-7B"
  - "google/gemma-2-9b"
  - "TrustLLMeu/baseline-7-8b_2-3t-tokens_llama"
  # Add custom tokenizers here

# Data paths
monolingual_data:
  English: "data/monolingual/english/*.parquet"
  Icelandic: "data/monolingual/icelandic/*.parquet"
  # ... more languages

parallel_data:
  English: "openlanguagedata/flores_plus/eng_Latn.dev"
  # ... more languages

# Evaluation parameters
sample_size_mb: 10  # MB of data per language for efficiency tests
text_column_name: "text"  # Column name in parquet files
```

## Metrics Explained

### 1. Compression Ratio
**Formula**: `Bytes / Tokens`

Measures how efficiently a tokenizer compresses text. Higher values indicate better compression.

### 2. Fertility
**Formula**: `Tokens / Words`

Average number of subword tokens per word. Lower values indicate more efficient word-level encoding.

### 3. Gini Coefficient
**Formula**: `Gini = (1/n) × (n + 1 - (2 × Σ[(n + 1 - i) × cᵢ] / Σcᵢ))`

Measures inequality in tokenization costs across languages using parallel data. Lower values indicate fairer multilingual support (0 = perfect equality, 1 = maximum inequality).

### 4. Vocabulary Utilization (Global)
**Formula**: `Unique Tokens Used / Total Vocabulary Size`

Percentage of the vocabulary actually used across all test data. Higher values indicate more efficient vocabulary design.

### 5. Vocabulary Utilization (Per Language)
**Formula**: `Unique Tokens Used in Language L / Total Vocabulary Size`

Language-specific vocabulary usage, revealing how different languages utilize the tokenizer's vocabulary.

### 6. Total Token Count
**Formula**: `Σ length(tokenize(textᵢ))`

Absolute number of tokens needed to encode the test corpus. Lower values indicate more efficient encoding.

For detailed metric explanations and interpretation, see [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

## Project Structure

```
wp4-tokenizer/
├── config/
│   └── config.yaml           # Main configuration file
├── data/
│   └── monolingual/          # Monolingual evaluation data
│       ├── english/
│       ├── icelandic/
│       └── ...
├── openlanguagedata/
│   └── flores_plus/          # FLORES+ parallel data
├── eval_results/
│   ├── results.json          # Main evaluation results
│   ├── granular_results.json # Detailed token counts and data sizes
│   └── plots/                # Generated visualizations
├── hf_cache/                 # Cached tokenizer files
├── download.py               # Download and cache tokenizers
├── evaluate.py               # Main evaluation script
├── visualize.py              # Generate plots from results
├── fetch_data.py             # Fetch monolingual data
├── fetch_flores.py           # Fetch FLORES+ parallel data
├── requirements.txt          # Python dependencies
├── EVALUATION_REPORT.md      # Detailed results and analysis
└── README.md                 # This file
```

## Output Files

### `eval_results/results.json`
Main results file containing all metrics for each tokenizer:
```json
{
  "mistralai/Mistral-Nemo-Base-2407": {
    "fertility": {"English": 1.33, "Icelandic": 2.48, ...},
    "compression": {"English": 4.65, "Icelandic": 2.74, ...},
    "vocab_utilization": 0.585,
    "vocab_utilization_per_lang": {"English": 0.346, ...},
    "gini": 0.097
  },
  ...
}
```

### `eval_results/granular_results.json`
Detailed statistics including:
- Data sizes (MB) per language
- Total token counts per language per tokenizer

### `eval_results/plots/`
Publication-ready visualizations for all metrics.

## Use Cases

- **Tokenizer Development**: Benchmark your custom tokenizer against established models
- **Model Selection**: Choose the most appropriate tokenizer for your multilingual application
- **Fairness Analysis**: Evaluate multilingual equity in tokenizer design
- **Research**: Support academic research on tokenizer efficiency and fairness

## Evaluated Tokenizers

The framework currently evaluates:
- **Mistral-Nemo-Base-2407**: Mistral AI's 2024 release
- **Meta-Llama-3.1-8B**: Meta's Llama 3.1 series
- **Qwen2.5-7B**: Alibaba's Qwen 2.5 model
- **gemma-2-9b**: Google's Gemma 2 series
- **baseline-7-8b_2-3t-tokens_llama**: TrustLLMeu custom tokenizer with enhanced Nordic language support

## Data Sources

### Monolingual Data
10MB per language collected from:
- FineWeb-Edu
- FineWeb2
- HPLT (High-Performance Language Technologies)
- Custom Faroese collections

### Parallel Data
FLORES+ development sets for fairness evaluation (Gini coefficient).

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{tokenizer_eval_framework,
  title={Tokenizer Evaluation Framework},
  author={TrustLLM WP4},
  year={2024},
  url={https://github.com/your-org/wp4-tokenizer}
}
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional metrics (e.g., character coverage, byte-pair encoding efficiency)
- More languages and language families
- Integration with more tokenizer types (SentencePiece, custom implementations)
- Performance optimizations for large-scale evaluations

## License

[Add your license information here]

## Acknowledgments

- Evaluation methodology inspired by Apertus and recent multilingual tokenizer research
- FLORES+ dataset for parallel evaluation data
- FineWeb, HPLT, and other data providers

## Contact

For questions or issues, please open a GitHub issue or contact [your contact information].

---

**Note**: This framework is part of the TrustLLM project Work Package 4, focusing on improving multilingual language model support for Nordic and European languages.

