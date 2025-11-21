import yaml
import os
from transformers import AutoTokenizer

def download_tokenizers(config_path=None):
    if config_path is None:
        # Try config/config.yaml first, then config.yaml
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
        if not os.path.exists(config_path):
            config_path = "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cache_dir = config.get("cache_dir", "./cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading tokenizers to {cache_dir}...")
    
    for model_name in config['tokenizers']:
        try:
            print(f"Fetching {model_name}...")
            # We only need the tokenizer, not the model weights
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            print(f"Successfully cached {model_name}")
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")

if __name__ == "__main__":
    download_tokenizers()