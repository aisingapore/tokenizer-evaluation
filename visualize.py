import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_results(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_metric_per_language(df, metric_name, output_dir, title_suffix=""):
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # df has columns: Tokenizer, Language, Value
    g = sns.barplot(
        data=df, 
        x="Language", 
        y="Value", 
        hue="Tokenizer",
        palette="viridis"
    )
    
    plt.title(f"{metric_name} per Language {title_suffix}")
    plt.ylabel(metric_name)
    plt.xlabel("Language")
    plt.xticks(rotation=45)
    plt.legend(title="Tokenizer", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = f"{metric_name.lower().replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Saved {filename}")
    plt.close()

def plot_simple_bar(df, metric_name, y_label, output_dir, title_suffix=""):
    """Plots a simple bar chart for global metrics like Gini or Vocab Util"""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.barplot(
        data=df,
        x="Tokenizer",
        y=metric_name,
        hue="Tokenizer",
        legend=False,
        palette="viridis"
    )
    
    plt.title(f"{metric_name} {title_suffix}")
    plt.ylabel(y_label)
    plt.xlabel("Tokenizer")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = f"{metric_name.lower().replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename))
    print(f"Saved {filename}")
    plt.close()

def main():
    results_path = "eval_results/results.json"
    granular_results_path = "eval_results/granular_results.json"
    output_dir = "eval_results/plots"
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_results(results_path)
    granular_data = load_results(granular_results_path)
    
    # Prepare dataframes
    fertility_data = []
    compression_data = []
    vocab_data = []
    vocab_per_lang_data = []
    gini_data = []
    
    # Granular dataframes
    token_count_data = []
    data_size_data = []
    
    if data:
        for tokenizer, metrics in data.items():
            # Shorten tokenizer name for better legend
            short_name = tokenizer.split('/')[-1]
            
            # Fertility
            for lang, val in metrics.get('fertility', {}).items():
                fertility_data.append({
                    "Tokenizer": short_name,
                    "Language": lang,
                    "Value": val
                })
                
            # Compression
            for lang, val in metrics.get('compression', {}).items():
                compression_data.append({
                    "Tokenizer": short_name,
                    "Language": lang,
                    "Value": val
                })
                
            # Vocab Global
            if 'vocab_utilization' in metrics:
                vocab_data.append({
                    "Tokenizer": short_name,
                    "Vocab Utilization": metrics.get('vocab_utilization', 0)
                })

            # Gini Coefficient
            if 'gini' in metrics:
                gini_data.append({
                    "Tokenizer": short_name,
                    "Gini Coefficient": metrics.get('gini', 0)
                })

            # Vocab Per Language
            for lang, val in metrics.get('vocab_utilization_per_lang', {}).items():
                vocab_per_lang_data.append({
                    "Tokenizer": short_name,
                    "Language": lang,
                    "Value": val
                })

    if granular_data:
        # Data Size is generally the same per language across tokenizers, 
        # but the structure is {Tokenizer: {data_size: {Lang: Size}}}
        # We can just pick the first tokenizer to plot data sizes, or plot them all to be safe.
        # Let's plot just once since it's the input data size.
        
        first_tok = list(granular_data.keys())[0]
        size_info = granular_data[first_tok].get('data_size', {})
        
        for lang, size_mb in size_info.items():
             data_size_data.append({
                "Language": lang,
                "Size (MB)": size_mb
             })

        for tokenizer, metrics in granular_data.items():
            short_name = tokenizer.split('/')[-1]
            
            # Token Counts
            for lang, count in metrics.get('token_counts', {}).items():
                token_count_data.append({
                    "Tokenizer": short_name,
                    "Language": lang,
                    "Value": count
                })

    # Plot Standard Metrics
    if fertility_data:
        df_fert = pd.DataFrame(fertility_data)
        plot_metric_per_language(df_fert, "Fertility", output_dir, "(Lower is better)")
        
    if compression_data:
        df_comp = pd.DataFrame(compression_data)
        plot_metric_per_language(df_comp, "Compression Ratio", output_dir, "(Higher is better)")

    if vocab_data:
        plot_simple_bar(pd.DataFrame(vocab_data), "Vocab Utilization", "Utilization Ratio", output_dir, "(Global)")
        
    if gini_data:
        plot_simple_bar(pd.DataFrame(gini_data), "Gini Coefficient", "Gini Index", output_dir, "(Lower is better / fairer)")

    if vocab_per_lang_data:
        df_vocab_lang = pd.DataFrame(vocab_per_lang_data)
        plot_metric_per_language(df_vocab_lang, "Vocab Utilization Per Language", output_dir, "")

    # Plot Granular Metrics
    if token_count_data:
        df_tokens = pd.DataFrame(token_count_data)
        plot_metric_per_language(df_tokens, "Total Token Count", output_dir, "")
        
    if data_size_data:
        df_size = pd.DataFrame(data_size_data)
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sns.barplot(data=df_size, x="Language", y="Size (MB)", hue="Language", legend=False, palette="viridis")
        plt.title("Input Data Size per Language (MB)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "data_size_mb.png"))
        print("Saved data_size_mb.png")
        plt.close()

if __name__ == "__main__":
    main()
