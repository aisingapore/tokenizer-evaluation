import os
import shutil
from datasets import load_dataset

# Mapping from ISO 639-3 (likely in dataset) to FLORES-200 style (expected by config)
# We will check what is actually in the dataset.
iso_map = {
    "eng": "eng_Latn",
    "ind": "ind_Latn",
    "mya": "mya_Mymr",
    "fil": "fil_Latn",
    "tha": "tha_Thai",
    "lao": "lao_Laoo",
    "tam": "tam_Taml",
    "vie": "vie_Latn",
    "zsm": "zsm_Latn",
    "cmn": "cmn_Hans",
    "khm": "khm_Khmr",
    "ban": "ban_Latn",
    "jav": "jav_Latn",
    "sun": "sun_Latn",
}

dataset_id = "openlanguagedata/flores_plus"
local_save_path = "openlanguagedata/flores_plus"
temp_dir = "temp_flores_download"

def fetch_and_save_flores():
    if os.path.exists(local_save_path):
        shutil.rmtree(local_save_path)
        
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    print(f"Loading {dataset_id}...")
    try:
        ds = load_dataset(dataset_id, split="dev", trust_remote_code=True)
        
        # Debug: print first few languages
        print("Sample languages in dataset:")
        sample_langs = set()
        for i, row in enumerate(ds):
            if i > 1000: break
            sample_langs.add(row.get('iso_639_3'))
        print(sample_langs)
        
        # Filter and save
        content_by_file = {filename: [] for filename in iso_map.values()}
        
        for row in ds:
            iso_code = row.get('iso_639_3')
            if iso_code in iso_map:
                target_filename = iso_map[iso_code]
                text = row.get('text', '').strip()
                if text:
                    content_by_file[target_filename].append(text)
        
        # Save to temp files
        for filename, texts in content_by_file.items():
            if not texts:
                print(f"Warning: No data found for {filename}")
                continue
            
            file_path = os.path.join(temp_dir, f"{filename}.dev")
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in texts:
                    f.write(line + '\n')
            print(f"Saved {len(texts)} lines to {file_path}")
            
        # Move to final destination
        os.makedirs(os.path.dirname(local_save_path), exist_ok=True)
        shutil.move(temp_dir, local_save_path)
        print(f"Successfully saved data to {local_save_path}")

    except Exception as e:
        print(f"Error processing dataset: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    fetch_and_save_flores()
