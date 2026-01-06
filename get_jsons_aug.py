import argparse
import json
import os
import sys
import pandas as pd
from tqdm import tqdm
import config_v2 as config  # Imports your original config

# ==========================================
# 1. RUNTIME CONFIGURATION OVERRIDES
# ==========================================
# Define the augmented root directly here (No changes to config.py needed)
AUG_ROOT = r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\DATA\Aug_Dataset" 

# Override the metadata CSV path in the loaded config object
# This ensures we read the COMBINED csv you created
config.dev_track1_meta = "metadata/datasplit_dev_aug.csv"

print("--- Configuration Override ---")
print(f"Augmented Data Root: {AUG_ROOT}")
print(f"Metadata CSV: {config.dev_track1_meta}")
print("------------------------------")

# ==========================================
# 2. HELPER: SCAN AUGMENTED FOLDER
# ==========================================
def index_augmented_files(root_dir):
    """
    Scans the Augmented Dataset and creates a dictionary:
    { 'filename.wav': '/full/path/to/filename.wav' }
    """
    print(f"Scanning {root_dir} for augmented files...")
    path_map = {}
    if not os.path.exists(root_dir):
        print(f"❌ ERROR: Augmented root '{root_dir}' not found on disk!")
        print("   Make sure you uploaded the 'Aug_Dataset' folder to the cluster root.")
        sys.exit(1)
        
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                # Map filename -> Full Path
                path_map[file] = os.path.join(root, file)
    
    print(f"✅ Found {len(path_map)} augmented files.")
    return path_map

# ==========================================
# 3. MAIN GENERATION LOGIC
# ==========================================
def main():
    # 1. Index the augmented files so we can find them instantly
    aug_file_map = index_augmented_files(AUG_ROOT)
    
    # 2. Setup Output Paths
    os.makedirs(config.metadata_json_file, exist_ok=True)
    train_meta_path = f"{config.metadata_json_file}/dev_track1_train_aug.json"
    
    train_data = []
    
    # 3. Read the Augmented CSV
    try:
        metadata = pd.read_csv(config.dev_track1_meta)
    except FileNotFoundError:
        sys.exit(f"❌ ERROR: Could not find CSV at {config.dev_track1_meta}")

    print("Processing metadata rows...")
    # csv columns: wavename, dataset, scene, event, caption, usage
    
    for i in tqdm(range(len(metadata))):
        wavename = metadata.iloc[i, 0]
        dataset = metadata.iloc[i, 1]
        event_col = metadata.iloc[i, 3] # We used this for 'augmented' tag
        usage = metadata.iloc[i, -1]
        
        # We only care about TRAIN data for augmentation
        if usage != 'train':
            continue

        # -------------------------------------------------
        # CASE A: NEW AUGMENTED DATA
        # -------------------------------------------------
        if event_col == 'augmented':
            # Look up the path in our map
            full_path = aug_file_map.get(wavename)
            
            if full_path:
                # Guess label from path/dataset name
                label = 'real' if 'real' in full_path.lower() else 'fake'
                
                train_data.append({
                    'file_path': full_path,
                    'label': label,
                    'attack_type': 'augmented',
                    'generative_model': dataset,
                })
            else:
                # File exists in CSV but not found in Aug_Dataset folder
                # This usually happens if CSV generation and upload aren't synced
                # We skip it to avoid crashing the training
                pass

        # -------------------------------------------------
        # CASE B: ORIGINAL DATA (Standard Logic)
        # -------------------------------------------------
        else:
            # 1. Add the Real File
            real_path = f'{config.dev_track1_audio}/real_audio/{dataset}/{wavename}'
            train_data.append({
                'file_path': real_path,
                'label': 'real',
                'attack_type': 'real',
                'generative_model': 'real',
            })
            
            # 2. Add Standard TTA Fakes
            for model_name in ['audiogen', 'audioldm1', 'audioldm2']:
                train_data.append({
                    'file_path': f'{config.dev_track1_audio}/fake_audio/TTA/{model_name}/{dataset}/{wavename}',
                    'label': 'fake',
                    'attack_type': 'tta',
                    'generative_model': model_name,
                })
            
            # 3. Add Standard ATA Fakes
            for model_name in ['audioldm1']:
                train_data.append({
                    'file_path': f'{config.dev_track1_audio}/fake_audio/ATA/{model_name}/{dataset}/{wavename}',
                    'label': 'fake',
                    'attack_type': 'ata',
                    'generative_model': model_name,
                })

    # Save Result
    print(f"Saving {len(train_data)} training items to {train_meta_path}")
    with open(train_meta_path, 'w') as f:
        json.dump(train_data, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    main()