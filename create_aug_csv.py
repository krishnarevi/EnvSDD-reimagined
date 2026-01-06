import os
import pandas as pd

# --- Configuration ---
original_csv = r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\EnvSDD-reimagined\metadata\dev_track1.csv"       # Your original CSV file
output_csv = r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\EnvSDD-reimagined\metadata\datasplit_dev_aug.csv"  # The new combined CSV
aug_root_dir = r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\DATA\Aug_Dataset"          # Your augmented data root

# 1. Load Original CSV
df_orig = pd.read_csv(original_csv)
print(f"Original CSV loaded: {len(df_orig)} rows.")

# 2. Scan Augmented Data and Build Rows
new_rows = []

for root, dirs, files in os.walk(aug_root_dir):
    for file in files:
        if file.endswith(".wav"):
            # Get full path parts to infer dataset name
            # Example: Aug_Dataset/fake_audio/ATA/audioldm1/file.wav
            path_parts = root.split(os.sep)
            
            # Logic to infer 'dataset' name from folder structure
            dataset_name = "augmented_unknown"
            
            if "fake_audio" in path_parts:
                # Try to grab the generator name (e.g., audioldm1)
                try:
                    # Find 'fake_audio' index and look 2 levels deeper (ATA -> audioldm1)
                    idx = path_parts.index("fake_audio")
                    if len(path_parts) > idx + 2:
                        dataset_name = path_parts[idx+2]
                    else:
                        dataset_name = path_parts[-1] # Fallback to immediate parent
                except ValueError:
                    dataset_name = "augmented_fake"
            elif "real_audio" in path_parts:
                dataset_name = "augmented_real"

            # Create entry matching EXACT schema: wavename,dataset,scene,event,caption,usage
            new_rows.append({
                'wavename': file,
                'dataset': dataset_name,
                'scene': 'none',           # Placeholder
                'event': 'augmented',      # Placeholder
                'caption': 'Augmented',    # Placeholder
                'usage': 'train'           # FORCE to train split
            })

# 3. Merge and Save
if new_rows:
    df_aug = pd.DataFrame(new_rows)
    
    # Ensure columns order matches the original CSV exactly
    # (wavename, dataset, scene, event, caption, usage)
    df_aug = df_aug[df_orig.columns]
    
    # Concatenate original + augmented
    df_combined = pd.concat([df_orig, df_aug], ignore_index=True)
    
    # Save to new file
    df_combined.to_csv(output_csv, index=False)
    print(f"Success! Created {output_csv}")
    print(f"Original rows: {len(df_orig)} | New rows: {len(df_aug)} | Total: {len(df_combined)}")
    print("Sample new row:")
    print(df_aug.iloc[0].to_dict())
else:
    print("No new wav files found. Check 'aug_root_dir' path.")