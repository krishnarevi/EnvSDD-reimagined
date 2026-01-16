import json
from pathlib import Path

INPUT_JSON = r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\EnvSDD-reimagined\jsons\test_track1.json"
OUTPUT_DIR = Path(INPUT_JSON).parent

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

real_samples = [x for x in data if x["label"] == "real"]

tta_samples = real_samples + [
    x for x in data if x["attack_type"] == "tta"
]

ata_samples = real_samples + [
    x for x in data if x["attack_type"] == "ata"
]

with open(OUTPUT_DIR / "test_track1_tta.json", "w") as f:
    json.dump(tta_samples, f, indent=4)

with open(OUTPUT_DIR / "test_track1_ata.json", "w") as f:
    json.dump(ata_samples, f, indent=4)

print("Saved test_track1_tta.json")
print("Saved test_track1_ata.json")
