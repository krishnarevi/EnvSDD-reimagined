# config_v2.py
from pathlib import Path

def p(path_str):
    # Accept Path or string, return string with forward slashes (no trailing slash)
    p = Path(path_str)
    s = str(p.as_posix())
    return s.rstrip("/")

# --- Windows locations 
beats_path = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\Models\BEATs_iter3_plus_AS2M.pt")
w2v2_path  = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\Models\xlsr2_300m.pt")

# main project folder 
main_folder = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\EnvSDD-reimagined")

# development track 1 audio folder and metadata CSV
dev_track1_audio = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\DATA\Track1\development")
dev_track1_meta  = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\EnvSDD-reimagined\metadata\dev_track1.csv")


test_track1_audio = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\DATA\Track1\test_track1_2\test_track1")
test_track1_meta  = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\EnvSDD-reimagined\metadata\test_track1.csv")

eval_track1_audio = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\DATA\Track1\eval_track1")
eval_track1_meta  = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\EnvSDD-reimagined\metadata\eval_track1.csv")

metadata_json_file = p(f"{main_folder}/jsons")


