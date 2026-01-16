from pathlib import Path

def p(path_str):
    # Accept Path or string, return string with forward slashes (no trailing slash)
    p = Path(path_str)
    s = str(p.as_posix())
    return s.rstrip("/")

beats_path = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\Models\BEATs_iter3_plus_AS2M.pt")
w2v2_path  = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\Models\xlsr2_300m.pt")

# ENVSDD development data
envsdd_dev_folder = r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\DATA\Track1\development"
envsdd_dev_split_path = r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\code\EnvSDD_project\datasplit_dev.csv"

# Metadata JSON folder 
metadata_json_file = r".\jsons"

# ENVSDD test data
envsdd_test_folder = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\DATA\Track1\test_track1_2\test_track1")
envsdd_test_meta_path  = p(r"C:\Users\krevi\Documents\Uni\SEM3\ENVSDD\EnvSDD-reimagined\metadata\test_track1.csv")
