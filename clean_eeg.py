import mne
import os

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = "./data"
OUTPUT_DIR = "./Cleaned"

os.makedirs(OUTPUT_DIR, exist_ok=True)

HP = 1.0
LP = 40.0

# -----------------------------
# CLEAN ONE FILE
# -----------------------------
def clean_file(fname):
    print(f"\n=== Cleaning {fname} ===")
    raw = mne.io.read_raw_gdf(fname, preload=True)

    # 1) Filtering
    raw.filter(HP, LP)

    # 2) Bad channels (none in your dataset)
    raw.info['bads'] = []
    raw.interpolate_bads()

    # 3) ICA
    ica = mne.preprocessing.ICA(n_components=20, random_state=42)
    ica.fit(raw)

    # SAFELY remove EOG only if exists
    try:
        eog_inds, _ = ica.find_bads_eog(raw)
        ica.exclude = eog_inds
        print("ICA removed EOG:", eog_inds)
    except:
        print("No EOG channels found - skipping EOG removal")

    raw_clean = ica.apply(raw.copy())

    # 4) Save
    base = os.path.basename(fname).replace(".gdf", "")
    outname = f"{OUTPUT_DIR}/{base}_cleaned_raw.fif"
    raw_clean.save(outname, overwrite=True)
    print(f"Saved -> {outname}")


# -----------------------------
# MAIN LOOP
# -----------------------------
all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.startswith("ID") and f.endswith(".gdf")])

print("FILES FOUND:", all_files)

for f in all_files:
    full_path = os.path.join(INPUT_DIR, f) # linha adicionada pq o MNE estava procurando na pasta raiz e não na data
    clean_file(full_path)
