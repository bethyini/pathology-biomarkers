#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import pickle

# ─── Adjust these ───────────────────────────────────────────
CSV_PATH   = Path("her2.csv")
SLIDES_DIR = Path("CLAM/svs_slides")
# ────────────────────────────────────────────────────────────

# 1) load CSV
df = pd.read_csv(CSV_PATH)

# 2) derive 'sample' by dropping the final "-XX"
df["sample"] = df["SAMPLE_ID"].str.rsplit("-", n=1).str[0]

# 3) build lookup: sample → (HER2_Amp, ERBB2)
lookup = (
    df.set_index("sample")[["HER2_Amp", "ERBB2"]]
      .apply(tuple, axis=1)
      .to_dict()
)

# 4) list slide IDs
slide_ids = {p.stem for p in SLIDES_DIR.glob("*.svs")}

# 5) intersect
common = slide_ids & lookup.keys()

# 6) final dict
common_dict = { sid: lookup[sid] for sid in common }

# 7) pickle
OUTFILE = Path("CLAM/her2_label_dict.pkl")
with open(OUTFILE, "wb") as f:
    pickle.dump(common_dict, f)

print(f"Pickled lookup ({len(common_dict)} entries) to {OUTFILE}")
