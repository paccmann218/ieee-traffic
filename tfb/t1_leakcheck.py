import sys, numpy as np, pandas as pd
from pathlib import Path
REL = Path(sys.argv[1]); panel = "D12_I5_N"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
tr = sorted((REL/"corridors"/panel/"train"/"mainline_states").glob("**/*.parquet"))
mk = {p.name: p for p in (REL/"corridors"/panel/"train"/"mainline_states_masked").glob("**/*.parquet")}
f = tr[40]; t = pd.read_parquet(f); m = pd.read_parquet(mk[f.name])
print(f.name, t.shape, m.shape); print(t.dtypes); print(t.head(3).T)
keys = ["timestamp", "station_id", "link_id"]
j = t.merge(m, on=keys, suffixes=("", "_m"))
tgt = j[j.speed_kmh_m.isna() & j.is_score_eligible.astype(bool)]
print("targets:", len(tgt), "of", len(j))
for col in m.columns:
    if col in keys: continue
    a, b = tgt[col].astype(str) if col in tgt else None, tgt[col + "_m"].astype(str) if col + "_m" in tgt else None
    if b is None: continue
    same = (tgt[col].astype(str) == tgt[col + "_m"].astype(str)).mean()
    nn = tgt[col + "_m"].notna().mean()
    print(f"{col:22s} masked_nonnull={nn:.3f} equal_to_truth={same:.3f}")
