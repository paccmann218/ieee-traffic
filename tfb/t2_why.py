"""For each actual ongoing train window, print selector quantities at slots s0-4..s0."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tfb"); import cache
from t2_simsel import vcut_fd, slot_of, REL

for panel in cache.PANELS[:8]:
    z = cache.load(panel, "train"); vc = vcut_fd(panel, z["links"]); dates = list(z["dates"])
    sp = z["speed"]; el = z["elig"].astype(bool); ms = z["mspeed"]
    wi = pd.read_csv(REL/"task2"/panel/"train"/"window_index.csv")
    for r in wi[wi.condition == "queue_ongoing"].itertuples():
        d = dates.index(r.date); s0 = slot_of(r.forecast_origin); out = []
        for s in range(s0 - 4, s0 + 1):
            hist = slice(s - 12, s); hor = slice(s + 1, s + 7)
            q = sp[d] <= vc[None, :]; qe = q & el[d]
            lab = q[hor]; pers = np.repeat(qe[s - 1][None], 6, 0)
            piou = (pers & lab).sum() / max(1, (pers | lab).sum())
            nq_last = int(qe[s - 1].sum()); nlab = int(lab.sum())
            # blanks in masked layer around (other windows' horizon/buffer)
            out.append(f"s{s-s0:+d}:cov={el[d,hist].mean():.3f},pi={piou:.2f},nq={nq_last},nl={nlab}")
        print(panel, r.window_id[-3:], r.date, s0, " | ".join(out), flush=True)
