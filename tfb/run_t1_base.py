"""Score base estimators on train with the local scorers. Usage: run_t1_base.py [panel ...]"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "tfb")
import cache, score_local as sl, t1_base as tb

panels = sys.argv[1:] or cache.PANELS
rows = []
for panel in panels:
    t0 = time.time(); z = cache.load(panel, "train"); lanes = sl.lanes_t1(panel, z["links"]); t3 = sl.T3(panel, z)
    cands = {
        "profile": (tb.profile(z, ch="speed"), tb.profile(z, ch="flow")),
        "tinterp": (tb.time_interp(z, "speed"), tb.time_interp(z, "flow")),
        "truth": (z["speed"], z["flow"]),
    }
    for name, (pv, pq) in cands.items():
        s1 = sl.score_t1(z, pv, pq, lanes); s3 = t3.score(z, pv, pq)
        rows.append(dict(panel=panel, model=name, S_state=s1["S_state"], S_phys=s3["S_physics"],
                         **{f"rmse_v_{r}": s1[r]["rmse_v"] for r in sl.REGIMES}, **{f"rmse_q_{r}": s1[r]["rmse_q"] for r in sl.REGIMES},
                         **{f"LWR_{r}": s3[r]["S_LWR"] for r in sl.REGIMES}, FD=np.mean([s3[r]["S_FD"] for r in sl.REGIMES])))
    print(panel, f"{time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(df.round(4).to_string())
fam = df.assign(fam=df.panel.str.rsplit("_", n=1).str[0]).groupby(["model", "fam"])[["S_state", "S_phys"]].mean().groupby("model").mean()
print(fam.round(4))
df.to_csv(f"{__import__('pathlib').Path.home()}/tfb/work/t1_base_scores.csv", index=False)
