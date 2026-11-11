"""DESIGN CHECK, breakdown of earlystop_check.py by run-name prefix."""
import json, pathlib, collections, statistics
import pandas as pd
TAU, W = 0.95, 20000
stats = collections.defaultdict(lambda: {"read": 0, "reached": 0, "reached_then_below": 0, "rule_stops": 0, "rule_wrong": 0, "stop_steps": []})
for r in sorted(pathlib.Path("runs").iterdir()):
    if r.name.startswith("_") or not (r / "DONE").exists() or not (r / "metrics.parquet").exists():
        continue
    fam = "why" if r.name.startswith("why_") else r.name.split("_")[0]
    if r.name.startswith("famA_d2"): fam = "famA_d2"
    try:
        df = pd.read_parquet(r / "metrics.parquet"); fin = json.load(open(r / "summary.json"))["final_accuracy"]
    except Exception:
        continue
    a = df[(df.split == "iid_val") & (df.metric_name == "accuracy") & (df.depth == -1) & (df.breadth == -1)].sort_values("step").drop_duplicates("step", keep="last")
    s, v = a.step.to_numpy(), a.value.to_numpy()
    if len(s) == 0: continue
    st = stats[fam]; st["read"] += 1
    if (v >= TAU).any():
        st["reached"] += 1
        if fin < TAU: st["reached_then_below"] += 1
    for i in range(len(s)):
        win = (s >= s[i]) & (s <= s[i] + W)
        if s[i] + W <= s[-1] and (v[win] >= TAU).all():
            st["rule_stops"] += 1; st["stop_steps"].append(int(s[i] + W))
            if fin < TAU: st["rule_wrong"] += 1
            break
for fam, st in sorted(stats.items()):
    ss = st.pop("stop_steps")
    med = statistics.median(ss) if ss else None
    print(f"{fam:10s} " + " ".join(f"{k}={v}" for k, v in st.items()) + f" median_stop_step={med}")
