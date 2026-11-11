"""DESIGN CHECK, not a result. Does any finished run reach 0.95 overall eval
accuracy and then finish below 0.95? And would the candidate stop rule,
'at or above 0.95 at every eval for W steps', ever stop a run that does not
finish at or above 0.95? Read only."""
import json, pathlib
import pandas as pd
TAU, W = 0.95, 20000
rows, viol_any, viol_rule = [], [], []
for r in sorted(pathlib.Path("runs").iterdir()):
    if r.name.startswith("_") or not (r / "DONE").exists() or not (r / "metrics.parquet").exists():
        continue
    try:
        df = pd.read_parquet(r / "metrics.parquet")
        fin = json.load(open(r / "summary.json"))["final_accuracy"]
    except Exception as e:
        continue
    a = df[(df.split == "iid_val") & (df.metric_name == "accuracy") & (df.depth == -1) & (df.breadth == -1)]
    a = a.sort_values("step").drop_duplicates("step", keep="last")
    s, v = a.step.to_numpy(), a.value.to_numpy()
    if len(s) == 0:
        continue
    reached = (v >= TAU).any()
    stop_at = None
    for i in range(len(s)):
        win = (s >= s[i]) & (s <= s[i] + W)
        if s[i] + W <= s[-1] and (v[win] >= TAU).all():
            stop_at = s[i] + W; break
    rows.append((r.name, len(s), reached, stop_at, fin))
    if reached and fin < TAU:
        first = s[v >= TAU][0]; after = v[s >= first]
        viol_any.append((r.name, int(first), float(after.min()), fin))
    if stop_at is not None and fin < TAU:
        viol_rule.append((r.name, int(stop_at), fin))
print(f"finished runs read: {len(rows)}")
print(f"reached {TAU} at some eval: {sum(1 for x in rows if x[2])}")
print(f"of those, finished below {TAU}: {len(viol_any)}")
for x in viol_any: print("   ", x)
print(f"rule 'all evals >= {TAU} for {W} steps' would stop: {sum(1 for x in rows if x[3] is not None)}")
print(f"of those, finished below {TAU}: {len(viol_rule)}")
for x in viol_rule: print("   ", x)
stopped = [x for x in rows if x[3] is not None]
if stopped:
    import statistics
    print(f"stop step among stopped runs: min {min(x[3] for x in stopped)}, median {statistics.median(x[3] for x in stopped)}, max {max(x[3] for x in stopped)}")
