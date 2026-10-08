"""What did each policy do on file tasks?  Localization, format and completion diagnostics.

    python scripts/analyze_files.py --tasks artifacts/data_r2/files_test_prof.jsonl --runs profile=.../profile_trajectories.jsonl full=...
"""
import argparse, collections, json

p = argparse.ArgumentParser()
p.add_argument("--tasks", required=True)
p.add_argument("--runs", nargs="+", required=True, help="name=path/to/*_trajectories.jsonl")
a = p.parse_args()
tasks = {t["task_id"]: t["meta"] for t in map(json.loads, open(a.tasks))}
slow_chance = sum(len(m["slow_functions"]) / len(m["functions"]) for m in tasks.values()) / len(tasks)
hot_chance = sum(1 / len(m["functions"]) for m in tasks.values()) / len(tasks)
print(f"chance: a random function is slow {slow_chance:.2f}; it is the hottest (largest step-0 time share) {hot_chance:.2f}")
for spec in a.runs:
    name, path = spec.split("=", 1)
    byd = collections.defaultdict(lambda: [0, 0, 0]); by_k = collections.defaultdict(lambda: [0, 0])
    pos, corr, first_slow, first_hot = collections.Counter(), collections.defaultdict(lambda: [0, 0]), [], []
    for r in map(json.loads, open(path)):
        m = tasks[r["task_id"]]; fns, slow = m["functions"], set(m["slow_functions"]); sp = 1 / r["best_ratio"]
        prof = m.get("profile0") or {}
        hottest = max(prof, key=prof.get) if prof else None
        b = byd[m["d"]]; b[0] += 1; b[1] += sp >= 1.5; b[2] += sp >= 0.8 * r["reference_speedup"]
        k = 0
        for s in r["steps"]:
            e = [f for f in s["edited"] if f in fns]
            if s["action_kind"] != "edit" or not e:
                continue
            k += 1; h = bool(slow & set(e))
            by_k[k][0] += h; by_k[k][1] += 1; pos[fns.index(e[0])] += 1
            corr["slow fn" if h else "fast fn"][0] += s["status"] == "ok"; corr["slow fn" if h else "fast fn"][1] += 1
            if k == 1:
                first_slow.append(h)
                if hottest:
                    first_hot.append(hottest in e)
    print(f"== {name}")
    print("   success>=1.5 / complete, by #slow functions:", {d: f"{v[1]/v[0]:.2f}/{v[2]/v[0]:.2f}" for d, v in sorted(byd.items())})
    print(f"   first edit: targets a slow function {sum(first_slow)/max(1,len(first_slow)):.2f}, the hottest function "
          f"{sum(first_hot)/max(1,len(first_hot)):.2f}  (n={len(first_slow)})")
    print("   edit #k targets a slow function:", {k: f"{v[0]/v[1]:.2f}" for k, v in sorted(by_k.items()) if v[1] >= 20})
    print("   position edited:", dict(sorted(pos.items())), "| correct-edit rate:", {k: f"{v[0]/v[1]:.2f}" for k, v in corr.items()})
