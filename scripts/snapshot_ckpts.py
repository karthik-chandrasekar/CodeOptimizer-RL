"""Copy each GRPO run's last.pt to <run>/snapshots/stepNNNN.pt whenever it is re-saved.

    python scripts/snapshot_ckpts.py --dir artifacts/files --runs grpo_agent grpo_nofb

Exits once every run has written final.pt, or has had no live process for 2 minutes (crashed).
"""
import argparse, os, shutil, subprocess, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True)
p.add_argument("--runs", nargs="+", required=True)
a = p.parse_args()
seen, t0, finished = {}, time.time(), set()
while len(finished) < len(a.runs):
    for r in a.runs:
        d = os.path.join(a.dir, r); ck = os.path.join(d, "last.pt")
        if r in finished:
            continue
        if os.path.exists(os.path.join(d, "final.pt")):
            # a checkpoint saved together with final.pt (e.g. at the last step) still gets its snapshot
            if os.path.exists(ck) and seen.get(r) != os.path.getmtime(ck):
                step = torch.load(ck, map_location="cpu", weights_only=False)["extra"]["step"]
                os.makedirs(os.path.join(d, "snapshots"), exist_ok=True)
                shutil.copy(ck, os.path.join(d, "snapshots", f"step{step:04d}.pt"))
                print(time.strftime("%T"), r, "snapshot step", step, "(at finish)", flush=True)
            finished.add(r); continue
        alive = subprocess.run(["pgrep", "-f", f"grpo.out_dir={d}( |$)"], capture_output=True).returncode == 0
        if not alive and time.time() - t0 > 120:
            print(time.strftime("%T"), r, "no live process and no final.pt - giving up", flush=True); finished.add(r); continue
        if not os.path.exists(ck):
            continue
        m = os.path.getmtime(ck)
        if seen.get(r) == m or time.time() - m < 20:
            continue
        step = torch.load(ck, map_location="cpu", weights_only=False)["extra"]["step"]
        os.makedirs(os.path.join(d, "snapshots"), exist_ok=True)
        shutil.copy(ck, os.path.join(d, "snapshots", f"step{step:04d}.pt")); seen[r] = m
        print(time.strftime("%T"), r, "snapshot step", step, flush=True)
    time.sleep(30)
