"""Follow-up experiments after the course paper.  Resumable, priority-ordered.

    python -u src/run_kops.py /kaggle/working/runs_kops --max-hours 8.5
    python -u src/run_kops.py /kaggle/working/runs_kops --dry-run      # list jobs

Jobs run seed-major: every phase on seed 0 first (calibrate, full, kops,
frozen), then seed 1, then seed 2, so a partial run already has one complete
seed of everything.

Phases
------
full       The original task, (a + b) mod 113, with a router whose two sides
           both read the whole input, as in MONET and PEER.  Same layers, expert
           counts, seeds and protocol as the main grid, so every number is
           directly comparable with results/summary_main.csv.  Answers "how much
           of the result was the router being handed a and b separately".
frozen     The original task with router keys frozen at their random
           initialisation.  If a layer still generalises as fast, its router
           needs no learned structure at all.
calibrate  Dense MLP on (a_1 + ... + a_k) mod P for k = 2, 3, 4 and several
           training-set sizes.  The main study found that a layer can look
           incapable when it is merely under-supplied with data, so the data
           budget for each k is fixed on the baseline first.
kops       The k-operand sweep: k = 2, 3, 4, four factorised layers plus the
           MLP, three expert counts and the router splits natural / pair / full.
           The training-set size for each k is the smallest one at which the MLP
           generalised in the calibration phase.  Seed 0 for everything first,
           then seeds 1 and 2.

Every finished run is an .npz in the output directory and is skipped on re-run.
Analyse with:  python src/kops_analysis.py <out_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ModelConfig            # noqa: E402
from train import TrainConfig, run_name, train_one   # noqa: E402

LAYERS = ("moe_hd", "moe_vd", "moe_cp", "moe_tucker", "moe_tt")
KOPS_LAYERS = ("moe_hd", "moe_cp", "moe_tucker", "moe_tt")
CALIB_SIZES = {2: (0,), 3: (10_000, 30_000, 80_000), 4: (30_000, 100_000, 300_000)}
DEFAULT_SIZES = {2: 0, 3: 30_000, 4: 100_000}
KOPS_BATCH = 4096
KOPS_TEST = 20_000


def main_tc(**kw) -> TrainConfig:
    """Protocol of the main study."""
    cfg = dict(steps=30_000, eval_every=100, train_frac=0.8, weight_decay=1.0,
               decay_router_keys=True, P=113, n_operands=2)
    cfg.update(kw)
    return TrainConfig(**cfg)


def kops_tc(P: int, k: int, n_train: int) -> TrainConfig:
    return main_tc(P=P, n_operands=k, n_train=n_train,
                   n_test=KOPS_TEST if n_train > 0 else 0,
                   batch_size=KOPS_BATCH)


def summaries(out: Path) -> list[dict]:
    rows = []
    for p in out.glob("*.npz"):
        try:
            with np.load(p, allow_pickle=True) as z:
                s = json.loads(str(z["config/summary"]))
                s["_tcfg"] = json.loads(str(z["config/train"]))
            rows.append(s)
        except Exception:                      # half-written archive
            continue
    return rows


def chosen_sizes(out: Path, P: int) -> dict[int, int]:
    """Smallest calibrated n_train at which the MLP generalised, per k."""
    sizes = dict(DEFAULT_SIZES)
    rows = [r for r in summaries(out)
            if r["arch"] == "mlp" and r["_tcfg"].get("P") == P]
    for k, cands in CALIB_SIZES.items():
        got = {r["_tcfg"].get("n_train", 0): r["grok_step"] for r in rows
               if r["_tcfg"].get("n_operands", 2) == k}
        if not got:
            continue
        ok = [n for n in cands if n in got and got[n] > 0]
        sizes[k] = min(ok, key=lambda n: (n == 0, n)) if ok else max(cands)
    return sizes


def jobs_for(phase: str, seed: int, out: Path, P: int):
    """Jobs of one phase for one seed.  Called lazily, so the 'kops' phase sees
    the calibration runs finished earlier in the same invocation."""
    todo = []
    if phase == "calibrate":
        if seed == 0:
            for k, cands in CALIB_SIZES.items():
                for n_train in cands:
                    todo.append((ModelConfig(arch="mlp", seed=0), kops_tc(P, k, n_train)))
    elif phase == "full":
        tc = main_tc()
        for n in (4, 8, 16, 32, 64):
            for a in LAYERS:
                todo.append((ModelConfig(arch=a, n_keys=n, seed=seed,
                                         router_split="full"), tc))
    elif phase == "frozen":
        tc = main_tc()
        for n in (16, 64):
            for a in ("moe_tucker", "moe_cp", "moe_hd"):
                todo.append((ModelConfig(arch=a, n_keys=n, seed=seed,
                                         freeze_router=True), tc))
    elif phase == "kops":
        sizes = chosen_sizes(out, P)
        print(f"  [kops, seed {seed}] training-set size per k: {sizes}", flush=True)
        for k in (2, 3, 4):
            tc = kops_tc(P, k, sizes[k])
            todo.append((ModelConfig(arch="mlp", seed=seed), tc))
            splits = ("natural", "full") if k == 2 else ("natural", "pair", "full")
            for n in (4, 16, 32):
                for split in splits:
                    for a in KOPS_LAYERS:
                        todo.append((ModelConfig(arch=a, n_keys=n, seed=seed,
                                                 router_split=split), tc))
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return todo


def all_jobs(phases, seeds, out: Path, P: int):
    """Seed-major order: every phase on seed 0 first, then seed 1, ...

    A run shared by two phases (the calibration MLP at the chosen size is also
    the MLP of the k-operand sweep) is listed once."""
    seen = set()
    for s in seeds:
        for ph in phases:
            for mc, tc in jobs_for(ph, s, out, P):
                name = run_name(mc, tc)
                if name not in seen:
                    seen.add(name)
                    yield mc, tc, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--phases", default="calibrate,full,kops,frozen")
    ap.add_argument("--max-hours", type=float, default=8.5)
    ap.add_argument("--P", type=int, default=47, help="modulus for the k-operand phases")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dry-run", action="store_true", help="list jobs and exit")
    args = ap.parse_args()

    import torch
    if not torch.backends.opt_einsum.is_available():
        # without it torch contracts multi-operand einsums left to right, and the
        # Tucker / TT layers build (batch, r, r, r) intermediates: ~10x slower
        print("WARNING: opt_einsum is not installed; Tucker and TT layers will be "
              "an order of magnitude slower.  pip install opt_einsum", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    phases = tuple(p.strip() for p in args.phases.split(","))
    seeds = tuple(int(s) for s in args.seeds.split(","))

    if args.dry_run:
        n = 0
        for mc, tc, ph in all_jobs(phases, seeds, out, args.P):
            done = (out / f"{run_name(mc, tc)}.npz").exists()
            print(f"{'done' if done else '    '}  {ph:10s} {run_name(mc, tc)}")
            n += 1
        print(f"{n} jobs")
        return

    t0, trained, cached = time.time(), 0, 0
    for i, (mc, tc, phase) in enumerate(all_jobs(phases, seeds, out, args.P), 1):
        path = out / f"{run_name(mc, tc)}.npz"
        if path.exists():
            cached += 1
            continue
        elapsed = time.time() - t0
        if elapsed > args.max_hours * 3600:
            print(f"\nstopping at {elapsed / 3600:.1f}h; re-run to continue")
            break
        print(f"[{i}] {phase:10s} {mc.arch:11s} n={mc.n_keys:<3} "
              f"split={mc.router_split:8s} k={tc.n_operands} seed={mc.seed} "
              f"({elapsed / 3600:.1f}h)", flush=True)
        train_one(mc, tc, out, verbose=False)
        trained += 1
    print(f"\ntrained {trained}, cached {cached}, "
          f"{len(list(out.glob('*.npz')))} archives, {(time.time() - t0) / 3600:.2f}h")


if __name__ == "__main__":
    main()
