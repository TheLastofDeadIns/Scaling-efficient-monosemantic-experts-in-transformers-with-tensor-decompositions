"""Priority-ordered, resumable experiment runner.

Designed for a session-limited environment: jobs are ordered so that the runs
carrying the paper's claims finish first, every completed run is cached on
disk, and the script stops cleanly before the session is killed.  Re-running it
picks up exactly where it stopped.

    python -u src/run_all.py /kaggle/working/runs --max-hours 8.5

Phases
------
1  Headline comparison at a fixed expert count, all seeds.  This is the table
   and Figure 4 of the report; without it there is no paper.
2  Scaling sweep: the expert-count axis, which carries the main claim that
   factorisation sets the sign of the effect.
3  Remaining architectures, for completeness of the frequency analysis.
4  Ablations: router input split, auxiliary loss weight, router-key decay.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ModelConfig
from train import TrainConfig, run_name, train_one

SEEDS = (0, 1, 2)
SHARED = ("moe_tucker", "moe_tt", "moe_cp")     # shared tensor basis
PRODUCT = ("moe_hd", "moe_vd")                  # product-key composition
ALL_MOE = SHARED + PRODUCT


def base_tc(**kw) -> TrainConfig:
    cfg = dict(steps=30_000, eval_every=100, train_frac=0.8,
               weight_decay=1.0, decay_router_keys=True)
    cfg.update(kw)
    return TrainConfig(**cfg)


def jobs() -> list[tuple[ModelConfig, TrainConfig, str]]:
    """(model config, train config, phase label), in priority order."""
    out = []
    tc = base_tc()

    # -- phase 1: headline comparison at N = 256 --------------------------- #
    for s in SEEDS:
        out.append((ModelConfig(arch="mlp", seed=s), tc, "1-headline"))
        for a in ALL_MOE:
            out.append((ModelConfig(arch=a, n_keys=16, seed=s), tc, "1-headline"))
        out.append((ModelConfig(arch="moe_full", n_keys=16, seed=s), tc, "1-headline"))

    # -- phase 2: expert-count sweep --------------------------------------- #
    for s in SEEDS:
        for n in (4, 8, 32, 64):
            for a in ("moe_tucker", "moe_hd"):          # the two opposite signs
                out.append((ModelConfig(arch=a, n_keys=n, seed=s), tc, "2-scaling"))
        for n in (4, 8, 32):                            # where it breaks down
            out.append((ModelConfig(arch="moe_full", n_keys=n, seed=s), tc, "2-scaling"))

    # -- phase 3: the rest of the grid ------------------------------------- #
    for s in SEEDS:
        for n in (4, 8, 32, 64):
            for a in ("moe_tt", "moe_cp", "moe_vd"):
                out.append((ModelConfig(arch=a, n_keys=n, seed=s), tc, "3-fill"))

    # -- phase 4: ablations, seed 0 only ----------------------------------- #
    for split in ("interleave", "random"):
        for a in ("moe_tucker", "moe_hd"):
            out.append((ModelConfig(arch=a, n_keys=16, router_split=split, seed=0),
                        tc, "4-router-split"))
    for lam in (0.0, 1e-2):
        for a in ("moe_tucker", "moe_hd"):
            out.append((ModelConfig(arch=a, n_keys=16, lambda_aux=lam, seed=0),
                        tc, "4-aux-loss"))
    for a in ("moe_tucker", "moe_hd", "moe_full"):
        for n in (16, 64):
            if a == "moe_full" and n > 16:
                continue
            out.append((ModelConfig(arch=a, n_keys=n, seed=0),
                        base_tc(decay_router_keys=False), "4-key-decay"))
    for frac in (0.4, 0.6):
        for a in ("moe_tucker", "moe_hd"):
            out.append((ModelConfig(arch=a, n_keys=16, seed=0),
                        base_tc(train_frac=frac), "4-data-fraction"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--max-hours", type=float, default=8.5,
                    help="stop before the session limit; re-run to continue")
    ap.add_argument("--phases", default="1,2,3,4",
                    help="comma-separated phase prefixes to run")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    want = tuple(args.phases.split(","))
    todo = [j for j in jobs() if j[2][0] in want]

    t0 = time.time()
    budget = args.max_hours * 3600
    trained = cached = 0

    for i, (mc, tc, phase) in enumerate(todo, 1):
        path = out / f"{run_name(mc, tc)}.npz"
        if path.exists():
            cached += 1
            continue
        elapsed = time.time() - t0
        if elapsed > budget:
            print(f"\nstopping at {elapsed/3600:.1f}h of {args.max_hours}h budget; "
                  f"re-run this script to continue")
            break
        print(f"[{i}/{len(todo)}] {phase:16s} {mc.arch:11s} n={mc.n_keys:<3} "
              f"seed={mc.seed} ({elapsed/3600:.1f}h elapsed)", flush=True)
        train_one(mc, tc, out, verbose=False)
        trained += 1

    n_done = len(list(out.glob("*.npz")))
    print(f"\ntrained {trained}, cached {cached}, {n_done} archives on disk, "
          f"{(time.time()-t0)/3600:.2f}h used")


if __name__ == "__main__":
    main()
