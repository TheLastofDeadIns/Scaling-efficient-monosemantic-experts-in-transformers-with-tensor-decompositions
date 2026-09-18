"""Extend the router-key decay ablation to three seeds.

The upper-edge mechanism of the report rests on a single seed: exempting the
router keys from weight decay costs MONET-HD generalisation at N = 4096.  This
runs the remaining seeds for both settings at N = 256 and N = 4096.  Runs that
already exist -- the decayed settings belong to the main grid -- are skipped.

Run from the repository root:  python src/run_decay_seeds.py [out_dir]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from model import ModelConfig
from train import TrainConfig, train_one, run_name


def main(out_dir: str = "runs") -> None:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    jobs = [(n, decay, seed)
            for n in (16, 64)
            for decay in (True, False)
            for seed in (1, 2)]
    t0 = time.time()
    for i, (n, decay, seed) in enumerate(jobs, 1):
        mcfg = ModelConfig(arch="moe_hd", n_keys=n, seed=seed)
        tcfg = TrainConfig(decay_router_keys=decay)
        stem = run_name(mcfg, tcfg)
        f = base / (stem if stem.endswith(".npz") else stem + ".npz")
        if f.exists():
            print(f"[{i}/{len(jobs)}] skip {f.name}", flush=True)
            continue
        print(f"[{i}/{len(jobs)}] moe_hd n={n} N={n * n} "
              f"keys={'decayed' if decay else 'exempt'} seed={seed}  "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
        train_one(mcfg, tcfg, base, verbose=False)
        torch.cuda.empty_cache()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
