"""Composition-rank sweep: from rank-one (product-key-like) to dense Tucker.

Run from the repository root:  python src/run_comp_rank.py [out_dir]

One subdirectory per composition rank, because run_name() does not encode it.
The dense-core Tucker runs of the main grid serve as the upper endpoint.
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

RANKS = (1, 2, 4, 8, 16, 32)
KEYS = (16, 32, 64)


def main(out_dir: str = "runs_comp") -> None:
    base, tcfg, t0 = Path(out_dir), TrainConfig(), time.time()
    jobs = [(r, n) for r in RANKS for n in KEYS]
    for i, (r, n) in enumerate(jobs, 1):
        out = base / f"r{r}"
        out.mkdir(parents=True, exist_ok=True)
        mcfg = ModelConfig(arch="moe_tucker", n_keys=n, seed=0, comp_rank=r)
        stem = run_name(mcfg, tcfg)
        f = out / (stem if stem.endswith(".npz") else stem + ".npz")
        if f.exists():
            print(f"[{i}/{len(jobs)}] skip r{r}/{f.name}", flush=True)
            continue
        print(f"[{i}/{len(jobs)}] comp_rank={r:2d}  n={n} N={n * n}  "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
        train_one(mcfg, tcfg, out, verbose=False)
        torch.cuda.empty_cache()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs_comp")
