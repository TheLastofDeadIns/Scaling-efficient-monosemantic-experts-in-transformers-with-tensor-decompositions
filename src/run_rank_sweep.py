"""Rank sweep: Tensor-Train rank against expert count, plus a Tucker control.

Run from the repository root:  python src/run_rank_sweep.py [out_dir]

One subdirectory per rank, because run_name() does not encode the tensor ranks
and runs at different ranks would otherwise overwrite each other.
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

JOBS = [("moe_tt", n, {"tt_rank": r}, f"tt_r{r}")
        for r in (8, 16, 32, 64) for n in (16, 32, 64)]
JOBS += [("moe_tucker", n, {"tucker_rank": 4}, "tucker_r4") for n in (16, 32, 64)]


def main(out_dir: str = "runs_rank") -> None:
    base, tcfg, t0 = Path(out_dir), TrainConfig(), time.time()
    for i, (arch, n, extra, sub) in enumerate(JOBS, 1):
        out = base / sub
        out.mkdir(parents=True, exist_ok=True)
        mcfg = ModelConfig(arch=arch, n_keys=n, seed=0, **extra)
        stem = run_name(mcfg, tcfg)
        f = out / (stem if stem.endswith(".npz") else stem + ".npz")
        if f.exists():
            print(f"[{i}/{len(JOBS)}] skip {sub}/{f.name}", flush=True)
            continue
        print(f"[{i}/{len(JOBS)}] {sub}  {arch} n={n} N={n * n} {extra}  "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
        train_one(mcfg, tcfg, out, verbose=False)
        torch.cuda.empty_cache()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs_rank")
