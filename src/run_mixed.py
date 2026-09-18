"""Training runs on the mixed add/multiply task.

This is the positive control for expert specialisation: the subtask is a
genuine partition of the input distribution, so class-based criteria apply.

Run from the repository root:  python src/run_mixed.py [out_dir]
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

JOBS = [("moe_tucker", 16), ("moe_hd", 16),
        ("moe_tucker", 32), ("moe_hd", 32),
        ("moe_tucker", 64), ("moe_hd", 64)]


def main(out_dir: str = "runs_mixed") -> None:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    tcfg, t0 = TrainConfig(task="mixed"), time.time()
    for i, (arch, n) in enumerate(JOBS, 1):
        mcfg = ModelConfig(arch=arch, n_keys=n, seed=0)
        stem = run_name(mcfg, tcfg)
        f = base / (stem if stem.endswith(".npz") else stem + ".npz")
        if f.exists():
            print(f"[{i}/{len(JOBS)}] skip {f.name}", flush=True)
            continue
        print(f"[{i}/{len(JOBS)}] {arch} n={n} N={n * n} "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)
        train_one(mcfg, tcfg, base, verbose=False)
        torch.cuda.empty_cache()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs_mixed")
