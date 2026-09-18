"""MONET's specialisation criterion, applied and checked against ablation.

On the mixed task a genuine partition of the input exists, so the criterion can
be run end to end: score every router factor separately on each subtask, flag
the factors whose score on one subtask exceeds the other by a factor of two,
ablate them, and measure the accuracy drop on each subtask.  A random control
ablates the same number of factors, averaged over N_CTRL draws.

Composed experts (i, j) cannot be ablated individually -- the layers never
materialise them, contracting g1 and g2 separately -- so we score and ablate
the 2*H*n router factors instead, zeroing their routing weight without
renormalising the remainder.

Run from the repository root:  python src/mixed_ablation.py [runs_dir]
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from data import make_data
from model import ModelConfig, ModAddModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_CTRL = 20
VALID = {f.name for f in dataclasses.fields(ModelConfig)}


def load(path: Path, dat):
    d = np.load(path, allow_pickle=True)
    raw = json.loads(str(d["config/model"]))
    cfg = ModelConfig(**{k: v for k, v in raw.items() if k in VALID})
    cfg.d_in, cfg.d_out = dat.d_in, dat.P
    m = ModAddModel(cfg).to(DEV).eval()
    sd = {}
    for k in d.files:
        if k.startswith("param/"):
            sd[k[6:]] = torch.as_tensor(d[k])
        if k.startswith("buffer/"):
            sd[k[7:]] = torch.as_tensor(d[k])
    miss, unexp = m.load_state_dict(sd, strict=False)
    if miss or unexp:
        print("  !! state_dict mismatch", miss, unexp)
    r = m.layer.router
    r._orig = r.forward

    def fwd(x, _r=r):
        g1, g2, p1, p2 = _r._orig(x)
        if _r.m1 is not None:
            g1 = g1 * _r.m1
        if _r.m2 is not None:
            g2 = g2 * _r.m2
        return g1, g2, p1, p2

    r.forward = fwd
    r.m1 = r.m2 = None
    return m, r, json.loads(str(d["config/summary"]))


@torch.no_grad()
def accuracy(m, x, y, t, chunk: int = 4096):
    """Accuracy on each subtask separately."""
    p = torch.cat([m(x[i:i + chunk]).argmax(-1) for i in range(0, len(x), chunk)])
    ok = p == y
    return ok[t == 0].float().mean().item(), ok[t == 1].float().mean().item()


@torch.no_grad()
def route_scores(m, x, t, chunk: int = 4096):
    """Mean routing weight of every factor, per side and per subtask."""
    H, n = m.cfg.n_heads, m.cfg.n_keys
    s = torch.zeros(2, 2, H, n, device=DEV)
    c = torch.zeros(2, device=DEV)
    for i in range(0, len(x), chunk):
        m(x[i:i + chunk])
        g1, g2 = m.routing()
        tt = t[i:i + chunk]
        for k in (0, 1):
            sel = tt == k
            if sel.any():
                s[0, k] += g1[sel].sum(0)
                s[1, k] += g2[sel].sum(0)
                c[k] += sel.sum()
    return (s / c[None, :, None, None]).cpu()


def main(runs_dir: str = "runs_mixed", P: int = 113, train_frac: float = 0.8) -> None:
    rng = np.random.default_rng(0)
    dat = make_data(P=P, task="mixed", train_frac=train_frac, seed=0).to(DEV)
    x, y, t = dat.x[dat.test_idx], dat.y[dat.test_idx], dat.task[dat.test_idx]

    print(f'{"arch":11s} {"N":>5s} | {"side1":>6s} {"side2":>6s} | '
          f'{"add":>7s} {"mul":>7s} {"select":>7s} | '
          f'{"r.add":>7s} {"r.mul":>7s} {"r.sel":>7s}')
    print("-" * 92)

    for f in sorted(Path(runs_dir).glob("*.npz")):
        m, r, summ = load(f, dat)
        if summ["final_test_acc"] < 0.90:
            print(f'{summ["arch"]:11s} {summ["n_experts"]:5d} | did not generalise')
            continue
        a0, m0 = accuracy(m, x, y, t)
        s = route_scores(m, x, t)
        flag = s[:, 0] > 2.0 * s[:, 1]            # specialised to addition
        k1, k2 = int(flag[0].sum()), int(flag[1].sum())
        if k1 + k2 == 0:
            print(f'{summ["arch"]:11s} {summ["n_experts"]:5d} | '
                  f"no factor meets the 2x criterion")
            continue

        r.m1 = (~flag[0]).float().to(DEV)
        r.m2 = (~flag[1]).float().to(DEV)
        a1, m1 = accuracy(m, x, y, t)
        r.m1 = r.m2 = None

        H, n = m.cfg.n_heads, m.cfg.n_keys
        da = dm = 0.0
        for _ in range(N_CTRL):
            c1, c2 = torch.ones(H * n), torch.ones(H * n)
            c1[rng.choice(H * n, k1, replace=False)] = 0
            c2[rng.choice(H * n, k2, replace=False)] = 0
            r.m1, r.m2 = c1.view(H, n).to(DEV), c2.view(H, n).to(DEV)
            ar, mr = accuracy(m, x, y, t)
            da += a0 - ar
            dm += m0 - mr
        r.m1 = r.m2 = None
        da, dm = da / N_CTRL, dm / N_CTRL

        print(f'{summ["arch"]:11s} {summ["n_experts"]:5d} | {k1:6d} {k2:6d} | '
              f"{a0 - a1:7.3f} {m0 - m1:7.3f} {(a0 - a1) - (m0 - m1):7.3f} | "
              f"{da:7.3f} {dm:7.3f} {da - dm:7.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs_mixed")
