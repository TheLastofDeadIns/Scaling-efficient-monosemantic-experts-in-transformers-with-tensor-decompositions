"""Fourier metrics for the k-operand runs and the new router splits.  NumPy only.

The main-study analysis (analysis.py) reads two operand blocks and assumes the
original perm-and-halve router.  This module generalises both:

* the shared basis is split into k one-hot blocks, one per operand, and every
  block is analysed as a function of that operand;
* router keys are mapped back to operands through the stored ``idx1``/``idx2``
  buffers ("full", "pair", k >= 3) or through ``perm`` (original path).  For a
  side that reads several operands, its logit is a sum of one lookup table per
  operand (one-hot input, linear keys), so each table is analysed separately.

Purity is reported next to its white-noise reference, H_m / m with
m = (P - 1) / 2, because that reference depends on P: 0.082 at P = 113,
0.163 at P = 47, 0.221 at P = 31.

    python src/kops_analysis.py runs_kops/            # per-run CSV + summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analysis import (Run, effective_n_freqs, fft_power, load_all,  # noqa: E402
                      weighted_purity)
from kops_data import noise_purity, operand_blocks  # noqa: E402


def input_factor(run: Run) -> np.ndarray | None:
    """The layer's first map from the input, as a (units, d_in) matrix."""
    a = run.arch
    if a == "mlp":
        return run.param("fc1.weight")
    if a in ("moe_hd", "moe_full"):
        U = run.param("U")
        return U.reshape(-1, U.shape[-1])
    if a == "moe_vd":
        U1, U2 = run.param("U1"), run.param("U2")
        return np.concatenate([U1.reshape(-1, U1.shape[-1]),
                               U2.reshape(-1, U2.shape[-1])], axis=0)
    if a == "moe_cp":
        return run.param("Gin")
    if a == "moe_tucker":
        return run.param("Gin").T
    if a == "moe_tt":
        G2 = run.param("G2")
        return G2.transpose(0, 2, 1).reshape(-1, G2.shape[1])
    return None


def _k(run: Run) -> int:
    return int(run.mcfg.get("n_operands", 2))


def basis_stats(run: Run) -> dict:
    W = input_factor(run)
    if W is None:
        return {}
    P, k = run.P, _k(run)
    pur, eff = [], []
    for j in range(k):
        t = W[:, j * P:(j + 1) * P]
        pw = fft_power(t)
        pur.append(weighted_purity(pw, t)[0])
        eff.append(effective_n_freqs(pw))
    return dict(basis_purity=float(np.mean(pur)), basis_eff_freqs=float(np.mean(eff)),
                basis_purity_by_op=",".join(f"{v:.3f}" for v in pur),
                basis_eff_by_op=",".join(f"{v:.1f}" for v in eff))


def router_side_columns(run: Run, side: int) -> np.ndarray | None:
    """Input coordinates read by router side 1 or 2."""
    b = run.buffers
    idx = b.get(f"layer.router.idx{side}")
    if idx is not None:
        return np.asarray(idx)
    perm = b.get("layer.router.perm")
    if perm is None:
        return None
    d1 = run.mcfg["d_in"] // 2
    return np.asarray(perm[:d1] if side == 1 else perm[d1:])


def router_stats(run: Run) -> dict:
    P, k = run.P, _k(run)
    pur, eff, seen = [], [], []
    for side in (1, 2):
        key = run.param(f"router.key{side}")            # (H, n, d_side)
        cols = router_side_columns(run, side)
        if key is None or cols is None:
            return {}
        for j, pos in operand_blocks(cols, P, k).items():
            t = key[:, :, pos].reshape(-1, P)             # (H * n, P), function of a_j
            pw = fft_power(t)
            pur.append(weighted_purity(pw, t)[0])
            eff.append(effective_n_freqs(pw))
            seen.append(f"s{side}a{j + 1}")
    if not pur:
        return {}
    return dict(router_purity=float(np.mean(pur)), router_eff_freqs=float(np.mean(eff)),
                router_tables=",".join(seen))


def analyse(run: Run) -> dict:
    s, m = run.summary, run.mcfg
    row = dict(arch=run.arch, k=_k(run), P=run.P, n_keys=m["n_keys"],
               n_experts=s["n_experts"], router_split=m.get("router_split", "natural"),
               frozen=bool(m.get("freeze_router", False)), seed=s["seed"],
               n_train=s.get("n_train"), n_params=s["n_params"],
               grok_step=s["grok_step"], test_acc=s["final_test_acc"],
               train_acc=s["final_train_acc"], noise_purity=noise_purity(run.P))
    row.update(basis_stats(run))
    row.update(router_stats(run))
    return row


def summarise(df):
    """Mean over seeds; grok step averaged over the seeds that generalised."""
    import pandas as pd

    keys = ["k", "P", "arch", "n_experts", "router_split", "frozen", "n_train"]
    rows = []
    for g, d in df.groupby(keys, dropna=False):
        ok = d[d.grok_step > 0]
        r = dict(zip(keys, g))
        r.update(seeds=len(d), generalised=len(ok),
                 grok_mean=float(ok.grok_step.mean()) if len(ok) else np.nan,
                 grok_std=float(ok.grok_step.std()) if len(ok) > 1 else np.nan,
                 test_acc=float(d.test_acc.mean()))
        for c in ("basis_purity", "basis_eff_freqs", "router_purity", "noise_purity"):
            if c in d:
                r[c] = float(d[c].mean())
        rows.append(r)
    return pd.DataFrame(rows).sort_values(keys)


def main(run_dir: str):
    import pandas as pd

    runs = load_all(run_dir)
    if not runs:
        print("no .npz archives in", run_dir)
        return
    df = pd.DataFrame([analyse(r) for r in runs])
    out = Path(run_dir)
    df.to_csv(out / "kops_runs.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "kops_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs_kops")
