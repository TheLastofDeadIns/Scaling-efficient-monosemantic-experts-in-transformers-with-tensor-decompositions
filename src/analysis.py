"""Fourier analysis of trained modular-arithmetic models.

Everything here runs on NumPy from the ``.npz`` archives written by
``train.py``.  No torch, no GPU: the router is re-derived analytically from the
saved key matrices, which is exact for one-hot inputs.

Why Fourier
-----------
For ``c = (a + b) mod P`` the ground-truth solution is known: a network that
generalises represents each operand through a small set of frequencies
``w = 2 pi k / P``, computes ``cos(w a + s1)`` and ``cos(w b + s2)``, and reads
out ``cos(w c + s1 + s2)``.  So "what feature does this unit represent" has an
exact answer here, unlike in a language model, and monosemanticity can be
measured rather than eyeballed.

Two objects are analysed:

* **Router tables.**  With one-hot inputs the logit of key ``i`` in head ``h``
  is literally ``key1[h, i, a]`` -- a learned lookup table indexed by the
  operand.  Its DFT says which frequency, if any, that key detects.
* **Shared expert basis.**  The bottom factor ``U`` of a MONET-style layer, or
  ``fc1`` of the dense MLP, maps the one-hot operand into the hidden space.
  Restricted to the ``a`` block it is again a function of ``a``, so the same
  DFT applies.  This is what tells us whether factorisation concentrates the
  basis on fewer frequencies than a dense layer.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

class Run:
    """One training run, loaded from disk."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        z = np.load(self.path, allow_pickle=True)
        self.summary = json.loads(str(z["config/summary"]))
        self.mcfg = json.loads(str(z["config/model"]))
        self.tcfg = json.loads(str(z["config/train"]))
        self.hist = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith("hist/")}
        self.params = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith("param/")}
        self.buffers = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith("buffer/")}

    # convenient accessors ------------------------------------------------- #
    @property
    def arch(self) -> str:
        return self.mcfg["arch"]

    @property
    def P(self) -> int:
        return self.mcfg["d_out"]

    @property
    def n_keys(self) -> int:
        return self.mcfg["n_keys"]

    def param(self, suffix: str) -> np.ndarray | None:
        """Fetch a parameter by the tail of its name, e.g. 'U' or 'router.key1'."""
        for k, v in self.params.items():
            if k == suffix or k.endswith("." + suffix):
                return v
        return None

    def __repr__(self) -> str:
        s = self.summary
        return (f"<Run {s['arch']} n={s['n_keys']} seed={s['seed']} "
                f"grok@{s['grok_step']} test={s['final_test_acc']:.3f}>")


def load_all(run_dir: str | Path, pattern: str = "*.npz") -> list[Run]:
    return [Run(p) for p in sorted(glob.glob(str(Path(run_dir) / pattern)))]


# --------------------------------------------------------------------------- #
# Fourier primitives
# --------------------------------------------------------------------------- #

def fft_power(f: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalised power spectrum over frequencies k = 1 .. (P-1)//2.

    The DC component is dropped: a constant offset carries no information about
    which residue class a unit responds to.  Each spectrum sums to one, so
    spectra of units with different scales are directly comparable.
    """
    f = np.asarray(f, dtype=np.float64)
    f = f - f.mean(axis=axis, keepdims=True)
    P = f.shape[axis]
    spec = np.fft.rfft(f, axis=axis)
    power = np.abs(spec) ** 2
    power = np.take(power, np.arange(1, P // 2 + 1), axis=axis)
    total = power.sum(axis=axis, keepdims=True)
    return power / np.maximum(total, 1e-30)


def spectral_purity(power: np.ndarray, axis: int = -1) -> np.ndarray:
    """Share of spectral mass in the single strongest frequency, in [0, 1].

    1.0 means the unit is a pure single-frequency detector -- monosemantic in
    the only sense this task admits.  A value near 2/(P-1) means a flat
    spectrum, i.e. no frequency structure at all.
    """
    return power.max(axis=axis)


def spectral_entropy(power: np.ndarray, axis: int = -1, normalise: bool = True) -> np.ndarray:
    """Shannon entropy of the normalised power spectrum, in nats.

    Lower is more monosemantic.  With ``normalise`` the value is divided by
    ``log(n_freqs)``, giving 0 for a pure tone and 1 for a flat spectrum, so it
    is comparable across different P.
    """
    p = np.clip(power, 1e-30, None)
    h = -(p * np.log(p)).sum(axis=axis)
    if normalise:
        h = h / np.log(power.shape[axis])
    return h


def dominant_freq(power: np.ndarray, axis: int = -1) -> np.ndarray:
    """Index (1-based frequency k) of the strongest component."""
    return power.argmax(axis=axis) + 1


def key_frequencies(power: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    """Frequencies carrying at least ``thresh`` of the pooled spectral mass.

    Pooling over all units first, then thresholding, follows the usual
    definition of the "key frequencies" a network has settled on.
    """
    pooled = power.reshape(-1, power.shape[-1]).mean(axis=0)
    pooled = pooled / pooled.sum()
    return np.flatnonzero(pooled >= thresh) + 1


# --------------------------------------------------------------------------- #
# router reconstruction
# --------------------------------------------------------------------------- #

def router_tables(run: Run) -> tuple[np.ndarray, np.ndarray] | None:
    """Routing logits as a function of each operand: two arrays (H, n, P).

    Exact for one-hot inputs.  ``perm`` is applied first, so this is correct for
    the interleaved and random router splits too -- except that under those
    splits a side's logit is no longer a function of a single operand, and the
    returned table is then the response to varying that operand with the other
    held at zero.  We flag that case rather than silently returning nonsense.
    """
    k1, k2 = run.param("router.key1"), run.param("router.key2")
    if k1 is None:
        return None
    perm = run.buffers.get("layer.router.perm")
    if perm is None:
        perm = next((v for k, v in run.buffers.items() if k.endswith("perm")), None)
    P, d_in = run.P, run.mcfg["d_in"]
    d1 = d_in // 2

    # one-hot basis for operand a (columns 0..P-1) and b (columns P..2P-1)
    eye_a = np.zeros((P, d_in)); eye_a[np.arange(P), np.arange(P)] = 1.0
    eye_b = np.zeros((P, d_in)); eye_b[np.arange(P), P + np.arange(P)] = 1.0
    if perm is not None:
        eye_a, eye_b = eye_a[:, perm], eye_b[:, perm]

    t1 = np.einsum("pd,hnd->hnp", eye_a[:, :d1], k1)
    t2 = np.einsum("pd,hnd->hnp", eye_b[:, d1:], k2)
    return t1, t2


def router_split_is_natural(run: Run) -> bool:
    return run.mcfg.get("router_split", "natural") == "natural"


# --------------------------------------------------------------------------- #
# shared basis reconstruction
# --------------------------------------------------------------------------- #

def basis_tables(run: Run) -> tuple[np.ndarray, np.ndarray] | None:
    """Hidden pre-activations as a function of each operand: two (units, P) arrays.

    For the dense MLP the units are neurons; for MONET-style layers they are
    the ``n_keys * expert_dim`` rows of the shared bottom factor.  Multilinear
    (mu-MoE) layers have no per-expert nonlinearity, and their input factor is
    returned instead, so the comparison stays like-for-like at the level of
    "what function of a does the layer compute first".
    """
    P = run.P
    arch = run.arch

    def split(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """W is (units, d_in); return its a-block and b-block."""
        return W[:, :P], W[:, P:2 * P]

    if arch == "mlp":
        W = run.param("fc1.weight")
        return split(W) if W is not None else None

    if arch in ("moe_hd",):
        U = run.param("U")                       # (n, m, d_in)
        return split(U.reshape(-1, U.shape[-1]))

    if arch == "moe_vd":
        U1, U2 = run.param("U1"), run.param("U2")
        W = np.concatenate([U1.reshape(-1, U1.shape[-1]),
                            U2.reshape(-1, U2.shape[-1])], axis=0)
        return split(W)

    if arch == "moe_full":
        U = run.param("U")                       # (n, n, m, d_in)
        return split(U.reshape(-1, U.shape[-1]))

    if arch == "moe_cp":
        return split(run.param("Gin"))           # (R, d_in)

    if arch == "moe_tucker":
        return split(run.param("Gin").T)         # (d_in, R) -> (R, d_in)

    if arch == "moe_tt":
        G2 = run.param("G2")                     # (R, d_in, R)
        return split(G2.transpose(0, 2, 1).reshape(-1, G2.shape[1]))

    return None


def readout_table(run: Run) -> np.ndarray | None:
    """Output weights as a function of the predicted class c: (units, P).

    Under the Fourier solution these should be sinusoids at the same
    frequencies as the input side, with phases that add.  The axis of length P
    is located rather than hard-coded, since the readout factor has a
    different rank in each architecture.
    """
    P = run.P
    if run.arch == "mlp":
        W = run.param("fc2.weight")
        return None if W is None else W.T
    for name in ("V", "Gout", "G1"):
        W = run.param(name)
        if W is None:
            continue
        axes = [i for i, s in enumerate(W.shape) if s == P]
        if not axes:
            continue
        return np.moveaxis(W, axes[0], -1).reshape(-1, P)
    return None


# --------------------------------------------------------------------------- #
# per-run report
# --------------------------------------------------------------------------- #

def analyse(run: Run, purity_thresh: float = 0.5) -> dict:
    """All Fourier statistics for one run, as a flat dict for a DataFrame row."""
    out = dict(
        run=run.summary["name"],
        arch=run.arch,
        n_keys=run.n_keys,
        n_experts=run.summary["n_experts"],
        seed=run.summary["seed"],
        n_params=run.summary["n_params"],
        grok_step=run.summary["grok_step"],
        test_acc=run.summary["final_test_acc"],
        train_frac=run.tcfg.get("train_frac"),
    )

    basis = basis_tables(run)
    if basis is not None:
        pa, pb = fft_power(basis[0]), fft_power(basis[1])
        pur = np.concatenate([spectral_purity(pa), spectral_purity(pb)])
        ent = np.concatenate([spectral_entropy(pa), spectral_entropy(pb)])
        out.update(
            basis_units=pa.shape[0] + pb.shape[0],
            basis_purity_mean=float(pur.mean()),
            basis_purity_median=float(np.median(pur)),
            basis_entropy_mean=float(ent.mean()),
            basis_mono_frac=float((pur >= purity_thresh).mean()),
            basis_key_freqs=",".join(map(str, key_frequencies(pa))),
            basis_n_key_freqs=len(key_frequencies(pa)),
        )

    tabs = router_tables(run)
    if tabs is not None:
        p1, p2 = fft_power(tabs[0]), fft_power(tabs[1])
        pur = np.concatenate([spectral_purity(p1).ravel(), spectral_purity(p2).ravel()])
        out.update(
            router_purity_mean=float(pur.mean()),
            router_purity_median=float(np.median(pur)),
            router_entropy_mean=float(np.concatenate(
                [spectral_entropy(p1).ravel(), spectral_entropy(p2).ravel()]).mean()),
            router_mono_frac=float((pur >= purity_thresh).mean()),
            router_key_freqs=",".join(map(str, key_frequencies(p1))),
            router_n_key_freqs=len(key_frequencies(p1)),
            router_natural=router_split_is_natural(run),
        )

    read = readout_table(run)
    if read is not None and read.shape[-1] == run.P:
        pr = fft_power(read)
        out["readout_purity_mean"] = float(spectral_purity(pr).mean())
        out["readout_key_freqs"] = ",".join(map(str, key_frequencies(pr)))
        if basis is not None:
            in_f = set(key_frequencies(fft_power(basis[0])).tolist())
            out_f = set(key_frequencies(pr).tolist())
            union = in_f | out_f
            out["freq_agreement"] = (len(in_f & out_f) / len(union)) if union else np.nan

    return out


def summarise(run_dir: str | Path, pattern: str = "*.npz"):
    """Analysis table over every run in a directory."""
    import pandas as pd
    rows = [analyse(r) for r in load_all(run_dir, pattern)]
    return pd.DataFrame(rows)


def random_baseline_purity(P: int = 113, units: int = 4096, seed: int = 0) -> float:
    """Mean spectral purity of white-noise units, for reference.

    Any purity claim must be read against this: a purity of 0.1 means nothing
    if random weights already score 0.1.
    """
    rng = np.random.default_rng(seed)
    return float(spectral_purity(fft_power(rng.normal(size=(units, P)))).mean())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = summarise(args.run_dir, args.pattern)
    cols = [c for c in ("arch", "n_experts", "grok_step", "test_acc",
                        "basis_purity_mean", "basis_n_key_freqs",
                        "router_purity_mean", "router_n_key_freqs",
                        "freq_agreement") if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nwhite-noise purity baseline: {random_baseline_purity():.4f}")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"written to {args.out}")
