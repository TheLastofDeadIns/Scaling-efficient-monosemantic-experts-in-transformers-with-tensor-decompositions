"""Figures for the report.

Runs on CPU from the ``.npz`` archives and the analysis table.  Every figure is
written as a PDF into ``figures/`` so LaTeX can include it at full quality.

    python src/figures.py runs/ --out figures/

Main experiments use a 0.8 train fraction; runs at other fractions are kept for
the ablation table and excluded here by default, since mixing them on one axis
would compare models trained on different amounts of data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import (Run, basis_tables, effective_n_freqs, fft_power, load_all,
                      pooled_spectrum, random_baseline_purity, router_tables,
                      spectral_purity, weighted_purity)

ARCH_LABEL = {
    "mlp": "Dense MLP",
    "moe_full": "Unfactorised MoE",
    "moe_hd": "MONET-HD",
    "moe_vd": "MONET-VD",
    "moe_cp": "CP",
    "moe_tucker": "Tucker",
    "moe_tt": "Tensor-Train",
}
ARCH_ORDER = ["mlp", "moe_full", "moe_hd", "moe_vd", "moe_cp", "moe_tucker", "moe_tt"]
COLORS = dict(zip(ARCH_ORDER, plt.cm.tab10.colors))

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def keep(runs: list[Run], train_frac: float | None = 0.8) -> list[Run]:
    if train_frac is None:
        return runs
    return [r for r in runs if abs(r.tcfg.get("train_frac", -1) - train_frac) < 1e-9]


def pick(runs: list[Run], arch: str, n_experts: int | None = None) -> Run | None:
    cand = [r for r in runs if r.arch == arch]
    if n_experts is not None:
        cand = [r for r in cand if r.summary["n_experts"] == n_experts]
    return cand[0] if cand else None


# --------------------------------------------------------------------------- #
# fig 1: grokking curves
# --------------------------------------------------------------------------- #

def fig_grokking(runs: list[Run], out: Path, n_experts: int = 256):
    sel = [(a, pick(runs, a, None if a == "mlp" else n_experts)) for a in ARCH_ORDER]
    sel = [(a, r) for a, r in sel if r is not None]
    if not sel:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for a, r in sel:
        st = r.hist["step"]
        axes[0].plot(st, r.hist["train_acc"], color=COLORS[a], lw=1.2,
                     label=ARCH_LABEL[a])
        axes[1].plot(st, r.hist["test_acc"], color=COLORS[a], lw=1.2)
    axes[0].set_title("Train accuracy")
    axes[1].set_title("Test accuracy")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_xscale("symlog", linthresh=1000)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Memorisation is immediate; generalisation is not "
                 f"(N = {n_experts} experts)", y=1.02)
    save(fig, out, "fig1_grokking_curves")


# --------------------------------------------------------------------------- #
# fig 2: grok step vs expert count
# --------------------------------------------------------------------------- #

def fig_scaling(runs: list[Run], out: Path):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    fails = []
    for a in ARCH_ORDER:
        if a == "mlp":
            continue
        pts = sorted((r.summary["n_experts"], r.summary["grok_step"])
                     for r in runs if r.arch == a)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] if p[1] > 0 else np.nan for p in pts]
        ax.plot(xs, ys, "o-", color=COLORS[a], lw=1.3, ms=4, label=ARCH_LABEL[a])
        fails.extend((p[0], a) for p in pts if p[1] < 0)

    if fails:
        y = ax.get_ylim()[1] * 1.04
        for x, a in fails:
            ax.plot([x], [y], "x", color=COLORS[a], ms=8, mew=2)
        ax.text(0.02, 0.97, "x = no generalisation within budget",
                transform=ax.transAxes, fontsize=7, va="top")

    mlp = pick(runs, "mlp")
    if mlp and mlp.summary["grok_step"] > 0:
        ax.axhline(mlp.summary["grok_step"], color="k", ls="--", lw=1,
                   label="Dense MLP")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of composed experts $N$")
    ax.set_ylabel("steps to 95% test accuracy")
    ax.set_title("Factorisation sets the sign of the expert-count effect")
    ax.legend(fontsize=7)
    save(fig, out, "fig2_grok_vs_experts")


# --------------------------------------------------------------------------- #
# fig 3: frequency budget vs grokking speed
# --------------------------------------------------------------------------- #

def fig_freq_vs_grok(runs: list[Run], out: Path):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for a in ARCH_ORDER:
        xs, ys = [], []
        for r in runs:
            if r.arch != a or r.summary["grok_step"] < 0:
                continue
            b = basis_tables(r)
            if b is None:
                continue
            xs.append(effective_n_freqs(fft_power(b[0])))
            ys.append(r.summary["grok_step"])
        if xs:
            ax.scatter(xs, ys, color=COLORS[a], s=28, label=ARCH_LABEL[a])
    ax.set_xscale("log")
    ax.set_xlabel("effective number of frequencies in the shared basis")
    ax.set_ylabel("steps to 95% test accuracy")
    ax.set_title("Narrower frequency bases generalise sooner")
    ax.legend(fontsize=7)
    save(fig, out, "fig3_freqs_vs_grok")


# --------------------------------------------------------------------------- #
# fig 4: where the structure lives
# --------------------------------------------------------------------------- #

def fig_purity(runs: list[Run], out: Path, n_experts: int = 256):
    labels, bas, rout = [], [], []
    for a in ARCH_ORDER:
        rs = [r for r in runs if r.arch == a and r.summary["grok_step"] > 0]
        if not rs:
            continue
        at_n = [x for x in rs if x.summary["n_experts"] == n_experts]
        r = at_n[0] if at_n else max(rs, key=lambda x: x.summary["n_experts"])
        b = basis_tables(r)
        if b is None:
            continue
        labels.append(f"{ARCH_LABEL[a]}\nN={r.summary['n_experts']}")
        bas.append(weighted_purity(fft_power(b[0]), b[0])[0])
        t = router_tables(r)
        rout.append(weighted_purity(fft_power(t[0]), t[0])[0] if t else np.nan)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x - 0.2, bas, 0.4, label="shared basis", color="#3b6ea5")
    ax.bar(x + 0.2, rout, 0.4, label="router keys", color="#d1874a")
    ax.axhline(random_baseline_purity(), color="k", ls=":", lw=1,
               label="white noise")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("norm-weighted spectral purity")
    ax.set_title("Interpretable structure sits in the basis, not in the router")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.text(0.0, -0.32, "Unfactorised MoE is shown at N=64: it does not "
            "generalise at N=256.", transform=ax.transAxes, fontsize=7)
    save(fig, out, "fig4_purity_basis_vs_router")


# --------------------------------------------------------------------------- #
# fig 5: pooled spectra
# --------------------------------------------------------------------------- #

def fig_spectra(runs: list[Run], out: Path, archs=("mlp", "moe_hd", "moe_tucker")):
    sel = [(a, pick(runs, a, None if a == "mlp" else 256)) for a in archs]
    sel = [(a, r) for a, r in sel if r is not None]
    if not sel:
        return
    fig, axes = plt.subplots(1, len(sel), figsize=(3.1 * len(sel), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (a, r) in zip(axes, sel):
        b = basis_tables(r)
        p = pooled_spectrum(fft_power(b[0]))
        ax.stem(np.arange(1, len(p) + 1), p, basefmt=" ",
                markerfmt="o", linefmt="-")
        ax.set_title(f"{ARCH_LABEL[a]}\n{effective_n_freqs(fft_power(b[0])):.1f} eff. freqs",
                     fontsize=8)
        ax.set_xlabel("frequency $k$")
    axes[0].set_ylabel("pooled power")
    fig.suptitle("Frequency content of the shared basis", y=1.04)
    save(fig, out, "fig5_spectra")


# --------------------------------------------------------------------------- #
# fig 6: raw tables
# --------------------------------------------------------------------------- #

def fig_tables(runs: list[Run], out: Path, arch: str = "moe_tucker", n_experts: int = 256):
    r = pick(runs, arch, n_experts) or pick(runs, arch)
    if r is None:
        return
    b = basis_tables(r)
    t = router_tables(r)
    if b is None or t is None:
        return

    def order(M):
        pw = fft_power(M)
        idx = np.lexsort((-spectral_purity(pw), pw.argmax(-1)))
        return M[idx]

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.2),
                             gridspec_kw={"height_ratios": [3, 1]})
    top, bot = axes[0], axes[1]
    bb = order(b[0])[:64]
    tt = order(t[0].reshape(-1, t[0].shape[-1]))[:64]
    for j, (M, title) in enumerate(((bb, "shared basis"), (tt, "router keys"))):
        Z = M / np.maximum(np.abs(M).max(-1, keepdims=True), 1e-12)
        top[j].imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                      interpolation="nearest")
        top[j].set_title(f"{ARCH_LABEL[r.arch]}: {title}", fontsize=9)
        top[j].set_xticks([]); top[j].grid(False)
        p = pooled_spectrum(fft_power(M))
        bot[j].stem(np.arange(1, len(p) + 1), p, basefmt=" ", markerfmt=".")
        bot[j].set_xlabel("frequency $k$")
        bot[j].set_ylim(0, max(p.max() * 1.15, 0.05))
    top[0].set_ylabel("unit (sorted by dominant freq.)")
    bot[0].set_ylabel("pooled power")
    fig.tight_layout()
    save(fig, out, "fig6_tables")


# --------------------------------------------------------------------------- #

FIGS = (fig_grokking, fig_scaling, fig_freq_vs_grok, fig_purity, fig_spectra, fig_tables)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--train_frac", type=float, default=0.8)
    args = ap.parse_args()

    runs = keep(load_all(args.run_dir), args.train_frac)
    print(f"{len(runs)} runs at train_frac={args.train_frac}")
    out = Path(args.out)
    for fn in FIGS:
        try:
            fn(runs, out)
        except Exception as e:                       # one bad figure must not
            print(f"  !! {fn.__name__}: {type(e).__name__}: {e}")  # kill the rest


if __name__ == "__main__":
    main()
