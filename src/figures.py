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
    out = [r for r in runs if abs(r.tcfg.get("train_frac", -1) - train_frac) < 1e-9]
    # keep only the main-grid configuration: ablation runs share arch and
    # n_experts with it and would otherwise be picked up by pick()
    return [r for r in out
            if r.mcfg.get("router_split", "natural") == "natural"
            and abs(float(r.mcfg.get("lambda_aux", 1e-3)) - 1e-3) < 1e-12
            and bool(r.tcfg.get("decay_router_keys", True))]


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
    """Grokking step against expert count, averaged over seeds."""
    from collections import defaultdict

    fig, ax = plt.subplots(figsize=(6, 3.8))
    fails, partial = [], []
    for a in ARCH_ORDER:
        if a == "mlp":
            continue
        by_n = defaultdict(list)
        for r in runs:
            if r.arch == a:
                by_n[r.summary["n_experts"]].append(r.summary["grok_step"])
        if not by_n:
            continue
        xs, ys, es = [], [], []
        for n in sorted(by_n):
            ok = [v for v in by_n[n] if v > 0]
            if not ok:
                fails.append((n, a))
                continue
            xs.append(n)
            ys.append(float(np.mean(ok)))
            es.append(float(np.std(ok, ddof=1)) if len(ok) > 1 else 0.0)
            if len(ok) < len(by_n[n]):
                partial.append((n, ys[-1], a))
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt="o-", color=COLORS[a], lw=1.3,
                        ms=4, capsize=3, label=ARCH_LABEL[a], zorder=3)

    mlp = [r.summary["grok_step"] for r in runs
           if r.arch == "mlp" and r.summary["grok_step"] > 0]
    if mlp:
        ax.axhline(float(np.mean(mlp)), color="k", ls="--", lw=1, label="Dense MLP")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    for n, y, a in partial:
        ax.plot([n], [y], "o", mfc="white", mec=COLORS[a], ms=7, mew=1.6, zorder=5)

    if fails:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 3.0)
        for n, a in fails:
            ax.plot([n], [hi * 1.5], "x", color=COLORS[a], ms=8, mew=2, zorder=5)

    ax.set_xlabel("number of composed experts $N$")
    ax.set_ylabel("steps to 95% test accuracy")
    ax.set_title("The expert-count effect has no single sign")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
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
    ax.set_title("Frequency budget against generalisation speed")
    ax.legend(fontsize=7)
    save(fig, out, "fig3_freqs_vs_grok")


# --------------------------------------------------------------------------- #
# fig 4: where the structure lives
# --------------------------------------------------------------------------- #

def fig_purity(runs: list[Run], out: Path, n_experts: int = 256):
    """Basis and router spectral purity against expert count."""
    from collections import defaultdict

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for a in ARCH_ORDER:
        if a == "mlp":            # no experts: nothing to place on this axis
            continue
        bas, rout = defaultdict(list), defaultdict(list)
        for r in runs:
            if r.arch != a:
                continue
            n = r.summary["n_experts"]
            b = basis_tables(r)
            if b is not None:
                bas[n].append(weighted_purity(fft_power(b[0]), b[0])[0])
            t = router_tables(r)
            if t is not None:
                rout[n].append(weighted_purity(fft_power(t[0]), t[0])[0])
        for ax, d in ((axes[0], bas), (axes[1], rout)):
            if not d:
                continue
            xs = sorted(d)
            ys = [float(np.mean(d[n])) for n in xs]
            es = [float(np.std(d[n], ddof=1)) if len(d[n]) > 1 else 0.0 for n in xs]
            ax.errorbar(xs, ys, yerr=es, fmt="o-", color=COLORS[a], lw=1.3,
                        ms=4, capsize=3, label=ARCH_LABEL[a])

    floor = random_baseline_purity()
    for ax, title in ((axes[0], "shared basis"), (axes[1], "router keys")):
        ax.axhline(floor, color="k", ls=":", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of composed experts $N$")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("norm-weighted spectral purity")
    axes[0].set_ylim(0, 1.02)
    axes[1].text(0.02, floor + 0.035, "white noise", fontsize=7, ha="left",
                 transform=axes[1].get_yaxis_transform())
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Scaling preserves the basis and empties the router", y=1.03)
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
        top[j].set_xlabel("operand $a$", fontsize=8)
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
