"""Training driver for the modular-arithmetic MoE study.

Design decisions worth knowing about
------------------------------------
*   **Full-batch training.**  The whole training split is ~3.8k one-hot rows,
    so mini-batching buys nothing and adds gradient noise that muddies the
    grokking transition.  Every step sees all training data.

*   **NumPy artifacts.**  Every run writes a single ``.npz`` holding (a) the
    full metric history, (b) every model parameter as a NumPy array, and (c)
    the config as a JSON string.  Nothing downstream needs torch: the Fourier
    analysis, the ablations and the figures all re-derive routing and expert
    activations from the stored weights.  This also keeps artifacts readable
    years later, which torch checkpoints are not.

*   **Gradient clipping.**  Initial gradient norms differ by ~30x across the
    seven architectures (Tucker highest, Tensor-Train lowest).  A single
    learning rate without clipping would silently favour some architectures,
    so we clip to ``clip_norm`` and log the pre-clip norm to document it.

*   **Weight decay.**  Grokking on modular addition depends on it; 1.0 with
    AdamW is in the range used by the reference experiments.  Decay is not
    applied to biases, norms or router keys.

Usage
-----
    python src/train.py --arch moe_hd --n_keys 16 --seed 0 --out runs/
    python src/train.py --grid main --out runs/          # whole sweep
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import make_data, full_grid_inputs
from model import ModelConfig, ModAddModel, router_aux_loss


# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class TrainConfig:
    steps: int = 30000
    eval_every: int = 100
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple = (0.9, 0.98)
    clip_norm: float = 1.0
    grok_threshold: float = 0.95
    device: str = "cuda"
    task: str = "single"
    P: int = 113
    train_frac: float = 0.8
    decay_router_keys: bool = True


NO_DECAY = ("bias", "b1", "b2", "b11", "b12", "b21", "b22", "bn")
ROUTER_KEYS = ("key1", "key2")


def build_optimizer(model: ModAddModel, tcfg: TrainConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        toks = NO_DECAY if tcfg.decay_router_keys else NO_DECAY + ROUTER_KEYS
        if p.ndim <= 1 or any(tok in name for tok in toks):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": tcfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tcfg.lr, betas=tcfg.betas)


@torch.no_grad()
def evaluate(model, x, y, chunk: int = 8192):
    """Loss and accuracy over a (possibly large) set, chunked to bound memory."""
    model.eval()
    tot_loss, tot_correct, n = 0.0, 0, x.shape[0]
    for i in range(0, n, chunk):
        xb, yb = x[i:i + chunk], y[i:i + chunk]
        logits = model(xb)
        tot_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        tot_correct += (logits.argmax(-1) == yb).sum().item()
    model.train()
    return tot_loss / n, tot_correct / n


@torch.no_grad()
def routing_stats(model, x, chunk: int = 8192):
    """Expert load and routing entropy, averaged over the given inputs.

    ``dead_frac`` is the fraction of key slots that never appear in any token's
    top-k support.  ``entropy`` is the mean per-token routing entropy in nats;
    low entropy means the router commits to few experts per token, which is
    what the ambiguity loss is supposed to encourage.
    """
    if model.cfg.arch == "mlp":
        return {}
    model.eval()
    n = x.shape[0]
    used1 = used2 = None
    ent1 = ent2 = 0.0
    for i in range(0, n, chunk):
        model(x[i:i + chunk])
        g1, g2 = model.routing()
        u1 = (g1 > 0).any(dim=0).any(dim=0)
        u2 = (g2 > 0).any(dim=0).any(dim=0)
        used1 = u1 if used1 is None else (used1 | u1)
        used2 = u2 if used2 is None else (used2 | u2)
        for g, acc in ((g1, "1"), (g2, "2")):
            p = g.clamp_min(1e-12)
            e = -(g * p.log()).sum(-1).mean().item() * g.shape[0]
            if acc == "1":
                ent1 += e
            else:
                ent2 += e
    model.train()
    return {
        "dead_frac_1": 1.0 - used1.float().mean().item(),
        "dead_frac_2": 1.0 - used2.float().mean().item(),
        "route_entropy_1": ent1 / n,
        "route_entropy_2": ent2 / n,
    }


def run_name(mcfg: ModelConfig, tcfg: TrainConfig) -> str:
    return (f"{mcfg.arch}_n{mcfg.n_keys}_m{mcfg.expert_dim}_h{mcfg.n_heads}"
            f"_k{mcfg.top_k}_lam{mcfg.lambda_aux:g}_{mcfg.router_split}"
            f"_{tcfg.task}_f{tcfg.train_frac:g}_wd{tcfg.weight_decay:g}_rk{int(tcfg.decay_router_keys)}"
            f"_s{mcfg.seed}")


def train_one(mcfg: ModelConfig, tcfg: TrainConfig, out_dir: Path, verbose: bool = True):
    device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")
    data = make_data(P=tcfg.P, task=tcfg.task, train_frac=tcfg.train_frac,
                     seed=0).to(device)

    mcfg = dataclasses.replace(mcfg, d_in=data.d_in, d_out=data.d_out)
    model = ModAddModel(mcfg).to(device)
    opt = build_optimizer(model, tcfg)

    xtr, ytr, xte, yte = data.split()
    hist = {k: [] for k in ("step", "train_loss", "test_loss", "train_acc",
                            "test_acc", "aux", "grad_norm")}
    grok_step = -1
    t0 = time.time()

    for step in range(tcfg.steps + 1):
        logits = model(xtr)
        ce = F.cross_entropy(logits, ytr)
        aux = model.aux_loss()
        loss = ce + aux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.clip_norm).item()
        opt.step()

        if step % tcfg.eval_every == 0:
            tr_loss, tr_acc = evaluate(model, xtr, ytr)
            te_loss, te_acc = evaluate(model, xte, yte)
            hist["step"].append(step)
            hist["train_loss"].append(tr_loss)
            hist["test_loss"].append(te_loss)
            hist["train_acc"].append(tr_acc)
            hist["test_acc"].append(te_acc)
            hist["aux"].append(float(aux.detach()))
            hist["grad_norm"].append(gn)
            if grok_step < 0 and te_acc >= tcfg.grok_threshold:
                grok_step = step
            if verbose and step % (tcfg.eval_every * 10) == 0:
                print(f"  step {step:6d}  train {tr_acc:.3f}  test {te_acc:.3f}  "
                      f"ce {ce.item():.4f}  |g| {gn:.2e}")

    stats = routing_stats(model, data.x)
    final_tr_loss, final_tr_acc = evaluate(model, xtr, ytr)
    final_te_loss, final_te_acc = evaluate(model, xte, yte)

    payload = {f"hist/{k}": np.asarray(v) for k, v in hist.items()}
    for name, p in model.named_parameters():
        payload[f"param/{name}"] = p.detach().cpu().numpy()
    for name, buf in model.named_buffers():
        payload[f"buffer/{name}"] = buf.detach().cpu().numpy()

    summary = {
        "name": run_name(mcfg, tcfg),
        "arch": mcfg.arch,
        "n_keys": mcfg.n_keys,
        "n_experts": mcfg.n_experts(),
        "expert_dim": mcfg.expert_dim,
        "n_heads": mcfg.n_heads,
        "top_k": mcfg.top_k,
        "lambda_aux": mcfg.lambda_aux,
        "router_split": mcfg.router_split,
        "unif_mode": mcfg.unif_mode,
        "activation": mcfg.activation,
        "seed": mcfg.seed,
        "task": tcfg.task,
        "n_params": model.n_params(),
        "grok_step": grok_step,
        "final_train_acc": final_tr_acc,
        "final_test_acc": final_te_acc,
        "final_train_loss": final_tr_loss,
        "final_test_loss": final_te_loss,
        "wall_time_s": time.time() - t0,
        **stats,
    }
    payload["config/model"] = np.array(json.dumps(dataclasses.asdict(mcfg)))
    payload["config/train"] = np.array(json.dumps(dataclasses.asdict(tcfg)))
    payload["config/summary"] = np.array(json.dumps(summary))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name(mcfg, tcfg)}.npz"
    np.savez_compressed(path, **payload)

    if verbose:
        print(f"  -> {path.name}  test_acc={final_te_acc:.4f}  "
              f"grok@{grok_step}  {summary['wall_time_s']:.0f}s")
    return summary


# --------------------------------------------------------------------------- #
# sweeps
# --------------------------------------------------------------------------- #

def grid_main(seeds=(0, 1, 2)):
    """Primary sweep: architecture x expert count x seed on (a + b) mod P.

    moe_full is capped at n_keys <= 16: its cost grows as O(N) and running it
    at N = 4096 would dominate the whole compute budget.  That is the point of
    including it, and the cap is stated in the report.
    """
    cfgs = []
    for seed in seeds:
        cfgs.append((ModelConfig(arch="mlp", hidden=100, seed=seed), "single"))
        for n in (4, 8, 16, 32, 64):
            for arch in ("moe_hd", "moe_vd", "moe_cp", "moe_tucker", "moe_tt"):
                cfgs.append((ModelConfig(arch=arch, n_keys=n, seed=seed), "single"))
            if n <= 16:
                cfgs.append((ModelConfig(arch="moe_full", n_keys=n, seed=seed), "single"))
    return cfgs


def grid_control(seeds=(0, 1, 2)):
    """Positive control: two subtasks with an explicit task token."""
    cfgs = []
    for seed in seeds:
        cfgs.append((ModelConfig(arch="mlp", hidden=200, seed=seed), "mixed"))
        for n in (8, 16, 32):
            for arch in ("moe_hd", "moe_vd", "moe_tucker"):
                cfgs.append((ModelConfig(arch=arch, n_keys=n, seed=seed), "mixed"))
    return cfgs


def grid_ablation(seeds=(0,)):
    """Ablations: router split, auxiliary loss weight, uniformity mode."""
    cfgs = []
    for seed in seeds:
        for split in ("natural", "interleave", "random"):
            cfgs.append((ModelConfig(arch="moe_hd", n_keys=16, seed=seed,
                                     router_split=split), "single"))
        for lam in (0.0, 1e-4, 1e-3, 5e-3):
            cfgs.append((ModelConfig(arch="moe_hd", n_keys=16, seed=seed,
                                     lambda_aux=lam), "single"))
        for mode in ("token", "batch"):
            cfgs.append((ModelConfig(arch="moe_hd", n_keys=16, seed=seed,
                                     unif_mode=mode), "single"))
        for act in ("relu", "sqrelu"):
            cfgs.append((ModelConfig(arch="moe_hd", n_keys=16, seed=seed,
                                     activation=act), "single"))
    return cfgs


GRIDS = {"main": grid_main, "control": grid_control, "ablation": grid_ablation}


def run_grid(name: str, out_dir: Path, tcfg: TrainConfig, skip_done: bool = True):
    import pandas as pd

    cfgs = GRIDS[name]()
    rows, out_dir = [], Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (mcfg, task) in enumerate(cfgs, 1):
        tc = dataclasses.replace(tcfg, task=task)
        path = out_dir / f"{run_name(mcfg, tc)}.npz"
        print(f"[{i}/{len(cfgs)}] {path.stem}")
        if skip_done and path.exists():
            with np.load(path, allow_pickle=True) as z:
                rows.append(json.loads(str(z["config/summary"])))
            print("  (cached)")
            continue
        rows.append(train_one(mcfg, tc, out_dir, verbose=True))
        pd.DataFrame(rows).to_csv(out_dir / f"summary_{name}.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"summary_{name}.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=list(GRIDS), default=None)
    ap.add_argument("--arch", default="moe_hd")
    ap.add_argument("--n_keys", type=int, default=16)
    ap.add_argument("--expert_dim", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--lambda_aux", type=float, default=1e-3)
    ap.add_argument("--router_split", default="natural")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="single")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    tcfg = TrainConfig(steps=args.steps, lr=args.lr,
                       weight_decay=args.weight_decay, task=args.task)

    if args.grid:
        df = run_grid(args.grid, Path(args.out), tcfg)
        print(df.to_string())
        return

    mcfg = ModelConfig(arch=args.arch, n_keys=args.n_keys,
                       expert_dim=args.expert_dim, n_heads=args.n_heads,
                       top_k=args.top_k, lambda_aux=args.lambda_aux,
                       router_split=args.router_split, seed=args.seed)
    train_one(mcfg, tcfg, Path(args.out))


if __name__ == "__main__":
    main()