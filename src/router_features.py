"""Где живёт структура: в ключах роутера или в факторах экспертов?

Для каждого сохранённого запуска (CP, Tucker, TT; натуральное разбиение входа)
считает три таблицы как функции операнда a (и отдельно b):

  keys   — логиты роутера, то есть таблица ключей (то, что мерилось раньше);
  g      — выход роутера после top-k softmax;
  u      — выход роутера, пропущенный через фактор экспертов:
           Tucker: u(a) = Ga^T g1(a),  CP: u(a) = Ga g1(a),  TT: u(a) = G3 x g1(a).

И для каждой — взвешенную спектральную чистоту и эффективное число частот,
теми же функциями, что и в основной работе.  Шум для чистоты ≈ 0,082.

Запуск (из корня репозитория, рядом с папкой с .npz):

    python src/router_features.py runs/

Работает на CPU, только numpy, меньше минуты на всю сетку.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analysis import (effective_n_freqs, fft_power, load_all,  # noqa: E402
                      weighted_purity)


def topk_softmax(z: np.ndarray, k: int) -> np.ndarray:
    """z: (..., n). Возвращает веса, ненулевые только на top-k, сумма = 1."""
    if k >= z.shape[-1]:
        e = np.exp(z - z.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    idx = np.argpartition(-z, k - 1, axis=-1)[..., :k]
    top = np.take_along_axis(z, idx, -1)
    w = np.exp(top - top.max(-1, keepdims=True))
    w = w / w.sum(-1, keepdims=True)
    out = np.zeros_like(z)
    np.put_along_axis(out, idx, w, -1)
    return out


def side_tables(run, side: int):
    """Таблицы (units, P) для одной стороны роутера: keys, g, u."""
    key = run.param(f"router.key{side}")          # (H, n, d_side)
    perm = run.buffers.get("layer.router.perm")
    if key is None:
        return None
    P = run.P if hasattr(run, "P") else 113
    if perm is not None and not np.array_equal(perm, np.arange(len(perm))):
        return None                               # только натуральное разбиение
    H, n, _ = key.shape
    keys = key[:, :, :P]                          # логиты при one-hot: (H, n, P)
    g = topk_softmax(np.moveaxis(keys, -1, 1), int(run.mcfg["top_k"]))  # (H, P, n)

    arch = run.arch if hasattr(run, "arch") else run.summary["arch"]
    if arch == "moe_tucker":
        F = run.param("Ga" if side == 1 else "Gb")              # (n, Ra)
        u = np.einsum("hpn,nr->hrp", g, F)                       # (H, Ra, P)
    elif arch == "moe_cp":
        F = run.param("Ga" if side == 1 else "Gb")              # (R, n)
        u = np.einsum("hpn,rn->hrp", g, F)
    elif arch == "moe_tt":
        F = run.param("G3" if side == 1 else "G4")              # (R, n, R')
        u = np.einsum("hpn,rns->hrsp", g, F).reshape(H, -1, P)
    else:
        u = None
    return {"keys": keys.reshape(-1, P),
            "g": np.moveaxis(g, 1, -1).reshape(-1, P),
            "u": None if u is None else u.reshape(-1, P)}


def describe(table: np.ndarray) -> tuple[float, float]:
    pw = fft_power(table)
    pur, _ = weighted_purity(pw, table)
    return pur, effective_n_freqs(pw)


def main(run_dir: str):
    runs = [r for r in load_all(run_dir)
            if r.summary["arch"] in ("moe_tucker", "moe_cp", "moe_tt")
            and r.summary.get("task", "single") == "single"
            and r.summary.get("router_split", "natural") == "natural"
            and abs(r.tcfg.get("train_frac", 0.8) - 0.8) < 1e-9
            and int(r.mcfg.get("comp_rank", 0) or 0) == 0
            and not r.mcfg.get("router_bn", False)
            and not r.mcfg.get("mu_nonlinear", False)]
    if not runs:
        print("не нашёл подходящих .npz в", run_dir)
        return
    print(f"{'arch':11s} {'N':>5s} {'seed':>4s} | "
          f"{'чист. ключей':>12s} {'чист. g':>8s} {'чист. u':>8s} | "
          f"{'частот u':>8s} | обобщился")
    rows = []
    for r in sorted(runs, key=lambda r: (r.summary["arch"], r.summary["n_experts"], r.summary["seed"])):
        s = r.summary
        res = []
        for side in (1, 2):
            t = side_tables(r, side)
            if t is None or t["u"] is None:
                break
            res.append((describe(t["keys"])[0], describe(t["g"])[0], *describe(t["u"])))
        if len(res) != 2:
            continue
        kp, gp, up, un = np.mean(res, axis=0)
        rows.append((s["arch"], s["n_experts"], kp, gp, up, un))
        print(f"{s['arch']:11s} {s['n_experts']:5d} {s['seed']:4d} | "
              f"{kp:12.3f} {gp:8.3f} {up:8.3f} | {un:8.1f} | "
              f"{'да' if s['grok_step'] > 0 else 'нет'}")
    print("\nСреднее по сидам (обе стороны роутера):")
    import collections
    agg = collections.defaultdict(list)
    for a, n, kp, gp, up, un in rows:
        agg[(a, n)].append((kp, gp, up, un))
    for (a, n), v in sorted(agg.items()):
        m = np.mean(v, axis=0)
        print(f"{a:11s} {n:5d} | ключи {m[0]:.3f}  g {m[1]:.3f}  u {m[2]:.3f}  частот u {m[3]:.1f}")
    print("\nКак читать: если у Tucker при большом N чистота ключей около 0,08, "
          "а чистота u высокая и частот u мало — роутер работает как хэш, "
          "а частоты хранятся в факторах Ga и Gb.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
