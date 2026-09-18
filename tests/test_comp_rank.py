"""Equivalence check for the rank-constrained Tucker composition.

The fast forward never materialises the N expert matrices.  This compares it
against an explicit sum over all n^2 composed experts, in double precision.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

from model import ModelConfig, MoETucker


@torch.no_grad()
def deviation(n: int, comp_rank: int, batch: int = 7) -> float:
    torch.manual_seed(0)
    cfg = ModelConfig(arch="moe_tucker", n_keys=n, comp_rank=comp_rank,
                      d_in=2 * 17, d_out=17, top_k=min(4, n))
    layer = MoETucker(cfg).double()
    x = torch.randn(batch, cfg.d_in, dtype=torch.float64)
    g1, g2, _, _ = layer.router(x)

    fast = layer.experts(x, g1, g2)

    core = layer.dense_core()
    wij = torch.einsum("op,pqac,dq,ia,jc->ijod", layer.Gout, core, layer.Gin,
                       layer.Ga, layer.Gb)
    gij = torch.einsum("bhi,bhj->bij", g1, g2)
    naive = torch.einsum("bij,ijod,bd->bo", gij, wij, x)

    return (fast - naive).abs().max().item()


if __name__ == "__main__":
    ok = True
    for n in (4, 8):
        for r in (0, 1, 2, 4, 8):
            d = deviation(n, r)
            tag = "dense" if r == 0 else f"rank {r}"
            flag = "OK " if d < 1e-9 else "BAD"
            ok &= d < 1e-9
            print(f"{flag} n={n:2d} {tag:8s}  max|fast - naive| = {d:.2e}")
    print("\nall equivalent" if ok else "\nMISMATCH -- do not use these runs")
