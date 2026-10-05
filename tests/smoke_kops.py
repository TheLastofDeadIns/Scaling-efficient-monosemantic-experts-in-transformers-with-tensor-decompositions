"""Smoke test for the follow-up code (k operands, new router splits, frozen router).

Needs torch.  Runs on CPU in about a minute -- do this before the long runs:

    python tests/smoke_kops.py
"""

import dataclasses
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from analysis import Run                                   # noqa: E402
from kops_analysis import analyse                          # noqa: E402
from model import ModAddModel, ModelConfig                 # noqa: E402
from train import TrainConfig, run_name, train_one         # noqa: E402

LAYERS = ("mlp", "moe_hd", "moe_vd", "moe_cp", "moe_tucker", "moe_tt")


def check_original_path_untouched():
    cfg = ModelConfig(arch="moe_tucker", n_keys=4)
    r = ModAddModel(cfg).layer.router
    assert hasattr(r, "perm") and not hasattr(r, "idx1"), "k=2 natural must use the old router"
    assert r.key1.shape == (4, 4, 113) and r.key2.shape == (4, 4, 113)
    old = ("moe_tucker_n4_m8_h4_k4_lam0.001_natural_single_f0.8_wd1_rk1_s0")
    assert run_name(cfg, TrainConfig()) == old, run_name(cfg, TrainConfig())
    print("ok  original router path and run names unchanged")


def check_forward_backward():
    P, k = 7, 3
    x = torch.zeros(10, k * P)
    ops = torch.randint(0, P, (10, k))
    for j in range(k):
        x[torch.arange(10), j * P + ops[:, j]] = 1.0
    for split, d1 in (("full", k * P), ("pair", P), ("natural", 2 * P)):
        for arch in LAYERS:
            cfg = ModelConfig(arch=arch, n_keys=4, top_k=2, d_in=k * P, d_out=P,
                              n_operands=k, router_split=split)
            m = ModAddModel(cfg)
            out = m(x)
            assert out.shape == (10, P), (arch, split, out.shape)
            (out.sum() + m.aux_loss()).backward()
            if arch != "mlp":
                r = m.layer.router
                assert r.key1.shape[-1] == d1, (arch, split, r.key1.shape)
                assert r.key1.grad is not None and torch.isfinite(r.key1.grad).all()
    # "pair": side 1 must ignore operands 2..k
    cfg = ModelConfig(arch="moe_tucker", n_keys=4, top_k=2, d_in=k * P, d_out=P,
                      n_operands=k, router_split="pair")
    r = ModAddModel(cfg).layer.router
    x2 = x.clone()
    x2[:, 2 * P:] = 0.0
    assert torch.equal(r(x)[0], r(x2)[0]), "pair split leaks operand 3 into the router"
    print("ok  forward/backward for every layer and split, k = 3")


def check_training(tmp: Path):
    jobs = [
        (ModelConfig(arch="moe_tucker", n_keys=4, top_k=2, router_split="full"),
         TrainConfig(P=7, n_operands=3, batch_size=64, steps=30, eval_every=10, device="cpu")),
        (ModelConfig(arch="moe_hd", n_keys=4, top_k=2, router_split="pair"),
         TrainConfig(P=11, n_operands=4, n_train=500, n_test=200, batch_size=128,
                     steps=30, eval_every=10, device="cpu")),
        (ModelConfig(arch="mlp"),
         TrainConfig(P=11, n_operands=4, n_train=500, n_test=200, batch_size=128,
                     steps=30, eval_every=10, device="cpu")),
        (ModelConfig(arch="moe_cp", n_keys=4, top_k=2, freeze_router=True),
         TrainConfig(P=7, steps=30, eval_every=10, device="cpu")),
    ]
    for mc, tc in jobs:
        s = train_one(mc, tc, tmp, verbose=False)
        path = tmp / f"{run_name(mc, tc)}.npz"
        assert path.exists(), path
        assert s["n_operands"] == tc.n_operands and s["P"] == tc.P
        row = analyse(Run(path))
        assert row["k"] == tc.n_operands, row
        if mc.arch != "mlp":
            assert "router_purity" in row and "basis_purity" in row, row

    # frozen router: saved keys equal the random initialisation
    mc, tc = jobs[-1]
    z = np.load(tmp / f"{run_name(mc, tc)}.npz", allow_pickle=True)
    fresh = ModAddModel(dataclasses.replace(mc, d_in=14, d_out=7)).layer.router
    assert np.array_equal(z["param/layer.router.key1"], fresh.key1.detach().numpy()), \
        "frozen router keys moved during training"
    print("ok  training, archives and analysis for k = 2, 3, 4, sampled data, frozen router")


def check_runner(tmp: Path):
    from run_kops import all_jobs
    jobs = list(all_jobs(("calibrate", "full", "kops", "frozen"), (0,), tmp, 47))
    names = [run_name(mc, tc) for mc, tc, _ in jobs]
    assert len(names) == len(set(names)), "two jobs share an archive name"
    print(f"ok  runner lists {len(jobs)} distinct jobs for seed 0")


if __name__ == "__main__":
    check_original_path_untouched()
    check_forward_backward()
    with tempfile.TemporaryDirectory() as d:
        check_training(Path(d))
        check_runner(Path(d))
    print("all passed")
