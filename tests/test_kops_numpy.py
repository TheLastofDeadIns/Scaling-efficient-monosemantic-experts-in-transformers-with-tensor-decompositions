"""Checks for the k-operand data, router splits and analysis.  NumPy only.

    python tests/test_kops_numpy.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from kops_data import (decode, encode, noise_purity, operand_blocks,  # noqa: E402
                       router_index_sets, sample_split, targets)


def test_full_grid():
    P, k = 7, 3
    ops, tr, te = sample_split(P, k, train_frac=0.8, seed=0)
    assert ops.shape == (P ** k, k)
    assert len(tr) + len(te) == P ** k and len(set(tr) & set(te)) == 0
    # every grid point exactly once
    flat = sum(ops[:, j] * P ** j for j in range(k))
    assert len(np.unique(flat)) == P ** k
    x = encode(ops, P)
    assert x.shape == (P ** k, k * P) and np.all(x.sum(1) == k)
    for j in range(k):
        assert np.all(x[np.arange(len(x)), j * P + ops[:, j]] == 1)
    y = targets(ops, P)
    assert np.all(y == ops.sum(1) % P)
    # deterministic in the data seed
    ops2, tr2, _ = sample_split(P, k, train_frac=0.8, seed=0)
    assert np.array_equal(ops, ops2) and np.array_equal(tr, tr2)


def test_sampled():
    P, k = 47, 5                       # 229M points: must sample, not enumerate
    ops, tr, te = sample_split(P, k, n_train=5000, n_test=1000, seed=3)
    assert ops.shape == (6000, k) and len(tr) == 5000 and len(te) == 1000
    flat = sum(ops[:, j].astype(np.int64) * P ** j for j in range(k))
    assert len(np.unique(flat)) == 6000          # distinct, so train and test disjoint
    assert ops.min() >= 0 and ops.max() < P
    try:
        sample_split(P, k, train_frac=0.8)
        raise AssertionError("enumerating 229M points should be refused")
    except ValueError:
        pass


def test_decode_roundtrip():
    P, k = 11, 4
    flat = np.arange(P ** k)
    ops = decode(flat, P, k)
    back = sum(ops[:, j] * P ** j for j in range(k))
    assert np.array_equal(back, flat)


def test_router_sets():
    P = 13
    assert router_index_sets("natural", 2 * P, P, 2) is None      # original path
    assert router_index_sets("random", 2 * P, P, 2) is None
    a, b = router_index_sets("full", 3 * P, P, 3)
    assert np.array_equal(a, np.arange(3 * P)) and np.array_equal(b, np.arange(3 * P))
    a, b = router_index_sets("pair", 4 * P, P, 4)
    assert np.array_equal(a, np.arange(P)) and np.array_equal(b, np.arange(P, 2 * P))
    a, b = router_index_sets("natural", 3 * P, P, 3)               # ceil(3/2) = 2 operands
    assert np.array_equal(a, np.arange(2 * P)) and np.array_equal(b, np.arange(2 * P, 3 * P))
    a, b = router_index_sets("natural", 4 * P, P, 4)
    assert np.array_equal(a, np.arange(2 * P)) and np.array_equal(b, np.arange(2 * P, 4 * P))
    a, b = router_index_sets("full", 2 * P, P, 2)                  # new path also for k = 2
    assert len(a) == 2 * P
    try:
        router_index_sets("interleave", 3 * P, P, 3)
        raise AssertionError("interleave is k = 2 only")
    except NotImplementedError:
        pass


def test_operand_blocks():
    P, k = 5, 3
    blocks = operand_blocks(np.arange(k * P), P, k)
    assert sorted(blocks) == [0, 1, 2]
    assert np.array_equal(blocks[1], np.arange(P, 2 * P))
    # a side reading a permuted subset: positions follow value order
    idx = np.array([2 * P + 3, 2 * P + 0, 2 * P + 4, 2 * P + 1, 2 * P + 2, 0])
    blocks = operand_blocks(idx, P, k)
    assert list(blocks) == [2]                       # operand 0 is incomplete
    assert np.array_equal(idx[blocks[2]], 2 * P + np.arange(P))


def test_noise_purity():
    rng = np.random.default_rng(0)
    for P in (31, 47, 113):
        f = rng.normal(size=(20000, P))
        f -= f.mean(1, keepdims=True)
        p = np.abs(np.fft.rfft(f, axis=1)) ** 2
        p = p[:, 1:P // 2 + 1]
        p /= p.sum(1, keepdims=True)
        assert abs(p.max(1).mean() - noise_purity(P)) < 0.003, P


def _fake_run(path, arch, P, k, split, Gin, key1, key2, idx1=None, idx2=None):
    summary = dict(name=path.stem, arch=arch, n_keys=key1.shape[1],
                   n_experts=key1.shape[1] ** 2, seed=0, n_params=1, grok_step=500,
                   final_test_acc=1.0, final_train_acc=1.0, n_train=100)
    mcfg = dict(arch=arch, d_in=Gin.shape[0], d_out=P, n_keys=key1.shape[1],
                n_operands=k, router_split=split)
    payload = {"config/summary": np.array(json.dumps(summary)),
               "config/model": np.array(json.dumps(mcfg)),
               "config/train": np.array(json.dumps(dict(P=P, n_operands=k))),
               "hist/step": np.arange(3),
               "param/layer.Gin": Gin,
               "param/layer.router.key1": key1,
               "param/layer.router.key2": key2}
    if idx1 is not None:
        payload["buffer/layer.router.idx1"] = idx1
        payload["buffer/layer.router.idx2"] = idx2
    np.savez(path, **payload)


def test_analysis():
    from analysis import Run
    from kops_analysis import analyse

    rng = np.random.default_rng(1)
    P, k, R, H, n = 31, 3, 6, 2, 4
    v = np.arange(P)
    # input factor: every unit a pure cosine in every operand block
    Gin = np.zeros((k * P, R))
    for j in range(k):
        for r in range(R):
            Gin[j * P:(j + 1) * P, r] = np.cos(2 * np.pi * (r % 3 + 1) * v / P + r)
    with tempfile.TemporaryDirectory() as d:
        # "full" router with random keys
        idx = np.arange(k * P)
        p1 = Path(d) / "tucker_full.npz"
        _fake_run(p1, "moe_tucker", P, k, "full", Gin,
                  rng.normal(size=(H, n, k * P)), rng.normal(size=(H, n, k * P)), idx, idx)
        row = analyse(Run(p1))
        assert row["k"] == 3 and abs(row["basis_purity"] - 1.0) < 1e-6
        assert 2.5 < row["basis_eff_freqs"] < 3.5           # three frequencies used
        assert row["router_tables"] == "s1a1,s1a2,s1a3,s2a1,s2a2,s2a3"
        assert abs(row["router_purity"] - noise_purity(P)) < 0.08

        # "pair" router whose side-1 keys are pure tones in a_1
        key1 = np.cos(2 * np.pi * 5 * v / P)[None, None, :] * rng.normal(size=(H, n, 1))
        p2 = Path(d) / "tucker_pair.npz"
        _fake_run(p2, "moe_tucker", P, k, "pair", Gin, key1,
                  rng.normal(size=(H, n, P)), np.arange(P), np.arange(P, 2 * P))
        row = analyse(Run(p2))
        assert row["router_tables"] == "s1a1,s2a2"
        assert row["router_purity"] > (1.0 + noise_purity(P)) / 2 - 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
