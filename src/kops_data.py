"""k-operand modular sum and router input splits.  NumPy only.

Task
----
    c = (a_1 + a_2 + ... + a_k) mod P

with every operand one-hot encoded, so the input is
``x = [onehot(a_1) | onehot(a_2) | ... | onehot(a_k)]`` of length ``k * P``.
For k = 2 this is exactly the task of the main study.  For k >= 3 the target
is no longer bilinear in two operands, which is the point: it removes the
structural match between the task and a two-expert-mode Tucker layer.

The full grid has P^k points.  Up to ``max_full`` points it is enumerated and
split by ``train_frac``; above that, ``n_train + n_test`` distinct points are
sampled without replacement.  Either way the split depends only on
``seed`` (the data seed), never on the model seed.

Router splits
-------------
The product-key router has two sides, and each side reads a subset of input
coordinates.  ``router_index_sets`` returns those subsets:

* ``natural``  k = 2: side 1 reads a, side 2 reads b -- the main study.
               Returned as ``None`` so that model.py keeps its original code
               path and old runs stay bit-for-bit reproducible.
               k >= 3: side 1 reads the first ceil(k/2) operands, side 2 the
               rest.
* ``pair``     side 1 reads a_1, side 2 reads a_2; operands 3..k reach the
               layer only through the experts.
* ``full``     both sides read the whole input, as in MONET and PEER, where
               both halves of the product key are computed from the full hidden
               state.  This removes the "router is handed the operands" hint.
* ``interleave``, ``random``  the ablation splits of the main study, k = 2
               only, also returned as ``None`` (original code path).
"""

from __future__ import annotations

import numpy as np

LEGACY_SPLITS = ("natural", "interleave", "random")
NEW_SPLITS = ("full", "pair")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def decode(flat: np.ndarray, P: int, k: int) -> np.ndarray:
    """Flat grid index -> operands, shape (n, k).  Operand j is digit j base P."""
    flat = np.asarray(flat, dtype=np.int64)
    out = np.empty((flat.shape[0], k), dtype=np.int64)
    rest = flat.copy()
    for j in range(k):
        out[:, j] = rest % P
        rest //= P
    return out


def sample_split(P: int, k: int, train_frac: float = 0.8, n_train: int = 0,
                 n_test: int = 0, seed: int = 0, max_full: int = 2_000_000):
    """Return (operands (n, k), train_idx, test_idx) into the operand array.

    n_train = 0  -> enumerate the whole grid and split it by ``train_frac``;
                    ``n_test > 0`` then caps the test set.
    n_train > 0  -> sample n_train training points and n_test test points
                    (n_test = 0 means n_train // 4), all distinct.
    """
    total = P ** k
    rng = np.random.default_rng(seed)
    if n_train <= 0:
        if total > max_full:
            raise ValueError(f"P^k = {total} points is too many to enumerate; "
                             f"set n_train (and n_test) to sample instead")
        perm = rng.permutation(total)
        cut = int(round(train_frac * total))
        tr, te = perm[:cut], perm[cut:]
        if n_test > 0:
            te = te[:n_test]
    else:
        nte = n_test if n_test > 0 else max(1, n_train // 4)
        need = n_train + nte
        if need > total:
            raise ValueError(f"asked for {need} distinct points, grid has {total}")
        if total <= max_full:
            pick = rng.permutation(total)[:need]
        else:
            pick = rng.choice(total, size=need, replace=False)
        tr, te = pick[:n_train], pick[n_train:]
    flat = np.concatenate([tr, te])
    ops = decode(flat, P, k)
    train_idx = np.arange(tr.shape[0], dtype=np.int64)
    test_idx = np.arange(tr.shape[0], flat.shape[0], dtype=np.int64)
    return ops, train_idx, test_idx


def encode(ops: np.ndarray, P: int) -> np.ndarray:
    """One-hot blocks: (n, k) operands -> (n, k * P) float32."""
    n, k = ops.shape
    x = np.zeros((n, k * P), dtype=np.float32)
    rows = np.arange(n)
    for j in range(k):
        x[rows, j * P + ops[:, j]] = 1.0
    return x


def targets(ops: np.ndarray, P: int) -> np.ndarray:
    return ops.sum(axis=1) % P


# --------------------------------------------------------------------------- #
# router splits
# --------------------------------------------------------------------------- #

def router_index_sets(split: str, d_in: int, P: int, k: int):
    """Input coordinates read by each router side, or None for the original path.

    Returns ``(idx1, idx2)`` as int64 arrays.  ``None`` means "use the original
    perm-and-halve router of the main study", which is only valid for k = 2.
    """
    if k == 2 and split in LEGACY_SPLITS:
        return None
    if split in ("interleave", "random"):
        raise NotImplementedError(f"router_split={split!r} is defined for k = 2 only")
    if d_in < k * P:
        raise ValueError(f"d_in={d_in} is smaller than k * P = {k * P}")
    if split == "full":
        idx = np.arange(d_in, dtype=np.int64)
        return idx, idx.copy()
    if split == "pair":
        return (np.arange(0, P, dtype=np.int64),
                np.arange(P, 2 * P, dtype=np.int64))
    if split == "natural":                      # k >= 3
        h = (k + 1) // 2
        return (np.arange(0, h * P, dtype=np.int64),
                np.arange(h * P, k * P, dtype=np.int64))
    raise ValueError(f"unknown router_split {split!r}")


def operand_blocks(idx: np.ndarray, P: int, k: int) -> dict[int, np.ndarray]:
    """For a router side reading coordinates ``idx``: operand j -> columns.

    Returns, for every operand whose whole one-hot block is read by this side,
    the positions (within ``idx``) of its P coordinates in value order.  Used by
    the analysis to turn a key matrix into one lookup table per operand.
    """
    pos = {int(c): i for i, c in enumerate(idx)}
    out = {}
    for j in range(k):
        cols = [pos.get(j * P + v) for v in range(P)]
        if all(c is not None for c in cols):
            out[j] = np.asarray(cols, dtype=np.int64)
    return out


def noise_purity(P: int) -> float:
    """Mean spectral purity of a white-noise unit: H_m / m with m = (P-1)//2."""
    m = (P - 1) // 2
    return float(sum(1.0 / i for i in range(1, m + 1)) / m)
