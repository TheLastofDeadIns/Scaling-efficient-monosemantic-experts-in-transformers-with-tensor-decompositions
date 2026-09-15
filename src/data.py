"""Datasets for modular-arithmetic interpretability experiments.

Two tasks are provided:

* ``single``  -- c = (a + b) mod P.  The classic grokking / mechanistic
  interpretability testbed (Nanda et al., 2023; "Interpreting modular addition
  in MLPs").
* ``mixed``   -- c = (a + b) mod P  or  c = (a * b) mod P, selected by an
  explicit task token appended to the input.  This is the *positive control*
  for expert specialisation: here a genuine two-way split of the input
  distribution exists, so experts that specialise by subtask should be
  detectable.  Without it, a negative specialisation result on ``single``
  cannot be distinguished from an implementation bug.

Inputs are one-hot encoded, following the reference MLP setup, so that the
first P coordinates of x depend only on a and the next P only on b.  This
matters downstream: MONET's product-key router splits x into two halves, and
with this encoding the split is exactly (a, b).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


TASKS = ("add", "mul")


@dataclass
class ModAddData:
    """Container for an encoded modular-arithmetic dataset."""

    x: torch.Tensor          # (n_samples, d_in) float32, one-hot blocks
    y: torch.Tensor          # (n_samples,) int64 labels in [0, P)
    a: torch.Tensor          # (n_samples,) int64 first operand
    b: torch.Tensor          # (n_samples,) int64 second operand
    task: torch.Tensor       # (n_samples,) int64 task id (all zeros if single)
    train_idx: torch.Tensor  # (n_train,) int64
    test_idx: torch.Tensor   # (n_test,) int64
    P: int
    n_tasks: int

    @property
    def d_in(self) -> int:
        return self.x.shape[1]

    @property
    def d_out(self) -> int:
        return self.P

    def to(self, device) -> "ModAddData":
        return ModAddData(
            x=self.x.to(device),
            y=self.y.to(device),
            a=self.a.to(device),
            b=self.b.to(device),
            task=self.task.to(device),
            train_idx=self.train_idx.to(device),
            test_idx=self.test_idx.to(device),
            P=self.P,
            n_tasks=self.n_tasks,
        )

    def split(self):
        """Return (x_train, y_train, x_test, y_test)."""
        return (
            self.x[self.train_idx],
            self.y[self.train_idx],
            self.x[self.test_idx],
            self.y[self.test_idx],
        )


def _apply_op(a: np.ndarray, b: np.ndarray, op: str, P: int) -> np.ndarray:
    if op == "add":
        return (a + b) % P
    if op == "mul":
        return (a * b) % P
    raise ValueError(f"unknown op {op!r}, expected one of {TASKS}")


def _encode(a: np.ndarray, b: np.ndarray, task: np.ndarray, P: int, n_tasks: int) -> np.ndarray:
    """One-hot encode into [onehot(a) | onehot(b) | onehot(task)]."""
    n = a.shape[0]
    d_in = 2 * P + (n_tasks if n_tasks > 1 else 0)
    x = np.zeros((n, d_in), dtype=np.float32)
    x[np.arange(n), a] = 1.0
    x[np.arange(n), P + b] = 1.0
    if n_tasks > 1:
        x[np.arange(n), 2 * P + task] = 1.0
    return x


def make_data(
    P: int = 113,
    task: str = "single",
    train_frac: float = 0.3,
    seed: int = 0,
) -> ModAddData:
    """Build a modular-arithmetic dataset.

    Parameters
    ----------
    P
        Modulus.  113 in the reference experiments (prime).
    task
        ``"single"`` for (a + b) mod P only, ``"mixed"`` for both addition and
        multiplication with a task token.
    train_frac
        Fraction of the full (a, b) grid used for training.  0.3 follows Nanda
        et al.; note that the LessWrong MLP post uses 0.8, which is too
        generous to exhibit a clear grokking transition.
    seed
        Controls the train/test split only, not model initialisation.
    """
    rng = np.random.default_rng(seed)

    if task == "single":
        ops, n_tasks = ["add"], 1
    elif task == "mixed":
        ops, n_tasks = list(TASKS), len(TASKS)
    else:
        raise ValueError(f"unknown task {task!r}")

    grid_a, grid_b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    grid_a, grid_b = grid_a.ravel(), grid_b.ravel()

    xs, ys, as_, bs_, ts_ = [], [], [], [], []
    for t, op in enumerate(ops):
        t_arr = np.full_like(grid_a, t)
        xs.append(_encode(grid_a, grid_b, t_arr, P, n_tasks))
        ys.append(_apply_op(grid_a, grid_b, op, P))
        as_.append(grid_a)
        bs_.append(grid_b)
        ts_.append(t_arr)

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    a = np.concatenate(as_, axis=0)
    b = np.concatenate(bs_, axis=0)
    t = np.concatenate(ts_, axis=0)

    # Split per task so both subtasks are represented in train and test.
    train_parts, test_parts = [], []
    for ti in range(len(ops)):
        idx = np.flatnonzero(t == ti)
        perm = rng.permutation(idx)
        cut = int(round(train_frac * perm.shape[0]))
        train_parts.append(perm[:cut])
        test_parts.append(perm[cut:])
    train_idx = np.sort(np.concatenate(train_parts))
    test_idx = np.sort(np.concatenate(test_parts))

    return ModAddData(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y).long(),
        a=torch.from_numpy(a).long(),
        b=torch.from_numpy(b).long(),
        task=torch.from_numpy(t).long(),
        train_idx=torch.from_numpy(train_idx).long(),
        test_idx=torch.from_numpy(test_idx).long(),
        P=P,
        n_tasks=n_tasks,
    )


def full_grid_inputs(P: int, task_id: int = 0, n_tasks: int = 1) -> torch.Tensor:
    """All P^2 inputs in row-major (a, b) order, for activation-map analysis.

    The returned tensor is ordered so that ``out.reshape(P, P, -1)`` indexes
    [a, b, :], which is what the Fourier analysis in ``analysis.py`` expects.
    """
    grid_a, grid_b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    grid_a, grid_b = grid_a.ravel(), grid_b.ravel()
    t = np.full_like(grid_a, task_id)
    return torch.from_numpy(_encode(grid_a, grid_b, t, P, n_tasks))
