# Follow-up experiments

Code for the experiments after the course paper. Everything in the paper still
reproduces exactly: with default settings the router, data, run names and
archives are unchanged.

## What is new

| file | what |
|---|---|
| `src/kops_data.py` | k-operand task `(a_1 + ... + a_k) mod P`, router input splits (NumPy) |
| `src/data.py` | `make_sum_data` — the k-operand dataset as tensors |
| `src/model.py` | router splits `full` and `pair`, k ≥ 3, `freeze_router` |
| `src/train.py` | k-operand data, sampled training sets, mini-batches, frozen router |
| `src/run_kops.py` | resumable runner for the four new phases |
| `src/kops_analysis.py` | purity and frequency counts per operand, any router split |
| `src/router_features.py` | spectrum of what the router feeds the experts (old archives) |
| `tests/test_kops_numpy.py` | data, splits and analysis checks, NumPy only |
| `tests/smoke_kops.py` | one-minute end-to-end check, needs torch |

Router splits (product-key router, two sides):

* `natural` — k = 2: side 1 reads a, side 2 reads b (the paper). k ≥ 3: side 1
  reads the first ⌈k/2⌉ operands, side 2 the rest.
* `pair` — side 1 reads a₁, side 2 reads a₂; other operands reach the layer only
  through the experts.
* `full` — both sides read the whole input, as in MONET and PEER.

## Phases

| phase | runs (1 seed / 3 seeds) | question |
|---|---|---|
| `calibrate` | 7 | MLP on k = 2, 3, 4 at several training-set sizes: how much data each k needs |
| `full` | 25 / 75 | the paper's grid with a router that sees the whole input: how much of the result was the a/b split |
| `kops` | ~96 / ~290 | k = 2, 3, 4 × 4 layers × N ∈ {16, 256, 1024} × splits: what survives when the task is no longer bilinear |
| `frozen` | 6 / 18 | router keys frozen at random init: does the router need learned structure at all |

The `kops` phase uses P = 47 (23 frequencies, noise purity 0.163) and, for each
k, the smallest calibrated training-set size at which the MLP generalised.
Jobs run seed-major, so a partial run already holds one full seed.

Rough GPU time from the paper's timings: about 14 h for seed 0 of everything,
about 40 h for three seeds. Run seed 0 first.

## Running on Kaggle

`opt_einsum` must be installed (it is on Kaggle's image). Without it torch
contracts the Tucker and TT einsums left to right and those layers run about ten
times slower; `run_kops.py` prints a warning in that case.

```bash
pip install pandas opt_einsum           # if missing
python tests/test_kops_numpy.py         # seconds
python tests/smoke_kops.py              # about a minute; must print "all passed"

python -u src/run_kops.py /kaggle/working/runs_kops --seeds 0 --max-hours 11
# re-run the same command in the next session; finished runs are skipped
python -u src/run_kops.py /kaggle/working/runs_kops --seeds 1,2 --max-hours 11

python src/kops_analysis.py /kaggle/working/runs_kops   # writes kops_runs.csv, kops_summary.csv
```

`--phases` picks a subset, for example `--phases calibrate,full`.
`--dry-run` lists the jobs and marks the finished ones.

Old archives from the paper: `python src/router_features.py <dir with .npz>`
prints the spectrum of the router keys, of the routing weights and of what the
router feeds the experts (F⁽¹⁾g⁽¹⁾(a)).
