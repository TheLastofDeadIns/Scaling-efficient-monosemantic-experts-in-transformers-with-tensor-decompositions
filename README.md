# Scaling Efficient Monosemantic Experts in Transformers with Tensor Decompositions

Course paper, HSE Faculty of Computer Science.

Mixture-of-Experts layers are proposed as a route to interpretable models, on the
argument that scaling the expert count makes individual experts monosemantic. That
argument has only ever been evaluated on natural language, where the features a
model *ought* to represent are unknown — so specialisation claims can be
illustrated but not checked.

This repository moves the question to modular addition, `c = (a + b) mod 113`,
whose generalising solution is known exactly: the network must represent each
operand through a small set of Fourier frequencies, and there are only 56 of them.
Monosemanticity becomes measurable.

Seven layer types replace the hidden layer of a reference MLP under one protocol,
swept from 16 to 4096 composed experts with three seeds, plus ablations, two rank
sweeps and a two-subtask positive control — about 150 runs.

## Findings

1. **The sign of the expert-count effect depends on the particular factorisation.**
   Tucker and CP improve with more experts, Tensor-Train and MONET's product-key
   layers degrade, independent experts fail outright past N = 64.

2. **A shared basis can act as a sparsity prior — but not always.** Tucker uses
   1.5–1.8 effective frequencies where a dense MLP uses 38.1; CP shares a basis and
   *broadens* it while its units become purer.

3. **Scaling drains the router of algorithmic structure, not of all structure.**
   Router purity falls to the white-noise floor everywhere, and this is not weight
   decay and not disuse. Yet on a control task whose subtask is signalled in the
   input, the same routers support MONET's criterion perfectly (selectivity 0.99).
   Routing scores recover partitions the input already carries and miss the
   structure that computes the answer.

4. **Rank is not the mechanism.** Raising the Tensor-Train rank does not remove its
   degradation; freezing Tucker's rank does remove its improvement.

5. **Tucker's dense core is over-provisioned tenfold.** Rank-32 coupling instead of
   1024 costs nothing and cuts 1.12M parameters to 108K.

Full report: [`paper/`](paper/).

## Layout

| path | what |
|---|---|
| `src/model.py` | seven layers + the shared product-key router |
| `src/train.py` | training loop, grokking detection, `.npz` artifacts |
| `src/run_*.py` | runners: main grid, rank sweeps, mixed task, extra seeds |
| `src/analysis.py` | Fourier metrics, computed from archived weights |
| `src/mixed_ablation.py` | MONET's criterion, factor ablation, random control |
| `tests/` | equivalence checks for the factorised forward passes |
| `results/` | per-run summaries (CSV) |
| `figures/`, `paper/` | figures and LaTeX source |

## Reproducing

```bash
python tests/test_layers_numpy.py
python -u src/run_all.py runs/
python -u src/run_rank_sweep.py runs_rank/
python -u src/run_comp_rank.py  runs_comp/
python -u src/run_mixed.py      runs_mixed/
python src/mixed_ablation.py    runs_mixed/
python src/figures.py runs/ --out figures/
```

Every runner skips completed runs, so an interrupted session continues.

## Notes

Rearranged forward passes are verified against a naive per-expert sum to machine
precision — a wrong contraction still trains, it just computes something else.
Analysis never runs the model: under one-hot inputs the router is re-derived
analytically from stored weights. Router keys are weight-decayed, contrary to the
usual convention, because they form a lookup table of `8nP` parameters addressed
by the operand; leaving it unregularised costs MONET-HD generalisation at N = 4096
on all three seeds. Run archives are too large for the repository; `results/` holds
the summaries they reduce to.
