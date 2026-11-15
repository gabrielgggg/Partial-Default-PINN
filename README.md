# Partial-Default-PINN

> **Note:** This is an experimental, automated solution written by Grok Bot. It aims at experimenting with
> physics-informed neural network (PINN) solutions of the diffusion model of partial default (Kim and Mihalache),
> benchmarked against the finite-difference upwind solution.

The official replication package is
[gabrielgggg/Partial-Default-Replication](https://github.com/gabrielgggg/Partial-Default-Replication).

The report is [`report/report.pdf`](report/report.pdf).

## Method

- Two networks, `V(B, z)` and `q(B, z)` (4 x 128 tanh), inputs `2B-1`, `2B^1.5-1` and `cos(k pi s)`, k = 1..8, so the
  reflecting (Neumann) conditions in `z` hold exactly.
- Implicit pseudo-time stepping, as in the paper's scheme, with the linear solves replaced by least-squares fits:
  frozen (EMA-averaged) copies give the policy (S, c, d) at each collocation point through the replication package's
  node problem (vectorized, fp64, exact network derivative `V_B`); the networks are then fit by Adam to the residuals of
  the implicit step. A gradient-enhanced term (B-derivative of the HJB residual) pins `V_B` on the zero-drift locus.
- Collocation points: uniform, ergodic density of the current policy, default threshold, boundary layers.
- Trained from the package's initial guess; the finite-difference (FD) solution is used only for evaluation.
- Evaluation on the paper grid (201 x 75) and the dense grid (2000 x 121); ergodic density and moments with the
  replication package's own code, for both the PINN and the FD solution.

## Usage

Requires PyTorch with CUDA and the replication package's `code/` and `data/` in `ref/` (or `KM_REF=...`), and the FD
results of the C++/CUDA port for comparison.

```
python -m pinn.train --out runs/t7 --b15 1 --outer 600 --inner 40 --batch 4096 --ema 0.99 --grad_w 1 \
    --dt0 0.05 --dt_grow 1.03 --dt_max 1.0 --decay_from 0.2 --eval_every 20      # ~28 min on an NVIDIA T1000
python -m pinn.evaluate runs/t7
python -m pinn.tables runs/t7 report/tables
python -m pinn.figures runs/t7 report/figures
python -m pytest tests                                                            # node problem vs the package
```

## Results (paper grid)

| | FD | PINN |
|---|---|---|
| mean debt / output | 0.300 | 0.304 |
| s.d. debt / output | 0.204 | 0.200 |
| mean spread | 1.98% | 1.92% |
| s.d. spread | 2.95% | 2.81% |
| frequency of partial default | 0.390 | 0.401 |
| mean partial default, in default | 0.477 | 0.465 |
| corr(spread, output) | -0.757 | -0.760 |
| corr(spread, debt) | 0.843 | 0.826 |
| loss | 0.2595 | 0.2516 |

Errors vs FD (max / ergodic-weighted): V 1.2e-2 / 3.1e-3, q 2.8e-2 / 3.8e-3, c 3.5e-2 / 2.0e-3, S 4.0e-2 / 2.4e-3,
d 2.3e-3 (weighted; pointwise maxima sit on the default threshold). Full tables and figures are in the report.
