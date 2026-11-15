# Working notes (for resuming)

- Method: PINN, implicit pseudo-time stepping with frozen copies (policy from frozen nets, fp64 node problem),
  inner Adam steps fit the implicit step; reference FD (~/KM_partial_cuda/results/p_cpu64, d2000_cpu64) only for evaluation.
- Runs: t2 (MLP, sin-z feature, no EMA): moments within ~0.01-0.02 of FD (pd_freq +0.022), EB 0.292 vs 0.284.
  t3 (modified MLP) slower, no better. t4 (cosine z features, edge/kink sampling, lr 1e-3 for 70%): unstable at k~420.
- Diagnosis: Adam noise at constant lr drives policy noise; HJB residual barely pins V_B where S ~ 0.
- Fix under test (t5): EMA (Polyak) weights as the frozen copies + lr decay after the dt ramp + gradient-enhanced
  term (B-derivative of the residual, central differences, masked across the default threshold).
- Commands: python -m pinn.train --out runs/tN ...; python -m pinn.evaluate runs/tN; python -m pinn.figures runs/tN report/figures
- t5 (EMA 0.99 + decay from k=120 + grad on V and q): stable (oscillation fixed) but S wrong near B=0 for high z
  (state-constraint layer, S ~ -sqrt(B)); mass moved from the atom at 0 to B~0.1, EB 0.225.
- t6: + input feature B^1.5 (V_B ~ sqrt(B) near B=0), grad term on V only (q pinned by its residual where S=0),
  sampling back toward uniform.
- t6/x1-x3 collapsed (EB -> 0): bug, torch.clamp(B, min=0) in the input features zeroed V_B at B = 0 exactly
  (S = Scap there). Fixed (no clamp on the linear input). B^1.5 feature now optional (--b15).
- t7: b15=1, grad on V only, EMA 0.99, decay from k=120, shares v2. Next: evaluate, then figures/report.
- t7 DONE (1667 s, 600x40 steps): paper grid EB 0.2881 vs 0.2842; vsFD V max 1.2e-2/erg 3.1e-3, q 2.8e-2/3.8e-3,
  S erg 2.4e-3, d erg 2.3e-3; loss 0.2516 vs FD 0.2595 (same combined_moments, seed 1). Gaps: debt_y 0.304/0.300,
  pd_freq 0.401/0.390, corr_spread_debt 0.826/0.843, spread_sd 0.028/0.0295. Best config so far = t7.
- t8 (running): continuation of t7 (--init runs/t7/model.pt, own PINN, not FD): batch 8192, dt 1 -> 5 (faster pseudo-time
  convergence; t7 EB still drifting at dt=1), lr 1e-4 -> 1e-6, 300x40. Next: evaluate t8, pick best, figures, report.
- t8 DONE (1572 s): pointwise V/c/S better (V erg 1.3e-3) but moments worse (EB 0.2903, debt_y 0.307, pd_freq 0.409,
  corr_spread_y -0.778; loss 0.2715 vs FD 0.2595). Final choice: t7. Next: tables, figures, report from runs/t7.
- FINAL: report/report.pdf (8 pp) from runs/t7, README, results/t7_*.json. Copied to box /workspace/KM_partial_PINN_report.pdf.
  Remaining gap documented in report: debt +0.004, pd_freq +0.011, spreads -0.06pp, corr(spread,debt) -0.017, atom 0.080/0.058,
  from q too high just past the default threshold. Possible next: wider net / adaptive features at the threshold.
