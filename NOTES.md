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
