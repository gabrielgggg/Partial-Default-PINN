"""evaluate.py -- the trained PINN on the paper grid (201 x 75) and the dense grid (2000 x 121) against the
finite-difference (FD) solution of the C++/CUDA port on the same grids: moments (replication package's
pdmoments.combined_moments for both), errors of V, q, c, d, S, and the ergodic densities.

Usage: python -m pinn.evaluate RUN_DIR [--fd_paper DIR] [--fd_dense DIR]
Writes RUN_DIR/eval.json and RUN_DIR/grid_{paper,dense}.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from pinn.core import Solution, load_params
from pinn.grid import eval_grid, ergodic, residuals

KM = os.path.expanduser("~/KM_partial_cuda/results")
GRIDS = dict(paper=(201, 75, os.path.join(KM, "p_cpu64")), dense=(2000, 121, os.path.join(KM, "d2000_cpu64")))
OBJ = ("V", "q", "c", "d", "S")


def moments_of(model, V, q, S, c, d, g):
    """Moments of Table 2 (ergodic and simulated, 16 paths of 25,000 years) and the loss, by the replication
    package's code, for a solution on the grid of `model`."""
    from pdmodel import Result
    from pdmoments import combined_moments, loss
    A = model.assemble(S)
    res = Result(model, V, q, S, c, d, np.zeros(S.shape, np.int8), np.zeros(model.nz), np.zeros(model.nz), A, g, True, 0,
                 0.0, 0.0)
    M, info = combined_moments(res, n_paths=16, T_years=25000, steps_per_year=100, seed=1)
    return M, info, float(loss(M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "train.json")))
    tr = cfg["args"]
    out = dict(train_seconds=cfg["train_seconds"], grids={})
    for name, (nB, nz, fddir) in GRIDS.items():
        p, par, zp = load_params(nz)
        from pdmodel import Model
        model = Model(par, zp, 0.0, p.Bmax, nB)
        sol = Solution(p, tr["width"], tr["depth"], arch=tr.get("arch", "mlp"), nk=tr.get("nk", 8)).cuda()
        sol.load_state_dict(torch.load(os.path.join(a.run, "model.pt")))
        t0 = time.time()
        E = eval_grid(sol, model.B, model.z)
        torch.cuda.synchronize()
        t_eval = time.time() - t0
        _, g = ergodic(model, E["S"])
        rV, rq = residuals(E, model.B, model.z, p)
        fd = {k: np.load(os.path.join(fddir, k + ".npy")) for k in ("V", "q", "S", "c", "d", "g", "B", "z")}
        assert np.allclose(fd["B"], model.B) and np.allclose(fd["z"], model.z)
        wf = fd["g"] / fd["g"].sum(); wp = g / g.sum()
        t0 = time.time()
        M, info, L = moments_of(model, E["V"], E["q"], E["S"], E["c"], E["d"], g)
        t_mom = time.time() - t0
        Mf, infof, Lf = moments_of(model, fd["V"], fd["q"], fd["S"], fd["c"], fd["d"], fd["g"])
        err = {}
        for k in OBJ:
            e = np.abs(E[k] - fd[k])
            err[k] = dict(max=float(e.max()), max_mass=float(e[wf > 1e-10].max()), erg=float((wf * e).sum()),
                          p99_mass=float(np.quantile(e[wf > 1e-10], 0.99)))
        eg = dict(l1=float(np.abs(wp - wf).sum()), EB_pinn=float((wp.sum(1) * model.B).sum()),
                  EB_fd=float((wf.sum(1) * model.B).sum()), atom0_pinn=float(wp[0].sum()), atom0_fd=float(wf[0].sum()))
        res = dict(max_V=float(np.abs(rV).max()), max_q=float(np.abs(rq).max()),
                   erg_V=float(np.sqrt((wp * rV ** 2).sum())), erg_q=float(np.sqrt((wp * rq ** 2).sum())))
        out["grids"][name] = dict(nB=nB, nz=nz, fd=fddir, eval_seconds=t_eval, moment_seconds=t_mom,
                                  moments_pinn=M, moments_fd=Mf, loss_pinn=L, loss_fd=Lf,
                                  ergodic_pinn=info["ergodic"], ergodic_fd=infof["ergodic"],
                                  sim_sd_pinn=info["sim_sd"], sim_sd_fd=infof["sim_sd"],
                                  errors=err, density=eg, residuals=res)
        np.savez(os.path.join(a.run, f"grid_{name}.npz"), B=model.B, z=model.z, g=g, rV=rV, rq=rq,
                 **{k: E[k] for k in ("V", "q", "c", "d", "S", "VB")})
        print(f"[{name} {nB}x{nz}] loss PINN {L:.4f} FD {Lf:.4f} | EB {eg['EB_pinn']:.4f} vs {eg['EB_fd']:.4f} | "
              f"|g-gFD|_1 {eg['l1']:.3f} | residuals max {res['max_V']:.1e}/{res['max_q']:.1e} erg {res['erg_V']:.1e}/{res['erg_q']:.1e}",
              flush=True)
        for k in OBJ:
            print(f"   {k}: max {err[k]['max']:.2e}  max on ergodic set {err[k]['max_mass']:.2e}  p99 {err[k]['p99_mass']:.2e}  ergodic-weighted {err[k]['erg']:.2e}")
        for k in M:
            print(f"   {k:18s} PINN {M[k]:8.4f}   FD {Mf[k]:8.4f}   diff {M[k] - Mf[k]:+.4f}")
    json.dump(out, open(os.path.join(a.run, "eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
