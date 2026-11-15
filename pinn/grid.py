"""grid.py -- evaluation of a trained solution on a (B, z) grid, and the ergodic density of its policy.

The policy at a grid node is the maximizer of the Hamiltonian at the network's exact derivative V_B (free inside,
S >= 0 at B = 0, S <= 0 at B = Bmax). The generator of (B, z) and the ergodic density are those of the replication
package (Model.assemble: upwind in the drift S; ergodic_density), so that moments are computed exactly as for the
finite-difference solution.
"""
from __future__ import annotations

import numpy as np
import torch

from pinn.core import best_node, derivs, mode_of




def eval_grid(sol, B, z, device="cuda", chunk=60000, policy_dtype=torch.float64):
    """V, q, V_B, V_z, V_zz, q_B, q_z, q_zz and the policy (S, c, d, H) on the tensor-product grid B x z (numpy
    arrays of shape (nB, nz))."""
    p = sol.p
    BB, ZZ = np.meshgrid(B, z, indexing="ij")
    Bf = torch.tensor(BB.ravel(), dtype=torch.float32, device=device)
    zf = torch.tensor(ZZ.ravel(), dtype=torch.float32, device=device)
    out = {k: [] for k in ("V", "VB", "Vz", "Vzz", "q", "qB", "qz", "qzz", "S", "c", "d", "H")}
    for s in range(0, Bf.numel(), chunk):
        b, zz = Bf[s:s + chunk], zf[s:s + chunk]
        V, VB, Vz, Vzz = derivs(sol.V, b, zz, create_graph=False)
        q, qB, qz, qzz = derivs(sol.q, b, zz, create_graph=False)
        t = lambda x: x.detach().to(policy_dtype)
        bm = t(b)
        H, S, c, d = best_node(t(VB), t(q), bm, t(zz), p, mode_of(bm, p))
        for k, v in zip(out, (V, VB, Vz, Vzz, q, qB, qz, qzz, S, c, d, H)):
            out[k].append(v.detach().double().cpu().numpy())
    return {k: np.concatenate(v).reshape(BB.shape) for k, v in out.items()}


def residuals(E, B, z, p):
    """Residuals of the stationary HJB and pricing equations at the grid (exact derivatives)."""
    BB, ZZ = np.meshgrid(B, z, indexing="ij")
    mz = -p.mu * ZZ; D = 0.5 * p.sigma ** 2
    u = -1.0 / E["c"]
    rV = p.rho * E["V"] - u - E["S"] * E["VB"] - mz * E["Vz"] - D * E["Vzz"]
    xi = p.r + p.delta - p.kappa * p.coupon * E["d"]
    rq = xi * E["q"] - (1 - E["d"]) * p.coupon - E["S"] * E["qB"] - mz * E["qz"] - D * E["qzz"]
    return rV, rq


def ergodic(model, S):
    """Ergodic density (masses on the grid) of the upwind generator of drift S (replication package)."""
    from pdmodel import ergodic_density
    A = model.assemble(S)
    return A, ergodic_density(model, A)
