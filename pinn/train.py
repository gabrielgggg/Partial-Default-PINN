"""train.py -- PINN solution of the stationary HJB and bond-pricing equations by implicit pseudo-time stepping.

Outer step k (pseudo-time step dt_k), with frozen copies (V^k, q^k) of the networks:
  policy    (S, c, d) = argmax of the Hamiltonian at (V^k_B, q^k), point by point
  update    train (V, q) on fresh collocation points to minimize the residuals of
              (rho + 1/dt) V - S V_B - mu(z) V_z - sigma^2/2 V_zz = u(c) + V^k/dt
              (xi(d) + 1/dt) q - S q_B - mu(z) q_z - sigma^2/2 q_zz = (1-d)(delta+lam) + q^k/dt
This is the implicit scheme of the paper with the linear solves replaced by least-squares fits of the networks.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch

from pinn.core import Solution, best_node, derivs, load_params, mode_of, F_fix
from pinn.grid import eval_grid, ergodic, residuals


def initial_guess(B, z, p):
    """The replication package's initial guess (Model.initial_guess, 'static'): V from consuming y - lam B (at least
    the consumption floor) forever, q the steady-state price under the interior default rule evaluated at q = 1."""
    y = torch.exp(z)
    c0 = torch.maximum(y - p.lam * B, p.cmf * y)
    V0 = -1.0 / c0 / p.rho
    Ff = F_fix(z, p)
    dint = torch.clamp((1 - p.kappa) * p.coupon * B / (p.g0 * p.g1 * Ff * y), min=0.0) ** (1.0 / (p.g1 - 1.0))
    dint = torch.clamp(dint, max=1.0)
    q0 = torch.clamp((1 - dint) * p.coupon / (p.r + p.delta - p.kappa * p.coupon * dint), min=0.05)
    return V0, q0


class Sampler:
    """Collocation points: uniform on the domain, on the boundaries B = 0 and B = Bmax, in the layer near B = 0, and
    from the ergodic density of the current policy (cells drawn by mass, uniform within the cell)."""

    def __init__(self, p, Bg, zg, shares, device):
        self.p, self.Bg, self.zg, self.sh, self.dev = p, Bg, zg, shares, device
        self.cdf = None

    def set_density(self, g):
        w = g.ravel() / g.sum()
        self.cdf = torch.tensor(np.cumsum(w), dtype=torch.float64, device=self.dev)

    def __call__(self, n):
        p, dev = self.p, self.dev
        U = lambda m: torch.rand(m, device=dev)
        zU = lambda m: p.zmin + (p.zmax - p.zmin) * U(m)
        k = {key: int(round(n * v)) for key, v in self.sh.items()}
        if self.cdf is None:
            k["uniform"] += k.pop("ergodic")
            k["ergodic"] = 0
        Bs, zs = [p.Bmax * U(k["uniform"])], [zU(k["uniform"])]
        Bs.append(torch.zeros(k["b0"], device=dev)); zs.append(zU(k["b0"]))
        Bs.append(torch.full((k["b1"],), p.Bmax, device=dev)); zs.append(zU(k["b1"]))
        Bs.append(0.05 * p.Bmax * U(k["layer"]) ** 2); zs.append(zU(k["layer"]))
        if k["ergodic"]:
            idx = torch.searchsorted(self.cdf, torch.rand(k["ergodic"], device=dev, dtype=torch.float64))
            idx = torch.clamp(idx, max=self.cdf.numel() - 1)
            nz = self.zg.size
            i, j = idx // nz, idx % nz
            Bt = torch.tensor(self.Bg, device=dev, dtype=torch.float32); zt = torch.tensor(self.zg, device=dev, dtype=torch.float32)
            hB, hz = float(self.Bg[1] - self.Bg[0]), float(self.zg[1] - self.zg[0])
            Bs.append(torch.clamp(Bt[i] + hB * (U(idx.numel()) - 0.5), 0.0, p.Bmax))
            zs.append(torch.clamp(zt[j] + hz * (U(idx.numel()) - 0.5), p.zmin, p.zmax))
        return torch.cat(Bs), torch.cat(zs)


def policy(target, B, z, p):
    """Policy at points from the frozen networks (fp64 node problem), and the frozen values."""
    V, VB, _, _ = derivs(target.V, B, z, second=False, create_graph=False)
    with torch.no_grad():
        q = target.q(B, z)
        t = lambda x: x.detach().double()
        Bd = t(B)
        H, S, c, d = best_node(t(VB), t(q), Bd, t(z), p, mode_of(Bd, p))
    f = lambda x: x.float()
    return V.detach(), q.detach(), f(S), f(c), f(d)


def step_loss(sol, B, z, Vk, qk, S, c, d, dt, p):
    """Mean squared residuals of the implicit step, each divided by its diagonal coefficient (units of V and q)."""
    V, VB, Vz, Vzz = derivs(sol.V, B, z)
    q, qB, qz, qzz = derivs(sol.q, B, z)
    mz = -p.mu * z; D = 0.5 * p.sigma ** 2
    inv = 1.0 / dt
    aV = p.rho + inv
    rV = (aV * V - S * VB - mz * Vz - D * Vzz - (-1.0 / c) - inv * Vk) / aV
    xi = p.r + p.delta - p.kappa * p.coupon * d
    aq = xi + inv
    rq = (aq * q - S * qB - mz * qz - D * qzz - (1 - d) * p.coupon - inv * qk) / aq
    return (rV ** 2).mean(), (rq ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--inner", type=int, default=100)
    ap.add_argument("--outer", type=int, default=300)
    ap.add_argument("--dt0", type=float, default=0.1)
    ap.add_argument("--dt_grow", type=float, default=1.05)
    ap.add_argument("--dt_max", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_end", type=float, default=1e-5)
    ap.add_argument("--decay_from", type=float, default=0.5, help="share of outer steps after which the lr decays")
    ap.add_argument("--pre", type=int, default=3000)
    ap.add_argument("--erg_every", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fd", default=os.path.expanduser("~/KM_partial_cuda/results/p_cpu64"), help="for monitoring only")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    p, par, zp = load_params(75)
    from pdmodel import Model
    model = Model(par, zp, 0.0, p.Bmax, 201)
    Bg, zg = model.B, model.z
    fd = {k: np.load(os.path.join(a.fd, k + ".npy")) for k in ("V", "q", "S", "c", "d", "g")} if a.fd else None
    sol = Solution(p, a.width, a.depth, arch=a.arch).to(dev)
    shares = dict(uniform=0.45, ergodic=0.35, b0=0.05, b1=0.03, layer=0.12)
    sampler = Sampler(p, Bg, zg, shares, dev)
    log = open(os.path.join(a.out, "log.txt"), "w")
    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    say(json.dumps(vars(a)))
    t0 = time.time()
    # fit the initial guess
    opt = torch.optim.Adam(sol.parameters(), lr=a.lr)
    for it in range(a.pre):
        B = p.Bmax * torch.rand(a.batch, device=dev); z = p.zmin + (p.zmax - p.zmin) * torch.rand(a.batch, device=dev)
        V0, q0 = initial_guess(B, z, p)
        loss = ((sol.V(B, z) - V0) ** 2).mean() + ((sol.q(B, z) - q0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    say(f"initial guess fitted: loss {loss.item():.2e}  {time.time() - t0:.1f}s")
    opt = torch.optim.Adam(sol.parameters(), lr=a.lr)
    n_decay = int(a.decay_from * a.outer)
    dt = a.dt0
    hist = []
    prev = None
    for k in range(a.outer):
        lr = a.lr if k < n_decay else a.lr * (a.lr_end / a.lr) ** ((k - n_decay) / max(1, a.outer - 1 - n_decay))
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        target = copy.deepcopy(sol).eval()
        for prm in target.parameters():
            prm.requires_grad_(False)
        if k % a.erg_every == 0:
            E = eval_grid(target, Bg, zg)
            _, g = ergodic(model, E["S"])
            sampler.set_density(g)
        for s in range(a.inner):
            B, z = sampler(a.batch)
            Vk, qk, S, c, d = policy(target, B, z, p)
            lV, lq = step_loss(sol, B, z, Vk, qk, S, c, d, dt, p)
            opt.zero_grad(); (lV + lq).backward(); opt.step()
        if k % a.eval_every == 0 or k == a.outer - 1:
            E = eval_grid(sol, Bg, zg)
            rV, rq = residuals(E, Bg, zg, p)
            _, g = ergodic(model, E["S"]); w = g / g.sum()
            dV = np.abs(E["V"] - prev["V"]).max() if prev else np.nan
            dq = np.abs(E["q"] - prev["q"]).max() if prev else np.nan
            prev = E
            msg = (f"k {k:4d} dt {dt:6.3f} lr {lr:.1e} loss {lV.item():.2e} {lq.item():.2e} | res max {np.abs(rV).max():.1e} "
                   f"{np.abs(rq).max():.1e} erg {np.sqrt((w * rV ** 2).sum()):.1e} {np.sqrt((w * rq ** 2).sum()):.1e} | "
                   f"chg {dV:.1e} {dq:.1e}")
            if fd is not None:
                wf = fd["g"] / fd["g"].sum()
                eq = np.abs(E["q"] - fd["q"]); eV = np.abs(E["V"] - fd["V"])
                Bm = (w * Bg[:, None]).sum()
                msg += (f" | vsFD V {eV.max():.1e}/{(wf * eV).sum():.1e} q {eq[:-1].max():.1e}/{(wf * eq).sum():.1e} "
                        f"S {(wf * np.abs(E['S'] - fd['S'])).sum():.1e} d {(wf * np.abs(E['d'] - fd['d'])).sum():.1e} "
                        f"EB {Bm:.4f}")
            msg += f" | {time.time() - t0:.0f}s"
            say(msg)
            hist.append(dict(k=k, dt=dt, t=time.time() - t0, resV=float(np.abs(rV).max()), resq=float(np.abs(rq).max())))
        dt = min(dt * a.dt_grow, a.dt_max)
    T = time.time() - t0
    torch.save(sol.state_dict(), os.path.join(a.out, "model.pt"))
    json.dump(dict(args=vars(a), train_seconds=T, hist=hist), open(os.path.join(a.out, "train.json"), "w"), indent=1)
    say(f"done {T:.0f}s")


if __name__ == "__main__":
    main()
