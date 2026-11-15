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
        self.kcdf = None

    def set_density(self, g, d=None):
        """Ergodic masses g on the grid; and, from the policy d on the grid, the cells at which the default
        intensity jumps (the default threshold), sampled uniformly."""
        w = g.ravel() / g.sum()
        self.cdf = torch.tensor(np.cumsum(w), dtype=torch.float64, device=self.dev)
        if d is not None:
            jump = np.zeros(d.shape, bool)
            jB = np.abs(np.diff(d, axis=0)) > 0.05
            jz = np.abs(np.diff(d, axis=1)) > 0.05
            jump[:-1] |= jB; jump[1:] |= jB; jump[:, :-1] |= jz; jump[:, 1:] |= jz
            k = jump.ravel().astype(float)
            self.kcdf = torch.tensor(np.cumsum(k / k.sum()), dtype=torch.float64, device=self.dev) if k.sum() else None

    def __call__(self, n):
        p, dev = self.p, self.dev
        U = lambda m: torch.rand(m, device=dev)
        zU = lambda m: p.zmin + (p.zmax - p.zmin) * U(m)
        k = {key: int(round(n * v)) for key, v in self.sh.items()}
        if self.cdf is None:
            k["uniform"] += k["ergodic"]; k["ergodic"] = 0
        if self.kcdf is None:
            k["uniform"] += k["kink"]; k["kink"] = 0
        Bs, zs = [p.Bmax * U(k["uniform"])], [zU(k["uniform"])]
        Bs.append(torch.zeros(k["b0"], device=dev)); zs.append(zU(k["b0"]))
        Bs.append(torch.full((k["b1"],), p.Bmax, device=dev)); zs.append(zU(k["b1"]))
        Bs.append(0.05 * p.Bmax * U(k["layer"]) ** 2); zs.append(zU(k["layer"]))
        # the reflecting barriers and the boundary layers next to them (width 0.15 of the z range)
        ze = 0.15 * (p.zmax - p.zmin) * U(k["zedge"]) ** 2
        ze[: k["zedge"] // 4] = 0.0
        Bs.append(p.Bmax * U(k["zedge"])); zs.append(torch.where(U(k["zedge"]) < 0.5, p.zmin + ze, p.zmax - ze))
        for cdf, key in ((self.cdf, "ergodic"), (self.kcdf, "kink")):
            if not k[key]:
                continue
            idx = torch.searchsorted(cdf, torch.rand(k[key], device=dev, dtype=torch.float64))
            idx = torch.clamp(idx, max=cdf.numel() - 1)
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


def step_residuals(sol, B, z, Vk, qk, S, c, d, dt, p):
    """Residuals of the implicit step at each point, each divided by its diagonal coefficient (units of V and q)."""
    V, VB, Vz, Vzz = derivs(sol.V, B, z)
    q, qB, qz, qzz = derivs(sol.q, B, z)
    mz = -p.mu * z; D = 0.5 * p.sigma ** 2
    inv = 1.0 / dt
    aV = p.rho + inv
    rV = (aV * V - S * VB - mz * Vz - D * Vzz - (-1.0 / c) - inv * Vk) / aV
    xi = p.r + p.delta - p.kappa * p.coupon * d
    aq = xi + inv
    rq = (aq * q - S * qB - mz * qz - D * qzz - (1 - d) * p.coupon - inv * qk) / aq
    return rV, rq


def step_loss(sol, target, B, z, dt, p, grad_w=0.0, h=2e-3, ell=0.05):
    """Mean squared residuals of the implicit step. With grad_w > 0, also the squared derivative in B of the
    residuals (gradient-enhanced PINN), by central differences of step h at the points whose neighbors B -/+ h are
    inside the domain and on the same side of the default threshold; the derivative is scaled by the length ell.
    Where the drift S vanishes the residual alone does not pin V_B and q_B, its derivative in B does."""
    n = B.numel()
    if grad_w > 0:
        Bm, Bp = B - h, B + h
        B3, z3 = torch.cat([B, Bm, Bp]), torch.cat([z, z, z])
        inside = (Bm > 0) & (Bp < p.Bmax)
        B3 = torch.clamp(B3, 0.0, p.Bmax)
    else:
        B3, z3 = B, z
    Vk, qk, S, c, d = policy(target, B3, z3, p)
    rV, rq = step_residuals(sol, B3, z3, Vk, qk, S, c, d, dt, p)
    lV, lq = (rV[:n] ** 2).mean(), (rq[:n] ** 2).mean()
    if grad_w > 0:
        dd = d > 0
        ok = (inside & (dd[n:2 * n] == dd[:n]) & (dd[2 * n:] == dd[:n])).float()
        s = ell / (2 * h)
        gV = (rV[2 * n:] - rV[n:2 * n]) * s
        lV = lV + grad_w * (ok * gV ** 2).mean()
        # the pricing equation pins q itself where S = 0, so only the HJB residual gets the derivative term
    return lV, lq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--nk", type=int, default=8)
    ap.add_argument("--grad_w", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0, help="Polyak averaging of the weights; the frozen copies are taken from the average")
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
    sol = Solution(p, a.width, a.depth, arch=a.arch, nk=a.nk).to(dev)
    shares = dict(uniform=0.40, ergodic=0.30, b0=0.04, b1=0.02, layer=0.08, zedge=0.08, kink=0.08)
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
    ema = copy.deepcopy(sol) if a.ema > 0 else None
    if ema is not None:
        for prm in ema.parameters():
            prm.requires_grad_(False)
    n_decay = int(a.decay_from * a.outer)
    dt = a.dt0
    hist = []
    prev = None
    for k in range(a.outer):
        lr = a.lr if k < n_decay else a.lr * (a.lr_end / a.lr) ** ((k - n_decay) / max(1, a.outer - 1 - n_decay))
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        target = copy.deepcopy(ema if ema is not None else sol).eval()
        for prm in target.parameters():
            prm.requires_grad_(False)
        if k % a.erg_every == 0:
            E = eval_grid(target, Bg, zg)
            _, g = ergodic(model, E["S"])
            sampler.set_density(g, E["d"])
        for s in range(a.inner):
            B, z = sampler(a.batch)
            lV, lq = step_loss(sol, target, B, z, dt, p, a.grad_w)
            opt.zero_grad(); (lV + lq).backward(); opt.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, ps in zip(ema.parameters(), sol.parameters()):
                        pe.mul_(a.ema).add_(ps, alpha=1.0 - a.ema)
        if k % a.eval_every == 0 or k == a.outer - 1:
            torch.save((ema if ema is not None else sol).state_dict(), os.path.join(a.out, "model.pt"))
            E = eval_grid(ema if ema is not None else sol, Bg, zg)
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
            hist.append(dict(k=k, dt=dt, t=time.time() - t0, resV=float(np.abs(rV).max()), resq=float(np.abs(rq).max()),
                             ergV=float(np.sqrt((w * rV ** 2).sum())), ergq=float(np.sqrt((w * rq ** 2).sum())),
                             EB=float((w.sum(1) * Bg).sum()),
                             EB_fd=float(((fd["g"] / fd["g"].sum()).sum(1) * Bg).sum()) if fd is not None else None))
        dt = min(dt * a.dt_grow, a.dt_max)
    T = time.time() - t0
    torch.save((ema if ema is not None else sol).state_dict(), os.path.join(a.out, "model.pt"))
    json.dump(dict(args=vars(a), train_seconds=T, hist=hist), open(os.path.join(a.out, "train.json"), "w"), indent=1)
    say(f"done {T:.0f}s")


if __name__ == "__main__":
    main()
