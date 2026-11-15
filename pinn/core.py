"""core.py -- the model in PyTorch: parameters, the Hamiltonian maximization, and the two networks.

The parameters, the endowment process, and the domain are those of the main calibration of the replication package
(data/ou_main.npz, read through calutil.unpack). The node problem (best_S, best_node) is the replication package's
pdmodel.best_S / best_node, vectorized over points, with the exact derivative V_B of the value network in place of the
one-sided differences of the upwind scheme.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

REF = os.environ.get("KM_REF", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ref"))
sys.path.insert(0, os.path.join(REF, "code"))

NEG = -1.0e30


@dataclass
class P:
    """Parameters of the main calibration (floats) and the domain."""
    nu: float; rho: float; delta: float; kappa: float; r: float; lam: float
    g0: float; g1: float; g2: float; zt: float; cmf: float; Scap: float; Fmin: float
    qf: float; qband: float; lband: float
    mu: float; sigma: float; zmin: float; zmax: float; Bmax: float = 1.0

    @property
    def coupon(self):
        return self.delta + self.lam


def load_params(nz=75):
    """Parameters and endowment process of data/ou_main.npz (the reported calibration), via the package's unpack."""
    from calutil import unpack
    a = np.load(os.path.join(REF, "data", "ou_main.npz"))
    par, zp = unpack(a["x"].astype(float), float(a["qf"]), nz, float(a["band"]), float(a["lband"]))
    assert par.fixed_cost_form == "level" and par.closed_d and par.nu == 2.0
    p = P(nu=par.nu, rho=par.rho, delta=par.delta, kappa=par.kappa, r=par.r, lam=par.lam, g0=par.g0, g1=par.g1,
          g2=par.g2, zt=par.ztilde, cmf=par.cmin_frac, Scap=par.S_cap, Fmin=par.F_min, qf=par.q_floor,
          qband=par.q_band, lband=par.l_band, mu=zp.mu, sigma=zp.sigma, zmin=float(zp.z[0]), zmax=float(zp.z[-1]))
    return p, par, zp


# ----------------------------------------------------------------------------- primitives

def F_fix(z, p: P):
    """Fixed-cost factor, level form: 1 - g2 (e^z - e^zt) above zt, floored at Fmin; 1 below zt."""
    v = torch.clamp(1.0 - p.g2 * (torch.exp(z) - math.exp(p.zt)), min=p.Fmin)
    return torch.where(z >= p.zt, v, torch.ones_like(z))


def issue_cap(q, p: P):
    """Underwriting standard: (capped, L). L = 0 below qf, lband w/(1-w) in the band, no cap above."""
    capped = q < p.qf + p.qband
    w = torch.clamp((q - p.qf) / p.qband, 0.0, 0.999999)
    L = torch.where(q < p.qf, torch.zeros_like(q), p.lband * w / (1.0 - w))
    return capped, L


def best_S(d, VB, q, B, y, Fz, p: P, mode):
    """max over admissible S of u(c(S, d)) + S VB for given d (pdmodel.best_S, nu = 2).
    mode: tensor with 1 (S >= 0), -1 (S <= 0), 2 (free). Returns H (NEG where infeasible), S, c."""
    ph = torch.where(d <= 0, torch.ones_like(d), (1.0 - p.g0 * torch.clamp(d, min=0.0) ** p.g1) * Fz)
    caut = ph * y - (1.0 - d) * p.coupon * B
    k = (p.delta - p.kappa * p.coupon * d) * B
    cmin = p.cmf * y
    capped, L = issue_cap(q, p)
    S_lo = torch.clamp((cmin - caut) / q - k, min=-p.Scap)
    S_hi = torch.where(capped, torch.clamp(L - k, max=p.Scap), torch.full_like(q, p.Scap))
    S_lo = torch.where(mode == 1, torch.clamp(S_lo, min=0.0), S_lo)
    S_hi = torch.where(mode == -1, torch.clamp(S_hi, max=0.0), S_hi)
    infeas = S_lo > S_hi
    neg = VB < 0
    cstar = torch.sqrt(q / torch.where(neg, -VB, torch.ones_like(VB)))
    Sstar = torch.where(neg, (cstar - caut) / q - k, S_hi)
    S = torch.minimum(torch.maximum(Sstar, S_lo), S_hi)
    c = caut + q * (S + k)
    H = -1.0 / c + S * VB
    H = torch.where(infeas | ~(c > 0), torch.full_like(H, NEG), H)
    S = torch.where(infeas, torch.zeros_like(S), S)
    c = torch.where(infeas, cmin, c)
    return H, S, c


def best_node(VB, q, B, z, p: P, mode, n_golden=28):
    """max over d in [0, 1] and S (pdmodel.best_node with closed_d): the closed-form interior intensity where it is
    admissible, otherwise a golden-section search over d, the candidate d = 1, and the clamped closed form; d = 0
    always. Returns H, S, c, d."""
    y = torch.exp(z)
    Fz = F_fix(z, p)
    zero = torch.zeros_like(q)
    H0, S0, c0 = best_S(zero, VB, q, B, y, Fz, p, mode)
    # closed-form interior intensity: dc/dd = 0 given S
    rhs = (1.0 - p.kappa * q) * p.coupon * B / (p.g0 * p.g1 * Fz * y)
    di = torch.clamp(rhs, min=0.0) ** (1.0 / (p.g1 - 1.0))
    dcl = torch.clamp(di, max=1.0)
    Hc, Sc, cc = best_S(dcl, VB, q, B, y, Fz, p, mode)
    capped, L = issue_cap(q, p)
    kd = (p.delta - p.kappa * p.coupon * dcl) * B
    closed_ok = (B > 0) & (rhs > 0) & (di < 1.0) & (Hc > -1e29) & (cc > p.cmf * y * (1.0 + 1e-6)) \
        & ((~capped) | (Sc < L - kd - 1e-9))
    # golden section on (1e-9, 1] (used where the closed form does not apply)
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a = torch.full_like(q, 1e-9); b = torch.ones_like(q)
    x1 = b - invphi * (b - a); x2 = a + invphi * (b - a)
    f1 = best_S(x1, VB, q, B, y, Fz, p, mode)[0]; f2 = best_S(x2, VB, q, B, y, Fz, p, mode)[0]
    for _ in range(n_golden):
        left = f1 <= f2
        a = torch.where(left, x1, a); b = torch.where(left, b, x2)
        nx1 = torch.where(left, x2, b - invphi * (b - a)); nx2 = torch.where(left, a + invphi * (b - a), x1)
        fn = best_S(torch.where(left, nx2, nx1), VB, q, B, y, Fz, p, mode)[0]
        f1, f2 = torch.where(left, f2, fn), torch.where(left, fn, f1)
        x1, x2 = nx1, nx2
    dg = 0.5 * (a + b)
    Hb, Sb, cb, db = H0, S0, c0, zero
    for dd in (dg, torch.ones_like(q), dcl):
        H, S, c = best_S(dd, VB, q, B, y, Fz, p, mode)
        better = H > Hb
        Hb, Sb, cb, db = (torch.where(better, H, Hb), torch.where(better, S, Sb), torch.where(better, c, cb),
                          torch.where(better, dd, db))
    # closed form where admissible (compared with d = 0 only)
    use_c = closed_ok & (Hc > H0)
    use_0 = closed_ok & ~(Hc > H0)
    Hb = torch.where(use_c, Hc, torch.where(use_0, H0, Hb))
    Sb = torch.where(use_c, Sc, torch.where(use_0, S0, Sb))
    cb = torch.where(use_c, cc, torch.where(use_0, c0, cb))
    db = torch.where(use_c, dcl, torch.where(use_0, zero, db))
    return Hb, Sb, cb, db


def mode_of(B, p: P):
    """State constraint: S >= 0 at B = 0, S <= 0 at B = Bmax, free inside."""
    m = torch.full_like(B, 2.0)
    m = torch.where(B <= 0.0, torch.ones_like(B), m)
    return torch.where(B >= p.Bmax, -torch.ones_like(B), m)


# ----------------------------------------------------------------------------- networks

class MLP(nn.Module):
    def __init__(self, width=128, depth=4, n_in=2):
        super().__init__()
        layers, k = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(k, width), nn.Tanh()]
            k = width
        layers.append(nn.Linear(k, 1))
        self.f = nn.Sequential(*layers)
        for m in self.f:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.f(x).squeeze(-1)


class ModMLP(nn.Module):
    """Modified MLP (Wang, Teng, and Perdikaris 2021): two encoders of the input gate every hidden layer,
    h <- h * U + (1 - h) * W."""

    def __init__(self, width=128, depth=4, n_in=2):
        super().__init__()
        self.U, self.W = nn.Linear(n_in, width), nn.Linear(n_in, width)
        self.inp = nn.Linear(n_in, width)
        self.hid = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        u, w = torch.tanh(self.U(x)), torch.tanh(self.W(x))
        h = torch.tanh(self.inp(x))
        h = h * u + (1 - h) * w
        for l in self.hid:
            h = torch.tanh(l(h))
            h = h * u + (1 - h) * w
        return self.out(h).squeeze(-1)


class Solution(nn.Module):
    """V(B, z) and q(B, z). Inputs: 2B/Bmax - 1 and sin(pi/2 z/zmax), whose derivative in z vanishes at the
    reflecting barriers, so the Neumann conditions V_z = q_z = 0 there hold exactly. Outputs: V = Vc + Vs N_V,
    q = sigmoid(N_q) in (0, 1)."""

    def __init__(self, p: P, width=128, depth=4, Vc=-20.0, Vs=2.0, arch="mlp"):
        super().__init__()
        self.p = p
        Net = ModMLP if arch == "mod" else MLP
        self.netV = Net(width, depth)
        self.netq = Net(width, depth)
        self.Vc, self.Vs = Vc, Vs

    def feats(self, B, z):
        zb = self.p.zmax
        return torch.stack([2.0 * B / self.p.Bmax - 1.0, torch.sin(0.5 * math.pi * z / zb)], -1)

    def V(self, B, z):
        return self.Vc + self.Vs * self.netV(self.feats(B, z))

    def q(self, B, z):
        return torch.sigmoid(self.netq(self.feats(B, z)))


def derivs(f, B, z, second=True, create_graph=True):
    """f(B, z) with f_B, f_z and f_zz by automatic differentiation."""
    B = B.detach().requires_grad_(True); z = z.detach().requires_grad_(True)
    v = f(B, z)
    vB, vz = torch.autograd.grad(v.sum(), (B, z), create_graph=create_graph or second)
    vzz = torch.autograd.grad(vz.sum(), z, create_graph=create_graph)[0] if second else None
    return v, vB, vz, vzz
