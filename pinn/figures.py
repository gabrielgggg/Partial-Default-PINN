"""figures.py -- figures of the report, in the style of the replication package (ref/code/make_figures.py, whose
rcParams, palette and helpers are used; fig_ou_slices and fig_ou_contours are the package's own functions, fed with
the PINN or the FD arrays).

Usage: python -m pinn.figures RUN_DIR FIG_DIR
Reads RUN_DIR/grid_{paper,dense}.npz (pinn.evaluate) and RUN_DIR/train.json.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

from pinn.core import REF  # noqa: F401  (puts the package's code on the path)
import make_figures as mf  # noqa: E402
from pinn.evaluate import GRIDS  # noqa: E402

plt = mf.plt
OBJ = [("V", "value $V$"), ("q", "bond price $q$"), ("c", "consumption $c$"), ("d", "default intensity $d$"),
       ("S", "drift of debt $S$")]


def load(run, name):
    a = dict(np.load(os.path.join(run, f"grid_{name}.npz")))
    fddir = GRIDS[name][2]
    f = {k: np.load(os.path.join(fddir, k + ".npy")) for k in ("B", "z", "V", "q", "S", "c", "d", "g")}
    return a, f


def package_figs(arr, figdir, suffix):
    """The package's fig_ou_slices and fig_ou_contours drawn from `arr` (paper grid)."""
    ref = np.load(os.path.join(mf.DATA, "ou_main.npz"))
    data = tempfile.mkdtemp()
    np.savez(os.path.join(data, "ou_main.npz"), **{k: arr[k] for k in ("B", "z", "V", "q", "S", "c", "d", "g")},
             x=ref["x"], names=ref["names"], qf=ref["qf"], band=ref["band"], lband=ref["lband"])
    old = mf.DATA
    mf.DATA, mf.FIG = data, figdir
    for n in ("ou_slices", "ou_contours"):
        mf.ALL[n]()
        shutil.move(os.path.join(figdir, n + ".pdf"), os.path.join(figdir, f"{n}_{suffix}.pdf"))
    mf.DATA = old
    shutil.rmtree(data)


def slice_js(z, p):
    sd = p["sd"]
    return [int(np.where(z < p["zt"])[0].max())] + [int(np.argmin(np.abs(z - k * sd))) for k in (-1, 1, 2)]


def overlay(a, f, figdir, name):
    """V, q, c, d, S against debt at the four endowment levels of the paper's Figure 6: FD solid, PINN dashed
    (same color per level); last panel: marginal ergodic density of debt."""
    _, p = mf.load_ou()
    B, z = a["B"], a["z"]
    js = slice_js(z, p)
    labs = [f"${k:+d}$ s.d." for k in (-2, -1, 1, 2)]
    cols = [mf.NAVY, mf.SCARLET, mf.INK, mf.MIDGRAY]
    fig, axes = plt.subplots(2, 3, figsize=(mf.W, 0.6 * mf.W))
    for ax, (k, t) in zip(axes.ravel(), OBJ):
        for n, j in enumerate(js):
            xf, yf = (mf.with_break(B[:-1], f[k][:-1, j]) if k == "d" else (B[:-1], f[k][:-1, j]))
            xp, yp = (mf.with_break(B[:-1], a[k][:-1, j]) if k == "d" else (B[:-1], a[k][:-1, j]))
            ax.plot(xf, yf, color=cols[n], ls="-", lw=1.0, label=labs[n] if k == "q" else None)
            ax.plot(xp, yp, color=cols[n], ls=(0, (3, 2)), lw=1.3)
        ax.set_title(t)
        if k == "S":
            mf.zero_line(ax)
    ax = axes[1, 2]
    h = B[1] - B[0]
    gf, gp = f["g"] / f["g"].sum(), a["g"] / a["g"].sum()
    ax.plot(B[1:], gf.sum(1)[1:] / h, color=mf.INK, lw=1.0, label="FD")
    ax.plot(B[1:], gp.sum(1)[1:] / h, color=mf.SCARLET, ls=(0, (3, 2)), lw=1.3, label="PINN")
    ax.set_title("ergodic density of debt ($B>0$)")
    ax.set_xlim(0, 0.8)
    ax.text(0.97, 0.95, f"atom at $B=0$: FD {gf[0].sum():.3f}, PINN {gp[0].sum():.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5)
    ax.legend(loc="center right", fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("debt $B$")
    for ax in axes.ravel():
        mf.clean(ax)
    h_, l_ = axes[0, 1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, handlelength=2.4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=1.2, h_pad=1.0, rect=(0, 0.04, 1, 1))
    mf.FIG = figdir
    mf.save(fig, name)


def diffmaps(a, f, figdir, name):
    """PINN minus FD in the (B, z) plane for V, q, c, d, S, and the PINN's residual of the pricing equation."""
    B, z = a["B"], a["z"]
    fig, axes = plt.subplots(2, 3, figsize=(mf.W, 0.62 * mf.W))
    short = dict(V="$V$", q="$q$", c="$c$", d="$d$", S="$S$")
    panels = [(k, short[k] + ": PINN $-$ FD", a[k] - f[k]) for k, t in OBJ] + [("rq", "pricing residual", a["rq"])]
    for ax, (k, t, D) in zip(axes.ravel(), panels):
        v = float(np.quantile(np.abs(D), 0.995)) if k == "d" else float(np.abs(D).max())
        e = int(np.floor(np.log10(v)))
        im = ax.pcolormesh(B, z, D.T / 10.0 ** e, cmap="RdBu_r", vmin=-v / 10.0 ** e, vmax=v / 10.0 ** e,
                           shading="nearest", rasterized=True)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        ax.set_title(t + f", $\\times 10^{{{e}}}$", fontsize=8)
        mf.draw_locus(ax, f["B"], f["z"], f["S"], color=mf.INK, lw=0.8)
    for ax in axes[1]:
        ax.set_xlabel("debt $B$")
    for ax in axes[:, 0]:
        ax.set_ylabel("log endowment $z$")
    fig.tight_layout(w_pad=0.6, h_pad=1.0)
    mf.FIG = figdir
    mf.save(fig, name)


def stages(run):
    """Training history of a run and of the stages it continues (--init), chained in time."""
    T = json.load(open(os.path.join(run, "train.json")))
    init = T["args"].get("init", "")
    if not init:
        return T["hist"], np.array([r["t"] for r in T["hist"]])
    h0, t0 = stages(os.path.dirname(init))
    off = json.load(open(os.path.join(os.path.dirname(init), "train.json")))["train_seconds"]
    return h0 + T["hist"], np.concatenate([t0, off + np.array([r["t"] for r in T["hist"]])])


def training(run, figdir):
    """Residuals and mean debt during training."""
    h, t = stages(run)
    t = t / 60
    fig, axes = plt.subplots(1, 3, figsize=(mf.W, 0.3 * mf.W))
    mf.line(axes[0], t, [r["ergV"] for r in h], 0, "HJB"); mf.line(axes[0], t, [r["ergq"] for r in h], 1, "pricing")
    axes[0].set_yscale("log"); axes[0].set_title("residuals, ergodic RMS"); axes[0].legend()
    mf.line(axes[1], t, [r["dt"] for r in h], 0); axes[1].set_yscale("log"); axes[1].set_title("pseudo-time step")
    mf.line(axes[2], t, [r["EB"] for r in h], 0, "PINN")
    axes[2].axhline(h[0].get("EB_fd", np.nan), color=mf.MIDGRAY, ls=":", lw=1.0, label="FD")
    axes[2].set_title("mean debt $E[B]$"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("minutes"); mf.clean(ax)
    fig.tight_layout(w_pad=1.0)
    mf.FIG = figdir
    mf.save(fig, "training")


def main():
    run, figdir = sys.argv[1], sys.argv[2]
    os.makedirs(figdir, exist_ok=True)
    a, f = load(run, "paper")
    package_figs(a, figdir, "pinn")
    package_figs(f, figdir, "fd")
    overlay(a, f, figdir, "overlay_paper")
    diffmaps(a, f, figdir, "diff_paper")
    a2, f2 = load(run, "dense")
    overlay(a2, f2, figdir, "overlay_dense")
    diffmaps(a2, f2, figdir, "diff_dense")
    training(run, figdir)


if __name__ == "__main__":
    main()
