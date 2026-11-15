"""Quick diagnostic: slices of the PINN against the FD solution, and |difference| maps (development aid)."""
import sys, os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pinn.core import Solution, load_params
from pinn.grid import eval_grid, ergodic
from pdmodel import Model
run = sys.argv[1]; width = int(sys.argv[2]) if len(sys.argv) > 2 else 128; depth = int(sys.argv[3]) if len(sys.argv) > 3 else 4
p, par, zp = load_params(75)
m = Model(par, zp, 0.0, 1.0, 201)
import json; tr = json.load(open(os.path.join(run, "train.json")))["args"]; sol = Solution(p, tr["width"], tr["depth"], nk=tr.get("nk", 8), b15=bool(tr.get("b15", 0))).cuda(); sol.load_state_dict(torch.load(os.path.join(run, "model.pt")))
E = eval_grid(sol, m.B, m.z)
_, g = ergodic(m, E["S"])
fd = {k: np.load(os.path.expanduser(f"~/KM_partial_cuda/results/p_cpu64/{k}.npy")) for k in "V q S c d g".split()}
E["g"] = g
js = [int(np.where(m.z < p.zt)[0].max()), 18, 56, 74]
fig, ax = plt.subplots(3, 6, figsize=(24, 11))
for col, k in enumerate("V q c d S g".split()):
    for n, j in enumerate(js):
        ax[0, col].plot(m.B, fd[k][:, j], color=f"C{n}", lw=1)
        ax[0, col].plot(m.B, E[k][:, j], color=f"C{n}", ls="--", lw=1)
    ax[0, col].set_title(k + " (FD solid, PINN dashed)")
    if k == "g":
        ax[1, col].plot(m.B, fd["g"].sum(1)); ax[1, col].plot(m.B, g.sum(1), "--"); ax[1, col].set_title("marginal g(B)")
        ax[2, col].plot(m.z, fd["g"].sum(0)); ax[2, col].plot(m.z, g.sum(0), "--"); ax[2, col].set_title("marginal g(z)")
        continue
    im = ax[1, col].imshow((E[k] - fd[k]).T, origin="lower", aspect="auto", extent=[0, 1, m.z[0], m.z[-1]], cmap="RdBu_r",
                           vmin=-np.abs(E[k] - fd[k]).max(), vmax=np.abs(E[k] - fd[k]).max())
    plt.colorbar(im, ax=ax[1, col]); ax[1, col].set_title(f"{k}: PINN - FD")
    for n, j in enumerate([30, 37, 44]):
        ax[2, col].plot(m.B, fd[k][:, j], color=f"C{n}"); ax[2, col].plot(m.B, E[k][:, j], color=f"C{n}", ls="--")
fig.tight_layout(); fig.savefig(os.path.join(run, "quick.png"), dpi=60)
print("saved")
