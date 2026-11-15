"""The vectorized node problem against the replication package's pdmodel.best_node on random inputs."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinn.core import load_params, best_node
from pdmodel import best_node as ref_node, Model
p, par, zp = load_params()
m = Model(par, zp, 0.0, 1.0, 201)
rng = np.random.default_rng(0)
n = 20000
B = rng.uniform(0, 1, n); B[:500] = 0.0; B[500:1000] = 1.0
z = rng.uniform(p.zmin, p.zmax, n)
q = rng.uniform(0.25, 1.0, n); q[1000:4000] = rng.uniform(0.57, 0.61, 3000)
VB = -rng.uniform(0.2, 3.0, n); VB[:200] = rng.uniform(0, 1, 200)
mode = np.where(B <= 0, 1, np.where(B >= 1, -1, 2)).astype(float)
R = np.array([ref_node(VB[i], q[i], B[i], z[i], np.exp(z[i]), m.p, int(mode[i]), 28) for i in range(n)])
for dt in (torch.float64, torch.float32):
    t = lambda x: torch.tensor(x, dtype=dt, device="cuda")
    H, S, c, d = (v.cpu().double().numpy() for v in best_node(t(VB), t(q), t(B), t(z), p, t(mode)))
    for name, a, b in (("H", H, R[:, 0]), ("S", S, R[:, 1]), ("c", c, R[:, 2]), ("d", d, R[:, 3])):
        err = np.abs(a - b)
        print(dt, name, f"max {err.max():.2e}  p99.9 {np.quantile(err, 0.999):.2e}  n>1e-3 {(err > 1e-3).sum()}")
