"""Quick test of two ideas on viscosity snapshots:
 (1) DeepONet two-level with lower Chebyshev degree
 (2) DeepONet trunk fields as near-null-space B of SA-AMG (top-k SVD columns)."""
import os, sys, numpy as np, torch, pyamg
from bench import to_scipy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vis_deeponet"))
from vis_don_data import VisSystem, find_snapshots, read_snapshot
from vis_don_model import Smoother, coarse_basis, cycle, estimate_lmax, galerkin_coarse, galerkin_matrix, load_checkpoint
from vis_don_solver import pcg
dev = torch.device("cuda")
model, ck = load_checkpoint(os.path.join(HERE, "..", "vis_deeponet", "trained_model", "best.pt"), dev); model.eval()

def amg_wu(ml, k=5):
    A0 = ml.levels[0].A.nnz; w = 0
    for i, L in enumerate(ml.levels):
        if i == len(ml.levels) - 1: w += 2 * L.A.nnz; break
        w += k * L.A.nnz + L.P.nnz + L.R.nnz
    return w / A0

if __name__ == "__main__":
  for d in sys.argv[1:]:
      acc = {}
      for f in find_snapshots([d])[::20]:
          s = read_snapshot(f); S = VisSystem(s, dev); A = to_scipy(s, S.m); n = s["n"]; nnz = A.nnz
          with torch.no_grad():
              V = coarse_basis(model.trunk(S.xhat, S.loc, S.glob), S.nc)
              C = galerkin_coarse(V, galerkin_matrix(V, S.matvec, 1e-10))
          lmax = estimate_lmax(S)
          for deg in (1, 2, 3, 6):
              sm = Smoother(cheb_degree=deg, cheb_ratio=30.0, lmax=lmax)
              M = lambda r, sm=sm: cycle(S, r[:, None], torch.zeros_like(r)[:, None], C, sm)[:, 0]
              _, it, rel = pcg(S, M, maxit=3000)
              wu = 2 * deg + 1 + 2 * S.m * V.shape[1] / nnz
              acc.setdefault(f"DON cheb{deg}", []).append((it, wu))
          pos = s["pos"]; c = pos.mean(0)
          Btr = np.kron(np.ones((n, 1)), np.eye(2))
          rot = np.zeros((2 * n, 1)); rot[0::2, 0] = -(pos[:, 1] - c[1]); rot[1::2, 0] = pos[:, 0] - c[0]
          Vn = V.detach().cpu().numpy()
          U, sv, _ = np.linalg.svd(Vn / np.linalg.norm(Vn, axis=0), full_matrices=False)
          for name, B in [("SA rigid", np.hstack([Btr, rot])),
                          ("SA rigid+NN4", np.hstack([Btr, rot, U[:, :4]])),
                          ("SA rigid+NN8", np.hstack([Btr, rot, U[:, :8]])),
                          ("SA NN8", U[:, :8])]:
              for smn, kw in [("GS", {}), ("Jac", dict(presmoother=("jacobi", {"omega": 2/3, "iterations": 2}),
                                                       postsmoother=("jacobi", {"omega": 2/3, "iterations": 2})))]:
                  ml = pyamg.smoothed_aggregation_solver(A, B=B, max_coarse=50, **kw)
                  Mn = ml.aspreconditioner(cycle="V")
                  M = lambda r, Mn=Mn: torch.as_tensor(Mn @ r.cpu().numpy(), device=dev)
                  _, it, rel = pcg(S, M, maxit=3000)
                  acc.setdefault(f"{name} ({smn})", []).append((it if rel <= 1 else np.nan, amg_wu(ml)))
      print("==", os.path.basename(d.rstrip("/")))
      for k, v in acc.items():
          it = np.array([a for a, _ in v], float); wu = np.mean([b for _, b in v])
          print(f"  {k:<22} it={np.nanmean(it):6.1f}  WU/apply={wu:5.2f}  total={np.nanmean(it) * (1 + wu):6.0f}", flush=True)
