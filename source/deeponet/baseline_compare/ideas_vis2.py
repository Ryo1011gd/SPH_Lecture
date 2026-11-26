import os, sys, numpy as np, torch, pyamg, scipy.sparse as sp
from bench import to_scipy
from ideas_vis import model, amg_wu, dev  # noqa (re-uses loaded model)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vis_deeponet"))
from vis_don_data import VisSystem, find_snapshots, read_snapshot
from vis_don_model import coarse_basis
from vis_don_solver import pcg
for d in sys.argv[1:]:
    acc = {}
    for f in find_snapshots([d])[::20]:
        s = read_snapshot(f); S = VisSystem(s, dev); A = to_scipy(s, S.m); n = s["n"]
        with torch.no_grad(): V = coarse_basis(model.trunk(S.xhat, S.loc, S.glob), S.nc).cpu().numpy()
        U = np.linalg.svd(V / np.linalg.norm(V, axis=0), full_matrices=False)[0]
        pos = s["pos"]; c = pos.mean(0)
        Btr = np.kron(np.ones((n, 1)), np.eye(2)); rot = np.zeros((2 * n, 1)); rot[0::2, 0] = -(pos[:, 1] - c[1]); rot[1::2, 0] = pos[:, 0] - c[0]
        for name, B in [("SA rigid", np.hstack([Btr, rot])), ("SA rigid+NN8", np.hstack([Btr, rot, U[:, :8]])), ("SA rigid+NN16", np.hstack([Btr, rot, U[:, :16]]))]:
            for deg in (2, 3):
                sm = ("chebyshev", {"degree": deg, "iterations": 1})
                ds = 1.0 / np.sqrt(A.diagonal()); Dm = sp.diags(ds); As = (Dm @ A @ Dm).tocsr()
                ml = pyamg.smoothed_aggregation_solver(As, B=B / ds[:, None], max_coarse=50, presmoother=sm, postsmoother=sm)
                Mn = ml.aspreconditioner(cycle="V")
                _, it, rel = pcg(S, lambda r: torch.as_tensor(ds * (Mn @ (ds * r.cpu().numpy())), device=dev), maxit=3000)
                acc.setdefault(f"{name} (cheb{deg})", []).append((it if rel <= 1 else np.nan, amg_wu(ml, k=2 * deg + 1)))
    print("==", os.path.basename(d.rstrip("/")))
    for k, v in acc.items():
        it = np.array([a for a, _ in v], float); wu = np.mean([b for _, b in v])
        print(f"  {k:<24} it={np.nanmean(it):6.1f} WU/apply={wu:5.2f} total={np.nanmean(it) * (1 + wu):6.0f}", flush=True)
