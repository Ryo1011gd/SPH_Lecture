"""SA-AMG with a GPU-friendly smoother (2 sweeps of damped Jacobi, pre and post)."""
import json, os, sys
import numpy as np, torch, pyamg
from bench import to_scipy, amg_precond
HERE = os.path.dirname(os.path.abspath(__file__))
kind, out, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
dev = torch.device("cuda")
sm = (("jacobi", {"omega": 2.0 / 3.0, "iterations": 2}),) * 1
res = {}
if kind == "vis":
    sys.path.insert(0, os.path.join(HERE, "..", "vis_deeponet"))
    from vis_don_data import VisSystem, find_snapshots, read_snapshot
    from vis_don_solver import pcg
    for d in dirs:
        its = []
        for f in find_snapshots([d])[::10]:
            s = read_snapshot(f); S = VisSystem(s, dev); A = to_scipy(s, S.m); n = s["n"]
            pos = s["pos"]; c = pos.mean(0)
            B = np.kron(np.ones((n, 1)), np.eye(2))
            rot = np.zeros((2 * n, 1)); rot[0::2, 0] = -(pos[:, 1] - c[1]); rot[1::2, 0] = pos[:, 0] - c[0]
            ml = pyamg.smoothed_aggregation_solver(A, B=np.hstack([B, rot]), max_coarse=50,
                                                   presmoother=sm[0], postsmoother=sm[0])
            M, wu = amg_precond(ml, dev)
            _, it, rel = pcg(S, M, maxit=3000); its.append((it if rel <= 1 else np.nan, wu))
        res[os.path.basename(d.rstrip("/"))] = its; print(d, np.nanmean([i for i, _ in its]), np.mean([w for _, w in its]), flush=True)
else:
    sys.path.insert(0, os.path.join(HERE, "..", "ppe_deeponet"))
    from ppe_don_data import PpeSystem, find_snapshots, read_snapshot
    from ppe_don_solver import bicgstab
    for d in dirs:
        its = []
        for f in find_snapshots([d])[::10]:
            s = read_snapshot(f); S = PpeSystem(s, dev); A = to_scipy(s, s["n"])
            ml = pyamg.smoothed_aggregation_solver(A, symmetry="nonsymmetric", max_coarse=50,
                                                   presmoother=sm[0], postsmoother=sm[0])
            M, wu = amg_precond(ml, dev)
            _, it, rel = bicgstab(S, M, tol=s["tol"], maxit=3000, initial_guess=True)
            its.append((it if rel <= s["tol"] * 1.0001 else np.nan, wu))
        res[os.path.basename(d.rstrip("/"))] = its; print(d, np.nanmean([i for i, _ in its]), np.mean([w for _, w in its]), flush=True)
json.dump(res, open(out, "w"))
