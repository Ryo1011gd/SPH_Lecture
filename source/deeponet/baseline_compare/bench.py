"""Offline baseline comparison of the DeepONet two-level preconditioners
against classical preconditioners, on solver snapshots (ppe_*.bin / vis_*.bin).

    python bench.py vis --data DIR... --every 10 --out vis.json
    python bench.py ppe --data DIR... --every 10 --model ../ppe_deeponet/trained_model.pre_alldata_20261014/best.pt --out ppe.json

Every method is run inside the torch replica of the production Krylov solver
(PCG for viscosity, right-preconditioned BiCGStab for the PPE): same warm start,
same stopping test.  Work per preconditioner application is reported in units of
one fine-level matvec (WU) so that iteration counts can be turned into cost.
"""
import argparse, json, os, sys, time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import pyamg

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- helpers
def to_scipy(s, m):
    return sp.csr_matrix((s["val"], s["col"], s["rowptr"]), shape=(m, m))


def wrap_np(fn, dev):
    """numpy preconditioner -> torch preconditioner"""
    def apply(r):
        return torch.as_tensor(fn(r.detach().cpu().numpy()), device=dev, dtype=torch.float64)
    return apply


def amg_precond(ml, dev):
    M = ml.aspreconditioner(cycle="V")
    return wrap_np(lambda r: M @ r, dev), float(ml.cycle_complexity(cycle="V"))


def ilu_precond(A, dev):
    """ILU with fill restricted to roughly the sparsity of A (scipy SuperLU ILUTP)."""
    ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=1.0, permc_spec="NATURAL")
    wu = (ilu.L.nnz + ilu.U.nnz) / A.nnz
    return wrap_np(ilu.solve, dev), wu


def cheb_wu(deg, passes, nc, n, nnz, extra_matvec=1):
    """WU of passes x Chebyshev(deg) + residual + dense coarse correction with nc columns."""
    return passes * deg + extra_matvec + (2.0 * n * nc / nnz if nc else 0.0)


def geometric_basis(pos, nbox_target, ncomp, rigid, null_const=False, D=None):
    """Non-learned coarse space: k-means-free box aggregation of the particles.
    Each aggregate contributes its indicator (scalar) or, for vectors,
    translations (+ rotation when rigid=True)."""
    n, sdim = pos.shape
    lo, hi = pos.min(0), pos.max(0)
    ext = np.maximum(hi - lo, 1e-12)
    # boxes per direction proportional to the extent so that #aggregates ~ nbox_target
    vol = np.prod(ext)
    h = (vol / nbox_target) ** (1.0 / sdim)
    nb = np.maximum(1, np.round(ext / h)).astype(int)
    idx = np.minimum(((pos - lo) / ext * nb).astype(int), nb - 1)
    key = np.ravel_multi_index(idx.T, nb)
    _, agg = np.unique(key, return_inverse=True)
    na = agg.max() + 1
    cols = []
    for a in range(na):
        sel = agg == a
        if ncomp == 1:
            v = np.zeros(n); v[sel] = 1.0; cols.append(v)
            continue
        c = pos[sel].mean(0)
        for d in range(ncomp):
            v = np.zeros(n * ncomp); v[np.where(sel)[0] * ncomp + d] = 1.0; cols.append(v)
        if rigid and ncomp == 2:
            v = np.zeros(n * ncomp)
            ii = np.where(sel)[0]
            v[ii * 2 + 0] = -(pos[ii, 1] - c[1]); v[ii * 2 + 1] = pos[ii, 0] - c[0]
            cols.append(v)
    V = np.stack(cols, 1)
    if null_const and ncomp == 1:
        w = 1.0 / D
        V = V - (w @ V) / w.sum()
        V = V[:, np.linalg.norm(V, axis=0) > 1e-12]
    return V


# ---------------------------------------------------------------- viscosity
def run_vis(args):
    sys.path.insert(0, os.path.join(HERE, "..", "vis_deeponet"))
    from vis_don_data import VisSystem, find_snapshots, read_snapshot, trajectory_of
    from vis_don_model import Smoother, cycle, estimate_lmax, galerkin_coarse, galerkin_matrix, load_checkpoint
    from vis_don_solver import deeponet_preconditioner, jacobi_preconditioner, pcg

    dev = torch.device(args.device)
    model, ck = load_checkpoint(args.model, dev); model.eval()
    ta = ck.get("args", {})
    deg, ratio = ta.get("cheb_degree", 6), ta.get("cheb_ratio", 30.0)
    P = ck["cfg"]["basis"]
    files = find_snapshots(args.data)[::args.every]
    res = defaultdict(lambda: defaultdict(list))
    for k, f in enumerate(files):
        s = read_snapshot(f)
        S = VisSystem(s, dev)
        A = to_scipy(s, S.m)
        n, nc = s["n"], s["ncomp"]
        sm = Smoother(cheb_degree=deg, cheb_ratio=ratio, lmax=estimate_lmax(S))
        row = res[os.path.basename(trajectory_of(f))]
        row["time"].append(s["time"]); row["n"].append(n)
        row["stiff"].append(float(np.expm1(10.0 * s["glob"][4])))
        methods = {}
        methods["Jacobi"] = (lambda: (jacobi_preconditioner(S), 0.0))
        methods["Cheb"] = (lambda: ((lambda r: sm(S, torch.zeros_like(r)[:, None], r[:, None])[:, 0]),
                                    cheb_wu(deg, 1, 0, n, A.nnz, 0)))
        B_tr = np.kron(np.ones((n, 1)), np.eye(nc))
        pos = s["pos"]; c = pos.mean(0)
        rot = np.zeros((n * nc, 1)); rot[0::2, 0] = -(pos[:, 1] - c[1]); rot[1::2, 0] = pos[:, 0] - c[0]
        B_rb = np.hstack([B_tr, rot])
        methods["SA(並進)"] = (lambda: amg_precond(pyamg.smoothed_aggregation_solver(A, B=B_tr, max_coarse=50), dev))
        methods["SA(剛体)"] = (lambda: amg_precond(pyamg.smoothed_aggregation_solver(A, B=B_rb, max_coarse=50), dev))
        methods["IC/ILU"] = (lambda: ilu_precond(A, dev))

        def geo():
            V = torch.as_tensor(geometric_basis(pos, P * nc // 3, nc, rigid=True), device=dev)
            C = galerkin_coarse(V, galerkin_matrix(V, S.matvec, 1e-10))
            return (lambda r: cycle(S, r[:, None], torch.zeros_like(r)[:, None], C, sm)[:, 0],
                    cheb_wu(deg, 2, V.shape[1], S.m, A.nnz))
        methods["Cheb+幾何剛体粗空間"] = geo
        methods["DeepONet"] = (lambda: (deeponet_preconditioner(model, S, sm), cheb_wu(deg, 2, P * nc, S.m, A.nnz)))
        for name, mk in methods.items():
            t0 = time.perf_counter()
            try:
                M, wu = mk()
            except Exception as e:  # noqa
                print(f"  {name}: setup failed {e}"); row["it_" + name].append(np.nan); continue
            ts = time.perf_counter() - t0
            _, it, rel = pcg(S, M, maxit=args.maxit)
            ok = rel <= 1.0
            row["it_" + name].append(it if ok else np.nan)
            row["wu_" + name].append(wu); row["setup_" + name].append(ts)
        print(f"[{k+1}/{len(files)}] {os.path.basename(os.path.dirname(f))} t={s['time']:.3f} n={n} "
              + " ".join(f"{m}={row['it_'+m][-1]}" for m in methods), flush=True)
    return res


# ---------------------------------------------------------------- PPE
def run_ppe(args):
    sys.path.insert(0, os.path.join(HERE, "..", "ppe_deeponet"))
    from ppe_don_data import PpeSystem, find_snapshots, read_snapshot, trajectory_of
    from ppe_don_model import PpeDeepONet, Smoother, cycle, estimate_lmax, load_checkpoint
    from ppe_don_solver import bicgstab, deeponet_preconditioner, jacobi_preconditioner

    dev = torch.device(args.device)
    model, ck = load_checkpoint(args.model, dev); model.eval()
    ta = ck.get("args", {})
    deg, ratio = ta.get("cheb_degree", 10), ta.get("cheb_ratio", 30.0)
    if args.cheb_degree is not None: deg = args.cheb_degree
    P = ck["cfg"]["basis"]
    files = find_snapshots(args.data)[::args.every]
    res = defaultdict(lambda: defaultdict(list))
    for k, f in enumerate(files):
        s = read_snapshot(f)
        S = PpeSystem(s, dev)
        n = s["n"]
        A = to_scipy(s, n)
        if s["pure_neumann"]:
            A = (sp.eye(n) - sp.csr_matrix(np.ones((n, n)) / n)) @ A if n < 0 else A  # projection applied in matvec only
        sm = Smoother(cheb_degree=deg, cheb_ratio=ratio, lmax=estimate_lmax(S))
        row = res[os.path.basename(trajectory_of(f))]
        row["time"].append(s["time"]); row["n"].append(n)
        tol = s["tol"]
        methods = {}
        methods["Jacobi"] = lambda: (jacobi_preconditioner(S), 0.0)
        methods["ILU"] = lambda: ilu_precond(A, dev)
        methods["SA"] = lambda: amg_precond(pyamg.smoothed_aggregation_solver(
            A, symmetry="nonsymmetric", max_coarse=50), dev)
        methods["AIR"] = lambda: amg_precond(pyamg.air_solver(A, max_coarse=50), dev)

        def geo():
            V = torch.as_tensor(geometric_basis(s["pos"], P, 1, False, S.null_const, s["diag"]), device=dev)
            Ac = PpeDeepONet.galerkin_factors(V, S.D, S.matvec, 1e-10)
            Q, R = V @ torch.linalg.inv(Ac), V / S.D[:, None]
            C = lambda r: Q @ (R.T @ r)
            def ap(r):
                z = cycle(S, r[:, None], torch.zeros_like(r)[:, None], C, sm, sm)[:, 0]
                return z - z.mean() if S.null_const else z
            return ap, cheb_wu(deg, 2, V.shape[1], n, A.nnz)
        methods["Cheb+幾何粗空間"] = geo
        methods["DeepONet"] = lambda: (deeponet_preconditioner(model, S, "galerkin", sm, 1, 1),
                                       cheb_wu(deg, 2, P, n, A.nnz))
        for name, mk in methods.items():
            t0 = time.perf_counter()
            try:
                M, wu = mk()
            except Exception as e:  # noqa
                print(f"  {name}: setup failed {e}"); row["it_" + name].append(np.nan); continue
            ts = time.perf_counter() - t0
            ig = name != "Jacobi"
            _, it, rel = bicgstab(S, M, tol=tol, maxit=args.maxit, initial_guess=ig)
            ok = np.isfinite(rel) and rel <= tol * 1.0001
            row["it_" + name].append(it if ok else np.nan)
            row["wu_" + name].append(wu); row["setup_" + name].append(ts)
        print(f"[{k+1}/{len(files)}] {os.path.basename(os.path.dirname(f))} n={n} neu={int(S.null_const)} "
              + " ".join(f"{m}={row['it_'+m][-1]}" for m in methods), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["vis", "ppe"])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--maxit", type=int, default=3000)
    ap.add_argument("--cheb-degree", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.model is None:
        a.model = os.path.join(HERE, "..", "vis_deeponet" if a.kind == "vis" else "ppe_deeponet", "trained_model", "best.pt")
    res = run_vis(a) if a.kind == "vis" else run_ppe(a)
    with open(a.out, "w") as fp:
        json.dump({k: {m: [float(x) for x in v] for m, v in d.items()} for k, d in res.items()}, fp, ensure_ascii=False)


if __name__ == "__main__":
    main()
