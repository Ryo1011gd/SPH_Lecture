"""Offline evaluation of a trained PPE DeepONet on solver snapshots.

    python eval_ppe_don.py --model model_dir/best.pt --data RUN_A/snaps RUN_B/snaps [--every 5]

For each snapshot the solver's BiCGStab is replayed in torch (same recurrences,
same warm start p0, same tolerance) with
    jacobi    the current solver
    galerkin  DeepONet trunk basis + Galerkin coarse operator (PpeDonCoarse 1)
    learned   pure DeepONet coarse operator                 (PpeDonCoarse 0)
and the iteration counts are reported per run directory, together with the
two-level convergence factor rho(E) estimated by power iteration and the
relative error of the DeepONet one-shot prediction of p (Ax=b).
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from ppe_don_data import PpeSystem, find_snapshots, read_snapshot, trajectory_of
from ppe_don_model import PpeDeepONet, Smoother, estimate_lmax, load_checkpoint, two_level
from ppe_don_solver import bicgstab, deeponet_preconditioner, jacobi_preconditioner


@torch.no_grad()
def convergence_factor(model, S, coarse, sm, npre, npost, iters=30, reg=1e-10):
    T, Psi, K = model.fields(S.xhat, S.loc, S.glob, need_branch=(coarse == "learned"))
    if coarse == "galerkin":
        T = PpeDeepONet.galerkin_basis(T, S.D, S.null_const)
        Ac = PpeDeepONet.galerkin_factors(T, S.D, S.matvec, reg)
        C = lambda r: PpeDeepONet.galerkin_coarse(T, S.D, Ac, r)
    elif coarse == "learned":
        C = lambda r: model.learned_coarse(T, Psi, K, S.D, r)
    else:
        C = lambda r: torch.zeros_like(r)
    e = torch.randn(S.n, 1, dtype=torch.float64, device=S.D.device)
    rho = 0.0
    for _ in range(iters):
        e = e / e.norm()
        e2 = two_level(e, S, C, sm if npre else None, sm if npost else None)
        if S.pure_neumann:
            e2 = e2 - e2.mean()
        rho = e2.norm().item()
        e = e2
    return rho


@torch.no_grad()
def one_shot_error(model, S):
    T, Psi, K = model.fields(S.xhat, S.loc, S.glob)
    ph = model.learned_coarse(T, Psi, K, S.D, S.b[:, None])[:, 0]
    y = S.p
    if S.pure_neumann:
        ph, y = ph - ph.mean(), y - y.mean()
    return ((ph - y).norm() / y.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=1, help="use every k-th snapshot")
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--npre", type=int, default=None)
    ap.add_argument("--npost", type=int, default=None)
    ap.add_argument("--smooth-sweeps", type=int, default=None)
    ap.add_argument("--cheb-degree", type=int, default=None)
    ap.add_argument("--cheb-ratio", type=float, default=None)
    ap.add_argument("--tol", type=float, default=None, help="default: tolerance stored in the snapshot")
    ap.add_argument("--modes", nargs="+", default=["jacobi", "galerkin", "learned"])
    ap.add_argument("--no-initial-guess", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model, ck = load_checkpoint(args.model, dev)
    model.eval()
    ta = ck.get("args", {})
    omega = args.omega if args.omega is not None else ta.get("omega", 0.7)
    npre = args.npre if args.npre is not None else ta.get("npre", 1)
    npost = args.npost if args.npost is not None else ta.get("npost", 1)
    sweeps = args.smooth_sweeps if args.smooth_sweeps is not None else ta.get("smooth_sweeps", 1)
    deg = args.cheb_degree if args.cheb_degree is not None else ta.get("cheb_degree", 0)
    ratio = args.cheb_ratio if args.cheb_ratio is not None else ta.get("cheb_ratio", 15.0)
    files = find_snapshots(args.data)[::args.every]
    print(f"{len(files)} snapshots  omega={omega} pre={npre} post={npost} sweeps={sweeps} cheb_degree={deg} cheb_ratio={ratio}")

    per_run = defaultdict(lambda: defaultdict(list))
    for f in files:
        s = read_snapshot(f)
        S = PpeSystem(s, dev)
        tol = args.tol if args.tol is not None else s["tol"]
        sm = Smoother(omega, sweeps, deg, ratio, estimate_lmax(S) if deg > 0 else None)
        row = per_run[trajectory_of(f)]
        for m in args.modes:
            if m == "jacobi":
                M = jacobi_preconditioner(S)
                ig = False
            else:
                M = deeponet_preconditioner(model, S, m, sm, npre, npost)
                ig = not args.no_initial_guess
            _, it, rel = bicgstab(S, M, tol=tol, initial_guess=ig)
            row["iter_" + m].append(it)
            if m != "jacobi":
                row["rho_" + m].append(convergence_factor(model, S, m, sm, npre, npost))
        row["rho_jacobi_smoother"].append(convergence_factor(model, S, "none", sm, npre, npost))
        row["oneshot_err"].append(one_shot_error(model, S))
        row["n"].append(s["n"])

    summary = {}
    hdr = f"{'run':<45} {'n':>6} " + " ".join(f"{'it_'+m:>12}" for m in args.modes) + \
          f" {'rho_gal':>8} {'rho_lrn':>8} {'oneshot':>8}"
    print(hdr)
    for run, row in per_run.items():
        name = os.path.basename(os.path.dirname(run)) if os.path.basename(run) == "snaps" else os.path.basename(run)
        summary[name] = {k: float(np.mean(v)) for k, v in row.items()}
        summary[name]["snapshots"] = len(row["n"])
        line = f"{name[:45]:<45} {int(np.mean(row['n'])):>6} " + \
               " ".join(f"{np.mean(row['iter_'+m]):>12.1f}" for m in args.modes) + \
               f" {np.mean(row.get('rho_galerkin', [np.nan])):>8.3f} {np.mean(row.get('rho_learned', [np.nan])):>8.3f}" + \
               f" {np.mean(row['oneshot_err']):>8.3f}"
        print(line)
    if args.json:
        with open(args.json, "w") as fp:
            json.dump(dict(omega=omega, npre=npre, npost=npost, sweeps=sweeps, cheb_degree=deg, cheb_ratio=ratio, summary=summary), fp, indent=1)


if __name__ == "__main__":
    main()
