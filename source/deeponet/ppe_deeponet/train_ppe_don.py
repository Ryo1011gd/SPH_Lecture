"""Train the PPE DeepONet preconditioner from solver snapshots.

    python train_ppe_don.py --data RUN1/trained_data RUN2/trained_data ... --out model_dir

Losses (all use the exported matrix A of each snapshot, no extra PDE solves):

  galerkin : || S^post (I - T Ac^-1 T^T D^-1 A) S^pre e ||^2      two-grid error propagation
  learned  : || S^post (I - Q R^T A) S^pre e ||^2                  same cycle with the learned operator
  solution : || G(A y) - y ||^2 / ||y||^2 on random trunk queries   y in {p, p - p0}  (Ax=b and A d = r0)
  ortho    : || T^T T / n - I ||^2                                  keeps the Galerkin system well conditioned

e are probe vectors: white noise and algebraically smooth vectors (white noise
after k damped-Jacobi sweeps) -- exactly the error components that Jacobi
cannot remove and that the coarse correction has to capture.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from ppe_don_data import PpeSystem, find_snapshots, read_snapshot, trajectory_of
from ppe_don_model import PpeDeepONet, Smoother, estimate_lmax, export_weights, two_level
from ppe_don_solver import bicgstab, deeponet_preconditioner, jacobi_preconditioner


def augment(sysm, rng):
    """Dihedral symmetries: A depends only on inter-particle distances, so a
    reflected/transposed coordinate frame with the same A is a valid sample."""
    xh = sysm.xhat.clone()
    gl = sysm.glob.clone()
    for d in range(sysm.sdim):
        if rng.random() < 0.5:
            xh[:, d] = -xh[:, d]
    if sysm.sdim == 2 and rng.random() < 0.5:
        xh = xh[:, [1, 0]]
        gl[[1, 2]] = gl[[2, 1]]
    sysm.xhat, sysm.glob = xh, gl
    return sysm


@torch.no_grad()
def make_probes(sysm, n_white, n_smooth, sweeps, omega, gen):
    n = sysm.n
    cols = [torch.randn(n, n_white, dtype=torch.float64, device=sysm.D.device, generator=gen)]
    if n_smooth > 0:
        e = torch.randn(n, n_smooth, dtype=torch.float64, device=sysm.D.device, generator=gen)
        ks = sorted(random.choice(sweeps) for _ in range(n_smooth))
        done = 0
        out = []
        for j, k in enumerate(ks):                       # incremental smoothing, reuse sweeps
            while done < k:
                e[:, j:] = e[:, j:] - omega * sysm.matvec(e[:, j:]) / sysm.D[:, None]
                done += 1
            out.append(e[:, j:j + 1].clone())
        cols += out
    E = torch.cat(cols, dim=1)
    if sysm.pure_neumann:
        E = E - E.mean(dim=0, keepdim=True)
    return E / E.norm(dim=0, keepdim=True).clamp_min(1e-300)


def make_smoother(args, sysm):
    lmax = None
    if args.cheb_degree > 0:
        lmax = sysm.meta.get("lmax")
        if lmax is None:
            lmax = sysm.meta["lmax"] = estimate_lmax(sysm)
    return Smoother(args.omega, args.smooth_sweeps, args.cheb_degree, args.cheb_ratio, lmax)


def snapshot_losses(model, sysm, args, gen, train=True):
    n = sysm.n
    need_branch = args.w_learned > 0 or args.w_solution > 0
    T, Psi, K = model.fields(sysm.xhat, sysm.loc, sysm.glob, need_branch=need_branch)
    D = sysm.D
    E0 = make_probes(sysm, args.white_probes, args.smooth_probes, args.sweeps, args.omega, gen)
    out = {}
    sm = make_smoother(args, sysm)
    pre = sm if args.npre else None
    post = sm if args.npost else None

    def centred(e):
        return e - e.mean(dim=0, keepdim=True) if sysm.pure_neumann else e

    if args.w_galerkin > 0:
        Tg = PpeDeepONet.galerkin_basis(T, D, sysm.null_const)
        Ac = PpeDeepONet.galerkin_factors(Tg, D, sysm.matvec, args.reg)
        cg = lambda R_in: PpeDeepONet.galerkin_coarse(Tg, D, Ac, R_in)
        E = centred(two_level(E0, sysm, cg, pre, post))
        out["galerkin"] = (E.norm(dim=0) ** 2).mean()
    if args.w_learned > 0:
        cl = lambda R_in: model.learned_coarse(T, Psi, K, D, R_in)
        E = centred(two_level(E0, sysm, cl, pre, post))
        out["learned"] = (E.norm(dim=0) ** 2).mean()
    if args.w_solution > 0:
        Y = torch.stack([sysm.p, sysm.p - sysm.p0], dim=1)
        Y = centred(Y)
        rhs = sysm.matvec(Y)
        q = torch.randperm(n, device=D.device, generator=gen)[:min(args.queries, n)] if train else None
        Tq = T if q is None else T[q]
        Yh = model.learned_coarse(T, Psi, K, D, rhs, Tq=Tq)
        Yq = Y if q is None else Y[q]
        if sysm.pure_neumann:
            Yh, Yq = centred(Yh), centred(Yq)
        out["solution"] = ((Yh - Yq).norm(dim=0) ** 2 / Yq.norm(dim=0).clamp_min(1e-300) ** 2).mean()
    if args.w_ortho > 0:
        G = T.T @ T / n
        out["ortho"] = ((G - torch.eye(model.P, dtype=G.dtype, device=G.device)) ** 2).mean()
    return out


def total(losses, args):
    return sum(getattr(args, "w_" + k) * v for k, v in losses.items())


def load_all(files, device_cache="cpu"):
    data = []
    for f in files:
        data.append(read_snapshot(f))
    return data


def evaluate(model, val, args, device, n_solve=3):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(1234)
    acc = {}
    with torch.no_grad():
        for s in val:
            L = snapshot_losses(model, PpeSystem(s, device), args, gen, train=False)
            for k, v in L.items():
                acc.setdefault(k, []).append(v.item())
        its = {"jacobi": [], "galerkin": [], "learned": []}
        for s in (val[:: max(1, len(val) // n_solve)][:n_solve] if n_solve > 0 else []):
            S = PpeSystem(s, device)
            its["jacobi"].append(bicgstab(S, jacobi_preconditioner(S), tol=args.tol)[1])
            for c in ("galerkin", "learned"):
                M = deeponet_preconditioner(model, S, c, make_smoother(args, S), args.npre, args.npost, args.reg)
                its[c].append(bicgstab(S, M, tol=args.tol, initial_guess=True)[1])
    model.train()
    res = {k: float(np.mean(v)) for k, v in acc.items()}
    res.update({"iter_" + k: float(np.mean(v)) for k, v in its.items() if v})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="snapshot directories/files")
    ap.add_argument("--val-data", nargs="*", default=None, help="explicit validation runs (default: hold out whole runs)")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--out", default="ppe_don_model")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--basis", type=int, default=64)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--fourier", type=int, default=16)
    ap.add_argument("--fourier-max", type=float, default=6.0)
    ap.add_argument("--white-probes", type=int, default=2)
    ap.add_argument("--smooth-probes", type=int, default=10)
    ap.add_argument("--sweeps", type=int, nargs="+", default=[16, 64, 256, 512])
    ap.add_argument("--queries", type=int, default=2048, help="random trunk query particles for the solution loss")
    ap.add_argument("--omega", type=float, default=0.7)
    ap.add_argument("--npre", type=int, default=1)
    ap.add_argument("--npost", type=int, default=1)
    ap.add_argument("--smooth-sweeps", type=int, default=1, help="Jacobi sweeps per smoothing step")
    ap.add_argument("--cheb-degree", type=int, default=10, help=">0: Chebyshev smoother of this degree, 0: damped Jacobi")
    ap.add_argument("--cheb-ratio", type=float, default=30.0, help="Chebyshev interval [lmax/ratio, lmax]")
    ap.add_argument("--reg", type=float, default=1e-10)
    ap.add_argument("--w-galerkin", type=float, default=1.0)
    ap.add_argument("--w-learned", type=float, default=1.0)
    ap.add_argument("--w-solution", type=float, default=0.3)
    ap.add_argument("--w-ortho", type=float, default=1e-3)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--max-snapshots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    files = find_snapshots(args.data)
    if args.max_snapshots:
        random.shuffle(files); files = sorted(files[:args.max_snapshots])
    if args.val_data:
        val_files = find_snapshots(args.val_data)
        train_files = [f for f in files if f not in set(val_files)]
    elif args.val_fraction <= 0:            # train on everything; monitor on every 10th snapshot (not held out)
        train_files, val_files = files, files[::10]
    else:                                   # hold out complete runs, never random time steps
        trajs = sorted({trajectory_of(f) for f in files})
        random.shuffle(trajs)
        nval = max(1, int(round(args.val_fraction * len(trajs)))) if len(trajs) > 1 else 0
        vt = set(trajs[:nval])
        val_files = [f for f in files if trajectory_of(f) in vt]
        train_files = [f for f in files if trajectory_of(f) not in vt]
        if not val_files:                   # single run: last 20 % in time
            k = max(1, int(len(files) * args.val_fraction))
            train_files, val_files = files[:-k], files[-k:]
    print(f"train snapshots {len(train_files)}  val snapshots {len(val_files)}")
    train = load_all(train_files)
    val = load_all(val_files)
    sdim = train[0]["sdim"]
    os.makedirs(args.out, exist_ok=True)

    dev = torch.device(args.device)
    model = PpeDeepONet(sdim=sdim, n_fourier=args.fourier, fourier_max=args.fourier_max, width=args.width,
                        depth=args.depth, latent=args.latent, basis=args.basis, seed=args.seed).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    rng = random.Random(args.seed)
    best = math.inf
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        run = {}
        for idx in order:
            S = PpeSystem(train[idx], dev)
            if not args.no_augment:
                S = augment(S, rng)
            L = snapshot_losses(model, S, args, gen)
            loss = total(L, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k, v in L.items():
                run.setdefault(k, []).append(v.item())
        msg = " ".join(f"{k}={np.mean(v):.3e}" for k, v in run.items())
        V = evaluate(model, val, args, dev, n_solve=3 if (ep % 5 == 4 or ep == args.epochs - 1) else 0)
        score = V.get("galerkin", 0) * args.w_galerkin + V.get("learned", 0) * args.w_learned + V.get("solution", 0) * args.w_solution
        hist.append(dict(epoch=ep, train={k: float(np.mean(v)) for k, v in run.items()}, val=V))
        print(f"ep {ep:3d} {time.time()-t0:7.1f}s train {msg} | val " +
              " ".join(f"{k}={v:.3e}" if not k.startswith("iter") else f"{k}={v:.1f}" for k, v in V.items()), flush=True)
        if score < best:
            best = score
            torch.save(dict(cfg=model.cfg, state=model.state_dict(), args=vars(args)), os.path.join(args.out, "best.pt"))
            export_weights(model, os.path.join(args.out, "ppe_deeponet.bin"))
    torch.save(dict(cfg=model.cfg, state=model.state_dict(), args=vars(args)), os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(dict(args=vars(args), train_files=train_files, val_files=val_files, history=hist), f, indent=1)
    with open(os.path.join(args.out, "ppe_deeponet.data.txt"), "w") as f:
        f.write("# paste into the solver .data file (settings this model was trained for)\n")
        f.write(f"PpeDon              1\nPpeDonModel         {os.path.abspath(os.path.join(args.out, 'ppe_deeponet.bin'))}\n")
        f.write("PpeDonCoarse        1\n")
        f.write(f"PpeDonPreSmooth     {1 if args.npre else 0}\nPpeDonPostSmooth    {1 if args.npost else 0}\n")
        f.write(f"PpeDonChebDegree    {args.cheb_degree}\nPpeDonChebRatio     {args.cheb_ratio}\n")
        f.write(f"PpeDonSmoothSweeps  {args.smooth_sweeps}\nPpeDonOmega         {args.omega}\n")
        f.write("PpeDonInitialGuess  1\n")
    print("best model ->", os.path.join(args.out, "ppe_deeponet.bin"))
    print("solver settings ->", os.path.join(args.out, "ppe_deeponet.data.txt"))


if __name__ == "__main__":
    main()
