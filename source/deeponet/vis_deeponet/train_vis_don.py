"""Train the DeepONet coarse space for the implicit-viscosity solve from solver snapshots.

    python train_vis_don.py --data RUN1 RUN2 ... --out model_dir

Losses (use only the exported matrix A of each snapshot, no extra solves):

  energy : || E e ||_A^2 / || e ||_A^2,   E = S (I - V (V^T A V)^-1 V^T A) S
           the A-norm error propagation of the symmetric two-level cycle, i.e. exactly
           what governs the CG convergence rate
  ortho  : || T^T T / n - I ||^2           keeps V^T A V well conditioned

Probes e: white noise, algebraically smooth vectors (white noise after k damped
Jacobi sweeps: the components Jacobi cannot remove, e.g. rigid motions of a
solidified region), and the actual warm-start error u - u0 of the snapshot.
Directories without vis_*.bin (PPE-only runs) are skipped.
"""
import argparse
import glob as _glob
import json
import math
import os
import random
import time

import numpy as np
import torch

from vis_don_data import VisSystem, find_snapshots, read_snapshot, trajectory_of
from vis_don_model import (Smoother, VisDeepONet, coarse_basis, estimate_lmax, export_weights,
                           galerkin_coarse, galerkin_matrix, two_level)
from vis_don_solver import deeponet_preconditioner, jacobi_preconditioner, pcg


def augment(sysm, rng):
    """Reflections / axis swap of the input frame.  The coarse space T (x) I is
    invariant under the matching component transformation of A, so the same A
    with transformed coordinates is the matrix of the mirrored flow."""
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
    m = sysm.m
    dev = sysm.D.device
    cols = [torch.randn(m, n_white, dtype=torch.float64, device=dev, generator=gen)]
    if n_smooth > 0:
        e = torch.randn(m, n_smooth, dtype=torch.float64, device=dev, generator=gen)
        ks = sorted(random.choice(sweeps) for _ in range(n_smooth))
        done = 0
        for j, k in enumerate(ks):                     # incremental smoothing, reuse sweeps
            while done < k:
                e[:, j:] = e[:, j:] - omega * sysm.matvec(e[:, j:]) / sysm.D[:, None]
                done += 1
            cols.append(e[:, j:j + 1].clone())
    e0 = (sysm.u - sysm.u0)[:, None]
    if e0.norm() > 0:
        cols.append(e0)
    E = torch.cat(cols, dim=1)
    en = torch.sqrt((E * sysm.matvec(E)).sum(0)).clamp_min(1e-300)  # unit A-norm
    return E / en


def make_smoother(args, sysm):
    lmax = None
    if args.cheb_degree > 0:
        lmax = sysm.meta.get("lmax")
        if lmax is None:
            lmax = sysm.meta["lmax"] = estimate_lmax(sysm)
    return Smoother(args.omega, args.smooth_sweeps, args.cheb_degree, args.cheb_ratio, lmax)


def snapshot_losses(model, sysm, args, gen):
    T = model.trunk(sysm.xhat, sysm.loc, sysm.glob)
    E0 = make_probes(sysm, args.white_probes, args.smooth_probes, args.sweeps, args.omega, gen)
    V = coarse_basis(T, sysm.nc)
    C = galerkin_coarse(V, galerkin_matrix(V, sysm.matvec, args.reg))
    E = two_level(E0, sysm, C, make_smoother(args, sysm))
    out = {"energy": (E * sysm.matvec(E)).sum(0).mean()}
    if args.w_ortho > 0:
        G = T.T @ T / T.shape[0]
        out["ortho"] = ((G - torch.eye(model.P, dtype=G.dtype, device=G.device)) ** 2).mean()
    return out


def total(losses, args):
    return sum(getattr(args, "w_" + k) * v for k, v in losses.items())


def evaluate(model, val, args, device, n_solve=3):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(1234)
    acc = {}
    its = {"jacobi": [], "deeponet": []}
    with torch.no_grad():
        for s in val:
            L = snapshot_losses(model, VisSystem(s, device), args, gen)
            for k, v in L.items():
                acc.setdefault(k, []).append(v.item())
        # the hardest validation systems (most Jacobi iterations in the run)
        hard = sorted(val, key=lambda s: -s["solver_iter"])[:n_solve] if n_solve > 0 else []
        for s in hard:
            S = VisSystem(s, device)
            its["jacobi"].append(pcg(S, jacobi_preconditioner(S))[1])
            M = deeponet_preconditioner(model, S, make_smoother(args, S), args.reg)
            its["deeponet"].append(pcg(S, M)[1])
    model.train()
    res = {k: float(np.mean(v)) for k, v in acc.items()}
    res.update({"iter_" + k: float(np.mean(v)) for k, v in its.items() if v})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="snapshot directories/files")
    ap.add_argument("--val-data", nargs="*", default=None, help="explicit validation runs (default: hold out whole runs)")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--out", default="vis_don_model")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--basis", type=int, default=64)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--fourier", type=int, default=16)
    ap.add_argument("--fourier-max", type=float, default=10.0)
    ap.add_argument("--white-probes", type=int, default=2)
    ap.add_argument("--smooth-probes", type=int, default=10)
    ap.add_argument("--sweeps", type=int, nargs="+", default=[16, 64, 256, 512])
    ap.add_argument("--omega", type=float, default=0.7)
    ap.add_argument("--smooth-sweeps", type=int, default=1, help="Jacobi sweeps when --cheb-degree 0")
    ap.add_argument("--cheb-degree", type=int, default=6, help=">0: Chebyshev smoother of this degree, 0: damped Jacobi")
    ap.add_argument("--cheb-ratio", type=float, default=30.0, help="Chebyshev interval [lmax/ratio, lmax]")
    ap.add_argument("--reg", type=float, default=1e-10)
    ap.add_argument("--w-energy", type=float, default=1.0)
    ap.add_argument("--w-ortho", type=float, default=1e-3)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--max-snapshots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    files = find_snapshots(args.data)
    if not files:
        ppe_dirs = sorted({os.path.dirname(os.path.abspath(f)) for r in args.data if os.path.isdir(r)
                           for f in _glob.glob(os.path.join(r, "**", "ppe_*.bin"), recursive=True)})
        msg = "no vis_*.bin snapshots found (generate them with make GPU=1 DATASET=1 VIS=1 or ALL=1)"
        if ppe_dirs:
            msg += (f"\n{len(ppe_dirs)} director{'y' if len(ppe_dirs) == 1 else 'ies'} contain only ppe_*.bin, i.e. they were"
                    " written by the PPE-only SPH_acc_dataset. Re-run those cases with SPH_acc_dataset_all"
                    " (or SPH_acc_dataset_vis), or add 'VisDonExportEvery 50' to the .data file of a binary"
                    " that includes vis_deeponet.inc:\n  " + "\n  ".join(ppe_dirs))
        raise SystemExit(msg)
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
        if not val_files:                   # single run: last part in time
            k = max(1, int(len(files) * args.val_fraction))
            train_files, val_files = files[:-k], files[-k:]
    print(f"runs {len({trajectory_of(f) for f in files})}  train snapshots {len(train_files)}  val snapshots {len(val_files)}")
    train = [read_snapshot(f) for f in train_files]
    val = [read_snapshot(f) for f in val_files]
    sdim = train[0]["sdim"]
    os.makedirs(args.out, exist_ok=True)

    dev = torch.device(args.device)
    model = VisDeepONet(sdim=sdim, n_fourier=args.fourier, fourier_max=args.fourier_max, width=args.width,
                        depth=args.depth, latent=args.latent, basis=args.basis, seed=args.seed).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train), pct_start=0.05)
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
            S = VisSystem(train[idx], dev)
            if not args.no_augment:
                S = augment(S, rng)
            L = snapshot_losses(model, S, args, gen)
            loss = total(L, args)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k, v in L.items():
                run.setdefault(k, []).append(v.item())
        msg = " ".join(f"{k}={np.mean(v):.3e}" for k, v in run.items())
        V = evaluate(model, val, args, dev, n_solve=3 if (ep % 5 == 4 or ep == args.epochs - 1) else 0)
        score = V.get("energy", math.inf)
        hist.append(dict(epoch=ep, train={k: float(np.mean(v)) for k, v in run.items()}, val=V))
        print(f"ep {ep:3d} {time.time()-t0:7.1f}s train {msg} | val " +
              " ".join(f"{k}={v:.3e}" if not k.startswith("iter") else f"{k}={v:.1f}" for k, v in V.items()), flush=True)
        if score < best:
            best = score
            torch.save(dict(cfg=model.cfg, state=model.state_dict(), args=vars(args)), os.path.join(args.out, "best.pt"))
            export_weights(model, os.path.join(args.out, "vis_deeponet.bin"))
    torch.save(dict(cfg=model.cfg, state=model.state_dict(), args=vars(args)), os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(dict(args=vars(args), train_files=train_files, val_files=val_files, history=hist), f, indent=1)
    with open(os.path.join(args.out, "vis_deeponet.data.txt"), "w") as f:
        f.write("# paste into the solver .data file (settings this model was trained for)\n")
        f.write(f"VisDon              1\nVisDonModel         {os.path.abspath(os.path.join(args.out, 'vis_deeponet.bin'))}\n")
        f.write(f"VisDonChebDegree    {args.cheb_degree}\nVisDonChebRatio     {args.cheb_ratio}\n")
        f.write(f"VisDonSmoothSweeps  {args.smooth_sweeps}\nVisDonOmega         {args.omega}\n")
    print("best model ->", os.path.join(args.out, "vis_deeponet.bin"))
    print("solver settings ->", os.path.join(args.out, "vis_deeponet.data.txt"))


if __name__ == "__main__":
    main()
