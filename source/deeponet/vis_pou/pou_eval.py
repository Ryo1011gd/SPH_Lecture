"""Iteration counts of the two-level PCG with PoU coarse spaces vs the global DeepONet basis.

python3 pou_eval.py [--model best.pt] [--hsub 16 ...] SNAPDIR ...
"""
import argparse, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vis_deeponet"))
from vis_don_data import VisSystem, find_snapshots, read_snapshot
from vis_don_model import Smoother, coarse_basis, cycle, estimate_lmax, load_checkpoint
from vis_don_solver import pcg
from pou_model import PouGrid, PouNet, pou_basis


def make_coarse(S, V):
    """r -> V Ac^+ V^T r, pseudo-inverse via eigh of the Galerkin matrix of orthonormalised V."""
    Q, _ = torch.linalg.qr(V)
    Ac = Q.T @ S.matvec(Q)
    w, U = torch.linalg.eigh(0.5 * (Ac + Ac.T))
    keep = w > 1e-12 * w.max()
    QU = Q @ U[:, keep]
    wk = w[keep]
    return (lambda R: QU @ ((QU.T @ R) / wk[:, None])), int(keep.sum())


def load_pou(path, dev):
    ck = torch.load(path, map_location=dev)
    m = PouNet(**ck["cfg"]).to(dev)
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--model", nargs="*", default=[], help="trained PouNet checkpoints")
    ap.add_argument("--hsub", type=float, nargs="+", default=[16.0, 11.3])
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--no-don", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    don = None
    if not a.no_don:
        don, _ = load_checkpoint(os.path.join(HERE, "..", "vis_deeponet", "trained_model", "best.pt"), dev); don.eval()
    pous = [load_pou(p, dev) for p in a.model]
    for d in a.dirs:
        res = {}
        files = find_snapshots([d])[::a.every]
        for f in files:
            s = read_snapshot(f); S = VisSystem(s, dev); nnz = s["nnz"]
            sm = Smoother(cheb_degree=6, cheb_ratio=30.0, lmax=estimate_lmax(S))
            feat = torch.as_tensor(s["loc"][:, 1:], device=dev, dtype=torch.float64)
            Vs = {}
            with torch.no_grad():
                if don is not None:
                    Vs["DON global P=64"] = coarse_basis(don.trunk(S.xhat, S.loc, S.glob), S.nc)
                for h in a.hsub:
                    g = PouGrid(s["pos"], s["dp"], h, device=dev)
                    one = torch.ones(g.n, 4, 1, dtype=torch.float64, device=dev)
                    Vs[f"PoU const H={h:g}dp"] = pou_basis(g, one)
                    Vs[f"PoU 1,xi H={h:g}dp"] = pou_basis(g, torch.cat([one, g.xi], -1))
                for (m, ck) in pous:
                    h = ck["args"]["hsub"]
                    g = PouGrid(s["pos"], s["dp"], h, device=dev)
                    Vs[f"PoU 1,xi+NN{m.k} H={h:g}dp [{os.path.basename(os.path.dirname(a.model[pous.index((m, ck))]))}]"] = pou_basis(g, m.fields(g, feat))
            line = []
            for name, V in Vs.items():
                t0 = time.time()
                C, pc = make_coarse(S, V)
                M = lambda r, C=C: cycle(S, r[:, None], torch.zeros_like(r)[:, None], C, sm)[:, 0]
                _, it, rel = pcg(S, M, maxit=3000)
                res.setdefault(name, []).append((it if rel <= 1 else np.nan, pc))
                line.append(f"{it}")
            print(f"  {os.path.basename(f)} n={s['n']} " + " ".join(line), flush=True)
        print(f"== {d.rstrip('/')}  ({len(files)} snapshots)")
        for name, v in res.items():
            it = np.array([x for x, _ in v], float); pc = np.mean([y for _, y in v])
            print(f"  {name:<44} iters={np.nanmean(it):6.1f}  coarse dim={pc:7.1f}  failed={int(np.isnan(it).sum())}", flush=True)
