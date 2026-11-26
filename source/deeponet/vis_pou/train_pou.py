"""Train the learned part of the PoU coarse space (same unsupervised A-norm two-grid loss as
../vis_deeponet/train_vis_don.py).

python3 train_pou.py --data FILES_OR_DIRS --val VAL_FILES --hsub 16 --k 3 --out DIR
"""
import argparse, json, math, os, random, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vis_deeponet"))
from vis_don_data import VisSystem, find_snapshots, read_snapshot
from vis_don_model import Smoother, estimate_lmax, two_level, galerkin_matrix, galerkin_coarse
from train_vis_don import make_probes
from pou_model import PouGrid, PouNet, pou_basis


def loss_of(model, s, S, args, gen):
    g = s["_grid"]
    feat = s["_feat"]
    V = pou_basis(g, model.fields(g, feat))
    C = galerkin_coarse(V, galerkin_matrix(V, S.matvec, args.reg))
    E0 = make_probes(S, args.white_probes, args.smooth_probes, args.sweeps, 0.7, gen)
    E = two_level(E0, S, C, s["_sm"])
    return (E * S.matvec(E)).sum(0).mean()


def prep(s, dev, hsub):
    S = VisSystem(s, dev)
    s["_grid"] = PouGrid(s["pos"], s["dp"], hsub, device=dev)
    s["_feat"] = torch.as_tensor(s["loc"][:, 1:], device=dev, dtype=torch.float64)
    s["_sm"] = Smoother(cheb_degree=6, cheb_ratio=30.0, lmax=estimate_lmax(S))
    return S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hsub", type=float, default=16.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--reg", type=float, default=1e-8)
    ap.add_argument("--white-probes", type=int, default=2)
    ap.add_argument("--smooth-probes", type=int, default=10)
    ap.add_argument("--sweeps", type=int, nargs="+", default=[16, 64, 256, 512])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    tr = [read_snapshot(f) for f in find_snapshots(args.data)]
    va = [read_snapshot(f) for f in find_snapshots(args.val)]
    trS = [prep(s, dev, args.hsub) for s in tr]
    vaS = [prep(s, dev, args.hsub) for s in va]
    print(f"train {len(tr)}  val {len(va)}  nodes(train first) {tr[0]['_grid'].nodes}", flush=True)
    model = PouNet(k_learned=args.k, width=args.width, latent=args.latent, seed=args.seed).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr), pct_start=0.05)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    best, hist, t0 = math.inf, [], time.time()
    for ep in range(args.epochs):
        order = list(range(len(tr))); random.shuffle(order); tl = []
        for i in order:
            L = loss_of(model, tr[i], trS[i], args, gen)
            if not torch.isfinite(L):
                continue
            opt.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl.append(L.item())
        with torch.no_grad():
            vg = torch.Generator(device=dev).manual_seed(1234)
            vl = float(np.mean([loss_of(model, s, S, args, vg).item() for s, S in zip(va, vaS)]))
        hist.append(dict(epoch=ep, train=float(np.mean(tl)), val=vl))
        print(f"ep {ep:3d} {time.time()-t0:6.1f}s train {np.mean(tl):.4e} val {vl:.4e}", flush=True)
        if vl < best:
            best = vl
            torch.save(dict(cfg=model.cfg, state=model.state_dict(), args=vars(args)), os.path.join(args.out, "best.pt"))
    json.dump(dict(args=vars(args), train=[s["path"] for s in tr], val=[s["path"] for s in va], history=hist),
              open(os.path.join(args.out, "history.json"), "w"), indent=1)
