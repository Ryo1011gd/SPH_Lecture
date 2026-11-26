"""Offline test of two training-free coarse-space enrichments for the viscosity
two-level preconditioner (same cycle as vis_deeponet.inc: Cheb6 / Galerkin / Cheb6):

  (2) rigid-body modes of the stiff (solidified) clusters, detected per snapshot
      from the stiffness feature s_i and the matrix graph; optionally the cluster
      is cut into square chunks of edge `chunk` dp (piecewise-rigid modes);
  (3) Krylov recycling: the smallest Ritz vectors of B A from the previous PCG
      solve (Lanczos tridiagonal of PCG), mapped to the new particles by nearest
      neighbour, are appended to the coarse space.

usage: python3 ideas_coarse.py SNAPDIR [SNAPDIR ...]   (snapshots in solve order)
"""
import os, sys, time
import numpy as np, scipy.sparse as sp, torch
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vis_deeponet"))
from vis_don_data import VisSystem, find_snapshots, read_snapshot
from vis_don_model import Smoother, coarse_basis, cycle, estimate_lmax, load_checkpoint

torch.set_num_threads(int(os.environ.get("NTHREADS", "16")))
dev = torch.device(os.environ.get("DEVICE", "cpu"))
model, _ = load_checkpoint(os.path.join(HERE, "..", "vis_deeponet", "trained_model", "best.pt"), dev)
model.eval()
NRITZ = int(os.environ.get("NRITZ", "16"))     # Ritz vectors kept from one solve
RMAX = int(os.environ.get("RMAX", "32"))       # recycled space cap
STIFF_THR = 30.0                               # s_i > 30 x (5th percentile) -> solid


def make_coarse(S, V):
    """Orthonormalise V and return r -> V Ac^+ V^T r (pseudo-inverse drops null directions)."""
    Q, _ = torch.linalg.qr(V)
    Ac = Q.T @ S.matvec(Q)
    Ac = 0.5 * (Ac + Ac.T)
    w, U = torch.linalg.eigh(Ac)
    keep = w > 1e-12 * w.max()
    U, w = U[:, keep], w[keep]
    QU = Q @ U
    return (lambda R: QU @ ((QU.T @ R) / w[:, None])), QU.shape[1]


def rigid_modes(s, chunk=None):
    n, pos, dp = s["n"], s["pos"], s["dp"]
    st = np.expm1(10 * s["loc"][:, 0])
    idx = np.where(st > STIFF_THR * np.percentile(st, 5))[0]
    if len(idx) == 0:
        return np.zeros((2 * n, 0))
    rows = np.repeat(np.arange(s["m"]), np.diff(s["rowptr"]))
    G = sp.coo_matrix((np.ones(len(rows)), (rows // 2, s["col"] // 2)), shape=(n, n)).tocsr()
    _, lab = connected_components(G[idx][:, idx], directed=False)
    if chunk:   # cut each cluster into square chunks of `chunk` dp
        cell = np.floor(pos[idx] / (chunk * dp)).astype(np.int64)
        key = lab * 10**8 + (cell[:, 0] + 50000) * 10**4 + (cell[:, 1] + 5000)
        _, lab = np.unique(key, return_inverse=True)
    cols = []
    for c in range(lab.max() + 1):
        p = idx[lab == c]
        if len(p) < 10:
            continue
        ctr = pos[p].mean(0)
        for mode in range(3):
            v = np.zeros(2 * n)
            if mode < 2:
                v[2 * p + mode] = 1.0
            else:
                v[2 * p] = -(pos[p, 1] - ctr[1]) / dp
                v[2 * p + 1] = (pos[p, 0] - ctr[0]) / dp
            cols.append(v / np.linalg.norm(v))
    return np.stack(cols, 1) if cols else np.zeros((2 * n, 0))


@torch.no_grad()
def pcg_lanczos(S, precond, maxit=3000, nritz=0):
    """pcg() of vis_don_solver.py + Lanczos bookkeeping -> smallest Ritz vectors of B A."""
    A, b, t2 = S.matvec, S.b, S.target2
    x = S.u0.clone(); r = b - A(x); res2 = S.res2(r)
    if res2 <= t2:
        return 0, None
    z = precond(r); p = z.clone(); rz = torch.dot(r, z).item()
    Z, al, be = [], [], []
    rec = nritz > 0
    for it in range(1, maxit + 1):
        if rec and not rz > 0:
            rec = False; nneg[0] += 1                # B not SPD here (e.g. lmax underestimated)
        if rec:
            Z.append(((-1) ** (it - 1)) * z / np.sqrt(rz))
        ap = A(p); alpha = rz / torch.dot(p, ap).item()
        x = x + alpha * p; r = r - alpha * ap; res2 = S.res2(r)
        refresh = res2 <= t2 or it % 50 == 0
        if refresh:
            r = b - A(x); res2 = S.res2(r)
        if res2 <= t2:
            break
        z = precond(r); rz_new = torch.dot(r, z).item()
        beta = 0.0 if refresh else rz_new / rz
        if rec:
            al.append(alpha); be.append(beta)
            if refresh or len(Z) >= 200 or not (beta >= 0 and rz_new > 0):
                rec = False                      # Lanczos relation broken by the restart
        p = z + beta * p; rz = rz_new
    if nritz == 0 or len(al) < 2:
        return it, None
    k = min(len(al), len(Z) - 1)
    al, be = al[:k], be[:k]
    if k < 2:
        return it, None
    al, be = np.array(al), np.array(be)
    T = np.zeros((k, k))
    for j in range(k):
        T[j, j] = 1 / al[j] + (be[j - 1] / al[j - 1] if j else 0.0)
        if j + 1 < k:
            T[j, j + 1] = T[j + 1, j] = np.sqrt(be[j]) / al[j]
    th, Y = np.linalg.eigh(T)
    q = min(nritz, k)
    W = torch.stack(Z[:k], 1) @ torch.as_tensor(Y[:, :q], device=dev)
    W = W[:, torch.isfinite(W).all(0)]
    return it, W


nneg = [0]


def map_to(W, pos_old, pos_new):
    """Nearest-neighbour transfer of per-particle 2-vectors."""
    j = cKDTree(pos_old).query(pos_new)[1]
    Wp = W.reshape(len(pos_old), 2, -1)[torch.as_tensor(j, device=W.device)]
    return Wp.reshape(2 * len(pos_new), -1)


VARIANTS = [  # name, NN, rigid(chunk or None/False), recycle
    ("DON (baseline)",            True,  False, False),
    ("DON + rigid(cluster)",      True,  None,  False),
    ("DON + rigid(chunk 16dp)",   True,  16,    False),
    ("DON + recycle",             True,  False, True),
    ("DON + rigid(chunk)+recycle", True, 16,    True),
    ("Cheb + rigid(chunk)",       False, 16,    False),
    ("Cheb + rigid(chunk)+recycle", False, 16,  True),
]

if __name__ == "__main__":
    for d in sys.argv[1:]:
        files = find_snapshots([d])
        res = {v[0]: [] for v in VARIANTS}
        rec_state = {v[0]: None for v in VARIANTS}
        for f in files:
            s = read_snapshot(f); S = VisSystem(s, dev); nnz = s["nnz"]
            lmax = estimate_lmax(S); sm = Smoother(cheb_degree=6, cheb_ratio=30.0, lmax=lmax)
            with torch.no_grad():
                Vnn = coarse_basis(model.trunk(S.xhat, S.loc, S.glob), S.nc)
            Rg = {c: torch.as_tensor(rigid_modes(s, c), device=dev) for c in (None, 16)}
            for name, nn, rig, rcy in VARIANTS:
                if os.environ.get("ONLY") and os.environ["ONLY"] not in name:
                    continue
                parts = [Vnn] if nn else []
                if rig is not False:
                    parts.append(Rg[rig])
                st = rec_state[name]
                if rcy and st is not None:
                    parts.append(map_to(st[0], st[1], s["pos"]))
                V = torch.cat(parts, 1)
                C, pc = make_coarse(S, V)
                M = lambda r, C=C: cycle(S, r[:, None], torch.zeros_like(r)[:, None], C, sm)[:, 0]
                it, W = pcg_lanczos(S, M, nritz=NRITZ if rcy else 0)
                if rcy and W is not None:
                    old = map_to(st[0], st[1], s["pos"]) if st is not None else W[:, :0]
                    U = torch.cat([W, old], 1)[:, :RMAX]
                    rec_state[name] = (U, s["pos"].copy())
                wu = 13 + 2 * S.m * pc / nnz
                res[name].append((it, wu, pc))
            res = {k: v for k, v in res.items() if v}
            print(f"  [rz<=0 events so far: {nneg[0]}]", end="")
            print(f"  {os.path.basename(f)} n={s['n']} " +
                  " ".join(f"{k[:12]}:{v[-1][0]}" for k, v in res.items()), flush=True)
        print(f"== {d.rstrip('/')}  ({len(files)} snapshots; first snapshot has no recycled space)")
        for name, v in res.items():
            it = np.array([a for a, _, _ in v], float); wu = np.mean([b for _, b, _ in v]); pc = np.mean([c for _, _, c in v])
            it2 = it[1:].mean() if len(it) > 1 else np.nan
            print(f"  {name:<30} iters={it.mean():6.1f} (excl. 1st {it2:6.1f})  coarse dim={pc:6.1f}  WU/apply={wu:6.2f}  total WU={it.mean() * (1 + wu):7.0f}", flush=True)
