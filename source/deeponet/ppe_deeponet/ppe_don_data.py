"""Reader for PPE snapshots written by donMaybeExport() in ppe_deeponet.inc.

A snapshot holds one pressure Poisson system A p = b of the ISPH solver,
restricted to the pressure unknowns (fluid particles with PpeFlag != 0):

    pos, xhat [n, sdim]  particle position / normalised position
    loc       [n, nloc]  local geometry descriptors (see donComputeGeometry)
    glob      [nglob]    global descriptors
    diag      [n]        operator diagonal D
    b, p, p0  [n]        right-hand side, converged pressure, warm start
    A                    CSR (rowptr, col, val), (A x)_i = sum_j A_ij x_j
    pure_neumann         1 -> the solver applies (I - 1 1^T/n) after A
"""
import glob as _glob
import os
import struct

import numpy as np
import torch

MAGIC = b"PPEDON01"


def read_snapshot(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != MAGIC:
        raise ValueError(f"{path}: not a PPEDON01 snapshot")
    off = 8
    sdim, nloc, nglob, n, pure_neumann, two_phase, solve_index, _ = struct.unpack_from("<8i", buf, off)
    off += 32
    (nnz,) = struct.unpack_from("<q", buf, off)
    off += 8

    def take(dtype, count):
        nonlocal off
        a = np.frombuffer(buf, dtype=dtype, count=count, offset=off)
        off += a.nbytes
        return a

    dp, dt, time, tol = take("<f8", 4)
    s = dict(path=path, sdim=sdim, nloc=nloc, nglob=nglob, n=n, nnz=nnz,
             pure_neumann=bool(pure_neumann), two_phase=bool(two_phase),
             solve_index=solve_index, dp=dp, dt=dt, time=time, tol=tol)
    s["glob"] = take("<f8", nglob).copy()
    s["pos"] = take("<f8", n * sdim).reshape(n, sdim).copy()
    s["xhat"] = take("<f8", n * sdim).reshape(n, sdim).copy()
    s["loc"] = take("<f8", n * nloc).reshape(n, nloc).copy()
    s["diag"] = take("<f8", n).copy()
    s["b"] = take("<f8", n).copy()
    s["p"] = take("<f8", n).copy()
    s["p0"] = take("<f8", n).copy()
    s["rowptr"] = take("<i8", n + 1).copy()
    s["col"] = take("<i4", nnz).astype(np.int64)
    s["val"] = take("<f8", nnz).copy()
    if off != len(buf):
        raise ValueError(f"{path}: {len(buf) - off} trailing bytes")
    return s


def find_snapshots(roots):
    """All ppe_*.bin below the given directories (or explicit files)."""
    files = []
    for r in roots:
        if os.path.isfile(r):
            files.append(r)
        else:
            files += sorted(_glob.glob(os.path.join(r, "**", "ppe_*.bin"), recursive=True))
    return files


def trajectory_of(path):
    """Snapshots are grouped by the directory they were exported to (= one run)."""
    return os.path.dirname(os.path.abspath(path))


class PpeSystem:
    """Torch view of one snapshot on a device (float64 operator)."""

    def __init__(self, s, device):
        self.meta = s
        self.n = s["n"]
        self.sdim = s["sdim"]
        self.pure_neumann = s["pure_neumann"]
        crow = torch.as_tensor(s["rowptr"], device=device)
        col = torch.as_tensor(s["col"], device=device)
        val = torch.as_tensor(s["val"], device=device, dtype=torch.float64)
        self.A = torch.sparse_csr_tensor(crow, col, val, size=(self.n, self.n))
        self.D = torch.as_tensor(s["diag"], device=device, dtype=torch.float64)
        self.b = torch.as_tensor(s["b"], device=device, dtype=torch.float64)
        self.p = torch.as_tensor(s["p"], device=device, dtype=torch.float64)
        self.p0 = torch.as_tensor(s["p0"], device=device, dtype=torch.float64)
        self.xhat = torch.as_tensor(s["xhat"], device=device, dtype=torch.float32)
        self.loc = torch.as_tensor(s["loc"], device=device, dtype=torch.float32)
        self.glob = torch.as_tensor(s["glob"], device=device, dtype=torch.float32)
        # constant null space (closed domain, no Dirichlet row): A 1 = 0
        one = torch.ones(self.n, dtype=torch.float64, device=device)
        self.null_const = bool((self.A @ one[:, None]).abs().max() <= 1e-10 * self.D.max())

    def matvec(self, x):
        """A x for x of shape [n] or [n, m] (float64), including the
        pure-Neumann projection used by applyPpeOperator()."""
        vec = x.dim() == 1
        X = x[:, None] if vec else x
        Y = self.A @ X
        if self.pure_neumann:
            Y = Y - Y.mean(dim=0, keepdim=True)
        return Y[:, 0] if vec else Y


def check_snapshot(path, device="cpu"):
    s = read_snapshot(path)
    S = PpeSystem(s, device)
    r = S.b - S.matvec(S.p)
    rel = (r.norm() / S.b.norm()).item()
    r0 = (S.b - S.matvec(S.p0)).norm().item() / S.b.norm().item()
    return dict(n=s["n"], nnz=s["nnz"], rel_residual=rel, warm_start_rel=r0,
                pure_neumann=s["pure_neumann"], two_phase=s["two_phase"])


if __name__ == "__main__":
    import sys
    for f in find_snapshots(sys.argv[1:]):
        print(os.path.basename(f), check_snapshot(f))
