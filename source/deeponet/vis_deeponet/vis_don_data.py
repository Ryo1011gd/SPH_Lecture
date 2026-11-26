"""Reader for implicit-viscosity snapshots written by visDonMaybeExport() in vis_deeponet.inc.

A snapshot holds one backward-Euler viscosity system  (M + Dt K) u = b  of the
ISPH solver, restricted to the fluid particles and the in-plane components
(row/column (a, d) -> a*ncomp + d):

    pos, xhat [n, sdim]   particle position / normalised position
    loc       [n, nloc]   local descriptors (see visComputeGeometry)
    glob      [nglob]     global descriptors
    mass      [n]         particle mass (the solver's residual norm is sum |r_i|^2 / M_i)
    diag      [m]         point-Jacobi diagonal D, m = n*ncomp
    b, u, u0  [m]         right-hand side, converged velocity, warm start (old velocity)
    A                     CSR (rowptr, col, val), symmetric positive definite
"""
import glob as _glob
import os
import struct

import numpy as np
import torch

MAGIC = b"VISDON01"


def read_snapshot(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != MAGIC:
        raise ValueError(f"{path}: not a VISDON01 snapshot")
    off = 8
    sdim, nloc, nglob, n, ncomp, solve_index, iters, _ = struct.unpack_from("<8i", buf, off)
    off += 32
    (nnz,) = struct.unpack_from("<q", buf, off)
    off += 8

    def take(dtype, count):
        nonlocal off
        a = np.frombuffer(buf, dtype=dtype, count=count, offset=off)
        off += a.nbytes
        return a

    dp, dt, time, tol, abstol = take("<f8", 5)
    m = n * ncomp
    s = dict(path=path, sdim=sdim, nloc=nloc, nglob=nglob, n=n, ncomp=ncomp, m=m, nnz=nnz,
             solve_index=solve_index, solver_iter=iters, dp=dp, dt=dt, time=time, tol=tol, abstol=abstol)
    s["glob"] = take("<f8", nglob).copy()
    s["pos"] = take("<f8", n * sdim).reshape(n, sdim).copy()
    s["xhat"] = take("<f8", n * sdim).reshape(n, sdim).copy()
    s["loc"] = take("<f8", n * nloc).reshape(n, nloc).copy()
    s["mass"] = take("<f8", n).copy()
    s["diag"] = take("<f8", m).copy()
    s["b"] = take("<f8", m).copy()
    s["u"] = take("<f8", m).copy()
    s["u0"] = take("<f8", m).copy()
    s["rowptr"] = take("<i8", m + 1).copy()
    s["col"] = take("<i4", nnz).astype(np.int64)
    s["val"] = take("<f8", nnz).copy()
    if off != len(buf):
        raise ValueError(f"{path}: {len(buf) - off} trailing bytes")
    return s


def find_snapshots(roots):
    """All vis_*.bin below the given directories (or explicit files)."""
    files = []
    for r in roots:
        if os.path.isfile(r):
            files.append(r)
        else:
            files += sorted(_glob.glob(os.path.join(r, "**", "vis_*.bin"), recursive=True))
    return files


def trajectory_of(path):
    """Snapshots are grouped by the directory they were exported to (= one run)."""
    return os.path.dirname(os.path.abspath(path))


class VisSystem:
    """Torch view of one snapshot on a device (float64 operator)."""

    def __init__(self, s, device):
        self.meta = s
        self.n, self.nc, self.m = s["n"], s["ncomp"], s["m"]
        self.sdim = s["sdim"]
        crow = torch.as_tensor(s["rowptr"], device=device)
        col = torch.as_tensor(s["col"], device=device)
        val = torch.as_tensor(s["val"], device=device, dtype=torch.float64)
        self.A = torch.sparse_csr_tensor(crow, col, val, size=(self.m, self.m))
        f64 = dict(device=device, dtype=torch.float64)
        self.D = torch.as_tensor(s["diag"], **f64)
        self.b = torch.as_tensor(s["b"], **f64)
        self.u = torch.as_tensor(s["u"], **f64)
        self.u0 = torch.as_tensor(s["u0"], **f64)
        self.mass = torch.as_tensor(s["mass"], **f64)
        self.w = (1.0 / self.mass).repeat_interleave(self.nc)       # residual-norm weights per row
        self.xhat = torch.as_tensor(s["xhat"], device=device, dtype=torch.float32)
        self.loc = torch.as_tensor(s["loc"], device=device, dtype=torch.float32)
        self.glob = torch.as_tensor(s["glob"], device=device, dtype=torch.float32)
        # same stopping test as solveImplicitViscosity()
        self.target2 = (s["tol"] ** 2 * torch.sum(self.b * self.b * self.w).item()
                        + s["abstol"] ** 2 * self.mass.sum().item())

    def matvec(self, x):
        """A x for x of shape [m] or [m, k] (float64)."""
        if x.dim() == 1:
            return (self.A @ x[:, None])[:, 0]
        return self.A @ x

    def res2(self, r):
        return torch.sum(r * r * self.w).item()


def check_snapshot(path, device="cpu"):
    s = read_snapshot(path)
    S = VisSystem(s, device)
    r = S.b - S.matvec(S.u)
    r0 = S.b - S.matvec(S.u0)
    return dict(n=s["n"], nnz=s["nnz"], solver_iter=s["solver_iter"],
                res2_over_target=S.res2(r) / S.target2, warm_start_res2_over_target=S.res2(r0) / S.target2,
                stiffness_max=float(np.expm1(10.0 * s["glob"][4])))


if __name__ == "__main__":
    import sys
    for f in find_snapshots(sys.argv[1:]):
        print(os.path.basename(f), check_snapshot(f))
