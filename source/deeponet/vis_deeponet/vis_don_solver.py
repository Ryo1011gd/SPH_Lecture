"""Torch replica of the viscosity CG (solveImplicitViscosity / visPcg in
vis_deeponet.inc) and of the DeepONet two-level preconditioner, for offline
evaluation before running the SPH code."""
import torch

from vis_don_model import Smoother, coarse_basis, cycle, estimate_lmax, galerkin_coarse, galerkin_matrix


def jacobi_preconditioner(sysm):
    D = sysm.D
    return lambda r: r / D


@torch.no_grad()
def deeponet_preconditioner(model, sysm, smoother=None, reg=1e-10):
    """M^-1 r = cycle(r, 0) exactly as visApply()."""
    V = coarse_basis(model.trunk(sysm.xhat, sysm.loc, sysm.glob), sysm.nc)
    C = galerkin_coarse(V, galerkin_matrix(V, sysm.matvec, reg))
    if smoother is None:
        smoother = Smoother(lmax=estimate_lmax(sysm))
    return lambda r: cycle(sysm, r[:, None], torch.zeros_like(r)[:, None], C, smoother)[:, 0]


@torch.no_grad()
def pcg(sysm, precond, maxit=20000, x0=None):
    """Same recurrences and stopping test (sum |r|^2/M <= target2, true residual
    every 50 iterations) as the solver. Returns (x, iters, res2/target2)."""
    A, b, t2 = sysm.matvec, sysm.b, sysm.target2
    x = (sysm.u0 if x0 is None else x0).clone()
    r = b - A(x)
    res2 = sysm.res2(r)
    if res2 <= t2:
        return x, 0, res2 / t2
    z = precond(r)
    p = z.clone()
    rz = torch.dot(r, z).item()
    for it in range(1, maxit + 1):
        ap = A(p)
        alpha = rz / torch.dot(p, ap).item()
        x = x + alpha * p
        r = r - alpha * ap
        res2 = sysm.res2(r)
        refresh = res2 <= t2 or it % 50 == 0
        if refresh:
            r = b - A(x)
            res2 = sysm.res2(r)
        if res2 <= t2:
            return x, it, res2 / t2
        z = precond(r)
        rz_new = torch.dot(r, z).item()
        beta = 0.0 if refresh else rz_new / rz
        p = z + beta * p
        rz = rz_new
    return x, maxit, res2 / t2
