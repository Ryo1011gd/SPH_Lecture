"""Torch replica of the solver's right-preconditioned BiCGStab and of the
DeepONet two-level preconditioner (donApply / donBiCGStab in ppe_deeponet.inc),
used for offline evaluation before running the SPH code."""
import torch

from ppe_don_model import PpeDeepONet, Smoother, cycle


def jacobi_preconditioner(sysm):
    D = sysm.D
    return lambda r: r / D


@torch.no_grad()
def deeponet_preconditioner(model, sysm, coarse="galerkin", smoother=None, npre=1, npost=1, reg=1e-10):
    """M^-1 r = cycle(r, 0) exactly as donApply()."""
    T, Psi, K = model.fields(sysm.xhat, sysm.loc, sysm.glob, need_branch=(coarse == "learned"))
    D = sysm.D
    if coarse == "galerkin":
        T = PpeDeepONet.galerkin_basis(T, D, sysm.null_const)
        Ac = PpeDeepONet.galerkin_factors(T, D, sysm.matvec, reg)
        Q, R = T @ torch.linalg.inv(Ac), T / D[:, None]
    else:
        n = D.shape[0]
        Q, R = PpeDeepONet.amp(n, model.sdim) / n * (T @ K), Psi / D[:, None]
    sm = smoother if smoother is not None else Smoother()
    pre = sm if npre else None
    post = sm if npost else None
    C = lambda r: Q @ (R.T @ r)

    def apply(r):
        z = cycle(sysm, r[:, None], torch.zeros_like(r)[:, None], C, pre, post)[:, 0]
        if sysm.null_const:            # remove the gauge mode (as donApply)
            z = z - z.mean()
        return z
    return apply


@torch.no_grad()
def bicgstab(sysm, precond, tol=1e-8, maxit=3000, x0=None, initial_guess=False):
    """Same recurrences and stopping test as calculatePpe(). Returns (x, iters, rel)."""
    A = sysm.matvec
    b = sysm.b
    x = (sysm.p0 if x0 is None else x0).clone()
    r = b - A(x)
    bb = torch.dot(b, b).item()
    rr = torch.dot(r, r).item()
    tol2 = tol * tol * bb
    if initial_guess:
        x1 = x + precond(r)
        r1 = b - A(x1)
        rr1 = torch.dot(r1, r1).item()
        if rr1 < rr:
            x, r, rr = x1, r1, rr1
    if rr <= tol2:
        return x, 0, (rr / bb) ** 0.5
    r0 = r.clone()
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)
    rho_old = alpha = omega = 1.0
    for it in range(maxit):
        rho = torch.dot(r0, r).item()
        if abs(rho) < 1e-60:
            break
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        ph = precond(p)
        v = A(ph)
        alpha = rho / torch.dot(r0, v).item()
        s = r - alpha * v
        ss = torch.dot(s, s).item()
        if ss <= tol2:
            x = x + alpha * ph
            return x, it + 1, (ss / bb) ** 0.5
        sh = precond(s)
        t = A(sh)
        tt = torch.dot(t, t).item()
        if not tt > 1e-60:
            break
        omega = torch.dot(t, s).item() / tt
        x = x + alpha * ph + omega * sh
        r = s - omega * t
        rr = torch.dot(r, r).item()
        if rr <= tol2:
            return x, it + 1, (rr / bb) ** 0.5
        rho_old = rho
    return x, maxit, (rr / bb) ** 0.5
