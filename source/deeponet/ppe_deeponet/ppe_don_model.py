"""Particle-native, residual-linear DeepONet for preconditioning the ISPH PPE.

     enc_i  = [xhat_i, loc_i, sin(B xhat_i), cos(B xhat_i)]     multi-scale Fourier features
     z      = G( mean_i E(enc_i), glob )                         geometry latent (permutation invariant)
     T_ik   = trunk ([enc_i, z])   k = 1..P                      basis at query particle i
     Psi_ik = branch([enc_i, z])                                 integral kernel of the branch
     K      = Kmix(z)  (P x P)

Learned DeepONet coarse operator (PpeDonCoarse 0), LINEAR in the residual r:
     c      = (1/n) sum_i Psi_ik r_i / D_i                       branch = integral encoder over particles
     e_hat  = n^(2/d) T K c                                      n^(2/d): inverse-Laplacian scaling law

Galerkin coarse operator on the DeepONet trunk basis (PpeDonCoarse 1):
     e_hat  = T (T^T D^-1 A T)^-1 T^T D^-1 r                    coarse matrix from the CURRENT A

Both are embedded in a two-level cycle with damped Jacobi smoothing. The C++
inference in ppe_deeponet.inc evaluates exactly the same network from the
weights written by export_weights().
"""
import math
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

NLOC = 6
NGLOB = 8


class MLP(nn.Module):
    def __init__(self, sizes, act_last=False):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.act_last = act_last

    def acts(self):
        n = len(self.layers)
        return [1 if (i < n - 1 or self.act_last) else 0 for i in range(n)]

    def forward(self, x):
        for layer, a in zip(self.layers, self.acts()):
            x = layer(x)
            if a:
                x = F.silu(x)
        return x


class PpeDeepONet(nn.Module):
    def __init__(self, sdim=2, n_fourier=16, fourier_min=0.5, fourier_max=6.0,
                 width=64, depth=2, latent=32, basis=32, seed=0):
        super().__init__()
        self.cfg = dict(sdim=sdim, n_fourier=n_fourier, fourier_min=fourier_min, fourier_max=fourier_max,
                        width=width, depth=depth, latent=latent, basis=basis, seed=seed)
        self.sdim, self.P, self.Z = sdim, basis, latent
        g = torch.Generator().manual_seed(seed)
        # random directions, log-uniform wavenumbers (in units of pi / half-extent)
        d = torch.randn(n_fourier, sdim, generator=g)
        d = d / d.norm(dim=1, keepdim=True)
        k = torch.exp(torch.linspace(math.log(fourier_min), math.log(fourier_max), n_fourier))
        self.register_buffer("B", (math.pi * k[:, None] * d).to(torch.float32))
        nenc = sdim + NLOC + 2 * n_fourier
        self.nenc = nenc
        hid = [width] * depth
        self.E = MLP([nenc] + hid, act_last=True)
        self.G = MLP([width + NGLOB, width, latent])
        self.T = MLP([nenc + latent] + hid + [basis])
        self.R = MLP([nenc + latent] + hid + [basis])
        self.K = MLP([latent, basis * basis])
        # K starts at the identity: e_hat = n^(2/d) T T^T-like projection
        nn.init.zeros_(self.K.layers[0].weight)
        with torch.no_grad():
            self.K.layers[0].bias.copy_(torch.eye(basis).flatten() / basis)

    # ------------------------------------------------------------------
    def encode(self, xhat, loc):
        a = xhat @ self.B.T
        return torch.cat([xhat, loc, torch.sin(a), torch.cos(a)], dim=1)

    def latent(self, enc, glob):
        pool = self.E(enc).mean(dim=0)
        return self.G(torch.cat([pool, glob]))

    def fields(self, xhat, loc, glob, query=None, need_branch=True):
        """Returns trunk T [n or |query|, P], branch Psi [n, P], K [P, P] (float64)."""
        enc = self.encode(xhat, loc)
        z = self.latent(enc, glob)
        inp = torch.cat([enc, z.expand(enc.shape[0], -1)], dim=1)
        T = self.T(inp if query is None else inp[query]).double()
        Psi = self.R(inp).double() if need_branch else None
        K = self.K(z).view(self.P, self.P).double()
        return T, Psi, K

    # ------------------------------------------------------------------
    @staticmethod
    def amp(n, sdim):
        return float(n) ** (2.0 / sdim)

    def learned_coarse(self, T, Psi, K, D, R_in, Tq=None):
        """e_hat = n^(2/d) T K (Psi^T D^-1 r) / n   for r of shape [n, m]."""
        n = D.shape[0]
        c = Psi.T @ (R_in / D[:, None]) / n
        Tout = T if Tq is None else Tq
        return self.amp(n, self.sdim) * (Tout @ (K @ c))

    @staticmethod
    def galerkin_basis(T, D, null_const):
        """Pure-Neumann systems (A 1 = 0): remove the D^-1-weighted mean of every
        basis vector, so the constant (pressure gauge) is not in the coarse space
        and T^T D^-1 A T stays non-singular.  Same as donSetupModel()."""
        if not null_const:
            return T
        w = 1.0 / D
        return T - ((w[:, None] * T).sum(0) / w.sum())[None, :]

    @staticmethod
    def galerkin_factors(T, D, matvec, reg=1e-10):
        """T must already be the output of galerkin_basis()."""
        AT = matvec(T)
        Ac = (T / D[:, None]).T @ AT
        dmax = Ac.diagonal().abs().max()
        Ac = Ac + reg * dmax * torch.eye(Ac.shape[0], dtype=Ac.dtype, device=Ac.device)
        return Ac

    @staticmethod
    def galerkin_coarse(T, D, Ac, R_in):
        return T @ torch.linalg.solve(Ac, (T / D[:, None]).T @ R_in)


# ----------------------------------------------------------------------
class Smoother:
    """Damped Jacobi (cheb_degree=0) or Jacobi-preconditioned Chebyshev on
    [lmax/ratio, lmax] of D^-1 A.  Identical recurrences in ppe_deeponet.inc."""

    def __init__(self, omega=0.7, sweeps=1, cheb_degree=0, cheb_ratio=15.0, lmax=None):
        self.omega, self.sweeps, self.deg, self.ratio, self.lmax = omega, sweeps, cheb_degree, cheb_ratio, lmax

    def __call__(self, sysm, z, r):
        """Smooth A z = r starting from z (r, z: [n, m])."""
        D = sysm.D[:, None]
        if self.deg <= 0:
            for _ in range(self.sweeps):
                z = z + self.omega * (r - sysm.matvec(z)) / D
            return z
        b = self.lmax
        a = b / self.ratio
        theta, delta = 0.5 * (b + a), 0.5 * (b - a)
        sigma = theta / delta
        rho = 1.0 / sigma
        d = (r - sysm.matvec(z)) / D / theta
        z = z + d
        for _ in range(self.deg - 1):
            rho1 = 1.0 / (2.0 * sigma - rho)
            d = rho1 * rho * d + (2.0 * rho1 / delta) * (r - sysm.matvec(z)) / D
            z = z + d
            rho = rho1
        return z


@torch.no_grad()
def estimate_lmax(sysm, iters=15, safety=1.15):
    """Power iteration for the largest eigenvalue of D^-1 A (same as C++)."""
    n = sysm.n
    i = torch.arange(n, dtype=torch.float64, device=sysm.D.device)
    x = torch.sin(0.731 * i + 0.3) + 0.5 * torch.cos(0.1137 * i)
    lam = 1.0
    for _ in range(iters):
        x = x / x.norm()
        y = sysm.matvec(x) / sysm.D
        lam = torch.dot(x, y).item()
        x = y
    return safety * lam


def cycle(sysm, r, z, coarse, pre, post):
    """Two-level cycle on A z = r: pre-smooth, coarse correction, post-smooth."""
    if pre is not None:
        z = pre(sysm, z, r)
    z = z + coarse(r - sysm.matvec(z))
    if post is not None:
        z = post(sysm, z, r)
    return z


def two_level(e, sysm, coarse, pre, post):
    """Error propagation of the cycle: z* = 0 for r = 0, so e_new = cycle(0, e)."""
    return cycle(sysm, torch.zeros_like(e), e, coarse, pre, post)


# ----------------------------------------------------------------------
def export_weights(model, path):
    """Binary format read by donLoadModel() (ppe_deeponet.inc)."""
    m = model
    with open(path, "wb") as f:
        f.write(b"DONPPE01")
        f.write(struct.pack("<8i", m.sdim, NLOC, NGLOB, m.B.shape[0], m.cfg["width"], m.Z, m.P, 5))
        f.write(m.B.detach().double().cpu().numpy().astype("<f8").tobytes())
        for net in (m.E, m.G, m.T, m.R, m.K):
            f.write(struct.pack("<i", len(net.layers)))
            for layer, a in zip(net.layers, net.acts()):
                W = layer.weight.detach().double().cpu().numpy()
                bvec = layer.bias.detach().double().cpu().numpy()
                f.write(struct.pack("<3i", W.shape[1], W.shape[0], a))
                f.write(W.astype("<f8").tobytes())
                f.write(bvec.astype("<f8").tobytes())


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = PpeDeepONet(**ck["cfg"]).to(device)
    model.load_state_dict(ck["state"])
    return model, ck
