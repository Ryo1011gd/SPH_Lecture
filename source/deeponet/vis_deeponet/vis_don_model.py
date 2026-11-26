"""Particle-native DeepONet coarse space for the implicit-viscosity solve.

     enc_i  = [xhat_i, loc_i, sin(B xhat_i), cos(B xhat_i)]     multi-scale Fourier features
     z      = G( mean_i E(enc_i), glob )                         geometry/stiffness latent
     T_ik   = trunk([enc_i, z])   k = 1..P                       scalar basis fields

Coarse space  V = T (x) I_ncomp   (column k*ncomp+d = T_k e_d), Galerkin on the CURRENT operator:

     e_hat  = V (V^T A V)^-1 V^T r

embedded in a symmetric two-level cycle (same Chebyshev/Jacobi smoother before and
after), so M^-1 is SPD and the solver keeps plain CG.  vis_deeponet.inc evaluates
exactly the same network from the weights written by export_weights().
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


class VisDeepONet(nn.Module):
    def __init__(self, sdim=2, n_fourier=16, fourier_min=0.5, fourier_max=10.0,
                 width=64, depth=2, latent=32, basis=64, seed=0):
        super().__init__()
        self.cfg = dict(sdim=sdim, n_fourier=n_fourier, fourier_min=fourier_min, fourier_max=fourier_max,
                        width=width, depth=depth, latent=latent, basis=basis, seed=seed)
        self.sdim, self.P, self.Z = sdim, basis, latent
        g = torch.Generator().manual_seed(seed)
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

    def encode(self, xhat, loc):
        a = xhat @ self.B.T
        return torch.cat([xhat, loc, torch.sin(a), torch.cos(a)], dim=1)

    def trunk(self, xhat, loc, glob):
        """Scalar basis T [n, P] (float64)."""
        enc = self.encode(xhat, loc)
        z = self.G(torch.cat([self.E(enc).mean(dim=0), glob]))
        inp = torch.cat([enc, z.expand(enc.shape[0], -1)], dim=1)
        return self.T(inp).double()


# ----------------------------------------------------------------------
def coarse_basis(T, nc):
    """V[(i,d),(k,d')] = T_ik delta_dd'  ->  [n*nc, P*nc], same ordering as vis_deeponet.inc."""
    return torch.kron(T, torch.eye(nc, dtype=T.dtype, device=T.device))


def galerkin_matrix(V, matvec, reg=1e-10):
    Ac = V.T @ matvec(V)
    Ac = 0.5 * (Ac + Ac.T)
    dmax = Ac.diagonal().abs().max()
    return Ac + reg * dmax * torch.eye(Ac.shape[0], dtype=Ac.dtype, device=Ac.device)


def galerkin_coarse(V, Ac):
    """r -> V Ac^-1 V^T r  (r: [m, k])."""
    return lambda R: V @ torch.linalg.solve(Ac, V.T @ R)


class Smoother:
    """Damped Jacobi (cheb_degree=0) or Jacobi-preconditioned Chebyshev on
    [lmax/ratio, lmax] of D^-1 A.  Identical recurrences in vis_deeponet.inc."""

    def __init__(self, omega=0.7, sweeps=1, cheb_degree=6, cheb_ratio=30.0, lmax=None):
        self.omega, self.sweeps, self.deg, self.ratio, self.lmax = omega, sweeps, cheb_degree, cheb_ratio, lmax

    def __call__(self, sysm, z, r):
        D = sysm.D[:, None]
        if self.deg <= 0:
            for _ in range(max(1, self.sweeps)):
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
    i = torch.arange(sysm.m, dtype=torch.float64, device=sysm.D.device)
    x = torch.sin(0.731 * i + 0.3) + 0.5 * torch.cos(0.1137 * i)
    lam = 1.0
    for _ in range(iters):
        x = x / x.norm()
        y = sysm.matvec(x) / sysm.D
        lam = torch.dot(x, y).item()
        x = y
    return safety * lam


def cycle(sysm, r, z, coarse, smoother):
    """Symmetric two-level cycle on A z = r."""
    z = smoother(sysm, z, r)
    z = z + coarse(r - sysm.matvec(z))
    return smoother(sysm, z, r)


def two_level(e, sysm, coarse, smoother):
    """Error propagation of the cycle: e_new = cycle(r=0, z=e)."""
    return cycle(sysm, torch.zeros_like(e), e, coarse, smoother)


# ----------------------------------------------------------------------
def export_weights(model, path):
    """Binary format read by visLoadModel() (vis_deeponet.inc)."""
    m = model
    with open(path, "wb") as f:
        f.write(b"DONVIS01")
        f.write(struct.pack("<8i", m.sdim, NLOC, NGLOB, m.B.shape[0], m.cfg["width"], m.Z, m.P, 3))
        f.write(m.B.detach().double().cpu().numpy().astype("<f8").tobytes())
        for net in (m.E, m.G, m.T):
            f.write(struct.pack("<i", len(net.layers)))
            for layer, a in zip(net.layers, net.acts()):
                W = layer.weight.detach().double().cpu().numpy()
                bvec = layer.bias.detach().double().cpu().numpy()
                f.write(struct.pack("<3i", W.shape[1], W.shape[0], a))
                f.write(W.astype("<f8").tobytes())
                f.write(bvec.astype("<f8").tobytes())


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = VisDeepONet(**ck["cfg"]).to(device)
    model.load_state_dict(ck["state"])
    return model, ck
