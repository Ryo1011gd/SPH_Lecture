"""Partition-of-unity (PoU) learned coarse space for the implicit-viscosity solve.

The global DeepONet basis of ../vis_deeponet has a fixed number of columns (P = 64 scalar
fields), so the two-level iteration count doubles every time dp is halved. Here the
coarse space is built from LOCAL fields glued by a partition of unity:

    coarse grid   nodes x_j on a square lattice of spacing H = h_sub * dp   (h_sub fixed in dp
                  units -> number of nodes ~ n, H/dp constant -> resolution-independent)
    phi_j(x)      bilinear hat functions, renormalised over the kept nodes  (sum_j phi_j = 1)
    xi_ij         (x_i - x_j) / H in [-1, 1]^2                               local coordinate
    fields        f_0 = 1, f_1 = xi_x, f_2 = xi_y                          (fixed, poly part)
                  f_3.. = NN([enc(xi_ij), feat_i, z_j])                       (learned, k fields)
                  z_j = G(sum_i phi_ij E(enc_i) / sum_i phi_ij)            node-local context
    V             column (j, k, d) = phi_j(x_i) f_k(i, j) e_d               (sparse: <= 4 nodes/row)

Every input is measured in dp units or is a relative quantity, so a model trained at one
resolution applies unchanged at another. Rigid motions of each node patch, (-y, x),
are in the span through xi (x) I. The cycle is the same symmetric Chebyshev / Galerkin /
Chebyshev cycle as vis_deeponet, so CG stays valid.

feat_i = loc_i[1:] of the snapshot (relative stiffness, fluid / wall kernel sums, wall
distance / radius, log D_i/<D>); the absolute stiffness loc_i[0] ~ log(Dt mu / rho dp^2)
changes with dp and Dt and is therefore left out.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NFEAT = 5            # loc[1:6]
NPOLY = 3            # 1, xi_x, xi_y


class MLP(nn.Module):
    def __init__(self, sizes, act_last=False):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.act_last = act_last

    def forward(self, x):
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < n - 1 or self.act_last:
                x = F.silu(x)
        return x


class PouGrid:
    """Coarse lattice, kept nodes, per-particle (node index, weight) for up to 4 nodes."""

    def __init__(self, pos, dp, h_sub, min_fill=0.1, device="cpu"):
        pos = torch.as_tensor(pos, dtype=torch.float64, device=device)
        H = h_sub * dp
        lo = pos.min(0).values - 1e-9
        g = (pos - lo) / H                                  # lattice coordinates
        c = torch.floor(g).long()
        t = g - c
        nx = int(c[:, 0].max()) + 2
        ny = int(c[:, 1].max()) + 2
        nid, w = [], []
        for ox in (0, 1):
            for oy in (0, 1):
                nid.append((c[:, 0] + ox) * ny + (c[:, 1] + oy))
                wx = t[:, 0] if ox else 1 - t[:, 0]
                wy = t[:, 1] if oy else 1 - t[:, 1]
                w.append(wx * wy)
        nid, w = torch.stack(nid, 1), torch.stack(w, 1)     # [n, 4]
        fill = torch.zeros(nx * ny, dtype=torch.float64, device=device).index_add_(0, nid.reshape(-1), w.reshape(-1))
        keep = fill >= min_fill * h_sub ** 2                 # full interior node: fill = h_sub^2
        w = w * keep[nid]
        s = w.sum(1, keepdim=True)
        orphan = s[:, 0] <= 0                                # all 4 nodes dropped -> best node, weight 1
        if orphan.any():
            j = torch.argmax(fill[nid[orphan]], 1)
            nid_o = nid[orphan].gather(1, j[:, None])
            keep[nid_o[:, 0]] = True
            w[orphan] = 0.0
            w[orphan] = w[orphan].scatter(1, j[:, None], 1.0)
            s = w.sum(1, keepdim=True)
        w = w / s
        remap = torch.full((nx * ny,), -1, dtype=torch.long, device=device)
        kept = torch.nonzero(keep)[:, 0]
        remap[kept] = torch.arange(len(kept), device=device)
        self.nodes = len(kept)
        self.node = remap[nid]                               # [n, 4], -1 where weight 0
        self.w = w
        node_xy = torch.stack([kept // ny, kept % ny], 1).double() * H + lo
        self.xi = ((pos[:, None, :] - node_xy[self.node.clamp_min(0)]) / H)   # [n, 4, 2]
        self.valid = (self.node >= 0) & (w > 0)
        self.n = pos.shape[0]


class PouNet(nn.Module):
    def __init__(self, k_learned=3, width=48, depth=2, latent=16, n_fourier=4, seed=0):
        super().__init__()
        self.cfg = dict(k_learned=k_learned, width=width, depth=depth, latent=latent, n_fourier=n_fourier, seed=seed)
        self.k = k_learned
        g = torch.Generator().manual_seed(seed)
        d = torch.randn(n_fourier, 2, generator=g)
        d = d / d.norm(dim=1, keepdim=True)
        kk = torch.linspace(0.5, 2.0, n_fourier)
        self.register_buffer("B", (math.pi * kk[:, None] * d).double())
        nenc = 2 + 2 * n_fourier + NFEAT
        hid = [width] * depth
        self.E = MLP([nenc] + hid, act_last=True)
        self.G = MLP([width, width, latent])
        self.T = MLP([nenc + latent] + hid + [k_learned]) if k_learned > 0 else None

    def encode(self, xi, feat):
        a = xi @ self.B.T
        return torch.cat([xi, torch.sin(a), torch.cos(a), feat], -1)

    def fields(self, grid, feat):
        """f [n, 4, NPOLY + k] (float64): fields of every particle on each of its 4 nodes."""
        n = grid.n
        one = torch.ones(n, 4, 1, dtype=torch.float64, device=grid.w.device)
        poly = torch.cat([one, grid.xi], -1)
        if self.k == 0:
            return poly
        f4 = feat[:, None, :].expand(n, 4, NFEAT)
        enc = self.encode(grid.xi, f4).float()                         # [n, 4, nenc]
        h = self.E(enc)                                                # [n, 4, W]
        wv = (grid.w * grid.valid).float()
        idx = grid.node.clamp_min(0).reshape(-1)
        num = torch.zeros(grid.nodes, h.shape[-1], device=h.device).index_add_(0, idx, (h * wv[..., None]).reshape(-1, h.shape[-1]))
        den = torch.zeros(grid.nodes, device=h.device).index_add_(0, idx, wv.reshape(-1)).clamp_min(1e-12)
        z = self.G(num / den[:, None])                                 # [nodes, latent]
        zi = z[grid.node.clamp_min(0)]                                 # [n, 4, latent]
        learned = self.T(torch.cat([enc, zi], -1)).double()
        return torch.cat([poly, learned], -1)


def pou_basis(grid, f, nc=2, scale=True):
    """Dense V [n*nc, nodes*K*nc] with column (j*K + k)*nc + d (prototype; production would keep it
    block-sparse: <= 4 node blocks per row)."""
    n, K = grid.n, f.shape[-1]
    vals = (grid.w * grid.valid)[..., None] * f                        # [n, 4, K]
    Vs = torch.zeros(n, grid.nodes * K, dtype=torch.float64, device=f.device)
    cols = grid.node.clamp_min(0)[..., None] * K + torch.arange(K, device=f.device)
    Vs = Vs.scatter_add(1, cols.reshape(n, -1), vals.reshape(n, -1))
    if scale:
        Vs = Vs / Vs.norm(dim=0).clamp_min(1e-30)
    return torch.kron(Vs, torch.eye(nc, dtype=Vs.dtype, device=Vs.device))
