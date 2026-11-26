# DeepONet two-level preconditioner for the implicit viscosity solve

The backward-Euler viscosity step in `main_implicit.cpp` (`solveImplicitViscosity`) solves

```
(M + Dt K) u = b        SPD, unknowns = fluid particles x velocity components
```

It uses Jacobi-preconditioned CG to `sum_i |r_i|^2/M_i <= tol^2 sum_i |b_i|^2/M_i + abstol^2 sum_i M_i`.
When the melt solidifies, the thermal viscosity rises by orders of magnitude and `Dt K` dominates `M`.
The slow modes are then the near-rigid motions of the solid regions, which Jacobi cannot remove.
On T05 (freezing collapse) the iteration count grows from about 20 to more than 1000.

This module is the viscosity counterpart of `../ppe_deeponet`. **The tolerance, the stopping test and the
solution are unchanged.** The network only supplies a coarse space. If anything goes wrong, the original
Jacobi CG finishes the solve.

## Method

```
M⁻¹ r :  z  = S(r; 0)                              pre-smoothing  (Jacobi-preconditioned Chebyshev on D⁻¹A)
         z += V (Vᵀ A V)⁻¹ Vᵀ (r − A z)            Galerkin coarse correction, CURRENT A
         z  = S(r; z)                              post-smoothing (same polynomial)
V = T ⊗ I : column k·sdim+d = T_k e_d               T = P scalar DeepONet trunk fields at the particles
```

* **Symmetric, so CG stays valid.** The cycle uses the same smoother before and after and a symmetric
  Galerkin correction, so `M⁻¹` is SPD. This differs from the PPE, which uses BiCGStab and a
  Petrov–Galerkin `D⁻¹` weighting.
* **Coupled components.** The central-pair viscosity `c_ij e eᵀ` couples x and y. The stored operator keeps
  the full DIM×DIM diagonal block and `c, e` per fluid-neighbour slot. Every scalar field `T_k` enters
  once for each component, so rigid translations and rotations of a solid region (`(−y, x)` needs
  `y·e_x` and `x·e_y`) lie in the span.
* **Features.** These follow the PPE geometry encoder, plus the viscous stiffness
  `s_i = tr(B_i)/(sdim M_i) − 1 ≈ Dt μ/(ρ dp²)`:
  `loc = [log(1+s)/10, (log(1+s) − mean)/5, Σ_fluid w/N0, Σ_wall w/N0, d_wall/RadiusV, log(D_i/<D>)]`,
  `glob = [log n, extents, max/mean/std of log(1+s), wall fraction]`.
* **Training** (`train_vis_don.py`) is unsupervised and uses only the exported matrix. The loss is the
  A-norm two-grid error propagation `‖S(I − V Ac⁻¹ Vᵀ A)S e‖²_A`, the quantity that sets the CG
  rate. It is evaluated on white noise, Jacobi-smoothed noise and the true warm-start error `u − u0`.
* **Safety.**
  * The stored operator is checked against `implicitViscMatvec()` (rel. err < 1e-9) on the first
    solve and every `VisDonVerifyEvery` solves.
  * The device inference is checked against a host recomputation.
  * Acceptance always uses the true residual from `implicitViscMatvec()`.
  * After `VisDonMaxIteration` iterations, or on a breakdown, the original Jacobi CG takes over.
* **Cost control.**
  * The basis is reused for `VisDonSetupEvery` solves. Rows follow `ParticleIndex` across re-sorts.
  * `Vᵀ A V` is rebuilt every solve by default (`VisDonGalerkinEvery 1`). This is cheap, and a stale
    one would still give an SPD preconditioner.

## Usage (from `source/`)

```
make GPU=1 DATASET=1 VIS=1   # -> SPH_acc_dataset_vis : vis_*.bin every 50 viscosity solves -> results/trained_data/<case>/
make GPU=1 DATASET=1 ALL=1   # -> SPH_acc_dataset_all : ppe_*.bin and vis_*.bin in the same case directory
make train VIS=1             # all cases under results/trained_data (dirs without vis_*.bin are skipped)
make GPU=1 ONET=1 VIS=1      # -> SPH_acc_onet_vis    : DeepONet viscosity preconditioner on by default
make GPU=1 ONET=1 ALL=1      # -> SPH_acc_onet_all    : PPE and viscosity DeepONet
```

`VIS_EVERY=n` changes the snapshot interval, `VIS_MODEL=path` the model compiled in, and
`VIS_EPOCHS` the training length. The trainer writes `trained_model/vis_deeponet.data.txt` with the
settings it was trained for. Paste it into a `.data` file to enable the preconditioner in any binary.

```
VisDon              1
VisDonModel         .../vis_deeponet/trained_model/vis_deeponet.bin
VisDonChebDegree    6        # 0 -> damped Jacobi (VisDonSmoothSweeps, VisDonOmega)
VisDonChebRatio     30
VisDonSetupEvery    10
VisDonGalerkinEvery 1
VisDonMaxIteration  500      # then Jacobi CG fallback
```

With `LogVerbose 1` every solve prints `VIS-DeepONet: iter=… res2/target2=… setup=…ms solve=…ms`.
The `step=` line reports the iteration count in `visc_iter`. At the end, `Viscosity summary` lines
give the Jacobi and DeepONet totals and a per-phase breakdown.

Offline checks:

```bash
python vis_don_data.py results/trained_data/T05_*            # |b − A u| vs the solver's target, stiffness
python train_vis_don.py --data DIR1 DIR2 --out model --epochs 30 --val-fraction 0.2
```

## Files

| file | role |
|---|---|
| `../vis_deeponet.inc` | C++/OpenACC runtime: stored operator, inference, Galerkin setup, symmetric two-level cycle, PCG, snapshot export |
| `vis_don_data.py` | `VISDON01` snapshot reader |
| `vis_don_model.py` | model (E, G, T), coarse basis, smoother, weight export (`DONVIS01`) |
| `vis_don_solver.py` | torch replica of the solver's CG and of `visApply` |
| `train_vis_don.py` | training; writes `vis_deeponet.bin` and the matching `.data` settings |

## Notes / limitations
* A model is tied to the spatial dimension and to the feature definition in `visComputeGeometry()`.
* Snapshots contain only the in-plane components. In 2-D, the out-of-plane equation is `M u_z = b_z` and
  is handled exactly by the smoother.
* The data-file keys work with every binary, as long as the model file is given.
