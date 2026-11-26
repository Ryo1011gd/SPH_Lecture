# DeepONet two-level preconditioner for the ISPH pressure Poisson equation

The PPE `A p = b` in `main_implicit.cpp` is solved by BiCGStab to `|r|/|b| < PpeTolerance` (1e-8).
This module replaces the Jacobi preconditioner with a **DeepONet + Chebyshev two-level preconditioner**.
**The tolerance and the converged solution are unchanged.** The network only changes how quickly
the solver reaches them. If anything goes wrong, the original Jacobi BiCGStab finishes the solve.

## Measured results
Measured on an RTX 6000 Ada with nvc++ 24.1 at -O3 -acc. Each run is 300 steps of a V&V case at tol 1e-8, with `PpeDonChebDegree 10`, `PpeDonChebRatio 30` and `PpeDonSetupEvery 10`.
**All five cases are held out**: they were never used for training.

| case (held out) | particles | Jacobi iter | DeepONet iter | PPE time Jacobi → DeepONet | whole run |
|---|---|---|---|---|---|
| D01 viscous collapse, AR 2.8 | 9511 | 125.0 | **2.00** | 13.03 s → 0.53 s (**25×**) | 16.9 s → 4.0 s (4.2×) |
| D03 inclined plane 20° | 9562 | 124.4 | **2.01** | 12.73 s → 0.56 s (**23×**) | 17.2 s → 4.6 s (3.7×) |
| D04 Jeffreys film | 1181 | 37.6 | **1.00** | 3.78 s → 0.35 s (**11×**) | 6.6 s → 2.9 s (2.3×) |
| V05 Taylor–Green (closed) | 10000 | 57.1 | **1.03** | 5.90 s → 0.44 s (**13×**) | 10.3 s → 5.0 s (2.1×) |
| V03 spin-up (closed) | 9692 | 55.0 | **1.10** | 5.71 s → 0.49 s (**12×**) | 10.3 s → 5.2 s (2.0×) |

* No run fell back to Jacobi.
* Final states match the Jacobi runs: position difference ≤ 5e-5 dp, relative velocity difference ≤ 6e-7, and the two-phase diagnostics agree to 7 digits.
* The PPE time includes everything: inference, Galerkin setup, λmax estimation and the true-residual check.

Effect of the Chebyshev degree (D01 AR2.8; more smoothing means fewer host syncs, so total time does not increase):

| PpeDonChebDegree | 2 | 4 | 6 | 8 | **10** | 12 |
|---|---|---|---|---|---|---|
| mean iterations | 6.84 | 3.98 | 3.00 | 2.49 | **2.00** | 2.00 |
| PPE time [s] | 0.750 | 0.560 | 0.536 | 0.529 | **0.531** | 0.548 |

Ablation on D01 (torch replica, held-out time steps):

| preconditioner | iterations |
|---|---|
| Jacobi (original) | 112 |
| Chebyshev(2) smoother only, no DeepONet | 32 |
| + DeepONet coarse space, P=32 | 10.2 |
| + DeepONet coarse space, P=64 | **7.4** |
| pure learned DeepONet operator (PpeDonCoarse 0), P=64 | 19.8 |

## Method

```
M⁻¹ r :  z  = Cheb_k(r; 0)                       pre-smoothing, Jacobi-preconditioned Chebyshev on [λmax/15, λmax]
         z += T (TᵀD⁻¹AT)⁻¹ TᵀD⁻¹ (r − A z)       DeepONet coarse correction (Galerkin, CURRENT A)
         z  = Cheb_k(r; z)                       post-smoothing
```

1. **Particle-native DeepONet.** Particles move and are re-sorted, so the network has no sensor grid.
   * The trunk `T_ik = t_k(x̂_i, local geometry, z)` is evaluated at the particles. During training
     the queries are sampled at random particles (`--queries`).
   * The geometry latent `z = G(mean_i E(enc_i), glob)` is a permutation-invariant set encoder.
   * The branch of the learned operator is an integral over particles, `c_k = (1/n) Σ_i Ψ_ik r_i/D_i`.
   * The trunk inputs include multi-scale random Fourier features of the normalized position.
2. **Linear in the residual.** `r` never passes through a nonlinearity, so `M⁻¹` is a fixed linear
   operator in every solve. Plain BiCGStab stays valid, and the residual amplitude never needs to be
   extrapolated.
3. **The DeepONet trunk serves as the Galerkin coarse space.** The coarse matrix is built from the
   **actual** current matrix. The network only supplies the subspace, so an unseen geometry costs
   some iterations but never correctness. This is what makes the held-out cases work.
4. **Spectral split.** For these discretizations `D⁻¹A` has a real, positive spectrum in
   (0, ~1.45]. Degree-2 Chebyshev removes everything above about 0.1. The roughly 70–100 modes
   below that threshold are global and smooth, and the DeepONet basis (P=64) captures them. Damped
   Jacobi plus DeepONet plateaued at about 22–29 iterations; Chebyshev plus DeepONet reaches 5–7.
5. **Unsupervised, operator-consistent training.** The loss is the two-grid error propagator
   `‖S (I − C A) S e‖²`, evaluated on white-noise and algebraically smooth probe vectors. It uses
   only the exported matrix, so no extra solves are needed. Supervised `Ax=b` and `A(p−p0)=r0`
   terms are added on random trunk queries.
6. **Closed domains (A·1 = 0).** When `A·1 ≈ 0` is detected, the D⁻¹-weighted mean is removed from
   every basis vector, and the mean of the preconditioner output is removed as well. Without this,
   the coarse matrix becomes singular and the iteration stagnates; this is what initially broke V03
   and V05.
7. **Cost control.**
   * `PpeDonSetupEvery 10`: the basis and coarse operator are reused across steps. When particles
     are re-sorted, the rows are moved by particle id.
   * The DeepONet BiCGStab uses the stored-coefficient matvec.
   * The preconditioner kernels are queued with OpenACC `async` and only synchronized at scalar
     reductions.
   * λmax is re-estimated every 20 solves.
8. **Safety.** The run is protected by:
   * a check of the stored operator against `applyPpeOperator()`;
   * a check of the device inference against a host recomputation;
   * a final true residual computed with `applyPpeOperator()`;
   * a Jacobi fallback, starting from the warm start if the DeepONet iterate is worse.

## Files

| file | role |
|---|---|
| `../ppe_deeponet.inc` | C++/OpenACC runtime: inference, Galerkin setup, Chebyshev two-level cycle, BiCGStab, snapshot export |
| `ppe_don_data.py` | snapshot reader (`python ppe_don_data.py DIR` checks that `|b − A p|/|b|` is at tolerance) |
| `ppe_don_model.py` | model, smoother, two-level cycle, weight export |
| `ppe_don_solver.py` | torch replica of the solver's BiCGStab and of `donApply` |
| `train_ppe_don.py` | training; writes `ppe_deeponet.bin` and the matching `.data` settings |
| `eval_ppe_don.py` | offline per-run iteration comparison |
| `gen_snapshots.sh` | example: export training snapshots from existing cases |
| `trained_model/` | model trained on 8 V&V cases (800 snapshots, P=64) |
| `make_manual.py`, `PPE_DeepONet_manual.pdf` | Japanese user manual |

## Usage

**Build.** The makefile is unchanged apart from the dependency on `ppe_deeponet.inc`. The GPU,
g++ and g++-OpenMP builds were all tested.

**Build modes** (same convention as DeepONet-SPH; see `PPE_DeepONet_manual.pdf`):

```
make GPU=1 DATASET=1 PPE=1   # -> SPH_acc_dataset : writes results/trained_data/<case dir>/ppe_*.bin every 50 PPE solves
make train PPE=1             # all cases under results/trained_data (or DATA="dir1 dir2 ...")
make GPU=1 ONET=1 PPE=1      # -> SPH_acc_onet    : DeepONet preconditioner on by default
```
`PPE=1` is the default when no solver is selected. `VIS=1` selects the implicit-viscosity
preconditioner (`../vis_deeponet`), and `ALL=1` selects both (`SPH_acc_dataset_all`, `SPH_acc_onet_all`).

**Run with the preconditioner** using a normal binary: append `trained_model/ppe_deeponet.data.txt` to the `.data` file:

```
PpeDon              1
PpeDonModel         /media/ryo/NewVolume/Implicit_SPH/source/ppe_deeponet/trained_model/ppe_deeponet.bin
PpeDonCoarse        1        # 1: Galerkin on DeepONet basis (robust), 0: pure learned operator
PpeDonPreSmooth     1
PpeDonPostSmooth    1
PpeDonChebDegree    10       # 0 -> damped Jacobi (PpeDonSmoothSweeps, PpeDonOmega)
PpeDonChebRatio     30
PpeDonInitialGuess  1
PpeDonSetupEvery    10
```

The log prints one `PPE-DeepONet: iter=… setup=…ms solve=…ms` line per solve. At the end it prints
`PPE summary` lines with the total PPE time for both Jacobi and DeepONet and a per-phase time
breakdown. The `ISPH(...)` line still reports the total iteration count.

**New training data.** Normal Jacobi runs can export snapshots with:

```
PpeDonExportEvery 50
PpeDonExportDir   snaps
```

Then train and evaluate:

```bash
python train_ppe_don.py --data caseA/snaps caseB/snaps ... --out model --epochs 30 --fourier-max 10
python eval_ppe_don.py  --model model/best.pt --data heldout/snaps
```

## Notes / limitations
* A model is tied to the spatial dimension (2-D or 3-D build) and to the feature definition in
  `donComputeGeometry()`.
* The training set contained only one closed-domain case (V02). The closed-domain results above
  come from the null-space deflation, not from retraining. Retraining with more closed or two-phase
  cases with density ratios (not tested here) is recommended before relying on it for those.
* The run-to-run speed-up is larger for larger particle counts, because Jacobi's iteration count
  grows like `O(√n)` while the two-level count stays almost constant. At about 10k particles the
  remaining PPE cost is dominated by kernel-launch and synchronization latency.
* `nvc++` 23.x and 24.1 silently drop stores in one kernel shape at `-O1` and above: a per-thread
  local array plus an inner reduction plus strided stores. The code avoids it (`donFormQ`), and the
  start-up self-test would catch a recurrence.
