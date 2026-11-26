//================================================================================================//
//------------------------------------------------------------------------------------------------//
//    MPH-WC : Moving Particle Hydrodynamics for Weakly Compressible  (explicit calculation)      //
//------------------------------------------------------------------------------------------------//
//    Developed by    : Masahiro Kondo                                                            //
//    Distributed in  : 2022                                                                      //
//    Lisence         : GPLv3                                                                     //
//    For instruction : see README                                                                //
//    For theory      : see the following references                                              //
//     [1] CPM 8 (2021) 69-86,       https://doi.org/10.1007/s40571-020-00313-w                   //
//     [2] JSCES Paper No.20210006,  https://doi.org/10.11421/jsces.2021.20210006                 //
//     [3] CPM 9 (2022) 265-276,     https://doi.org/10.1007/s40571-021-00408-y                   //
//     [4] CMAME 385 (2021) 114072,  https://doi.org/10.1016/j.cma.2021.114072                    //
//     [5] JSCES Paper No.20210016,  https://doi.org/10.11421/jsces.2021.20210016                 //
//    (Please cite the references above when you make a publication using this program)           //
//    Copyright (c) 2022                                                                          //
//    Masahiro Kondo & National Institute of Advanced Industrial Science and Technology (AIST)    //
//================================================================================================//
//
// Two-phase stability patch, September 2026.
// IMPORTANT: the graph Laplacian is still not the composition D*G.
// This patch removes timestep-dependent density-penalty amplification, refreshes
// mass/viscosity after sorting, bounds explicit geometric repair by elapsed time,
// rejects runaway states, and avoids a roundoff-size extra step at EndTime.
// It does not establish Hysing accuracy or solve the operator-consistency issue.
//
// Recommended Case 2 overrides (keep physical rho=1000/1, mu=10/.1, sigma=1.96):
//   TwoPhaseSourceGamma       0.0
//   TwoPhaseDensityRelaxTime  0.01   // seconds; relevant only when gamma > 0
//   TwoPhaseRhoSmooth         0.0
//   TwoPhaseShiftCoef         0.1
//   TwoPhaseShiftVelCorr      1.0
//   TwoPhaseIfShiftNormal     0.0
//   TwoPhaseMaxSpeed          0.0    // 0 = 10*gravity/capillary/wall speed scale
//   TwoPhaseOpenSurface       0
//   TwoPhaseDtMinRatio        1e-4
//
// CHANGED SEMANTICS: density RHS is gamma*dn/tau^2, NOT gamma*dn/Dt^2.
// Gamma defaults to zero. A nonzero penalty is optional weak compressibility /
// density relaxation, not an exact incompressibility correction. tau is physical
// time and must be resolved (adaptive selector bounds Dt <= 0.1*tau/sqrt(gamma)).
// Speed guard REJECTS/rolls back; it does not clip accepted velocities. Explicit
// TwoPhaseMaxSpeed is needed if automatic characteristic scales do not describe
// the problem. A triggered guard is a failed simulation, not a validated result.
// Single-phase numerical path is unchanged. 2D only. Keep project log.h and
// errorfunc.h. Validation performed for this patch is reported to the user;
// historical validation claims from the incoming source have been removed.
// LIMITS: shifting is not exactly phase-volume/momentum conserving. Open-surface
// detection and the single global pressure nullspace retain their original
// limitations (unresolved voids and disconnected fluid components).
//
// LOCAL CHECKS (synthetic grids, not the user's 80000-fluid-particle case):
// g++ -std=c++11 -O2 -fopenmp; local logging-header stubs only for the test build.
// Hysing Case 2 physical properties, box 1x2, R=.25, dp=.05, 800 fluids + 544
// walls; all walls no-slip in this smoke test. Dt=1e-5, gamma=.05, rhoSmooth=.1,
// shiftCoef=.5, shiftVelCorr=.5, PPE tol=1e-6. On a +/- .001 m jittered fluid
// lattice (Python random.Random seed 42), 300 steps: peak speed original 3.352762,
// revised .006201698 m/s. Final max number-density error: original 1.594390e-4,
// revised 8.161716e-3 (tradeoff; density disorder is not corrected as strongly).
// Regular-grid run completed .02 s in exactly 2000 steps, without the old
// ~6e-16 s extra timestep; final peak speed .033098 m/s and |p|max ~1383 Pa.
// Five jittered-grid steps with two OpenMP threads under ASan/UBSan passed;
// leak detection disabled because the legacy program does not free all arrays.
// Forced speed-limit rejection restores t=0 and all 1344 particle rows to within
// 4.5e-16 in the text checkpoint. No NVHPC/GPU run or 3-second validation.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <assert.h>
#include <omp.h>

#include "errorfunc.h"
#include "log.h"

const double DOUBLE_ZERO[32]={0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0};

using namespace std;
#define TWO_DIMENSIONAL

#ifndef TWO_DIMENSIONAL
#error "This source uses 2D interface curvature and wall reconstruction."
#endif
#define DIM 3

// Property definition
#define TYPE_COUNT   4
#define FLUID_BEGIN  0
#define FLUID_END    2
#define WALL_BEGIN   2
#define WALL_END     4

#define  DEFAULT_LOG  "sample.log"
#define  DEFAULT_DATA "sample.data"
#define  DEFAULT_GRID "sample.grid"
#define  DEFAULT_PROF "sample%03d.prof"
#define  DEFAULT_VTK  "sample%03d.vtk"

// Calculation and Output
static double ParticleSpacing=0.0;
static double ParticleVolume=0.0;
static double OutputInterval=0.0;
static double OutputNext=0.0;
static double VtkOutputInterval=0.0;
static double VtkOutputNext=0.0;
static double EndTime=0.0;
static double Time=0.0;
static double Dt=1.0e100;
static double DomainMin[DIM];
static double DomainMax[DIM];
static double DomainWidth[DIM];
#pragma acc declare create(ParticleSpacing,ParticleVolume,Dt,DomainMin,DomainMax,DomainWidth)

#define Mod(x,w) ((x)-(w)*floor((x)/(w)))   // mod 

#define MAX_NEIGHBOR_COUNT 512
// Particle
static int ParticleCount;
static int *ParticleIndex;                // original particle id
static int *Property;                     // particle type
static double (*Mass);                    // mass
static double (*Position)[DIM];           // coordinate
static double (*Velocity)[DIM];           // momentum
static double (*Force)[DIM];              // total explicit force acting on the particle
static int (*NeighborCount);                  // [ParticleCount]
static long int (*NeighborPtr);           // [ParticleCount+1];
static int          (*NeighborInd);           // [ParticleCount x NeighborCount]
static long int   NeighborIndCount;
static double (*NeighborCalculatedPosition)[DIM];
static int    (*TmpIntScalar);                // [ParticleCount] to sort with cellId
static double (*TmpDoubleScalar);             // [ParticleCount]
static double (*TmpDoubleVector)[DIM];        // [ParticleCount]
#define MARGIN (0.5*ParticleSpacing)  // RK-safe Verlet skin
#pragma acc declare create(ParticleCount,ParticleIndex,Property,Mass,Position,Velocity,Force,NeighborCalculatedPosition)
#pragma acc declare create(NeighborCount,NeighborPtr,NeighborInd,NeighborIndCount)
#pragma acc declare create(TmpIntScalar,TmpDoubleScalar,TmpDoubleVector)


// BackGroundCells
#ifdef TWO_DIMENSIONAL
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1])
#else
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1]*CellCount[2] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1]*CellCount[2] + ((iCZ)%CellCount[2]+CellCount[2])%CellCount[2])
#endif

static int PowerParticleCount;
static int ParticleCountPower;                   
static double CellWidth[DIM];
static int CellCount[DIM];
static int TotalCellCount = 0;
static int *CellFluidParticleBegin;  // beginning of fluid particles in the cell
static int *CellFluidParticleEnd;  // end of fluid particles in the cell
static int *CellWallParticleBegin;   // beginning of wall particles in the cell
static int *CellWallParticleEnd;   // end of wall particles in the cell
static int *CellIndex;  // [ParticleCountPower>>1]
static int *CellParticle;       // array of particle id in the cells) [ParticleCountPower>>1]
#pragma acc declare create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount,TotalCellCount)
#pragma acc declare create(CellFluidParticleBegin,CellFluidParticleEnd,CellWallParticleBegin,CellWallParticleEnd,CellIndex,CellParticle)

// Type
static double Density[TYPE_COUNT];
static double ShearViscosity[TYPE_COUNT];
static double SurfaceTension[TYPE_COUNT];
static double InterfaceTension01 = 0.0; // physical Type0-Type1 interfacial tension [N/m]
static double CofA[TYPE_COUNT];   // coefficient for attractive pressure
static double CofK;               // coefficinet (ratio) for diffuse interface thickness normalized by ParticleSpacing
static double InteractionRatio[TYPE_COUNT][TYPE_COUNT];
#pragma acc declare create(Density,ShearViscosity,SurfaceTension,CofA,CofK,InteractionRatio)
#pragma acc declare create(InterfaceTension01)

// Kondo-Koshizuka-Takimoto (2007) inter-particle potential surface tension.
// The original paper uses re = 3.2 * initial nearest-neighbor spacing.
static double SurfacePotentialRadius = 0.0;
static double SurfacePotentialC[TYPE_COUNT] = {0.0,0.0,0.0,0.0};
#pragma acc declare create(SurfacePotentialRadius,SurfacePotentialC)


// Fluid
static int FluidParticleBegin;
static int FluidParticleEnd;
static double *DensityA;        // number density per unit volume for attractive pressure
static double (*GravityCenter)[DIM];
static double *PressureA;       // attractive pressure (surface tension)
static double *VolStrainP;        // number density per unit volume for base pressure
static double *DivergenceP;     // volumetric strainrate for pressure B
static double *PressureP;       // base pressure
static double *VirialPressureAtParticle; // VirialPressureInSingleParticleRegion
static double (*VirialStressAtParticle)[DIM][DIM];
static double *Mu;              // viscosity coefficient for shear
#pragma acc declare create(FluidParticleBegin,FluidParticleEnd)
#pragma acc declare create(DensityA,GravityCenter,PressureA,VolStrainP,DivergenceP,PressureP,VirialPressureAtParticle,VirialStressAtParticle,Mu)

static double Gravity[DIM] = {0.0,0.0,0.0};
#pragma acc declare create(Gravity)


// -----------------------------------------------------------------------------
// Robust baseline ISPH projection
// Standard MPS Laplacian + smooth free-surface relaxation (Chow/Skillen)
// + minimum-pressure MPS gradient + consistent virtual-wall Neumann BC + Lind Fickian shifting.
// -----------------------------------------------------------------------------
static int    PpeMaxIteration = 500;
static double PpeTolerance    = 1.0e-4;
// -----------------------------------------------------------------------------
// Free-surface detection  --  Marrone, Colagrossi, Le Touze & Graziani,
// "Fast free-surface detection and level-set function definition in SPH
//  solvers", J. Comput. Phys. 229 (2010) 3652-3663.
static double PpeSurfLambdaMin  = 0.75;
static double PpeSurfLambdaKeep = 0.55;
static double PpeSurfPndMin     = 0.85;
static double PpeSurfProbeRatio = 1.0;
#pragma acc declare create(PpeSurfLambdaMin,PpeSurfLambdaKeep,PpeSurfPndMin,PpeSurfProbeRatio)
static double PpeLambdaP = 1.0; // fallback second moment
static double PpeHlNorm = 1.0;
static double MrefP      = 1.0; // interior value of the renormalisation-matrix diagonal
#pragma acc declare create(MrefP)

// Internal numerical stabilization constants.
static const double IsphEcsAlpha      = 2.5e-3;
static const double IsphDensityErrMax = 1.0e-2;
static const double IsphMinPairRatio  = 0.35;
static const double IsphQmlsReg       = 1.0e-10;
static const int    IsphPressureFloor = 0;


// -----------------------------------------------------------------------------
// Two-phase branch.
//
// Automatically enabled only when BOTH fluid Type 0 and Type 1 are present.
// The single-phase pressure and force formulas are retained.
//
// Two-phase branch:
//   * all gas/liquid particles are pressure unknowns in default closed mode;
//     TwoPhaseOpenSurface=1 enables geometric free-surface detection,
//   * variable-density PPE: div[(1/rho) grad p] = source,
//   * harmonic interpolation of 1/rho: beta_ij = 2/(rho_i+rho_j),
//   * mixed divergence-free / density-invariant source term,
//   * pure-Neumann null space removed by projection, not by a pinned gauge,
//   * interface-aware Fickian shifting for both phases,
//   * selectable CSF (default), legacy CSS, or pairwise-potential surface tension.
// -----------------------------------------------------------------------------
static int TwoPhaseActive = 0;
static int TwoPhaseGauge  = -1;
#pragma acc declare create(TwoPhaseActive,TwoPhaseGauge)

// TwoPhaseSourceGamma: small density-error penalty added to full divergence
// source b=-div(u*)/Dt + gamma*dn/tau^2, tau=TwoPhaseDensityRelaxTime.
// The old gamma*dn/Dt^2 formula amplified disorder as adaptive Dt decreased.
// TwoPhaseRhoSmooth: optional density smoothing; default 0 preserves the jump.
// TwoPhaseShiftCoef: dt-scaled Fickian shifting; <=0 disables shifting/repair.
static double TwoPhaseSourceGamma = 0.0;
static double TwoPhaseDensityRelaxTime = 0.01;
static double TwoPhaseMaxSpeed = 0.0;
static double TwoPhaseSpeedLimit = 0.0; // derived on host, passed into kernels
#pragma acc declare create(TwoPhaseDensityRelaxTime)
static double TwoPhaseRhoSmooth   = 0.0;
static double TwoPhaseShiftCoef   = 0.1;
static double TwoPhaseIfThreshold = 0.02;
// Dt in the data file is an upper bound for two-phase runs.
static double TwoPhaseDtMax=0.0;
static double TwoPhaseDtMinRatio=1.0e-4; // stop rather than crawl through numerical blow-up
static int TwoPhaseOpenSurface=0;
static int TwoPhaseStepFailed=0; // host-only retry status
#pragma acc declare create(TwoPhaseOpenSurface)
static int    PpePureNeumann      = 0;   // 1: no Dirichlet row -> constant null space

// Fickian shift normal fraction (default 0), and Taylor velocity remapping.
// Close-pair geometric repair additionally separates pairs below 0.85 dp;
// the combined displacement is capped at 0.02 dp. It is not exactly conservative.
// TwoPhaseDuCap: any activation rejects and retries the entire step.
// TwoPhaseMinFluidNb: legacy parsed option, no longer creates p=0 particles.
static double TwoPhaseIfShiftNormal = 0.0;
static double TwoPhaseShiftVelCorr  = 1.0;
static double TwoPhaseDuCap         = 0.25;
static int    TwoPhaseMinFluidNb    = 4;
#pragma acc declare create(TwoPhaseSourceGamma,TwoPhaseRhoSmooth,TwoPhaseShiftCoef)
#pragma acc declare create(TwoPhaseIfThreshold,PpePureNeumann)
#pragma acc declare create(TwoPhaseIfShiftNormal,TwoPhaseShiftVelCorr,TwoPhaseDuCap,TwoPhaseMinFluidNb)

// -----------------------------------------------------------------------------
// Surface-tension model for the two-phase branch.
//
//  TwoPhaseStModel = 0 : CSF  - Brackbill/Morris continuum surface force.
//                              f = sigma * kappa * grad C, with kappa built
//                              from a reproducing (first-order consistent)
//                              divergence of the interface normal
//                              (Adami, Hu & Adams, JCP 229 (2010) 5011).
//                              Uses the same pair stencil as pressure gradients.
//                              StDensityScale/StZeroNet are ignored for model 0.
//                = 1 : CSS  - previous curvature-free stress form (comparison).
//                = 2 : PP   - pairwise inter-particle potential between unlike
//                              particles, coefficient calibrated at start-up
//                              from a broken-bond lattice sum so that the
//                              planar interfacial tension equals sigma_01.
//                              Exactly momentum conserving.
//
// TwoPhaseStDensityScale : 0 = force distributed equally (old behaviour),
//                          1 = full Brackbill weight rho_i/rho_avg.  ALWAYS
//                          leave at 1 for a density ratio above ~20.
// TwoPhaseStDeltaMin     : |grad C|*dp below which the normal is considered
//                          undefined (bulk particle, no curvature, no force).
// TwoPhaseCurvMax        : |kappa| <= TwoPhaseCurvMax/dp.  A curvature sharper
//                          than one particle spacing is noise by definition.
// -----------------------------------------------------------------------------
static int    TwoPhaseStModel        = 0;
static double TwoPhaseStSmoothRatio  = 1.0;   // colour smoothing radius / RadiusP
static int    TwoPhaseStDoubleSmooth = 1;     // 1: apply the Shepard filter twice
static double TwoPhaseStDensityScale = 1.0;
static double TwoPhaseStZeroNet      = 0.0;   // remove the residual net ST force
static double TwoPhaseStDeltaMin     = 0.05;
static double TwoPhaseCurvMax        = 2.0;
static double TwoPhasePpRadiusRatio  = 3.1;   // pairwise potential cut-off / dp
static int    WallNoSlip             = 1;     // Hysing: no-slip walls

static double RadiusST = 0.0;                 // colour smoothing radius
static double MrefST   = 1.0;                 // interior M diagonal at RadiusST
static double PpRadius = 0.0;
static double PpCoef   = 0.0;                 // calibrated pairwise coefficient
#pragma acc declare create(TwoPhaseStModel,TwoPhaseStSmoothRatio,TwoPhaseStDoubleSmooth)
#pragma acc declare create(TwoPhaseStDensityScale,TwoPhaseStZeroNet,TwoPhaseStDeltaMin)
#pragma acc declare create(TwoPhaseCurvMax,TwoPhasePpRadiusRatio,WallNoSlip)
#pragma acc declare create(RadiusST,MrefST,PpRadius,PpCoef)

static double *ColorS  = NULL;                // colour smoothing buffer
static double (*IfGrad)[DIM] = NULL;          // grad C  (= delta * n)
static double *IfKappa = NULL;                // interface curvature
#pragma acc declare create(ColorS,IfGrad,IfKappa)


static double *RhoEff  = NULL;           // density used by the projection
static double *ColorC  = NULL;           // Shepard-smoothed colour function
static double *IfDelta = NULL;           // |grad C| (interface surface delta)
static double (*IfNormal)[DIM] = NULL;   // interface unit normal
#pragma acc declare create(RhoEff,ColorC,IfDelta,IfNormal)

// -----------------------------------------------------------------------------
// Legacy WBP input parameters retained only for backward-compatible data files.
// -----------------------------------------------------------------------------
static double WbpActivationRatio = 0.06;
static double WbpWeightRatio     = 0.50;
static double WbpBcRelax         = 0.85;
static double WbpAccelCapFactor  = 30.0;
static double WbpDistanceMinRatio= 0.50;
static double WbpDistanceMaxRatio= 1.50;
#pragma acc declare create(WbpActivationRatio,WbpWeightRatio,WbpBcRelax,WbpAccelCapFactor,WbpDistanceMinRatio,WbpDistanceMaxRatio)

static int    *PpeFlag;      // 0: p=0 free-surface/inactive, 1: pressure unknown
static double *PpeAlpha;     // smooth near-free-surface PPE relaxation [0,1]
static double *PositionDivR; // normalized divergence of position (bulk ~= 2 in 2D)
static double PositionDivRef = 1.0; // raw complete-support reference
static double *PpeDiag;
static double *PpeB;
static double *PpeR;
static double *PpeR0;
static double *PpeP;
static double *PpePHat;
static double *PpeV;
static double *PpeS;
static double *PpeT;
static double *PpeOpX;
static double *PpeOpY;

// QMLS coefficients, one entry per NeighborInd entry.
static double *IsphGradX = NULL;
static double *IsphGradY = NULL;
static double *IsphLap   = NULL;
static int    *IsphQmlsUse = NULL; // 1: use common second-order QMLS D/G/L row
static long int IsphCoeffCount = 0;

// RK storage for explicit/non-pressure predictor.
static double (*RkX0)[DIM];
static double (*RkU0)[DIM];
static double (*RkK1X)[DIM];
static double (*RkK2X)[DIM];
static double (*RkK3X)[DIM];
static double (*RkK4X)[DIM];
static double (*RkK1U)[DIM];
static double (*RkK2U)[DIM];
static double (*RkK3U)[DIM];
static double (*RkK4U)[DIM];

#pragma acc declare create(PpeMaxIteration,PpeTolerance,PpeLambdaP,PpeHlNorm)
#pragma acc declare create(PpeFlag,PpeAlpha,PositionDivR,PositionDivRef,PpeDiag,PpeB,PpeR,PpeR0,PpeP,PpePHat,PpeV,PpeS,PpeT,PpeOpX,PpeOpY)
#pragma acc declare create(IsphGradX,IsphGradY,IsphLap,IsphQmlsUse,IsphCoeffCount)
#pragma acc declare create(RkX0,RkU0,RkK1X,RkK2X,RkK3X,RkK4X,RkK1U,RkK2U,RkK3U,RkK4U)

// Wall
static int WallParticleBegin;
static int WallParticleEnd;
static double WallCenter[WALL_END][DIM];
static double WallVelocity[WALL_END][DIM];
static double WallOmega[WALL_END][DIM];
static double WallRotation[WALL_END][DIM][DIM];
#pragma acc declare create(WallParticleBegin,WallParticleEnd)
#pragma acc declare create(WallCenter,WallVelocity,WallOmega,WallRotation)

// proceedures
static void readDataFile(char *filename);
static void readGridFile(char *filename);
static void writeProfFile(const char *filename);
static void writeVtkFile(const char *filename);
static void initializeWeight( void );
static void initializeDomain( void );
static void initializeFluid( void );
static void initializeWall( void );

static void calculateConvection();
static void selectTwoPhaseDt();
static void advanceTwoPhase();
static int invalidTwoPhaseState();
static void calculatePredictor();
static void evaluateExplicitAcceleration(double (*acc)[DIM]);
static void calculateWall();
static void calculatePeriodicBoundary();
static void resetForce();
static int neighborCalculation();
static void calculateCellParticle( void );
static void calculateNeighbor( void );
static void freeNeighbor( void );
static void calculateNeighbor();
static void calculatePhysicalCoefficients();
static void calculateDensityA();
static void calculatePressureA();
static void calculateGravityCenter();
static void calculateDiffuseInterface();
static void calculateKondoPotentialSurfaceTension();
static void calculateEffectiveDensity();
static void calculateInterfaceGeometry();
static void calculateTwoPhaseCSSSurfaceTension();
static void calculateTwoPhaseSurfaceTension();
static void calculateCsfSurfaceTension();
static void calculatePpSurfaceTension();
static void calculateDensityP();
static void calculateDivergenceP();
static void calculateDivergenceTwoPhase();
static void detectFreeSurface();
static void calculatePpe();
static void applyPpeOperator();
static void applyParticleShifting();
static void applyTwoPhaseShifting();
static void logTwoPhaseStabilityDiagnostics();
static void buildIsphHighOrderCoefficients();
static void freeIsphHighOrderCoefficients();
static void calculateViscosityV();
static void calculateGravity();
static void calculateAcceleration();
static void calculateVirialPressureAtParticle();
static void calculateVirialStressAtParticle();


// dual kernel functions
static double RadiusRatioA;
static double RadiusRatioG;
static double RadiusRatioP;
static double RadiusRatioV;

static double MaxRadius = 0.0;
static double RadiusA = 0.0;
static double RadiusG = 0.0;
static double RadiusP = 0.0;
static double RadiusV = 0.0;
static double Swa = 1.0;
static double Swg = 1.0;
static double Swp = 1.0;
static double Swv = 1.0;
static double N0a = 1.0;
static double N0p = 1.0;
static double R2g = 1.0;

#pragma acc declare create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)


#pragma acc routine seq
static double wa(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#else
    return 1.0/Swa * 1.0/(h*h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#endif
}

#pragma acc routine seq
static double dwadr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#else
    return 1.0/Swa * 1.0/(h*h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#endif
}

#pragma acc routine seq
static double wg(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwgdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double wp(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swp * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwpdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swp * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}


#pragma acc routine seq
static double d2wpdr2(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * (2.0/(h*h));
#else
    return 1.0/Swp * 1.0/(h*h*h) * (2.0/(h*h));
#endif
}

#pragma acc routine seq
static double wv(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swv * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwvdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swv * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}


#pragma acc routine seq
static int invert5x5(const double Ain[5][5], double Ainv[5][5])
{
    double a[5][10];
    for(int i=0;i<5;++i){
        for(int j=0;j<5;++j) a[i][j]=Ain[i][j];
        for(int j=0;j<5;++j) a[i][j+5]=(i==j ? 1.0 : 0.0);
    }
    for(int col=0;col<5;++col){
        int piv=col;
        double amax=fabs(a[col][col]);
        for(int r=col+1;r<5;++r){
            const double av=fabs(a[r][col]);
            if(av>amax){ amax=av; piv=r; }
        }
        if(!(amax>1.0e-18)) return 0;
        if(piv!=col){
            for(int j=0;j<10;++j){
                const double tmp=a[col][j];
                a[col][j]=a[piv][j];
                a[piv][j]=tmp;
            }
        }
        const double d=a[col][col];
        for(int j=0;j<10;++j) a[col][j]/=d;
        for(int r=0;r<5;++r){
            if(r==col) continue;
            const double f=a[r][col];
            for(int j=0;j<10;++j) a[r][j]-=f*a[col][j];
        }
    }
    for(int i=0;i<5;++i)
        for(int j=0;j<5;++j)
            Ainv[i][j]=a[i][j+5];
    return 1;
}


	clock_t cFrom, cTill, cStart, cEnd;
	clock_t cNeigh=0, cExplicit=0, cVirial=0, cOther=0;

int main(int argc, char *argv[])
{
	
    char logfilename[1024]  = DEFAULT_LOG;
    char datafilename[1024] = DEFAULT_DATA;
    char gridfilename[1024] = DEFAULT_GRID;
    char proffilename[1024] = DEFAULT_PROF;
    char vtkfilename[1024] = DEFAULT_VTK;
	    int numberofthread = 1;
    
    {
        if(argc>1)strcpy(datafilename,argv[1]);
        if(argc>2)strcpy(gridfilename,argv[2]);
        if(argc>3)strcpy(proffilename,argv[3]);
        if(argc>4)strcpy(vtkfilename,argv[4]);
        if(argc>5)strcpy( logfilename,argv[5]);
    	if(argc>6)numberofthread=atoi(argv[6]);
    }
    log_open(logfilename);
    {
        time_t t=time(NULL);
        log_printf("start reading files at %s\n",ctime(&t));
    }
	{
		#ifdef _OPENMP
		omp_set_num_threads( numberofthread );
		#pragma omp parallel
		{
			if(omp_get_thread_num()==0){
				log_printf("omp_get_num_threads()=%d\n", omp_get_num_threads() );
			}
		}
		#endif
    }
    readDataFile(datafilename);
    readGridFile(gridfilename);
    {
        time_t t=time(NULL);
        log_printf("start initialization at %s\n",ctime(&t));
    }
    initializeWeight();
    initializeFluid();
    TwoPhaseDtMax=Dt;
    #pragma acc update device(TwoPhaseOpenSurface,TwoPhaseDensityRelaxTime)
    initializeWall();
    initializeDomain();

	// data transfer from host to device
	#pragma acc update device(ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCount,ParticleIndex[0:ParticleCount],Property[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc update device(Density[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc update device(InterfaceTension01)
	#pragma acc update device(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc update device(Mu[0:ParticleCount],Gravity[0:DIM])
	#pragma acc update device(PpeMaxIteration,PpeTolerance)
	#pragma acc update device(PpeSurfLambdaMin,PpeSurfLambdaKeep,PpeSurfPndMin,PpeSurfProbeRatio)
	#pragma acc update device(TwoPhaseSourceGamma,TwoPhaseRhoSmooth,TwoPhaseShiftCoef,TwoPhaseIfThreshold,PpePureNeumann)
	#pragma acc update device(TwoPhaseIfShiftNormal,TwoPhaseShiftVelCorr,TwoPhaseDuCap,TwoPhaseMinFluidNb)
	#pragma acc update device(TwoPhaseStModel,TwoPhaseStSmoothRatio,TwoPhaseStDoubleSmooth)
	#pragma acc update device(TwoPhaseStDensityScale,TwoPhaseStZeroNet,TwoPhaseStDeltaMin)
	#pragma acc update device(TwoPhaseCurvMax,TwoPhasePpRadiusRatio,WallNoSlip)
	#pragma acc update device(WbpActivationRatio,WbpWeightRatio,WbpBcRelax,WbpAccelCapFactor,WbpDistanceMinRatio,WbpDistanceMaxRatio)
	#pragma acc update device(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	#pragma acc update device(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc update device(PowerParticleCount,ParticleCountPower,CellWidth[0:DIM],CellCount[0:DIM])
	#pragma acc update device(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount])
	#pragma acc update device(CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	

	{
		calculateCellParticle();
   		calculateNeighbor();
		log_printf("line:%d, NeighborIndCount = %ld\n",__LINE__,NeighborIndCount);

		calculateDensityA();
		calculateGravityCenter();
		calculateDensityP();

		#pragma acc kernels present(PressureP[0:ParticleCount],PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],PositionDivR[0:ParticleCount],RhoEff[0:ParticleCount],ColorC[0:ParticleCount],ColorS[0:ParticleCount],IfDelta[0:ParticleCount],IfGrad[0:ParticleCount][0:DIM],IfNormal[0:ParticleCount][0:DIM],IfKappa[0:ParticleCount],Property[0:ParticleCount],Density[0:TYPE_COUNT])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0;iP<ParticleCount;++iP){
			PressureP[iP]=0.0;
			PpeFlag[iP]=0;
			PpeAlpha[iP]=0.0;
			PositionDivR[iP]=0.0;
			RhoEff[iP]=Density[Property[iP]];
			ColorC[iP]=0.0;
			ColorS[iP]=0.0;
			IfDelta[iP]=0.0;
			IfGrad[iP][0]=0.0;
			IfGrad[iP][1]=0.0;
			IfGrad[iP][2]=0.0;
			IfNormal[iP][0]=0.0;
			IfNormal[iP][1]=0.0;
			IfNormal[iP][2]=0.0;
			IfKappa[iP]=0.0;
		}

		writeVtkFile("output.vtk");
		
		{
			time_t t=time(NULL);
			log_printf("start main roop at %s\n",ctime(&t));
		}
		int iStep=(int)(Time/Dt);
		cStart = clock();
		cFrom = cStart;
		while(TwoPhaseActive ? Time < EndTime-1.0e-12*TwoPhaseDtMax : Time < EndTime + 1.0e-5*Dt){
			
			if( Time + 1.0e-5*Dt >= OutputNext ){
				char filename[256];
				sprintf(filename,proffilename,iStep);
				writeProfFile(filename);
				log_printf("@ Prof Output Time : %e\n", Time );
				OutputNext += OutputInterval;
			}
			cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;
		
            if(TwoPhaseActive){
                advanceTwoPhase();
                Time += Dt; // outputs below describe the accepted new state
                logTwoPhaseStabilityDiagnostics();
            }else{
			// The predictor must start from a valid Verlet list.  Rebuild here
			// rather than inside the RK stages, because cell sorting changes
			// particle order.
			if(neighborCalculation()==1){
				freeNeighbor();
				calculateCellParticle();
				calculateNeighbor();
			}

			// RK2 predictor for explicit / non-pressure terms
			calculatePhysicalCoefficients();
			calculatePredictor();

			// prescribed wall motion once per full step
			calculateWall();
			calculatePeriodicBoundary();

			// neighbor rebuild only after the complete predictor
			if(neighborCalculation()==1){
				freeNeighbor();
				calculateCellParticle();
				calculateNeighbor();
			}
			cTill = clock(); cNeigh += (cTill-cFrom); cFrom = cTill;

			// final geometry/density for ISPH
			calculateDensityA();
			calculateGravityCenter();
			calculateDensityP();

			// robust baseline ISPH/MPS pressure projection
			calculatePpe();

            if(TwoPhaseActive){
                // Interface-aware Fickian shifting for BOTH phases.  The
                // component normal to the phase interface, to a free surface
                // and to a wall is projected out, so the two phases regularise
                // without mixing and without the energy injection of the old
                // pairwise velocity "collision" model.
                applyTwoPhaseShifting();
                logTwoPhaseStabilityDiagnostics();
            }else{
                // Preserve the validated one-phase cPS/DPC behavior.
                applyParticleShifting();
            }
			
            }

			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
			
			
			if( Time + 1.0e-5*Dt >= VtkOutputNext ){
				calculateVirialStressAtParticle();
				cTill = clock(); cVirial += (cTill-cFrom); cFrom = cTill;
				
				char filename [256];
				sprintf(filename,vtkfilename,iStep);
				writeVtkFile(filename);
				log_printf("@ Vtk Output Time : %e\n", Time );
				VtkOutputNext += VtkOutputInterval;
				cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;

			}
			
			if(!TwoPhaseActive) Time += Dt;
			iStep++;
			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
		}
	}
	cEnd = cTill;
	
	{
		time_t t=time(NULL);
		log_printf("end main roop at %s\n",ctime(&t));
		log_printf("neighbor search:         %lf [CPU sec]\n", (double)cNeigh/CLOCKS_PER_SEC);
		log_printf("explicit calculation:    %lf [CPU sec]\n", (double)cExplicit/CLOCKS_PER_SEC);
		log_printf("virial calculation:      %lf [CPU sec]\n", (double)cVirial/CLOCKS_PER_SEC);
		log_printf("other calculation:       %lf [CPU sec]\n", (double)cOther/CLOCKS_PER_SEC);
		log_printf("total:                   %lf [CPU sec]\n", (double)(cNeigh+cExplicit+cVirial+cOther)/CLOCKS_PER_SEC);
		log_printf("total (check):           %lf [CPU sec]\n", (double)(cEnd-cStart)/CLOCKS_PER_SEC);
	}
	
	
	freeIsphHighOrderCoefficients();

	#pragma acc exit data delete(ParticleCount,ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc exit data delete(ParticleIndex[0:ParticleCount],Property[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(Density[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc exit data delete(InterfaceTension01)
	#pragma acc exit data delete(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc exit data delete(Mu[0:ParticleCount],Gravity[0:DIM])
	#pragma acc exit data delete(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	#pragma acc exit data delete(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc exit data delete(PowerParticleCount,ParticleCountPower,CellWidth[0:DIM],CellCount[0:DIM])
	#pragma acc exit data delete(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount])
	#pragma acc exit data delete(CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc exit data delete(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc exit data delete(Force[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount],NeighborCalculatedPosition[0:ParticleCount][0:DIM],TmpIntScalar[0:ParticleCount],TmpDoubleScalar[0:ParticleCount],TmpDoubleVector[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
	#pragma acc exit data delete(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
	#pragma acc exit data delete(PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],PositionDivR[0:ParticleCount],PpeDiag[0:ParticleCount],PpeB[0:ParticleCount],PpeR[0:ParticleCount],PpeR0[0:ParticleCount],PpeP[0:ParticleCount],PpePHat[0:ParticleCount],PpeV[0:ParticleCount],PpeS[0:ParticleCount],PpeT[0:ParticleCount],PpeOpX[0:ParticleCount],PpeOpY[0:ParticleCount],IsphQmlsUse[0:ParticleCount])
	#pragma acc exit data delete(RhoEff[0:ParticleCount],ColorC[0:ParticleCount],ColorS[0:ParticleCount],IfDelta[0:ParticleCount],IfGrad[0:ParticleCount][0:DIM],IfNormal[0:ParticleCount][0:DIM],IfKappa[0:ParticleCount])
	#pragma acc exit data delete(WbpActivationRatio,WbpWeightRatio,WbpBcRelax,WbpAccelCapFactor,WbpDistanceMinRatio,WbpDistanceMaxRatio)
	#pragma acc exit data delete(RkX0[0:ParticleCount][0:DIM],RkU0[0:ParticleCount][0:DIM],RkK1X[0:ParticleCount][0:DIM],RkK2X[0:ParticleCount][0:DIM],RkK3X[0:ParticleCount][0:DIM],RkK4X[0:ParticleCount][0:DIM],RkK1U[0:ParticleCount][0:DIM],RkK2U[0:ParticleCount][0:DIM],RkK3U[0:ParticleCount][0:DIM],RkK4U[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(VirialPressureAtParticle[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	
	return 0;
	
}

static void readDataFile(char *filename)
{
    FILE * fp;
    char buf[1024];
    const int reading_global=0;
    int mode=reading_global;
    

    fp=fopen(filename,"r");
    if(!fp){ log_printf("Cannot open data file %s\n",filename); exit(EXIT_FAILURE); }
    mode=reading_global;
    while(fp!=NULL && !feof(fp) && !ferror(fp)){
        if(fgets(buf,sizeof(buf),fp)!=NULL){
            if(buf[0]=='#'){}
            else if(sscanf(buf," Dt %lf",&Dt)==1){mode=reading_global;}
            else if(sscanf(buf," OutputInterval %lf",&OutputInterval)==1){mode=reading_global;}
            else if(sscanf(buf," VtkOutputInterval %lf",&VtkOutputInterval)==1){mode=reading_global;}
            else if(sscanf(buf," EndTime %lf",&EndTime)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioA %lf",&RadiusRatioA)==1){mode=reading_global;}
        	// else if(sscanf(buf," RadiusRatioG %lf",&RadiusRatioG)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioP %lf",&RadiusRatioP)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioV %lf",&RadiusRatioV)==1){mode=reading_global;}
            else if(sscanf(buf," Density %lf %lf %lf %lf",&Density[0],&Density[1],&Density[2],&Density[3])==4){mode=reading_global;}
            else if(sscanf(buf," ShearViscosity %lf %lf %lf %lf",&ShearViscosity[0],&ShearViscosity[1],&ShearViscosity[2],&ShearViscosity[3])==4){mode=reading_global;}
            else if(sscanf(buf," PpeMaxIteration %d",&PpeMaxIteration)==1){mode=reading_global;}
            else if(sscanf(buf," PpeTolerance %lf",&PpeTolerance)==1){mode=reading_global;}
            else if(sscanf(buf," PpeSurfLambdaMin %lf",&PpeSurfLambdaMin)==1){mode=reading_global;}
            else if(sscanf(buf," PpeSurfLambdaKeep %lf",&PpeSurfLambdaKeep)==1){mode=reading_global;}
            else if(sscanf(buf," PpeSurfPndMin %lf",&PpeSurfPndMin)==1){mode=reading_global;}
            else if(sscanf(buf," PpeSurfProbeRatio %lf",&PpeSurfProbeRatio)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseDensityRelaxTime %lf",&TwoPhaseDensityRelaxTime)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseMaxSpeed %lf",&TwoPhaseMaxSpeed)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseDtMinRatio %lf",&TwoPhaseDtMinRatio)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseOpenSurface %d",&TwoPhaseOpenSurface)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseSourceGamma %lf",&TwoPhaseSourceGamma)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseRhoSmooth %lf",&TwoPhaseRhoSmooth)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseShiftCoef %lf",&TwoPhaseShiftCoef)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseIfThreshold %lf",&TwoPhaseIfThreshold)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseIfShiftNormal %lf",&TwoPhaseIfShiftNormal)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseShiftVelCorr %lf",&TwoPhaseShiftVelCorr)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseDuCap %lf",&TwoPhaseDuCap)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseMinFluidNb %d",&TwoPhaseMinFluidNb)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseStModel %d",&TwoPhaseStModel)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseStSmoothRatio %lf",&TwoPhaseStSmoothRatio)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseStDoubleSmooth %d",&TwoPhaseStDoubleSmooth)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseStDensityScale %lf",&TwoPhaseStDensityScale)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseStZeroNet %lf",&TwoPhaseStZeroNet)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseStDeltaMin %lf",&TwoPhaseStDeltaMin)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhaseCurvMax %lf",&TwoPhaseCurvMax)==1){mode=reading_global;}
            else if(sscanf(buf," TwoPhasePpRadiusRatio %lf",&TwoPhasePpRadiusRatio)==1){mode=reading_global;}
            else if(sscanf(buf," WallNoSlip %d",&WallNoSlip)==1){mode=reading_global;}
            else if(sscanf(buf," WbpActivationRatio %lf",&WbpActivationRatio)==1){mode=reading_global;}
            else if(sscanf(buf," WbpWeightRatio %lf",&WbpWeightRatio)==1){mode=reading_global;}
            else if(sscanf(buf," WbpBcRelax %lf",&WbpBcRelax)==1){mode=reading_global;}
            else if(sscanf(buf," WbpAccelCapFactor %lf",&WbpAccelCapFactor)==1){mode=reading_global;}
            else if(sscanf(buf," WbpDistanceMinRatio %lf",&WbpDistanceMinRatio)==1){mode=reading_global;}
            else if(sscanf(buf," WbpDistanceMaxRatio %lf",&WbpDistanceMaxRatio)==1){mode=reading_global;}
            else if(sscanf(buf," SurfaceTension %lf %lf %lf %lf",&SurfaceTension[0],&SurfaceTension[1],&SurfaceTension[2],&SurfaceTension[3])==4){mode=reading_global;}
            else if(sscanf(buf," InterfaceTension01 %lf",&InterfaceTension01)==1){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type0) %lf %lf %lf %lf",&InteractionRatio[0][0],&InteractionRatio[0][1],&InteractionRatio[0][2],&InteractionRatio[0][3])==4){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type1) %lf %lf %lf %lf",&InteractionRatio[1][0],&InteractionRatio[1][1],&InteractionRatio[1][2],&InteractionRatio[1][3])==4){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type2) %lf %lf %lf %lf",&InteractionRatio[2][0],&InteractionRatio[2][1],&InteractionRatio[2][2],&InteractionRatio[2][3])==4){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type3) %lf %lf %lf %lf",&InteractionRatio[3][0],&InteractionRatio[3][1],&InteractionRatio[3][2],&InteractionRatio[3][3])==4){mode=reading_global;}
            else if(sscanf(buf," Gravity %lf %lf %lf", &Gravity[0], &Gravity[1], &Gravity[2])==3){mode=reading_global;}
            else if(sscanf(buf," Wall2  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[2][0],  &WallCenter[2][1],  &WallCenter[2][2],  &WallVelocity[2][0],  &WallVelocity[2][1],  &WallVelocity[2][2],  &WallOmega[2][0],  &WallOmega[2][1],  &WallOmega[2][2])==9){mode=reading_global;}
            else if(sscanf(buf," Wall3  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[3][0],  &WallCenter[3][1],  &WallCenter[3][2],  &WallVelocity[3][0],  &WallVelocity[3][1],  &WallVelocity[3][2],  &WallOmega[3][0],  &WallOmega[3][1],  &WallOmega[3][2])==9){mode=reading_global;}
            else{
                log_printf("Invalid line in data file \"%s\"\n", buf);
            }
        }
    }
    fclose(fp);
	
	#pragma acc enter data create(ParticleCount,ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	#pragma acc enter data create(Density[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc enter data create(InterfaceTension01)
	#pragma acc enter data create(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc enter data create(Gravity[0:DIM])
	#pragma acc enter data create(PpeMaxIteration,PpeTolerance)
	#pragma acc enter data create(PpeSurfLambdaMin,PpeSurfLambdaKeep,PpeSurfPndMin,PpeSurfProbeRatio,MrefP)
	#pragma acc enter data create(TwoPhaseSourceGamma,TwoPhaseRhoSmooth,TwoPhaseShiftCoef,TwoPhaseIfThreshold,PpePureNeumann)
	#pragma acc enter data create(TwoPhaseIfShiftNormal,TwoPhaseShiftVelCorr,TwoPhaseDuCap,TwoPhaseMinFluidNb)
	#pragma acc enter data create(TwoPhaseStModel,TwoPhaseStSmoothRatio,TwoPhaseStDoubleSmooth)
	#pragma acc enter data create(TwoPhaseStDensityScale,TwoPhaseStZeroNet,TwoPhaseStDeltaMin)
	#pragma acc enter data create(TwoPhaseCurvMax,TwoPhasePpRadiusRatio,WallNoSlip)
	#pragma acc enter data create(WbpActivationRatio,WbpWeightRatio,WbpBcRelax,WbpAccelCapFactor,WbpDistanceMinRatio,WbpDistanceMaxRatio)
	#pragma acc enter data create(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth[0:DIM],CellCount[0:DIM])
	#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM])
	#pragma acc enter data create(WallRotation[0:WALL_END][0:DIM][0:DIM])
	
}

static void readGridFile(char *filename)
{
    FILE *fp=fopen(filename,"r");
    if(!fp){ log_printf("Cannot open grid file %s\n",filename); exit(EXIT_FAILURE); }
	char buf[1024];   
	
	
	try{
		
		if(fgets(buf,sizeof(buf),fp)==NULL){ log_printf("Truncated grid header\n"); exit(EXIT_FAILURE); }
		sscanf(buf,"%lf",&Time);
		if(fgets(buf,sizeof(buf),fp)==NULL){ log_printf("Truncated grid header\n"); exit(EXIT_FAILURE); }
		if(sscanf(buf,"%d  %lf  %lf %lf %lf  %lf %lf %lf",
			&ParticleCount,
			&ParticleSpacing,
			&DomainMin[0], &DomainMax[0],
			&DomainMin[1], &DomainMax[1],
			&DomainMin[2], &DomainMax[2])!=8 || ParticleCount<=0 ||
            !(ParticleSpacing>0.0) || !isfinite(ParticleSpacing) ||
            !(DomainMax[0]>DomainMin[0]) || !(DomainMax[1]>DomainMin[1]) || !(DomainMax[2]>DomainMin[2])){
            log_printf("Invalid grid header\n"); exit(EXIT_FAILURE);
        }
		#ifdef TWO_DIMENSIONAL
		ParticleVolume = ParticleSpacing*ParticleSpacing;
		#else
		ParticleVolume = ParticleSpacing*ParticleSpacing*ParticleSpacing;
		#endif
		
		ParticleIndex = (int *)malloc(ParticleCount*sizeof(int));
		Property = (int *)malloc(ParticleCount*sizeof(int));
		Position = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		Velocity = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		DensityA = (double *)malloc(ParticleCount*sizeof(double));
		GravityCenter = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		PressureA = (double *)malloc(ParticleCount*sizeof(double));
		VolStrainP = (double *)malloc(ParticleCount*sizeof(double));
		DivergenceP = (double *)malloc(ParticleCount*sizeof(double));
		PressureP = (double *)malloc(ParticleCount*sizeof(double));
		VirialPressureAtParticle = (double *)malloc(ParticleCount*sizeof(double));
		VirialStressAtParticle = (double (*) [DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
		Mass = (double (*))malloc(ParticleCount*sizeof(double));
		Force = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		Mu = (double (*))malloc(ParticleCount*sizeof(double));
		
		#pragma acc enter data create(ParticleIndex[0:ParticleCount])          attach(ParticleIndex)
		#pragma acc enter data create(Property[0:ParticleCount])               attach(Property)
		#pragma acc enter data create(Position[0:ParticleCount][0:DIM])        attach(Position)
		#pragma acc enter data create(Velocity[0:ParticleCount][0:DIM])        attach(Velocity)
		#pragma acc enter data create(DensityA[0:ParticleCount])               attach(DensityA)
		#pragma acc enter data create(GravityCenter[0:ParticleCount][0:DIM])   attach(GravityCenter)
		#pragma acc enter data create(PressureA[0:ParticleCount])              attach(PressureA)
		#pragma acc enter data create(VolStrainP[0:ParticleCount])             attach(VolStrainP)
		#pragma acc enter data create(DivergenceP[0:ParticleCount])            attach(DivergenceP)
		#pragma acc enter data create(PressureP[0:ParticleCount])              attach(PressureP)
		#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount])               attach(VirialPressureAtParticle)
		#pragma acc enter data create(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])   attach(VirialStressAtParticle)
		#pragma acc enter data create(Mass[0:ParticleCount])                attach(Mass)
		#pragma acc enter data create(Force[0:ParticleCount][0:DIM])        attach(Force)
		#pragma acc enter data create(Mu[0:ParticleCount])                  attach(Mu)

		PpeFlag = (int *)malloc(ParticleCount*sizeof(int));
		PpeAlpha = (double *)malloc(ParticleCount*sizeof(double));
		PositionDivR = (double *)malloc(ParticleCount*sizeof(double));
		PpeDiag = (double *)malloc(ParticleCount*sizeof(double));
		PpeB = (double *)malloc(ParticleCount*sizeof(double));
		PpeR = (double *)malloc(ParticleCount*sizeof(double));
		PpeR0 = (double *)malloc(ParticleCount*sizeof(double));
		PpeP = (double *)malloc(ParticleCount*sizeof(double));
		PpePHat = (double *)malloc(ParticleCount*sizeof(double));
		PpeV = (double *)malloc(ParticleCount*sizeof(double));
		PpeS = (double *)malloc(ParticleCount*sizeof(double));
		PpeT = (double *)malloc(ParticleCount*sizeof(double));
		PpeOpX = (double *)malloc(ParticleCount*sizeof(double));
		PpeOpY = (double *)malloc(ParticleCount*sizeof(double));
		IsphQmlsUse = (int *)malloc(ParticleCount*sizeof(int));

		#pragma acc enter data create(PpeFlag[0:ParticleCount]) attach(PpeFlag)
		#pragma acc enter data create(PpeAlpha[0:ParticleCount]) attach(PpeAlpha)
		#pragma acc enter data create(PositionDivR[0:ParticleCount]) attach(PositionDivR)
		#pragma acc enter data create(PpeDiag[0:ParticleCount]) attach(PpeDiag)
		#pragma acc enter data create(PpeB[0:ParticleCount]) attach(PpeB)
		#pragma acc enter data create(PpeR[0:ParticleCount]) attach(PpeR)
		#pragma acc enter data create(PpeR0[0:ParticleCount]) attach(PpeR0)
		#pragma acc enter data create(PpeP[0:ParticleCount]) attach(PpeP)
		#pragma acc enter data create(PpePHat[0:ParticleCount]) attach(PpePHat)
		#pragma acc enter data create(PpeV[0:ParticleCount]) attach(PpeV)
		#pragma acc enter data create(PpeS[0:ParticleCount]) attach(PpeS)
		#pragma acc enter data create(PpeT[0:ParticleCount]) attach(PpeT)
		#pragma acc enter data create(PpeOpX[0:ParticleCount]) attach(PpeOpX)
		#pragma acc enter data create(PpeOpY[0:ParticleCount]) attach(PpeOpY)
		#pragma acc enter data create(IsphQmlsUse[0:ParticleCount]) attach(IsphQmlsUse)

		// Two-phase auxiliary fields.
		RhoEff   = (double *)malloc(ParticleCount*sizeof(double));
		ColorC   = (double *)malloc(ParticleCount*sizeof(double));
		IfDelta  = (double *)malloc(ParticleCount*sizeof(double));
		IfNormal = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		ColorS  = (double *)malloc(ParticleCount*sizeof(double));
		IfKappa = (double *)malloc(ParticleCount*sizeof(double));
		IfGrad  = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));

		#pragma acc enter data create(RhoEff[0:ParticleCount]) attach(RhoEff)
		#pragma acc enter data create(ColorC[0:ParticleCount]) attach(ColorC)
		#pragma acc enter data create(IfDelta[0:ParticleCount]) attach(IfDelta)
		#pragma acc enter data create(IfNormal[0:ParticleCount][0:DIM]) attach(IfNormal)
		#pragma acc enter data create(ColorS[0:ParticleCount]) attach(ColorS)
		#pragma acc enter data create(IfKappa[0:ParticleCount]) attach(IfKappa)
		#pragma acc enter data create(IfGrad[0:ParticleCount][0:DIM]) attach(IfGrad)

		RkX0  = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		RkU0  = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		RkK1X = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		RkK2X = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		RkK3X = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		RkK4X = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		RkK1U = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		RkK2U = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		RkK3U = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		RkK4U = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));

		#pragma acc enter data create(RkX0[0:ParticleCount][0:DIM]) attach(RkX0)
		#pragma acc enter data create(RkU0[0:ParticleCount][0:DIM]) attach(RkU0)
		#pragma acc enter data create(RkK1X[0:ParticleCount][0:DIM]) attach(RkK1X)
		#pragma acc enter data create(RkK2X[0:ParticleCount][0:DIM]) attach(RkK2X)
		#pragma acc enter data create(RkK3X[0:ParticleCount][0:DIM]) attach(RkK3X)
		#pragma acc enter data create(RkK4X[0:ParticleCount][0:DIM]) attach(RkK4X)
		#pragma acc enter data create(RkK1U[0:ParticleCount][0:DIM]) attach(RkK1U)
		#pragma acc enter data create(RkK2U[0:ParticleCount][0:DIM]) attach(RkK2U)
		#pragma acc enter data create(RkK3U[0:ParticleCount][0:DIM]) attach(RkK3U)
		#pragma acc enter data create(RkK4U[0:ParticleCount][0:DIM]) attach(RkK4U)
		
		TmpIntScalar = (int *)malloc(ParticleCount*sizeof(int));
		TmpDoubleScalar = (double *)malloc(ParticleCount*sizeof(double));
		TmpDoubleVector = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		NeighborCount  = (int *)malloc(ParticleCount*sizeof(int));
		NeighborCalculatedPosition = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
		
		#pragma acc enter data create(TmpIntScalar[0:ParticleCount]) attach(TmpIntScalar)
		#pragma acc enter data create(TmpDoubleScalar[0:ParticleCount]) attach(TmpDoubleScalar)
		#pragma acc enter data create(TmpDoubleVector[0:ParticleCount][0:DIM]) attach(TmpDoubleVector)
		#pragma acc enter data create(NeighborCount[0:ParticleCount]) attach(NeighborCount)
		#pragma acc enter data create(NeighborCalculatedPosition[0:ParticleCount][0:DIM]) attach(NeighborCalculatedPosition)
		
		// calculate minimun PowerParticleCount which sataisfies  ParticleCount < PowerParticleCount = pow(2,ParticleCountPower) 
		ParticleCountPower=0;
		while((ParticleCount>>ParticleCountPower)!=0){
			++ParticleCountPower;
		}
		PowerParticleCount = (1<<ParticleCountPower);
		fprintf(stderr,"memory for CellIndex and CellParticle %d\n", PowerParticleCount );
		CellIndex    = (int *)malloc( (PowerParticleCount) * sizeof(int) );
		CellParticle = (int *)malloc( (PowerParticleCount) * sizeof(int) );
		#pragma acc enter data create(CellIndex   [0:PowerParticleCount]) attach(CellIndex)
		#pragma acc enter data create(CellParticle[0:PowerParticleCount]) attach(CellParticle)

		NeighborPtr  = (long int *)malloc( (PowerParticleCount) * sizeof(long int) );
		#pragma acc enter data create(NeighborPtr[0:PowerParticleCount]) attach(NeighborPtr) 
		#pragma acc update device(ParticleCountPower,PowerParticleCount)

		
		double (*q)[DIM] = Position;
		double (*v)[DIM] = Velocity;
		
		for(int iP=0;iP<ParticleCount;++iP){
			if(fgets(buf,sizeof(buf),fp)==NULL){ log_printf("Truncated grid particles\n"); exit(EXIT_FAILURE); }
			if(sscanf(buf,"%d  %lf %lf %lf  %lf %lf %lf",
				&Property[iP],
				&q[iP][0],&q[iP][1],&q[iP][2],
				&v[iP][0],&v[iP][1],&v[iP][2]
			)!=7 || Property[iP]<0 || Property[iP]>=TYPE_COUNT){ log_printf("Invalid particle row\n"); exit(EXIT_FAILURE); }
		}
	}catch(...){};
	
	fclose(fp);
	
	// set particle index
	for(int iP=0;iP<ParticleCount;++iP){
		ParticleIndex[iP]=iP;
        PressureP[iP]=0.0; PpeFlag[iP]=0;
	}
	
	// set begin & end
	FluidParticleBegin=0; FluidParticleEnd=0;
	WallParticleBegin=ParticleCount; WallParticleEnd=ParticleCount;
	for(int iP=0;iP<ParticleCount;++iP){
		if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END){
			FluidParticleEnd=iP+1;
		}
		if(WALL_BEGIN<=Property[iP] && Property[iP]<WALL_END){
			if(WallParticleBegin==ParticleCount) WallParticleBegin=iP;
			WallParticleEnd=iP+1;
		}
	}
	if(WallParticleBegin==ParticleCount) WallParticleBegin=WallParticleEnd=ParticleCount;
	
	#pragma acc update device(ParticleCount,ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(Property[0:ParticleCount],PressureP[0:ParticleCount],PpeFlag[0:ParticleCount])
	#pragma acc update device(Position[0:ParticleCount][0:DIM])
	#pragma acc update device(Velocity[0:ParticleCount][0:DIM])
	#pragma acc update device(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	
}

static void writeProfFile(const char *filename)
{
    #pragma acc update self(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
    FILE *fp=fopen(filename,"w");

    fprintf(fp,"%e\n",Time);
    fprintf(fp,"%d %e %e %e %e %e %e %e\n",
            ParticleCount,
            ParticleSpacing,
            DomainMin[0], DomainMax[0],
            DomainMin[1], DomainMax[1],
            DomainMin[2], DomainMax[2]);

    const double (*q)[DIM] = Position;
    const double (*v)[DIM] = Velocity;

    for(int iP=0;iP<ParticleCount;++iP){
            fprintf(fp,"%d %e %e %e  %e %e %e\n",
                    Property[iP],
                    q[iP][0], q[iP][1], q[iP][2],
                    v[iP][0], v[iP][1], v[iP][2]
            );
    }
    fflush(fp);
    fclose(fp);
}

static void writeVtkFile(const char *filename)
{
	// update parameters to be output
	#pragma acc update host(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],VirialPressureAtParticle[0:ParticleCount])
	#pragma acc update host(DensityA[0:ParticleCount],VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
	#pragma acc update host(PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],PositionDivR[0:ParticleCount],NeighborCount[0:ParticleCount],Force[0:ParticleCount][0:DIM])
	#pragma acc update host(RhoEff[0:ParticleCount],ColorC[0:ParticleCount],IfDelta[0:ParticleCount],IfKappa[0:ParticleCount],IfNormal[0:ParticleCount][0:DIM])

    const double (*q)[DIM] = Position;
    const double (*v)[DIM] = Velocity;

    FILE *fp=fopen(filename, "w");

    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "Unstructured Grid Example\n");
    fprintf(fp, "ASCII\n");

    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "POINTS %d float\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e %e %e\n", (float)q[iP][0], (float)q[iP][1], (float)q[iP][2]);
    }
    fprintf(fp, "CELLS %d %d\n", ParticleCount, 2*ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 %d ",iP);
    }
    fprintf(fp, "\n");
    fprintf(fp, "CELL_TYPES %d\n", ParticleCount);
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "1 ");
    }
    fprintf(fp, "\n");

    fprintf(fp, "\n");

    fprintf(fp, "POINT_DATA %d\n", ParticleCount);
    fprintf(fp, "SCALARS label float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", Property[iP]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS DensityA float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n", (float)DensityA[iP]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS VolStrainP float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n", (float)VolStrainP[iP]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS PressureP float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n", (float)PressureP[iP]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "SCALARS PressureP_Fluid float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         const float pp = (FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)
                        ? (float)PressureP[iP] : 0.0f;
         fprintf(fp, "%e\n", pp);
    }
    fprintf(fp, "\n");

    fprintf(fp, "SCALARS DivergenceP float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n", (float)DivergenceP[iP]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "SCALARS PpeActive float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", PpeFlag[iP]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS PpeAlpha float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){ fprintf(fp, "%e\n", (float)PpeAlpha[iP]); }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS PositionDivR float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){ fprintf(fp, "%e\n", (float)PositionDivR[iP]); }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS RhoEff float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){ fprintf(fp, "%e\n", (float)RhoEff[iP]); }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS ColorC float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){ fprintf(fp, "%e\n", (float)ColorC[iP]); }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS InterfaceDelta float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){ fprintf(fp, "%e\n", (float)IfDelta[iP]); }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS InterfaceKappa float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){ fprintf(fp, "%e\n", (float)IfKappa[iP]); }
    fprintf(fp, "\n");
    fprintf(fp, "VECTORS interface_normal float\n");
    for(int iP=0;iP<ParticleCount;++iP){
        fprintf(fp, "%e %e %e\n", (float)IfNormal[iP][0], (float)IfNormal[iP][1], (float)IfNormal[iP][2]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS VirialPressureAtParticle float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n", (float)VirialPressureAtParticle[iP]);
    }
	fprintf(fp, "\n");
    fprintf(fp, "SCALARS neighbor float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", NeighborCount[iP]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "VECTORS velocity float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)v[iP][0], (float)v[iP][1], (float)v[iP][2]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "VECTORS force float\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e %e %e\n", (float)Force[iP][0], (float)Force[iP][1], (float)Force[iP][2]);
    }
    fprintf(fp, "\n");
	
    fflush(fp);
    fclose(fp);
}

static void initializeWeight()
{
	RadiusRatioG = RadiusRatioA;
	
	RadiusA = RadiusRatioA*ParticleSpacing;
	RadiusG = RadiusRatioG*ParticleSpacing;
	RadiusP = RadiusRatioP*ParticleSpacing;
	RadiusV = RadiusRatioV*ParticleSpacing;
	
	
#ifdef TWO_DIMENSIONAL
		Swa = 1.0/2.0 * 2.0/15.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swg = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swp = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swv = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		R2g = 1.0/2.0 * 1.0/30.0* M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing /Swg;
#else	//code for three dimensional
		Swa = 1.0/3.0 * 1.0/5.0*M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swg = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swp = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swv = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		R2g = 1.0/3.0 * 4.0/105.0*M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing/ParticleSpacing /Swg;
#endif
	
	
	    {// N0a
        const double radius_ratio = RadiusA/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusA*RadiusA){
                        const double rij = sqrt(rij2);
                        const double wij = wa(rij,RadiusA);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else	//code for three dimensional
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusA*RadiusA){
                            const double rij = sqrt(rij2);
                            const double wij = wa(rij,RadiusA);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0a = sum;
        log_printf("N0a = %e, count=%d\n", N0a, count);
    }	

    {// N0p
        const double radius_ratio = RadiusP/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusP*RadiusP){
                        const double rij = sqrt(rij2);
                        const double wij = wp(rij,RadiusP);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else	//code for three dimensional
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusP*RadiusP){
                            const double rij = sqrt(rij2);
                            const double wij = wp(rij,RadiusP);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0p = sum;
        log_printf("N0p = %e, count=%d\n", N0p, count);
    }

    // Legacy second moment.
    {
        const double radius_ratio = RadiusP/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        double sumW = 0.0;
        double sumR2W = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(iX==0 && iY==0) continue;
                const double x=ParticleSpacing*(double)iX;
                const double y=ParticleSpacing*(double)iY;
                const double r2=x*x+y*y;
                if(r2<=RadiusP*RadiusP){
                    const double r=sqrt(r2);
                    const double w=wp(r,RadiusP);
                    sumW += w;
                    sumR2W += r2*w;
                }
            }
        }
#else
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(iX==0 && iY==0 && iZ==0) continue;
                    const double x=ParticleSpacing*(double)iX;
                    const double y=ParticleSpacing*(double)iY;
                    const double z=ParticleSpacing*(double)iZ;
                    const double r2=x*x+y*y+z*z;
                    if(r2<=RadiusP*RadiusP){
                        const double r=sqrt(r2);
                        const double w=wp(r,RadiusP);
                        sumW += w;
                        sumR2W += r2*w;
                    }
                }
            }
        }
#endif
        PpeLambdaP = (sumW>0.0 ? sumR2W/sumW : ParticleSpacing*ParticleSpacing);
        log_printf("PpeLambdaP = %e\n",PpeLambdaP);
    }

    // Discrete normalization for the higher-order Laplacian.
    {
        const double radius_ratio = RadiusP/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        double rawQuadratic = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(iX==0 && iY==0) continue;
                const double x=ParticleSpacing*(double)iX;
                const double y=ParticleSpacing*(double)iY;
                const double r2=x*x+y*y;
                if(r2>1.0e-30 && r2<RadiusP*RadiusP){
                    const double r=sqrt(r2);
                    const double rMin=IsphMinPairRatio*ParticleSpacing;
                    const double rEff=(r>rMin ? r : rMin);
                    const double cRaw=(d2wpdr2(r,RadiusP)-dwpdr(r,RadiusP)/rEff)/N0p;
                    if(cRaw>0.0) rawQuadratic += cRaw*r2;
                }
            }
        }
        PpeHlNorm=(rawQuadratic>1.0e-30 ? 4.0/rawQuadratic : 1.0);
#else
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(iX==0 && iY==0 && iZ==0) continue;
                    const double x=ParticleSpacing*(double)iX;
                    const double y=ParticleSpacing*(double)iY;
                    const double z=ParticleSpacing*(double)iZ;
                    const double r2=x*x+y*y+z*z;
                    if(r2>1.0e-30 && r2<RadiusP*RadiusP){
                        const double r=sqrt(r2);
                        const double rMin=IsphMinPairRatio*ParticleSpacing;
                        const double rEff=(r>rMin ? r : rMin);
                        const double cRaw=(d2wpdr2(r,RadiusP)-dwpdr(r,RadiusP)/rEff)/N0p;
                        if(cRaw>0.0) rawQuadratic += cRaw*r2;
                    }
                }
            }
        }
        PpeHlNorm=(rawQuadratic>1.0e-30 ? 6.0/rawQuadratic : 1.0);
#endif
        log_printf("PpeHlNorm = %e, raw quadratic Laplacian = %e\n",PpeHlNorm,rawQuadratic);
    }

    // Interior reference value of the renormalisation matrix.  It is used by
    // the free-surface detector AND as the fallback normalisation of the
    // renormalised colour gradient of the two-phase surface tension.
    {
        const double radius_ratio = RadiusP/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        const double Vj = ParticleVolume;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(iX==0 && iY==0) continue;
                const double x=ParticleSpacing*(double)iX;
                const double y=ParticleSpacing*(double)iY;
                const double r2=x*x+y*y;
                if(r2<=RadiusP*RadiusP){
                    const double r=sqrt(r2);
                    const double g=-dwpdr(r,RadiusP)/r;
                    sum += Vj*g*x*x;
                }
            }
        }
#else
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(iX==0 && iY==0 && iZ==0) continue;
                    const double x=ParticleSpacing*(double)iX;
                    const double y=ParticleSpacing*(double)iY;
                    const double z=ParticleSpacing*(double)iZ;
                    const double r2=x*x+y*y+z*z;
                    if(r2<=RadiusP*RadiusP){
                        const double r=sqrt(r2);
                        const double g=-dwpdr(r,RadiusP)/r;
                        sum += Vj*g*x*x;
                    }
                }
            }
        }
#endif
        MrefP = (sum>0.0 ? sum : 1.0);
        log_printf("MrefP = %e\n",MrefP);
    }


    {
        const double radius_ratio=RadiusP/ParticleSpacing;
        const int range=(int)(radius_ratio+3.0);
        double sum=0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX) for(int iY=-range;iY<=range;++iY){
            if(iX==0 && iY==0) continue;
            const double dx=ParticleSpacing*(double)iX, dy=ParticleSpacing*(double)iY;
            const double r2=dx*dx+dy*dy;
            if(r2>1.0e-30 && r2<RadiusP*RadiusP){ const double r=sqrt(r2); sum+=ParticleVolume*(-dwpdr(r,RadiusP))*r; }
        }
#else
        for(int iX=-range;iX<=range;++iX) for(int iY=-range;iY<=range;++iY) for(int iZ=-range;iZ<=range;++iZ){
            if(iX==0 && iY==0 && iZ==0) continue;
            const double dx=ParticleSpacing*(double)iX, dy=ParticleSpacing*(double)iY, dz=ParticleSpacing*(double)iZ;
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2>1.0e-30 && r2<RadiusP*RadiusP){ const double r=sqrt(r2); sum+=ParticleVolume*(-dwpdr(r,RadiusP))*r; }
        }
#endif
        PositionDivRef=(sum>1.0e-30 ? sum : 1.0);
        log_printf("PositionDivRef = %e\n",PositionDivRef);
    }

    // -------- colour-smoothing radius and its interior M diagonal -----------
    RadiusST=(TwoPhaseStSmoothRatio>0.0 ? TwoPhaseStSmoothRatio : 1.0)*RadiusP;
    {
        const int range=(int)(RadiusST/ParticleSpacing+3.0);
        const double Vj=ParticleVolume;
        double sum=0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX) for(int iY=-range;iY<=range;++iY){
            if(iX==0 && iY==0) continue;
            const double x=ParticleSpacing*(double)iX, y=ParticleSpacing*(double)iY;
            const double r2=x*x+y*y;
            if(r2<=RadiusST*RadiusST){
                const double r=sqrt(r2);
                sum+=Vj*(-dwpdr(r,RadiusST)/r)*x*x;
            }
        }
#else
        for(int iX=-range;iX<=range;++iX) for(int iY=-range;iY<=range;++iY) for(int iZ=-range;iZ<=range;++iZ){
            if(iX==0 && iY==0 && iZ==0) continue;
            const double x=ParticleSpacing*(double)iX, y=ParticleSpacing*(double)iY, z=ParticleSpacing*(double)iZ;
            const double r2=x*x+y*y+z*z;
            if(r2<=RadiusST*RadiusST){
                const double r=sqrt(r2);
                sum+=Vj*(-dwpdr(r,RadiusST)/r)*x*x;
            }
        }
#endif
        MrefST=(sum>0.0 ? sum : 1.0);
        log_printf("RadiusST/dp = %e, MrefST = %e\n",RadiusST/ParticleSpacing,MrefST);
    }

    // -------- pairwise-potential calibration (TwoPhaseStModel = 2) ----------
    // Broken-bond interfacial energy of a flat interface between two
    // half-lattices.  U(r) = PpCoef * (re/3)(1-r/re)^3 acts between UNLIKE
    // particles only, so it produces sigma_01 and nothing else:
    //     sigma = (1/dp) * sum_{i in one column, y<0} sum_{j, y>0} U(r_ij).
    PpRadius=(TwoPhasePpRadiusRatio>0.0 ? TwoPhasePpRadiusRatio : 3.1)*ParticleSpacing;
    {
        const int range=(int)ceil(PpRadius/ParticleSpacing)+1;
        double s=0.0;
#ifdef TWO_DIMENSIONAL
        for(int my=0;my<range;++my){
            for(int ny=0;ny<range;++ny){
                for(int kx=-range;kx<=range;++kx){
                    const double dx=ParticleSpacing*(double)kx;
                    const double dy=ParticleSpacing*(double)(my+ny+1);
                    const double r=sqrt(dx*dx+dy*dy);
                    if(r<PpRadius){
                        const double q=1.0-r/PpRadius;
                        s+=(PpRadius/3.0)*q*q*q;
                    }
                }
            }
        }
        s/=ParticleSpacing;
#else
        s=0.0;
#endif
        PpCoef=((s>1.0e-30 && InterfaceTension01>0.0) ? InterfaceTension01/s : 0.0);
        log_printf("PP surface tension: re/dp=%e latticeSum=%e PpCoef=%e\n",
                   PpRadius/ParticleSpacing,s,PpCoef);
    }

	#pragma acc update device(RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g,PpeLambdaP,PpeHlNorm,MrefP,PositionDivRef,RadiusST,MrefST,PpRadius,PpCoef)
	

}


static void initializeFluid()
{
    int nType0=0, nType1=0;
    for(int iP=0;iP<ParticleCount;++iP){
        if(Property[iP]==0) ++nType0;
        if(Property[iP]==1) ++nType1;
    }
    TwoPhaseActive=(nType0>0 && nType1>0) ? 1 : 0;
    TwoPhaseGauge=-1;
    log_printf("Fluid mode: %s (Type0=%d Type1=%d)\n",
               (TwoPhaseActive ? "TWO-PHASE variable-density ISPH (mixed source, projected null space)" : "single-phase baseline"),
               nType0,nType1);

    if(TwoPhaseActive){
        const double rhoMin=(Density[0]<Density[1] ? Density[0] : Density[1]);
        const double rhoMax=(Density[0]>Density[1] ? Density[0] : Density[1]);
        const double pi=3.14159265358979323846;
        log_printf("TwoPhase: density ratio = %e (rho_heavy/rho_light)\n",
                   rhoMax/(rhoMin+1.0e-30));
        log_printf("TwoPhase parameters: SourceGamma=%e RhoSmooth=%e ShiftCoef=%e IfThreshold=%e\n",
                   TwoPhaseSourceGamma,TwoPhaseRhoSmooth,TwoPhaseShiftCoef,TwoPhaseIfThreshold);
        log_printf("TwoPhase parameters: IfShiftNormal=%e ShiftVelCorr=%e DuCap=%e MinFluidNb=%d\n",
                   TwoPhaseIfShiftNormal,TwoPhaseShiftVelCorr,TwoPhaseDuCap,TwoPhaseMinFluidNb);
        log_printf("TwoPhase surface tension: model=%d (0=CSF,1=CSS,2=PP) smooth=%e doubleSmooth=%d densityScale=%e zeroNet=%e deltaMin=%e curvMax=%e ppRe/dp=%e WallNoSlip=%d\n",
                   TwoPhaseStModel,TwoPhaseStSmoothRatio,TwoPhaseStDoubleSmooth,
                   TwoPhaseStDensityScale,TwoPhaseStZeroNet,TwoPhaseStDeltaMin,
                   TwoPhaseCurvMax,TwoPhasePpRadiusRatio,WallNoSlip);
        if(!(InterfaceTension01>0.0)) log_printf("InterfaceTension01=0: capillary forces disabled.\n");
        log_printf("Two-phase CSF model 0 uses pair-stencil forces; StDensityScale and StZeroNet are legacy options.\n");
        if(!(Dt>0.0) || !isfinite(Dt) || !(ParticleSpacing>0.0) ||
           !(Density[0]>0.0) || !(Density[1]>0.0) || ShearViscosity[0]<0.0 || ShearViscosity[1]<0.0 ||
           !(PpeTolerance>0.0) || PpeMaxIteration<1 ||
           !(TwoPhaseDtMinRatio>0.0 && TwoPhaseDtMinRatio<1.0) ||
           !isfinite(TwoPhaseSourceGamma) || TwoPhaseSourceGamma<0.0 || TwoPhaseSourceGamma>1.0 ||
           !(TwoPhaseDensityRelaxTime>0.0) || !isfinite(TwoPhaseDensityRelaxTime) ||
           TwoPhaseMaxSpeed<0.0 || !isfinite(TwoPhaseMaxSpeed)){

            log_printf("Invalid two-phase physical/solver parameters\n"); exit(EXIT_FAILURE);
        }


        double wallSpeed=0.0, initialSpeed=0.0;
        const double lx=DomainMax[0]-DomainMin[0], ly=DomainMax[1]-DomainMin[1];
        const double length=sqrt(lx*lx+ly*ly);
        const double grav=sqrt(Gravity[0]*Gravity[0]+Gravity[1]*Gravity[1]);
        for(int i=0;i<ParticleCount;++i){
            const double u=sqrt(Velocity[i][0]*Velocity[i][0]+Velocity[i][1]*Velocity[i][1]);
            if(u>initialSpeed) initialSpeed=u;
        }
        for(int t=WALL_BEGIN;t<WALL_END;++t){
            const double u=sqrt(WallVelocity[t][0]*WallVelocity[t][0]+WallVelocity[t][1]*WallVelocity[t][1]);
            const double w=sqrt(WallOmega[t][0]*WallOmega[t][0]+WallOmega[t][1]*WallOmega[t][1]+WallOmega[t][2]*WallOmega[t][2]);
            wallSpeed=fmax(wallSpeed,u+w*length);
        }
        const double capSpeed=(InterfaceTension01>0.0 ? sqrt(InterfaceTension01/(rhoMin*ParticleSpacing)) : 0.0);
        const double scale=fmax(initialSpeed,fmax(wallSpeed,fmax(sqrt(grav*length),capSpeed)));
        // A strictly quiescent, unforced case gets a finite roundoff allowance.
        TwoPhaseSpeedLimit=(TwoPhaseMaxSpeed>0.0 ? TwoPhaseMaxSpeed : 10.0*fmax(scale,1e-6));
        log_printf("TwoPhase guards: speed_limit=%e density_relax_time=%e gamma=%e (fixed-time penalty)\n",
                   TwoPhaseSpeedLimit,TwoPhaseDensityRelaxTime,TwoPhaseSourceGamma);

        // ---------------------------------------------------------------
        // Explicit-stage time-step advisories.  At a 1:1000 density ratio the
        // light phase sets every explicit limit, because a = F/m and m is
        // 1000x smaller.  Exceeding the capillary or viscous limit is by far
        // the most common cause of "the bubble explodes after a few steps".
        // ---------------------------------------------------------------
        const double gmag=sqrt(Gravity[0]*Gravity[0]+Gravity[1]*Gravity[1]+Gravity[2]*Gravity[2]);
        const double dtGrav=(gmag>0.0 ? 0.25*sqrt(ParticleSpacing/gmag) : 1.0e100);
        double nuMax=0.0;
        for(int iT=FLUID_BEGIN;iT<FLUID_END;++iT){
            const double nu=ShearViscosity[iT]/(Density[iT]+1.0e-30);
            if(nu>nuMax) nuMax=nu;
        }
        const double dtVisc=(nuMax>0.0 ? 0.125*ParticleSpacing*ParticleSpacing/nuMax : 1.0e100);
const double rhoStCap=(TwoPhaseStDensityScale>0.0 ? 0.5*(Density[0]+Density[1]) : rhoMin);
        const double dtCap=(InterfaceTension01>0.0
            ? 0.5*sqrt(rhoStCap*ParticleSpacing*ParticleSpacing*ParticleSpacing/(2.0*pi*InterfaceTension01))
            : 1.0e100);
        log_printf("TwoPhase dt limits: gravity=%e viscous=%e capillary=%e  (current Dt=%e)\n",
                   dtGrav,dtVisc,dtCap,Dt);
        if(Dt>dtVisc || Dt>dtCap || Dt>dtGrav){
            log_printf("*** WARNING: Dt exceeds an explicit stability limit above. ***\n");
            log_printf("*** Reduce Dt; the two-phase explicit stage (viscosity + surface tension) will diverge otherwise. ***\n");
        }
    }
    #pragma acc update device(TwoPhaseActive,TwoPhaseGauge)
	for(int iP=0;iP<ParticleCount;++iP){
		Mass[iP]=Density[Property[iP]]*ParticleVolume;
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Mu[iP]=ShearViscosity[Property[iP]];
	}
	#ifdef TWO_DIMENSIONAL
	CofK = 0.350778153;
	double integN=0.024679383;
	double integX=0.226126699;
	#else 
	CofK = 0.326976006;
	double integN=0.021425779;
	double integX=0.233977488;
	#endif
	
	for(int iT=0;iT<TYPE_COUNT;++iT){
		CofA[iT]=SurfaceTension[iT] / ((RadiusG/ParticleSpacing)*(integN+CofK*CofK*integX));
	}

    // Kondo, Koshizuka & Takimoto (2007) surface-tension potential
    // (single-phase free-surface cohesion only).
    SurfacePotentialRadius = 3.2*ParticleSpacing;
    double surfaceShapeSum = 0.0;
#ifdef TWO_DIMENSIONAL
    {
        const int range=(int)ceil(SurfacePotentialRadius/ParticleSpacing);
        for(int iy=1; iy<=range; ++iy){
            for(int ix=-range; ix<=range; ++ix){
                const double dx=ParticleSpacing*(double)ix;
                const double dy=ParticleSpacing*(double)iy;
                const double r=sqrt(dx*dx+dy*dy);
                if(r>=SurfacePotentialRadius || r<=1.0e-30) continue;
                const double shape=(r-1.5*ParticleSpacing+0.5*SurfacePotentialRadius)
                                  *(r-SurfacePotentialRadius)*(r-SurfacePotentialRadius);
                surfaceShapeSum += (double)iy*shape;
            }
        }
    }
#else
    surfaceShapeSum = 0.0;
#endif
    for(int iT=0;iT<TYPE_COUNT;++iT){
#ifdef TWO_DIMENSIONAL
        SurfacePotentialC[iT]=(surfaceShapeSum>1.0e-30
            ? 6.0*SurfaceTension[iT]*ParticleSpacing*ParticleSpacing/surfaceShapeSum
            : 0.0);
#else
        SurfacePotentialC[iT]=0.0;
#endif
    }
    log_printf("KondoST: re/dp=%e shapeSum/dp^3=%e C=[%e %e %e %e]\n",
               SurfacePotentialRadius/ParticleSpacing,
               (ParticleSpacing>0.0 ? surfaceShapeSum/(ParticleSpacing*ParticleSpacing*ParticleSpacing) : 0.0),
               SurfacePotentialC[0],SurfacePotentialC[1],SurfacePotentialC[2],SurfacePotentialC[3]);
	
	#pragma acc update device(Mass[0:ParticleCount])
	#pragma acc update device(Mu[0:ParticleCount])
	#pragma acc update device(CofK,CofA[0:TYPE_COUNT])
    #pragma acc update device(SurfacePotentialRadius,SurfacePotentialC[0:TYPE_COUNT])
}



static void initializeWall()
{
	
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		
		double theta;
		double normal[DIM]={0.0,0.0,0.0};
		double q[DIM+1];
		double t[DIM];
		double (&R)[DIM][DIM]=WallRotation[iProp];
		
		theta = sqrt(WallOmega[iProp][0]*WallOmega[iProp][0]+WallOmega[iProp][1]*WallOmega[iProp][1]+WallOmega[iProp][2]*WallOmega[iProp][2]);
		if(theta!=0.0){
			for(int iD=0;iD<DIM;++iD){
				normal[iD]=WallOmega[iProp][iD]/theta;
			}
		}
		q[0]=normal[0]*sin(theta*Dt/2.0);
		q[1]=normal[1]*sin(theta*Dt/2.0);
		q[2]=normal[2]*sin(theta*Dt/2.0);
		q[3]=cos(theta*Dt/2.0);
		t[0]=WallVelocity[iProp][0]*Dt;
		t[1]=WallVelocity[iProp][1]*Dt;
		t[2]=WallVelocity[iProp][2]*Dt;
		
		R[0][0] = q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[0][1] = 2.0*(q[0]*q[1]-q[2]*q[3]);
		R[0][2] = 2.0*(q[0]*q[2]+q[1]*q[3]);
		
		R[1][0] = 2.0*(q[0]*q[1]+q[2]*q[3]);
		R[1][1] = -q[0]*q[0]+q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[1][2] = 2.0*(q[1]*q[2]-q[0]*q[3]);
		
		R[2][0] = 2.0*(q[0]*q[2]-q[1]*q[3]);
		R[2][1] = 2.0*(q[1]*q[2]+q[0]*q[3]);
		R[2][2] = -q[0]*q[0]-q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
		
	}
	#pragma acc update device(WallRotation[0:WALL_END][0:DIM][0:DIM])
}

static void initializeDomain( void )
{
	
	
	MaxRadius = ((RadiusA>MaxRadius) ? RadiusA : MaxRadius);
	MaxRadius = ((RadiusG>MaxRadius) ? RadiusG : MaxRadius);
	MaxRadius = ((RadiusP>MaxRadius) ? RadiusP : MaxRadius);
	MaxRadius = ((RadiusV>MaxRadius) ? RadiusV : MaxRadius);
    if(TwoPhaseActive){
        MaxRadius = ((RadiusST>MaxRadius) ? RadiusST : MaxRadius);
        if(TwoPhaseStModel==2){
            MaxRadius = ((PpRadius>MaxRadius) ? PpRadius : MaxRadius);
        }
    }
    // The Kondo potential is single-phase only; do not inflate the cell size
    // (and hence the neighbour list) for a two-phase run.
    if(!TwoPhaseActive){
        MaxRadius = ((SurfacePotentialRadius>MaxRadius) ? SurfacePotentialRadius : MaxRadius);
    }

	DomainWidth[0] = DomainMax[0] - DomainMin[0];
	DomainWidth[1] = DomainMax[1] - DomainMin[1];
	DomainWidth[2] = DomainMax[2] - DomainMin[2];
	
	double cellCount[DIM];
	
	cellCount[0] = floor((DomainMax[0] - DomainMin[0])/(MaxRadius+MARGIN));
	cellCount[1] = floor((DomainMax[1] - DomainMin[1])/(MaxRadius+MARGIN));
	#ifdef TWO_DIMENSIONAL
	cellCount[2] = 1;
	#else
	cellCount[2] = floor((DomainMax[2] - DomainMin[2])/(MaxRadius+MARGIN));
	#endif
	
	CellCount[0] = (int)fmax(1.0,cellCount[0]);
	CellCount[1] = (int)fmax(1.0,cellCount[1]);
	CellCount[2] = (int)cellCount[2];
	TotalCellCount   = CellCount[0]*CellCount[1]*CellCount[2];
	log_printf("line:%d CellCount[DIM]: %d %d %d\n", __LINE__, CellCount[0], CellCount[1], CellCount[2]);
	log_printf("line:%d TotalCellCount: %d\n", __LINE__, TotalCellCount);
	
	CellWidth[0]=DomainWidth[0]/CellCount[0];
	CellWidth[1]=DomainWidth[1]/CellCount[1];
	CellWidth[2]=DomainWidth[2]/CellCount[2];
	log_printf("line:%d CellWidth[DIM]: %e %e %e\n", __LINE__, CellWidth[0], CellWidth[1], CellWidth[2]);

	
	CellFluidParticleBegin = (int *)malloc( (TotalCellCount) * sizeof(int) );
	CellFluidParticleEnd   = (int *)malloc( (TotalCellCount) * sizeof(int) );
	CellWallParticleBegin = (int *)malloc( (TotalCellCount) * sizeof(int) );
	CellWallParticleEnd   = (int *)malloc( (TotalCellCount) * sizeof(int) );
	#pragma acc enter data create(CellFluidParticleBegin[0:TotalCellCount]) attach(CellFluidParticleBegin)
	#pragma acc enter data create(CellFluidParticleEnd[0:TotalCellCount]) attach(CellFluidParticleEnd)
	#pragma acc enter data create(CellWallParticleBegin[0:TotalCellCount]) attach(CellWallParticleBegin)
	#pragma acc enter data create(CellWallParticleEnd[0:TotalCellCount]) attach(CellWallParticleEnd)
	
	#pragma acc update device(MaxRadius)	
	#pragma acc update device(CellWidth[0:DIM],CellCount[0:DIM],TotalCellCount)
	#pragma acc update device(DomainMax[0:DIM],DomainMin[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCountPower,PowerParticleCount)
}

static int neighborCalculation( void ){
	double maxShift2=0.0;
	#pragma acc parallel loop reduction(max:maxShift2) present(NeighborCalculatedPosition[0:ParticleCount][0:DIM],Position[0:ParticleCount][0:DIM])
	#pragma omp parallel for reduction (max:maxShift2)
	for(int iP=0;iP<ParticleCount;++iP){
		 double disp[DIM];
         #pragma acc loop seq
         for(int iD=0;iD<DIM;++iD){
            disp[iD] = Position[iP][iD] - NeighborCalculatedPosition[iP][iD];
         }
		const double shift2 = disp[0]*disp[0]+disp[1]*disp[1]+disp[2]*disp[2];
		if(shift2>maxShift2){
			maxShift2=shift2;
		}
	}
	
	if(maxShift2>0.5*MARGIN*0.5*MARGIN){
		return 1;
	}
	else{
		return 0;
	}
}


static void calculateCellParticle()
{
	// store and sort with cells
	
	#pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0; iP<PowerParticleCount; ++iP){
		if(iP<ParticleCount){
			int iCX=(int)floor((Position[iP][0]-DomainMin[0])/CellWidth[0]);
			int iCY=(int)floor((Position[iP][1]-DomainMin[1])/CellWidth[1]);
			int iCZ=(int)floor((Position[iP][2]-DomainMin[2])/CellWidth[2]);
			if(iCX<0) iCX=0; else if(iCX>=CellCount[0]) iCX=CellCount[0]-1;
			if(iCY<0) iCY=0; else if(iCY>=CellCount[1]) iCY=CellCount[1]-1;
			#ifdef TWO_DIMENSIONAL
			iCZ=0;
			#else
			if(iCZ<0) iCZ=0; else if(iCZ>=CellCount[2]) iCZ=CellCount[2]-1;
			#endif
			CellIndex[iP]=(iCX*CellCount[1]+iCY)*CellCount[2]+iCZ;
			if(WALL_BEGIN<=Property[iP] && Property[iP]<WALL_END){
				CellIndex[iP] += TotalCellCount;
			}
			CellParticle[iP]=iP;
		}
		else{
			CellIndex[ iP ]    = 2*TotalCellCount;
			CellParticle[ iP ] = ParticleCount;
		}
	}
	
	// sort with CellIndex
	for(int iMain=0;iMain<ParticleCountPower;++iMain){
		for(int iSub=0;iSub<=iMain;++iSub){
			
			int dist = (1<< (iMain-iSub));
			
			#pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
			#pragma acc loop independent
			#pragma omp parallel for
			for(int iP=0;iP<(1<<ParticleCountPower);++iP){
				bool up = ((iP >> iMain) & 2) == 0;
				
				if(  (( iP & dist )==0) && ( CellIndex[ iP ] > CellIndex[ iP | dist ] == up) ){
					int tmpCellIndex    = CellIndex[ iP ];
					int tmpCellParticle = CellParticle[ iP ];
					CellIndex[ iP ]     = CellIndex[ iP | dist ];
					CellParticle[ iP ]  = CellParticle[ iP | dist ];
					CellIndex[ iP | dist ]    = tmpCellIndex;
					CellParticle[ iP | dist ] = tmpCellParticle;
				}
			}
		}
	}
	
	#pragma acc kernels present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		CellFluidParticleBegin[iC]=0;
		CellFluidParticleEnd[iC]=0;
		CellWallParticleBegin[iC]=0;
		CellWallParticleEnd[iC]=0;
	}
	
	#pragma acc kernels present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellIndex[0:PowerParticleCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=1; iP<ParticleCount+1; ++iP){
		if( CellIndex[iP-1]<CellIndex[iP] ){
			if( CellIndex[iP-1] < TotalCellCount ){
				CellFluidParticleEnd[ CellIndex[iP-1] ] = iP;
			}
			else if( CellIndex[iP-1] - TotalCellCount < TotalCellCount ){
				CellWallParticleEnd[ CellIndex[iP-1]-TotalCellCount ] = iP;
			}
			if( CellIndex[iP] < TotalCellCount ){
				CellFluidParticleBegin[ CellIndex[iP] ] = iP;
			}
			else if( CellIndex[iP] - TotalCellCount < TotalCellCount ){
				CellWallParticleBegin[ CellIndex[iP]-TotalCellCount ] = iP;
			}
		}
	}
	
	// Fill zeros in CellParticleBegin and CellParticleEnd
	int power = 0;
	const int N = 2*TotalCellCount;
	while( (N>>power) != 0 ){
		power++;
	}
	const int powerN = (1<<power);
	
	int * ptr = (int *)malloc( powerN * sizeof(int));
	#pragma acc enter data create(ptr[0:powerN])
	
	#pragma acc kernels present(ptr[0:powerN])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<powerN;++iRow){
		ptr[iRow]=0;
	}
	
	#pragma acc kernels present(ptr[0:powerN],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		ptr[iC]               =CellFluidParticleEnd[iC]-CellFluidParticleBegin[iC];
		ptr[iC+TotalCellCount]=CellWallParticleEnd[iC] -CellWallParticleBegin[iC];
	}
	
	// Convert ptr to cumulative sum
	for(int iMain=0;iMain<power;++iMain){
		const int dist = (1<<iMain);	
		#pragma acc kernels present(ptr[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			ptr[iRow]+=ptr[iRow+dist];
		}
	}
	for(int iMain=0;iMain<power;++iMain){
		const int dist = (powerN>>(iMain+1));	
		#pragma acc kernels present(ptr[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			ptr[iRow]-=ptr[iRow+dist];
			ptr[iRow+dist]+=ptr[iRow];
		}
	}
	
	#pragma acc kernels present(ptr[0:powerN],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		if(iC==0){	CellFluidParticleBegin[iC]=0;	}
		else     { 	CellFluidParticleBegin[iC]=ptr[iC-1];	}
		CellFluidParticleEnd[iC]  =ptr[iC];
		CellWallParticleBegin[iC] =ptr[iC-1+TotalCellCount];
		CellWallParticleEnd[iC]   =ptr[iC+TotalCellCount];
	}
	
	free(ptr);
	#pragma acc exit data delete(ptr[0:powerN])
	
	#pragma acc kernels present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	{
		FluidParticleBegin = CellFluidParticleBegin[0];
		FluidParticleEnd   = CellFluidParticleEnd[TotalCellCount-1];
		WallParticleBegin  = CellWallParticleBegin[0];
		WallParticleEnd    = CellWallParticleEnd[TotalCellCount-1];
	}
	#pragma acc update host(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	
	// re-arange particles in CellIndex order
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],ParticleIndex[0:ParticleCount],TmpIntScalar[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		TmpIntScalar[iP]=ParticleIndex[CellParticle[iP]];
	}
	#pragma acc kernels present(ParticleIndex[0:ParticleCount],TmpIntScalar[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		ParticleIndex[iP]=TmpIntScalar[iP];
	}
	
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],Property[0:ParticleCount],TmpIntScalar[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		TmpIntScalar[iP]=Property[CellParticle[iP]];
	}
	#pragma acc kernels present(Property[0:ParticleCount],TmpIntScalar[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Property[iP]=TmpIntScalar[iP];
	}
	
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],TmpDoubleVector[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			TmpDoubleVector[iP][iD]=Position[CellParticle[iP]][iD];
		}
	}
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],TmpDoubleVector[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Position[iP][iD]=TmpDoubleVector[iP][iD];
		}
	}
		
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],TmpDoubleVector[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			TmpDoubleVector[iP][iD]=Velocity[CellParticle[iP]][iD];
		}
	}
	#pragma acc kernels present(TmpDoubleVector[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Velocity[iP][iD]=TmpDoubleVector[iP][iD];
		}
	}

	// Carry the ISPH pressure and the free-surface flag along with the particle
	// permutation, so the warm start and the free-surface hysteresis remain
	// valid across a cell re-sort.
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],PressureP[0:ParticleCount],TmpDoubleScalar[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){ TmpDoubleScalar[iP]=PressureP[CellParticle[iP]]; }
	#pragma acc kernels present(PressureP[0:ParticleCount],TmpDoubleScalar[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){ PressureP[iP]=TmpDoubleScalar[iP]; }

	#pragma acc kernels present(CellParticle[0:PowerParticleCount],PpeFlag[0:ParticleCount],TmpIntScalar[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){ TmpIntScalar[iP]=PpeFlag[CellParticle[iP]]; }
	#pragma acc kernels present(PpeFlag[0:ParticleCount],TmpIntScalar[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){ PpeFlag[iP]=TmpIntScalar[iP]; }
}


static void calculateNeighbor( void )
{
    // Robust CLOSED-TANK neighbor search (no periodic wrapping).
    #pragma acc kernels present(NeighborCount[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        NeighborCount[iP]=0;
    }

    const double rSearch=MaxRadius+MARGIN;
    const double rSearch2=rSearch*rSearch;
    const int rangeX=(int)ceil(rSearch/CellWidth[0]);
    const int rangeY=(int)ceil(rSearch/CellWidth[1]);
    #ifdef TWO_DIMENSIONAL
    const int rangeZ=0;
    #else
    const int rangeZ=(int)ceil(rSearch/CellWidth[2]);
    #endif

    // ---------------------------- count pass -------------------------------
    #pragma acc kernels present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount],CellIndex[0:PowerParticleCount],NeighborCount[0:ParticleCount],Position[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        int baseCell=CellIndex[iP];
        if(baseCell>=TotalCellCount) baseCell-=TotalCellCount;
        if(baseCell<0) baseCell=0;
        if(baseCell>=TotalCellCount) baseCell=TotalCellCount-1;

        const int iCX=baseCell/(CellCount[1]*CellCount[2]);
        const int rem=baseCell-iCX*CellCount[1]*CellCount[2];
        const int iCY=rem/CellCount[2];
        const int iCZ=rem-iCY*CellCount[2];

        const int x0=(iCX-rangeX>0 ? iCX-rangeX : 0);
        const int x1=(iCX+rangeX<CellCount[0]-1 ? iCX+rangeX : CellCount[0]-1);
        const int y0=(iCY-rangeY>0 ? iCY-rangeY : 0);
        const int y1=(iCY+rangeY<CellCount[1]-1 ? iCY+rangeY : CellCount[1]-1);
        #ifdef TWO_DIMENSIONAL
        const int z0=0, z1=0;
        #else
        const int z0=(iCZ-rangeZ>0 ? iCZ-rangeZ : 0);
        const int z1=(iCZ+rangeZ<CellCount[2]-1 ? iCZ+rangeZ : CellCount[2]-1);
        #endif

        int count=0;
        #pragma acc loop seq
        for(int jCX=x0;jCX<=x1;++jCX){
            #pragma acc loop seq
            for(int jCY=y0;jCY<=y1;++jCY){
                #pragma acc loop seq
                for(int jCZ=z0;jCZ<=z1;++jCZ){
                    const int jC=(jCX*CellCount[1]+jCY)*CellCount[2]+jCZ;

                    #pragma acc loop seq
                    for(int jP=CellFluidParticleBegin[jC];jP<CellFluidParticleEnd[jC];++jP){
                        if(iP==jP) continue;
                        const double dx=Position[jP][0]-Position[iP][0];
                        const double dy=Position[jP][1]-Position[iP][1];
                        const double dz=Position[jP][2]-Position[iP][2];
                        const double r2=dx*dx+dy*dy+dz*dz;
                        if(r2<=rSearch2) ++count;
                    }

                    #pragma acc loop seq
                    for(int jP=CellWallParticleBegin[jC];jP<CellWallParticleEnd[jC];++jP){
                        if(iP==jP) continue;
                        const double dx=Position[jP][0]-Position[iP][0];
                        const double dy=Position[jP][1]-Position[iP][1];
                        const double dz=Position[jP][2]-Position[iP][2];
                        const double r2=dx*dx+dy*dy+dz*dz;
                        if(r2<=rSearch2) ++count;
                    }
                }
            }
        }
        NeighborCount[iP]=count;
    }

    // ---------------------------- CSR prefix -------------------------------
    #pragma acc kernels present(NeighborPtr[0:PowerParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<PowerParticleCount;++iP) NeighborPtr[iP]=0;

    #pragma acc kernels present(NeighborCount[0:ParticleCount],NeighborPtr[0:PowerParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP) NeighborPtr[iP+1]=NeighborCount[iP];

    for(int iMain=0;iMain<ParticleCountPower;++iMain){
        const int dist=(1<<iMain);
        #pragma acc kernels present(NeighborPtr[0:PowerParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<PowerParticleCount;iP+=(dist<<1)){
            NeighborPtr[iP]+=NeighborPtr[iP+dist];
        }
    }
    for(int iMain=0;iMain<ParticleCountPower;++iMain){
        const int dist=(PowerParticleCount>>(iMain+1));
        #pragma acc kernels present(NeighborPtr[0:PowerParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<PowerParticleCount;iP+=(dist<<1)){
            NeighborPtr[iP]-=NeighborPtr[iP+dist];
            NeighborPtr[iP+dist]+=NeighborPtr[iP];
        }
    }

    #pragma acc kernels present(NeighborPtr[0:PowerParticleCount])
    { NeighborIndCount=NeighborPtr[ParticleCount]; }
    #pragma acc update host(NeighborIndCount)

    NeighborInd=(int *)malloc(NeighborIndCount*sizeof(int));
    #pragma acc enter data create(NeighborInd[0:NeighborIndCount]) attach(NeighborInd)

    // ----------------------------- fill pass -------------------------------
    #pragma acc kernels present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount],CellIndex[0:PowerParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        int baseCell=CellIndex[iP];
        if(baseCell>=TotalCellCount) baseCell-=TotalCellCount;
        if(baseCell<0) baseCell=0;
        if(baseCell>=TotalCellCount) baseCell=TotalCellCount-1;

        const int iCX=baseCell/(CellCount[1]*CellCount[2]);
        const int rem=baseCell-iCX*CellCount[1]*CellCount[2];
        const int iCY=rem/CellCount[2];
        const int iCZ=rem-iCY*CellCount[2];

        const int x0=(iCX-rangeX>0 ? iCX-rangeX : 0);
        const int x1=(iCX+rangeX<CellCount[0]-1 ? iCX+rangeX : CellCount[0]-1);
        const int y0=(iCY-rangeY>0 ? iCY-rangeY : 0);
        const int y1=(iCY+rangeY<CellCount[1]-1 ? iCY+rangeY : CellCount[1]-1);
        #ifdef TWO_DIMENSIONAL
        const int z0=0, z1=0;
        #else
        const int z0=(iCZ-rangeZ>0 ? iCZ-rangeZ : 0);
        const int z1=(iCZ+rangeZ<CellCount[2]-1 ? iCZ+rangeZ : CellCount[2]-1);
        #endif

        int iN=0;
        #pragma acc loop seq
        for(int jCX=x0;jCX<=x1;++jCX){
            #pragma acc loop seq
            for(int jCY=y0;jCY<=y1;++jCY){
                #pragma acc loop seq
                for(int jCZ=z0;jCZ<=z1;++jCZ){
                    const int jC=(jCX*CellCount[1]+jCY)*CellCount[2]+jCZ;

                    #pragma acc loop seq
                    for(int jP=CellFluidParticleBegin[jC];jP<CellFluidParticleEnd[jC];++jP){
                        if(iP==jP) continue;
                        const double dx=Position[jP][0]-Position[iP][0];
                        const double dy=Position[jP][1]-Position[iP][1];
                        const double dz=Position[jP][2]-Position[iP][2];
                        const double r2=dx*dx+dy*dy+dz*dz;
                        if(r2<=rSearch2){
                            if(iN<NeighborCount[iP]) NeighborInd[NeighborPtr[iP]+iN]=jP;
                            ++iN;
                        }
                    }

                    #pragma acc loop seq
                    for(int jP=CellWallParticleBegin[jC];jP<CellWallParticleEnd[jC];++jP){
                        if(iP==jP) continue;
                        const double dx=Position[jP][0]-Position[iP][0];
                        const double dy=Position[jP][1]-Position[iP][1];
                        const double dz=Position[jP][2]-Position[iP][2];
                        const double r2=dx*dx+dy*dy+dz*dz;
                        if(r2<=rSearch2){
                            if(iN<NeighborCount[iP]) NeighborInd[NeighborPtr[iP]+iN]=jP;
                            ++iN;
                        }
                    }
                }
            }
        }

        #pragma acc loop seq
        for(int k=iN;k<NeighborCount[iP];++k){
            NeighborInd[NeighborPtr[iP]+k]=iP;
        }
    }

    #pragma acc kernels present(NeighborCalculatedPosition[0:ParticleCount][0:DIM],Position[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        #pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            NeighborCalculatedPosition[iP][iD]=Position[iP][iD];
        }
    }
}


static void freeNeighbor()
{
	#pragma acc exit data delete(NeighborInd[0:NeighborIndCount])
	free(NeighborInd);
	NeighborInd=NULL;
	NeighborIndCount=0;
}


static void calculateConvection()
{
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Position[iP][0] += Velocity[iP][0]*Dt;
        Position[iP][1] += Velocity[iP][1]*Dt;
        Position[iP][2] += Velocity[iP][2]*Dt;
    }
}


static void evaluateExplicitAcceleration(double (*acc)[DIM])
{
	resetForce();
	calculateDensityA();
	calculateGravityCenter();
    // Surface tension models are intentionally separated:
    //   single phase -> validated Kondo free-surface potential
    //   two phase    -> selectable CSF / legacy CSS / pairwise potential
if(TwoPhaseActive) calculateTwoPhaseSurfaceTension();
    else               calculateKondoPotentialSurfaceTension();
	calculateViscosityV();
	calculateGravity();

	// Explicit-stage acceleration, with a safety cap for the light phase.
	// a = F/m and the gas mass is 1000x smaller, so a single bad interface
	// pair can otherwise launch one gas particle across the domain in one
	// step; that particle then corrupts the neighbour list and the PPE.
	const double accCap=(TwoPhaseActive
	                     ? TwoPhaseDuCap*ParticleSpacing/((Dt+1.0e-300)*(Dt+1.0e-300))
	                     : 1.0e300);
	int clipped=0;
	#pragma acc parallel loop reduction(+:clipped) present(acc[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM],Mass[0:ParticleCount])
	#pragma omp parallel for reduction(+:clipped)
	for(int iP=0;iP<ParticleCount;++iP){
		if(FluidParticleBegin<=iP && iP<FluidParticleEnd){
			double a0=Force[iP][0]/Mass[iP];
			double a1=Force[iP][1]/Mass[iP];
			double a2=Force[iP][2]/Mass[iP];
			const double am=sqrt(a0*a0+a1*a1+a2*a2);
            if(!isfinite(am)) ++clipped;
			if(am>accCap && am>1.0e-30){
				++clipped;
                const double sc=accCap/am;
				a0*=sc; a1*=sc; a2*=sc;
			}
			acc[iP][0]=a0;
			acc[iP][1]=a1;
			acc[iP][2]=a2;
		}else{
			acc[iP][0]=0.0;
			acc[iP][1]=0.0;
			acc[iP][2]=0.0;
		}
	}
    if(TwoPhaseActive && clipped) TwoPhaseStepFailed=1;
}


// Return before any cell indexing if a stage produced an invalid state.
static int invalidTwoPhaseState()
{
    int bad=0;
    const double limit=TwoPhaseSpeedLimit;
    #pragma acc parallel loop reduction(+:bad) present(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
    #pragma omp parallel for reduction(+:bad)
    for(int i=0;i<ParticleCount;++i){
        for(int d=0;d<DIM;++d)
            if(!isfinite(Position[i][d]) || !isfinite(Velocity[i][d])) ++bad;
        const double u2=Velocity[i][0]*Velocity[i][0]+Velocity[i][1]*Velocity[i][1]+Velocity[i][2]*Velocity[i][2];
        if(limit>0.0 && u2>limit*limit) ++bad;
    }
    if(bad) log_printf("REJECT state: invalid or speed above %e, count=%d\n",limit,bad);
    return bad;
}

static void selectTwoPhaseDt()
{
    calculatePhysicalCoefficients();
    resetForce();
    calculateTwoPhaseSurfaceTension();
    calculateViscosityV();
    calculateGravity();
    double umax=0.0, amax=0.0, rateMax=0.0;
    int bad=0;
    // Bound the actual explicit viscosity row, including high-contrast pairs
    // and walls, rather than relying only on bulk nu_max.
    #pragma acc parallel loop reduction(max:umax,amax,rateMax) reduction(+:bad) present(Velocity[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM],Mass[0:ParticleCount],Mu[0:ParticleCount],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for reduction(max:umax,amax,rateMax) reduction(+:bad)
    for(int i=0;i<ParticleCount;++i){
        const double u=sqrt(Velocity[i][0]*Velocity[i][0]+Velocity[i][1]*Velocity[i][1]+Velocity[i][2]*Velocity[i][2]);
        if(u>umax) umax=u;
        if(Property[i]<FLUID_BEGIN || Property[i]>=FLUID_END) continue;
        const double a=sqrt(Force[i][0]*Force[i][0]+Force[i][1]*Force[i][1]+Force[i][2]*Force[i][2])/Mass[i];
        if(!isfinite(a) || !isfinite(u)) ++bad;
        if(a>amax) amax=a;
        double rate=0.0;
        #pragma acc loop seq
        for(int jn=0;jn<NeighborCount[i];++jn){
            const int j=NeighborInd[NeighborPtr[i]+jn];
            if(i==j) continue;
            const int wall=(Property[j]>=WALL_BEGIN && Property[j]<WALL_END);
            if(wall && !WallNoSlip) continue;
            const double dx=Position[j][0]-Position[i][0],dy=Position[j][1]-Position[i][1];
            const double r2=dx*dx+dy*dy;
            if(r2<=1.0e-30 || r2>=RadiusV*RadiusV) continue;
            const double r=sqrt(r2), re=fmax(r,IsphMinPairRatio*ParticleSpacing);
            const double mu=(wall ? Mu[i] : 2.0*Mu[i]*Mu[j]/(Mu[i]+Mu[j]+1.0e-300));
            rate+=8.0*mu*(-dwvdr(r,RadiusV))/re*ParticleVolume/Mass[i];
        }
        if(rate>rateMax) rateMax=rate;
    }
    if(bad){ log_printf("Nonfinite acceleration at accepted state\n"); exit(EXIT_FAILURE); }
    double next=fmin(TwoPhaseDtMax,1.2*Dt);
    if(TwoPhaseSourceGamma>0.0)
        next=fmin(next,0.1*TwoPhaseDensityRelaxTime/sqrt(TwoPhaseSourceGamma));
    if(umax>0.0) next=fmin(next,0.10*ParticleSpacing/umax);
    if(amax>0.0) next=fmin(next,sqrt(0.05*ParticleSpacing/amax));
    if(rateMax>0.0) next=fmin(next,0.45/rateMax);
    if(InterfaceTension01>0.0){
        const double rho=fmin(Density[0],Density[1]);
        next=fmin(next,0.25*sqrt(rho*pow(ParticleSpacing,3)/(2.0*M_PI*InterfaceTension01)));
    }
    if(next<TwoPhaseDtMinRatio*TwoPhaseDtMax){
        writeProfFile("isph_last_accepted.prof");
        log_printf("Stable Dt=%e below configured minimum=%e; stopped at Time=%e\n",
                   next,TwoPhaseDtMinRatio*TwoPhaseDtMax,Time);
        exit(EXIT_FAILURE);
    }
    // Absorb a roundoff-size final remainder in the last full step. Otherwise
    // 2000 additions of 1e-5 can leave ~1e-15 s and produce enormous p~1/Dt.
    // The relative extension is at most 1e-8 of an already stable step.
    const double remaining=EndTime-Time;
    Dt=(remaining<=next*(1.0+1.0e-8) ? remaining : next);
    if(!(Dt>0.0) || !isfinite(Dt) || Time+Dt==Time){
        log_printf("No representable stable time step\n"); exit(EXIT_FAILURE);
    }
    #pragma acc update device(Dt)
    log_printf("TwoPhase adaptive: time=%e Dt=%e Umax=%e Amax=%e viscRate=%e\n",Time,Dt,umax,amax,rateMax);
}

static void advanceTwoPhase()
{
    if(invalidTwoPhaseState()){ log_printf("Invalid initial/accepted state\n"); exit(EXIT_FAILURE); }
    // A fresh list leaves the full skin available to both predictor stages.
    freeNeighbor(); calculateCellParticle(); calculateNeighbor();
    selectTwoPhaseDt();
    #pragma acc update self(WallCenter[0:WALL_END][0:DIM])
    double wall0[WALL_END][DIM];
    memcpy(wall0,WallCenter,sizeof(wall0));
    // Backups are indexed by immutable particle ID, not the sortable row.
    // RK3/4 position buffers are unused by the RK2 integrator.
    #pragma acc parallel loop present(ParticleIndex[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],PressureP[0:ParticleCount],RkK3X[0:ParticleCount][0:DIM],RkK3U[0:ParticleCount][0:DIM],RkK4X[0:ParticleCount][0:DIM])
    #pragma omp parallel for
    for(int i=0;i<ParticleCount;++i){
        const int id=ParticleIndex[i];
        for(int d=0;d<DIM;++d){ RkK3X[id][d]=Position[i][d]; RkK3U[id][d]=Velocity[i][d]; }
        RkK4X[id][0]=PressureP[i];
    }
    for(int attempt=0;;++attempt){
        TwoPhaseStepFailed=0;
        initializeWall(); // rotation matrix depends on the current adaptive Dt
        calculatePhysicalCoefficients();
        calculatePredictor();
        if(!TwoPhaseStepFailed){
            calculateWall();
            calculatePeriodicBoundary();
            TwoPhaseStepFailed=invalidTwoPhaseState();
        }
        if(!TwoPhaseStepFailed){
            freeNeighbor(); calculateCellParticle(); calculateNeighbor();
            // Mass and Mu are derived arrays, not carried by the permutation.
            calculatePhysicalCoefficients();
            calculateDensityA(); calculateGravityCenter(); calculateDensityP();
            detectFreeSurface();
            applyTwoPhaseShifting();
            TwoPhaseStepFailed=invalidTwoPhaseState();
        }
        if(!TwoPhaseStepFailed){
            // Shift precedes projection; rebuild geometry after changing x.
            freeNeighbor(); calculateCellParticle(); calculateNeighbor();
            calculatePhysicalCoefficients();
            calculateDensityA(); calculateGravityCenter(); calculateDensityP();
            calculatePpe();
            if(invalidTwoPhaseState()) TwoPhaseStepFailed=1;
        }
        if(!TwoPhaseStepFailed){
            freeNeighbor(); calculateCellParticle(); calculateNeighbor();
            calculateDensityP();
            calculateEffectiveDensity(); calculateInterfaceGeometry();
            return;
        }
        #pragma acc parallel loop present(ParticleIndex[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],PressureP[0:ParticleCount],RkK3X[0:ParticleCount][0:DIM],RkK3U[0:ParticleCount][0:DIM],RkK4X[0:ParticleCount][0:DIM])
        #pragma omp parallel for
        for(int i=0;i<ParticleCount;++i){
            const int id=ParticleIndex[i];
            for(int d=0;d<DIM;++d){ Position[i][d]=RkK3X[id][d]; Velocity[i][d]=RkK3U[id][d]; }
            PressureP[i]=RkK4X[id][0];
        }
        memcpy(WallCenter,wall0,sizeof(wall0));
        #pragma acc update device(WallCenter[0:WALL_END][0:DIM])
        if(attempt>=10 || Dt<TwoPhaseDtMinRatio*TwoPhaseDtMax){
            char checkpoint[]="isph_last_accepted.prof";
            writeProfFile(checkpoint);
            log_printf("Stopped after failed retries at Time=%e; restored accepted state in %s\n",Time,checkpoint);
            exit(EXIT_FAILURE);
        }
        Dt*=0.5;
        #pragma acc update device(Dt)
        log_printf("Retry two-phase step %d at Time=%e Dt=%e\n",attempt+1,Time,Dt);
        freeNeighbor(); calculateCellParticle(); calculateNeighbor();
    }
}

static void calculatePredictor()
{
	// RK2 midpoint for the explicit (non-pressure) part.
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],RkX0[0:ParticleCount][0:DIM],RkU0[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			RkX0[iP][iD]=Position[iP][iD];
			RkU0[iP][iD]=Velocity[iP][iD];
		}
	}

	#pragma acc kernels present(Velocity[0:ParticleCount][0:DIM],RkK1X[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD) RkK1X[iP][iD]=Velocity[iP][iD];
	}
	evaluateExplicitAcceleration(RkK1U);
    if(TwoPhaseActive && TwoPhaseStepFailed) return;

	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],RkX0[0:ParticleCount][0:DIM],RkU0[0:ParticleCount][0:DIM],RkK1X[0:ParticleCount][0:DIM],RkK1U[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Position[iP][iD]=RkX0[iP][iD]+0.5*Dt*RkK1X[iP][iD];
			Velocity[iP][iD]=RkU0[iP][iD]+0.5*Dt*RkK1U[iP][iD];
		}
	}
	calculatePeriodicBoundary();

	#pragma acc kernels present(Velocity[0:ParticleCount][0:DIM],RkK2X[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD) RkK2X[iP][iD]=Velocity[iP][iD];
	}
	evaluateExplicitAcceleration(RkK2U);
    if(TwoPhaseActive && TwoPhaseStepFailed) return;

	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],RkX0[0:ParticleCount][0:DIM],RkU0[0:ParticleCount][0:DIM],RkK2X[0:ParticleCount][0:DIM],RkK2U[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Position[iP][iD]=RkX0[iP][iD]+Dt*RkK2X[iP][iD];
			Velocity[iP][iD]=RkU0[iP][iD]+Dt*RkK2U[iP][iD];
		}
	}
	calculatePeriodicBoundary();
}


static void resetForce()
{
	#pragma acc kernels present(Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	#pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Force[iP][iD]=0.0;
        }
    }
}


static void calculatePhysicalCoefficients()
{
	#pragma acc kernels present(Mass[0:ParticleCount],Property[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Mass[iP]=Density[Property[iP]]*ParticleVolume;
	}

	#pragma acc kernels present(Mu[0:ParticleCount],Property[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Mu[iP]=ShearViscosity[Property[iP]];
	}
}


static void calculateDensityA()
{ 
	#pragma acc kernels present(DensityA[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum = 0.0;
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double ratio = InteractionRatio[Property[iP]][Property[jP]];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Position[jP][iD] - Position[iP][iD];
			}
			const double radius = RadiusA;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double weight = ratio * wa(rij,radius);
				sum += weight;
			}
		}
		DensityA[iP]=sum;
	}
}


static void calculateGravityCenter()
{
	#pragma acc kernels present(GravityCenter[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double ratio = InteractionRatio[Property[iP]][Property[jP]];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Position[jP][iD] - Position[iP][iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(RadiusG*RadiusG - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double weight = ratio * wg(rij,RadiusG);
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					sum[iD] += xij[iD]*weight/R2g*RadiusG;
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			GravityCenter[iP][iD] = sum[iD];
		}
	}
}

static void calculatePressureA()
{

	#pragma acc kernels present(DensityA[0:ParticleCount],Position[0:ParticleCount][0:DIM],PressureA[0:ParticleCount],Property[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		PressureA[iP] = CofA[Property[iP]]*(DensityA[iP]-N0a)/ParticleSpacing;
		if(N0a<=DensityA[iP]){
			PressureA[iP] = 0.0;
		}
	}
	
	#pragma acc kernels present(Force[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],PressureA[0:ParticleCount],Property[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	double force[DIM]={0.0,0.0,0.0};
    	#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
        	double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Position[jP][iD] - Position[iP][iD];
            }
            const double radius = RadiusA;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 > 0 && rij2 > 1.0e-30){
                const double rij = sqrt(rij2);
                const double dwij = ratio_ij * dwadr(rij,radius);
            	const double dwji = ratio_ji * dwadr(rij,radius);
                const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
            	#pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    force[iD] += (PressureA[iP]*dwij+PressureA[jP]*dwji)*eij[iD]* ParticleVolume;
                }
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}


static void calculateEffectiveDensity()
{
    // -------------------------------------------------------------------------
    // Density used by the two-phase projection (PPE coefficient and pressure
    // gradient).
    //
    //   rho_eff = (1-theta) rho(type) + theta * <rho>_Shepard ,
    //   theta = TwoPhaseRhoSmooth (default zero: sharp physical density).
    //
    // theta = 0 keeps the physical sharp jump and is the default.  A small
    // amount of smoothing (0.1 - 0.3) widens the coefficient jump of the
    // variable-density PPE over roughly one particle spacing; that is the
    // classical remedy if a very large density ratio still produces interface
    // noise, at the cost of a slightly smeared buoyancy.
    // -------------------------------------------------------------------------
    #pragma acc kernels present(RhoEff[0:ParticleCount],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Density[0:TYPE_COUNT])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const double rho0=Density[Property[iP]];
        RhoEff[iP]=rho0;
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;
        if(!(TwoPhaseRhoSmooth>0.0)) continue;

        double sw=wp(0.0,RadiusP);
        double sr=sw*rho0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double w=wp(sqrt(r2),RadiusP);
            sw+=w;
            sr+=w*Density[Property[jP]];
        }
        const double rsm=(sw>1.0e-30 ? sr/sw : rho0);
        double th=TwoPhaseRhoSmooth;
        if(th>1.0) th=1.0;
        RhoEff[iP]=(1.0-th)*rho0+th*rsm;
    }
}


static void calculateInterfaceGeometry()
{
    // -------------------------------------------------------------------------
    // Colour function, first-order-consistent colour gradient, interface unit
    // normal, and CURVATURE.
    //
    //   PASS A : raw colour  (C = 0 for Type0, C = 1 for Type1)
    //   PASS B : Shepard smoothing (applied twice by default).  Wall neighbours
    //            contribute the MIRRORED value C_i, which imposes a zero normal
    //            colour gradient at the wall, i.e. a neutral 90 deg contact
    //            angle, and keeps the filter from being truncated there.
    //   PASS C : renormalised gradient
    //               grad C_i = M_i^{-1} sum_j V_j (C_j - C_i) grad_i W_ij
    //   PASS D : curvature by a REPRODUCING divergence of the unit normal,
    //               kappa_i = - sum_j V_j (n_j - n_i).(M_i^{-1} grad_i W_ij),
    //            evaluated only over neighbours that carry a valid normal, with
    //            M_i built on the same restricted set so the operator stays
    //            first-order exact there.  This is the step the previous
    //            curvature-free CSS was missing, and it is what makes the
    //            Laplace pressure correct instead of merely non-zero.
    // -------------------------------------------------------------------------
    if(!TwoPhaseActive) return;

    const double hC=(RadiusST>1.0e-30 ? RadiusST : RadiusP);

    // ------------------------------- PASS A --------------------------------
    #pragma acc kernels present(ColorS[0:ParticleCount],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        ColorS[iP]=((FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END) && Property[iP]==1) ? 1.0 : 0.0;
    }

    // ------------------------------- PASS B --------------------------------
    const int nSmooth=(TwoPhaseStDoubleSmooth ? 2 : 1);
    for(int iSm=0;iSm<nSmooth;++iSm){
        #pragma acc kernels present(ColorC[0:ParticleCount],ColorS[0:ParticleCount],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP){
            ColorC[iP]=ColorS[iP];
            if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

            const double ci=ColorS[iP];
            double sw=wp(0.0,hC);
            double sc=sw*ci;

            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const int jP=NeighborInd[NeighborPtr[iP]+jN];
                if(iP==jP) continue;
                const int isF=(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END);
                const int isW=(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END);
                if(!isF && !isW) continue;
                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double dz=Position[jP][2]-Position[iP][2];
                const double r2=dx*dx+dy*dy+dz*dz;
                if(r2<=1.0e-30 || r2>=hC*hC) continue;
                const double w=wp(sqrt(r2),hC);
                sw+=w;
                sc+=w*(isF ? ColorS[jP] : ci);   // wall: mirrored colour
            }
            ColorC[iP]=(sw>1.0e-30 ? sc/sw : ci);
        }

        #pragma acc kernels present(ColorC[0:ParticleCount],ColorS[0:ParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP) ColorS[iP]=ColorC[iP];
    }

    // ------------------------------- PASS C --------------------------------
    #pragma acc kernels present(ColorC[0:ParticleCount],IfGrad[0:ParticleCount][0:DIM],IfDelta[0:ParticleCount],IfNormal[0:ParticleCount][0:DIM],IfKappa[0:ParticleCount],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        IfGrad[iP][0]=IfGrad[iP][1]=IfGrad[iP][2]=0.0;
        IfNormal[iP][0]=IfNormal[iP][1]=IfNormal[iP][2]=0.0;
        IfDelta[iP]=0.0;
        IfKappa[iP]=0.0;
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

        const double ci=ColorC[iP];
        double gx=0.0, gy=0.0, gz=0.0;
        double Mxx=0.0, Mxy=0.0, Myy=0.0;

        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            const int isF=(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END);
            const int isW=(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END);
            if(!isF && !isW) continue;

            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=hC*hC) continue;

            const double r=sqrt(r2);
            const double rMin=IsphMinPairRatio*ParticleSpacing;
            const double rEff=(r>rMin ? r : rMin);
            const double g=(-dwpdr(r,hC))/rEff;

            Mxx+=ParticleVolume*g*dx*dx;
            Mxy+=ParticleVolume*g*dx*dy;
            Myy+=ParticleVolume*g*dy*dy;

            if(!isF) continue;                       // wall: dc = 0 (mirrored)
            const double dc=ColorC[jP]-ci;
            gx+=ParticleVolume*dc*g*dx;
            gy+=ParticleVolume*dc*g*dy;
            gz+=ParticleVolume*dc*g*dz;
        }

        double rx=gx/MrefST, ry=gy/MrefST, rz=gz/MrefST;
#ifdef TWO_DIMENSIONAL
        {
            const double det=Mxx*Myy-Mxy*Mxy;
            const double sc=Mxx+Myy;
            if(det>1.0e-10*(sc*sc+1.0e-30)){
                rx=( Myy*gx-Mxy*gy)/det;
                ry=(-Mxy*gx+Mxx*gy)/det;
                rz=0.0;
            }
        }
#endif
        double gm=sqrt(rx*rx+ry*ry+rz*rz);
        const double gmMax=1.0/(ParticleSpacing+1.0e-30);
        if(gm>gmMax){ const double s=gmMax/gm; rx*=s; ry*=s; rz*=s; gm=gmMax; }

        IfGrad[iP][0]=rx; IfGrad[iP][1]=ry; IfGrad[iP][2]=rz;
        IfDelta[iP]=gm;
        if(gm*ParticleSpacing>TwoPhaseStDeltaMin){
            IfNormal[iP][0]=rx/gm;
            IfNormal[iP][1]=ry/gm;
            IfNormal[iP][2]=rz/gm;
        }
    }

    // ------------------------------- PASS D --------------------------------
    #pragma acc kernels present(IfNormal[0:ParticleCount][0:DIM],IfKappa[0:ParticleCount],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        IfKappa[iP]=0.0;
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

        const double nix=IfNormal[iP][0], niy=IfNormal[iP][1], niz=IfNormal[iP][2];
        if(nix*nix+niy*niy+niz*niz<0.5) continue;    // no valid normal here

        double Mxx=0.0, Mxy=0.0, Myy=0.0;
        double Txx=0.0, Txy=0.0, Tyx=0.0, Tyy=0.0;   // T_ab = sum_j dn_a G_b
        int nValid=0;

        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;

            const double njx=IfNormal[jP][0], njy=IfNormal[jP][1], njz=IfNormal[jP][2];
            if(njx*njx+njy*njy+njz*njz<0.5) continue;  // bulk neighbour: skip

            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=hC*hC) continue;

            const double r=sqrt(r2);
            const double rMin=IsphMinPairRatio*ParticleSpacing;
            const double rEff=(r>rMin ? r : rMin);
            const double g=(-dwpdr(r,hC))/rEff;

            const double Gx=ParticleVolume*g*dx;
            const double Gy=ParticleVolume*g*dy;

            Mxx+=Gx*dx; Mxy+=Gx*dy; Myy+=Gy*dy;

            const double dnx=njx-nix;
            const double dny=njy-niy;
            Txx+=dnx*Gx; Txy+=dnx*Gy;
            Tyx+=dny*Gx; Tyy+=dny*Gy;
            ++nValid;
        }

        if(nValid<3) continue;

        double divN=(Txx+Tyy)/MrefST;
#ifdef TWO_DIMENSIONAL
        {
            const double det=Mxx*Myy-Mxy*Mxy;
            const double sc=Mxx+Myy;
            if(det>1.0e-10*(sc*sc+1.0e-30)){
                divN=(Myy*Txx-Mxy*Txy-Mxy*Tyx+Mxx*Tyy)/det;
            }
        }
#endif
        double kap=-divN;
        const double kMax=TwoPhaseCurvMax/(ParticleSpacing+1.0e-30);
        if(!(kap==kap)) kap=0.0;
        if(kap> kMax) kap= kMax;
        if(kap<-kMax) kap=-kMax;
        IfKappa[iP]=kap;
    }
}

static void calculateTwoPhaseCSSSurfaceTension()
{
    // -------------------------------------------------------------------------
    // Two-phase Continuum Surface Stress (CSS), curvature-free form.
    //
    //     T_i     = sigma_01 * delta_i * (I - n_i n_i)
    //     f_sigma = div(T)
    //     F_i    += V_i V_j (T_i + T_j) . grad_i(W_ij)
    //
    // Because grad_i W_ij = -grad_j W_ji, the pair contribution is exactly
    // antisymmetric, so the model injects no net linear momentum.  No curvature
    // is reconstructed, and the force is non-zero only where the colour
    // gradient is non-zero, i.e. strictly at the interface.
    //
    // delta_i and n_i now come from the renormalised colour gradient computed
    // in calculateInterfaceGeometry(), which is the fix for the wrong Laplace
    // pressure of the previous version.
    // -------------------------------------------------------------------------
    if(!TwoPhaseActive || !(InterfaceTension01>0.0)) return;

    calculateInterfaceGeometry();

    #pragma acc kernels present(Force[0:ParticleCount][0:DIM],IfNormal[0:ParticleCount][0:DIM],IfDelta[0:ParticleCount],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],InterfaceTension01)
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

        const double di=IfDelta[iP];
        const double nix=IfNormal[iP][0];
        const double niy=IfNormal[iP][1];
        const double niz=IfNormal[iP][2];

        const double Tixx=di*(1.0-nix*nix);
        const double Tixy=   -di*(nix*niy);
        const double Tixz=   -di*(nix*niz);
        const double Tiyy=di*(1.0-niy*niy);
        const double Tiyz=   -di*(niy*niz);
        const double Tizz=di*(1.0-niz*niz);

        double fx=0.0,fy=0.0,fz=0.0;

        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;

            const double dj=IfDelta[jP];
            if(di<=1.0e-30 && dj<=1.0e-30) continue; // both bulk

            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;

            const double r=sqrt(r2);
            const double rMin=IsphMinPairRatio*ParticleSpacing;
            const double rEff=(r>rMin ? r : rMin);
            const double gradFac=(-dwpdr(r,RadiusP))/rEff;

            const double gwx=gradFac*dx;
            const double gwy=gradFac*dy;
            const double gwz=gradFac*dz;

            const double njx=IfNormal[jP][0];
            const double njy=IfNormal[jP][1];
            const double njz=IfNormal[jP][2];

            const double Tjxx=dj*(1.0-njx*njx);
            const double Tjxy=   -dj*(njx*njy);
            const double Tjxz=   -dj*(njx*njz);
            const double Tjyy=dj*(1.0-njy*njy);
            const double Tjyz=   -dj*(njy*njz);
            const double Tjzz=dj*(1.0-njz*njz);

            const double qx=(Tixx+Tjxx)*gwx + (Tixy+Tjxy)*gwy + (Tixz+Tjxz)*gwz;
            const double qy=(Tixy+Tjxy)*gwx + (Tiyy+Tjyy)*gwy + (Tiyz+Tjyz)*gwz;
            const double qz=(Tixz+Tjxz)*gwx + (Tiyz+Tjyz)*gwy + (Tizz+Tjzz)*gwz;

            const double pairScale=InterfaceTension01*ParticleVolume*ParticleVolume;
            fx+=pairScale*qx;
            fy+=pairScale*qy;
            fz+=pairScale*qz;
        }

        Force[iP][0]+=fx;
        Force[iP][1]+=fy;
        Force[iP][2]+=fz;
    }
}


static void calculateCsfSurfaceTension()
{
    if(!TwoPhaseActive || !(InterfaceTension01>0.0)) return;
    calculateEffectiveDensity();
    calculateInterfaceGeometry();
    // Same pair stencil as pressure acceleration. For constant curvature k,
    // p_i=sigma*k*C_i cancels this force exactly on this fixed geometry.
    // This does not make the approximate graph projection a balanced D*G solve.
    const double gc=2.0/N0p; // this implementation is 2D
    #pragma acc parallel loop present(Force[0:ParticleCount][0:DIM],Mass[0:ParticleCount],RhoEff[0:ParticleCount],ColorC[0:ParticleCount],IfKappa[0:ParticleCount],IfNormal[0:ParticleCount][0:DIM],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for
    for(int i=0;i<ParticleCount;++i){
        if(Property[i]<FLUID_BEGIN || Property[i]>=FLUID_END) continue;
        double a[DIM]={0.0,0.0,0.0};
        #pragma acc loop seq
        for(int jn=0;jn<NeighborCount[i];++jn){
            const int j=NeighborInd[NeighborPtr[i]+jn];
            if(j==i || Property[j]<FLUID_BEGIN || Property[j]>=FLUID_END) continue;
            const double dx=Position[j][0]-Position[i][0], dy=Position[j][1]-Position[i][1];
            const double r2=dx*dx+dy*dy;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double ni=IfNormal[i][0]*IfNormal[i][0]+IfNormal[i][1]*IfNormal[i][1];
            const double nj=IfNormal[j][0]*IfNormal[j][0]+IfNormal[j][1]*IfNormal[j][1];
            const double wi=(ni>0.5 ? 1.0 : 0.0), wj=(nj>0.5 ? 1.0 : 0.0);
            if(wi+wj==0.0) continue;
            const double k=(wi*IfKappa[i]+wj*IfKappa[j])/(wi+wj);
            const double rmin=IsphMinPairRatio*ParticleSpacing;
            const double q=gc*2.0/(RhoEff[i]+RhoEff[j])*InterfaceTension01*k*
                (ColorC[j]-ColorC[i])*wp(sqrt(r2),RadiusP)/(r2>rmin*rmin ? r2 : rmin*rmin);
            a[0]+=q*dx; a[1]+=q*dy;
        }
        Force[i][0]+=Mass[i]*a[0]; Force[i][1]+=Mass[i]*a[1];
    }
}

static void calculatePpSurfaceTension()
{
    // -------------------------------------------------------------------------
    // Pairwise inter-particle potential ("many-body potential") model.
    //
    //   U_ij(r) = PpCoef * (re/3) (1 - r/re)^3   between UNLIKE fluid types
    //   F_i     = -grad_i U = -PpCoef (1 - r/re)^2 e_ij       (repulsive)
    //
    // Like-type pairs carry no potential, so the model adds nothing in the bulk
    // and produces exactly the prescribed interfacial tension: PpCoef is fixed
    // at start-up by the broken-bond lattice sum in initializeWeight().
    //
    // The force is exactly antisymmetric, so linear momentum is conserved to
    // round-off, and the interface energy is bounded - this is the robust
    // choice at moderate density ratio (Hysing case 1) or when the curvature
    // reconstruction of the CSF is under-resolved.  At 1:1000, however, it
    // suffers the same acceleration asymmetry as any pairwise model, so CSF
    // with the density weight remains the recommended default there.
    // -------------------------------------------------------------------------
    if(!TwoPhaseActive || !(PpCoef>0.0)) return;

    const double re=PpRadius;

    #pragma acc kernels present(Force[0:ParticleCount][0:DIM],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

        double fx=0.0, fy=0.0, fz=0.0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
            if(Property[jP]==Property[iP]) continue;      // unlike pairs only

            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=re*re) continue;

            const double r=sqrt(r2);
            const double q=1.0-r/re;
            const double fmag=-PpCoef*q*q;               // negative = repulsive
            const double invr=1.0/r;
            fx+=fmag*dx*invr;
            fy+=fmag*dy*invr;
            fz+=fmag*dz*invr;
        }
        Force[iP][0]+=fx;
        Force[iP][1]+=fy;
        Force[iP][2]+=fz;
    }
}


static void calculateTwoPhaseSurfaceTension()
{
    if(!TwoPhaseActive) return;
    if(TwoPhaseStModel==2){
        // The shifting still needs the interface normal.
        calculateInterfaceGeometry();
        calculatePpSurfaceTension();
    }else if(TwoPhaseStModel==1){
        calculateTwoPhaseCSSSurfaceTension();
    }else{
        calculateCsfSurfaceTension();
    }
}

static void calculateKondoPotentialSurfaceTension()
{
    // Kondo, Koshizuka & Takimoto (2007), single-phase free-surface cohesion.
    //   F_ij = -C (r-l0)(r-re),  r < re
    #pragma acc kernels present(Force[0:ParticleCount][0:DIM],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],SurfacePotentialC[0:TYPE_COUNT])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

        double fx=0.0, fy=0.0, fz=0.0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;

            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=SurfacePotentialRadius*SurfacePotentialRadius) continue;
            const double r=sqrt(r2);

            const double Cij=0.5*(SurfacePotentialC[Property[iP]]+SurfacePotentialC[Property[jP]]);
            if(Cij==0.0) continue;

            const double fmag=-Cij*(r-ParticleSpacing)*(r-SurfacePotentialRadius);
            const double invr=1.0/r;
            fx+=fmag*dx*invr;
            fy+=fmag*dy*invr;
            fz+=fmag*dz*invr;
        }
        Force[iP][0]+=fx;
        Force[iP][1]+=fy;
        Force[iP][2]+=fz;
    }
}


static void calculateDiffuseInterface()
{
	#pragma acc kernels present(Force[0:ParticleCount][0:DIM],GravityCenter[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		const double ai = CofA[Property[iP]]*(CofK)*(CofK);
		double force[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			const double aj = CofA[Property[iP]]*(CofK)*(CofK);
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
			double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Position[jP][iD] - Position[iP][iD];
			}
			
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(RadiusG*RadiusG - rij2 > 0 && rij2 > 1.0e-30){
				const double rij = sqrt(rij2);
				const double wij = ratio_ij * wg(rij,RadiusG);
				const double wji = ratio_ji * wg(rij,RadiusG);
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (aj*GravityCenter[jP][iD]*wji-ai*GravityCenter[iP][iD]*wij)/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				const double dwij = ratio_ij * dwgdr(rij,RadiusG);
				const double dwji = ratio_ji * dwgdr(rij,RadiusG);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				double gr=0.0;
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					gr += (aj*GravityCenter[jP][iD]*dwji-ai*GravityCenter[iP][iD]*dwij)*xij[iD];
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (gr)*eij[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Force[iP][iD]+=force[iD];
		}
	}
}

static void calculateDensityP()
{
	#pragma acc kernels present(NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount],VolStrainP[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum = 0.0;
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Position[jP][iD] - Position[iP][iD];
			}
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double weight = wp(rij,radius);
				sum += weight;
			}
		}
		VolStrainP[iP] = (sum - N0p);
	}
}

static void freeIsphHighOrderCoefficients()
{
	if(IsphCoeffCount>0){
		#pragma acc exit data delete(IsphGradX[0:IsphCoeffCount],IsphGradY[0:IsphCoeffCount],IsphLap[0:IsphCoeffCount])
		free(IsphGradX);
		free(IsphGradY);
		free(IsphLap);
		IsphGradX=NULL;
		IsphGradY=NULL;
		IsphLap=NULL;
		IsphCoeffCount=0;
	}
}


static void buildIsphHighOrderCoefficients()
{
    // Common second-order QMLS reconstruction for the single-phase projection.
    // (Disabled for two-phase runs: no smooth polynomial reconstruction is
    //  admissible across a 1:1000 density jump.  The arrays are still built so
    //  that the device pointers listed in later present(...) clauses stay valid.)
    if(IsphCoeffCount!=NeighborIndCount){
        freeIsphHighOrderCoefficients();
        IsphCoeffCount=NeighborIndCount;
        IsphGradX=(double *)malloc(IsphCoeffCount*sizeof(double));
        IsphGradY=(double *)malloc(IsphCoeffCount*sizeof(double));
        IsphLap  =(double *)malloc(IsphCoeffCount*sizeof(double));

        #pragma acc enter data create(IsphGradX[0:IsphCoeffCount]) attach(IsphGradX)
        #pragma acc enter data create(IsphGradY[0:IsphCoeffCount]) attach(IsphGradY)
        #pragma acc enter data create(IsphLap[0:IsphCoeffCount]) attach(IsphLap)
        #pragma acc update device(IsphCoeffCount)
    }

    #pragma acc kernels present(IsphGradX[0:IsphCoeffCount],IsphGradY[0:IsphCoeffCount],IsphLap[0:IsphCoeffCount],IsphQmlsUse[0:ParticleCount],PpeAlpha[0:ParticleCount],Property[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){

        IsphQmlsUse[iP]=0;

        double M[5][5];
        #pragma acc loop seq
        for(int a=0;a<5;++a){
            #pragma acc loop seq
            for(int b=0;b<5;++b) M[a][b]=0.0;
        }

        int hasWall=0;
        int nFluidSupport=0;

        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const long int k=NeighborPtr[iP]+jN;
            const int jP=NeighborInd[k];

            IsphGradX[k]=0.0;
            IsphGradY[k]=0.0;
            IsphLap[k]=0.0;

            if(iP==jP) continue;

            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;

            if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
                hasWall=1;
                continue;
            }
            if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;

#ifdef TWO_DIMENSIONAL
            const double r=sqrt(r2);
            const double xi=dx/RadiusP;
            const double et=dy/RadiusP;
            const double ss=1.0-r/RadiusP;
            const double w=ss*ss;
            const double q[5]={xi,et,0.5*xi*xi,xi*et,0.5*et*et};

            #pragma acc loop seq
            for(int a=0;a<5;++a){
                #pragma acc loop seq
                for(int b=0;b<5;++b){
                    M[a][b]+=w*q[a]*q[b];
                }
            }
            ++nFluidSupport;
#endif
        }

        double Minv[5][5];
        int ok=0;
#ifdef TWO_DIMENSIONAL
        if(nFluidSupport>=5){
            double tr=0.0;
            #pragma acc loop seq
            for(int a=0;a<5;++a) tr+=fabs(M[a][a]);
            const double reg=IsphQmlsReg*(tr/5.0+1.0);
            #pragma acc loop seq
            for(int a=0;a<5;++a) M[a][a]+=reg;
            ok=invert5x5(M,Minv);
        }
#endif

        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const long int k=NeighborPtr[iP]+jN;
            const int jP=NeighborInd[k];
            if(iP==jP) continue;
            if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;

            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double r=sqrt(r2);

#ifdef TWO_DIMENSIONAL
            if(ok){
                const double xi=dx/RadiusP;
                const double et=dy/RadiusP;
                const double ss=1.0-r/RadiusP;
                const double w=ss*ss;
                const double q[5]={xi,et,0.5*xi*xi,xi*et,0.5*et*et};

                double gx=0.0, gy=0.0, lp=0.0;
                #pragma acc loop seq
                for(int a=0;a<5;++a){
                    gx+=Minv[0][a]*q[a];
                    gy+=Minv[1][a]*q[a];
                    lp+=(Minv[2][a]+Minv[4][a])*q[a];
                }

                IsphGradX[k]=w*gx/RadiusP;
                IsphGradY[k]=w*gy/RadiusP;
                IsphLap[k]=w*lp/(RadiusP*RadiusP);
            }else
#endif
            {
#ifdef TWO_DIMENSIONAL
                const double wij=wp(r,RadiusP);
                const double rMin=IsphMinPairRatio*ParticleSpacing;
                const double r2Eff=(r2>rMin*rMin ? r2 : rMin*rMin);
                IsphGradX[k]=2.0/N0p*wij*dx/r2Eff;
                IsphGradY[k]=2.0/N0p*wij*dy/r2Eff;
                IsphLap[k]=4.0/(PpeLambdaP*N0p)*wij;
#else
                IsphGradX[k]=0.0;
                IsphGradY[k]=0.0;
                IsphLap[k]=0.0;
#endif
            }
        }

#ifdef TWO_DIMENSIONAL
        int useQ=(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END &&
                  PpeAlpha[iP]>=0.999999999 && !hasWall && ok);
        if(useQ){
            int nPos=0;
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const long int k=NeighborPtr[iP]+jN;
                const int jP=NeighborInd[k];
                if(iP==jP) continue;
                if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double r2=dx*dx+dy*dy;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                const double c=IsphLap[k];
                if(!(c==c) || !(c>0.0) || !(IsphGradX[k]==IsphGradX[k]) || !(IsphGradY[k]==IsphGradY[k])){
                    useQ=0;
                }else{
                    ++nPos;
                }
            }
            if(nPos<5) useQ=0;
        }
        IsphQmlsUse[iP]=useQ;
#else
        IsphQmlsUse[iP]=0;
#endif
    }
}


static void calculateDivergenceP()
{
#ifdef TWO_DIMENSIONAL
    const double divCoef=2.0/N0p;
#else
    const double divCoef=3.0/N0p;
#endif
    #pragma acc kernels present(DivergenceP[0:ParticleCount],IsphQmlsUse[0:ParticleCount],IsphGradX[0:IsphCoeffCount],IsphGradY[0:IsphCoeffCount],Property[0:ParticleCount],Velocity[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)){
            DivergenceP[iP]=0.0;
            continue;
        }

#ifdef TWO_DIMENSIONAL
        if(IsphQmlsUse[iP]){
            double div=0.0;
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const long int k=NeighborPtr[iP]+jN;
                const int jP=NeighborInd[k];
                if(iP==jP) continue;
                if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double r2=dx*dx+dy*dy;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                div += IsphGradX[k]*(Velocity[jP][0]-Velocity[iP][0])
                     + IsphGradY[k]*(Velocity[jP][1]-Velocity[iP][1]);
            }
            DivergenceP[iP]=div;
            continue;
        }
#endif

        double sum=0.0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            if(!((FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END) ||
                 (WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END))) continue;
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double r=sqrt(r2);
            const double dux=Velocity[jP][0]-Velocity[iP][0];
            const double duy=Velocity[jP][1]-Velocity[iP][1];
            const double duz=Velocity[jP][2]-Velocity[iP][2];
            const double rMin=IsphMinPairRatio*ParticleSpacing;
            const double r2Eff=(r2>rMin*rMin ? r2 : rMin*rMin);
            sum += (dux*dx+duy*dy+duz*dz)/r2Eff * wp(r,RadiusP);
        }
        DivergenceP[iP]=divCoef*sum;
    }
}


static void calculateDivergenceTwoPhase()
{
#ifdef TWO_DIMENSIONAL
    const double divCoef=2.0/N0p;
#else
    const double divCoef=3.0/N0p;
#endif
    #pragma acc kernels present(DivergenceP[0:ParticleCount],Property[0:ParticleCount],Velocity[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)){
            DivergenceP[iP]=0.0;
            continue;
        }

        double sum=0.0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            // Two-phase PPE divergence is fluid-fluid only.  The solid-wall
            // normal flux is imposed once, through the PPE Neumann BC.
            if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;

            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;

            const double r=sqrt(r2);
            const double rMin=IsphMinPairRatio*ParticleSpacing;
            const double r2Eff=(r2>rMin*rMin ? r2 : rMin*rMin);
            const double dux=Velocity[jP][0]-Velocity[iP][0];
            const double duy=Velocity[jP][1]-Velocity[iP][1];
            const double duz=Velocity[jP][2]-Velocity[iP][2];

            sum+=(dux*dx+duy*dy+duz*dz)/r2Eff*wp(r,RadiusP);
        }
        DivergenceP[iP]=divCoef*sum;
    }
}


static void applyPpeOperator()
{
#ifdef TWO_DIMENSIONAL
    const double lapCoef=4.0/(PpeLambdaP*N0p);
#else
    const double lapCoef=6.0/(PpeLambdaP*N0p);
#endif

    // ---------------------------------------------------------------------
    // Two-phase variable-density operator:
    //
    //   A p = -div[(1/rho) grad p],   beta_ij = 2/(rho_i+rho_j)
    //
    // beta_ij is the harmonic mean of a=1/rho, i.e. the flux-consistent face
    // value.  In the same-phase limit beta_ij = 1/rho, so this reduces exactly
    // to the single-density equation divided by rho.
    // ---------------------------------------------------------------------
    if(TwoPhaseActive){
        #pragma acc kernels present(PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],PpeOpX[0:ParticleCount],PpeOpY[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount],RhoEff[0:ParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP){
            if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END) || PpeFlag[iP]==0){
                PpeOpY[iP]=0.0;
                continue;
            }

            const double rhoi=RhoEff[iP];
            const double ai=PpeAlpha[iP];
            double diag=0.0, off=0.0;

            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const int jP=NeighborInd[NeighborPtr[iP]+jN];
                if(iP==jP) continue;
                if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;

                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double dz=Position[jP][2]-Position[iP][2];
                const double r2=dx*dx+dy*dy+dz*dz;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;

                const double r=sqrt(r2);
                const double rhoj=RhoEff[jP];
                const double beta=2.0/(rhoi+rhoj+1.0e-30);
                const double c=lapCoef*wp(r,RadiusP)*beta;

                diag+=c;
                off+=c*(PpeFlag[jP]!=0 ? PpeOpX[jP] : 0.0);
            }

            PpeOpY[iP]=(diag>1.0e-30 ? diag*PpeOpX[iP]-ai*off : PpeOpX[iP]);
        }

        // Pure-Neumann problem (a closed tank completely filled by the two
        // phases): the operator has the constant null space.  Projecting it out
        // of every operator application keeps the Krylov iteration inside the
        // range space. A consistently eliminated gauge row would also be valid.
        if(PpePureNeumann){
            double sy=0.0; int ny=0;
            #pragma acc parallel loop reduction(+:sy,ny) present(PpeOpY[0:ParticleCount],PpeFlag[0:ParticleCount])
            #pragma omp parallel for reduction(+:sy,ny)
            for(int iP=0;iP<ParticleCount;++iP){ if(PpeFlag[iP]!=0){ sy+=PpeOpY[iP]; ++ny; } }
            const double my=(ny>0 ? sy/(double)ny : 0.0);
            #pragma acc kernels present(PpeOpY[0:ParticleCount],PpeFlag[0:ParticleCount])
            #pragma acc loop independent
            #pragma omp parallel for
            for(int iP=0;iP<ParticleCount;++iP){ if(PpeFlag[iP]!=0) PpeOpY[iP]-=my; }
        }
        return;
    }

    // ---------------------------------------------------------------------
    // Single-phase operator (unchanged validated path).
    // ---------------------------------------------------------------------
    #pragma acc kernels present(IsphQmlsUse[0:ParticleCount],IsphLap[0:IsphCoeffCount],PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],PpeOpX[0:ParticleCount],PpeOpY[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)){ PpeOpY[iP]=0.0; continue; }

#ifdef TWO_DIMENSIONAL
        if(IsphQmlsUse[iP]){
            double diag=0.0, off=0.0;
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const long int k=NeighborPtr[iP]+jN;
                const int jP=NeighborInd[k];
                if(iP==jP || !(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double r2=dx*dx+dy*dy;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                const double c=IsphLap[k];
                diag+=c;
                off+=c*(PpeFlag[jP]!=0 ? PpeOpX[jP] : 0.0);
            }
            PpeOpY[iP]=(diag>1.0e-30 ? diag*PpeOpX[iP]-off : PpeOpX[iP]);
            continue;
        }
#endif

        int hasWall=0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(!(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END)) continue;
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2>1.0e-30 && r2<RadiusP*RadiusP) hasWall=1;
        }

        const int useHL=(PpeAlpha[iP]>=0.999999999 && !hasWall);
        const double ai=PpeAlpha[iP];
        double diag=0.0, off=0.0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP || !(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double r=sqrt(r2);
            double c;
            if(useHL){
                const double rMin=IsphMinPairRatio*ParticleSpacing;
                const double rEff=(r>rMin ? r : rMin);
                const double cRaw=(d2wpdr2(r,RadiusP)-dwpdr(r,RadiusP)/rEff)/N0p;
                c=PpeHlNorm*cRaw;
                if(!(c>0.0)) c=lapCoef*wp(r,RadiusP);
            }else{
                c=lapCoef*wp(r,RadiusP);
            }
            diag+=c;
            off+=c*(PpeFlag[jP]!=0 ? PpeOpX[jP] : 0.0);
        }
        PpeOpY[iP]=(diag>1.0e-30 ? diag*PpeOpX[iP]-ai*off : PpeOpX[iP]);
    }
}


// -----------------------------------------------------------------------------
// Free-surface detector.  Fills PpeFlag with
//   1 : interior fluid  (PPE unknown)
//   0 : free-surface fluid (Dirichlet p = 0) or non-fluid
//
// It scans ALL fluid particles and ALL wall particles, therefore for a
// two-phase run the support is complete across the gas/liquid interface and
// the material interface is correctly NOT reported as a free surface.  Only a
// genuine free surface (fluid adjacent to vacuum) is detected.
// -----------------------------------------------------------------------------
static void detectFreeSurface()
{
#ifdef TWO_DIMENSIONAL
    const double alphaL=1.6, alphaU=1.8, dimRef=2.0;
#else
    const double alphaL=2.6, alphaU=2.8, dimRef=3.0;
#endif
    const double pi=3.14159265358979323846;
    #pragma acc kernels present(PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],PositionDivR[0:ParticleCount],Property[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)){
            PpeFlag[iP]=0; PpeAlpha[iP]=0.0; PositionDivR[iP]=0.0; continue;
        }
        double raw=0.0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            if(!((FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END) || (WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END))) continue;
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double r=sqrt(r2);
            raw += ParticleVolume*(-dwpdr(r,RadiusP))*r;
        }
        const double divR=dimRef*raw/(PositionDivRef+1.0e-30);
        PositionDivR[iP]=divR;
        double a;
        if(divR<=alphaL) a=0.0;
        else if(divR>=alphaU) a=1.0;
        else{ const double s=(divR-alphaL)/(alphaU-alphaL); a=0.5*(1.0-cos(pi*s)); }
        PpeAlpha[iP]=a;
        PpeFlag[iP]=(a>1.0e-12 ? 1 : 0);
    }
}


static void calculatePpe()
{
    if(TwoPhaseActive){
        // -----------------------------------------------------------------
        // Two-phase projection.
        //  * both phases are pressure unknowns,
        //  * the SAME free-surface detector as the single-phase branch is run
        //    on the union of the two phases, so a real free surface (open tank)
        //    still gets p = 0 while the internal gas/liquid interface does not,
        //  * if the domain is fully enclosed there is no Dirichlet row: the
        //    pure-Neumann null space is removed by a compatibility correction
        //    of the RHS and by projecting the operator, instead of by pinning
        //    one gauge particle.
        // -----------------------------------------------------------------
        TwoPhaseGauge=-1;
        #pragma acc update device(TwoPhaseGauge)

        calculateEffectiveDensity();
        detectFreeSurface();

        // Device pointers for the QMLS arrays must stay valid because later
        // kernels list them in present(...).  Allocate as usual, then disable
        // every QMLS row.
        buildIsphHighOrderCoefficients();
        #pragma acc kernels present(IsphQmlsUse[0:ParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP) IsphQmlsUse[iP]=0;

        // A gas/liquid interface is not an atmospheric pressure boundary.
        // Closed mode retains every fluid row, including under-supported rows.
        // Open mode uses binary flags; fractional off-diagonal relaxation would
        // destroy symmetry and the constant nullspace of the Neumann operator.
        #pragma acc parallel loop present(PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],Property[0:ParticleCount])
        #pragma omp parallel for
        for(int i=0;i<ParticleCount;++i){
            if(FLUID_BEGIN<=Property[i] && Property[i]<FLUID_END){
                if(!TwoPhaseOpenSurface) PpeFlag[i]=1;
                PpeAlpha[i]=(PpeFlag[i] ? 1.0 : 0.0);
            }
        }

        int nDirichlet=0;
        #pragma acc parallel loop reduction(+:nDirichlet) present(PpeFlag[0:ParticleCount],Property[0:ParticleCount])
        #pragma omp parallel for reduction(+:nDirichlet)
        for(int iP=0;iP<ParticleCount;++iP){
            if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END && PpeFlag[iP]==0) ++nDirichlet;
        }
        PpePureNeumann=(nDirichlet==0 ? 1 : 0);
        #pragma acc update device(PpePureNeumann)

        calculateDivergenceTwoPhase();
    }else{
        detectFreeSurface();
        buildIsphHighOrderCoefficients();
        calculateDivergenceP();
    }

#ifdef TWO_DIMENSIONAL
    const double lapCoef=4.0/(PpeLambdaP*N0p);
    const double gradCoef=2.0/N0p;
#else
    const double lapCoef=6.0/(PpeLambdaP*N0p);
    const double gradCoef=3.0/N0p;
#endif

    // Pre-projection diagnostic.
    double maxPreDiv=0.0, sumPreDiv2=0.0;
    int nPreDiv=0;
    #pragma acc parallel loop reduction(max:maxPreDiv) reduction(+:sumPreDiv2,nPreDiv) present(DivergenceP[0:ParticleCount],PpeAlpha[0:ParticleCount],Property[0:ParticleCount])
    #pragma omp parallel for reduction(max:maxPreDiv) reduction(+:sumPreDiv2,nPreDiv)
    for(int iP=0;iP<ParticleCount;++iP){
        if(PpeAlpha[iP]>0.99 && FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END){
            const double ad=fabs(DivergenceP[iP]);
            if(ad>maxPreDiv) maxPreDiv=ad;
            sumPreDiv2+=DivergenceP[iP]*DivergenceP[iP];
            ++nPreDiv;
        }
    }
    const double rmsPreDiv=(nPreDiv>0 ? sqrt(sumPreDiv2/(double)nPreDiv) : 0.0);

    int qmlsRows=0;
    #pragma acc parallel loop reduction(+:qmlsRows) present(IsphQmlsUse[0:ParticleCount])
    #pragma omp parallel for reduction(+:qmlsRows)
    for(int iP=0;iP<ParticleCount;++iP) qmlsRows+=IsphQmlsUse[iP];

    // Jacobi diagonal and RHS.
    #pragma acc kernels present(IsphQmlsUse[0:ParticleCount],IsphLap[0:IsphCoeffCount],PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],PpeDiag[0:ParticleCount],PpeB[0:ParticleCount],PpeR[0:ParticleCount],PpeR0[0:ParticleCount],PpeP[0:ParticleCount],PpeV[0:ParticleCount],PpeS[0:ParticleCount],PpeT[0:ParticleCount],PpePHat[0:ParticleCount],PpeOpX[0:ParticleCount],PressureP[0:ParticleCount],DivergenceP[0:ParticleCount],VolStrainP[0:ParticleCount],RhoEff[0:ParticleCount],Property[0:ParticleCount],Density[0:TYPE_COUNT],Velocity[0:ParticleCount][0:DIM],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const int isFluid=(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END);

        if(TwoPhaseActive){
            double diagTP=0.0;
            double bTP=0.0;

            if(isFluid && PpeFlag[iP]!=0){
                const double rhoi=RhoEff[iP];

                #pragma acc loop seq
                for(int jN=0;jN<NeighborCount[iP];++jN){
                    const int jP=NeighborInd[NeighborPtr[iP]+jN];
                    if(iP==jP) continue;
                    if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;

                    const double dx=Position[jP][0]-Position[iP][0];
                    const double dy=Position[jP][1]-Position[iP][1];
                    const double dz=Position[jP][2]-Position[iP][2];
                    const double r2=dx*dx+dy*dy+dz*dz;
                    if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;

                    const double r=sqrt(r2);
                    const double rhoj=RhoEff[jP];
                    const double beta=2.0/(rhoi+rhoj+1.0e-30);
                    diagTP+=lapCoef*wp(r,RadiusP)*beta;
                }

                // Full divergence source plus bounded density-error penalty.
                double dn=VolStrainP[iP]/(N0p+1.0e-30);
                if(dn> IsphDensityErrMax) dn= IsphDensityErrMax;
                if(dn<-IsphDensityErrMax) dn=-IsphDensityErrMax;
                double g=TwoPhaseSourceGamma;
                if(g<0.0) g=0.0;
                if(g>1.0) g=1.0;
                // Fixed physical relaxation time: reducing Dt must not increase
                // the density-penalty acceleration as 1/Dt^2.
                bTP = -DivergenceP[iP]/Dt + g*dn/(TwoPhaseDensityRelaxTime*TwoPhaseDensityRelaxTime);

                // Virtual-wall Neumann BC.  beta_wall = 1/rho_i and
                // dp/dn = rho_i*vn/dt cancel exactly, so no density multiplier
                // remains in the known RHS wall flux.
                double fluidSw=0.0;
                double wallSw=0.0, wallNx=0.0, wallNy=0.0;
                double wallUx=0.0, wallUy=0.0;
                double minWallR=1.0e100;

                #pragma acc loop seq
                for(int jN=0;jN<NeighborCount[iP];++jN){
                    const int jP=NeighborInd[NeighborPtr[iP]+jN];
                    if(iP==jP) continue;

                    const double dx=Position[jP][0]-Position[iP][0];
                    const double dy=Position[jP][1]-Position[iP][1];
                    const double dz=Position[jP][2]-Position[iP][2];
                    const double r2=dx*dx+dy*dy+dz*dz;
                    if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;

                    const double r=sqrt(r2);
                    const double w=wp(r,RadiusP);
                    if(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END){
                        fluidSw+=w;
                    }else if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
                        wallSw+=w;
                        wallNx+=w*(-dx/r);
                        wallNy+=w*(-dy/r);
                        wallUx+=w*Velocity[jP][0];
                        wallUy+=w*Velocity[jP][1];
                        if(r<minWallR) minWallR=r;
                    }
                }

                const double wallNmag=sqrt(wallNx*wallNx+wallNy*wallNy);
                if(wallSw>1.0e-30 && wallNmag>1.0e-30 && minWallR<1.0e99){
                    double wMiss=N0p-fluidSw;
                    if(wMiss<0.0) wMiss=0.0;
                    if(wMiss>wallSw) wMiss=wallSw;
                    if(wMiss>1.0e-30){
                        const double nx=wallNx/wallNmag;
                        const double ny=wallNy/wallNmag;
                        const double uwx=wallUx/wallSw;
                        const double uwy=wallUy/wallSw;
                        double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny;
                        // Signed wall flux: clamping separation breaks hydrostatic balance.
                        bTP-=lapCoef*wMiss*minWallR*vn/Dt;
                    }
                }

                bTP*=PpeAlpha[iP];
            }

            PpeDiag[iP]=(isFluid && PpeFlag[iP]!=0 && diagTP>1.0e-30 ? diagTP : 1.0);
            PpeB[iP]=(isFluid && PpeFlag[iP]!=0 ? bTP : 0.0);

            double p0=PressureP[iP];
            if(!isFluid || PpeFlag[iP]==0 || !(p0==p0) || fabs(p0)>1.0e12) p0=0.0;
            PressureP[iP]=p0;

            PpeOpX[iP]=(PpeFlag[iP]!=0 ? p0 : 0.0);
            PpeP[iP]=0.0;
            PpeV[iP]=0.0;
            PpeS[iP]=0.0;
            PpeT[iP]=0.0;
            PpePHat[iP]=0.0;
            continue;
        }

        double diag=0.0;
        int hasWallForLap=0;
        if(isFluid){
#ifdef TWO_DIMENSIONAL
            if(IsphQmlsUse[iP]){
                #pragma acc loop seq
                for(int jN=0;jN<NeighborCount[iP];++jN){
                    const long int k=NeighborPtr[iP]+jN;
                    const int jP=NeighborInd[k];
                    if(iP==jP || !(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                    const double dx=Position[jP][0]-Position[iP][0];
                    const double dy=Position[jP][1]-Position[iP][1];
                    const double r2=dx*dx+dy*dy;
                    if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                    diag+=IsphLap[k];
                }
            }else
#endif
            {
                #pragma acc loop seq
                for(int jN=0;jN<NeighborCount[iP];++jN){
                    const int jP=NeighborInd[NeighborPtr[iP]+jN];
                    if(iP==jP) continue;
                    if(!(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END)) continue;
                    const double dx=Position[jP][0]-Position[iP][0];
                    const double dy=Position[jP][1]-Position[iP][1];
                    const double dz=Position[jP][2]-Position[iP][2];
                    const double r2=dx*dx+dy*dy+dz*dz;
                    if(r2>1.0e-30 && r2<RadiusP*RadiusP) hasWallForLap=1;
                }
                const int useHLdiag=(PpeAlpha[iP]>=0.999999999 && !hasWallForLap);
                #pragma acc loop seq
                for(int jN=0;jN<NeighborCount[iP];++jN){
                    const int jP=NeighborInd[NeighborPtr[iP]+jN];
                    if(iP==jP) continue;
                    if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                    const double dx=Position[jP][0]-Position[iP][0];
                    const double dy=Position[jP][1]-Position[iP][1];
                    const double dz=Position[jP][2]-Position[iP][2];
                    const double r2=dx*dx+dy*dy+dz*dz;
                    if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                    const double r=sqrt(r2);
                    double c;
                    if(useHLdiag){
                        const double rMin=IsphMinPairRatio*ParticleSpacing;
                        const double rEff=(r>rMin ? r : rMin);
                        const double cRaw=(d2wpdr2(r,RadiusP)-dwpdr(r,RadiusP)/rEff)/N0p;
                        c=PpeHlNorm*cRaw;
                        if(!(c>0.0)) c=lapCoef*wp(r,RadiusP);
                    }else{
                        c=lapCoef*wp(r,RadiusP);
                    }
                    diag+=c;
                }
            }
        }
        PpeDiag[iP]=(isFluid && diag>1.0e-30 ? diag : 1.0);

        double b=0.0;
        if(isFluid && PpeAlpha[iP]>0.0){
            const double rho=Density[Property[iP]];
            b=-rho*DivergenceP[iP]/Dt;

            double fluidSw=0.0;
            double wallSw=0.0, wallNx=0.0, wallNy=0.0;
            double wallUx=0.0, wallUy=0.0;
            double minWallR=1.0e100;
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const int jP=NeighborInd[NeighborPtr[iP]+jN];
                if(iP==jP) continue;
                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double dz=Position[jP][2]-Position[iP][2];
                const double r2=dx*dx+dy*dy+dz*dz;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                const double r=sqrt(r2);
                const double w=wp(r,RadiusP);
                if(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END){
                    fluidSw+=w;
                }else if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
                    wallSw+=w;
                    wallNx+=w*(-dx/r);
                    wallNy+=w*(-dy/r);
                    wallUx+=w*Velocity[jP][0];
                    wallUy+=w*Velocity[jP][1];
                    if(r<minWallR) minWallR=r;
                }
            }
            const double wallNmag=sqrt(wallNx*wallNx+wallNy*wallNy);
            if(wallSw>1.0e-30 && wallNmag>1.0e-30 && minWallR<1.0e99){
                double wMiss=N0p-fluidSw;
                if(wMiss<0.0) wMiss=0.0;
                if(wMiss>wallSw) wMiss=wallSw;
                if(wMiss>1.0e-30){
                    const double nx=wallNx/wallNmag;
                    const double ny=wallNy/wallNmag;
                    const double uwx=wallUx/wallSw;
                    const double uwy=wallUy/wallSw;
                    double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny;
                    if(vn>0.0) vn=0.0;
                    const double gradPn=rho*vn/Dt;
                    const double cW=lapCoef*wMiss;
                    b-=cW*minWallR*gradPn;
                }
            }

            b*=PpeAlpha[iP];
        }
        PpeB[iP]=b;

        double p0=PressureP[iP];
        if(!isFluid || PpeAlpha[iP]<=1.0e-12 || !(p0==p0) || p0<0.0 || p0>1.0e12) p0=0.0;
        else p0*=PpeAlpha[iP];
        PressureP[iP]=p0;
        PpeOpX[iP]=p0;
        PpeP[iP]=0.0;
        PpeV[iP]=0.0;
        PpeS[iP]=0.0;
        PpeT[iP]=0.0;
        PpePHat[iP]=0.0;
    }

    if(TwoPhaseActive && PpePureNeumann){
        // Compatibility condition of the pure-Neumann system: b must be
        // orthogonal to the constant null vector.
        double sumBtp=0.0;
        int nBtp=0;
        #pragma acc parallel loop reduction(+:sumBtp,nBtp) present(PpeB[0:ParticleCount],PpeFlag[0:ParticleCount])
        #pragma omp parallel for reduction(+:sumBtp,nBtp)
        for(int iP=0;iP<ParticleCount;++iP){
            if(PpeFlag[iP]!=0){ sumBtp+=PpeB[iP]; ++nBtp; }
        }
        const double meanBtp=(nBtp>0 ? sumBtp/(double)nBtp : 0.0);

        #pragma acc kernels present(PpeB[0:ParticleCount],PpeFlag[0:ParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP){
            if(PpeFlag[iP]!=0) PpeB[iP]-=meanBtp;
        }

        log_printf("TwoPhase PPE: pure Neumann, compatibility mean removed=%e active=%d\n",
                   meanBtp,nBtp);
    }

    applyPpeOperator();
    #pragma acc kernels present(PpeFlag[0:ParticleCount],PpeB[0:ParticleCount],PpeOpY[0:ParticleCount],PpeR[0:ParticleCount],PpeR0[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const double r=(PpeFlag[iP]!=0 ? PpeB[iP]-PpeOpY[iP] : 0.0);
        PpeR[iP]=r;
        PpeR0[iP]=r;
    }

    double bb=0.0, rr=0.0;
    int fluidUnknown=0, freeFluid=0, transitionFluid=0;
    #pragma acc parallel loop reduction(+:bb,rr,fluidUnknown,freeFluid,transitionFluid) present(PpeB[0:ParticleCount],PpeR[0:ParticleCount],PpeAlpha[0:ParticleCount],Property[0:ParticleCount])
    #pragma omp parallel for reduction(+:bb,rr,fluidUnknown,freeFluid,transitionFluid)
    for(int iP=0;iP<ParticleCount;++iP){
        bb+=PpeB[iP]*PpeB[iP];
        rr+=PpeR[iP]*PpeR[iP];
        if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END){
            if(PpeAlpha[iP]<=1.0e-12) ++freeFluid;
            else if(PpeAlpha[iP]<0.999999) ++transitionFluid;
            else ++fluidUnknown;
        }
    }

    // Jacobi-preconditioned BiCGStab.
    double rhoOld=1.0, alpha=1.0, omega=1.0;
    int iter=0, breakdown=0;
    if(bb>1.0e-60 && rr>PpeTolerance*PpeTolerance*bb){
        for(iter=0;iter<PpeMaxIteration;++iter){
            double rhoNew=0.0;
            #pragma acc parallel loop reduction(+:rhoNew) present(PpeR0[0:ParticleCount],PpeR[0:ParticleCount])
            #pragma omp parallel for reduction(+:rhoNew)
            for(int iP=0;iP<ParticleCount;++iP) rhoNew+=PpeR0[iP]*PpeR[iP];
            if(fabs(rhoNew)<1.0e-60){ breakdown=1; break; }
            const double beta=(rhoNew/rhoOld)*(alpha/omega);

            #pragma acc kernels present(PpeP[0:ParticleCount],PpeR[0:ParticleCount],PpeV[0:ParticleCount],PpePHat[0:ParticleCount],PpeDiag[0:ParticleCount],PpeFlag[0:ParticleCount],PpeOpX[0:ParticleCount])
            #pragma acc loop independent
            #pragma omp parallel for
            for(int iP=0;iP<ParticleCount;++iP){
                if(PpeFlag[iP]!=0){
                    PpeP[iP]=PpeR[iP]+beta*(PpeP[iP]-omega*PpeV[iP]);
                    PpePHat[iP]=PpeP[iP]/PpeDiag[iP];
                    PpeOpX[iP]=PpePHat[iP];
                }else PpeP[iP]=PpePHat[iP]=PpeOpX[iP]=0.0;
            }
            applyPpeOperator();
            #pragma acc kernels present(PpeV[0:ParticleCount],PpeOpY[0:ParticleCount])
            #pragma acc loop independent
            #pragma omp parallel for
            for(int iP=0;iP<ParticleCount;++iP) PpeV[iP]=PpeOpY[iP];

            double r0v=0.0;
            #pragma acc parallel loop reduction(+:r0v) present(PpeR0[0:ParticleCount],PpeV[0:ParticleCount])
            #pragma omp parallel for reduction(+:r0v)
            for(int iP=0;iP<ParticleCount;++iP) r0v+=PpeR0[iP]*PpeV[iP];
            if(fabs(r0v)<1.0e-60){ breakdown=1; break; }
            alpha=rhoNew/r0v;

            double ss=0.0;
            #pragma acc parallel loop reduction(+:ss) present(PpeS[0:ParticleCount],PpeR[0:ParticleCount],PpeV[0:ParticleCount],PpeFlag[0:ParticleCount])
            #pragma omp parallel for reduction(+:ss)
            for(int iP=0;iP<ParticleCount;++iP){
                PpeS[iP]=(PpeFlag[iP]!=0 ? PpeR[iP]-alpha*PpeV[iP] : 0.0);
                ss+=PpeS[iP]*PpeS[iP];
            }
            if(ss<=PpeTolerance*PpeTolerance*bb){
                #pragma acc kernels present(PressureP[0:ParticleCount],PpePHat[0:ParticleCount],PpeFlag[0:ParticleCount])
                #pragma acc loop independent
                #pragma omp parallel for
                for(int iP=0;iP<ParticleCount;++iP) if(PpeFlag[iP]!=0) PressureP[iP]+=alpha*PpePHat[iP];
                rr=ss; ++iter; break;
            }

            #pragma acc kernels present(PpeS[0:ParticleCount],PpeDiag[0:ParticleCount],PpeFlag[0:ParticleCount],PpeOpX[0:ParticleCount])
            #pragma acc loop independent
            #pragma omp parallel for
            for(int iP=0;iP<ParticleCount;++iP) PpeOpX[iP]=(PpeFlag[iP]!=0 ? PpeS[iP]/PpeDiag[iP] : 0.0);
            applyPpeOperator();
            #pragma acc kernels present(PpeT[0:ParticleCount],PpeOpY[0:ParticleCount])
            #pragma acc loop independent
            #pragma omp parallel for
            for(int iP=0;iP<ParticleCount;++iP) PpeT[iP]=PpeOpY[iP];

            double ts=0.0, tt=0.0;
            #pragma acc parallel loop reduction(+:ts,tt) present(PpeT[0:ParticleCount],PpeS[0:ParticleCount])
            #pragma omp parallel for reduction(+:ts,tt)
            for(int iP=0;iP<ParticleCount;++iP){ ts+=PpeT[iP]*PpeS[iP]; tt+=PpeT[iP]*PpeT[iP]; }
            if(!(tt>1.0e-60)){ breakdown=1; break; }
            omega=ts/tt;
            if(fabs(omega)<1.0e-14){ breakdown=1; break; }

            rr=0.0;
            #pragma acc parallel loop reduction(+:rr) present(PressureP[0:ParticleCount],PpePHat[0:ParticleCount],PpeS[0:ParticleCount],PpeT[0:ParticleCount],PpeR[0:ParticleCount],PpeDiag[0:ParticleCount],PpeFlag[0:ParticleCount])
            #pragma omp parallel for reduction(+:rr)
            for(int iP=0;iP<ParticleCount;++iP){
                if(PpeFlag[iP]!=0){
                    const double shat=PpeS[iP]/PpeDiag[iP];
                    PressureP[iP]+=alpha*PpePHat[iP]+omega*shat;
                    PpeR[iP]=PpeS[iP]-omega*PpeT[iP];
                }else{
                    PressureP[iP]=0.0;
                    PpeR[iP]=0.0;
                }
                rr+=PpeR[iP]*PpeR[iP];
            }
            if(rr<=PpeTolerance*PpeTolerance*bb){ ++iter; break; }
            rhoOld=rhoNew;
        }
    }else if(bb<=1.0e-60){
        rr=0.0;
        if(TwoPhaseActive){
            #pragma acc parallel loop present(PressureP[0:ParticleCount])
            #pragma omp parallel for
            for(int i=0;i<ParticleCount;++i) PressureP[i]=0.0;
        }
    }

    if(TwoPhaseActive){
        #pragma acc parallel loop present(PpeOpX[0:ParticleCount],PressureP[0:ParticleCount],PpeFlag[0:ParticleCount])
        #pragma omp parallel for
        for(int i=0;i<ParticleCount;++i) PpeOpX[i]=(PpeFlag[i] ? PressureP[i] : 0.0);
        applyPpeOperator();
        rr=0.0;
        int bad=0;
        #pragma acc parallel loop reduction(+:rr,bad) present(PpeB[0:ParticleCount],PpeOpY[0:ParticleCount],PressureP[0:ParticleCount],PpeFlag[0:ParticleCount])
        #pragma omp parallel for reduction(+:rr,bad)
        for(int i=0;i<ParticleCount;++i){
            if(!PpeFlag[i]) continue;
            const double r=PpeB[i]-PpeOpY[i];
            if(!isfinite(r) || !isfinite(PressureP[i])) ++bad;
            else rr+=r*r;
        }
        const double target=PpeTolerance*PpeTolerance*(bb>1.0e-30 ? bb : 1.0e-30);
        if(bad || !isfinite(rr) || rr>target){
            log_printf("REJECT PPE: iter=%d true_rel=%e bad=%d breakdown=%d\n",
                       iter,sqrt(rr/(bb>1.0e-30 ? bb : 1.0e-30)),bad,breakdown);
            TwoPhaseStepFailed=1;
            return;
        }
    }

    if(TwoPhaseActive){
        // Do NOT clip negative pressure in a two-phase system: with a
        // pure-Neumann gauge the pressure is defined up to a constant, and
        // clipping destroys the pressure difference across the interface.
        #pragma acc kernels present(PressureP[0:ParticleCount],PpeFlag[0:ParticleCount],Property[0:ParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP){
            if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END) || PpeFlag[iP]==0){
                PressureP[iP]=0.0;
            } // finite pressure has already passed the true residual test
        }

        if(PpePureNeumann){
            // Zero-mean normalisation of the arbitrary additive constant.
            double sp=0.0; int np=0;
            #pragma acc parallel loop reduction(+:sp,np) present(PressureP[0:ParticleCount],PpeFlag[0:ParticleCount])
            #pragma omp parallel for reduction(+:sp,np)
            for(int iP=0;iP<ParticleCount;++iP){ if(PpeFlag[iP]!=0){ sp+=PressureP[iP]; ++np; } }
            const double mp=(np>0 ? sp/(double)np : 0.0);
            #pragma acc kernels present(PressureP[0:ParticleCount],PpeFlag[0:ParticleCount])
            #pragma acc loop independent
            #pragma omp parallel for
            for(int iP=0;iP<ParticleCount;++iP){ if(PpeFlag[iP]!=0) PressureP[iP]-=mp; }
        }
    }else{
        // Non-negative pressure for violent one-phase free-surface MPS.
        #pragma acc kernels present(PressureP[0:ParticleCount],PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],Property[0:ParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP){
            if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END) || PpeFlag[iP]==0) PressureP[iP]=0.0;
            else{
                if(!(PressureP[iP]==PressureP[iP]) || PressureP[iP]<0.0) PressureP[iP]=0.0;
                PressureP[iP]*=PpeAlpha[iP];
            }
        }
    }

    const double rel=sqrt(rr/(bb>1.0e-60 ? bb : 1.0));
    log_printf("ISPH(%s): iter=%d rel=%e q2=%d bulk=%d transition=%d free=%d%s%s\n",
               (TwoPhaseActive ? "TwoPhase-GraphPPE+FixedTimePenalty+StateGuard"
                               : "QMLS2-DGL+HL/GCMfallback+VirtualWall+KondoST+cPS-DPC,RK2,BiCGStab"),
               iter,rel,qmlsRows,fluidUnknown,transitionFluid,freeFluid,
               (iter>=PpeMaxIteration ? " (not converged)" : ""),
               (breakdown ? " (breakdown)" : ""));

    // ---------------------------------------------------------------------
    // Pressure gradient / velocity correction.
    // ---------------------------------------------------------------------
    int pressureClipped=0;
    #pragma acc parallel loop reduction(+:pressureClipped) present(RkK4U[0:ParticleCount][0:DIM],IsphQmlsUse[0:ParticleCount],IsphGradX[0:IsphCoeffCount],IsphGradY[0:IsphCoeffCount],Force[0:ParticleCount][0:DIM],Mass[0:ParticleCount],Density[0:TYPE_COUNT],RhoEff[0:ParticleCount],Property[0:ParticleCount],PressureP[0:ParticleCount],PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],Velocity[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM])
    #pragma omp parallel for reduction(+:pressureClipped)
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

        if(TwoPhaseActive){
            // Pressure-difference gradient with the PPE pair density.
            // Constant pressure gives zero acceleration. This collocated
            // gradient is not pair-antisymmetric and the PPE is not exactly D*G.
            const double rhoi=RhoEff[iP];
            double ax=0.0, ay=0.0, az=0.0;   // capped below

            double fluidSw=0.0;
            double wallSw=0.0, wallNx=0.0, wallNy=0.0, wallUx=0.0, wallUy=0.0;
            double minWallR=1.0e100;

            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const int jP=NeighborInd[NeighborPtr[iP]+jN];
                if(iP==jP) continue;

                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double dz=Position[jP][2]-Position[iP][2];
                const double r2=dx*dx+dy*dy+dz*dz;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;

                const double r=sqrt(r2);
                const double w=wp(r,RadiusP);

                if(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END){
                    fluidSw+=w;

                    const double rhoj=RhoEff[jP];
                    const double beta=2.0/(rhoi+rhoj+1.0e-30);
                    const double rMin=IsphMinPairRatio*ParticleSpacing;
                    const double r2Eff=(r2>rMin*rMin ? r2 : rMin*rMin);
                    const double pj=(PpeFlag[jP]!=0 ? PressureP[jP] : 0.0);
                    const double dpij=pj-PressureP[iP];
                    const double q=gradCoef*beta*dpij*w/r2Eff;

                    ax-=q*dx;
                    ay-=q*dy;
                    az-=q*dz;
                }else if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
                    wallSw+=w;
                    wallNx+=w*(-dx/r);
                    wallNy+=w*(-dy/r);
                    wallUx+=w*Velocity[jP][0];
                    wallUy+=w*Velocity[jP][1];
                    if(r<minWallR) minWallR=r;
                }
            }

            const double wallNmag=sqrt(wallNx*wallNx+wallNy*wallNy);
            if(wallSw>1.0e-30 && wallNmag>1.0e-30 && minWallR<1.0e99){
                double wMiss=N0p-fluidSw;
                if(wMiss<0.0) wMiss=0.0;
                if(wMiss>wallSw) wMiss=wallSw;

                if(wMiss>1.0e-30){
                    const double nx=wallNx/wallNmag;
                    const double ny=wallNy/wallNmag;
                    const double uwx=wallUx/wallSw;
                    const double uwy=wallUy/wallSw;
                    double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny;
                    // Enforce signed no-penetration correction, including separation.

                    const double rMin=IsphMinPairRatio*ParticleSpacing;
                    const double rEff=(minWallR>rMin ? minWallR : rMin);
                    const double dpGhost=-minWallR*rhoi*vn/Dt;
                    const double betaWall=1.0/(rhoi+1.0e-30);
                    const double q=gradCoef*betaWall*dpGhost*wMiss/(rEff*rEff);

                    ax-=q*(-minWallR*nx);
                    ay-=q*(-minWallR*ny);
                }
            }

            // Safety cap on the velocity change from a single projection.
            // In a healthy step this never activates; when one pathological
            // particle appears it stops that particle from taking the whole
            // run with it.
            {
                const double duCap=TwoPhaseDuCap*ParticleSpacing/(Dt+1.0e-300);
                const double dum=Dt*sqrt(ax*ax+ay*ay+az*az);
                if(dum>duCap && dum>1.0e-30){
                    ++pressureClipped;
                    const double sc=duCap/dum;
                    ax*=sc; ay*=sc; az*=sc;
                }
            }

            RkK4U[iP][0]=Dt*ax;
            RkK4U[iP][1]=Dt*ay;
            RkK4U[iP][2]=Dt*az;
            Velocity[iP][0]+=Dt*ax;
            Velocity[iP][1]+=Dt*ay;
            Velocity[iP][2]+=Dt*az;
            Force[iP][0]+=Mass[iP]*ax;
            Force[iP][1]+=Mass[iP]*ay;
            Force[iP][2]+=Mass[iP]*az;
            continue;
        }

        // ---------------- single-phase path (unchanged) ----------------
        double pMin=PressureP[iP];
        double fluidSw=0.0;
        double wallSw=0.0, wallNx=0.0, wallNy=0.0, wallUx=0.0, wallUy=0.0;
        double minWallR=1.0e100;

        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double r=sqrt(r2);
            const double w=wp(r,RadiusP);

            if(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END){
                fluidSw+=w;
                const double pj=(PpeFlag[jP]!=0 ? PressureP[jP] : 0.0);
                if(pj<pMin) pMin=pj;
            }else if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
                wallSw+=w;
                wallNx+=w*(-dx/r);
                wallNy+=w*(-dy/r);
                wallUx+=w*Velocity[jP][0];
                wallUy+=w*Velocity[jP][1];
                if(r<minWallR) minWallR=r;
            }
        }
        if(pMin<0.0) pMin=0.0;

        double gx=0.0, gy=0.0, gz=0.0;
        int usedQMLS=0;
        int usedGCM=0;

#ifdef TWO_DIMENSIONAL
        if(IsphQmlsUse[iP]){
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const long int k=NeighborPtr[iP]+jN;
                const int jP=NeighborInd[k];
                if(iP==jP) continue;
                if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double r2=dx*dx+dy*dy;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                const double pj=(PpeFlag[jP]!=0 ? PressureP[jP] : 0.0);
                const double dpij=pj-pMin;
                gx+=IsphGradX[k]*dpij;
                gy+=IsphGradY[k]*dpij;
            }
            usedQMLS=1;
        }
#endif

        if(!usedQMLS){
            double Mxx=0.0, Mxy=0.0, Myy=0.0;
            double Bx=0.0, By=0.0;
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const int jP=NeighborInd[NeighborPtr[iP]+jN];
                if(iP==jP) continue;
                if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double dz=Position[jP][2]-Position[iP][2];
                const double r2=dx*dx+dy*dy+dz*dz;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                const double r=sqrt(r2);
                const double w=wp(r,RadiusP);
                const double rMin=IsphMinPairRatio*ParticleSpacing;
                const double r2Eff=(r2>rMin*rMin ? r2 : rMin*rMin);
                const double q=w/r2Eff;
                const double pj=(PpeFlag[jP]!=0 ? PressureP[jP] : 0.0);

                Mxx+=q*dx*dx;
                Mxy+=q*dx*dy;
                Myy+=q*dy*dy;

                const double dpij=pj-pMin;
                Bx+=q*dpij*dx;
                By+=q*dpij*dy;
            }

#ifdef TWO_DIMENSIONAL
            if(PpeAlpha[iP]>=0.999999999 && wallSw<=1.0e-30){
                const double det=Mxx*Myy-Mxy*Mxy;
                const double scale=Mxx+Myy;
                if(det>1.0e-12*(scale*scale+1.0e-30)){
                    gx=( Myy*Bx-Mxy*By)/det;
                    gy=(-Mxy*Bx+Mxx*By)/det;
                    gz=0.0;
                    usedGCM=1;
                }
            }
#endif
        }

        if(!usedQMLS && !usedGCM){
            gx=0.0; gy=0.0; gz=0.0;
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const int jP=NeighborInd[NeighborPtr[iP]+jN];
                if(iP==jP) continue;
                if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double dz=Position[jP][2]-Position[iP][2];
                const double r2=dx*dx+dy*dy+dz*dz;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                const double r=sqrt(r2);
                const double w=wp(r,RadiusP);
                const double rMin=IsphMinPairRatio*ParticleSpacing;
                const double r2Eff=(r2>rMin*rMin ? r2 : rMin*rMin);
                const double pj=(PpeFlag[jP]!=0 ? PressureP[jP] : 0.0);
                const double q=(pj-pMin)*w/r2Eff;
                gx+=q*dx; gy+=q*dy; gz+=q*dz;
            }
            gx*=gradCoef; gy*=gradCoef; gz*=gradCoef;
        }

        const double rho=Density[Property[iP]];
        const double wallNmag=sqrt(wallNx*wallNx+wallNy*wallNy);
        if(wallSw>1.0e-30 && wallNmag>1.0e-30 && minWallR<1.0e99){
            double wMiss=N0p-fluidSw;
            if(wMiss<0.0) wMiss=0.0;
            if(wMiss>wallSw) wMiss=wallSw;
            if(wMiss>1.0e-30){
                const double nx=wallNx/wallNmag;
                const double ny=wallNy/wallNmag;
                const double uwx=wallUx/wallSw;
                const double uwy=wallUy/wallSw;
                double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny;
                if(vn>0.0) vn=0.0;
                const double gradPn=rho*vn/Dt;
                double pGhost=PressureP[iP]-minWallR*gradPn;
                if(pGhost<0.0) pGhost=0.0;
                const double rMin=IsphMinPairRatio*ParticleSpacing;
                const double rEff=(minWallR>rMin ? minWallR : rMin);
                const double q=(pGhost-pMin)*wMiss/(rEff*rEff);
                gx+=gradCoef*q*(-minWallR*nx);
                gy+=gradCoef*q*(-minWallR*ny);
            }
        }

        const double ax=-gx/rho, ay=-gy/rho, az=-gz/rho;
        Velocity[iP][0]+=Dt*ax;
        Velocity[iP][1]+=Dt*ay;
        Velocity[iP][2]+=Dt*az;
        Force[iP][0]+=Mass[iP]*ax;
        Force[iP][1]+=Mass[iP]*ay;
        Force[iP][2]+=Mass[iP]*az;
    }

    if(TwoPhaseActive){
        // Complete the midpoint position update with the pressure impulse.
        // Separate kernel: never modify positions while neighbors read them.
        #pragma acc parallel loop present(Position[0:ParticleCount][0:DIM],RkK4U[0:ParticleCount][0:DIM],Property[0:ParticleCount])
        #pragma omp parallel for
        for(int i=0;i<ParticleCount;++i){
            if(Property[i]>=FLUID_BEGIN && Property[i]<FLUID_END)
                for(int d=0;d<DIM;++d) Position[i][d]+=0.5*Dt*RkK4U[i][d];
        }
    }

    // Wall-only diagnostic after the pressure correction.
    double maxWallPen=0.0;
    int nWallNear=0;
    #pragma acc parallel loop reduction(max:maxWallPen) reduction(+:nWallNear) present(Property[0:ParticleCount],Velocity[0:ParticleCount][0:DIM],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for reduction(max:maxWallPen) reduction(+:nWallNear)
    for(int iP=0;iP<ParticleCount;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;
        double wallSw=0.0, wallNx=0.0, wallNy=0.0, wallUx=0.0, wallUy=0.0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(!(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END)) continue;
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double r2=dx*dx+dy*dy;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double r=sqrt(r2);
            const double w=wp(r,RadiusP);
            wallSw+=w;
            wallNx+=w*(-dx/r);
            wallNy+=w*(-dy/r);
            wallUx+=w*Velocity[jP][0];
            wallUy+=w*Velocity[jP][1];
        }
        const double wn=sqrt(wallNx*wallNx+wallNy*wallNy);
        if(wallSw>1.0e-30 && wn>1.0e-30){
            const double nx=wallNx/wn, ny=wallNy/wn;
            const double uwx=wallUx/wallSw, uwy=wallUy/wallSw;
            const double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny;
            const double pen=(vn<0.0 ? -vn : 0.0);
            if(pen>maxWallPen) maxWallPen=pen;
            ++nWallNear;
        }
    }
    if(TwoPhaseActive && pressureClipped) TwoPhaseStepFailed=1;
    log_printf("ISPH wall diagnostic: near=%d max_penetration_velocity=%e\n",nWallNear,maxWallPen);

    if(TwoPhaseActive) calculateDivergenceTwoPhase();
    else               calculateDivergenceP();
    double maxPostDiv=0.0, sumPostDiv2=0.0;
    int nPostDiv=0;
    #pragma acc parallel loop reduction(max:maxPostDiv) reduction(+:sumPostDiv2,nPostDiv) present(DivergenceP[0:ParticleCount],PpeAlpha[0:ParticleCount],Property[0:ParticleCount])
    #pragma omp parallel for reduction(max:maxPostDiv) reduction(+:sumPostDiv2,nPostDiv)
    for(int iP=0;iP<ParticleCount;++iP){
        if(PpeAlpha[iP]>0.99 && FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END){
            const double ad=fabs(DivergenceP[iP]);
            if(ad>maxPostDiv) maxPostDiv=ad;
            sumPostDiv2+=DivergenceP[iP]*DivergenceP[iP];
            ++nPostDiv;
        }
    }
    const double rmsPostDiv=(nPostDiv>0 ? sqrt(sumPostDiv2/(double)nPostDiv) : 0.0);
    log_printf("ISPH divergence bulk: pre max=%e rms=%e | post max=%e rms=%e\n",
               maxPreDiv,rmsPreDiv,maxPostDiv,rmsPostDiv);
}


static void applyTwoPhaseShifting()
{
    // Pre-projection Fickian shift. wp is already a dimensionless particle
    // weight (its normalization includes dp^DIM); do not multiply by V again.
    if(!TwoPhaseActive) return;

    calculateInterfaceGeometry();

    const double Rps=0.2;
    double shiftCoef=TwoPhaseShiftCoef;
    if(shiftCoef<=0.0) return;
    double umax=0.0;
    #pragma acc parallel loop reduction(max:umax) present(Velocity[0:ParticleCount][0:DIM])
    #pragma omp parallel for reduction(max:umax)
    for(int i=0;i<ParticleCount;++i){
        const double u=sqrt(Velocity[i][0]*Velocity[i][0]+Velocity[i][1]*Velocity[i][1]+Velocity[i][2]*Velocity[i][2]);
        if(u>umax) umax=u;
    }
    const double Dshift=shiftCoef*ParticleSpacing*umax*Dt;
    const double Wdp=wp(ParticleSpacing,RadiusP);
    const double repairScale=fmin(1.0,Dt/TwoPhaseDtMax);
    const double maxShift=0.02*ParticleSpacing*repairScale;
    const double duShiftCap=0.10*ParticleSpacing/(Dt+1.0e-300);

    double keepN=TwoPhaseIfShiftNormal;
    if(keepN<0.0) keepN=0.0;
    if(keepN>1.0) keepN=1.0;
    double velCorr=TwoPhaseShiftVelCorr;
    if(velCorr<0.0) velCorr=0.0;
    if(velCorr>1.0) velCorr=1.0;

#ifdef TWO_DIMENSIONAL
    const double dimRef=2.0;
    const double AFST=1.5;
#else
    const double dimRef=3.0;
    const double AFST=2.75;
#endif

    // TmpDoubleVector : position shift.   RkK4U : matching velocity correction.
    #pragma acc kernels present(Mass[0:ParticleCount],TmpDoubleVector[0:ParticleCount][0:DIM],RkK4U[0:ParticleCount][0:DIM],Property[0:ParticleCount],PositionDivR[0:ParticleCount],IfDelta[0:ParticleCount],IfNormal[0:ParticleCount][0:DIM],IfGrad[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        TmpDoubleVector[iP][0]=0.0;
        TmpDoubleVector[iP][1]=0.0;
        TmpDoubleVector[iP][2]=0.0;
        RkK4U[iP][0]=0.0;
        RkK4U[iP][1]=0.0;
        RkK4U[iP][2]=0.0;

        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

        // Free-surface correction factor.  A_C <= 0 -> genuine free surface or
        // splash: no shifting at all.
        double AC=(TwoPhaseOpenSurface ? (PositionDivR[iP]-AFST)/(dimRef-AFST) : 1.0);
        if(AC<=0.0) continue;
        if(AC>1.0) AC=1.0;

        double gcx=0.0, gcy=0.0, gcz=0.0;
        double sepX=0.0,sepY=0.0,sepZ=0.0;
        double wallSw=0.0, wallNx=0.0, wallNy=0.0;

        // Renormalised velocity gradient, fluid neighbours only.
        double Mxx=0.0, Mxy=0.0, Myy=0.0;
        double Bux=0.0, Buy=0.0, Bvx=0.0, Bvy=0.0;

        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            const int isFluidJ=(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END);
            const int isWallJ =(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END);
            if(!isFluidJ && !isWallJ) continue;

            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double r=sqrt(r2);
            const double w=wp(r,RadiusP);
            const double dw=dwpdr(r,RadiusP);
            const double gradFactor=-dw/r;
            // Bounded geometric separation of close pairs, with no velocity
            // impulse. Interface-normal separation is allowed only for actual
            // overlap; bulk/interface Fickian shifts remain tangential.
            const double target=0.85*ParticleSpacing;
            if(r<target){
                const double share=(isWallJ ? 1.0 : Mass[jP]/(Mass[iP]+Mass[jP]));
                const double repair=0.25*repairScale*share*(target-r)/r;
                sepX-=repair*dx; sepY-=repair*dy; sepZ-=repair*dz;
            }

            double fij=0.0;
            if(Wdp>1.0e-30){
                const double wr=w/Wdp;
                const double wr2=wr*wr;
                fij=Rps*wr2*wr2;
            }
            const double fac=(1.0+fij)*gradFactor/(N0p+wp(0.0,RadiusP));
            gcx+=fac*dx;
            gcy+=fac*dy;
            gcz+=fac*dz;

            if(isWallJ){
                wallSw+=w;
                wallNx+=w*(-dx/r);
                wallNy+=w*(-dy/r);
                continue;
            }

            const double g=ParticleVolume*gradFactor;
            Mxx+=g*dx*dx;
            Mxy+=g*dx*dy;
            Myy+=g*dy*dy;

            const double dux=Velocity[jP][0]-Velocity[iP][0];
            const double dvy=Velocity[jP][1]-Velocity[iP][1];
            Bux+=g*dux*dx;
            Buy+=g*dux*dy;
            Bvx+=g*dvy*dx;
            Bvy+=g*dvy*dy;
        }

        double sx=-AC*Dshift*gcx;
        double sy=-AC*Dshift*gcy;
        double sz=-AC*Dshift*gcz;

        // Damp (do not remove) the phase-interface normal component.
        if(IfDelta[iP]*ParticleSpacing>TwoPhaseIfThreshold){
            const double nx=IfGrad[iP][0]/IfDelta[iP];
            const double ny=IfGrad[iP][1]/IfDelta[iP];
            const double nz=IfGrad[iP][2]/IfDelta[iP];
            const double sn=sx*nx+sy*ny+sz*nz;
            const double cut=(1.0-keepN)*sn;
            sx-=cut*nx;
            sy-=cut*ny;
            sz-=cut*nz;
        }

        // Remove the wall-normal component: preserve the wall spacing.
        const double wn=sqrt(wallNx*wallNx+wallNy*wallNy);
        if(wallSw>1.0e-30 && wn>1.0e-30){
            if(IfDelta[iP]*ParticleSpacing>TwoPhaseIfThreshold){ sx=sy=sz=0.0; }
            const double nx=wallNx/wn, ny=wallNy/wn;
            const double sn=sx*nx+sy*ny;
            sx-=sn*nx;
            sy-=sn*ny;
        }

        sx+=sepX; sy+=sepY; sz+=sepZ;
        const double smag=sqrt(sx*sx+sy*sy+sz*sz);
        if(smag>maxShift && smag>1.0e-30){
            const double q=maxShift/smag;
            sx*=q; sy*=q; sz*=q;
        }

        TmpDoubleVector[iP][0]=sx;
        TmpDoubleVector[iP][1]=sy;
        TmpDoubleVector[iP][2]=sz;

#ifdef TWO_DIMENSIONAL
        // u_i += (delta_r . grad) u_i, with the renormalised gradient.
        if(velCorr>0.0){
            const double det=Mxx*Myy-Mxy*Mxy;
            const double sc=Mxx+Myy;
            if(det>1.0e-10*(sc*sc+1.0e-30)){
                const double dudx=( Myy*Bux-Mxy*Buy)/det;
                const double dudy=(-Mxy*Bux+Mxx*Buy)/det;
                const double dvdx=( Myy*Bvx-Mxy*Bvy)/det;
                const double dvdy=(-Mxy*Bvx+Mxx*Bvy)/det;

                double du=velCorr*(sx*dudx+sy*dudy);
                double dv=velCorr*(sx*dvdx+sy*dvdy);

                const double dum=sqrt(du*du+dv*dv);
                if(dum>duShiftCap && dum>1.0e-30){
                    const double q=duShiftCap/dum;
                    du*=q; dv*=q;
                }
                RkK4U[iP][0]=du;
                RkK4U[iP][1]=dv;
            }
        }
#endif
    }

    #pragma acc kernels present(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],TmpDoubleVector[0:ParticleCount][0:DIM],RkK4U[0:ParticleCount][0:DIM],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END){
            Position[iP][0]+=TmpDoubleVector[iP][0];
            Position[iP][1]+=TmpDoubleVector[iP][1];
            Position[iP][2]+=TmpDoubleVector[iP][2];
            Velocity[iP][0]+=RkK4U[iP][0];
            Velocity[iP][1]+=RkK4U[iP][1];
            Velocity[iP][2]+=RkK4U[iP][2];
        }
    }
}


static void logTwoPhaseStabilityDiagnostics()
{
    if(!TwoPhaseActive) return;

    // Diagnostic only; does not change the solution.
    calculateDivergenceTwoPhase();

    double minPairRatio=1.0e100;
    double maxUg=0.0,maxUl=0.0;
    double maxPg=0.0,maxPl=0.0;
    double maxDiv=0.0,sumDiv2=0.0;
    double maxDens=0.0;
    int nDiv=0;

    #pragma acc parallel loop reduction(min:minPairRatio) reduction(max:maxUg,maxUl,maxPg,maxPl,maxDiv,maxDens) reduction(+:sumDiv2,nDiv) present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],PressureP[0:ParticleCount],DivergenceP[0:ParticleCount],VolStrainP[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for reduction(min:minPairRatio) reduction(max:maxUg,maxUl,maxPg,maxPl,maxDiv,maxDens) reduction(+:sumDiv2,nDiv)
    for(int iP=0;iP<ParticleCount;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

        const double ux=Velocity[iP][0],uy=Velocity[iP][1],uz=Velocity[iP][2];
        const double umag=sqrt(ux*ux+uy*uy+uz*uz);
        const double pp=fabs(PressureP[iP]);
        if(Property[iP]==0){
            if(umag>maxUg) maxUg=umag;
            if(pp>maxPg) maxPg=pp;
        }else if(Property[iP]==1){
            if(umag>maxUl) maxUl=umag;
            if(pp>maxPl) maxPl=pp;
        }

        const double ad=fabs(DivergenceP[iP]);
        if(ad>maxDiv) maxDiv=ad;
        sumDiv2+=DivergenceP[iP]*DivergenceP[iP];
        ++nDiv;

        const double dn=fabs(VolStrainP[iP])/(N0p+1.0e-30);
        if(dn>maxDens) maxDens=dn;

        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(jP<=iP) continue;
            if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;

            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r=sqrt(dx*dx+dy*dy+dz*dz);
            if(r>1.0e-30 && r<RadiusP){
                const double q=r/(ParticleSpacing+1.0e-30);
                if(q<minPairRatio) minPairRatio=q;
            }
        }
    }

    const double rmsDiv=(nDiv>0 ? sqrt(sumDiv2/(double)nDiv) : 0.0);
    if(minPairRatio>1.0e50) minPairRatio=0.0;

    log_printf("TwoPhase stability: min_r/dp=%e max|dn/n0|=%e Umax(type0,type1)=(%e,%e) |P|max(type0,type1)=(%e,%e) postDiv(max,rms)=(%e,%e)\n",
               minPairRatio,maxDens,maxUg,maxUl,maxPg,maxPl,maxDiv,rmsDiv);
}



static void applyParticleShifting()
{
    // -------------------------------------------------------------------------
    // Single-phase cPS + DPC free-surface regularization (unchanged).
    // -------------------------------------------------------------------------

    const double Rps=0.2;
    const double Dshift=0.5*ParticleSpacing*ParticleSpacing;
    const double Wdp=wp(ParticleSpacing,RadiusP);
    const double maxShift=0.10*ParticleSpacing;

#ifdef TWO_DIMENSIONAL
    const double dimRef=2.0;
    const double AFST=1.5;
#else
    const double dimRef=3.0;
    const double AFST=2.75;
#endif

    const double lambdaDpc=0.10;
    const double eta=0.01*ParticleSpacing;
    const double gmag=sqrt(Gravity[0]*Gravity[0]+Gravity[1]*Gravity[1]+Gravity[2]*Gravity[2]);

    double pMaxNow=0.0;
    #pragma acc parallel loop reduction(max:pMaxNow) present(PressureP[0:ParticleCount],Property[0:ParticleCount])
    #pragma omp parallel for reduction(max:pMaxNow)
    for(int iP=0;iP<ParticleCount;++iP){
        if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END){
            const double pp=PressureP[iP];
            if(pp>pMaxNow) pMaxNow=pp;
        }
    }

    #pragma acc kernels present(TmpDoubleVector[0:ParticleCount][0:DIM],RkK4U[0:ParticleCount][0:DIM],Property[0:ParticleCount],PositionDivR[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],PressureP[0:ParticleCount],Density[0:TYPE_COUNT])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        TmpDoubleVector[iP][0]=0.0;
        TmpDoubleVector[iP][1]=0.0;
        TmpDoubleVector[iP][2]=0.0;
        RkK4U[iP][0]=0.0;
        RkK4U[iP][1]=0.0;
        RkK4U[iP][2]=0.0;

        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

        double AC=(PositionDivR[iP]-AFST)/(dimRef-AFST);
        const int useDpc=(AC<=0.0 ? 1 : 0);
        if(AC<0.0) AC=0.0;
        if(AC>1.0) AC=1.0;

        if(!useDpc){
            double gcx=0.0, gcy=0.0, gcz=0.0;
            double wallSw=0.0, wallNx=0.0, wallNy=0.0;

            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const int jP=NeighborInd[NeighborPtr[iP]+jN];
                if(iP==jP) continue;
                if(!((FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END) ||
                     (WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END))) continue;

                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double dz=Position[jP][2]-Position[iP][2];
                const double r2=dx*dx+dy*dy+dz*dz;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                const double r=sqrt(r2);
                const double w=wp(r,RadiusP);
                const double dw=dwpdr(r,RadiusP);
                const double gradFactor=-dw/r;

                double fij=0.0;
                if(Wdp>1.0e-30){
                    const double wr=w/Wdp;
                    const double wr2=wr*wr;
                    fij=Rps*wr2*wr2;
                }
                const double fac=ParticleVolume*(1.0+fij)*gradFactor;
                gcx+=fac*dx;
                gcy+=fac*dy;
                gcz+=fac*dz;

                if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
                    wallSw+=w;
                    wallNx+=w*(-dx/r);
                    wallNy+=w*(-dy/r);
                }
            }

            double sx=-AC*Dshift*gcx;
            double sy=-AC*Dshift*gcy;
            double sz=-AC*Dshift*gcz;

            const double wn=sqrt(wallNx*wallNx+wallNy*wallNy);
            if(wallSw>1.0e-30 && wn>1.0e-30){
                const double nx=wallNx/wn, ny=wallNy/wn;
                const double sn=sx*nx+sy*ny;
                sx-=sn*nx;
                sy-=sn*ny;
            }

            const double smag=sqrt(sx*sx+sy*sy+sz*sz);
            if(smag>maxShift && smag>1.0e-30){
                const double q=maxShift/smag;
                sx*=q; sy*=q; sz*=q;
            }
            TmpDoubleVector[iP][0]=sx;
            TmpDoubleVector[iP][1]=sy;
            TmpDoubleVector[iP][2]=sz;
        }
        else{
            double dvx=0.0, dvy=0.0, dvz=0.0;
            const double rhoi=Density[Property[iP]];
            const double pMin=rhoi*gmag*ParticleSpacing;

            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const int jP=NeighborInd[NeighborPtr[iP]+jN];
                if(iP==jP) continue;
                if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;

                const double rx=Position[iP][0]-Position[jP][0];
                const double ry=Position[iP][1]-Position[jP][1];
                const double rz=Position[iP][2]-Position[jP][2];
                const double r2=rx*rx+ry*ry+rz*rz;
                if(r2<=1.0e-30 || r2>=ParticleSpacing*ParticleSpacing) continue;
                const double r=sqrt(r2);
                const double q=r/ParticleSpacing;

                const double oneMinusQ=1.0-q;
                const double wq=oneMinusQ*oneMinusQ*oneMinusQ*oneMinusQ*(4.0*q+1.0);
                const double q05=0.5;
                const double om05=1.0-q05;
                const double w05=om05*om05*om05*om05*(4.0*q05+1.0);
                double chi=(w05>1.0e-30 ? sqrt(wq/w05) : 0.0);
                if(chi<0.0) chi=0.0;
                if(chi>1.0) chi=1.0;
                const double kappa=(q<0.5 ? 1.0 : chi);

                const double vx=Velocity[iP][0]-Velocity[jP][0];
                const double vy=Velocity[iP][1]-Velocity[jP][1];
                const double vz=Velocity[iP][2]-Velocity[jP][2];
                const double vr=vx*rx+vy*ry+vz*rz;
                const double denom=r2+eta*eta;

                if(vr<0.0){
                    const double s=vr/denom;
                    dvx+=kappa*s*(-rx);
                    dvy+=kappa*s*(-ry);
                    dvz+=kappa*s*(-rz);
                }
                else{
                    double ptilde=lambdaDpc*fabs(PressureP[iP]+PressureP[jP]);
                    const double pUpper=lambdaDpc*pMaxNow;
                    if(pUpper>0.0 && ptilde>pUpper) ptilde=pUpper;
                    if(ptilde<pMin) ptilde=pMin;
                    const double pb=ptilde*chi;
                    const double fac=Dt/rhoi*pb/denom;
                    dvx+=fac*rx;
                    dvy+=fac*ry;
                    dvz+=fac*rz;
                }
            }

            double sx=Dt*dvx, sy=Dt*dvy, sz=Dt*dvz;
            const double smag=sqrt(sx*sx+sy*sy+sz*sz);
            if(smag>maxShift && smag>1.0e-30){
                const double q=maxShift/smag;
                sx*=q; sy*=q; sz*=q;
                dvx*=q; dvy*=q; dvz*=q;
            }

            TmpDoubleVector[iP][0]=sx;
            TmpDoubleVector[iP][1]=sy;
            TmpDoubleVector[iP][2]=sz;
            RkK4U[iP][0]=dvx;
            RkK4U[iP][1]=dvy;
            RkK4U[iP][2]=dvz;
        }
    }

    #pragma acc kernels present(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],TmpDoubleVector[0:ParticleCount][0:DIM],RkK4U[0:ParticleCount][0:DIM],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END){
            Position[iP][0]+=TmpDoubleVector[iP][0];
            Position[iP][1]+=TmpDoubleVector[iP][1];
            Position[iP][2]+=TmpDoubleVector[iP][2];
            Velocity[iP][0]+=RkK4U[iP][0];
            Velocity[iP][1]+=RkK4U[iP][1];
            Velocity[iP][2]+=RkK4U[iP][2];
        }
    }
}

static void calculateViscosityV(){

	#pragma acc kernels present(Force[0:ParticleCount][0:DIM],Mu[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount],Velocity[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
    	double force[DIM]={0.0,0.0,0.0};
    	#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;

			const int isWallJ=(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END);
			// Preserve the validated single-phase free-slip path exactly; WallNoSlip
			// is a two-phase/Hysing option only.
			if(isWallJ && (!TwoPhaseActive || !WallNoSlip)) continue;
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Position[jP][iD] - Position[iP][iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            
        	if(RadiusV*RadiusV - rij2 > 0 && rij2 > 1.0e-30){
                const double rij = sqrt(rij2);
                const double rijMin=IsphMinPairRatio*ParticleSpacing;
                const double rijEff=(rij>rijMin ? rij : rijMin);
                const double dwij = -dwvdr(rij,RadiusV);
            	const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
        		double uij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
				}
				// Wall pairs use the fluid viscosity (the wall carries none, so
				// a harmonic mean would silently switch the no-slip BC off).
				const double muij = (isWallJ
				                     ? Mu[iP]
				                     : 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]+1.0e-300));
        		#pragma acc loop seq
            	for(int iD=0;iD<DIM;++iD){
            		#ifdef TWO_DIMENSIONAL
            		force[iD] += 8.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rijEff*ParticleVolume;
            		#else
            		force[iD] += 10.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rijEff*ParticleVolume;
            		#endif
            	}
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}


static void calculateGravity(){
	
	#pragma acc kernels present(Force[0:ParticleCount][0:DIM],Mass[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Force[iP][0] += Mass[iP]*Gravity[0];
        Force[iP][1] += Mass[iP]*Gravity[1];
        Force[iP][2] += Mass[iP]*Gravity[2];
    }
}

static void calculateAcceleration()
{
	#pragma acc kernels present(Force[0:ParticleCount][0:DIM],Mass[0:ParticleCount],Velocity[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Velocity[iP][0] += Force[iP][0]/Mass[iP]*Dt;
        Velocity[iP][1] += Force[iP][1]/Mass[iP]*Dt;
        Velocity[iP][2] += Force[iP][2]/Mass[iP]*Dt;
    }
}


static void calculateWall()
{
	
	#pragma acc kernels present(Force[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){
        Force[iP][0] = 0.0;
        Force[iP][1] = 0.0;
        Force[iP][2] = 0.0;
    }
	
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM],Property[0:ParticleCount],Velocity[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){
		const int iProp = Property[iP];
		double r0[DIM] = {Position[iP][0]-WallCenter[iProp][0],Position[iP][1]-WallCenter[iProp][1],Position[iP][2]-WallCenter[iProp][2]};
		const double (&R)[DIM][DIM] = WallRotation[iProp];
		const double (&w)[DIM] = WallOmega[iProp];

		double r[DIM];
		r[0] = R[0][0]*r0[0]+R[0][1]*r0[1]+R[0][2]*r0[2];
		r[1] = R[1][0]*r0[0]+R[1][1]*r0[1]+R[1][2]*r0[2];
		r[2] = R[2][0]*r0[0]+R[2][1]*r0[1]+R[2][2]*r0[2];
		Velocity[iP][0] = w[1]*r[2]-w[2]*r[1] + WallVelocity[iProp][0];
		Velocity[iP][1] = w[2]*r[0]-w[0]*r[2] + WallVelocity[iProp][1];
		Velocity[iP][2] = w[0]*r[1]-w[1]*r[0] + WallVelocity[iProp][2];
		Position[iP][0] = r[0] + WallCenter[iProp][0] + WallVelocity[iProp][0]*Dt;
		Position[iP][1] = r[1] + WallCenter[iProp][1] + WallVelocity[iProp][1]*Dt;
		Position[iP][2] = r[2] + WallCenter[iProp][2] + WallVelocity[iProp][2]*Dt;
		
	}
	
	#pragma acc kernels present(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		WallCenter[iProp][0] += WallVelocity[iProp][0]*Dt;
		WallCenter[iProp][1] += WallVelocity[iProp][1]*Dt;
		WallCenter[iProp][2] += WallVelocity[iProp][2]*Dt;
	}
}



static void calculateVirialStressAtParticle()
{
	const double (*x)[DIM] = Position;
	const double (*v)[DIM] = Velocity;
	

	#pragma acc kernels present(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD]=0.0;
			}
		}
	}
	
	#pragma acc kernels present(NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],PressureP[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM],x[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = x[jP][iD] - x[iP][iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			if(RadiusP*RadiusP - rij2 > 0 && rij2 > 1.0e-30){
				const double rij = sqrt(rij2);
				const double dwij = dwpdr(rij,RadiusP);
				double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = (PressureP[iP])*gradw[iD]*ParticleVolume;
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}
	
	#pragma acc kernels present(Mu[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM],v[0:ParticleCount][0:DIM],x[0:ParticleCount][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP=NeighborInd[ NeighborPtr[iP]+jN ];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = x[jP][iD] - x[iP][iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			if(RadiusV*RadiusV - rij2 > 0 && rij2 > 1.0e-30){
				const double rij = sqrt(rij2);
				const double dwij = -dwvdr(rij,RadiusV);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				const double vij[DIM] = {v[jP][0]-v[iP][0],v[jP][1]-v[iP][1],v[jP][2]-v[iP][2]};
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]+1.0e-300);
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#ifdef TWO_DIMENSIONAL
					fij[iD] = 8.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#else
					fij[iD] = 10.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#endif
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=0.5*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}

	#pragma acc kernels present(VirialPressureAtParticle[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#ifdef TWO_DIMENSIONAL
		VirialPressureAtParticle[iP]=-1.0/2.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]);
		#else 
		VirialPressureAtParticle[iP]=-1.0/3.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]+VirialStressAtParticle[iP][2][2]);
		#endif
	}

}



static void calculatePeriodicBoundary( void )
{
    // Closed dam-break tank: intentionally no periodic wrapping.
    // Physical wall particles provide the boundary.
}
