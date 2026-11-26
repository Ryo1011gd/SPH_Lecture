
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cerrno>
#include <assert.h>
#include <omp.h>
#if defined(SPH_FVM_AIR) && !defined(THREE_DIMENSIONAL)   // 2D spray scavenging build (make SPRAY=1)
#include "fvm/fvm_core.inc"         // VOF_FVM_2Phase solver as namespace fvm (before the macros below)
#endif

#include "errorfunc.h"
#include "log.h"

const double DOUBLE_ZERO[32]={0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0};

using namespace std;
// 2D unless built with -DTHREE_DIMENSIONAL (make DIM=3). The 3D build has no
// interface curvature, QMLS or Kondo surface tension (2D-only code paths): it
// runs the single-phase solver, e.g. the 3D spray in FVM air (make DIM=3 SPRAY=1).
#ifndef THREE_DIMENSIONAL
#define TWO_DIMENSIONAL
#endif

// Comment out this ONE line to compile without thermal/phase-change physics.
#define TEMPERATURE
// Comment out this ONE line to compile without meltable rigid bodies (rigid_body.inc).
#define RIGID
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
// Backward Euler viscosity; Jacobi-preconditioned coupled-vector CG.
static int ViscosityMaxIteration = 3000; // finite default; override in .data
static double ViscosityTolerance = 1.0e-8;
static double ViscosityAbsoluteTolerance = 1.0e-12; // velocity units
static int    PpeMaxIteration = 500;
static double PpeTolerance    = 1.0e-7;
// Single-phase PND drift correction (Asai et al. 2012 style):
//   b += PpeDensityGamma*rho*max(n-n0,0)/(n0*Dt^2)
// The predictor moves x with the un-projected u*, so the bulk compresses by
// O(Dt^2) per step and the velocity-divergence source alone never removes it.
// Gamma is the fraction of the compression removed per step; 0 = off.
static double PpeDensityGamma = 0.0;
// Single-phase projection limiter (0 = off): after the pressure correction
// |v_i| <= PpeDuLimit*max_{j in N(i)+i}|v*_j| + 2|g|Dt, i.e. the projection may not
// create a new local speed maximum. Stops isolated pressure spikes next to
// p=0 surface rows (3D impact) from launching particles.
static double PpeDuLimit = 0.0;
// Position correction after the projection (0 = off, legacy behaviour):
//   x^{n+1} = x* + Dt (u^{n+1} - u*)   (Koshizuka MPS / Cummins-Rudman ISPH)
// Without it x moves with the un-projected u* only, so gravity alone compresses a
// resting column by g Dt^2 per step (hydrostatic V&V: ~8 % after 2000 steps).
static int PpePositionCorrection = 0;
// 2D single-phase pressure gradient (0 = off, legacy): 1 uses the renormalised
// (GCM) gradient also on free-surface/transition rows and wall rows, with wall
// particles as hydrostatic ghosts p_w = p_i + f rho g.(x_w-x_i).  The legacy
// uncorrected kernel sum on one-sided rows under-supports gravity (hydrostatic
// V&V: surface rows fall at ~0.6 g, the wall column slides down at ~0.3 g).
static int PpeWallGCM = 0;
#pragma acc declare create(PpeDensityGamma,PpeWallGCM)
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

// Log verbosity (data file: LogVerbose 0|1).  0 = one line per step
// (time, Dt, PPE/viscosity iterations); 1 = full per-step diagnostics.
static int LogVerbose      = 0;
static int LogLastPpeIter  = 0;
static int LogLastViscIter = 0;

// Internal numerical stabilization constants.
static const double IsphEcsAlpha      = 2.5e-3;
static const double IsphDensityErrMax = 1.0e-2;
static const double IsphMinPairRatio  = 0.35;
// MPS-specific pressure rows (Khayyer-Gotoh higher-order Laplacian, QMLS rows)
// are used only with -DLEGACY_MPS_OPERATORS; the default ISPH build uses the SPH
// kernel-gradient operators everywhere.
#ifdef LEGACY_MPS_OPERATORS
#define ISPH_USE_HL   1
#define ISPH_USE_QMLS 1
#else
#define ISPH_USE_HL   0
#define ISPH_USE_QMLS 0
#endif
// Wall-normal coherence |sum w e|/sum w is >=0.67 for a flat 3D wall (>=0.8 in 2D).
// Below this the walls oppose each other (narrow gap): the averaged normal is
// shrunk instead of renormalised, so 1/|N| cannot amplify noise.
#ifdef TWO_DIMENSIONAL
static const double WallNormalCoherence = 0.0;   // 2D V&V path unchanged
#else
static const double WallNormalCoherence = 0.5;
#endif
// Single-phase pressure gradient near walls.  2D: one ghost pressure on the
// averaged normal (V&V path).  3D: every wall particle in the kernel is a ghost
// neighbour with p_w = p_i + rho g.(x_w-x_i) (+ impact term very close to the
// wall).  The wall particles fill the missing support exactly, so the pMin
// offset cancels at 3D edges, corners and narrow gaps, where the single ghost
// on one normal pushed fluid out of the corner.
#ifdef TWO_DIMENSIONAL
#define WALL_GHOST_PER_PARTICLE 0
#else
#define WALL_GHOST_PER_PARTICLE 1
#endif
// p_w adds WallGhostHydrostatic*f*rho g.(x_w-x_i), f=min(1,p_i/(rho|g|RadiusP)):
// the exact ghost is rho(g-a_i).(x_w-x_i); a body in free fall (a=g) has p~0,
// so f switches the hydrostatic part off there and on inside a resting pool.
static double WallGhostHydrostatic = 1.0;
#pragma acc declare create(WallGhostHydrostatic)
static const double IsphQmlsReg       = 1.0e-10;
static const int    IsphPressureFloor = 0;


static int TwoPhaseActive = 0;
static int TwoPhaseGauge  = -1;
#pragma acc declare create(TwoPhaseActive,TwoPhaseGauge)

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
static double TwoPhaseDuCap         = 0.10;
static int    TwoPhaseMinFluidNb    = 4;
#pragma acc declare create(TwoPhaseSourceGamma,TwoPhaseRhoSmooth,TwoPhaseShiftCoef)
#pragma acc declare create(TwoPhaseIfThreshold,PpePureNeumann)
#pragma acc declare create(TwoPhaseIfShiftNormal,TwoPhaseShiftVelCorr,TwoPhaseDuCap,TwoPhaseMinFluidNb)


static int    TwoPhaseStModel        = 0;
static double TwoPhaseStSmoothRatio  = 1.0;   // colour smoothing radius / RadiusP
static int    TwoPhaseStDoubleSmooth = 1;     // 1: apply the Shepard filter twice
static double TwoPhaseStDensityScale = 1.0;
static double TwoPhaseStZeroNet      = 0.0;   // remove the residual net ST force
static double TwoPhaseStDeltaMin     = 0.05;
static double TwoPhaseCurvMax        = 2.0;
static double TwoPhasePpRadiusRatio  = 3.1;   // pairwise potential cut-off / dp
static int    WallNoSlip             = 1;     // no-slip traction close to the wall
static int    WallHardNonPenetration = 1;     // geometric/kinematic wall barrier
static int    WallViscMorris         = 0;     // [V&V] 1: Morris (1997) extrapolated wall velocity in the
                                              //      viscous wall coupling (factor 1+min(dj/di,1.5))
static int    WallViscTangential     = 0;     // [V&V] 1: wall viscous coupling acts on the tangential
                                              //      velocity only (normal left to the PPE)
// Adaptive wall model: ordinary near-wall flow is weakly affected, while a
// predicted impact is stopped before crossing the wall layer.
static double WallMinGapRatio        = 0.75;  // hard geometric centre-distance / dp
static double WallImpactBufferRatio  = 0.10;  // predictive collision buffer / dp
static double WallImpactCfl          = 0.50;  // max fraction of remaining clearance per step
static double WallPressureFullRatio  = 1.05;  // full virtual-wall pressure BC below this r/dp
static double WallPressureRangeRatio = 1.75;  // wall pressure BC fades to zero here
static double WallViscFullRatio      = 1.10;  // full no-slip viscous coupling below this r/dp
static double WallViscRangeRatio     = 1.80;  // no-slip viscous coupling fades to zero here
// [V&V, ported from results/HighViscosity_VnV/bin/NP_patch.diff; defaults = previous behaviour]
static int    WallNoSeparation       = 0;     // 1: wall-contact particles may not leave the wall (u_n>0 removed)
static double PpeMinPressure         = 0.0;   // single-phase pressure floor (0: original non-negative clip)
static double WallContactRatio       = 1.2;   // wall-contact distance to the nearest wall particle, in dp

static double RadiusST = 0.0;                 // colour smoothing radius
static double MrefST   = 1.0;                 // interior M diagonal at RadiusST
static double PpRadius = 0.0;
static double PpCoef   = 0.0;                 // calibrated pairwise coefficient
#pragma acc declare create(TwoPhaseStModel,TwoPhaseStSmoothRatio,TwoPhaseStDoubleSmooth)
#pragma acc declare create(TwoPhaseStDensityScale,TwoPhaseStZeroNet,TwoPhaseStDeltaMin)
#pragma acc declare create(TwoPhaseCurvMax,PpeMinPressure,WallNoSeparation,WallContactRatio,TwoPhasePpRadiusRatio,WallNoSlip,WallHardNonPenetration,WallViscTangential,WallViscMorris,WallMinGapRatio,WallImpactBufferRatio,WallImpactCfl,WallPressureFullRatio,WallPressureRangeRatio,WallViscFullRatio,WallViscRangeRatio)
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
// time after which a wall type stops moving (e.g. inflow piston); default: never
static double WallMotionEndTime[WALL_END]={1e30,1e30,1e30,1e30};
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
static void ppeSaveUStar();
static void ppeCorrectPosition();
static int enforceWallNonPenetration(int correctPosition, int correctVelocity);
static void applyPpeOperator();
static void applyParticleShifting();
static void applyTwoPhaseShifting();
static void logTwoPhaseStabilityDiagnostics();
static void buildIsphHighOrderCoefficients();
static void freeIsphHighOrderCoefficients();
static int solveImplicitViscosity();
static void applyWallNoSeparation();
static int visDonParseDataLine(const char *buf);   // vis_deeponet.inc
static void visDonLogSettings();
static void visDonLogSummary();
#ifdef USE_AMGX
static void amgxLogSummary();
#endif
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

// -----------------------------------------------------------------------------
// Pressure / viscous kernels.
// Default (ISPH): Wendland C2 kernel with compact support h (= RadiusP, RadiusV),
//   W(r) = sigma/h^d (1-s)^4 (1+4s),  s = r/h,  sigma = 7/pi (2D), 21/(2 pi) (3D),
// returned multiplied by the particle volume V0 = dx^d, i.e. integrating to V0,
// which is the normalization of the legacy weights: every caller (viscous pair
// 8 mu V0 (-w')/r, kernel sums normalized by N0p, ...) keeps its meaning.
// -DLEGACY_MPS_OPERATORS: the previous MPS-type weight (1-r/h)^2 / Sw.
// -----------------------------------------------------------------------------
#ifndef LEGACY_MPS_OPERATORS
#ifdef TWO_DIMENSIONAL
#define WENDLAND_SIGMA (7.0/M_PI)
#define WENDLAND_HD(h) ((h)*(h))
#else
#define WENDLAND_SIGMA (21.0/(2.0*M_PI))
#define WENDLAND_HD(h) ((h)*(h)*(h))
#endif
#pragma acc routine seq
static double wendlandW(const double r, const double h){
    const double s=r/h; if(s>=1.0) return 0.0;
    const double a=1.0-s;
    return ParticleVolume*WENDLAND_SIGMA/WENDLAND_HD(h)*a*a*a*a*(1.0+4.0*s);
}
#pragma acc routine seq
static double wendlanddW(const double r, const double h){
    const double s=r/h; if(s>=1.0) return 0.0;
    const double a=1.0-s;
    return ParticleVolume*WENDLAND_SIGMA/WENDLAND_HD(h)*(-20.0*s*a*a*a)/h;
}
#pragma acc routine seq
static double wendlandd2W(const double r, const double h){
    const double s=r/h; if(s>=1.0) return 0.0;
    const double a=1.0-s;
    return ParticleVolume*WENDLAND_SIGMA/WENDLAND_HD(h)*(-20.0*a*a*(1.0-4.0*s))/(h*h);
}
#pragma acc routine seq
static double wp(const double r, const double h){ return wendlandW(r,h); }
#pragma acc routine seq
static double dwpdr(const double r, const double h){ return wendlanddW(r,h); }
#pragma acc routine seq
static double d2wpdr2(const double r, const double h){ return wendlandd2W(r,h); }
#pragma acc routine seq
static double wv(const double r, const double h){ return wendlandW(r,h); }
#pragma acc routine seq
static double dwvdr(const double r, const double h){ return wendlanddW(r,h); }

#else  // LEGACY_MPS_OPERATORS
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


#endif // LEGACY_MPS_OPERATORS

// -----------------------------------------------------------------------------
// Pair weights of the pressure operators (one place for the discretization).
//   Laplacian  <lap p>_i  = sum_j isphLapW(r_ij) (p_j - p_i)
//   gradient   <grad p>_i = gradCoef * sum_j isphGradQ(r_ij) (p_j - p_ref) x_ij   (gradCoef = d/N0p)
//   divergence <div u>_i  = gradCoef * sum_j isphGradQ(r_ij) (u_j - u_i).x_ij
// ISPH (default): SPH kernel-gradient forms (Cummins & Rudman / Morris Laplacian
//   2 V (-W') r/(r^2+eta^2), eta = 0.05 h; gradient V (-W')/r x_ij).
// LEGACY_MPS_OPERATORS: MPS forms 2d/(lambda n0) w and (d/n0) w/r^2.
// The missing-support wall terms convert a missing number density wMiss at the
// wall distance r into the same pair weights.
// -----------------------------------------------------------------------------
#pragma acc routine seq
static double isphLapW(const double r){
#ifdef LEGACY_MPS_OPERATORS
#ifdef TWO_DIMENSIONAL
    return 4.0/(PpeLambdaP*N0p)*wp(r,RadiusP);
#else
    return 6.0/(PpeLambdaP*N0p)*wp(r,RadiusP);
#endif
#else
    const double eta2=0.0025*RadiusP*RadiusP;
    return 2.0*(-dwpdr(r,RadiusP))*r/(r*r+eta2);          // dwpdr already carries V0
#endif
}
#pragma acc routine seq
static double isphGradQ(const double r){
#ifdef LEGACY_MPS_OPERATORS
    const double rMin=IsphMinPairRatio*ParticleSpacing;
    const double r2Eff=(r>rMin ? r*r : rMin*rMin);
    return wp(r,RadiusP)/r2Eff;
#else
#ifdef TWO_DIMENSIONAL
    const double gradCoefRef=2.0/N0p;
#else
    const double gradCoefRef=3.0/N0p;
#endif
    return (-dwpdr(r,RadiusP))/fmax(r,1.0e-12*RadiusP)/gradCoefRef;   // dwpdr already carries V0
#endif
}
#pragma acc routine seq
static double isphMissLap(const double rWall, const double wMiss){
    const double rMin=IsphMinPairRatio*ParticleSpacing;
    const double r=(rWall>rMin ? rWall : rMin);
    const double w=wp(r,RadiusP);
    return (w>1.0e-300 ? wMiss*isphLapW(r)/w : 0.0);
}
#pragma acc routine seq
static double isphMissGrad(const double rWall, const double wMiss){
    const double rMin=IsphMinPairRatio*ParticleSpacing;
    const double r=(rWall>rMin ? rWall : rMin);
    const double w=wp(r,RadiusP);
    return (w>1.0e-300 ? wMiss*isphGradQ(r)/w : 0.0);
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


#ifdef TEMPERATURE
// ----------------------- Thermal / phase-change model -----------------------
// Specific enthalpy [J/kg], reference h=0 at solidus (solid side).
// Pairwise conduction is conservative for symmetric neighbor lists. 2D uses
// energy per unit out-of-plane depth, consistently with Mass=rho*dp^2.
static int ThermalEnabled=0; // enabled only by a successfully opened thermal.data
static char ThermalDataFile[1024]="thermal.data";
static int ThermalCoolingEnabled=0;
static int ThermalMaxSubsteps=100000;
static double ThermalSafety=0.45;
static double ThermalAmbient=273.0, ThermalCoolingH=1200.0;
static double ThermalEmissivity=0.0, ThermalAreaFactor=6.0;
static const double ThermalSBC=5.670374419e-8;
static double InitialTemperature[TYPE_COUNT]={NAN,NAN,NAN,NAN};
static double HeatCapacityType[TYPE_COUNT]={NAN,NAN,NAN,NAN};
static double ConductivityType[TYPE_COUNT]={NAN,NAN,NAN,NAN};
static double SolidusType[TYPE_COUNT]={NAN,NAN,NAN,NAN};
static double LiquidusType[TYPE_COUNT]={NAN,NAN,NAN,NAN};
static double LatentHeatType[TYPE_COUNT]={0.0,0.0,0.0,0.0};
// Volumetric heat input per unit mass [W/kg] (e.g. decay heat / direct electrical heating). 0 = off.
static double HeatSourceType[TYPE_COUNT]={0.0,0.0,0.0,0.0};
// Conductivity [W/(m K)] used while a particle is liquid (fs < CriticalSolidFraction), e.g. molten
// concrete mixed into a gas-sparged pool. <0 = off (always Conductivity).
static double LiquidConductivityType[TYPE_COUNT]={-1.0,-1.0,-1.0,-1.0};
static double CriticalSolidFraction[TYPE_COUNT]={0.6,0.6,0.6,0.6};
static double ThermalViscosityExponent[TYPE_COUNT]={15.0,15.0,20.0,20.0};
static double ThermalViscosityJump[TYPE_COUNT]={100.0,100.0,1.0,1.0};
static double ThermalViscosityMaxMultiplier=1.0e12;
static int ThermalFixedTemperature[TYPE_COUNT]={0,0,1,1};
static int ThermalCoolingType[TYPE_COUNT]={1,1,0,0};
// ThermalFreezeSolid 1: a fluid particle with fs >= CriticalSolidFraction[Type] is a stationary
// wall (Type WALL_BEGIN, u = 0) during the flow step, like a fixed rigid body (rigidFixedAsWall);
// the thermal step sees its own Type again, so it re-melts when fs drops. 0 (default) = off.
static int ThermalFreezeSolid=0;
static int *FreezeOwnType=NULL;   // [ParticleCount] by original id: own Type while frozen, -1 otherwise
static int FreezeWallOn=0, FreezeLastCount=0;
static double *Temperature=NULL,*TemperatureOld=NULL,*SolidFraction=NULL;
static double *SpecificEnthalpy=NULL,*Cp=NULL,*Conductivity=NULL;
static double *SolidusTemp=NULL,*LiquidusTemp=NULL,*LatentH=NULL;
static double *ThermalRhs=NULL,*ThermalRow=NULL;
#pragma acc declare create(Temperature,TemperatureOld,SolidFraction,SpecificEnthalpy,Cp,Conductivity,SolidusTemp,LiquidusTemp,LatentH,ThermalRhs,ThermalRow)
#pragma acc declare create(HeatCapacityType,ConductivityType,SolidusType,LiquidusType,LatentHeatType,CriticalSolidFraction,HeatSourceType,LiquidConductivityType)
#pragma acc declare create(ThermalFixedTemperature,ThermalCoolingType,ThermalViscosityExponent,ThermalViscosityJump,ThermalViscosityMaxMultiplier)

static int thermalParseDataLine(const char *line)
{
    char key[80]; if(sscanf(line,"%79s",key)!=1) return 0;
    const char *v=line+strspn(line," \t"); v+=strlen(key);
    int parsed=0, expected=1;
    if(!strcmp(key,"ThermalEnabled")) parsed=sscanf(v,"%d",&ThermalEnabled);
    else if(!strcmp(key,"ThermalCoolingEnabled")) parsed=sscanf(v,"%d",&ThermalCoolingEnabled);
    else if(!strcmp(key,"ThermalMaxSubsteps")) parsed=sscanf(v,"%d",&ThermalMaxSubsteps);
    else if(!strcmp(key,"ThermalFreezeSolid")) parsed=sscanf(v,"%d",&ThermalFreezeSolid);
    else if(!strcmp(key,"ThermalSafety")) parsed=sscanf(v,"%lf",&ThermalSafety);
    else if(!strcmp(key,"ThermalAmbient")) parsed=sscanf(v,"%lf",&ThermalAmbient);
    else if(!strcmp(key,"ThermalCoolingH")) parsed=sscanf(v,"%lf",&ThermalCoolingH);
    else if(!strcmp(key,"ThermalEmissivity")) parsed=sscanf(v,"%lf",&ThermalEmissivity);
    else if(!strcmp(key,"ThermalAreaFactor")) parsed=sscanf(v,"%lf",&ThermalAreaFactor);
    else if(!strcmp(key,"ThermalViscosityMaxMultiplier")) parsed=sscanf(v,"%lf",&ThermalViscosityMaxMultiplier);
    else if(!strcmp(key,"ThermalFixedTemperature")){
        expected=TYPE_COUNT;parsed=sscanf(v,"%d %d %d %d",&ThermalFixedTemperature[0],&ThermalFixedTemperature[1],&ThermalFixedTemperature[2],&ThermalFixedTemperature[3]);
    }else if(!strcmp(key,"ThermalCoolingType")){
        expected=TYPE_COUNT;parsed=sscanf(v,"%d %d %d %d",&ThermalCoolingType[0],&ThermalCoolingType[1],&ThermalCoolingType[2],&ThermalCoolingType[3]);
    }else{
        double *a=NULL;
        if(!strcmp(key,"InitialTemperature")) a=InitialTemperature;
        else if(!strcmp(key,"HeatCapacity")) a=HeatCapacityType;
        else if(!strcmp(key,"Conductivity")) a=ConductivityType;
        else if(!strcmp(key,"SolidusTemperature")) a=SolidusType;
        else if(!strcmp(key,"LiquidusTemperature")) a=LiquidusType;
        else if(!strcmp(key,"LatentHeat")) a=LatentHeatType;
        else if(!strcmp(key,"ThermalHeatSource")) a=HeatSourceType;
        else if(!strcmp(key,"ThermalLiquidConductivity")) a=LiquidConductivityType;
        else if(!strcmp(key,"CriticalSolidFraction")) a=CriticalSolidFraction;
        else if(!strcmp(key,"ThermalViscosityExponent")) a=ThermalViscosityExponent;
        else if(!strcmp(key,"ThermalViscosityJump")) a=ThermalViscosityJump;
        if(!a) return 0;
        expected=TYPE_COUNT;parsed=sscanf(v,"%lf %lf %lf %lf",a,a+1,a+2,a+3);
    }
    if(parsed!=expected){log_printf("Invalid thermal data: %s\n",line);exit(EXIT_FAILURE);}
    return 1;
}

// The main .data selects the filename only. Physical thermal settings live in
// that file; the file is resolved relative to the MAIN .data directory.
static int thermalParseMainLine(const char *line)
{
    char key[80]={0}; if(sscanf(line,"%79s",key)!=1) return 0;
    if(!strcmp(key,"ThermalDataFile")){
        const char *v=line+strspn(line," \t");v+=strlen(key);
        char path[1024]={0};
        if(sscanf(v,"%1023s",path)!=1){log_printf("ThermalDataFile requires a filename\n");exit(EXIT_FAILURE);}
        strcpy(ThermalDataFile,path);return 1;
    }
    if(!strncmp(key,"Thermal",7) || !strcmp(key,"InitialTemperature") || !strcmp(key,"HeatCapacity") ||
       !strcmp(key,"Conductivity") || !strcmp(key,"SolidusTemperature") || !strcmp(key,"LiquidusTemperature") ||
       !strcmp(key,"LatentHeat") || !strcmp(key,"CriticalSolidFraction")){
        log_printf("Thermal setting %s in main .data ignored; place it in thermal.data\n",key);
        return 1;
    }
    return 0;
}

static void readThermalDataFile(const char *mainDataFilename)
{
    ThermalEnabled=0; // missing file always means isothermal, including restarts
    char path[2048];int length=0;
    if(ThermalDataFile[0]=='/') length=snprintf(path,sizeof(path),"%s",ThermalDataFile);
    else{
        const char *slash=strrchr(mainDataFilename,'/');
        const int prefix=(slash ? (int)(slash-mainDataFilename+1):0);
        length=snprintf(path,sizeof(path),"%.*s%s",prefix,mainDataFilename,ThermalDataFile);
    }
    if(length<0 || (size_t)length>=sizeof(path)){log_printf("Thermal data path too long\n");exit(EXIT_FAILURE);}
    errno=0;FILE *fp=(fopen)(path,"r"); // bypass err_fopen macro: a missing file must fall back to isothermal
    if(!fp){
        if(errno==ENOENT || errno==ENOTDIR){
            log_printf("Thermal: %s not found; temperature/solidification disabled, isothermal flow continues\n",path);
            return;
        }
        log_printf("Cannot read thermal data %s: %s\n",path,strerror(errno));exit(EXIT_FAILURE);
    }
    ThermalEnabled=1; // existing file opts in, unless it explicitly says 0
    char line[2048];int number=0;
    while(fgets(line,sizeof(line),fp)){
        ++number;
        if(!strchr(line,'\n') && !feof(fp)){
            log_printf("Thermal data line too long: %s:%d\n",path,number);fclose(fp);exit(EXIT_FAILURE);
        }
        char *comment=strchr(line,'#');if(comment)*comment='\0';
        const char *content=line+strspn(line," \t\r\n");if(!*content)continue;
        if(!thermalParseDataLine(content)){
            log_printf("Unknown thermal setting at %s:%d: %s\n",path,number,content);fclose(fp);exit(EXIT_FAILURE);
        }
    }
    const int failed=ferror(fp);fclose(fp);
    if(failed){log_printf("Error reading thermal data %s\n",path);exit(EXIT_FAILURE);}
    if(ThermalEnabled!=0 && ThermalEnabled!=1){log_printf("ThermalEnabled must be 0 or 1\n");exit(EXIT_FAILURE);}
    log_printf("Thermal: loaded %s, enabled=%d\n",path,ThermalEnabled);
}

#pragma acc routine seq
static double thermalHFromT(double T,double cp,double Ts,double Tl,double L)
{
    double fl=0.0;
    if(Tl>Ts) fl=fmin(1.0,fmax(0.0,(T-Ts)/(Tl-Ts)));
    else fl=(T>Ts ? 1.0:0.0); // T==Tm defaults to solid; restart h specifies mixtures.
    return cp*(T-Ts)+L*fl;
}

#pragma acc routine seq
static double thermalTFromH(double h,double cp,double Ts,double Tl,double L)
{
    const double hl=cp*(Tl-Ts)+L;
    if(h<=0.0) return Ts+h/cp;
    if(h>=hl) return Tl+(h-hl)/cp;
    if(Tl>Ts) return Ts+h/(cp+L/(Tl-Ts));
    return Ts; // isothermal latent-heat plateau, including pure metals
}

#pragma acc routine seq
static double thermalFsFromH(double h,double cp,double Ts,double Tl,double L)
{
    const double hl=cp*(Tl-Ts)+L;
    if(h<=0.0) return 1.0;
    if(h>=hl) return 0.0;
    return 1.0-h/hl;
}

static void thermalAllocate()
{
    if(!ThermalEnabled) return;
    Temperature=(double*)malloc(ParticleCount*sizeof(double));
    TemperatureOld=(double*)malloc(ParticleCount*sizeof(double));
    SpecificEnthalpy=(double*)malloc(ParticleCount*sizeof(double));
    SolidFraction=(double*)malloc(ParticleCount*sizeof(double));
    Cp=(double*)malloc(ParticleCount*sizeof(double));
    Conductivity=(double*)malloc(ParticleCount*sizeof(double));
    SolidusTemp=(double*)malloc(ParticleCount*sizeof(double));
    LiquidusTemp=(double*)malloc(ParticleCount*sizeof(double));
    LatentH=(double*)malloc(ParticleCount*sizeof(double));
    ThermalRhs=(double*)malloc(ParticleCount*sizeof(double));
    ThermalRow=(double*)malloc(ParticleCount*sizeof(double));
    if(!Temperature||!TemperatureOld||!SpecificEnthalpy||!SolidFraction||!Cp||!Conductivity||!SolidusTemp||!LiquidusTemp||!LatentH||!ThermalRhs||!ThermalRow){log_printf("Thermal allocation failed\n");exit(EXIT_FAILURE);}
    #pragma acc enter data create(Temperature[0:ParticleCount])
    #pragma acc enter data attach(Temperature)
    #pragma acc enter data create(TemperatureOld[0:ParticleCount])
    #pragma acc enter data attach(TemperatureOld)
    #pragma acc enter data create(SpecificEnthalpy[0:ParticleCount])
    #pragma acc enter data attach(SpecificEnthalpy)
    #pragma acc enter data create(SolidFraction[0:ParticleCount])
    #pragma acc enter data attach(SolidFraction)
    #pragma acc enter data create(Cp[0:ParticleCount])
    #pragma acc enter data attach(Cp)
    #pragma acc enter data create(Conductivity[0:ParticleCount])
    #pragma acc enter data attach(Conductivity)
    #pragma acc enter data create(SolidusTemp[0:ParticleCount])
    #pragma acc enter data attach(SolidusTemp)
    #pragma acc enter data create(LiquidusTemp[0:ParticleCount])
    #pragma acc enter data attach(LiquidusTemp)
    #pragma acc enter data create(LatentH[0:ParticleCount])
    #pragma acc enter data attach(LatentH)
    #pragma acc enter data create(ThermalRhs[0:ParticleCount])
    #pragma acc enter data attach(ThermalRhs)
    #pragma acc enter data create(ThermalRow[0:ParticleCount])
    #pragma acc enter data attach(ThermalRow)
}

static void calculateSolidFraction()
{
    if(!ThermalEnabled) return;
#ifdef _OPENACC
    #pragma acc parallel loop present(Property[0:ParticleCount],Temperature[0:ParticleCount],SpecificEnthalpy[0:ParticleCount],SolidFraction[0:ParticleCount],Cp[0:ParticleCount],Conductivity[0:ParticleCount],SolidusTemp[0:ParticleCount],LiquidusTemp[0:ParticleCount],LatentH[0:ParticleCount],HeatCapacityType[0:TYPE_COUNT],ConductivityType[0:TYPE_COUNT],SolidusType[0:TYPE_COUNT],LiquidusType[0:TYPE_COUNT],LatentHeatType[0:TYPE_COUNT],LiquidConductivityType[0:TYPE_COUNT],CriticalSolidFraction[0:TYPE_COUNT])
#else
    #pragma omp parallel for
#endif
    for(int i=0;i<ParticleCount;++i){
        const int t=Property[i];
        Cp[i]=HeatCapacityType[t];Conductivity[i]=ConductivityType[t];
        SolidusTemp[i]=SolidusType[t];LiquidusTemp[i]=LiquidusType[t];LatentH[i]=LatentHeatType[t];
        Temperature[i]=thermalTFromH(SpecificEnthalpy[i],Cp[i],SolidusTemp[i],LiquidusTemp[i],LatentH[i]);
        SolidFraction[i]=(t>=WALL_BEGIN ? 0.0 : thermalFsFromH(SpecificEnthalpy[i],Cp[i],SolidusTemp[i],LiquidusTemp[i],LatentH[i]));
        if(t<WALL_BEGIN && LiquidConductivityType[t]>=0.0 && SolidFraction[i]<CriticalSolidFraction[t]) Conductivity[i]=LiquidConductivityType[t];
    }
}

static void calculateViscosity()
{
    if(!ThermalEnabled) return;
#ifdef _OPENACC
    #pragma acc parallel loop present(Property[0:ParticleCount],Mu[0:ParticleCount],SolidFraction[0:ParticleCount],ShearViscosity[0:TYPE_COUNT],CriticalSolidFraction[0:TYPE_COUNT],ThermalViscosityExponent[0:TYPE_COUNT],ThermalViscosityJump[0:TYPE_COUNT],LatentHeatType[0:TYPE_COUNT])
#else
    #pragma omp parallel for
#endif
    for(int i=0;i<ParticleCount;++i){
        const int t=Property[i]; double factor=1.0;
        if(t>=FLUID_BEGIN && t<FLUID_END && LatentHeatType[t]>0.0){
            const double logcap=log(ThermalViscosityMaxMultiplier);
            double logfactor=ThermalViscosityExponent[t]*SolidFraction[i];
            if(SolidFraction[i]>=CriticalSolidFraction[t]) logfactor+=log(ThermalViscosityJump[t]);
            factor=exp(fmin(logfactor,logcap));
        }
        Mu[i]=ShearViscosity[t]*factor;
    }
}

static void initializeThermal()
{
    if(ThermalEnabled!=0 && ThermalEnabled!=1){log_printf("ThermalEnabled must be 0 or 1\n");exit(EXIT_FAILURE);}
    if(!ThermalEnabled) return;
    if((ThermalCoolingEnabled!=0 && ThermalCoolingEnabled!=1) || !(Dt>0.0) || !isfinite(Dt) ||
       !(ThermalSafety>0.0 && ThermalSafety<=0.5) || ThermalMaxSubsteps<1 ||
       !(ThermalAmbient>=0.0) || !isfinite(ThermalAmbient) || !(ThermalCoolingH>=0.0) || !isfinite(ThermalCoolingH) ||
       !(ThermalEmissivity>=0.0 && ThermalEmissivity<=1.0) || !(ThermalAreaFactor>=0.0) || !isfinite(ThermalAreaFactor) ||
       !(ThermalViscosityMaxMultiplier>=1.0) || !isfinite(ThermalViscosityMaxMultiplier)){
        log_printf("Invalid thermal settings (temperatures must be Kelvin)\n");exit(EXIT_FAILURE);
    }
    int used[TYPE_COUNT]={0};for(int i=0;i<ParticleCount;++i) used[Property[i]]=1;
    for(int t=0;t<TYPE_COUNT;++t){
        if(!used[t]) continue;
        if(!(HeatCapacityType[t]>0.0)||!isfinite(HeatCapacityType[t]) || !(ConductivityType[t]>=0.0)||!isfinite(ConductivityType[t]) ||
           !(SolidusType[t]>=0.0)||!isfinite(SolidusType[t]) || !(LiquidusType[t]>=SolidusType[t])||!isfinite(LiquidusType[t]) ||
           !(LatentHeatType[t]>=0.0)||!isfinite(LatentHeatType[t]) || !(Density[t]>0.0)||!isfinite(Density[t]) || !isfinite(HeatSourceType[t]) ||
           !(CriticalSolidFraction[t]>=0.0 && CriticalSolidFraction[t]<=1.0) ||
           !(ThermalViscosityExponent[t]>=0.0)||!isfinite(ThermalViscosityExponent[t]) ||
           !(ThermalViscosityJump[t]>=1.0)||!isfinite(ThermalViscosityJump[t]) ||
           (ThermalFixedTemperature[t]!=0 && ThermalFixedTemperature[t]!=1) || (ThermalCoolingType[t]!=0 && ThermalCoolingType[t]!=1)){
            log_printf("Missing/invalid thermal material parameters for Type%d\n",t);exit(EXIT_FAILURE);
        }
    }
    for(int i=0;i<ParticleCount;++i){
        const int t=Property[i];
        if(!isfinite(SpecificEnthalpy[i])){
            if(!isfinite(Temperature[i]) || Temperature[i]<0.0){log_printf("Missing/invalid initial Kelvin temperature for particle %d\n",i);exit(EXIT_FAILURE);}
            SpecificEnthalpy[i]=thermalHFromT(Temperature[i],HeatCapacityType[t],SolidusType[t],LiquidusType[t],LatentHeatType[t]);
        }
        const double T=thermalTFromH(SpecificEnthalpy[i],HeatCapacityType[t],SolidusType[t],LiquidusType[t],LatentHeatType[t]);
        if(!isfinite(T)||T<0.0){log_printf("Invalid thermal restart enthalpy\n");exit(EXIT_FAILURE);}
    }
    #pragma acc update device(HeatCapacityType[0:TYPE_COUNT],ConductivityType[0:TYPE_COUNT],SolidusType[0:TYPE_COUNT],LiquidusType[0:TYPE_COUNT],LatentHeatType[0:TYPE_COUNT],CriticalSolidFraction[0:TYPE_COUNT],HeatSourceType[0:TYPE_COUNT],LiquidConductivityType[0:TYPE_COUNT])
    #pragma acc update device(ThermalFixedTemperature[0:TYPE_COUNT],ThermalCoolingType[0:TYPE_COUNT],ThermalViscosityExponent[0:TYPE_COUNT],ThermalViscosityJump[0:TYPE_COUNT],ThermalViscosityMaxMultiplier)
    #pragma acc update device(SpecificEnthalpy[0:ParticleCount])
    #pragma acc update device(ShearViscosity[0:TYPE_COUNT])
    calculateSolidFraction();calculateViscosity();
    log_printf("Thermal: enthalpy, explicit stable substeps, safety=%g, cooling=%d ambient=%g K h=%g eps=%g\n",ThermalSafety,ThermalCoolingEnabled,ThermalAmbient,ThermalCoolingH,ThermalEmissivity);
    log_printf("Thermal: heat source [W/kg] = %g %g %g %g\n",HeatSourceType[0],HeatSourceType[1],HeatSourceType[2],HeatSourceType[3]);
    log_printf("Thermal: liquid conductivity (<0 off) = %g %g %g %g\n",LiquidConductivityType[0],LiquidConductivityType[1],LiquidConductivityType[2],LiquidConductivityType[3]);
    log_printf("Thermal: freeze solid (fs >= CriticalSolidFraction -> stationary wall in the flow step) = %d\n",ThermalFreezeSolid);
}

static void thermalSortFields()
{
    if(!ThermalEnabled) return;
#ifdef _OPENACC
    #pragma acc parallel loop present(SpecificEnthalpy[0:ParticleCount],TmpDoubleScalar[0:ParticleCount],CellParticle[0:PowerParticleCount])
#else
    #pragma omp parallel for
#endif
    for(int i=0;i<ParticleCount;++i) TmpDoubleScalar[i]=SpecificEnthalpy[CellParticle[i]];
#ifdef _OPENACC
    #pragma acc parallel loop present(SpecificEnthalpy[0:ParticleCount],TmpDoubleScalar[0:ParticleCount])
#else
    #pragma omp parallel for
#endif
    for(int i=0;i<ParticleCount;++i) SpecificEnthalpy[i]=TmpDoubleScalar[i];
    calculateSolidFraction(); // Property has already been permuted; material fields are derived.
}

static void calculateEnergyConservation()
{
    // Assemble dh/dt and a conservative explicit stability bound; no in-place T update.
#ifdef _OPENACC
    #pragma acc parallel loop present(Temperature[0:ParticleCount],TemperatureOld[0:ParticleCount])
#else
    #pragma omp parallel for
#endif
    for(int i=0;i<ParticleCount;++i) TemperatureOld[i]=Temperature[i];
#ifdef _OPENACC
    #pragma acc parallel loop present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],TemperatureOld[0:ParticleCount],Cp[0:ParticleCount],Conductivity[0:ParticleCount],ThermalRhs[0:ParticleCount],ThermalRow[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount],Density[0:TYPE_COUNT],ThermalFixedTemperature[0:TYPE_COUNT],HeatSourceType[0:TYPE_COUNT])
#else
    #pragma omp parallel for
#endif
    for(int i=0;i<ParticleCount;++i){
        double rhs=0.0,row=0.0;
        const int ti=Property[i];
        if(!ThermalFixedTemperature[ti]){
            #pragma acc loop seq
            for(int jn=0;jn<NeighborCount[i];++jn){
                const int j=NeighborInd[NeighborPtr[i]+jn];if(i==j) continue;
                // Existing solver is a CLOSED tank: do not wrap periodically.
                double r2=0.0;
                #pragma acc loop seq
                for(int d=0;d<DIM;++d){const double dx=Position[j][d]-Position[i][d];r2+=dx*dx;}
                if(r2<=1e-30 || r2>=RadiusV*RadiusV) continue;
                const double ki=Conductivity[i],kj=Conductivity[j];
                const double hi=fmax(ki,kj),lo=fmin(ki,kj);
                const double kij=(hi>0.0 ? 2.0*lo/(1.0+lo/hi):0.0);
                const double r=sqrt(r2);
                // Swv makes sum_j W_ij ~ 1 (W already carries dp^DIM), so no extra ParticleVolume
                // here: same scaling as implicitViscPair()/Mass. Per unit mass -> /rho.
                const double a=2.0*kij*(-dwvdr(r,RadiusV))/r/Density[ti];
                rhs+=a*(TemperatureOld[j]-TemperatureOld[i]);row+=a/Cp[i];
            }
        }
        // Heat source does not affect the explicit stability bound (row).
        if(!ThermalFixedTemperature[ti]) rhs+=HeatSourceType[ti];
        ThermalRhs[i]=rhs;ThermalRow[i]=row;
    }
}

static void calculateRadiation()
{
    if(!ThermalCoolingEnabled) return;
    const double ambient=ThermalAmbient,h=ThermalCoolingH,em=ThermalEmissivity;
    const double areaFactor=ThermalAreaFactor,baseArea=ParticleVolume/ParticleSpacing;
#ifdef _OPENACC
    #pragma acc parallel loop present(Property[0:ParticleCount],TemperatureOld[0:ParticleCount],Cp[0:ParticleCount],Mass[0:ParticleCount],VolStrainP[0:ParticleCount],ThermalRhs[0:ParticleCount],ThermalRow[0:ParticleCount],ThermalCoolingType[0:TYPE_COUNT],ThermalFixedTemperature[0:TYPE_COUNT])
#else
    #pragma omp parallel for
#endif
    for(int i=0;i<ParticleCount;++i){
        const int t=Property[i];
        if(!ThermalCoolingType[t] || ThermalFixedTemperature[t]) continue;
        // Empirical exposed-area estimate requested by the user; not geometric area.
        const double ratio=sqrt(fmin(1.0,fmax(0.0,-VolStrainP[i]/N0p)));
        const double area=areaFactor*baseArea*ratio;
        const double T=TemperatureOld[i];
        // Factorization avoids subtracting nearly equal fourth powers.
        const double heff=h+em*ThermalSBC*(T+ambient)*(T*T+ambient*ambient);
        ThermalRhs[i]-=area*heff*(T-ambient)/Mass[i];
        ThermalRow[i]+=area*heff/(Mass[i]*Cp[i]);
    }
}

static void advanceThermalOnCurrentNeighbors()
{
    if(!ThermalEnabled) return;
    double elapsed=0.0;int steps=0;
    while(elapsed<Dt){
        if(++steps>ThermalMaxSubsteps){log_printf("Thermal substeps exceeded limit at Time=%e\n",Time);exit(EXIT_FAILURE);}
        calculateEnergyConservation();calculateRadiation();
        double maxrow=0.0;int bad=0;
#ifdef _OPENACC
        #pragma acc parallel loop reduction(max:maxrow) reduction(+:bad) present(ThermalRow[0:ParticleCount],ThermalRhs[0:ParticleCount])
#else
        #pragma omp parallel for reduction(max:maxrow) reduction(+:bad)
#endif
        for(int i=0;i<ParticleCount;++i){
            if(!isfinite(ThermalRow[i]) || !isfinite(ThermalRhs[i]) || ThermalRow[i]<0.0) ++bad;
            if(ThermalRow[i]>maxrow) maxrow=ThermalRow[i];
        }
        double step=Dt-elapsed;
        if(maxrow>0.0) step=fmin(step,ThermalSafety/maxrow);
        if(bad || !(step>0.0) || !isfinite(step) || elapsed+step==elapsed){log_printf("Invalid thermal rate/time step\n");exit(EXIT_FAILURE);}
#ifdef _OPENACC
        #pragma acc parallel loop present(SpecificEnthalpy[0:ParticleCount],ThermalRhs[0:ParticleCount])
#else
        #pragma omp parallel for
#endif
        for(int i=0;i<ParticleCount;++i) SpecificEnthalpy[i]+=step*ThermalRhs[i];
        calculateSolidFraction();
        bad=0;
#ifdef _OPENACC
        #pragma acc parallel loop reduction(+:bad) present(Temperature[0:ParticleCount],SpecificEnthalpy[0:ParticleCount])
#else
        #pragma omp parallel for reduction(+:bad)
#endif
        for(int i=0;i<ParticleCount;++i) if(!isfinite(Temperature[i])||Temperature[i]<0.0||!isfinite(SpecificEnthalpy[i])) ++bad;
        if(bad){log_printf("Nonphysical thermal state\n");exit(EXIT_FAILURE);}
        elapsed+=step;
    }
    calculateViscosity();
    if(LogVerbose) log_printf("Thermal: substeps=%d Dt=%e\n",steps,Dt);
}

static void advanceThermal()
{
    if(!ThermalEnabled) return;
    // Thermal fields are committed only after an accepted hydrodynamic step.
    freeNeighbor();calculateCellParticle();calculateNeighbor();
    calculatePhysicalCoefficients();calculateDensityP();
    advanceThermalOnCurrentNeighbors();
}

// ---- Gravity hold: g=0 until a chosen material starts to melt ----
// box.data:  GravityHoldUntilMelt <type> [fs_threshold=0.99] [ramp_time=0] [hold_dt=0] [hold_output=1]
// Gravity stays zero until min(SolidFraction) of that type drops below
// fs_threshold, then ramps linearly to the Gravity input over ramp_time.
// hold_dt>0: during the hold the fluid is at rest and only conduction is
// advanced, with time step hold_dt and output every hold_output s; at the
// onset the step returns to Dt and the flow solver starts.
static int    GravityHoldType=-1;            // -1: off
static double GravityHoldFs=0.99, GravityHoldRamp=0.0;
static double GravityHoldDt=0.0, GravityHoldOut=1.0, GravityHoldOutNext=0.0, GravityHoldDtFlow=0.0;
static double GravityInput[DIM]={0.0,0.0,0.0};
static int    GravityHoldState=0;            // 0 off, 1 holding, 2 ramping, 3 done
static double GravityHoldStart=0.0;

static int gravityHoldParseDataLine(const char *buf)
{
    int t; double fs=0.99, ramp=0.0, hdt=0.0, hout=1.0;
    const int n=sscanf(buf," GravityHoldUntilMelt %d %lf %lf %lf %lf",&t,&fs,&ramp,&hdt,&hout);
    if(n<1) return 0;
    if(t<FLUID_BEGIN || t>=FLUID_END || !(fs>0.0 && fs<=1.0) || ramp<0.0 || hdt<0.0 || !(hout>0.0)){
        log_printf("GravityHoldUntilMelt: need a fluid type, 0<fs<=1, ramp>=0, hold_dt>=0, hold_output>0\n"); exit(EXIT_FAILURE);
    }
    GravityHoldType=t; GravityHoldFs=fs; GravityHoldRamp=ramp; GravityHoldDt=hdt; GravityHoldOut=hout;
    return 1;
}

static void gravityHoldInitialize()
{
    if(GravityHoldType<0) return;
    if(!ThermalEnabled){ log_printf("GravityHoldUntilMelt ignored: thermal model disabled\n"); GravityHoldType=-1; return; }
    for(int d=0;d<DIM;++d){ GravityInput[d]=Gravity[d]; Gravity[d]=0.0; }
    GravityHoldState=1;
    log_printf("GravityHoldUntilMelt: g=0 until type %d has fs<%g (ramp %g s)\n",GravityHoldType,GravityHoldFs,GravityHoldRamp);
    if(GravityHoldDt>0.0){
        GravityHoldDtFlow=Dt; Dt=GravityHoldDt;      // device copy follows the global update after init
        initializeWall();
        GravityHoldOutNext=Time+GravityHoldOut;
        log_printf("GravityHoldUntilMelt: conduction only during the hold, Dt=%g, output every %g s\n",GravityHoldDt,GravityHoldOut);
    }
}

// Hold phase with hold_dt>0: the flow solver is skipped (fluid at rest, g=0).
static int gravityHoldThermalOnly(){ return GravityHoldState==1 && GravityHoldDt>0.0; }

static void gravityHoldUpdate()
{
    if(GravityHoldState==0 || GravityHoldState==3) return;
    if(GravityHoldState==1){
        const int ht=GravityHoldType; const double hfs=GravityHoldFs;
        int nMelt=0;
        #pragma acc parallel loop reduction(+:nMelt) present(Property[0:ParticleCount],SolidFraction[0:ParticleCount])
        #pragma omp parallel for reduction(+:nMelt)
        for(int i=0;i<ParticleCount;++i) if(Property[i]==ht && SolidFraction[i]<hfs) ++nMelt;
        if(nMelt==0){
            if(GravityHoldDt>0.0){          // sparse output while only conduction runs
                if(Time+1.0e-6*Dt>=GravityHoldOutNext){ OutputNext=VtkOutputNext=Time; GravityHoldOutNext+=GravityHoldOut; }
                else{ OutputNext=VtkOutputNext=GravityHoldOutNext; }
            }
            return;
        }
        GravityHoldState=2; GravityHoldStart=Time;
        log_printf("GravityHoldUntilMelt: melting onset at t=%e (%d particles of type %d), gravity on\n",Time,nMelt,ht);
        if(GravityHoldDt>0.0){
            Dt=GravityHoldDtFlow;
            #pragma acc update device(Dt)
            initializeWall();
            OutputNext=VtkOutputNext=Time;   // write the onset state, then the normal intervals
            log_printf("GravityHoldUntilMelt: flow solver on, Dt=%g\n",Dt);
        }
    }
    double f=(GravityHoldRamp>0.0 ? (Time-GravityHoldStart)/GravityHoldRamp : 1.0);
    if(f>=1.0){ f=1.0; GravityHoldState=3; }
    for(int d=0;d<DIM;++d) Gravity[d]=f*GravityInput[d];
    #pragma acc update device(Gravity[0:DIM])
}

static void thermalFree()
{
    if(!ThermalEnabled) return;
    #pragma acc exit data delete(Temperature[0:ParticleCount],TemperatureOld[0:ParticleCount],SpecificEnthalpy[0:ParticleCount],SolidFraction[0:ParticleCount],Cp[0:ParticleCount],Conductivity[0:ParticleCount],SolidusTemp[0:ParticleCount],LiquidusTemp[0:ParticleCount],LatentH[0:ParticleCount],ThermalRhs[0:ParticleCount],ThermalRow[0:ParticleCount])
    free(Temperature);free(TemperatureOld);free(SpecificEnthalpy);free(SolidFraction);free(Cp);free(Conductivity);free(SolidusTemp);free(LiquidusTemp);free(LatentH);free(ThermalRhs);free(ThermalRow);
}
// -------------------------- Thermal model end -------------------------------
#endif // TEMPERATURE


#include "deeponet/ppe_deeponet.inc"   // DeepONet two-level PPE preconditioner + training-data export
#ifdef USE_AMGX
#include "amgx/amgx_ppe.inc"       // AmgX alternative solvers (PpeAmgx / VisAmgx)
#endif
#include "spray/aerosol.inc"        // Property 5: Lagrangian aerosol + spray scavenging (one-way)
#include "body/solid_body.inc"     // resolved rigid particles (solid aerosol) with wettability
#ifdef RIGID
#include "body/rigid_body.inc"     // meltable rigid bodies per rigid property (MPH-PMS port)

#ifdef TEMPERATURE
// ThermalFreezeSolid: solidified fluid particles are stationary walls during the flow step.
// on=1 (start of the flow step): fluid rows with fs >= CriticalSolidFraction[Type] get the wall
//   Type WALL_BEGIN and u = 0, so the PPE (Neumann row, hydrostatic ghost), the implicit
//   viscosity (prescribed u = 0, eliminated from the unknowns) and the wall non-penetration
//   treat the crust as a wall. Removes the stiff (mu x 1e4-1e5) rows from the viscous solve.
// on=0 (before the thermal step): own Types back (the thermal step re-sorts).
// Returns 1 when a Type changed (cell sort / neighbour lists must be rebuilt).
static int thermalFreezeAsWall(const int on)
{
    if(!ThermalEnabled || !ThermalFreezeSolid || TwoPhaseActive) return 0;
    if(!FreezeOwnType){
        FreezeOwnType=(int*)malloc(ParticleCount*sizeof(int));
        for(int i=0;i<ParticleCount;++i) FreezeOwnType[i]=-1;
        #pragma acc enter data copyin(FreezeOwnType[0:ParticleCount])
    }
    if(!on && !FreezeWallOn) return 0;
    const int wt=WALL_BEGIN;
    int count=0;
    #pragma acc parallel loop reduction(+:count) present(FreezeOwnType[0:ParticleCount],ParticleIndex[0:ParticleCount],Property[0:ParticleCount],SolidFraction[0:ParticleCount],CriticalSolidFraction[0:TYPE_COUNT],Velocity[0:ParticleCount][0:DIM])
    #pragma omp parallel for reduction(+:count)
    for(int iP=0;iP<ParticleCount;++iP){
        const int id=ParticleIndex[iP];
        if(on){
            const int t=Property[iP];
            if(t<FLUID_BEGIN || t>=FLUID_END) continue;
            if(!(SolidFraction[iP]>=CriticalSolidFraction[t])) continue;
            FreezeOwnType[id]=t; Property[iP]=wt; ++count;
            #pragma acc loop seq
            for(int d=0;d<DIM;++d) Velocity[iP][d]=0.0;
        }else if(FreezeOwnType[id]>=0){
            Property[iP]=FreezeOwnType[id]; FreezeOwnType[id]=-1; ++count;
        }
    }
    if(on){
        FreezeWallOn=(count>0);
        if(count!=FreezeLastCount && (LogVerbose || count/50!=FreezeLastCount/50))
            log_printf("Thermal freeze: %d solidified particles act as wall (Time=%e)\n",count,Time);
        FreezeLastCount=count;
    }else FreezeWallOn=0;
    return count>0;
}
#endif
#endif
#include "body/surface_tension.inc" // single-phase surface tension: Akinci (2013) / Laplace pressure BC, 2D/3D
#include "spray/aerosol_capillary.inc" // point aerosol + capillary force of the resolved liquid surface
#ifdef SPH_FVM_AIR
#ifdef THREE_DIMENSIONAL
#include "fvm/fvm3d_air.inc"         // carrier air on the 3D FVM grid (own gas-only solver)
#include "spray/aerosol_field3d.inc"   // aerosol mass fractions on the 3D FVM grid (Liang et al. 2020)
#else
#include "fvm/fvm_air.inc"        // carrier air on the FVM grid
#include "spray/aerosol_field.inc"  // aerosol mass fractions on the FVM grid (Liang et al. 2020)
#endif
#include "spray/spray.inc"          // SPH spray (recycled water particles), drag coupling with the FVM air
#endif

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
#ifdef TEMPERATURE
    readThermalDataFile(datafilename); // BEFORE allocation and grid temperature columns
#endif
    readGridFile(gridfilename);
    donLogSettings();
    visDonLogSettings();
#ifdef USE_AMGX
    amgxLogSettings();
#endif
    {
        time_t t=time(NULL);
        log_printf("start initialization at %s\n",ctime(&t));
    }
    if(ViscosityMaxIteration<1 || !isfinite(ViscosityTolerance) ||
       !(ViscosityTolerance>0.0 && ViscosityTolerance<1.0) ||
       !isfinite(ViscosityAbsoluteTolerance) || ViscosityAbsoluteTolerance<0.0){
        log_printf("Invalid implicit-viscosity solver settings\n"); exit(EXIT_FAILURE);
    }
    for(int t=FLUID_BEGIN;t<FLUID_END;++t){
        if(!isfinite(Density[t]) || !(Density[t]>0.0) ||
           !isfinite(ShearViscosity[t]) || ShearViscosity[t]<0.0){
            log_printf("Implicit viscosity requires positive density and nonnegative viscosity\n");
            exit(EXIT_FAILURE);
        }
    }
    log_printf("Viscosity: backward Euler, coupled PCG, maxIter=%d relTol=%e absTol=%e\n",
               ViscosityMaxIteration,ViscosityTolerance,ViscosityAbsoluteTolerance);
    if(!isfinite(PpeDensityGamma) || PpeDensityGamma<0.0 || PpeDensityGamma>=1.0){
        log_printf("PpeDensityGamma must be in [0,1)\n"); exit(EXIT_FAILURE);
    }
    log_printf("PPE: single-phase PND drift correction gamma=%e (compression only, |dn|<=%e)\n",
               PpeDensityGamma,IsphDensityErrMax);
    if(PpeDuLimit<0.0){ log_printf("PpeDuLimit must be >= 0\n"); exit(EXIT_FAILURE); }
    if(PpeDuLimit>0.0) log_printf("PPE: single-phase projection limiter |v_i|<=%g*max_N|v*_j|+2|g|Dt\n",PpeDuLimit);
    log_printf("PPE: position correction x += Dt (u^{n+1}-u*) after projection = %d\n",PpePositionCorrection);
    log_printf("PPE: 2D corrected gradient on surface/wall rows with hydrostatic wall ghosts = %d\n",PpeWallGCM);
    initializeWeight();
    initializeFluid();
    TwoPhaseDtMax=Dt;
    #pragma acc update device(TwoPhaseOpenSurface,TwoPhaseDensityRelaxTime)
    initializeWall();
    initializeDomain();
#ifdef TEMPERATURE
    initializeThermal();
    gravityHoldInitialize();
#endif
    aerosolInitialize();
    surfaceTensionInitialize();
    solidInitialize();
#ifdef RIGID
    rigidInitialize();
#endif
    capAerosolInitialize();
#ifdef SPH_FVM_AIR
    fvmInitialize();
    aerFieldInitialize();
    sprayInitialize();
#endif

	// data transfer from host to device
	#pragma acc update device(ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCount,ParticleIndex[0:ParticleCount],Property[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc update device(Density[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc update device(InterfaceTension01)
	#pragma acc update device(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc update device(Mu[0:ParticleCount],Gravity[0:DIM])
	#pragma acc update device(WallGhostHydrostatic)
	#pragma acc update device(PpeMaxIteration,PpeTolerance,PpeDensityGamma,PpeWallGCM)
	#pragma acc update device(PpeSurfLambdaMin,PpeSurfLambdaKeep,PpeSurfPndMin,PpeSurfProbeRatio)
	#pragma acc update device(TwoPhaseSourceGamma,TwoPhaseRhoSmooth,TwoPhaseShiftCoef,TwoPhaseIfThreshold,PpePureNeumann)
	#pragma acc update device(TwoPhaseIfShiftNormal,TwoPhaseShiftVelCorr,TwoPhaseDuCap,TwoPhaseMinFluidNb)
	#pragma acc update device(TwoPhaseStModel,TwoPhaseStSmoothRatio,TwoPhaseStDoubleSmooth)
	#pragma acc update device(TwoPhaseStDensityScale,TwoPhaseStZeroNet,TwoPhaseStDeltaMin)
	#pragma acc update device(TwoPhaseCurvMax,PpeMinPressure,WallNoSeparation,WallContactRatio,TwoPhasePpRadiusRatio,WallNoSlip,WallHardNonPenetration,WallViscTangential,WallViscMorris,WallMinGapRatio,WallImpactBufferRatio,WallImpactCfl,WallPressureFullRatio,WallPressureRangeRatio,WallViscFullRatio,WallViscRangeRatio)
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
		aerosolInitialState();
		
		{
			time_t t=time(NULL);
			log_printf("start main roop at %s\n",ctime(&t));
		}
		int iStep=(int)(Time/Dt);
		cStart = clock();
		cFrom = cStart;
		#ifdef TEMPERATURE
        while((TwoPhaseActive || ThermalEnabled) ? Time < EndTime-1.0e-12*Dt : Time < EndTime + 1.0e-5*Dt){
#else
        while(TwoPhaseActive ? Time < EndTime-1.0e-12*TwoPhaseDtMax : Time < EndTime + 1.0e-5*Dt){
#endif
			
			if( Time + 1.0e-5*Dt >= OutputNext ){
				char filename[256];
				sprintf(filename,proffilename,iStep);
				writeProfFile(filename);
				if(Time>0.0) aerosolWriteHistory();
				solidWriteHistory();
#ifdef RIGID
				rigidWriteHistory();
#endif
				capAerosolWriteHistory();
#ifdef SPH_FVM_AIR
				if(Time>0.0) fvmLogStatus();
				if(Time>0.0) aerFieldWriteHistory();
#endif
				if(LogVerbose) log_printf("@ Prof Output Time : %e\n", Time );
				OutputNext += OutputInterval;
			}
			cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;
		
#ifdef TEMPERATURE
            if(ThermalEnabled && !TwoPhaseActive && Dt>EndTime-Time){
                Dt=EndTime-Time;
                #pragma acc update device(Dt)
                initializeWall();
            }
#endif
            if(TwoPhaseActive){
                advanceTwoPhase();
                Time += Dt; // outputs below describe the accepted new state
                logTwoPhaseStabilityDiagnostics();
            }
#ifdef TEMPERATURE
            else if(gravityHoldThermalOnly()){
                // GravityHoldUntilMelt hold: fluid at rest under g=0, conduction only (below)
            }
#endif
            else{
#ifdef RIGID
			// fixed rigid bodies act as walls in the flow step (Type swap, re-sort)
			if(rigidFixedAsWall(1)){
				freeNeighbor();
				calculateCellParticle();
				calculateNeighbor();
			}
#endif
#ifdef TEMPERATURE
			// solidified particles (ThermalFreezeSolid) are stationary walls in the flow step
			if(thermalFreezeAsWall(1)){
				freeNeighbor();
				calculateCellParticle();
				calculateNeighbor();
			}
#endif
			// The predictor must start from a valid Verlet list.  Rebuild here
			// rather than inside the RK stages, because cell sorting changes
			// particle order.
			if(neighborCalculation()==1){
				freeNeighbor();
				calculateCellParticle();
				calculateNeighbor();
			}

			// IMEX predictor: explicit non-viscous force + backward Euler viscosity
			calculatePhysicalCoefficients();
			calculatePredictor();
#ifdef RIGID
			rigidHoldFixed();          // anchored bodies stay put in the PPE
#endif

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
			ppeSaveUStar();
			calculatePpe();
			ppeCorrectPosition();

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
                // Shifting is numerical particle motion; never allow it to put a
                // fluid particle inside the wall layer.
                (void)enforceWallNonPenetration(1,1);
                solidRigidProjection();   // solid aerosol: rigid motion of its particles
#ifdef RIGID
                rigidProjection();        // meltable rigid bodies: rigid motion per rigid property
#endif
            }
			
            }

#ifdef RIGID
            rigidFixedAsWall(0);          // own Types again: the thermal step re-sorts
#endif
#ifdef TEMPERATURE
            thermalFreezeAsWall(0);       // own Types again: the thermal step re-sorts
#endif
#ifdef TEMPERATURE
            advanceThermal();
#endif
#ifdef RIGID
            rigidRelease();               // melted members continue as fluid
#endif
            if(!TwoPhaseActive) Time += Dt; // accepted fields and VTK timestamp agree
#ifdef TEMPERATURE
            gravityHoldUpdate();          // GravityHoldUntilMelt: switch g on at melting onset
#endif
#ifdef SPH_FVM_AIR
            sprayStep();     // spray drain/injection, drag coupling, FVM air to the new time
#endif
            aerosolStep();
            capAerosolStep();
			cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
			
			
			if( Time + 1.0e-5*Dt >= VtkOutputNext ){
				calculateVirialStressAtParticle();
				cTill = clock(); cVirial += (cTill-cFrom); cFrom = cTill;
				
				char filename [256];
				sprintf(filename,vtkfilename,iStep);
				writeVtkFile(filename);
				aerosolWriteVtk(filename);
#ifdef SPH_FVM_AIR
				fvmWriteVtk();
				aerFieldWriteVtk();
#endif
				if(LogVerbose) log_printf("@ Vtk Output Time : %e\n", Time );
				VtkOutputNext += VtkOutputInterval;
				cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;

			}
			

			iStep++;
			if(!LogVerbose) log_printf("step=%d time=%e Dt=%e ppe_iter=%d visc_iter=%d\n",
			                           iStep,Time,Dt,LogLastPpeIter,LogLastViscIter);
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
		donLogSummary();
		aerosolLogSummary();
#ifdef SPH_FVM_AIR
		sprayLogSummary();
		fvmLogStatus();
		aerFieldLogSummary();
#endif
		visDonLogSummary();
#ifdef USE_AMGX
		amgxLogSummary();
#endif
	}
	
	
#ifdef TEMPERATURE
    thermalFree();
#endif
	freeIsphHighOrderCoefficients();
#ifdef SPH_FVM_AIR
	fvmFinalize();
#endif

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
#ifndef TEMPERATURE
            char thermalKey[80]={0}; sscanf(buf,"%79s",thermalKey);
#endif
            if(buf[0]=='#'){}
            else if(sscanf(buf," Dt %lf",&Dt)==1){mode=reading_global;}
            else if(sscanf(buf," WallMotionEndTime %lf %lf",&WallMotionEndTime[2],&WallMotionEndTime[3])==2){mode=reading_global;}
            else if(sscanf(buf," OutputInterval %lf",&OutputInterval)==1){mode=reading_global;}
            else if(sscanf(buf," VtkOutputInterval %lf",&VtkOutputInterval)==1){mode=reading_global;}
            else if(sscanf(buf," EndTime %lf",&EndTime)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioA %lf",&RadiusRatioA)==1){mode=reading_global;}
        	// else if(sscanf(buf," RadiusRatioG %lf",&RadiusRatioG)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioP %lf",&RadiusRatioP)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioV %lf",&RadiusRatioV)==1){mode=reading_global;}
            else if(sscanf(buf," Density %lf %lf %lf %lf",&Density[0],&Density[1],&Density[2],&Density[3])==4){mode=reading_global;}
            else if(sscanf(buf," ShearViscosity %lf %lf %lf %lf",&ShearViscosity[0],&ShearViscosity[1],&ShearViscosity[2],&ShearViscosity[3])==4){mode=reading_global;}
            else if(sscanf(buf," ViscosityMaxIteration %d",&ViscosityMaxIteration)==1){mode=reading_global;}
            else if(sscanf(buf," ViscosityTolerance %lf",&ViscosityTolerance)==1){mode=reading_global;}
            else if(sscanf(buf," ViscosityAbsoluteTolerance %lf",&ViscosityAbsoluteTolerance)==1){mode=reading_global;}
            else if(sscanf(buf," PpeMaxIteration %d",&PpeMaxIteration)==1){mode=reading_global;}
            else if(sscanf(buf," PpeTolerance %lf",&PpeTolerance)==1){mode=reading_global;}
            else if(sscanf(buf," PpeDensityGamma %lf",&PpeDensityGamma)==1){mode=reading_global;}
            else if(sscanf(buf," PpeDuLimit %lf",&PpeDuLimit)==1){mode=reading_global;}
            else if(sscanf(buf," PpePositionCorrection %d",&PpePositionCorrection)==1){mode=reading_global;}
            else if(sscanf(buf," PpeWallGCM %d",&PpeWallGCM)==1){mode=reading_global;}
            else if(sscanf(buf," WallGhostHydrostatic %lf",&WallGhostHydrostatic)==1){mode=reading_global;}
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
            else if(sscanf(buf," WallHardNonPenetration %d",&WallHardNonPenetration)==1){mode=reading_global;}
            else if(sscanf(buf," WallViscTangential %d",&WallViscTangential)==1){mode=reading_global;}
            else if(sscanf(buf," WallViscMorris %d",&WallViscMorris)==1){mode=reading_global;}
            else if(sscanf(buf," WallMinGapRatio %lf",&WallMinGapRatio)==1){mode=reading_global;}
            else if(sscanf(buf," WallImpactBufferRatio %lf",&WallImpactBufferRatio)==1){mode=reading_global;}
            else if(sscanf(buf," WallImpactCfl %lf",&WallImpactCfl)==1){mode=reading_global;}
            else if(sscanf(buf," WallPressureFullRatio %lf",&WallPressureFullRatio)==1){mode=reading_global;}
            else if(sscanf(buf," WallPressureRangeRatio %lf",&WallPressureRangeRatio)==1){mode=reading_global;}
            else if(sscanf(buf," WallViscFullRatio %lf",&WallViscFullRatio)==1){mode=reading_global;}
            else if(sscanf(buf," WallViscRangeRatio %lf",&WallViscRangeRatio)==1){mode=reading_global;}
            else if(sscanf(buf," WallNoSeparation %d",&WallNoSeparation)==1){mode=reading_global;}
            else if(sscanf(buf," WallContactRatio %lf",&WallContactRatio)==1){mode=reading_global;}
            else if(sscanf(buf," PpeMinPressure %lf",&PpeMinPressure)==1){mode=reading_global;}
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
            else if(sscanf(buf," LogVerbose %d",&LogVerbose)==1){mode=reading_global;}
#ifdef TEMPERATURE
            else if(thermalParseMainLine(buf)){mode=reading_global;}
#else
            else if(!strncmp(thermalKey,"Thermal",7) || !strcmp(thermalKey,"InitialTemperature") || !strcmp(thermalKey,"HeatCapacity") || !strcmp(thermalKey,"Conductivity") || !strcmp(thermalKey,"SolidusTemperature") || !strcmp(thermalKey,"LiquidusTemperature") || !strcmp(thermalKey,"LatentHeat") || !strcmp(thermalKey,"CriticalSolidFraction")){mode=reading_global;}
#endif
            else if(aerosolParseDataLine(buf)){mode=reading_global;}
            else if(surfaceTensionParseDataLine(buf)){mode=reading_global;}
            else if(solidParseDataLine(buf)){mode=reading_global;}
#ifdef TEMPERATURE
            else if(gravityHoldParseDataLine(buf)){mode=reading_global;}
#endif
#ifdef RIGID
            else if(rigidParseDataLine(buf)){mode=reading_global;}
#endif
            else if(capAerosolParseDataLine(buf)){mode=reading_global;}
#ifdef SPH_FVM_AIR
            else if(sprayParseDataLine(buf)){mode=reading_global;}
            else if(fvmParseDataLine(buf)){mode=reading_global;}
            else if(aerFieldParseDataLine(buf)){mode=reading_global;}
#endif
            else if(donParseDataLine(buf)){mode=reading_global;}
            else if(visDonParseDataLine(buf)){mode=reading_global;}
#ifdef USE_AMGX
            else if(amgxParseDataLine(buf)){mode=reading_global;}
#endif
            else if(buf[strspn(buf," \t\r\n")]!='\0'){
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
	#pragma acc enter data create(TwoPhaseCurvMax,PpeMinPressure,WallNoSeparation,WallContactRatio,TwoPhasePpRadiusRatio,WallNoSlip,WallHardNonPenetration,WallViscTangential,WallViscMorris,WallMinGapRatio,WallImpactBufferRatio,WallImpactCfl,WallPressureFullRatio,WallPressureRangeRatio,WallViscFullRatio,WallViscRangeRatio)
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
        const int gridRowCount=ParticleCount;
        ParticleCount-=aerosolPrescanGrid(fp,gridRowCount);   // Property 5 rows are not SPH particles
        if(ParticleCount<=0){ log_printf("Grid has no SPH particles\n"); exit(EXIT_FAILURE); }
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
#ifdef TEMPERATURE
        thermalAllocate();
#endif
		
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
			{ int rowType; if(sscanf(buf,"%d",&rowType)==1 && rowType==AEROSOL_TYPE){ --iP; continue; } }
#ifdef TEMPERATURE
            double tempIn=NAN,enthalpyIn=NAN;
            const int fields=sscanf(buf,"%d %lf %lf %lf %lf %lf %lf %lf %lf",
                &Property[iP],&q[iP][0],&q[iP][1],&q[iP][2],&v[iP][0],&v[iP][1],&v[iP][2],&tempIn,&enthalpyIn);
#else
            const int fields=sscanf(buf,"%d %lf %lf %lf %lf %lf %lf",
                &Property[iP],&q[iP][0],&q[iP][1],&q[iP][2],&v[iP][0],&v[iP][1],&v[iP][2]);
#endif
            if(fields<7 || Property[iP]<0 || Property[iP]>=TYPE_COUNT){log_printf("Invalid particle row\n");exit(EXIT_FAILURE);}
#ifdef TEMPERATURE
            if(ThermalEnabled){
                if((fields>=8 && (!isfinite(tempIn)||tempIn<0.0)) || (fields>=9 && !isfinite(enthalpyIn))){log_printf("Invalid thermal grid columns\n");exit(EXIT_FAILURE);}
                Temperature[iP]=(fields>=8 ? tempIn:InitialTemperature[Property[iP]]);
                SpecificEnthalpy[iP]=(fields>=9 ? enthalpyIn:NAN);
            }
#endif
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
#ifdef TEMPERATURE
    if(ThermalEnabled){
        #pragma acc update self(Temperature[0:ParticleCount],SpecificEnthalpy[0:ParticleCount])
    }
#endif
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
            fprintf(fp,"%d %.17g %.17g %.17g %.17g %.17g %.17g",
                    Property[iP],q[iP][0],q[iP][1],q[iP][2],v[iP][0],v[iP][1],v[iP][2]);
#ifdef TEMPERATURE
            if(ThermalEnabled) fprintf(fp," %.17g %.17g",Temperature[iP],SpecificEnthalpy[iP]);
#endif
            fprintf(fp,"\n");
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
#ifdef TEMPERATURE
    if(ThermalEnabled){
        #pragma acc update self(Temperature[0:ParticleCount],SpecificEnthalpy[0:ParticleCount],SolidFraction[0:ParticleCount],Mu[0:ParticleCount])
        const char *names[4]={"Temperature_K","SpecificEnthalpy_J_per_kg","SolidFraction","DynamicViscosity"};
        const double *values[4]={Temperature,SpecificEnthalpy,SolidFraction,Mu};
        for(int field=0;field<4;++field){
            fprintf(fp,"SCALARS %s double 1\nLOOKUP_TABLE default\n",names[field]);
            for(int i=0;i<ParticleCount;++i) fprintf(fp,"%.17g\n",values[field][i]);
        }
    }
#endif
    fprintf(fp, "SCALARS label float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%d\n", Property[iP]);
    }
    fprintf(fp, "\n");
    // original grid row (follows a particle across the cell sort) and solid body
    #pragma acc update self(ParticleIndex[0:ParticleCount])
    fprintf(fp, "SCALARS id float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP) fprintf(fp, "%d\n", ParticleIndex[iP]);
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS body float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP) fprintf(fp, "%d\n", SolidBodyOfId[ParticleIndex[iP]]);
    fprintf(fp, "\n");
#ifdef RIGID
    rigidWriteVtkField(fp);
#endif
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
    if(Temperature){ // only allocated when thermal.data enables the thermal model
    fprintf(fp, "SCALARS Temperature float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
         fprintf(fp, "%e\n", (float)Temperature[iP]);
    }
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
        if(LogVerbose) log_printf("RadiusST/dp = %e, MrefST = %e\n",RadiusST/ParticleSpacing,MrefST);
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
        if(LogVerbose) log_printf("PP surface tension: re/dp=%e latticeSum=%e PpCoef=%e\n",
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
#ifdef RIGID
    // rigid bodies: Type 0 (structure) and Type 1 (melt) share the single-phase path
    if(rigidActive() && TwoPhaseActive){
        TwoPhaseActive=0;
        log_printf("Rigid: Type 0 + Type 1 as multi-material single-phase (no two-phase solver)\n");
    }
#endif
    TwoPhaseGauge=-1;
#ifndef TWO_DIMENSIONAL
    if(TwoPhaseActive){ log_printf("3D build: the two-phase solver (Type 0 + Type 1) is 2D only\n"); exit(EXIT_FAILURE); }
    if(SurfaceTensionModel==0)
        for(int iT=0;iT<TYPE_COUNT;++iT)
            if(SurfaceTension[iT]!=0.0){ log_printf("3D build: Kondo SurfaceTension of Type %d ignored (2D only; use SurfaceTensionModel 2)\n",iT); break; }
#endif
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
        if(LogVerbose) log_printf("TwoPhase surface tension: model=%d (0=CSF,1=CSS,2=PP) smooth=%e doubleSmooth=%d densityScale=%e zeroNet=%e deltaMin=%e curvMax=%e ppRe/dp=%e WallNoSlip=%d\n",
                   TwoPhaseStModel,TwoPhaseStSmoothRatio,TwoPhaseStDoubleSmooth,
                   TwoPhaseStDensityScale,TwoPhaseStZeroNet,TwoPhaseStDeltaMin,
                   TwoPhaseCurvMax,TwoPhasePpRadiusRatio,WallNoSlip);
        log_printf("Wall viscous coupling: tangential-only=%d Morris=%d\n",WallViscTangential,WallViscMorris);
        log_printf("Wall no-separation=%d contact r/dp<%e PpeMinPressure=%e\n",WallNoSeparation,WallContactRatio,PpeMinPressure);
        log_printf("Wall adaptive: hard=%d minGap/dp=%e impactBuffer/dp=%e impactCfl=%e\n",
                   WallHardNonPenetration,WallMinGapRatio,WallImpactBufferRatio,WallImpactCfl);
        log_printf("Wall adaptive: pressure full/range=%e/%e viscous full/range=%e/%e dp\n",
                   WallPressureFullRatio,WallPressureRangeRatio,WallViscFullRatio,WallViscRangeRatio);
        if(LogVerbose && !(InterfaceTension01>0.0)) log_printf("InterfaceTension01=0: capillary forces disabled.\n");
        if(LogVerbose) log_printf("Two-phase CSF model 0 uses pair-stencil forces; StDensityScale and StZeroNet are legacy options.\n");
        if(!(Dt>0.0) || !isfinite(Dt) || !(ParticleSpacing>0.0) ||
           !(Density[0]>0.0) || !(Density[1]>0.0) || ShearViscosity[0]<0.0 || ShearViscosity[1]<0.0 ||
           !(PpeTolerance>0.0) || PpeMaxIteration<1 ||
           !(WallMinGapRatio>0.0 && WallMinGapRatio<1.5) ||
           !(WallImpactBufferRatio>=0.0 && WallImpactBufferRatio<1.0) ||
           !(WallImpactCfl>0.0 && WallImpactCfl<=1.0) ||
           !(WallPressureFullRatio>WallMinGapRatio && WallPressureRangeRatio>WallPressureFullRatio) ||
           !(WallViscFullRatio>WallMinGapRatio && WallViscRangeRatio>WallViscFullRatio) ||
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

        // Viscous diffusion is implicit. Gravity and surface tension remain explicit.
        const double gmag=sqrt(Gravity[0]*Gravity[0]+Gravity[1]*Gravity[1]+Gravity[2]*Gravity[2]);
        const double dtGrav=(gmag>0.0 ? 0.25*sqrt(ParticleSpacing/gmag) : 1.0e100);
const double rhoStCap=(TwoPhaseStDensityScale>0.0 ? 0.5*(Density[0]+Density[1]) : rhoMin);
        const double dtCap=(InterfaceTension01>0.0
            ? 0.5*sqrt(rhoStCap*ParticleSpacing*ParticleSpacing*ParticleSpacing/(2.0*pi*InterfaceTension01))
            : 1.0e100);
        log_printf("TwoPhase dt limits: gravity=%e capillary=%e (viscosity implicit)  (current Dt=%e)\n",
                   dtGrav,dtCap,Dt);
        if(Dt>dtCap || Dt>dtGrav){
            log_printf("*** WARNING: Dt exceeds an explicit stability limit above. ***\n");
            log_printf("*** Reduce Dt; explicit surface tension and particle motion still require a small Dt. ***\n");
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
    if(LogVerbose) log_printf("KondoST: re/dp=%e shapeSum/dp^3=%e C=[%e %e %e %e]\n",
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
#ifdef TEMPERATURE
    thermalSortFields();
#endif
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
if(TwoPhaseActive)              calculateTwoPhaseSurfaceTension();
    else if(SurfaceTensionModel==1) calculateAkinciSurfaceTension();
    else if(SurfaceTensionModel==0) calculateKondoPotentialSurfaceTension();   // 2: in the PPE
    if(!TwoPhaseActive) calculateSolidAdhesion();                              // solid aerosol wettability

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

#pragma acc routine seq
static double wallSmoothActivation(const double r, const double fullRatio, const double rangeRatio)
{
    const double q=r/(ParticleSpacing+1.0e-300);
    if(q<=fullRatio) return 1.0;
    if(q>=rangeRatio) return 0.0;
    double t=(rangeRatio-q)/(rangeRatio-fullRatio+1.0e-300);
    if(t<0.0) t=0.0;
    if(t>1.0) t=1.0;
    return t*t*(3.0-2.0*t);
}

// Reduce Dt only for an approaching wall impact.  Tangential near-wall motion
// does not activate this limiter.  This prevents high-speed tunnelling without
// imposing a globally stronger wall force.
static double limitDtForWallImpact(double next)
{
    if(!WallHardNonPenetration || !(WallImpactCfl>0.0)) return next;
    const double hardGap=WallMinGapRatio*ParticleSpacing;
    double dtWall=1.0e100;
    #pragma acc parallel loop reduction(min:dtWall) present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount])
    #pragma omp parallel for reduction(min:dtWall)
    for(int iP=0;iP<ParticleCount;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;
        double wallSw=0.0, nx=0.0, ny=0.0, nz=0.0, uwx=0.0, uwy=0.0, uwz=0.0;
        double minR=1.0e100;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(!(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END)) continue;
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double r=sqrt(r2), w=wp(r,RadiusP);
            wallSw+=w; nx+=w*(-dx/r); ny+=w*(-dy/r); nz+=w*(-dz/r);
            uwx+=w*Velocity[jP][0]; uwy+=w*Velocity[jP][1]; uwz+=w*Velocity[jP][2];
            if(r<minR) minR=r;
        }
        const double nn=sqrt(nx*nx+ny*ny+nz*nz);
        if(!(wallSw>1.0e-30) || !(nn>1.0e-30) || !(minR>hardGap)) continue;
        { const double nd=fmax(nn,WallNormalCoherence*wallSw); nx/=nd; ny/=nd; nz/=nd; } uwx/=wallSw; uwy/=wallSw; uwz/=wallSw;
        const double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny+(Velocity[iP][2]-uwz)*nz;
        if(vn<0.0){
            const double clearance=minR-hardGap;
            const double cand=WallImpactCfl*clearance/(-vn+1.0e-300);
            if(cand>0.0 && cand<dtWall) dtWall=cand;
        }
    }
    if(dtWall<next) next=dtWall;
    return next;
}

static void selectTwoPhaseDt()
{
    calculatePhysicalCoefficients();
    resetForce();
    calculateTwoPhaseSurfaceTension();

    calculateGravity();
    double umax=0.0, amax=0.0;
    int bad=0;
    // Only non-viscous explicit acceleration limits the predictor.
    #pragma acc parallel loop reduction(max:umax,amax) reduction(+:bad) present(Velocity[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM],Mass[0:ParticleCount],Property[0:ParticleCount])
    #pragma omp parallel for reduction(max:umax,amax) reduction(+:bad)
    for(int i=0;i<ParticleCount;++i){
        const double u=sqrt(Velocity[i][0]*Velocity[i][0]+Velocity[i][1]*Velocity[i][1]+Velocity[i][2]*Velocity[i][2]);
        if(u>umax) umax=u;
        if(Property[i]<FLUID_BEGIN || Property[i]>=FLUID_END) continue;
        const double a=sqrt(Force[i][0]*Force[i][0]+Force[i][1]*Force[i][1]+Force[i][2]*Force[i][2])/Mass[i];
        if(!isfinite(a) || !isfinite(u)) ++bad;
        if(a>amax) amax=a;
    }
    if(bad){ log_printf("Nonfinite acceleration at accepted state\n"); exit(EXIT_FAILURE); }
    double next=fmin(TwoPhaseDtMax,1.2*Dt);
    if(TwoPhaseSourceGamma>0.0)
        next=fmin(next,0.1*TwoPhaseDensityRelaxTime/sqrt(TwoPhaseSourceGamma));
    if(umax>0.0) next=fmin(next,0.10*ParticleSpacing/umax);
    if(amax>0.0) next=fmin(next,sqrt(0.05*ParticleSpacing/amax));
    if(InterfaceTension01>0.0){
        const double rho=fmin(Density[0],Density[1]);
        next=fmin(next,0.25*sqrt(rho*pow(ParticleSpacing,3)/(2.0*M_PI*InterfaceTension01)));
    }
    next=limitDtForWallImpact(next);
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
    if(LogVerbose) log_printf("TwoPhase adaptive: time=%e Dt=%e Umax=%e Amax=%e viscosity=implicit\n",Time,Dt,umax,amax);
}

static void advanceTwoPhase()
{
    if(invalidTwoPhaseState()){ log_printf("Invalid initial/accepted state\n"); exit(EXIT_FAILURE); }
    // A fresh symmetric neighbor list is used by the implicit predictor.
    freeNeighbor(); calculateCellParticle(); calculateNeighbor();
    selectTwoPhaseDt();
    #pragma acc update self(WallCenter[0:WALL_END][0:DIM])
    double wall0[WALL_END][DIM];
    memcpy(wall0,WallCenter,sizeof(wall0));
    // Backups are indexed by immutable particle ID, not the sortable row.
    // RK3/4 buffers are reserved for accepted-state backups.
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
            // Hard geometric wall barrier before any density/interface reconstruction.
            // This keeps a predictor overshoot from entering the PPE geometry.
            if(enforceWallNonPenetration(1,0)>0){
                // Geometry-only wall repair before PPE. Do NOT clamp predictor vn here:
                // the PPE Neumann condition needs that wall-normal flux to build pressure.
                // Positions changed: rebuild once so density/interface operators see
                // the corrected geometry rather than the penetrated neighbour graph.
                freeNeighbor(); calculateCellParticle(); calculateNeighbor();
            }
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
            // Shifting may alter the wall clearance. Repair POSITION only here, while
            // retaining the predictor wall-normal velocity for the Neumann PPE flux.
            if(enforceWallNonPenetration(1,0)>0){
                freeNeighbor(); calculateCellParticle(); calculateNeighbor();
            }
            calculatePhysicalCoefficients();
            calculateDensityA(); calculateGravityCenter(); calculateDensityP();
            ppeSaveUStar();
            calculatePpe();
            ppeCorrectPosition();
            // Final safety constraint: remove only any residual inward wall-normal
            // velocity (and, with PpePositionCorrection, repair the wall clearance).
            (void)enforceWallNonPenetration(PpePositionCorrection ? 1 : 0,1);
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

// u* (velocity entering the projection) is kept in the otherwise unused RkU0.
static void ppeSaveUStar()
{
    if(!PpePositionCorrection) return;
    #pragma acc parallel loop present(Velocity[0:ParticleCount][0:DIM],RkU0[0:ParticleCount][0:DIM])
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP)
        for(int d=0;d<DIM;++d) RkU0[iP][d]=Velocity[iP][d];
}

// x += Dt (u^{n+1} - u*) for fluid rows only (walls and fixed rigid bodies, which
// carry a wall Type during the flow step, stay put).
static void ppeCorrectPosition()
{
    if(!PpePositionCorrection) return;
    #pragma acc parallel loop present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],RkU0[0:ParticleCount][0:DIM])
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const int t=Property[iP];
        if(t<FLUID_BEGIN || t>=FLUID_END) continue;
        for(int d=0;d<DIM;++d) Position[iP][d]+=Dt*(Velocity[iP][d]-RkU0[iP][d]);
    }
}

static void calculatePredictor()
{
    // First-order IMEX predictor at x^n:
    // (M + Dt*K) u* = M*u^n + Dt*F_nonvisc + Dt*wall_rhs.
    // The old RK2 viscosity stages MUST NOT be retained: they are explicit.
    evaluateExplicitAcceleration(RkK1U);
    if(TwoPhaseActive && TwoPhaseStepFailed) return;
    if(!solveImplicitViscosity()){
        if(TwoPhaseActive){ TwoPhaseStepFailed=1; return; }
        log_printf("Implicit viscosity failed at Time=%e; velocity/position not committed\n",Time);
        exit(EXIT_FAILURE);
    }
    // [V&V] Remove the separating wall-normal component of u* before it moves
    // the particles; otherwise x ratchets off the wall by u*.Dt every step.
    applyWallNoSeparation();
    // Advect with the viscosity-damped predictor, then perform the existing PPE.
    calculateConvection();
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
#ifdef TEMPERATURE
    calculateViscosity();
#endif
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
        IsphQmlsUse[iP]=(ISPH_USE_QMLS ? useQ : 0);
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
            sum += (dux*dx+duy*dy+duz*dz)*isphGradQ(r);
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

            sum+=(dux*dx+duy*dy+duz*dz)*isphGradQ(r);
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
                const double c=isphLapW(r)*beta;

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

        const int useHL=(ISPH_USE_HL && PpeAlpha[iP]>=0.999999999 && !hasWall);
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
                if(!(c>0.0)) c=isphLapW(r);
            }else{
                c=isphLapW(r);
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
    const double dimRef=2.0;
#else
    const double dimRef=3.0;
#endif
    const double alphaL=FreeSurfaceDivRLower, alphaU=FreeSurfaceDivRUpper;   // surface_tension.inc
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
                    diagTP+=isphLapW(r)*beta;
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
                double wallSw=0.0, wallNx=0.0, wallNy=0.0, wallNz=0.0;
                double wallUx=0.0, wallUy=0.0, wallUz=0.0;
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
                        wallNz+=w*(-dz/r);
                        wallUx+=w*Velocity[jP][0];
                        wallUy+=w*Velocity[jP][1];
                        wallUz+=w*Velocity[jP][2];
                        if(r<minWallR) minWallR=r;
                    }
                }

                const double wallNmag=sqrt(wallNx*wallNx+wallNy*wallNy+wallNz*wallNz);
                if(wallSw>1.0e-30 && wallNmag>1.0e-30 && minWallR<1.0e99){
                    double wMiss=N0p-fluidSw;
                    if(wMiss<0.0) wMiss=0.0;
                    if(wMiss>wallSw) wMiss=wallSw;
                    if(wMiss>1.0e-30){
                        const double nx=wallNx/fmax(wallNmag,WallNormalCoherence*wallSw);
                        const double ny=wallNy/fmax(wallNmag,WallNormalCoherence*wallSw);
                        const double nz=wallNz/fmax(wallNmag,WallNormalCoherence*wallSw);
                        const double uwx=wallUx/wallSw;
                        const double uwy=wallUy/wallSw;
                        const double uwz=wallUz/wallSw;
                        double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny+(Velocity[iP][2]-uwz)*nz;
                        // Local wall BC: full near actual contact, smoothly zero at
                        // WallPressureRangeRatio*dp. This avoids a 3.5dp-long wall halo.
                        const double wallAct=wallSmoothActivation(minWallR,WallPressureFullRatio,WallPressureRangeRatio);
                        // Signed wall flux is retained inside the active wall layer.
                        bTP-=wallAct*isphMissLap(minWallR,wMiss)*minWallR*vn/Dt;
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
                const int useHLdiag=(ISPH_USE_HL && PpeAlpha[iP]>=0.999999999 && !hasWallForLap);
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
                        if(!(c>0.0)) c=isphLapW(r);
                    }else{
                        c=isphLapW(r);
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
            if(PpeDensityGamma>0.0){
                // Compression only: the PND deficit at a free surface is not
                // a density error and must not pull particles inward.
                double dn=VolStrainP[iP]/(N0p+1.0e-30);
                if(dn<0.0) dn=0.0;
                if(dn>IsphDensityErrMax) dn=IsphDensityErrMax;
                b+=PpeDensityGamma*rho*dn/(Dt*Dt);
            }

            double fluidSw=0.0;
            double wallSw=0.0, wallNx=0.0, wallNy=0.0, wallNz=0.0;
            double wallUx=0.0, wallUy=0.0, wallUz=0.0;
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
                    wallNz+=w*(-dz/r);
                    wallUx+=w*Velocity[jP][0];
                    wallUy+=w*Velocity[jP][1];
                    wallUz+=w*Velocity[jP][2];
                    if(r<minWallR) minWallR=r;
                }
            }
            const double wallNmag=sqrt(wallNx*wallNx+wallNy*wallNy+wallNz*wallNz);
            if(wallSw>1.0e-30 && wallNmag>1.0e-30 && minWallR<1.0e99){
                double wMiss=N0p-fluidSw;
                if(wMiss<0.0) wMiss=0.0;
                if(wMiss>wallSw) wMiss=wallSw;
                if(wMiss>1.0e-30){
                    const double nx=wallNx/fmax(wallNmag,WallNormalCoherence*wallSw);
                    const double ny=wallNy/fmax(wallNmag,WallNormalCoherence*wallSw);
                    const double nz=wallNz/fmax(wallNmag,WallNormalCoherence*wallSw);
                    const double uwx=wallUx/wallSw;
                    const double uwy=wallUy/wallSw;
                    const double uwz=wallUz/wallSw;
                    double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny+(Velocity[iP][2]-uwz)*nz;
                    if(vn>0.0) vn=0.0;
                    const double gradPn=rho*vn/Dt;
                    const double cW=isphMissLap(minWallR,wMiss);
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

        if(LogVerbose) log_printf("TwoPhase PPE: pure Neumann, compatibility mean removed=%e active=%d\n",
                   meanBtp,nBtp);
    }
    if(!TwoPhaseActive) addSurfacePressureRhs();   // Laplace pressure BC (SurfaceTensionModel 2)

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

    // DeepONet two-level preconditioned BiCGStab (PpeDon 1). If it converges,
    // rr<=tol^2*bb and the Jacobi loop below is skipped; otherwise it leaves a
    // valid residual and the Jacobi loop finishes the solve.
    int donIter=0;
    donSolvePpe(bb,&rr,&donIter);
#ifdef USE_AMGX
    amgxSolvePpe(bb,&rr,&donIter);   // PpeAmgx 1
#endif

    // Jacobi-preconditioned BiCGStab.
    double rhoOld=1.0, alpha=1.0, omega=1.0;
    int iter=0, breakdown=0;
    const double tJacobi0=donWtime();
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

    donAccountJacobi(donWtime()-tJacobi0,iter);

    // Training snapshot for the DeepONet preconditioner (PpeDonExportEvery>0),
    // written before clipping so that p solves A p = b.
    donMaybeExport(bb,rr);

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
        // Laplace BC: free rows hold p_s and pressures may be negative.
        #pragma acc kernels present(PressureP[0:ParticleCount],PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],Property[0:ParticleCount],SurfacePressure[0:ParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP){
            if(LaplaceSurfaceBC){
                if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) PressureP[iP]=0.0;
                else if(PpeFlag[iP]==0 || !(PressureP[iP]==PressureP[iP])) PressureP[iP]=SurfacePressure[iP];
                continue;
            }
            if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END) || PpeFlag[iP]==0) PressureP[iP]=0.0;
            else{
                if(!(PressureP[iP]==PressureP[iP])) PressureP[iP]=0.0;
                if(PressureP[iP]<PpeMinPressure) PressureP[iP]=PpeMinPressure;
                PressureP[iP]*=PpeAlpha[iP];
            }
        }
    }

    const double rel=sqrt(rr/(bb>1.0e-60 ? bb : 1.0));
    LogLastPpeIter=iter+donIter;
    if(LogVerbose || iter>=PpeMaxIteration || breakdown) log_printf("ISPH(%s): iter=%d rel=%e q2=%d bulk=%d transition=%d free=%d%s%s\n",
               (TwoPhaseActive ? "TwoPhase-GraphPPE+FixedTimePenalty+StateGuard"
                               : "QMLS2-DGL+HL/GCMfallback+VirtualWall+KondoST+cPS-DPC,IMEX-BE,BiCGStab"),
               iter+donIter,rel,qmlsRows,fluidUnknown,transitionFluid,freeFluid,
               (iter>=PpeMaxIteration ? " (not converged)" : ""),
               (breakdown ? " (breakdown)" : ""));

    // ---------------------------------------------------------------------
    // Pressure gradient / velocity correction.
    // ---------------------------------------------------------------------
    // PpeDuLimit: per-row speed bound from the neighbourhood predictor speed, stored in
    // PpeOpY (free scratch after the solve) before velocities are overwritten.
    const int duLimitOn=(PpeDuLimit>0.0 && !TwoPhaseActive);
    if(duLimitOn){
        const double duFloor=2.0*sqrt(Gravity[0]*Gravity[0]+Gravity[1]*Gravity[1]+Gravity[2]*Gravity[2])*Dt;
        const double duK=PpeDuLimit;
        #pragma acc parallel loop present(PpeOpY[0:ParticleCount],Property[0:ParticleCount],Velocity[0:ParticleCount][0:DIM],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP){
            double vrMax=Velocity[iP][0]*Velocity[iP][0]+Velocity[iP][1]*Velocity[iP][1]+Velocity[iP][2]*Velocity[iP][2];
            if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END){
                #pragma acc loop seq
                for(int jN=0;jN<NeighborCount[iP];++jN){
                    const int jP=NeighborInd[NeighborPtr[iP]+jN];
                    if(iP==jP || Property[jP]<FLUID_BEGIN || Property[jP]>=WALL_END) continue;
                    const double dx=Position[jP][0]-Position[iP][0];
                    const double dy=Position[jP][1]-Position[iP][1];
                    const double dz=Position[jP][2]-Position[iP][2];
                    if(dx*dx+dy*dy+dz*dz>=RadiusP*RadiusP) continue;
                    const double u2=Velocity[jP][0]*Velocity[jP][0]+Velocity[jP][1]*Velocity[jP][1]+Velocity[jP][2]*Velocity[jP][2];
                    if(u2>vrMax) vrMax=u2;
                }
            }
            PpeOpY[iP]=duK*sqrt(vrMax)+duFloor;
        }
    }
    int pressureClipped=0, duLimited=0;
    #pragma acc parallel loop reduction(+:pressureClipped,duLimited) present(PpeOpY[0:ParticleCount],RkK4U[0:ParticleCount][0:DIM],IsphQmlsUse[0:ParticleCount],IsphGradX[0:IsphCoeffCount],IsphGradY[0:IsphCoeffCount],Force[0:ParticleCount][0:DIM],Mass[0:ParticleCount],Density[0:TYPE_COUNT],RhoEff[0:ParticleCount],Property[0:ParticleCount],PressureP[0:ParticleCount],PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],Velocity[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM])
    #pragma omp parallel for reduction(+:pressureClipped,duLimited)
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

        if(TwoPhaseActive){
            // Pressure-difference gradient with the PPE pair density.
            // Constant pressure gives zero acceleration. This collocated
            // gradient is not pair-antisymmetric and the PPE is not exactly D*G.
            const double rhoi=RhoEff[iP];
            double ax=0.0, ay=0.0, az=0.0;   // capped below

            double fluidSw=0.0;
            double wallSw=0.0, wallNx=0.0, wallNy=0.0, wallNz=0.0, wallUx=0.0, wallUy=0.0, wallUz=0.0;
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
                    const double q=gradCoef*beta*dpij*isphGradQ(r);

                    ax-=q*dx;
                    ay-=q*dy;
                    az-=q*dz;
                }else if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
                    wallSw+=w;
                    wallNx+=w*(-dx/r);
                    wallNy+=w*(-dy/r);
                    wallNz+=w*(-dz/r);
                    wallUx+=w*Velocity[jP][0];
                    wallUy+=w*Velocity[jP][1];
                    wallUz+=w*Velocity[jP][2];
                    if(r<minWallR) minWallR=r;
                }
            }

            const double wallNmag=sqrt(wallNx*wallNx+wallNy*wallNy+wallNz*wallNz);
            if(wallSw>1.0e-30 && wallNmag>1.0e-30 && minWallR<1.0e99){
                double wMiss=N0p-fluidSw;
                if(wMiss<0.0) wMiss=0.0;
                if(wMiss>wallSw) wMiss=wallSw;

                if(wMiss>1.0e-30){
                    const double nx=wallNx/fmax(wallNmag,WallNormalCoherence*wallSw);
                    const double ny=wallNy/fmax(wallNmag,WallNormalCoherence*wallSw);
                    const double nz=wallNz/fmax(wallNmag,WallNormalCoherence*wallSw);
                    const double uwx=wallUx/wallSw;
                    const double uwy=wallUy/wallSw;
                    const double uwz=wallUz/wallSw;
                    double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny+(Velocity[iP][2]-uwz)*nz;
                    // Enforce signed no-penetration correction, including separation.

                    const double rMin=IsphMinPairRatio*ParticleSpacing;
                    const double rEff=(minWallR>rMin ? minWallR : rMin);
                    const double dpGhost=-minWallR*rhoi*vn/Dt;
                    const double betaWall=1.0/(rhoi+1.0e-30);
                    const double wallAct=wallSmoothActivation(minWallR,WallPressureFullRatio,WallPressureRangeRatio);
                    const double q=wallAct*gradCoef*betaWall*dpGhost*isphMissGrad(minWallR,wMiss);

                    ax-=q*(-minWallR*nx);
                    ay-=q*(-minWallR*ny);
                    az-=q*(-minWallR*nz);
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
        double wallSw=0.0, wallNx=0.0, wallNy=0.0, wallNz=0.0, wallUx=0.0, wallUy=0.0, wallUz=0.0;
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
                const double pj=(PpeFlag[jP]!=0 || LaplaceSurfaceBC ? PressureP[jP] : 0.0);
                if(pj<pMin) pMin=pj;
            }else if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
                wallSw+=w;
                wallNx+=w*(-dx/r);
                wallNy+=w*(-dy/r);
                wallNz+=w*(-dz/r);
                wallUx+=w*Velocity[jP][0];
                wallUy+=w*Velocity[jP][1];
                wallUz+=w*Velocity[jP][2];
                if(r<minWallR) minWallR=r;
            }
        }
        if(pMin<PpeMinPressure && !LaplaceSurfaceBC) pMin=PpeMinPressure;
        // Laplace BC: one-sided band rows take p_i as the reference, otherwise
        // any neighbour below p_i pushes the surface outward (with p_s = 0 that
        // push is the physical free-surface force; with p_s = sigma kappa it is not)
        if(LaplaceSurfaceBC && PpeAlpha[iP]<0.999999999) pMin=PressureP[iP];

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
                const double pj=(PpeFlag[jP]!=0 || LaplaceSurfaceBC ? PressureP[jP] : 0.0);
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
            // PpeWallGCM: wall particles enter the corrected gradient as hydrostatic
            // ghosts p_w = p_i + f rho g.(x_w-x_i) (f as in the 3D per-particle ghost)
            double fHydW=0.0;
            if(PpeWallGCM){
                const double gMag=sqrt(Gravity[0]*Gravity[0]+Gravity[1]*Gravity[1]+Gravity[2]*Gravity[2]);
                fHydW=(gMag>0.0 ? PressureP[iP]/(Density[Property[iP]]*gMag*RadiusP) : 0.0);
                if(fHydW<0.0) fHydW=0.0;
                if(fHydW>1.0) fHydW=1.0;
                fHydW*=WallGhostHydrostatic;
            }
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const int jP=NeighborInd[NeighborPtr[iP]+jN];
                if(iP==jP) continue;
                const int isFluid=(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END);
                const int isWall=(PpeWallGCM && WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END);
                if(!isFluid && !isWall) continue;
                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double dz=Position[jP][2]-Position[iP][2];
                const double r2=dx*dx+dy*dy+dz*dz;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                const double r=sqrt(r2);
                const double w=wp(r,RadiusP);
                const double rMin=IsphMinPairRatio*ParticleSpacing;
                const double r2Eff=(r2>rMin*rMin ? r2 : rMin*rMin);
                const double q=isphGradQ(r);
                const double pj=(isWall ? PressureP[iP]+fHydW*Density[Property[iP]]*(Gravity[0]*dx+Gravity[1]*dy+Gravity[2]*dz)
                                        : (PpeFlag[jP]!=0 || LaplaceSurfaceBC ? PressureP[jP] : 0.0));

                Mxx+=q*dx*dx;
                Mxy+=q*dx*dy;
                Myy+=q*dy*dy;

                const double dpij=pj-pMin;
                Bx+=q*dpij*dx;
                By+=q*dpij*dy;
            }

#ifdef TWO_DIMENSIONAL
            const int interiorRow=(PpeAlpha[iP]>=0.999999999 && wallSw<=1.0e-30);
            if(interiorRow || (PpeWallGCM && fluidSw+wallSw>=0.4*N0p)){
                const double det=Mxx*Myy-Mxy*Mxy;
                const double scale=Mxx+Myy;
                const double detMin=(interiorRow ? 1.0e-12 : 0.05);
                if(det>detMin*(scale*scale+1.0e-30)){
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
                const double pj=(PpeFlag[jP]!=0 || LaplaceSurfaceBC ? PressureP[jP] : 0.0);
                const double q=(pj-pMin)*isphGradQ(r);
                gx+=q*dx; gy+=q*dy; gz+=q*dz;
            }
#if WALL_GHOST_PER_PARTICLE
            if(wallSw>1.0e-30){
                const double rhoI=Density[Property[iP]];
                const double gMag=sqrt(Gravity[0]*Gravity[0]+Gravity[1]*Gravity[1]+Gravity[2]*Gravity[2]);
                double fHyd=(gMag>0.0 ? PressureP[iP]/(rhoI*gMag*RadiusP) : 0.0);
                if(fHyd<0.0) fHyd=0.0;
                if(fHyd>1.0) fHyd=1.0;
                fHyd*=WallGhostHydrostatic;
                // Impact uses the approach speed along the wall normal only:
                // sliding along a wall is not an approach to the wall particles ahead.
                const double nd=fmax(sqrt(wallNx*wallNx+wallNy*wallNy+wallNz*wallNz),WallNormalCoherence*wallSw)+1.0e-300;
                const double enx=wallNx/nd, eny=wallNy/nd, enz=wallNz/nd;
                double vApp=-((Velocity[iP][0]-wallUx/wallSw)*enx+(Velocity[iP][1]-wallUy/wallSw)*eny+(Velocity[iP][2]-wallUz/wallSw)*enz);
                if(vApp<0.0) vApp=0.0;
                #pragma acc loop seq
                for(int jN=0;jN<NeighborCount[iP];++jN){
                    const int jP=NeighborInd[NeighborPtr[iP]+jN];
                    if(!(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END)) continue;
                    const double dx=Position[jP][0]-Position[iP][0];
                    const double dy=Position[jP][1]-Position[iP][1];
                    const double dz=Position[jP][2]-Position[iP][2];
                    const double r2=dx*dx+dy*dy+dz*dz;
                    if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                    const double r=sqrt(r2);
                    const double w=wp(r,RadiusP);
                    const double rMin=IsphMinPairRatio*ParticleSpacing;
                    const double r2Eff=(r2>rMin*rMin ? r2 : rMin*rMin);
                    double pw=PressureP[iP]+fHyd*rhoI*(Gravity[0]*dx+Gravity[1]*dy+Gravity[2]*dz);
                    // Neumann impact pressure, projected on this wall particle's direction
                    const double cs=-(dx*enx+dy*eny+dz*enz)/r;
                    if(vApp>0.0 && cs>0.0) pw+=wallSmoothActivation(r,WallPressureFullRatio,WallPressureRangeRatio)*rhoI*vApp*cs*r/Dt;
                    if(pw<pMin) pw=pMin;   // a wall never pulls fluid
                    const double q=(pw-pMin)*isphGradQ(r);
                    gx+=q*dx; gy+=q*dy; gz+=q*dz;
                }
            }
#endif
            gx*=gradCoef; gy*=gradCoef; gz*=gradCoef;
        }

        const double rho=Density[Property[iP]];
        const double wallNmag=sqrt(wallNx*wallNx+wallNy*wallNy+wallNz*wallNz);
        if(!WALL_GHOST_PER_PARTICLE && wallSw>1.0e-30 && wallNmag>1.0e-30 && minWallR<1.0e99){
            double wMiss=N0p-fluidSw;
            if(wMiss<0.0) wMiss=0.0;
            if(wMiss>wallSw) wMiss=wallSw;
            if(wMiss>1.0e-30){
                const double nx=wallNx/fmax(wallNmag,WallNormalCoherence*wallSw);
                const double ny=wallNy/fmax(wallNmag,WallNormalCoherence*wallSw);
                const double nz=wallNz/fmax(wallNmag,WallNormalCoherence*wallSw);
                const double uwx=wallUx/wallSw;
                const double uwy=wallUy/wallSw;
                const double uwz=wallUz/wallSw;
                double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny+(Velocity[iP][2]-uwz)*nz;
                if(vn>0.0) vn=0.0;
                const double gradPn=rho*vn/Dt;
                double pGhost=PressureP[iP]-minWallR*gradPn;
                if(pGhost<PpeMinPressure) pGhost=PpeMinPressure;
                const double rMin=IsphMinPairRatio*ParticleSpacing;
                const double rEff=(minWallR>rMin ? minWallR : rMin);
                // GCM rows already carry the full (renormalised) gradient: add only the impact part
                const double q=(pGhost-(usedGCM ? PressureP[iP] : pMin))*isphMissGrad(minWallR,wMiss);
                gx+=gradCoef*q*(-minWallR*nx);
                gy+=gradCoef*q*(-minWallR*ny);
                gz+=gradCoef*q*(-minWallR*nz);
            }
        }

        double ax=-gx/rho, ay=-gy/rho, az=-gz/rho;
        if(duLimitOn){
            const double nx=Velocity[iP][0]+Dt*ax, ny=Velocity[iP][1]+Dt*ay, nz=Velocity[iP][2]+Dt*az;
            const double vn=sqrt(nx*nx+ny*ny+nz*nz);
            if(vn>PpeOpY[iP]){
                const double sc=PpeOpY[iP]/vn;   // keep the direction, bound the speed
                ax=(sc*nx-Velocity[iP][0])/Dt; ay=(sc*ny-Velocity[iP][1])/Dt; az=(sc*nz-Velocity[iP][2])/Dt;
                ++duLimited;
            }
        }
        Velocity[iP][0]+=Dt*ax;
        Velocity[iP][1]+=Dt*ay;
        Velocity[iP][2]+=Dt*az;
        Force[iP][0]+=Mass[iP]*ax;
        Force[iP][1]+=Mass[iP]*ay;
        Force[iP][2]+=Mass[iP]*az;
    }

    // IMPORTANT: do not move particle positions with the pressure impulse here.
    // The PPE was assembled on the predictor geometry x*. Changing x immediately
    // after solving A(x*)p=b(x*,u*) makes the accepted geometry inconsistent with
    // the solved pressure field and can generate step-to-step pressure oscillations,
    // especially at large density ratio and near hard walls. The predictor convection
    // already advances x; pressure projection corrects velocity only.

    // Wall-only diagnostic after the pressure correction.
    double maxWallPen=0.0;
    int nWallNear=0;
    #pragma acc parallel loop reduction(max:maxWallPen) reduction(+:nWallNear) present(Property[0:ParticleCount],Velocity[0:ParticleCount][0:DIM],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for reduction(max:maxWallPen) reduction(+:nWallNear)
    for(int iP=0;iP<ParticleCount;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;
        double wallSw=0.0, wallNx=0.0, wallNy=0.0, wallNz=0.0, wallUx=0.0, wallUy=0.0, wallUz=0.0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(!(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END)) continue;
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double r=sqrt(r2);
            const double w=wp(r,RadiusP);
            wallSw+=w;
            wallNx+=w*(-dx/r);
            wallNy+=w*(-dy/r);
            wallNz+=w*(-dz/r);
            wallUx+=w*Velocity[jP][0];
            wallUy+=w*Velocity[jP][1];
            wallUz+=w*Velocity[jP][2];
        }
        const double wn=sqrt(wallNx*wallNx+wallNy*wallNy+wallNz*wallNz);
        if(wallSw>1.0e-30 && wn>1.0e-30){
            const double nx=wallNx/wn, ny=wallNy/wn, nz=wallNz/wn;
            const double uwx=wallUx/wallSw, uwy=wallUy/wallSw, uwz=wallUz/wallSw;
            const double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny+(Velocity[iP][2]-uwz)*nz;
            const double pen=(vn<0.0 ? -vn : 0.0);
            if(pen>maxWallPen) maxWallPen=pen;
            ++nWallNear;
        }
    }
    if(TwoPhaseActive && pressureClipped) TwoPhaseStepFailed=1;
    if(LogVerbose) log_printf("ISPH wall diagnostic: near=%d max_penetration_velocity=%e\n",nWallNear,maxWallPen);
    if(duLimited>0 && LogVerbose) log_printf("PpeDuLimit: %d rows limited\n",duLimited);

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
    if(LogVerbose) log_printf("ISPH divergence bulk: pre max=%e rms=%e | post max=%e rms=%e\n",
               maxPreDiv,rmsPreDiv,maxPostDiv,rmsPostDiv);
}


// Hard wall constraint, deliberately separated from no-slip viscosity and from
// particle shifting.  The pressure Neumann condition suppresses wall-normal
// velocity but does not geometrically forbid x_i from entering the wall layer.
//
// For every fluid particle with wall neighbours:
//   1) reconstruct a smooth outward wall normal from wall particles,
//   2) if the closest wall-particle distance is below WallMinGapRatio*dp, move the fluid
//      particle outward to that clearance,
//   3) remove only the inward wall-normal relative velocity.
//
// Tangential velocity is NOT overwritten here.  No-slip tangential traction is
// handled by the implicit-viscosity wall coupling.  Thus no-penetration and
// no-slip remain distinct boundary conditions.
static int enforceWallNonPenetration(int correctPosition, int correctVelocity)
{
    if(!WallHardNonPenetration) return 0;
    const double minGap=WallMinGapRatio*ParticleSpacing;
    if(!(minGap>0.0)) return 0;

    int nPos=0, nVel=0, nNear=0, nFallback=0;
    double maxCorr=0.0, maxClosing=0.0, minBefore=1.0e100;

    #pragma acc parallel loop reduction(+:nPos,nVel,nNear,nFallback) reduction(max:maxCorr,maxClosing) reduction(min:minBefore) present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount])
    #pragma omp parallel for reduction(+:nPos,nVel,nNear,nFallback) reduction(max:maxCorr,maxClosing) reduction(min:minBefore)
    for(int iP=0;iP<ParticleCount;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;

        double wallSw=0.0, wallNx=0.0, wallNy=0.0, wallNz=0.0;
        double wallUx=0.0, wallUy=0.0, wallUz=0.0;
        double minR=1.0e100;
        double nearDx=0.0, nearDy=0.0, nearDz=0.0;

        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(!(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END)) continue;

            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double r=sqrt(r2);
            const double w=wp(r,RadiusP);

            // -r_ij/r points from the wall particle toward the fluid particle.
            wallSw+=w;
            wallNx+=w*(-dx/r);
            wallNy+=w*(-dy/r);
            wallNz+=w*(-dz/r);
            wallUx+=w*Velocity[jP][0];
            wallUy+=w*Velocity[jP][1];
            wallUz+=w*Velocity[jP][2];

            if(r<minR){ minR=r; nearDx=dx; nearDy=dy; nearDz=dz; }
        }

        if(!(wallSw>1.0e-30) || !(minR<1.0e99)) continue;
        ++nNear;
        if(minR<minBefore) minBefore=minR;

        // Prefer the smooth wall normal.  At corners, in narrow gaps or inside a
        // locally symmetric wall cloud it cancels (low coherence); then fall back
        // to the nearest wall-particle radial direction so the hard barrier never
        // disappears.
        double nx=wallNx, ny=wallNy, nz=wallNz;
        double wn=sqrt(nx*nx+ny*ny+nz*nz);
        const double nearEx=-nearDx/(minR+1.0e-300);
        const double nearEy=-nearDy/(minR+1.0e-300);
        const double nearEz=-nearDz/(minR+1.0e-300);
        if(wn>WallNormalCoherence*wallSw && wn>1.0e-30){
            nx/=wn; ny/=wn; nz/=wn;
            // Keep the reconstructed normal on the same outward hemisphere as
            // the closest wall particle.
            if(nx*nearEx+ny*nearEy+nz*nearEz<0.0){ nx=-nx; ny=-ny; nz=-nz; }
        }else{
            nx=nearEx; ny=nearEy; nz=nearEz;
            ++nFallback;
        }

        // Geometric non-penetration.  The clearance is enforced exactly and
        // independently of particle shifting, so ShiftCoef=0 does not disable
        // the wall barrier.
        if(correctPosition && minR<minGap){
            const double corr=minGap-minR;
            Position[iP][0]+=corr*nx;
            Position[iP][1]+=corr*ny;
            Position[iP][2]+=corr*nz;
            if(corr>maxCorr) maxCorr=corr;
            ++nPos;
        }

        // Kinematic non-penetration relative to a translating/rotating wall.
        // Only the inward normal component is removed.  Tangential velocity is
        // left to the implicit no-slip viscous operator.
        const double uwx=wallUx/wallSw;
        const double uwy=wallUy/wallSw;
        const double uwz=wallUz/wallSw;
        const double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny+(Velocity[iP][2]-uwz)*nz;
        if(correctVelocity && vn<0.0){
            // Predictive collision constraint.  Do NOT zero every inward velocity:
            // only limit vn if the next step would enter the impact buffer.
            const double impactGap=(WallMinGapRatio+WallImpactBufferRatio)*ParticleSpacing;
            double vnLimit=(impactGap-minR)/(Dt+1.0e-300);
            if(vnLimit>0.0) vnLimit=0.0; // no artificial rebound when already close
            if(vn<vnLimit){
                const double dvn=vnLimit-vn;
                Velocity[iP][0]+=dvn*nx;
                Velocity[iP][1]+=dvn*ny;
                Velocity[iP][2]+=dvn*nz;
                if(dvn>maxClosing) maxClosing=dvn;
                ++nVel;
            }
        }
        // [V&V] No separation from a no-slip wall (WallNoSeparation 1). With
        // WallViscTangential the wall carries no normal viscous force and the PPE
        // wall flux is one-sided, so nothing else stops a wall-contact particle
        // from leaving the wall (shear-traction peel at a free-surface contact line).
        if(correctVelocity && WallNoSeparation && vn>0.0 && minR<WallContactRatio*ParticleSpacing){
            Velocity[iP][0]-=vn*nx;
            Velocity[iP][1]-=vn*ny;
            Velocity[iP][2]-=vn*nz;
            ++nVel;
        }
    }

    if(nPos>0 || nVel>0 || nFallback>0){
        log_printf("Hard wall projection(pos=%d,vel=%d): near=%d correctedPos=%d correctedVel=%d fallback=%d min_r/dp=%e max_dx/dp=%e max_closing=%e\n",
                   correctPosition,correctVelocity,nNear,nPos,nVel,nFallback,
                   (minBefore<1.0e99 ? minBefore/(ParticleSpacing+1.0e-300) : 1.0e100),
                   maxCorr/(ParticleSpacing+1.0e-300),maxClosing);
    }
    return nPos;
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
        double wallSw=0.0, wallNx=0.0, wallNy=0.0, wallNz=0.0;

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
                wallNz+=w*(-dz/r);
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
        const double wn=sqrt(wallNx*wallNx+wallNy*wallNy+wallNz*wallNz);
        if(wallSw>1.0e-30 && wn>1.0e-30){
            if(IfDelta[iP]*ParticleSpacing>TwoPhaseIfThreshold){ sx=sy=sz=0.0; }
            const double nx=wallNx/wn, ny=wallNy/wn, nz=wallNz/wn;
            const double sn=sx*nx+sy*ny+sz*nz;
            sx-=sn*nx;
            sy-=sn*ny;
            sz-=sn*nz;
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

    if(LogVerbose) log_printf("TwoPhase stability: min_r/dp=%e max|dn/n0|=%e Umax(type0,type1)=(%e,%e) |P|max(type0,type1)=(%e,%e) postDiv(max,rms)=(%e,%e)\n",
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

    #pragma acc kernels present(TmpDoubleVector[0:ParticleCount][0:DIM],RkK4U[0:ParticleCount][0:DIM],Property[0:ParticleCount],PositionDivR[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],PressureP[0:ParticleCount],SurfacePressure[0:ParticleCount],Density[0:TYPE_COUNT])
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
            double wallSw=0.0, wallNx=0.0, wallNy=0.0, wallNz=0.0;

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
                    wallNz+=w*(-dz/r);
                }
            }

            double sx=-AC*Dshift*gcx;
            double sy=-AC*Dshift*gcy;
            double sz=-AC*Dshift*gcz;

            const double wn=sqrt(wallNx*wallNx+wallNy*wallNy+wallNz*wallNz);
            if(wallSw>1.0e-30 && wn>1.0e-30){
                const double nx=wallNx/wn, ny=wallNy/wn, nz=wallNz/wn;
                const double sn=sx*nx+sy*ny+sz*nz;
                sx-=sn*nx;
                sy-=sn*ny;
                sz-=sn*nz;
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
                    // Laplace BC: dynamic part only (p_s = 0 for the other models)
                    double ptilde=lambdaDpc*fabs(PressureP[iP]-SurfacePressure[iP]+PressureP[jP]-SurfacePressure[jP]);
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

// Original central pair viscosity: C_ij = c_ij e_ij e_ij^T.
// Retaining this tensor is essential: Cartesian components are coupled.
// With reciprocal fluid neighbors and nonnegative mu, K is symmetric PSD;
// M + Dt*K is SPD even for unequal phase densities. Do not row-divide by M
// before CG. Prescribed walls are eliminated into the right-hand side.
#pragma acc routine seq
static double implicitViscPair(int i, int j, double e[DIM])
{
    if(i==j) return 0.0;
    const int wall=(WALL_BEGIN<=Property[j] && Property[j]<WALL_END);
    // [V&V patch] WallNoSlip is honoured in single-phase runs too. The original
    // line returned 0 for every wall pair unless TwoPhaseActive, so single-phase
    // walls were free-slip for the viscous term even with WallNoSlip 1.
    if(wall && !WallNoSlip) return 0.0;
    double r2=0.0;
    for(int d=0;d<DIM;++d){ e[d]=Position[j][d]-Position[i][d]; r2+=e[d]*e[d]; }
    if(r2<=1.0e-30 || r2>=RadiusV*RadiusV) return 0.0;
    const double r=sqrt(r2);
    for(int d=0;d<DIM;++d) e[d]/=r;
    // Stable harmonic mean without multiplying two potentially large viscosities.
    const double lo=fmin(Mu[i],Mu[j]), hi=fmax(Mu[i],Mu[j]);
    const double mu=wall ? Mu[i] : (hi>0.0 ? lo*(2.0/(1.0+lo/hi)) : 0.0);
    double wallAct=1.0;
    if(wall) wallAct=wallSmoothActivation(r,WallViscFullRatio,WallViscRangeRatio);
    if(!(wallAct>0.0)) return 0.0;
    return wallAct*8.0*mu*(-dwvdr(r,RadiusV))*ParticleVolume/
           fmax(r,IsphMinPairRatio*ParticleSpacing);
}

// [V&V] Tangential-only wall coupling. A wall pair contributes c (P e)(P e)^T
// with P = I - n n^T, n = unit wall normal at particle i (nw). The tangential
// no-slip shear of the central-pair model is unchanged, but the wall no longer
// carries a normal viscous force: at a no-slip wall 2 mu du_n/dn = 0 by
// continuity, and the normal velocity is the PPE's job. Without this the
// stationary wall supports a resting heavy fluid through viscosity instead of
// pressure (hydrostatic deficit within RadiusV of the wall, growing with mu).
#pragma acc routine seq
static void projectWallPair(const double *nw, double e[DIM])
{
    const double en=e[0]*nw[0]+e[1]*nw[1]+e[2]*nw[2];
    for(int d=0;d<DIM;++d) e[d]-=en*nw[d];
}

// [V&V] Morris, Fox & Zhu (1997) no-slip: the wall particle j carries the
// extrapolated velocity u_j = u_w - (d_j/d_i)(u_i - u_w), so the pair sees
// (u_i - u_w)(1 + d_j/d_i). d_i, d_j are the distances of i and j from the
// wall surface along the wall normal; the ratio is capped at 1.5 as in Morris.
// Wall particles carrying the plain rigid-body velocity under-estimate the
// wall shear by a factor ~2 (slip of O(dp) at moving walls).
#pragma acc routine seq
static double morrisWallFactor(int i, int j, const double *nw, const double *dw)
{
    if(!WallViscMorris) return 1.0;
    const double di=dw[i];
    if(!(di>0.0)) return 1.0;
    double dij=0.0;
    for(int d=0;d<DIM;++d) dij+=(Position[i][d]-Position[j][d])*nw[DIM*i+d];
    double dj=dij-di;
    if(dj<0.0) dj=0.0;
    double beta=dj/di;
    if(beta>1.5) beta=1.5;
    return 1.0+beta;
}

// [V&V] Velocity of the rigid wall at the surface point x_j + d_j n_i that the
// Morris extrapolation refers to. For a rotating wall the particle velocity
// u_j = V + Omega x (x_j - c) differs from the surface velocity by Omega x (d_j n);
// using u_j directly over-spins the fluid in a rotating drum and under-spins
// it next to a rotating inner cylinder.
#pragma acc routine seq
static void morrisWallVelocity(int i, int j, const double *nw, const double *dw, double uw[DIM])
{
    for(int d=0;d<DIM;++d) uw[d]=Velocity[j][d];
    if(!WallViscMorris || !(dw[i]>0.0)) return;
    double dij=0.0;
    for(int d=0;d<DIM;++d) dij+=(Position[i][d]-Position[j][d])*nw[DIM*i+d];
    double dj=dij-dw[i];
    if(dj<0.0) dj=0.0;
    double a[DIM];
    for(int d=0;d<DIM;++d) a[d]=dj*nw[DIM*i+d];
    const double *om=WallOmega[Property[j]];
    uw[0]+=om[1]*a[2]-om[2]*a[1];
    uw[1]+=om[2]*a[0]-om[0]*a[2];
    uw[2]+=om[0]*a[1]-om[1]*a[0];
}

static void implicitViscMatvec(const double *x, double *y, const double *nw, const double *dw)
{
    const int n=ParticleCount*DIM;
    #pragma acc parallel loop present(x[0:n],y[0:n],nw[0:n],dw[0:ParticleCount],Mass[0:ParticleCount],Property[0:ParticleCount],Mu[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount])
    #pragma omp parallel for
    for(int i=0;i<ParticleCount;++i){
        if(Property[i]<FLUID_BEGIN || Property[i]>=FLUID_END){
            for(int d=0;d<DIM;++d) y[DIM*i+d]=0.0;
            continue;
        }
        double v[DIM];
        for(int d=0;d<DIM;++d) v[d]=Mass[i]*x[DIM*i+d];
        #pragma acc loop seq
        for(int jn=0;jn<NeighborCount[i];++jn){
            const int j=NeighborInd[NeighborPtr[i]+jn];
            double e[DIM]={0.0,0.0,0.0};
            double c=Dt*implicitViscPair(i,j,e);
            if(c==0.0) continue;
            const int fluid=(Property[j]>=FLUID_BEGIN && Property[j]<FLUID_END);
            if(!fluid){
                if(WallViscTangential) projectWallPair(&nw[DIM*i],e);
                c*=morrisWallFactor(i,j,nw,dw);
            }
            double dot=0.0;
            for(int d=0;d<DIM;++d) dot+=e[d]*(x[DIM*i+d]-(fluid ? x[DIM*j+d] : 0.0));
            for(int d=0;d<DIM;++d) v[d]+=c*e[d]*dot;
        }
        for(int d=0;d<DIM;++d) y[DIM*i+d]=v[d];
    }
}

#include "deeponet/vis_deeponet.inc"   // DeepONet two-level viscosity preconditioner + training-data export
#ifdef USE_AMGX
#include "amgx/amgx_vis.inc"
#endif

static int solveImplicitViscosity()
{
    const int n=ParticleCount*DIM;
    // O(N) workspace; all Krylov vectors remain on the accelerator throughout
    // the solve. Only reduction scalars return to the host. The geometry is
    // frozen during CG, and the existing CSR neighbor list is not modified.
    double *x=(double*)malloc(sizeof(double)*n), *b=(double*)malloc(sizeof(double)*n);
    double *r=(double*)malloc(sizeof(double)*n), *z=(double*)malloc(sizeof(double)*n);
    double *p=(double*)malloc(sizeof(double)*n), *ap=(double*)malloc(sizeof(double)*n);
    double *diag=(double*)malloc(sizeof(double)*n);
    double *nw=(double*)malloc(sizeof(double)*n);
    double *dw=(double*)malloc(sizeof(double)*ParticleCount);
    if(!x || !b || !r || !z || !p || !ap || !diag || !nw || !dw){
        log_printf("Implicit viscosity workspace allocation failed\n"); exit(EXIT_FAILURE);
    }
    int bad=0, converged=0, iterations=0;
    double bnorm2=0.0, massSum=0.0, res2=0.0, rz=0.0;
    #pragma acc data create(x[0:n],b[0:n],r[0:n],z[0:n],p[0:n],ap[0:n],diag[0:n],nw[0:n],dw[0:ParticleCount])
    {
        // Unit wall normal (into the fluid) at each fluid particle: kernel-
        // weighted mean of the directions from the wall neighbours. Zero when
        // no wall particle is inside RadiusV.
        #pragma acc parallel loop present(nw[0:n],dw[0:ParticleCount],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount])
        #pragma omp parallel for
        for(int i=0;i<ParticleCount;++i){
            double s[DIM]={0.0,0.0,0.0};
            dw[i]=0.0;
            if((WallViscTangential || WallViscMorris) && Property[i]>=FLUID_BEGIN && Property[i]<FLUID_END){
                #pragma acc loop seq
                for(int jn=0;jn<NeighborCount[i];++jn){
                    const int j=NeighborInd[NeighborPtr[i]+jn];
                    if(Property[j]<WALL_BEGIN || Property[j]>=WALL_END) continue;
                    double d[DIM], r2=0.0;
                    for(int k=0;k<DIM;++k){ d[k]=Position[i][k]-Position[j][k]; r2+=d[k]*d[k]; }
                    if(r2<=1.0e-30 || r2>=RadiusV*RadiusV) continue;
                    const double r=sqrt(r2), w=wv(r,RadiusV)/r;
                    for(int k=0;k<DIM;++k) s[k]+=w*d[k];
                }
            }
            const double m=sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
            for(int k=0;k<DIM;++k) nw[DIM*i+k]=(m>1.0e-30 ? s[k]/m : 0.0);
            // distance of i from the wall surface: outermost wall layer + dp/2
            if(m>1.0e-30){
                double dmin=1.0e100;
                #pragma acc loop seq
                for(int jn=0;jn<NeighborCount[i];++jn){
                    const int j=NeighborInd[NeighborPtr[i]+jn];
                    if(Property[j]<WALL_BEGIN || Property[j]>=WALL_END) continue;
                    double dn=0.0, r2=0.0;
                    for(int k=0;k<DIM;++k){
                        const double dk=Position[i][k]-Position[j][k];
                        dn+=dk*nw[DIM*i+k]; r2+=dk*dk;
                    }
                    if(r2>=RadiusV*RadiusV) continue;
                    if(dn<dmin) dmin=dn;
                }
                double di=dmin-0.5*ParticleSpacing;
                if(di<0.1*ParticleSpacing) di=0.1*ParticleSpacing;
                dw[i]=(dmin<1.0e99 ? di : 0.0);
            }
        }
        #pragma acc parallel loop reduction(+:bad,bnorm2,massSum) present(Velocity[0:ParticleCount][0:DIM],RkK1U[0:ParticleCount][0:DIM],Mass[0:ParticleCount],Mu[0:ParticleCount],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount])
        #pragma omp parallel for reduction(+:bad,bnorm2,massSum)
        for(int i=0;i<ParticleCount;++i){
            for(int d=0;d<DIM;++d){ x[DIM*i+d]=0.0; b[DIM*i+d]=0.0; diag[DIM*i+d]=1.0; }
            if(Property[i]<FLUID_BEGIN || Property[i]>=FLUID_END) continue;
            if(!(Mass[i]>0.0) || !isfinite(Mass[i]) || Mu[i]<0.0 || !isfinite(Mu[i])){ ++bad; continue; }
            double rhs[DIM], dd[DIM];
            for(int d=0;d<DIM;++d){
                x[DIM*i+d]=Velocity[i][d];
                rhs[d]=Mass[i]*(Velocity[i][d]+Dt*RkK1U[i][d]);
                dd[d]=Mass[i];
            }
            #pragma acc loop seq
            for(int jn=0;jn<NeighborCount[i];++jn){
                const int j=NeighborInd[NeighborPtr[i]+jn];
                double e[DIM]={0.0,0.0,0.0};
                double c=Dt*implicitViscPair(i,j,e);
                if(!isfinite(c) || c<0.0){ ++bad; continue; }
                const int wall=(Property[j]>=WALL_BEGIN && Property[j]<WALL_END);
                if(wall && WallViscTangential) projectWallPair(&nw[DIM*i],e);
                if(wall) c*=morrisWallFactor(i,j,nw,dw);
                double uw=0.0;
                if(wall){
                    double us[DIM];
                    morrisWallVelocity(i,j,nw,dw,us);
                    for(int d=0;d<DIM;++d) uw+=e[d]*us[d];
                }
                for(int d=0;d<DIM;++d){ dd[d]+=c*e[d]*e[d]; if(wall) rhs[d]+=c*e[d]*uw; }
            }
            massSum+=Mass[i];
            for(int d=0;d<DIM;++d){
                b[DIM*i+d]=rhs[d]; diag[DIM*i+d]=dd[d];
                bnorm2+=rhs[d]*rhs[d]/Mass[i];
                if(!isfinite(rhs[d]) || !isfinite(dd[d]) || !(dd[d]>0.0)) ++bad;
            }
        }
        const double target2=ViscosityTolerance*ViscosityTolerance*bnorm2+
            ViscosityAbsoluteTolerance*ViscosityAbsoluteTolerance*massSum;
        // DeepONet two-level preconditioned CG (VisDon 1). If it converges the
        // Jacobi CG below is skipped; otherwise x is a valid start for it.
        if(!bad && isfinite(target2))
#ifdef USE_AMGX
            converged=(VisAmgx ? amgxSolveVis(x,b,nw,dw,target2,&iterations,&res2)
                               : visDonSolve(x,b,nw,dw,target2,&iterations,&res2));
#else
            converged=visDonSolve(x,b,nw,dw,target2,&iterations,&res2);
#endif
        const double tJacobi0=donWtime();
        int jacobiIter=0;
        if(!bad && isfinite(target2) && !converged){
            implicitViscMatvec(x,ap,nw,dw);
            #pragma acc parallel loop reduction(+:rz,res2) present(Mass[0:ParticleCount],Property[0:ParticleCount])
            #pragma omp parallel for reduction(+:rz,res2)
            for(int i=0;i<ParticleCount;++i){
                const int fluid=(Property[i]>=FLUID_BEGIN && Property[i]<FLUID_END);
                for(int d=0;d<DIM;++d){
                    const int k=DIM*i+d;
                    r[k]=b[k]-ap[k]; z[k]=r[k]/diag[k]; p[k]=z[k];
                    if(fluid){ rz+=r[k]*z[k]; res2+=r[k]*r[k]/Mass[i]; }
                }
            }
            converged=isfinite(res2) && res2<=target2;
            for(int it=0;it<ViscosityMaxIteration && !converged;++it){
                if(!isfinite(rz) || !(rz>0.0)) break;
                implicitViscMatvec(p,ap,nw,dw);
                double pap=0.0;
                #pragma acc parallel loop reduction(+:pap)
                #pragma omp parallel for reduction(+:pap)
                for(int k=0;k<n;++k) pap+=p[k]*ap[k];
                if(!isfinite(pap) || !(pap>0.0)) break;
                const double alpha=rz/pap;
                if(!isfinite(alpha)) break;
                res2=0.0;
                #pragma acc parallel loop reduction(+:res2) present(Mass[0:ParticleCount],Property[0:ParticleCount])
                #pragma omp parallel for reduction(+:res2)
                for(int i=0;i<ParticleCount;++i){
                    const int fluid=(Property[i]>=FLUID_BEGIN && Property[i]<FLUID_END);
                    for(int d=0;d<DIM;++d){
                        const int k=DIM*i+d;
                        x[k]+=alpha*p[k]; r[k]-=alpha*ap[k];
                        if(fluid) res2+=r[k]*r[k]/Mass[i];
                    }
                }
                iterations=it+1; jacobiIter=iterations;
                // Recompute the true residual before acceptance, periodically,
                // and at the iteration limit. Restart CG after replacement.
                const int refresh=(res2<=target2 || iterations%50==0 || iterations==ViscosityMaxIteration);
                if(refresh){
                    implicitViscMatvec(x,ap,nw,dw);
                    res2=0.0;
                    #pragma acc parallel loop reduction(+:res2) present(Mass[0:ParticleCount],Property[0:ParticleCount])
                    #pragma omp parallel for reduction(+:res2)
                    for(int i=0;i<ParticleCount;++i){
                        const int fluid=(Property[i]>=FLUID_BEGIN && Property[i]<FLUID_END);
                        for(int d=0;d<DIM;++d){
                            const int k=DIM*i+d; r[k]=b[k]-ap[k];
                            if(fluid) res2+=r[k]*r[k]/Mass[i];
                        }
                    }
                }
                if(!isfinite(res2)) break;
                converged=res2<=target2;
                if(converged) break;
                double rzNew=0.0;
                #pragma acc parallel loop reduction(+:rzNew)
                #pragma omp parallel for reduction(+:rzNew)
                for(int k=0;k<n;++k){ z[k]=r[k]/diag[k]; rzNew+=r[k]*z[k]; }
                const double beta=refresh ? 0.0 : rzNew/rz;
                #pragma acc parallel loop
                #pragma omp parallel for
                for(int k=0;k<n;++k) p[k]=z[k]+beta*p[k];
                rz=rzNew;
            }
        }
        visDonAccountJacobi(donWtime()-tJacobi0,jacobiIter);
        // Transactional commit: a failed solve leaves velocities untouched.
        if(converged){
            #pragma acc parallel loop reduction(+:bad)
            #pragma omp parallel for reduction(+:bad)
            for(int k=0;k<n;++k) if(!isfinite(x[k])) ++bad;
            if(bad) converged=0;
        }
        // Training snapshot (VisDonExportEvery>0), before Velocity is overwritten
        if(converged) visDonMaybeExport(x,b,nw,dw,iterations);
        if(converged){
            #pragma acc parallel loop present(Velocity[0:ParticleCount][0:DIM],Property[0:ParticleCount])
            #pragma omp parallel for
            for(int i=0;i<ParticleCount;++i){
                if(Property[i]>=FLUID_BEGIN && Property[i]<FLUID_END)
                    for(int d=0;d<DIM;++d) Velocity[i][d]=x[DIM*i+d];
            }
        }
    }
    LogLastViscIter=iterations;
    if(LogVerbose) log_printf("Implicit viscosity: iter=%d rel=%e rms_residual=%e status=%s\n",iterations,
        sqrt(res2/fmax(bnorm2,1.0e-300)),sqrt(res2/fmax(massSum,1.0e-300)),
        converged ? "OK" : "FAILED");
    free(x); free(b); free(r); free(z); free(p); free(ap); free(diag); free(nw); free(dw);
    return converged;
}


// [V&V] No-separation constraint on the current velocity: a wall-contact fluid
// particle (nearest wall particle closer than WallContactRatio*dp) loses any
// wall-normal velocity component pointing away from the wall. Inward vn is not
// touched (the PPE wall flux needs it).  Ported from the NP patch (2-D form).
static void applyWallNoSeparation()
{
    if(!WallNoSeparation) return;
    #pragma acc parallel loop present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount])
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;
        double wallSw=0.0, nx=0.0, ny=0.0, uwx=0.0, uwy=0.0, minR=1.0e100;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(!(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END)) continue;
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double r2=dx*dx+dy*dy;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double r=sqrt(r2), w=wp(r,RadiusP);
            wallSw+=w; nx+=w*(-dx/r); ny+=w*(-dy/r);
            uwx+=w*Velocity[jP][0]; uwy+=w*Velocity[jP][1];
            if(r<minR) minR=r;
        }
        const double nn=sqrt(nx*nx+ny*ny);
        if(!(wallSw>1.0e-30) || !(nn>1.0e-30) || !(minR<WallContactRatio*ParticleSpacing)) continue;
        nx/=nn; ny/=nn; uwx/=wallSw; uwy/=wallSw;
        const double vn=(Velocity[iP][0]-uwx)*nx+(Velocity[iP][1]-uwy)*ny;
        if(vn>0.0){ Velocity[iP][0]-=vn*nx; Velocity[iP][1]-=vn*ny; }
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
	// WallMotionEndTime <t2> <t3>: wall type 2/3 stops (velocity and omega set to 0) at this time
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		if(Time+0.5*Dt>=WallMotionEndTime[iProp] && (WallVelocity[iProp][0]!=0.0||WallVelocity[iProp][1]!=0.0||WallVelocity[iProp][2]!=0.0||WallOmega[iProp][0]!=0.0||WallOmega[iProp][1]!=0.0||WallOmega[iProp][2]!=0.0)){
			for(int d=0;d<DIM;++d){WallVelocity[iProp][d]=0.0;WallOmega[iProp][d]=0.0;}
			#pragma acc update device(WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM])
			log_printf("Wall type %d stopped at Time=%e\n",iProp,Time);
		}
	}
	
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
