// =============================================================================
// surface_tension.inc -- single-phase free-surface tension, 2D and 3D
//
// Included once by main_implicit.cpp (after log.h, before main()).
//
//   SurfaceTensionModel 0 : Kondo potential (main_implicit.cpp, default)
//   SurfaceTensionModel 1 : Akinci et al. (2013) cohesion + curvature force
//   SurfaceTensionModel 2 : Laplace pressure boundary condition (recommended)
//
// ---------------------------------------------------------------------------
// Model 2: Laplace pressure at the free surface
//   The Dirichlet value of the free-surface rows of the PPE is p_s = sigma kappa
//   instead of 0; no explicit force. The jump is carried by the implicit
//   projection, so the surface band (alpha < 1, no divergence constraint) is
//   not driven inward by an unopposed explicit force (models 0/1 compress the
//   band and break the drop at physical sigma: Oh_dp = mu/sqrt(rho sigma dp)
//   ~ 0.05 for water at dp = 5 um).
//     n_i   = -h sum_j V_j dW/dr e_ij        outward, unnormalised (P kernel)
//     valid : |n_i| > SurfaceNormalThreshold * |n_flat|
//     kappa = div(n^) by least squares over valid neighbours,
//             G = (sum w dn dx^T / r^2)(sum w dx dx^T / r^2)^-1,  kappa = tr G
//     smoothed SurfaceKappaSmoothing times (P-kernel average over valid
//     neighbours; the last pass also fills invalid particles) -> p_s = sigma kappa
//   PPE rows: diag p_i - a_i sum_flag c_ij p_j
//               = a_i b_i + a_i sum_free c_ij p_s,j + (1-a_i) diag_i p_s,i
//   The coupling sum is evaluated with applyPpeOperator itself, so it uses the
//   exact plain / HL / QMLS coefficients of the matrix. Pressures may be
//   negative (concave necks); the free-surface rows hold p_s afterwards.
//   The DPC collision term of the shifting uses the dynamic part p - p_s
//   (with p_s it would inflate the free-surface band).
//   The pressure gradient of the band rows (alpha < 1) uses p_i, not the
//   neighbour minimum, as reference: uniform p gives no force on a one-sided row.
//   Bodies further apart than the kernel radius do not interact (no air: there
//   is no long-range capillary attraction); within the kernel the facing
//   surfaces see a concave gap, which gives a short-range pull (~ contact).
//
// ---------------------------------------------------------------------------
// Model 1: Akinci et al. (2013)
//   F_i = sum_j K_ij [ -g_coh m_i m_j c(r/h) e_ij  -  g_curv (n_i - n_j) ]
//     e_ij = (x_i - x_j)/r,  K_ij = 2 n0 / (n_i + n_j)  (fluid number density, P kernel)
//     c(q) = (1-q)^3 q^3              , 1/2 < q <= 1   (attraction)
//          = 2 (1-q)^3 q^3 - 1/64     ,   0 < q <= 1/2 (repulsive core)
//   Cohesion calibrated so that separating two half lattices costs 2 sigma:
//     2 sigma A = g_coh m^2 h sum_{layers z>=1} z I(r/h),   I(q) = int_q^1 c dt
//   A = dp (2D, per unit depth) or dp^2 (3D), m = rho dp^DIM.
//     g_curv = AkinciCurvature * sigma * dp^(DIM-2)        (0 = off)
//
// SurfaceTension (per Type) is the physical sigma [N/m] for models 1 and 2.
//
// Data file keys (all optional):
//   SurfaceTensionModel      0|1|2      0
//   AkinciRadiusRatio        h/dp       3.2   (<= RadiusRatioP)
//   AkinciCurvature          -          0.0
//   SurfaceNormalThreshold   -          0.25  (model 2)
//   SurfaceKappaSmoothing    passes     3     (model 2, >= 1)
//   FreeSurfaceDivRLower     -          1.6 (2D) / 2.6 (3D)   free-surface detector:
//   FreeSurfaceDivRUpper     -          1.8 (2D) / 2.8 (3D)   alpha 0 below, 1 above
//     (any model; a band of one layer only, e.g. 1.25/1.45 in 2D, is unstable:
//      the second layer's PPE row has a truncated kernel)
// =============================================================================

static int    SurfaceTensionModel = 0;
static int    LaplaceSurfaceBC    = 0;     // SurfaceTensionModel==2, device copy
static double AkinciRadiusRatio   = 3.2;
static double AkinciCurvature     = 0.0;
static double AkinciRadius        = 0.0;
static double AkinciGammaCoh[TYPE_COUNT]  = {0.0,0.0,0.0,0.0};
static double AkinciGammaCurv[TYPE_COUNT] = {0.0,0.0,0.0,0.0};
static double SurfaceNormalThreshold = 0.25;
static int    SurfaceKappaSmoothing  = 3;
#ifdef TWO_DIMENSIONAL
static double FreeSurfaceDivRLower=1.6, FreeSurfaceDivRUpper=1.8;
#else
static double FreeSurfaceDivRLower=2.6, FreeSurfaceDivRUpper=2.8;
#endif
#pragma acc declare create(FreeSurfaceDivRLower,FreeSurfaceDivRUpper)
static double SurfaceNormalMin    = 0.0;   // threshold * |n| of a flat surface
#pragma acc declare create(LaplaceSurfaceBC,AkinciRadius,AkinciGammaCoh,AkinciGammaCurv,SurfaceNormalMin)

static double *SurfaceKappa    = NULL;     // raw curvature (valid particles)
static double *SurfacePressure = NULL;     // sigma * smoothed curvature
static double *SurfaceSaveX    = NULL;
static int    *SurfaceValid    = NULL;
static int    *SurfaceSaveFlag = NULL;


static int surfaceTensionParseDataLine(const char *buf)
{
    if(sscanf(buf," SurfaceTensionModel %d",&SurfaceTensionModel)==1)       return 1;
    if(sscanf(buf," AkinciRadiusRatio %lf",&AkinciRadiusRatio)==1)         return 1;
    if(sscanf(buf," AkinciCurvature %lf",&AkinciCurvature)==1)             return 1;
    if(sscanf(buf," SurfaceNormalThreshold %lf",&SurfaceNormalThreshold)==1) return 1;
    if(sscanf(buf," SurfaceKappaSmoothing %d",&SurfaceKappaSmoothing)==1)   return 1;
    if(sscanf(buf," FreeSurfaceDivRLower %lf",&FreeSurfaceDivRLower)==1)    return 1;
    if(sscanf(buf," FreeSurfaceDivRUpper %lf",&FreeSurfaceDivRUpper)==1)    return 1;
    return 0;
}


#pragma acc routine seq
static double akinciC(const double q)
{
    if(q<=0.0 || q>=1.0) return 0.0;
    const double a=(1.0-q)*(1.0-q)*(1.0-q)*q*q*q;
    return (q>0.5 ? a : 2.0*a-1.0/64.0);
}


// I(q) = int_q^1 c(t) dt (Simpson)
static double akinciPotential(const double q)
{
    const int n=400;
    const double hs=(1.0-q)/n;
    double s=akinciC(q)+akinciC(1.0);
    for(int k=1;k<n;++k) s+=(k%2 ? 4.0 : 2.0)*akinciC(q+k*hs);
    return s*hs/3.0;
}


static void surfaceTensionInitialize()
{
    LaplaceSurfaceBC=(SurfaceTensionModel==2 ? 1 : 0);
    if(!(FreeSurfaceDivRUpper>FreeSurfaceDivRLower)){ log_printf("FreeSurfaceDivRUpper must exceed FreeSurfaceDivRLower\n"); exit(EXIT_FAILURE); }
    #pragma acc update device(LaplaceSurfaceBC,FreeSurfaceDivRLower,FreeSurfaceDivRUpper)
    log_printf("Free-surface detector: divR %e .. %e\n",FreeSurfaceDivRLower,FreeSurfaceDivRUpper);

    // allocated for every model: the PPE and shifting kernels list SurfacePressure
    SurfaceKappa   =(double *)malloc(ParticleCount*sizeof(double));
    SurfacePressure=(double *)malloc(ParticleCount*sizeof(double));
    SurfaceSaveX   =(double *)malloc(ParticleCount*sizeof(double));
    SurfaceValid   =(int *)malloc(ParticleCount*sizeof(int));
    SurfaceSaveFlag=(int *)malloc(ParticleCount*sizeof(int));
    for(int i=0;i<ParticleCount;++i){ SurfaceKappa[i]=0.0; SurfacePressure[i]=0.0; SurfaceValid[i]=0; }
    #pragma acc enter data copyin(SurfaceKappa[0:ParticleCount],SurfacePressure[0:ParticleCount],SurfaceValid[0:ParticleCount])
    #pragma acc enter data create(SurfaceSaveX[0:ParticleCount],SurfaceSaveFlag[0:ParticleCount])

    if(SurfaceTensionModel==0) return;
    const double dp=ParticleSpacing;

    if(SurfaceTensionModel==1){
        AkinciRadius=AkinciRadiusRatio*dp;
        if(AkinciRadius>RadiusP*(1.0+1.0e-12)){
            log_printf("AkinciRadiusRatio=%e exceeds RadiusRatioP: neighbour list too short\n",AkinciRadiusRatio);
            exit(EXIT_FAILURE);
        }
        const int range=(int)ceil(AkinciRadiusRatio);
        double sum=0.0;
#ifdef TWO_DIMENSIONAL
        for(int iz=1;iz<=range;++iz) for(int ix=-range;ix<=range;++ix){
            const double q=sqrt((double)(ix*ix+iz*iz))/AkinciRadiusRatio;
            if(q<1.0) sum+=(double)iz*akinciPotential(q);
        }
        const double area=dp, curvScale=1.0;
#else
        for(int iz=1;iz<=range;++iz) for(int iy=-range;iy<=range;++iy) for(int ix=-range;ix<=range;++ix){
            const double q=sqrt((double)(ix*ix+iy*iy+iz*iz))/AkinciRadiusRatio;
            if(q<1.0) sum+=(double)iz*akinciPotential(q);
        }
        const double area=dp*dp, curvScale=dp;
#endif
        for(int iT=0;iT<TYPE_COUNT;++iT){
            const double m=Density[iT]*ParticleVolume;
            AkinciGammaCoh[iT]=(sum>1.0e-30 && m>0.0 ? 2.0*SurfaceTension[iT]*area/(m*m*AkinciRadius*sum) : 0.0);
            AkinciGammaCurv[iT]=AkinciCurvature*SurfaceTension[iT]*curvScale;
        }
        log_printf("Akinci surface tension: h/dp=%e latticeSum=%e gammaCoh(T1)=%e gammaCurv(T1)=%e\n",
                   AkinciRadiusRatio,sum,AkinciGammaCoh[1],AkinciGammaCurv[1]);
        #pragma acc update device(AkinciRadius,AkinciGammaCoh[0:TYPE_COUNT],AkinciGammaCurv[0:TYPE_COUNT])
        return;
    }

    if(SurfaceTensionModel!=2){ log_printf("Unknown SurfaceTensionModel %d\n",SurfaceTensionModel); exit(EXIT_FAILURE); }
    // |n| of a particle on a flat surface of a half lattice (z <= 0)
    const int range=(int)ceil(RadiusP/dp);
    double nFlat=0.0;
#ifdef TWO_DIMENSIONAL
    for(int iz=-range;iz<=0;++iz) for(int ix=-range;ix<=range;++ix){
        const double r=dp*sqrt((double)(ix*ix+iz*iz));
        if(r<=1.0e-30 || r>=RadiusP) continue;
        nFlat+=-RadiusP*ParticleVolume*dwpdr(r,RadiusP)*(-dp*iz)/r;
    }
#else
    for(int iz=-range;iz<=0;++iz) for(int iy=-range;iy<=range;++iy) for(int ix=-range;ix<=range;++ix){
        const double r=dp*sqrt((double)(ix*ix+iy*iy+iz*iz));
        if(r<=1.0e-30 || r>=RadiusP) continue;
        nFlat+=-RadiusP*ParticleVolume*dwpdr(r,RadiusP)*(-dp*iz)/r;
    }
#endif
    SurfaceNormalMin=SurfaceNormalThreshold*nFlat;
    #pragma acc update device(SurfaceNormalMin)

    if(SurfaceKappaSmoothing<1) SurfaceKappaSmoothing=1;
    log_printf("Laplace surface pressure BC: |n_flat|=%e valid if |n| > %e (threshold %e), %d smoothing pass(es)\n",
               nFlat,SurfaceNormalMin,SurfaceNormalThreshold,SurfaceKappaSmoothing);
}


// outward normal (IfNormal) and fluid number density (IfDelta); both are
// two-phase fields, unused by the single-phase solver. Liquid rows only;
// solid particles (solid_body.inc) are not part of the liquid surface.
static void calculateSurfaceNormal()
{
    #pragma acc kernels present(IfNormal[0:ParticleCount][0:DIM],IfDelta[0:ParticleCount],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],SolidBodyOfId[0:ParticleCount],ParticleIndex[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        double nx=0.0, ny=0.0, nz=0.0, nd=0.0;
        if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END && solidBodyOf(iP)<0){
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const int jP=NeighborInd[NeighborPtr[iP]+jN];
                if(iP==jP) continue;
                if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                const double dx=Position[iP][0]-Position[jP][0];
                const double dy=Position[iP][1]-Position[jP][1];
                const double dz=Position[iP][2]-Position[jP][2];
                const double r2=dx*dx+dy*dy+dz*dz;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                const double r=sqrt(r2);
                if(solidBodyOf(jP)>=0) continue;
                const double f=-RadiusP*ParticleVolume*dwpdr(r,RadiusP)/r;
                nx+=f*dx; ny+=f*dy; nz+=f*dz;
                nd+=wp(r,RadiusP);
            }
        }
        IfNormal[iP][0]=nx; IfNormal[iP][1]=ny; IfNormal[iP][2]=nz;
        IfDelta[iP]=nd;
    }
}


static void calculateAkinciSurfaceTension()
{
    calculateSurfaceNormal();

    #pragma acc kernels present(Force[0:ParticleCount][0:DIM],IfNormal[0:ParticleCount][0:DIM],IfDelta[0:ParticleCount],Mass[0:ParticleCount],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],AkinciGammaCoh[0:TYPE_COUNT],AkinciGammaCurv[0:TYPE_COUNT])
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
            const double dx=Position[iP][0]-Position[jP][0];
            const double dy=Position[iP][1]-Position[jP][1];
            const double dz=Position[iP][2]-Position[jP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=AkinciRadius*AkinciRadius) continue;
            const double r=sqrt(r2);
            const double gCoh=0.5*(AkinciGammaCoh[Property[iP]]+AkinciGammaCoh[Property[jP]]);
            const double gCurv=0.5*(AkinciGammaCurv[Property[iP]]+AkinciGammaCurv[Property[jP]]);
            const double K=2.0*N0p/(IfDelta[iP]+IfDelta[jP]+1.0e-30);
            const double coh=-gCoh*Mass[iP]*Mass[jP]*akinciC(r/AkinciRadius)/r;
            fx+=K*(coh*dx-gCurv*(IfNormal[iP][0]-IfNormal[jP][0]));
            fy+=K*(coh*dy-gCurv*(IfNormal[iP][1]-IfNormal[jP][1]));
            fz+=K*(coh*dz-gCurv*(IfNormal[iP][2]-IfNormal[jP][2]));
        }
        Force[iP][0]+=fx;
        Force[iP][1]+=fy;
        Force[iP][2]+=fz;
    }
}


// p_s = sigma * kappa on every fluid particle (used on free-surface rows)
static void calculateSurfacePressure()
{
    calculateSurfaceNormal();

    // raw curvature: least-squares divergence of the unit normal
    #pragma acc kernels present(SurfaceKappa[0:ParticleCount],SurfaceValid[0:ParticleCount],IfNormal[0:ParticleCount][0:DIM],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        SurfaceKappa[iP]=0.0; SurfaceValid[iP]=0;
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;
        const double ni=sqrt(IfNormal[iP][0]*IfNormal[iP][0]+IfNormal[iP][1]*IfNormal[iP][1]+IfNormal[iP][2]*IfNormal[iP][2]);
        if(!(ni>SurfaceNormalMin)) continue;
        const double ux=IfNormal[iP][0]/ni, uy=IfNormal[iP][1]/ni, uz=IfNormal[iP][2]/ni;
        double M[3][3]={{0,0,0},{0,0,0},{0,0,0}}, B[3][3]={{0,0,0},{0,0,0},{0,0,0}};
        int cnt=0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP) continue;
            if(!(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
            const double nj=sqrt(IfNormal[jP][0]*IfNormal[jP][0]+IfNormal[jP][1]*IfNormal[jP][1]+IfNormal[jP][2]*IfNormal[jP][2]);
            if(!(nj>SurfaceNormalMin)) continue;
            const double d[3]={Position[jP][0]-Position[iP][0],Position[jP][1]-Position[iP][1],Position[jP][2]-Position[iP][2]};
            const double r2=d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double q=wp(sqrt(r2),RadiusP)/r2;
            const double dn[3]={IfNormal[jP][0]/nj-ux,IfNormal[jP][1]/nj-uy,IfNormal[jP][2]/nj-uz};
            #pragma acc loop seq
            for(int a=0;a<3;++a){
                #pragma acc loop seq
                for(int b=0;b<3;++b){ M[a][b]+=q*d[a]*d[b]; B[a][b]+=q*dn[a]*d[b]; }
            }
            ++cnt;
        }
#ifdef TWO_DIMENSIONAL
        if(cnt<3) continue;
        const double det=M[0][0]*M[1][1]-M[0][1]*M[1][0];
        const double sc=M[0][0]+M[1][1];
        if(!(det>1.0e-6*sc*sc)) continue;
        // kappa = tr(B M^-1)
        const double kappa=(B[0][0]*M[1][1]-B[0][1]*M[1][0]-B[1][0]*M[0][1]+B[1][1]*M[0][0])/det;
#else
        if(cnt<5) continue;
        const double c00=M[1][1]*M[2][2]-M[1][2]*M[2][1];
        const double c01=M[1][2]*M[2][0]-M[1][0]*M[2][2];
        const double c02=M[1][0]*M[2][1]-M[1][1]*M[2][0];
        const double det=M[0][0]*c00+M[0][1]*c01+M[0][2]*c02;
        const double sc=M[0][0]+M[1][1]+M[2][2];
        if(!(det>1.0e-6*sc*sc*sc)) continue;
        double Mi[3][3];
        Mi[0][0]=c00/det; Mi[1][0]=c01/det; Mi[2][0]=c02/det;
        Mi[0][1]=(M[0][2]*M[2][1]-M[0][1]*M[2][2])/det;
        Mi[1][1]=(M[0][0]*M[2][2]-M[0][2]*M[2][0])/det;
        Mi[2][1]=(M[0][1]*M[2][0]-M[0][0]*M[2][1])/det;
        Mi[0][2]=(M[0][1]*M[1][2]-M[0][2]*M[1][1])/det;
        Mi[1][2]=(M[0][2]*M[1][0]-M[0][0]*M[1][2])/det;
        Mi[2][2]=(M[0][0]*M[1][1]-M[0][1]*M[1][0])/det;
        double kappa=0.0;
        #pragma acc loop seq
        for(int a=0;a<3;++a){
            #pragma acc loop seq
            for(int b=0;b<3;++b) kappa+=B[a][b]*Mi[b][a];
        }
#endif
        SurfaceKappa[iP]=kappa;
        SurfaceValid[iP]=1;
    }

    // smoothing among the valid particles (SurfaceSaveX as scratch)
    for(int pass=1;pass<SurfaceKappaSmoothing;++pass){
        #pragma acc kernels present(SurfaceKappa[0:ParticleCount],SurfaceValid[0:ParticleCount],SurfaceSaveX[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP){
            SurfaceSaveX[iP]=SurfaceKappa[iP];
            if(!SurfaceValid[iP]) continue;
            double sw=wp(0.0,RadiusP), sk=sw*SurfaceKappa[iP];
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[iP];++jN){
                const int jP=NeighborInd[NeighborPtr[iP]+jN];
                if(iP==jP || !SurfaceValid[jP]) continue;
                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double dz=Position[jP][2]-Position[iP][2];
                const double r2=dx*dx+dy*dy+dz*dz;
                if(r2>=RadiusP*RadiusP) continue;
                const double w=wp(sqrt(r2),RadiusP);
                sw+=w; sk+=w*SurfaceKappa[jP];
            }
            SurfaceSaveX[iP]=sk/sw;
        }
        #pragma acc kernels present(SurfaceKappa[0:ParticleCount],SurfaceSaveX[0:ParticleCount])
        #pragma acc loop independent
        #pragma omp parallel for
        for(int iP=0;iP<ParticleCount;++iP) SurfaceKappa[iP]=SurfaceSaveX[iP];
    }

    // last pass over valid neighbours; also fills the invalid ones
    #pragma acc kernels present(SurfaceKappa[0:ParticleCount],SurfaceValid[0:ParticleCount],SurfacePressure[0:ParticleCount],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],SurfaceTension[0:TYPE_COUNT],SolidBodyOfId[0:ParticleCount],ParticleIndex[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        SurfacePressure[iP]=0.0;
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;
        if(solidBodyOf(iP)>=0) continue;              // solid surface rows: p = 0
        const double w0=wp(0.0,RadiusP);
        double sw=(SurfaceValid[iP] ? w0 : 0.0), sk=(SurfaceValid[iP] ? w0*SurfaceKappa[iP] : 0.0);
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP || !SurfaceValid[jP]) continue;
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2>=RadiusP*RadiusP) continue;
            const double w=wp(sqrt(r2),RadiusP);
            sw+=w; sk+=w*SurfaceKappa[jP];
        }
        SurfacePressure[iP]=(sw>0.0 ? SurfaceTension[Property[iP]]*sk/sw : 0.0);
    }
}


// Adds the Laplace pressure of the free-surface rows to PpeB (single phase,
// after the RHS is assembled, before the first operator call of the solve).
static void addSurfacePressureRhs()
{
    if(!LaplaceSurfaceBC) return;
    calculateSurfacePressure();

    // x = p_s on free-surface rows, 0 elsewhere; every fluid row active, so
    // y_i = -a_i sum_free c_ij p_s,j on the interior rows
    #pragma acc kernels present(SurfaceSaveX[0:ParticleCount],SurfaceSaveFlag[0:ParticleCount],SurfacePressure[0:ParticleCount],PpeOpX[0:ParticleCount],PpeFlag[0:ParticleCount],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        SurfaceSaveX[iP]=PpeOpX[iP];
        SurfaceSaveFlag[iP]=PpeFlag[iP];
        const int fluid=(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END);
        PpeOpX[iP]=(fluid && PpeFlag[iP]==0 ? SurfacePressure[iP] : 0.0);
        if(fluid) PpeFlag[iP]=1;
    }
    applyPpeOperator();
    #pragma acc kernels present(SurfaceSaveX[0:ParticleCount],SurfaceSaveFlag[0:ParticleCount],SurfacePressure[0:ParticleCount],PpeOpX[0:ParticleCount],PpeOpY[0:ParticleCount],PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],PpeDiag[0:ParticleCount],PpeB[0:ParticleCount],PressureP[0:ParticleCount],Property[0:ParticleCount])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        PpeFlag[iP]=SurfaceSaveFlag[iP];
        PpeOpX[iP]=SurfaceSaveX[iP];
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;
        if(PpeFlag[iP]!=0) PpeB[iP]+=-PpeOpY[iP]+(1.0-PpeAlpha[iP])*PpeDiag[iP]*SurfacePressure[iP];
        else               PressureP[iP]=SurfacePressure[iP];
    }
}
