// =============================================================================
// vis_deeponet.inc -- DeepONet two-level preconditioner for the implicit
// (backward-Euler) viscosity solve
//
// Included once by main_implicit.cpp, after implicitViscMatvec() and before
// solveImplicitViscosity().  It shares the particle arrays and the helpers of
// ppe_deeponet.inc (donMlp, donEncode, donInvertHost, donWtime, ...).
//
// System (unknowns = fluid particles x DIM velocity components, SPD):
//
//     A u = (M + Dt K) u = b,   (A u)_i = B_i u_i - sum_j c_ij e_ij (e_ij . u_j)
//
// B_i = M_i I + sum_j c_ij e e^T is a full DIM x DIM block (the central-pair
// tensor couples the Cartesian components), j runs over fluid neighbours; wall
// pairs only enter B_i.  The solver is CG with the original mass-weighted
// stopping test, so the preconditioner must be SPD:
//
//     z0 = S(r; 0)                                 pre-smoothing  (Chebyshev / Jacobi on D^-1 A)
//     z1 = z0 + V (V^T A V)^-1 V^T (r - A z0)      Galerkin coarse correction
//     z  = S(r; z1)                                post-smoothing (same polynomial)
//
// V = { T_k e_d }  k = 1..P, d = 1..sdim: the P scalar DeepONet trunk fields
// times the unit vectors, so the coarse space has P*sdim columns.  The coarse
// matrix is formed from the CURRENT operator; an unseen geometry or a new
// solid region only degrades the basis, never correctness.  Symmetric
// smoothing + Galerkin coarse correction gives an SPD M^-1, so plain CG is valid.
// Convergence is judged with implicitViscMatvec() itself; a failed DeepONet
// solve is handed to the original Jacobi CG.
//
// Data file keys (all optional):
//   VisDon               0|1      enable the DeepONet preconditioner     [0]
//   VisDonModel          path     weights from train_vis_don.py          [vis_deeponet.bin]
//   VisDonChebDegree     k        Chebyshev smoother degree (0: Jacobi)  [6]
//   VisDonChebRatio      q        Chebyshev interval [lmax/q, lmax]      [30]
//   VisDonSmoothSweeps   n        Jacobi sweeps when VisDonChebDegree 0  [1]
//   VisDonOmega          w        Jacobi damping                         [0.7]
//   VisDonLmaxEvery      n        re-estimate lmax(D^-1 A) every n solves [20]
//   VisDonSetupEvery     n        re-run the DeepONet (basis) every n solves; rows follow
//                                 the particle ids in between                    [10]
//   VisDonGalerkinEvery  n        rebuild V^T A V every n solves (stale = still SPD) [1]
//   VisDonMaxIteration   n        DeepONet-CG iterations before the Jacobi fallback [500]
//   VisDonVerifyEvery    n        check stored A against implicitViscMatvec every n solves [500]
//   VisDonExportEvery    n        write a training snapshot every n viscosity solves (0 off) [0]
//   VisDonExportMax      n        stop exporting after n snapshots       [100000]
//   VisDonExportDir      path     snapshot directory   [VIS_DON_DATA_ROOT/<case dir name>]
//                                 (files vis_*.bin; PPE snapshots ppe_*.bin may share it)
// =============================================================================

#define VIS_SD   DON_SDIM          // velocity components in the coarse space
#define VIS_NMLP 3                 // 0:E geometry encoder 1:G global 2:T trunk
#define VIS_BLK  (DIM*DIM)

#ifdef VIS_DON_ONET
static int    VisDon              = 1;
#else
static int    VisDon              = 0;
#endif
#ifndef VIS_DON_MODEL_DEFAULT
#define VIS_DON_MODEL_DEFAULT "vis_deeponet.bin"
#endif
#ifdef VIS_DON_DATASET
#ifndef VIS_DON_DATASET_EVERY
#define VIS_DON_DATASET_EVERY 50
#endif
#define VIS_DON_EXPORT_DEFAULT VIS_DON_DATASET_EVERY
#else
#define VIS_DON_EXPORT_DEFAULT 0
#endif
#ifndef VIS_DON_DATA_ROOT
#define VIS_DON_DATA_ROOT PPE_DON_DATA_ROOT
#endif
static int    VisDonChebDegree    = 6;
static double VisDonChebRatio     = 30.0;
static int    VisDonSmoothSweeps  = 1;
static double VisDonOmega         = 0.7;
static int    VisDonLmaxEvery     = 20;
static int    VisDonLmaxIters     = 15;    // power iterations for lmax(D^-1 A)
static int    VisDonSetupEvery    = 10;
static int    VisDonGalerkinEvery = 1;
static int    VisDonMaxIteration  = 500;
static int    VisDonVerifyEvery   = 500;
static int    VisDonExportEvery   = VIS_DON_EXPORT_DEFAULT;
static int    VisDonExportMax     = 100000;
static double VisDonGalerkinReg   = 1.0e-10;
static char   VisDonModel[1024]     = VIS_DON_MODEL_DEFAULT;
static char   VisDonExportDir[1024] = "";

// ---- timing breakdown (seconds, accumulated) ----
enum { VT_OPER, VT_VERIFY, VT_GEOM, VT_LMAX, VT_NET, VT_GALERKIN, VT_APPLY, VT_MATVEC, VT_KRYLOV, VT_COUNT };
static const char *VdTmName[VT_COUNT]={"operator","verify","geometry","lmax","deeponet","galerkin",
                                        "precond-apply","matvec","krylov-other"};
static double VdTm[VT_COUNT];
static long   VdApplyCount=0;

// ---- model ----
static int     VdModelState=0;          // 0 not loaded, 1 ok, -1 failed
static int     VdNf=0, VdHe=0, VdZ=0, VdP=0, VdPc=0;
static double *VdW=NULL;  static long VdWCount=0;
static int    *VdTab=NULL;              // [VIS_NMLP*DON_TAB]

// ---- per-solve arrays ----
static int     VdN=0;
static int    *VdAct=NULL, *VdPid=NULL, *VdSlot=NULL;
static double *VdBlk=NULL;     // [n*DIM*DIM]  diagonal blocks B_i
static double *VdDiag=NULL;    // [n*DIM]      point Jacobi diagonal (= diag of B_i, 1 on inactive rows)
static double *VdX=NULL;       // [n*DON_NX]   xhat | local descriptors
static double *VdX0=NULL;      // [n*DIM]      warm start
static double *VdR=NULL, *VdZ0=NULL, *VdPv=NULL, *VdAp=NULL;      // CG vectors [n*DIM]
static double *VdZa=NULL, *VdZb=NULL, *VdDv=NULL, *VdRc=NULL;     // preconditioner scratch [n*DIM]
static long    VdNb=0;
static double *VdCoef=NULL, *VdE=NULL;   // fluid-neighbour slots: c_k, e_k [nb], [nb*DIM]
static int     VdPAlloc=0, VdHAlloc=0;
static double *VdHid=NULL, *VdPool=NULL, *VdZv=NULL;
static double *VdT=NULL, *VdTn=NULL;     // [P*n] column-major trunk basis
static double *VdAV=NULL;                // [Pc*n*SD]  A V
static double *VdAc=NULL, *VdAinv=NULL, *VdCc=NULL, *VdYc=NULL;

static double VdGlob[DON_NGLOB];
static int    VdNAct=0, VdSetupValid=0, VdGalerkinValid=0, VdOperatorOk=1, VdSelfTestDone=0;
static long   VdSolveCount=0, VdLastSetup=-1, VdLastGalerkin=-1, VdReuseCount=0, VdExportCount=0;
static double VdSetupTime=0.0, VdSolveTime=0.0;
static long   VdIterTotal=0, VdFallbackCount=0, VdRunCount=0, VdLmaxRetryCount=0;
static double VdJacobiTime=0.0;
static long   VdJacobiIter=0, VdJacobiRuns=0;

#define VD_TIC(v) const double v=donWtime()
#define VD_TOC(k,v) (VdTm[k]+=donWtime()-(v))
#define VD_NEW(ptr,type,count) do{ ptr=(type*)malloc(sizeof(type)*(size_t)(count)); \
    if(!ptr){ log_printf("VIS-DeepONet: allocation failed\n"); exit(EXIT_FAILURE);} }while(0)

#ifdef VIS_POU
#include "vis_pou.inc"     // partition-of-unity coarse space (VisPou 1)
#endif


static int visDonParseDataLine(const char *buf)
{
    if(sscanf(buf," VisDon %d",&VisDon)==1) return 1;
    if(sscanf(buf," VisDonModel %1023s",VisDonModel)==1) return 1;
    if(sscanf(buf," VisDonChebDegree %d",&VisDonChebDegree)==1) return 1;
    if(sscanf(buf," VisDonChebRatio %lf",&VisDonChebRatio)==1) return 1;
    if(sscanf(buf," VisDonSmoothSweeps %d",&VisDonSmoothSweeps)==1) return 1;
    if(sscanf(buf," VisDonOmega %lf",&VisDonOmega)==1) return 1;
    if(sscanf(buf," VisDonLmaxEvery %d",&VisDonLmaxEvery)==1) return 1;
    if(sscanf(buf," VisDonLmaxIters %d",&VisDonLmaxIters)==1) return 1;
#ifdef VIS_POU
    if(visPouParseDataLine(buf)) return 1;
#endif
    if(sscanf(buf," VisDonSetupEvery %d",&VisDonSetupEvery)==1) return 1;
    if(sscanf(buf," VisDonGalerkinEvery %d",&VisDonGalerkinEvery)==1) return 1;
    if(sscanf(buf," VisDonMaxIteration %d",&VisDonMaxIteration)==1) return 1;
    if(sscanf(buf," VisDonVerifyEvery %d",&VisDonVerifyEvery)==1) return 1;
    if(sscanf(buf," VisDonExportEvery %d",&VisDonExportEvery)==1) return 1;
    if(sscanf(buf," VisDonExportMax %d",&VisDonExportMax)==1) return 1;
    if(sscanf(buf," VisDonExportDir %1023s",VisDonExportDir)==1) return 1;
    if(sscanf(buf," VisDonGalerkinReg %lf",&VisDonGalerkinReg)==1) return 1;
    return 0;
}

static void visResolveExportDir()
{
    if(VisDonExportDir[0]) return;
    char cwd[1024]; const char *name="case";
    if(getcwd(cwd,sizeof(cwd))){ const char *sl=strrchr(cwd,'/'); if(sl && sl[1]) name=sl+1; }
    snprintf(VisDonExportDir,sizeof(VisDonExportDir),"%s/%s",VIS_DON_DATA_ROOT,name);
}

static void visDonLogSettings()
{
    visResolveExportDir();
#ifdef VIS_POU
    pouLogSettings();
#endif
    log_printf("VIS-DeepONet: enabled=%d model=%s smoother=%s(degree/sweeps=%d ratio=%g omega=%g) setupEvery=%d galerkinEvery=%d maxIter=%d verifyEvery=%d\n",
               VisDon,VisDonModel,(VisDonChebDegree>0 ? "Chebyshev" : "Jacobi"),
               (VisDonChebDegree>0 ? VisDonChebDegree : VisDonSmoothSweeps),VisDonChebRatio,VisDonOmega,
               VisDonSetupEvery,VisDonGalerkinEvery,VisDonMaxIteration,VisDonVerifyEvery);
    if(VisDonExportEvery>0)
        log_printf("VIS-DeepONet: exporting a training snapshot every %d viscosity solves to %s (max %d)\n",
                   VisDonExportEvery,VisDonExportDir,VisDonExportMax);
}

static void visMlpHost(const int m, const std::vector<double> &in, std::vector<double> &out)
{
    const int *t=VdTab+m*DON_TAB;
    std::vector<double> x(in), y;
    for(int l=0;l<t[0];++l){
        const int ni=t[1+5*l], no=t[2+5*l], act=t[3+5*l], oW=t[4+5*l], oB=t[5+5*l];
        y.assign(no,0.0);
        for(int o=0;o<no;++o){
            double s=VdW[oB+o];
            for(int k=0;k<ni;++k) s+=VdW[oW+(long)o*ni+k]*x[k];
            if(act==1) s=s/(1.0+exp(-s));
            y[o]=s;
        }
        x.swap(y);
    }
    out.swap(x);
}


// -----------------------------------------------------------------------------
// Buffers
// -----------------------------------------------------------------------------
static void visAllocateParticleBuffers()
{
    if(VdN==ParticleCount && VdAct) return;
    if(VdAct){
        const int n=VdN;
        int *act=VdAct, *pid=VdPid, *slot=VdSlot;
        double *blk=VdBlk,*dg=VdDiag,*X=VdX,*x0=VdX0,*r=VdR,*z=VdZ0,*p=VdPv,*ap=VdAp,*za=VdZa,*zb=VdZb,*dv=VdDv,*rc=VdRc;
        #pragma acc exit data delete(act[0:n],pid[0:n],slot[0:n],blk[0:n*VIS_BLK],dg[0:n*DIM],X[0:n*DON_NX],x0[0:n*DIM])
        #pragma acc exit data delete(r[0:n*DIM],z[0:n*DIM],p[0:n*DIM],ap[0:n*DIM],za[0:n*DIM],zb[0:n*DIM],dv[0:n*DIM],rc[0:n*DIM])
        free(act); free(pid); free(slot); free(blk); free(dg); free(X); free(x0);
        free(r); free(z); free(p); free(ap); free(za); free(zb); free(dv); free(rc);
    }
    VdN=ParticleCount;
    const int n=VdN;
    VD_NEW(VdAct,int,n); VD_NEW(VdPid,int,n); VD_NEW(VdSlot,int,n);
    VD_NEW(VdBlk,double,(long)n*VIS_BLK); VD_NEW(VdDiag,double,(long)n*DIM); VD_NEW(VdX,double,(long)n*DON_NX);
    VD_NEW(VdX0,double,(long)n*DIM);
    VD_NEW(VdR,double,(long)n*DIM); VD_NEW(VdZ0,double,(long)n*DIM); VD_NEW(VdPv,double,(long)n*DIM); VD_NEW(VdAp,double,(long)n*DIM);
    VD_NEW(VdZa,double,(long)n*DIM); VD_NEW(VdZb,double,(long)n*DIM); VD_NEW(VdDv,double,(long)n*DIM); VD_NEW(VdRc,double,(long)n*DIM);
    int *act=VdAct, *pid=VdPid, *slot=VdSlot;
    double *blk=VdBlk,*dg=VdDiag,*X=VdX,*x0=VdX0,*r=VdR,*z=VdZ0,*p=VdPv,*ap=VdAp,*za=VdZa,*zb=VdZb,*dv=VdDv,*rc=VdRc;
    #pragma acc enter data create(act[0:n],pid[0:n],slot[0:n],blk[0:n*VIS_BLK],dg[0:n*DIM],X[0:n*DON_NX],x0[0:n*DIM])
    #pragma acc enter data create(r[0:n*DIM],z[0:n*DIM],p[0:n*DIM],ap[0:n*DIM],za[0:n*DIM],zb[0:n*DIM],dv[0:n*DIM],rc[0:n*DIM])
    VdSetupValid=0; VdGalerkinValid=0;
    VdPAlloc=0;   // model buffers follow the new N
}

static void visAllocateNeighborBuffers()
{
    if(VdCoef && VdNb==NeighborIndCount) return;
    if(VdCoef){
        double *c=VdCoef, *e=VdE; const long nb=VdNb;
        #pragma acc exit data delete(c[0:nb],e[0:nb*DIM])
        free(c); free(e);
    }
    VdNb=NeighborIndCount;
    VD_NEW(VdCoef,double,VdNb); VD_NEW(VdE,double,VdNb*DIM);
    double *c=VdCoef, *e=VdE; const long nb=VdNb;
    #pragma acc enter data create(c[0:nb],e[0:nb*DIM])
}

static void visAllocateModelBuffers()
{
    if(VdPAlloc==VdP && VdHAlloc==VdHe && VdT) return;
    const int n=VdN;
    if(VdT){
        const int P=VdPAlloc, H=VdHAlloc, Pc=P*VIS_SD;
        double *T=VdT,*Tn=VdTn,*AV=VdAV,*hid=VdHid,*pool=VdPool,*zv=VdZv,*Ac=VdAc,*Ai=VdAinv,*cc=VdCc,*yc=VdYc;
        #pragma acc exit data delete(T[0:P*n],Tn[0:P*n],AV[0:Pc*n*VIS_SD],hid[0:n*H],pool[0:H],zv[0:DON_MAXW],Ac[0:Pc*Pc],Ai[0:Pc*Pc],cc[0:Pc],yc[0:Pc])
        free(T); free(Tn); free(AV); free(hid); free(pool); free(zv); free(Ac); free(Ai); free(cc); free(yc);
    }
    const int P=VdP, H=VdHe, Pc=P*VIS_SD;
    VD_NEW(VdT,double,(long)P*n); VD_NEW(VdTn,double,(long)P*n); VD_NEW(VdAV,double,(long)Pc*n*VIS_SD);
    VD_NEW(VdHid,double,(long)n*H); VD_NEW(VdPool,double,H); VD_NEW(VdZv,double,DON_MAXW);
    VD_NEW(VdAc,double,Pc*Pc); VD_NEW(VdAinv,double,Pc*Pc); VD_NEW(VdCc,double,Pc); VD_NEW(VdYc,double,Pc);
    double *T=VdT,*Tn=VdTn,*AV=VdAV,*hid=VdHid,*pool=VdPool,*zv=VdZv,*Ac=VdAc,*Ai=VdAinv,*cc=VdCc,*yc=VdYc;
    #pragma acc enter data create(T[0:P*n],Tn[0:P*n],AV[0:Pc*n*VIS_SD],hid[0:n*H],pool[0:H],zv[0:DON_MAXW],Ac[0:Pc*Pc],Ai[0:Pc*Pc],cc[0:Pc],yc[0:Pc])
    VdPAlloc=P; VdHAlloc=H;
    VdSetupValid=0; VdGalerkinValid=0;
}


// -----------------------------------------------------------------------------
// Model file (written by train_vis_don.py, little endian)
//   char[8] "DONVIS01"; int32 sdim,nloc,nglob,nf,He,Z,P,nmlp(=3)
//   double B[nf*sdim]
//   nmlp x { int32 nl; nl x { int32 in,out,act; double W[out*in]; double b[out] } }
// -----------------------------------------------------------------------------
static int visLoadModel()
{
    FILE *fp=fopen(VisDonModel,"rb");
    if(!fp){ log_printf("VIS-DeepONet: cannot open model %s -> Jacobi CG only\n",VisDonModel); return 0; }
    char magic[8]; int hdr[8];
    if(fread(magic,1,8,fp)!=8 || memcmp(magic,"DONVIS01",8)!=0 || fread(hdr,sizeof(int),8,fp)!=8){
        log_printf("VIS-DeepONet: %s is not a DONVIS01 model\n",VisDonModel); fclose(fp); return 0;
    }
    const int sdim=hdr[0], nloc=hdr[1], nglob=hdr[2], nf=hdr[3], He=hdr[4], Z=hdr[5], P=hdr[6], nmlp=hdr[7];
    if(sdim!=DON_SDIM || nloc!=DON_NLOC || nglob!=DON_NGLOB || nmlp!=VIS_NMLP || P<1 || P>DON_MAXP ||
       2*nf+DON_NX+Z>DON_MAXW || He>DON_MAXW || Z>DON_MAXW){
        log_printf("VIS-DeepONet: model dimensions (sdim=%d nloc=%d nglob=%d nf=%d He=%d Z=%d P=%d nmlp=%d) do not match this build\n",
                   sdim,nloc,nglob,nf,He,Z,P,nmlp);
        fclose(fp); return 0;
    }
    std::vector<double> w(nf*sdim);
    if(fread(w.data(),sizeof(double),w.size(),fp)!=w.size()){ fclose(fp); return 0; }
    VdTab=(int*)calloc(VIS_NMLP*DON_TAB,sizeof(int));
    for(int m=0;m<VIS_NMLP;++m){
        int nl=0;
        if(fread(&nl,sizeof(int),1,fp)!=1 || nl<1 || nl>DON_MAX_LAYERS){ fclose(fp); log_printf("VIS-DeepONet: bad layer count\n"); return 0; }
        VdTab[m*DON_TAB]=nl;
        for(int l=0;l<nl;++l){
            int lio[3];
            if(fread(lio,sizeof(int),3,fp)!=3){ fclose(fp); return 0; }
            const long nw=(long)lio[0]*lio[1];
            int *t=VdTab+m*DON_TAB+1+5*l;
            t[0]=lio[0]; t[1]=lio[1]; t[2]=lio[2]; t[3]=(int)w.size(); t[4]=(int)(w.size()+nw);
            if(m!=1 && lio[1]>DON_MAXW){ fclose(fp); log_printf("VIS-DeepONet: layer too wide for DON_MAXW\n"); return 0; }
            w.resize(w.size()+nw+lio[1]);
            if(fread(w.data()+t[3],sizeof(double),nw+lio[1],fp)!=(size_t)(nw+lio[1])){ fclose(fp); return 0; }
        }
    }
    fclose(fp);
    const int nenc=DON_NX+2*nf;
    const int *tE=VdTab+0*DON_TAB, *tG=VdTab+1*DON_TAB, *tT=VdTab+2*DON_TAB;
    const int lastE=1+5*(tE[0]-1), lastG=1+5*(tG[0]-1), lastT=1+5*(tT[0]-1);
    if(tE[1]!=nenc || tE[lastE+1]!=He || tG[1]!=He+DON_NGLOB || tG[lastG+1]!=Z || tT[1]!=nenc+Z || tT[lastT+1]!=P){
        log_printf("VIS-DeepONet: inconsistent layer shapes in %s\n",VisDonModel); return 0;
    }
    VdNf=nf; VdHe=He; VdZ=Z; VdP=P; VdPc=P*VIS_SD;
    VdWCount=(long)w.size();
    VD_NEW(VdW,double,VdWCount);
    memcpy(VdW,w.data(),sizeof(double)*VdWCount);
    double *W=VdW; int *tab=VdTab; const long nw=VdWCount;
    #pragma acc enter data copyin(W[0:nw],tab[0:VIS_NMLP*DON_TAB])
    log_printf("VIS-DeepONet: loaded %s  P=%d (coarse %d) He=%d Z=%d fourier=%d params=%ld\n",VisDonModel,P,VdPc,He,Z,nf,VdWCount);
    return 1;
}


// -----------------------------------------------------------------------------
// Stored viscosity operator: exact mirror of implicitViscMatvec().  Must be
// kept in sync with it; visVerifyOperator() detects any drift.
// -----------------------------------------------------------------------------
static void visBuildOperator(const double *nw, const double *dw)
{
    const int n=ParticleCount, n3=ParticleCount*DIM;
    int *act=VdAct; double *blk=VdBlk, *dg=VdDiag, *coef=VdCoef, *E=VdE; const long nb=VdNb;
    #pragma acc parallel loop present(act[0:n],blk[0:n*VIS_BLK],dg[0:n3],coef[0:nb],E[0:nb*DIM],nw[0:n3],dw[0:n],Mass[0:ParticleCount],Property[0:ParticleCount],Mu[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        const long k0=NeighborPtr[i];
        const int nn=NeighborCount[i];
        double *bi=blk+(long)i*VIS_BLK;
        #pragma acc loop seq
        for(int jn=0;jn<nn;++jn){
            coef[k0+jn]=0.0;
            for(int d=0;d<DIM;++d) E[(k0+jn)*DIM+d]=0.0;
        }
        for(int q=0;q<VIS_BLK;++q) bi[q]=0.0;
        act[i]=(Property[i]>=FLUID_BEGIN && Property[i]<FLUID_END);
        if(!act[i]){
            for(int d=0;d<DIM;++d){ bi[d*DIM+d]=1.0; dg[DIM*i+d]=1.0; }
            continue;
        }
        for(int d=0;d<DIM;++d) bi[d*DIM+d]=Mass[i];
        #pragma acc loop seq
        for(int jn=0;jn<nn;++jn){
            const int j=NeighborInd[k0+jn];
            double e[DIM]={0.0,0.0,0.0};
            double c=Dt*implicitViscPair(i,j,e);
            if(c==0.0) continue;
            const int fluid=(Property[j]>=FLUID_BEGIN && Property[j]<FLUID_END);
            if(!fluid){
                if(WallViscTangential) projectWallPair(&nw[DIM*i],e);
                c*=morrisWallFactor(i,j,nw,dw);
            }
            for(int d=0;d<DIM;++d)
                for(int q=0;q<DIM;++q) bi[d*DIM+q]+=c*e[d]*e[q];
            if(fluid){
                coef[k0+jn]=c;
                for(int d=0;d<DIM;++d) E[(k0+jn)*DIM+d]=e[d];
            }
        }
        for(int d=0;d<DIM;++d) dg[DIM*i+d]=bi[d*DIM+d];
    }
}

// (A x)_i of the stored operator (fluid row i)
#pragma acc routine seq
static void visRowOp(const int i, const double *blk, const double *coef, const double *E, const double *x, double *v)
{
    const double *bi=blk+(long)i*VIS_BLK;
    for(int d=0;d<DIM;++d){
        double s=0.0;
        for(int q=0;q<DIM;++q) s+=bi[d*DIM+q]*x[DIM*i+q];
        v[d]=s;
    }
    const long k0=NeighborPtr[i];
    #pragma acc loop seq
    for(int jn=0;jn<NeighborCount[i];++jn){
        const long k=k0+jn;
        const double c=coef[k];
        if(c==0.0) continue;
        const int j=NeighborInd[k];
        double dot=0.0;
        for(int d=0;d<DIM;++d) dot+=E[k*DIM+d]*x[DIM*j+d];
        for(int d=0;d<DIM;++d) v[d]-=c*E[k*DIM+d]*dot;
    }
}

// y = A x with the stored operator (inactive rows -> 0), queued on async(1)
static void visMatvecImpl(const double *x, double *y)
{
    const int n=ParticleCount, n3=ParticleCount*DIM;
    const int *act=VdAct; const double *blk=VdBlk, *coef=VdCoef, *E=VdE; const long nb=VdNb;
    #pragma acc parallel loop async(1) present(x[0:n3],y[0:n3],act[0:n],blk[0:n*VIS_BLK],coef[0:nb],E[0:nb*DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(!act[i]){ for(int d=0;d<DIM;++d) y[DIM*i+d]=0.0; continue; }
        double v[DIM];
        visRowOp(i,blk,coef,E,x,v);
        for(int d=0;d<DIM;++d) y[DIM*i+d]=v[d];
    }
}
static void visMatvec(const double *x, double *y)
{
    VD_TIC(t); visMatvecImpl(x,y); VD_TOC(VT_MATVEC,t);
}

// Compare the stored operator with implicitViscMatvec on a deterministic probe.
static double visVerifyOperator(const double *nw, const double *dw)
{
    const int n=ParticleCount, n3=ParticleCount*DIM;
    const int *act=VdAct; double *za=VdZa, *zb=VdZb, *ref=VdAp;
    #pragma acc parallel loop present(act[0:n],za[0:n3])
    #pragma omp parallel for
    for(int i=0;i<n;++i)
        for(int d=0;d<DIM;++d) za[DIM*i+d]=(act[i] ? sin(0.7310*(DIM*i+d)+0.3)+0.5*cos(0.1137*(DIM*i+d)) : 0.0);
    implicitViscMatvec(za,ref,nw,dw);
    visMatvecImpl(za,zb);
    #pragma acc wait(1)
    double e2=0.0, y2=0.0;
    #pragma acc parallel loop reduction(+:e2,y2) present(zb[0:n3],ref[0:n3],act[0:n])
    #pragma omp parallel for reduction(+:e2,y2)
    for(int i=0;i<n;++i){
        if(!act[i]) continue;
        for(int d=0;d<DIM;++d){
            const double q=ref[DIM*i+d]-zb[DIM*i+d];
            e2+=q*q; y2+=ref[DIM*i+d]*ref[DIM*i+d];
        }
    }
    return sqrt(e2/(y2>1.0e-300 ? y2 : 1.0e-300));
}


// -----------------------------------------------------------------------------
// Geometry descriptors (identical definition at export and inference).
//   s_i  = tr(B_i)/(sdim M_i) - 1        viscous stiffness relative to inertia (~ Dt mu / (rho dp^2))
//   xhat = (x - c)/L
//   loc  = [log(1+s)/10, (log(1+s) - <log(1+s)>)/5, sum_fluid w/N0, sum_wall w/N0,
//           wall distance/RadiusV, log(D_i/<D>)]
//   glob = [log n/10, log(ext_x/dp)/5, log(ext_y/dp)/5, log(ext_z/dp)/5,
//           max log(1+s)/10, <log(1+s)>/10, std log(1+s)/5, wall-adjacent fraction]
// -----------------------------------------------------------------------------
#pragma acc routine seq
static double visLogStiff(const int i, const double *blk)
{
    const double *bi=blk+(long)i*VIS_BLK;
    double tr=0.0;
    for(int d=0;d<VIS_SD;++d) tr+=bi[d*DIM+d];
    double s=tr/(VIS_SD*Mass[i])-1.0;
    if(!(s>0.0)) s=0.0;
    return log1p(s);
}

static int visComputeGeometry(const double *dw)
{
    const int n=ParticleCount, n3=ParticleCount*DIM;
    const int *act=VdAct; const double *blk=VdBlk, *dg=VdDiag;
    double xmn=1e300,ymn=1e300,zmn=1e300,xmx=-1e300,ymx=-1e300,zmx=-1e300, sumD=0.0, sl=0.0, sl2=0.0, slmax=0.0;
    int nAct=0, nWall=0;
    #pragma acc parallel loop reduction(min:xmn,ymn,zmn) reduction(max:xmx,ymx,zmx,slmax) reduction(+:sumD,sl,sl2,nAct,nWall) present(act[0:n],blk[0:n*VIS_BLK],dg[0:n3],dw[0:n],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM])
    #pragma omp parallel for reduction(min:xmn,ymn,zmn) reduction(max:xmx,ymx,zmx,slmax) reduction(+:sumD,sl,sl2,nAct,nWall)
    for(int i=0;i<n;++i){
        if(!act[i]) continue;
        ++nAct;
        double dm=0.0;
        for(int d=0;d<VIS_SD;++d) dm+=dg[DIM*i+d];
        sumD+=dm/VIS_SD;
        const double ls=visLogStiff(i,blk);
        sl+=ls; sl2+=ls*ls; if(ls>slmax) slmax=ls;
        if(dw[i]>0.0) ++nWall;
        const double x=Position[i][0], y=Position[i][1], z=Position[i][2];
        if(x<xmn) xmn=x; if(x>xmx) xmx=x;
        if(y<ymn) ymn=y; if(y>ymx) ymx=y;
        if(z<zmn) zmn=z; if(z>zmx) zmx=z;
    }
    VdNAct=nAct;
    if(nAct<1) return 0;
    const double ex=xmx-xmn, ey=ymx-ymn, ez=zmx-zmn;
    double L=(ex>ey ? ex : ey);
#ifndef TWO_DIMENSIONAL
    if(ez>L) L=ez;
#endif
    const double halfL=0.5*L+0.5*ParticleSpacing;
    const double dp=ParticleSpacing, meanLs=sl/nAct;
    const double varLs=sl2/nAct-meanLs*meanLs;
    VdGlob[0]=log((double)nAct)/10.0;
    VdGlob[1]=log((ex+dp)/dp)/5.0;
    VdGlob[2]=log((ey+dp)/dp)/5.0;
#ifdef TWO_DIMENSIONAL
    VdGlob[3]=0.0;
#else
    VdGlob[3]=log((ez+dp)/dp)/5.0;
#endif
    VdGlob[4]=slmax/10.0;
    VdGlob[5]=meanLs/10.0;
    VdGlob[6]=sqrt(varLs>0.0 ? varLs : 0.0)/5.0;
    VdGlob[7]=(double)nWall/nAct;

    const double meanD=sumD/nAct, cx=0.5*(xmn+xmx), cy=0.5*(ymn+ymx), cz=0.5*(zmn+zmx), iL=1.0/halfL;
    const double iRv=1.0/RadiusV;
    double *X=VdX;
    #pragma acc parallel loop present(X[0:n*DON_NX],act[0:n],blk[0:n*VIS_BLK],dg[0:n3],dw[0:n],Mass[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        double *xi=X+(long)i*DON_NX;
        if(!act[i]){
            #pragma acc loop seq
            for(int k=0;k<DON_NX;++k) xi[k]=0.0;
            continue;
        }
        xi[0]=(Position[i][0]-cx)*iL;
        xi[1]=(Position[i][1]-cy)*iL;
#ifndef TWO_DIMENSIONAL
        xi[2]=(Position[i][2]-cz)*iL;
#endif
        double fs=0.0, ws=0.0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[i];++jN){
            const int j=NeighborInd[NeighborPtr[i]+jN];
            if(i==j) continue;
            const double dx=Position[j][0]-Position[i][0];
            const double dy=Position[j][1]-Position[i][1];
            const double dz=Position[j][2]-Position[i][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
            const double w=wp(sqrt(r2),RadiusP);
            if(FLUID_BEGIN<=Property[j] && Property[j]<FLUID_END) fs+=w;
            else if(WALL_BEGIN<=Property[j] && Property[j]<WALL_END) ws+=w;
        }
        double dm=0.0;
        for(int d=0;d<VIS_SD;++d) dm+=dg[DIM*i+d];
        dm/=VIS_SD;
        const double ls=visLogStiff(i,blk);
        double *loc=xi+DON_SDIM;
        loc[0]=ls/10.0;
        loc[1]=(ls-meanLs)/5.0;
        loc[2]=fs/N0p;
        loc[3]=ws/N0p;
        loc[4]=dw[i]*iRv;
        loc[5]=log(dm/meanD);
    }
    return 1;
}


// -----------------------------------------------------------------------------
// DeepONet basis T (P scalar fields) and the Galerkin coarse operator
// -----------------------------------------------------------------------------
static int visSelfTest()
{
    if(VdSelfTestDone) return VdSelfTestDone>0;
    const int n=ParticleCount, P=VdP, Z=VdZ, nf=VdNf;
    double *T=VdT, *X=VdX; int *act=VdAct;
    #pragma acc update self(T[0:P*n],X[0:n*DON_NX],act[0:n])
    double eT=0.0, nT=0.0;
    int tested=0;
    for(int i=0;i<n && tested<64;i+=1+n/64){
        if(!act[i]) continue;
        std::vector<double> inp(DON_MAXW), t;
        donEncode(X+(long)i*DON_NX,VdW,nf,inp.data());
        for(int k=0;k<Z;++k) inp[DON_NX+2*nf+k]=VdZv[k];
        inp.resize(DON_NX+2*nf+Z);
        visMlpHost(2,inp,t);
        for(int k=0;k<P;++k){ const double d=t[k]-T[(long)k*n+i]; eT+=d*d; nT+=t[k]*t[k]; }
        ++tested;
    }
    const double rT=sqrt(eT/(nT+1e-300));
    const int ok=(tested>0 && rT<1.0e-10);
    log_printf("VIS-DeepONet: device inference self-test on %d particles: trunk err=%e -> %s\n",
               tested,rT,(ok ? "ok" : "FAILED, DeepONet path disabled"));
    VdSelfTestDone=(ok ? 1 : -1);
    if(!ok) VdModelState=-1;
    return ok;
}

static int visSetupBasis()
{
    const int n=ParticleCount, P=VdP, He=VdHe, Z=VdZ, nf=VdNf;
    const int *act=VdAct; const double *X=VdX, *W=VdW; const int *tab=VdTab;
    const long nw=VdWCount;
    double *hid=VdHid, *pool=VdPool, *zv=VdZv, *T=VdT;

    // 1) geometry encoder + mean pooling over the fluid particles
    #pragma acc parallel loop present(W[0:nw],tab[0:VIS_NMLP*DON_TAB],X[0:n*DON_NX],act[0:n],hid[0:n*He])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        double *h=hid+(long)i*He;
        if(!act[i]){
            #pragma acc loop seq
            for(int k=0;k<He;++k) h[k]=0.0;
            continue;
        }
        double enc[DON_MAXW];
        donEncode(X+(long)i*DON_NX,W,nf,enc);
        donMlp(W,tab,0,enc,h);
    }
    const double invN=1.0/VdNAct;
    #pragma acc parallel loop gang present(hid[0:n*He],pool[0:He])
    #pragma omp parallel for
    for(int k=0;k<He;++k){
        double s=0.0;
        #pragma acc loop vector reduction(+:s)
        for(int i=0;i<n;++i) s+=hid[(long)i*He+k];
        pool[k]=s*invN;
    }
    #pragma acc update self(pool[0:He])

    // 2) global latent z (host, tiny)
    std::vector<double> gin(He+DON_NGLOB), z;
    for(int k=0;k<He;++k) gin[k]=pool[k];
    for(int k=0;k<DON_NGLOB;++k) gin[He+k]=VdGlob[k];
    visMlpHost(1,gin,z);
    for(int k=0;k<Z;++k){ zv[k]=z[k]; if(!isfinite(zv[k])) return 0; }
    #pragma acc update device(zv[0:Z])

    // 3) trunk T at every fluid particle
    #pragma acc parallel loop present(W[0:nw],tab[0:VIS_NMLP*DON_TAB],X[0:n*DON_NX],act[0:n],zv[0:Z],T[0:P*n])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(!act[i]){
            #pragma acc loop seq
            for(int k=0;k<P;++k) T[(long)k*n+i]=0.0;
            continue;
        }
        double inp[DON_MAXW], t[DON_MAXP];
        donEncode(X+(long)i*DON_NX,W,nf,inp);
        #pragma acc loop seq
        for(int k=0;k<Z;++k) inp[DON_NX+2*nf+k]=zv[k];
        donMlp(W,tab,2,inp,t);
        #pragma acc loop seq
        for(int k=0;k<P;++k) T[(long)k*n+i]=t[k];
    }
    return visSelfTest();
}

// Ac = V^T A V  with  V_(i,d),(k,d') = T_ik delta_dd'   (coarse index k*SD+d), Ainv = Ac^-1
static int visSetupGalerkin()
{
    const int n=ParticleCount, P=VdP, Pc=VdPc;
    const int *act=VdAct; const double *blk=VdBlk, *coef=VdCoef, *E=VdE, *T=VdT; const long nb=VdNb;
    double *AV=VdAV, *Ac=VdAc, *Ai=VdAinv;
    // A V column c=(l,q): (A T_l e_q)_(i,d) = B_i[d][q] T_il - sum_k c_k e_d e_q T_jl
    #pragma acc parallel loop collapse(3) present(act[0:n],blk[0:n*VIS_BLK],coef[0:nb],E[0:nb*DIM],T[0:P*n],AV[0:Pc*n*VIS_SD],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for collapse(3)
    for(int c=0;c<Pc;++c){
        for(int i=0;i<n;++i){
            for(int d=0;d<VIS_SD;++d){
                const int l=c/VIS_SD, q=c%VIS_SD;
                double s=0.0;
                if(act[i]){
                    s=blk[(long)i*VIS_BLK+d*DIM+q]*T[(long)l*n+i];
                    const long k0=NeighborPtr[i];
                    #pragma acc loop seq
                    for(int jn=0;jn<NeighborCount[i];++jn){
                        const long k=k0+jn;
                        const double cf=coef[k];
                        if(cf==0.0) continue;
                        s-=cf*E[k*DIM+d]*E[k*DIM+q]*T[(long)l*n+NeighborInd[k]];
                    }
                }
                AV[((long)c*n+i)*VIS_SD+d]=s;
            }
        }
    }
    #pragma acc parallel loop gang collapse(2) present(T[0:P*n],AV[0:Pc*n*VIS_SD],Ac[0:Pc*Pc])
    #pragma omp parallel for collapse(2)
    for(int rw=0;rw<Pc;++rw){
        for(int c=0;c<Pc;++c){
            const int k=rw/VIS_SD, d=rw%VIS_SD;
            double s=0.0;
            #pragma acc loop vector reduction(+:s)
            for(int i=0;i<n;++i) s+=T[(long)k*n+i]*AV[((long)c*n+i)*VIS_SD+d];
            Ac[rw*Pc+c]=s;
        }
    }
    #pragma acc update self(Ac[0:Pc*Pc])
    std::vector<double> a(Pc*Pc), inv;
    double dmax=0.0;
    for(int r=0;r<Pc;++r) for(int c=0;c<Pc;++c) a[r*Pc+c]=0.5*(Ac[r*Pc+c]+Ac[c*Pc+r]);   // exact symmetry -> SPD M^-1
    for(int k=0;k<Pc;++k) if(fabs(a[k*Pc+k])>dmax) dmax=fabs(a[k*Pc+k]);
    if(!(dmax>0.0) || !isfinite(dmax)) return 0;
    for(int k=0;k<Pc;++k) a[k*Pc+k]+=VisDonGalerkinReg*dmax;
    if(!donInvertHost(Pc,a,inv)) return 0;
    for(int r=0;r<Pc;++r) for(int c=0;c<Pc;++c){
        const double v=0.5*(inv[r*Pc+c]+inv[c*Pc+r]);
        if(!isfinite(v)) return 0;
        Ai[r*Pc+c]=v;
    }
    #pragma acc update device(Ai[0:Pc*Pc])
    return 1;
}


// -----------------------------------------------------------------------------
// Smoother: VisDonChebDegree k>0 -> Jacobi-preconditioned Chebyshev of degree k
// on [lmax/VisDonChebRatio, lmax] of D^-1 A; k=0 -> VisDonSmoothSweeps damped
// Jacobi sweeps.  Steps  d = c1 d + c2 D^-1 (r - A src),  dst = src + d;
// identical to Smoother in vis_don_model.py.
// -----------------------------------------------------------------------------
static int    VdNSteps=0;
static double VdC1[DON_MAX_STEPS], VdC2[DON_MAX_STEPS];
static double VdLmax=0.0;
static long   VdLmaxSolve=-1000000;

static double visEstimateLmax()
{
    const int n=ParticleCount, n3=ParticleCount*DIM; const int *act=VdAct; const double *dg=VdDiag;
    double *x=VdZa, *y=VdZb;
    #pragma acc parallel loop present(x[0:n3],act[0:n])
    #pragma omp parallel for
    for(int i=0;i<n;++i)
        for(int d=0;d<DIM;++d) x[DIM*i+d]=(act[i] ? sin(0.731*(DIM*i+d)+0.3)+0.5*cos(0.1137*(DIM*i+d)) : 0.0);
    double lam=1.0;
    for(int it=0;it<VisDonLmaxIters;++it){
        visMatvecImpl(x,y);
        #pragma acc wait(1)
        double xy=0.0, xx=0.0, yy=0.0;
        #pragma acc parallel loop reduction(+:xy,xx,yy) present(x[0:n3],y[0:n3],dg[0:n3])
        #pragma omp parallel for reduction(+:xy,xx,yy)
        for(int k=0;k<n3;++k){ y[k]/=dg[k]; xy+=x[k]*y[k]; xx+=x[k]*x[k]; yy+=y[k]*y[k]; }
        lam=xy/(xx>1e-300 ? xx : 1e-300);
        const double inv=1.0/sqrt(yy>1e-300 ? yy : 1e-300);
        #pragma acc parallel loop present(x[0:n3],y[0:n3])
        #pragma omp parallel for
        for(int k=0;k<n3;++k) x[k]=y[k]*inv;
    }
    return 1.15*lam;
}

static int visSetupSmoother()
{
    if(VisDonChebDegree>0){
        if(VdSolveCount-VdLmaxSolve>=VisDonLmaxEvery || !(VdLmax>0.0)){
            VdLmax=visEstimateLmax();
            VdLmaxSolve=VdSolveCount;
            if(!(VdLmax>0.0) || !isfinite(VdLmax)) return 0;
        }
        const int k=(VisDonChebDegree<DON_MAX_STEPS ? VisDonChebDegree : DON_MAX_STEPS);
        const double b=VdLmax, a=b/VisDonChebRatio;
        const double theta=0.5*(b+a), delta=0.5*(b-a), sigma=theta/delta;
        double rho=1.0/sigma;
        VdC1[0]=0.0; VdC2[0]=1.0/theta;
        for(int j=1;j<k;++j){
            const double rho1=1.0/(2.0*sigma-rho);
            VdC1[j]=rho1*rho; VdC2[j]=2.0*rho1/delta;
            rho=rho1;
        }
        VdNSteps=k;
    }else{
        const int k=(VisDonSmoothSweeps>0 ? (VisDonSmoothSweeps<DON_MAX_STEPS ? VisDonSmoothSweeps : DON_MAX_STEPS) : 1);
        for(int j=0;j<k;++j){ VdC1[j]=0.0; VdC2[j]=VisDonOmega; }
        VdNSteps=k;
    }
    return 1;
}

static void visSmoothStep(const double *r, const double *src, const int hasSrc, double *dst, const double c1, const double c2)
{
    const int n=ParticleCount, n3=ParticleCount*DIM;
    const int *act=VdAct; const double *blk=VdBlk, *coef=VdCoef, *E=VdE, *dg=VdDiag; const long nb=VdNb;
    double *dv=VdDv;
    #pragma acc parallel loop async(1) present(r[0:n3],src[0:n3],dst[0:n3],dv[0:n3],dg[0:n3],act[0:n],blk[0:n*VIS_BLK],coef[0:nb],E[0:nb*DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(!act[i]){ for(int d=0;d<DIM;++d){ dst[DIM*i+d]=0.0; dv[DIM*i+d]=0.0; } continue; }
        double v[DIM]={0.0,0.0,0.0};
        if(hasSrc) visRowOp(i,blk,coef,E,src,v);
        for(int d=0;d<DIM;++d){
            const int k=DIM*i+d;
            const double res=r[k]-v[d];
            const double di=(c1!=0.0 ? c1*dv[k] : 0.0)+c2*res/dg[k];
            dv[k]=di;
            dst[k]=(hasSrc ? src[k] : 0.0)+di;
        }
    }
}

// -----------------------------------------------------------------------------
// z = M^-1 r  (symmetric two-level cycle, linear in r). out must not be a
// preconditioner scratch array.
// -----------------------------------------------------------------------------
static void visApplyImpl(const double *in, double *out)
{
    const int n=ParticleCount, n3=ParticleCount*DIM, P=VdP, Pc=VdPc;
    const int *act=VdAct; const double *blk=VdBlk, *coef=VdCoef, *E=VdE, *T=VdT, *Ai=VdAinv; const long nb=VdNb;
    double *za=VdZa, *zb=VdZb, *rc=VdRc, *cc=VdCc, *yc=VdYc;
    const int nS=VdNSteps;

    // pre-smoothing from 0
    double *z=NULL;
    for(int s=0;s<nS;++s){
        double *dst=(z==za ? zb : za);
        visSmoothStep(in,(z ? z : in),(z!=NULL),dst,VdC1[s],VdC2[s]);
        z=dst;
    }
    // coarse residual rc = in - A z
    #pragma acc parallel loop async(1) present(in[0:n3],z[0:n3],rc[0:n3],act[0:n],blk[0:n*VIS_BLK],coef[0:nb],E[0:nb*DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(!act[i]){ for(int d=0;d<DIM;++d) rc[DIM*i+d]=0.0; continue; }
        double v[DIM];
        visRowOp(i,blk,coef,E,z,v);
        for(int d=0;d<DIM;++d) rc[DIM*i+d]=in[DIM*i+d]-v[d];
    }
#ifdef VIS_POU
    if(VisPou){
        double *dst=(z==za ? zb : za);
        pouCoarseApply(rc,z,dst);
        z=dst;
    }else
#endif
    {
    // cc = V^T rc
    #pragma acc parallel loop gang async(1) present(T[0:P*n],rc[0:n3],cc[0:Pc])
    #pragma omp parallel for
    for(int rw=0;rw<Pc;++rw){
        const int k=rw/VIS_SD, d=rw%VIS_SD;
        double s=0.0;
        #pragma acc loop vector reduction(+:s)
        for(int i=0;i<n;++i) s+=T[(long)k*n+i]*rc[DIM*i+d];
        cc[rw]=s;
    }
    // yc = Ac^-1 cc
    #pragma acc parallel loop gang async(1) present(Ai[0:Pc*Pc],cc[0:Pc],yc[0:Pc])
    #pragma omp parallel for
    for(int rw=0;rw<Pc;++rw){
        double s=0.0;
        #pragma acc loop vector reduction(+:s)
        for(int c=0;c<Pc;++c) s+=Ai[rw*Pc+c]*cc[c];
        yc[rw]=s;
    }
    // z <- z + V yc
    double *dst=(z==za ? zb : za);
    #pragma acc parallel loop collapse(2) async(1) present(z[0:n3],dst[0:n3],T[0:P*n],yc[0:Pc],act[0:n])
    #pragma omp parallel for collapse(2)
    for(int i=0;i<n;++i){
        for(int d=0;d<DIM;++d){
            const int k=DIM*i+d;
            if(!act[i]){ dst[k]=0.0; continue; }
            double s=z[k];
            if(d<VIS_SD){
                #pragma acc loop seq
                for(int l=0;l<P;++l) s+=T[(long)l*n+i]*yc[l*VIS_SD+d];
            }
            dst[k]=s;
        }
    }
    z=dst;
    }
    // post-smoothing (same polynomial -> symmetric cycle)
    for(int s=0;s<nS;++s){
        double *d2=(s==nS-1 ? out : (z==za ? zb : za));
        visSmoothStep(in,z,1,d2,VdC1[s],VdC2[s]);
        z=d2;
    }
    #pragma acc wait(1)
}
static void visApply(const double *in, double *out)
{
    VD_TIC(t); visApplyImpl(in,out); VD_TOC(VT_APPLY,t); ++VdApplyCount;
}


// -----------------------------------------------------------------------------
// DeepONet-preconditioned CG.  Same stopping test as solveImplicitViscosity:
//     sum_fluid |r_i|^2 / M_i <= target2,
// with the true residual from implicitViscMatvec() before acceptance.
// -----------------------------------------------------------------------------
// r = b - A x (true operator), returns sum_fluid |r|^2/M
static double visTrueResidual(const double *x, const double *b, double *r, const double *nw, const double *dw)
{
    const int n=ParticleCount, n3=ParticleCount*DIM; double *ap=VdAp;
    #pragma acc wait(1)
    implicitViscMatvec(x,ap,nw,dw);
    double res2=0.0;
    #pragma acc parallel loop reduction(+:res2) present(x[0:n3],b[0:n3],r[0:n3],ap[0:n3],Mass[0:ParticleCount],Property[0:ParticleCount])
    #pragma omp parallel for reduction(+:res2)
    for(int i=0;i<n;++i){
        const int fluid=(Property[i]>=FLUID_BEGIN && Property[i]<FLUID_END);
        for(int d=0;d<DIM;++d){
            const int k=DIM*i+d;
            r[k]=(fluid ? b[k]-ap[k] : 0.0);
            if(fluid) res2+=r[k]*r[k]/Mass[i];
        }
    }
    return res2;
}

static int visPcg(double *x, const double *b, const double *nw, const double *dw, const double target2,
                  int *iterOut, double *res2IO)
{
    const int n=ParticleCount, n3=ParticleCount*DIM;
    double *r=VdR, *z=VdZ0, *p=VdPv, *ap=VdAp;
    int iter=0, converged=0;
    double res2=visTrueResidual(x,b,r,nw,dw);
    if(isfinite(res2) && res2<=target2){ *iterOut=0; *res2IO=res2; return 1; }
    visApply(r,z);
    double rz=0.0;
    #pragma acc parallel loop reduction(+:rz) present(r[0:n3],z[0:n3],p[0:n3])
    #pragma omp parallel for reduction(+:rz)
    for(int k=0;k<n3;++k){ p[k]=z[k]; rz+=r[k]*z[k]; }
    for(int it=0;it<VisDonMaxIteration;++it){
        if(!isfinite(rz) || !(rz>0.0)) break;
        visMatvec(p,ap);
        #pragma acc wait(1)
        double pap=0.0;
        #pragma acc parallel loop reduction(+:pap) present(p[0:n3],ap[0:n3])
        #pragma omp parallel for reduction(+:pap)
        for(int k=0;k<n3;++k) pap+=p[k]*ap[k];
        if(!isfinite(pap) || !(pap>0.0)) break;
        const double alpha=rz/pap;
        res2=0.0;
        #pragma acc parallel loop reduction(+:res2) present(x[0:n3],r[0:n3],p[0:n3],ap[0:n3],Mass[0:ParticleCount],Property[0:ParticleCount])
        #pragma omp parallel for reduction(+:res2)
        for(int i=0;i<n;++i){
            const int fluid=(Property[i]>=FLUID_BEGIN && Property[i]<FLUID_END);
            for(int d=0;d<DIM;++d){
                const int k=DIM*i+d;
                x[k]+=alpha*p[k]; r[k]-=alpha*ap[k];
                if(fluid) res2+=r[k]*r[k]/Mass[i];
            }
        }
        iter=it+1;
        const int refresh=(res2<=target2 || iter%50==0);
        if(refresh) res2=visTrueResidual(x,b,r,nw,dw);
        if(!isfinite(res2)) break;
        if(res2<=target2){ converged=1; break; }
        visApply(r,z);
        double rzNew=0.0;
        #pragma acc parallel loop reduction(+:rzNew) present(r[0:n3],z[0:n3])
        #pragma omp parallel for reduction(+:rzNew)
        for(int k=0;k<n3;++k) rzNew+=r[k]*z[k];
        const double beta=(refresh ? 0.0 : rzNew/rz);
        #pragma acc parallel loop present(z[0:n3],p[0:n3])
        #pragma omp parallel for
        for(int k=0;k<n3;++k) p[k]=z[k]+beta*p[k];
        rz=rzNew;
    }
    *iterOut=iter; *res2IO=res2;
    return converged;
}


// -----------------------------------------------------------------------------
// Hook called by solveImplicitViscosity() once b, x0 (= Velocity), nw, dw and
// the tolerance are ready.  Returns 1 if x solves the system to target2 (the
// Jacobi CG is skipped).  Otherwise x is a valid starting point (the warm
// start, or the DeepONet iterate if that is better) for the Jacobi CG.
// -----------------------------------------------------------------------------
#ifdef VIS_POU
// PoU coarse space: same operator, verification, smoother, CG and fallback as the
// DeepONet path; the lattice and V^T A V are rebuilt every solve (cheap, O(n)).
static int visPouSolve(double *x, const double *b, const double *nw, const double *dw,
                       const double target2, int *iterOut, double *res2Out)
{
    const double t0=donWtime();
    VdModelState=1;                 // no network: the summary lines report this path
    visAllocateNeighborBuffers();
    { VD_TIC(t); visBuildOperator(nw,dw); VD_TOC(VT_OPER,t); }
    if(VisDonVerifyEvery>0 && (VdRunCount==0 || VdSolveCount%VisDonVerifyEvery==0 || !VdOperatorOk)){
        VD_TIC(tv);
        const double err=visVerifyOperator(nw,dw);
        VD_TOC(VT_VERIFY,tv);
        const int ok=(isfinite(err) && err<1.0e-9);
        if(!ok || !VdOperatorOk || VdRunCount==0)
            log_printf("VIS-PoU: stored-operator check rel.err=%e -> %s\n",err,(ok ? "ok" : "MISMATCH, PoU path disabled"));
        VdOperatorOk=ok;
    }
    if(!VdOperatorOk) return 0;
    int setupOk=1;
    { VD_TIC(t); setupOk=pouBuildGrid(); VD_TOC(VT_GEOM,t); }
    if(setupOk){ VD_TIC(t); setupOk=visSetupSmoother(); VD_TOC(VT_LMAX,t); }
    if(setupOk){ VD_TIC(t); setupOk=pouSetupGalerkin(); VD_TOC(VT_GALERKIN,t); }
    if(!setupOk){
        log_printf("VIS-PoU: setup failed -> Jacobi CG\n");
        ++VdFallbackCount;
        return 0;
    }
    ++PgSetups; PgPcSum+=PgPc;
    const double t1=donWtime();
    int iter=0; double res2=0.0;
    int ok=visPcg(x,b,nw,dw,target2,&iter,&res2);
    if(!ok && VisDonChebDegree>0 && VdLmaxSolve!=VdSolveCount){
        const int n3=ParticleCount*DIM; const double *x0=VdX0;
        #pragma acc parallel loop present(x[0:n3],x0[0:n3])
        #pragma omp parallel for
        for(int k=0;k<n3;++k) x[k]=x0[k];
        VdLmaxSolve=-1000000;
        if(visSetupSmoother()){
            int iter2=0;
            ok=visPcg(x,b,nw,dw,target2,&iter2,&res2);
            iter+=iter2;
            ++VdLmaxRetryCount;
        }
    }
    const double t2=donWtime();
    VdSetupTime+=t1-t0; VdSolveTime+=t2-t1; VdIterTotal+=iter; ++VdRunCount;
    *iterOut=iter; *res2Out=res2;
    if(!ok){
        ++VdFallbackCount;
        const int n3=ParticleCount*DIM; const double *x0=VdX0; int bad=!isfinite(res2);
        #pragma acc parallel loop reduction(+:bad) present(x[0:n3])
        #pragma omp parallel for reduction(+:bad)
        for(int k=0;k<n3;++k) if(!isfinite(x[k])) ++bad;
        if(bad){
            #pragma acc parallel loop present(x[0:n3],x0[0:n3])
            #pragma omp parallel for
            for(int k=0;k<n3;++k) x[k]=x0[k];
        }
        log_printf("VIS-PoU: not converged (iter=%d res2/target2=%e) -> Jacobi CG fallback%s\n",
                   iter,res2/(target2>1e-300 ? target2 : 1e-300),(bad ? " from the warm start" : ""));
    }
    if(LogVerbose) log_printf("VIS-PoU: iter=%d res2/target2=%.3e setup=%.3fms solve=%.3fms Pc=%d nodes=%d lmax=%.4f\n",
               iter,res2/(target2>1e-300 ? target2 : 1e-300),1e3*(t1-t0),1e3*(t2-t1),PgPc,PgNodes,VdLmax);
    return ok;
}
#endif

static int visDonSolve(double *x, const double *b, const double *nw, const double *dw,
                       const double target2, int *iterOut, double *res2Out)
{
    *iterOut=0;
    ++VdSolveCount;
    if(!VisDon && VisDonExportEvery<=0) return 0;
    visAllocateParticleBuffers();
    {
        const int n3=ParticleCount*DIM; double *x0=VdX0;
        #pragma acc parallel loop present(x0[0:n3],x[0:n3])
        #pragma omp parallel for
        for(int k=0;k<n3;++k) x0[k]=x[k];
    }
    if(!VisDon) return 0;
#ifdef VIS_POU
    if(VisPou) return visPouSolve(x,b,nw,dw,target2,iterOut,res2Out);
#endif
    if(VdModelState==0){
        VdModelState=(visLoadModel() ? 1 : -1);
        if(VdModelState<0) log_printf("VIS-DeepONet: disabled for this run\n");
    }
    if(VdModelState<0) return 0;

    const double t0=donWtime();
    visAllocateNeighborBuffers();
    visAllocateModelBuffers();
    { VD_TIC(t); visBuildOperator(nw,dw); VD_TOC(VT_OPER,t); }
    if(VisDonVerifyEvery>0 && (VdRunCount==0 || VdSolveCount%VisDonVerifyEvery==0 || !VdOperatorOk)){
        VD_TIC(tv);
        const double err=visVerifyOperator(nw,dw);
        VD_TOC(VT_VERIFY,tv);
        const int ok=(isfinite(err) && err<1.0e-9);
        if(!ok || !VdOperatorOk || VdRunCount==0)
            log_printf("VIS-DeepONet: stored-operator check rel.err=%e -> %s\n",err,(ok ? "ok" : "MISMATCH, DeepONet path disabled"));
        VdOperatorOk=ok;
    }
    if(!VdOperatorOk) return 0;

    // Reuse the trunk basis of a recent solve; rows follow the particle ids.
    int full=(!VdSetupValid || VisDonSetupEvery<=1 || VdSolveCount-VdLastSetup>=VisDonSetupEvery);
    if(!full){
        const int n=ParticleCount, P=VdP;
        int *pid=VdPid, *slot=VdSlot; int mism=0, bad=0;
        #pragma acc parallel loop reduction(+:mism,bad) present(pid[0:n],ParticleIndex[0:n])
        #pragma omp parallel for reduction(+:mism,bad)
        for(int i=0;i<n;++i){ mism+=(pid[i]!=ParticleIndex[i]); bad+=(ParticleIndex[i]<0 || ParticleIndex[i]>=n); }
        if(bad) full=1;
        else if(mism){
            double *T=VdT, *Tn=VdTn;
            #pragma acc parallel loop collapse(2) present(T[0:P*n],Tn[0:P*n],slot[0:n],ParticleIndex[0:n])
            #pragma omp parallel for collapse(2)
            for(int k=0;k<P;++k)
                for(int i=0;i<n;++i) Tn[(long)k*n+i]=T[(long)k*n+slot[ParticleIndex[i]]];
            VdT=Tn; VdTn=T;
            #pragma acc parallel loop present(pid[0:n],slot[0:n],ParticleIndex[0:n])
            #pragma omp parallel for
            for(int i=0;i<n;++i){ pid[i]=ParticleIndex[i]; slot[ParticleIndex[i]]=i; }
            // a fluid particle that changed activity would carry a stale row; the
            // Galerkin coarse operator below is rebuilt from the moved rows anyway
        }
    }
    int setupOk=1;
    if(full){
        VdSetupValid=0;
        { VD_TIC(t); setupOk=visComputeGeometry(dw); VD_TOC(VT_GEOM,t); }
        if(setupOk){ VD_TIC(t); setupOk=visSetupBasis(); VD_TOC(VT_NET,t); }
        if(setupOk){
            int *pid=VdPid, *slot=VdSlot; int bad=0;
            #pragma acc parallel loop reduction(+:bad) present(pid[0:ParticleCount],slot[0:ParticleCount],ParticleIndex[0:ParticleCount])
            #pragma omp parallel for reduction(+:bad)
            for(int i=0;i<ParticleCount;++i){
                const int id=ParticleIndex[i];
                pid[i]=id;
                if(id>=0 && id<ParticleCount) slot[id]=i; else ++bad;
            }
            if(!bad){ VdSetupValid=1; VdLastSetup=VdSolveCount; }
            VdGalerkinValid=0;
        }
    }
    if(setupOk){ VD_TIC(t); setupOk=visSetupSmoother(); VD_TOC(VT_LMAX,t); }
    if(setupOk && (full || !VdGalerkinValid || VisDonGalerkinEvery<=1 || VdSolveCount-VdLastGalerkin>=VisDonGalerkinEvery)){
        VD_TIC(t); setupOk=visSetupGalerkin(); VD_TOC(VT_GALERKIN,t);
        VdGalerkinValid=setupOk; VdLastGalerkin=VdSolveCount;
    }
    if(!full) ++VdReuseCount;
    if(!setupOk){
        log_printf("VIS-DeepONet: setup failed (singular coarse operator?) -> Jacobi CG\n");
        ++VdFallbackCount;
        return 0;
    }
    const double t1=donWtime();

    int iter=0; double res2=0.0;
    int ok=visPcg(x,b,nw,dw,target2,&iter,&res2);
    if(!ok && VisDonChebDegree>0 && VdLmaxSolve!=VdSolveCount){
        // A stale lmax below the true one (viscosity rising quickly) makes the
        // Chebyshev smoother divergent and M^-1 indefinite -> CG breakdown.
        // Re-estimate lmax and retry once from the warm start.
        const int n3=ParticleCount*DIM; const double *x0=VdX0;
        #pragma acc parallel loop present(x[0:n3],x0[0:n3])
        #pragma omp parallel for
        for(int k=0;k<n3;++k) x[k]=x0[k];
        VdLmaxSolve=-1000000;
        if(visSetupSmoother()){
            int iter2=0;
            ok=visPcg(x,b,nw,dw,target2,&iter2,&res2);
            iter+=iter2;
            ++VdLmaxRetryCount;
        }
    }
    const double t2=donWtime();
    VdSetupTime+=t1-t0; VdSolveTime+=t2-t1; VdIterTotal+=iter; ++VdRunCount;
    *iterOut=iter; *res2Out=res2;
    if(!ok){
        ++VdFallbackCount;
        // keep the DeepONet iterate only if it is finite (the Jacobi CG restarts from x)
        const int n3=ParticleCount*DIM; const double *x0=VdX0; int bad=!isfinite(res2);
        #pragma acc parallel loop reduction(+:bad) present(x[0:n3])
        #pragma omp parallel for reduction(+:bad)
        for(int k=0;k<n3;++k) if(!isfinite(x[k])) ++bad;
        if(bad){
            #pragma acc parallel loop present(x[0:n3],x0[0:n3])
            #pragma omp parallel for
            for(int k=0;k<n3;++k) x[k]=x0[k];
        }
        log_printf("VIS-DeepONet: not converged (iter=%d res2/target2=%e) -> Jacobi CG fallback%s\n",
                   iter,res2/(target2>1e-300 ? target2 : 1e-300),(bad ? " from the warm start" : ""));
    }
    if(LogVerbose) log_printf("VIS-DeepONet: iter=%d res2/target2=%.3e setup=%.3fms solve=%.3fms lmax=%.4f%s | mean iter=%.2f fallback=%ld\n",
               iter,res2/(target2>1e-300 ? target2 : 1e-300),1e3*(t1-t0),1e3*(t2-t1),VdLmax,(full ? " (new basis)" : ""),
               (double)VdIterTotal/VdRunCount,VdFallbackCount);
    return ok;
}


// -----------------------------------------------------------------------------
// Training snapshot (little endian):
//   char[8] "VISDON01"
//   int32 sdim,nloc,nglob,n,ncomp(=sdim),solveIndex,solverIter,reserved
//   int64 nnz
//   double dp,dt,time,tol,abstol, glob[nglob]
//   double pos[n*sdim], xhat[n*sdim], loc[n*nloc], mass[n],
//          diag[n*ncomp], b[n*ncomp], u[n*ncomp], u0[n*ncomp]
//   int64 rowptr[n*ncomp+1]; int32 col[nnz]; double val[nnz]
// Rows/columns (a,d) -> a*ncomp+d over the fluid particles a; only the in-plane
// components (the out-of-plane equation of a 2-D run is M u_z = b_z).
// u is the converged velocity, u0 the warm start.
// -----------------------------------------------------------------------------
static void visDonMaybeExport(double *x, double *b, const double *nw, const double *dw, const int iterations)
{
    if(VisDonExportEvery<=0 || VdExportCount>=VisDonExportMax) return;
    if(VdSolveCount%VisDonExportEvery!=0) return;

    visAllocateParticleBuffers();
    visAllocateNeighborBuffers();
    visBuildOperator(nw,dw);
    if(!visComputeGeometry(dw) || VdNAct<2) return;

    const int n=ParticleCount, n3=ParticleCount*DIM; const long nb=VdNb;
    int *act=VdAct; double *X=VdX, *blk=VdBlk, *dg=VdDiag, *coef=VdCoef, *E=VdE, *x0=VdX0;
    // x, b are the solver's own work arrays: refreshing their host copies is harmless
    #pragma acc update self(x[0:n3],b[0:n3])
    const double *px=x, *pb=b;
    #pragma acc update self(act[0:n],X[0:n*DON_NX],blk[0:n*VIS_BLK],dg[0:n3],coef[0:nb],E[0:nb*DIM],x0[0:n3])
    #pragma acc update self(Mass[0:n],Position[0:n][0:DIM],NeighborCount[0:n],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount])

    const int SD=VIS_SD;
    std::vector<int> map(n,-1);
    int m=0;
    for(int i=0;i<n;++i) if(act[i]) map[i]=m++;
    std::vector<long long> rowptr((size_t)m*SD+1,0);
    std::vector<int> col; std::vector<double> val;
    col.reserve((size_t)m*SD*40); val.reserve((size_t)m*SD*40);
    for(int i=0;i<n;++i){
        if(!act[i]) continue;
        const int a=map[i];
        for(int d=0;d<SD;++d){
            for(int q=0;q<SD;++q){
                const double v=blk[(long)i*VIS_BLK+d*DIM+q];
                if(v!=0.0 || q==d){ col.push_back(a*SD+q); val.push_back(v); }
            }
            for(int jN=0;jN<NeighborCount[i];++jN){
                const long k=NeighborPtr[i]+jN;
                const double c=coef[k];
                if(c==0.0) continue;
                const int jm=map[NeighborInd[k]];
                if(jm<0) continue;
                for(int q=0;q<SD;++q){
                    const double v=-c*E[k*DIM+d]*E[k*DIM+q];
                    if(v!=0.0){ col.push_back(jm*SD+q); val.push_back(v); }
                }
            }
            rowptr[(size_t)a*SD+d+1]=(long long)col.size();
        }
    }

    visResolveExportDir();
    donMkdirP(VisDonExportDir);
    char fn[2048];
    snprintf(fn,sizeof(fn),"%s/vis_%07ld.bin",VisDonExportDir,VdSolveCount);
    FILE *fp=fopen(fn,"wb");
    if(!fp){ log_printf("VIS-DeepONet: cannot write %s\n",fn); return; }
    const int hdr[8]={DON_SDIM,DON_NLOC,DON_NGLOB,m,SD,(int)VdSolveCount,iterations,0};
    const long long nnz=(long long)col.size();
    const double dh[5]={ParticleSpacing,Dt,Time,ViscosityTolerance,ViscosityAbsoluteTolerance};
    fwrite("VISDON01",1,8,fp); fwrite(hdr,sizeof(int),8,fp); fwrite(&nnz,sizeof(long long),1,fp);
    fwrite(dh,sizeof(double),5,fp); fwrite(VdGlob,sizeof(double),DON_NGLOB,fp);
    std::vector<double> buf((size_t)m*DON_NX);
    for(int i=0,a=0;i<n;++i) if(act[i]){ for(int d=0;d<DON_SDIM;++d) buf[(size_t)a*DON_SDIM+d]=Position[i][d]; ++a; }
    fwrite(buf.data(),sizeof(double),(size_t)m*DON_SDIM,fp);
    for(int i=0,a=0;i<n;++i) if(act[i]){ for(int d=0;d<DON_SDIM;++d) buf[(size_t)a*DON_SDIM+d]=X[(long)i*DON_NX+d]; ++a; }
    fwrite(buf.data(),sizeof(double),(size_t)m*DON_SDIM,fp);
    for(int i=0,a=0;i<n;++i) if(act[i]){ for(int d=0;d<DON_NLOC;++d) buf[(size_t)a*DON_NLOC+d]=X[(long)i*DON_NX+DON_SDIM+d]; ++a; }
    fwrite(buf.data(),sizeof(double),(size_t)m*DON_NLOC,fp);
    for(int i=0,a=0;i<n;++i) if(act[i]) buf[a++]=Mass[i];
    fwrite(buf.data(),sizeof(double),m,fp);
    const double *fields[4]={dg,pb,px,x0};
    for(int f=0;f<4;++f){
        for(int i=0,a=0;i<n;++i) if(act[i]){ for(int d=0;d<SD;++d) buf[(size_t)a*SD+d]=fields[f][DIM*i+d]; ++a; }
        fwrite(buf.data(),sizeof(double),(size_t)m*SD,fp);
    }
    fwrite(rowptr.data(),sizeof(long long),rowptr.size(),fp);
    fwrite(col.data(),sizeof(int),col.size(),fp);
    fwrite(val.data(),sizeof(double),val.size(),fp);
    fclose(fp);
    ++VdExportCount;
    if(LogVerbose) log_printf("VIS-DeepONet: exported %s (n=%d nnz=%lld)\n",fn,m,nnz);
}

static void visDonAccountJacobi(const double t, const int iter)
{
    VdJacobiTime+=t;
    if(iter>0){ VdJacobiIter+=iter; ++VdJacobiRuns; }
}

static void visDonLogSummary()
{
#ifdef VIS_POU
    pouLogSummary();
#endif
    log_printf("Viscosity summary: Jacobi CG loop: runs=%ld iter=%ld (mean %.2f) time=%.3fs\n",
               VdJacobiRuns,VdJacobiIter,(VdJacobiRuns>0 ? (double)VdJacobiIter/VdJacobiRuns : 0.0),VdJacobiTime);
    if(!VisDon || VdModelState!=1) return;
    log_printf("Viscosity summary: DeepONet CG: runs=%ld mean iter=%.2f lmax-retry=%ld fallback=%ld setup=%.3fs solve=%.3fs total=%.3fs\n",
               VdRunCount,(VdRunCount>0 ? (double)VdIterTotal/VdRunCount : 0.0),VdLmaxRetryCount,VdFallbackCount,
               VdSetupTime,VdSolveTime,VdSetupTime+VdSolveTime);
    VdTm[VT_KRYLOV]=VdSolveTime-VdTm[VT_APPLY]-VdTm[VT_MATVEC];
    const double per=(VdRunCount>0 ? 1e3/VdRunCount : 0.0);
    log_printf("Viscosity summary: DeepONet time per solve [ms]:");
    for(int k=0;k<VT_COUNT;++k) log_printf(" %s=%.3f",VdTmName[k],VdTm[k]*per);
    log_printf("  (precond applies/solve=%.1f, basis reused in %ld of %ld solves)\n",(VdRunCount>0 ? (double)VdApplyCount/VdRunCount : 0.0),VdReuseCount,VdRunCount);
    if(VdJacobiRuns>0) log_printf("Viscosity summary: Jacobi time per solve [ms]: %.3f (%.4f per iteration)\n",
                                  1e3*VdJacobiTime/VdJacobiRuns,1e3*VdJacobiTime/(VdJacobiIter>0 ? VdJacobiIter : 1));
}
