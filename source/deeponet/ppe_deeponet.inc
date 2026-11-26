// =============================================================================
// ppe_deeponet.inc -- DeepONet two-level preconditioner for the ISPH PPE
//
// Included once by main_implicit.cpp (after the kernel functions, before
// main()).  Everything lives in the same translation unit as the solver, so the
// static particle arrays, applyPpeOperator() and the OpenACC data region are
// shared directly.
//
// Preconditioner (one application z = M^{-1} r, linear in r):
//
//     z0 = S(r; 0)                          pre-smoothing  (Chebyshev or Jacobi)
//     z1 = z0 + Q R^T (r - A z0)            DeepONet coarse correction
//     z  = S(r; z1)                         post-smoothing
//
// Q, R are N x P matrices built ONCE per PPE solve from per-particle MLPs:
//
//   PpeDonCoarse 1 (Galerkin, default, robust):
//       R = D^-1 T,  Q = T (T^T D^-1 A T)^-1        T = DeepONet trunk basis
//       -> the coarse operator is formed from the CURRENT matrix A, so an
//          unseen geometry only degrades the basis, never the coarse solve.
//   PpeDonCoarse 0 (pure DeepONet operator):
//       R = D^-1 Psi,  Q = (n^(2/d)/n) T K(z)        Psi = branch integral
//       encoder, K(z) = geometry-conditioned P x P mixing, no extra matvecs.
//
// The network never sees r inside a nonlinearity, so M^{-1} is a fixed linear
// operator during a solve and plain right-preconditioned BiCGStab stays valid.
// A failed DeepONet solve is detected and handed to the original Jacobi
// BiCGStab, so the tolerance PpeTolerance is always honoured.
//
// Data file keys (all optional):
//   PpeDon               0|1      enable the DeepONet preconditioner     [0]
//   PpeDonModel          path     weights from train_ppe_don.py          [ppe_deeponet.bin]
//   PpeDonCoarse         0|1      0 learned coarse, 1 Galerkin           [1]
//   PpeDonPreSmooth      0|1      pre-smoothing                          [1]
//   PpeDonPostSmooth     0|1      post-smoothing                         [1]
//   PpeDonChebDegree     k        Chebyshev smoother degree (0: Jacobi)  [10]
//   PpeDonChebRatio      q        Chebyshev interval [lmax/q, lmax]      [30]
//   PpeDonLmaxEvery      n        re-estimate lmax(D^-1 A) every n solves [20]
//   PpeDonSetupEvery     n        rebuild DeepONet basis + coarse operator every n solves;
//                                 forced whenever the particle order changes   [1]
//   PpeDonSmoothSweeps   n        Jacobi sweeps when PpeDonChebDegree 0  [1]
//   PpeDonOmega          w        Jacobi damping                         [0.7]
//   PpeDonInitialGuess   0|1      x0 += M^-1 r0 before BiCGStab          [1]
//   PpeDonVerifyEvery    n        check stored A against applyPpeOperator every n solves [500]
//   PpeDonExportEvery    n        write a training snapshot every n PPE solves (0 off)   [0]
//   PpeDonExportMax      n        stop exporting after n snapshots       [100000]
//   PpeDonExportDir      path     snapshot directory   [PPE_DON_DATA_ROOT/<case dir name>]
//                                 (makefile DATA_ROOT, default <repo>/results/trained_data)
// =============================================================================
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <chrono>

// wall clock that does not require linking OpenMP
static double donWtime()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#ifdef TWO_DIMENSIONAL
#define DON_SDIM 2
#else
#define DON_SDIM 3
#endif
#define DON_NLOC        6     // per-particle geometry descriptors
#define DON_NGLOB       8     // global descriptors
#define DON_NX          (DON_SDIM+DON_NLOC)
#define DON_MAXW        128   // max width of a per-particle MLP layer
#define DON_MAXP        128   // max basis size P
#define DON_MAX_LAYERS  8
#define DON_NMLP        5     // 0:E geometry encoder 1:G global 2:T trunk 3:R branch 4:K mixing
#define DON_TAB         (1+5*DON_MAX_LAYERS)

// Build-time defaults (makefile: make GPU=1 DATASET=1 / make GPU=1 ONET=1).
// Every value can still be overridden by the data-file keys below.
#ifdef PPE_DON_ONET
static int    PpeDon             = 1;
#else
static int    PpeDon             = 0;
#endif
#ifndef PPE_DON_MODEL_DEFAULT
#define PPE_DON_MODEL_DEFAULT "ppe_deeponet.bin"
#endif
#ifdef PPE_DON_DATASET
#ifndef PPE_DON_DATASET_EVERY
#define PPE_DON_DATASET_EVERY 50
#endif
#define PPE_DON_EXPORT_DEFAULT PPE_DON_DATASET_EVERY
#else
#define PPE_DON_EXPORT_DEFAULT 0
#endif
static int    PpeDonCoarse       = 1;
static int    PpeDonPreSmooth    = 1;
static int    PpeDonPostSmooth   = 1;
static double PpeDonOmega        = 0.7;
static int    PpeDonSmoothSweeps = 1;
static int    PpeDonChebDegree   = 10;
static double PpeDonChebRatio    = 30.0;
static int    PpeDonLmaxEvery    = 20;
static int    PpeDonSetupEvery   = 1;
static int    PpeDonInitialGuess = 1;
static int    PpeDonVerifyEvery  = 500;
static int    PpeDonExportEvery  = PPE_DON_EXPORT_DEFAULT;
static int    PpeDonExportMax    = 100000;
static double PpeDonGalerkinReg  = 1.0e-10;
static char   PpeDonModel[1024]     = PPE_DON_MODEL_DEFAULT;
#ifndef PPE_DON_DATA_ROOT
#define PPE_DON_DATA_ROOT "trained_data"
#endif
static char   PpeDonExportDir[1024] = "";   // "" -> PPE_DON_DATA_ROOT/<basename of cwd>

// ---- timing breakdown (seconds, accumulated; kernels are synchronous) ----
enum { DT_OPER, DT_VERIFY, DT_GEOM, DT_LMAX, DT_ENC, DT_HOST, DT_TRUNK, DT_AT, DT_AC, DT_INVQ, DT_SELF,
       DT_APPLY, DT_MATVEC, DT_KRYLOV, DT_COUNT };
static const char *DonTmName[DT_COUNT]={"operator","verify","geometry","lmax","encoder","host-mlp","trunk",
                                         "A*T","Ac","inv+Q","selftest","precond-apply","matvec","krylov-other"};
static double DonTm[DT_COUNT];
static long   DonApplyCount=0;
#define DON_TIC(v) const double v=donWtime()
#define DON_TOC(k,v) (DonTm[k]+=donWtime()-(v))

// ---- model ----
static int     DonModelState=0;          // 0 not loaded, 1 ok, -1 failed
static int     DonNf=0, DonHe=0, DonZ=0, DonP=0, DonNenc=0;
static double *DonW=NULL;  static long DonWCount=0;   // flat weights (host+device)
static int    *DonTabArr=NULL;                        // [DON_NMLP*DON_TAB] (host+device)
static int     DonOffB=0;                             // offset of the Fourier matrix in DonW

// ---- per-solve work arrays ----
static int     DonN=0;            // ParticleCount the buffers were allocated for
static int    *DonPid=NULL;       // ParticleIndex of the rows currently stored in Q, R
static int    *DonSlot=NULL;      // DonSlot[particle id] = row of that particle in Q, R
static int     DonSetupValid=0;
static long    DonLastSetup=-1, DonReuseCount=0;
static long    DonNbCount=0;      // NeighborIndCount DonCoef was allocated for
static int     DonPAlloc=0, DonHAlloc=0;
static int    *DonAct=NULL;       // 1: PPE unknown
static double *DonX=NULL;         // [N*DON_NX]   xhat | local descriptors
static double *DonAdiag=NULL;     // operator diagonal
static double *DonCoef=NULL;      // off-diagonal coefficients in neighbour-slot layout:  (A x)_i = adiag_i x_i - sum coef_k x_j
static double *DonP0=NULL;        // warm-start pressure (fallback restore / export)
static double *DonRc=NULL, *DonZa=NULL, *DonZb=NULL, *DonSHat=NULL, *DonDv=NULL;
static double *DonHid=NULL;       // [N*He] geometry encoder output
static double *DonT=NULL, *DonR=NULL, *DonQ=NULL, *DonAT=NULL;   // [P*N] column-major
static double *DonZvec=NULL, *DonK=NULL, *DonAc=NULL, *DonAinv=NULL, *DonC=NULL, *DonPool=NULL;

static double DonGlob[DON_NGLOB];
static double DonCenter[3], DonHalfL=1.0;
static int    DonNAct=0;
static long   DonSolveCount=0, DonExportCount=0;
static int    DonNullConst=0;      // A 1 = 0 detected at the last setup
static double *DonMeanBuf=NULL;     // [2] device scratch for the gauge projection
static int    DonOperatorOk=1;     // cleared if the stored operator disagrees with applyPpeOperator
static double DonSetupTime=0.0, DonSolveTime=0.0;
static long   DonIterTotal=0, DonFallbackCount=0, DonRunCount=0;


static int donParseDataLine(const char *buf)
{
    if(sscanf(buf," PpeDon %d",&PpeDon)==1) return 1;
    if(sscanf(buf," PpeDonModel %1023s",PpeDonModel)==1) return 1;
    if(sscanf(buf," PpeDonCoarse %d",&PpeDonCoarse)==1) return 1;
    if(sscanf(buf," PpeDonPreSmooth %d",&PpeDonPreSmooth)==1) return 1;
    if(sscanf(buf," PpeDonPostSmooth %d",&PpeDonPostSmooth)==1) return 1;
    if(sscanf(buf," PpeDonOmega %lf",&PpeDonOmega)==1) return 1;
    if(sscanf(buf," PpeDonSmoothSweeps %d",&PpeDonSmoothSweeps)==1) return 1;
    if(sscanf(buf," PpeDonChebDegree %d",&PpeDonChebDegree)==1) return 1;
    if(sscanf(buf," PpeDonChebRatio %lf",&PpeDonChebRatio)==1) return 1;
    if(sscanf(buf," PpeDonLmaxEvery %d",&PpeDonLmaxEvery)==1) return 1;
    if(sscanf(buf," PpeDonSetupEvery %d",&PpeDonSetupEvery)==1) return 1;
    if(sscanf(buf," PpeDonInitialGuess %d",&PpeDonInitialGuess)==1) return 1;
    if(sscanf(buf," PpeDonVerifyEvery %d",&PpeDonVerifyEvery)==1) return 1;
    if(sscanf(buf," PpeDonExportEvery %d",&PpeDonExportEvery)==1) return 1;
    if(sscanf(buf," PpeDonExportMax %d",&PpeDonExportMax)==1) return 1;
    if(sscanf(buf," PpeDonExportDir %1023s",PpeDonExportDir)==1) return 1;
    if(sscanf(buf," PpeDonGalerkinReg %lf",&PpeDonGalerkinReg)==1) return 1;
    return 0;
}

// mkdir -p
static void donMkdirP(const char *path)
{
    char tmp[1024]; snprintf(tmp,sizeof(tmp),"%s",path);
    for(char *q=tmp+1;*q;++q){ if(*q=='/'){ *q='\0'; mkdir(tmp,0755); *q='/'; } }
    mkdir(tmp,0755);
}

// default export directory: PPE_DON_DATA_ROOT/<name of the case (working) directory>
static void donResolveExportDir()
{
    if(PpeDonExportDir[0]) return;
    char cwd[1024]; const char *name="case";
    if(getcwd(cwd,sizeof(cwd))){ const char *sl=strrchr(cwd,'/'); if(sl && sl[1]) name=sl+1; }
    snprintf(PpeDonExportDir,sizeof(PpeDonExportDir),"%s/%s",PPE_DON_DATA_ROOT,name);
}

static void donLogSettings()
{
    donResolveExportDir();
    log_printf("PPE-DeepONet: enabled=%d model=%s coarse=%s pre=%d post=%d smoother=%s(degree/sweeps=%d ratio=%g omega=%g) initialGuess=%d verifyEvery=%d\n",
               PpeDon,PpeDonModel,(PpeDonCoarse ? "Galerkin" : "learned"),PpeDonPreSmooth,PpeDonPostSmooth,
               (PpeDonChebDegree>0 ? "Chebyshev" : "Jacobi"),(PpeDonChebDegree>0 ? PpeDonChebDegree : PpeDonSmoothSweeps),
               PpeDonChebRatio,PpeDonOmega,PpeDonInitialGuess,PpeDonVerifyEvery);
    if(PpeDonExportEvery>0)
        log_printf("PPE-DeepONet: exporting a training snapshot every %d PPE solves to %s (max %d)\n",
                   PpeDonExportEvery,PpeDonExportDir,PpeDonExportMax);
}


// -----------------------------------------------------------------------------
// MLP evaluation (shared by host and device). Activation 1 = SiLU.
// tab[m*DON_TAB] = nLayers, then per layer: in, out, act, offW, offB
// -----------------------------------------------------------------------------
#pragma acc routine seq
static void donMlp(const double *W, const int *tab, const int m, const double *in, double *out)
{
    double h0[DON_MAXW], h1[DON_MAXW];
    const int *t=tab+m*DON_TAB;
    const int nl=t[0];
    const double *x=in;
    #pragma acc loop seq
    for(int l=0;l<nl;++l){
        const int ni=t[1+5*l], no=t[2+5*l], act=t[3+5*l], oW=t[4+5*l], oB=t[5+5*l];
        double *y=(l==nl-1 ? out : ((l&1) ? h1 : h0));
        #pragma acc loop seq
        for(int o=0;o<no;++o){
            double s=W[oB+o];
            const double *w=W+oW+(long)o*ni;
            #pragma acc loop seq
            for(int k=0;k<ni;++k) s+=w[k]*x[k];
            if(act==1) s=s/(1.0+exp(-s));
            y[o]=s;
        }
        x=y;
    }
}

// enc = [xhat, loc, sin(B xhat), cos(B xhat)]
#pragma acc routine seq
static void donEncode(const double *xi, const double *B, const int nf, double *enc)
{
    #pragma acc loop seq
    for(int k=0;k<DON_NX;++k) enc[k]=xi[k];
    #pragma acc loop seq
    for(int f=0;f<nf;++f){
        double a=0.0;
        #pragma acc loop seq
        for(int d=0;d<DON_SDIM;++d) a+=B[f*DON_SDIM+d]*xi[d];
        enc[DON_NX+f]=sin(a);
        enc[DON_NX+nf+f]=cos(a);
    }
}

static void donMlpHost(const int m, const std::vector<double> &in, std::vector<double> &out)
{
    const int *t=DonTabArr+m*DON_TAB;
    std::vector<double> x(in), y;
    for(int l=0;l<t[0];++l){
        const int ni=t[1+5*l], no=t[2+5*l], act=t[3+5*l], oW=t[4+5*l], oB=t[5+5*l];
        y.assign(no,0.0);
        for(int o=0;o<no;++o){
            double s=DonW[oB+o];
            for(int k=0;k<ni;++k) s+=DonW[oW+(long)o*ni+k]*x[k];
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
#define DON_NEW(ptr,type,count) do{ ptr=(type*)malloc(sizeof(type)*(size_t)(count)); \
    if(!ptr){ log_printf("PPE-DeepONet: allocation failed\n"); exit(EXIT_FAILURE);} }while(0)

static void donAllocateParticleBuffers()
{
    if(DonN==ParticleCount && DonAct) return;
    if(DonAct){
        int *act=DonAct; double *x=DonX,*ad=DonAdiag,*p0=DonP0,*rc=DonRc,*za=DonZa,*zb=DonZb,*sh=DonSHat,*dv=DonDv; const int n=DonN;
        #pragma acc exit data delete(act[0:n],x[0:n*DON_NX],ad[0:n],p0[0:n],rc[0:n],za[0:n],zb[0:n],sh[0:n],dv[0:n])
        free(act); free(x); free(ad); free(p0); free(rc); free(za); free(zb); free(sh); free(dv);
        int *pid=DonPid, *slot=DonSlot;
        #pragma acc exit data delete(pid[0:n],slot[0:n])
        free(pid); free(slot);
    }
    DonN=ParticleCount;
    const int n=DonN;
    DON_NEW(DonAct,int,n); DON_NEW(DonX,double,(long)n*DON_NX); DON_NEW(DonAdiag,double,n); DON_NEW(DonP0,double,n);
    DON_NEW(DonRc,double,n); DON_NEW(DonZa,double,n); DON_NEW(DonZb,double,n); DON_NEW(DonSHat,double,n); DON_NEW(DonDv,double,n);
    int *act=DonAct; double *x=DonX,*ad=DonAdiag,*p0=DonP0,*rc=DonRc,*za=DonZa,*zb=DonZb,*sh=DonSHat,*dv=DonDv;
    #pragma acc enter data create(act[0:n],x[0:n*DON_NX],ad[0:n],p0[0:n],rc[0:n],za[0:n],zb[0:n],sh[0:n],dv[0:n])
    if(!DonMeanBuf){
        DON_NEW(DonMeanBuf,double,2);
        double *mb=DonMeanBuf;
        #pragma acc enter data create(mb[0:2])
    }
    DON_NEW(DonPid,int,n); DON_NEW(DonSlot,int,n);
    int *pid=DonPid, *slot=DonSlot;
    #pragma acc enter data create(pid[0:n],slot[0:n])
    DonSetupValid=0;
    DonPAlloc=0; // force model buffers to follow the new N
}

static void donAllocateNeighborBuffers()
{
    if(DonCoef && DonNbCount==NeighborIndCount) return;
    if(DonCoef){
        double *c=DonCoef; const long nb=DonNbCount;
        #pragma acc exit data delete(c[0:nb])
        free(c);
    }
    DonNbCount=NeighborIndCount;
    DON_NEW(DonCoef,double,DonNbCount);
    double *c=DonCoef; const long nb=DonNbCount;
    #pragma acc enter data create(c[0:nb])
}

static void donAllocateModelBuffers()
{
    if(DonPAlloc==DonP && DonHAlloc==DonHe && DonT) return;
    const int n=DonN;
    if(DonT){
        const int P=DonPAlloc, H=DonHAlloc;
        double *T=DonT,*R=DonR,*Q=DonQ,*AT=DonAT,*hid=DonHid,*zv=DonZvec,*K=DonK,*Ac=DonAc,*Ai=DonAinv,*c=DonC,*pool=DonPool;
        #pragma acc exit data delete(T[0:P*n],R[0:P*n],Q[0:P*n],AT[0:P*n],hid[0:n*H],zv[0:DON_MAXW],K[0:P*P],Ac[0:P*P],Ai[0:P*P],c[0:P],pool[0:H])
        free(T); free(R); free(Q); free(AT); free(DonHid); free(zv); free(K); free(Ac); free(Ai); free(c); free(pool);
    }
    const int P=DonP, H=DonHe;
    DON_NEW(DonT,double,(long)P*n); DON_NEW(DonR,double,(long)P*n); DON_NEW(DonQ,double,(long)P*n); DON_NEW(DonAT,double,(long)P*n);
    DON_NEW(DonHid,double,(long)n*H); DON_NEW(DonZvec,double,DON_MAXW); DON_NEW(DonK,double,P*P);
    DON_NEW(DonAc,double,P*P); DON_NEW(DonAinv,double,P*P); DON_NEW(DonC,double,P); DON_NEW(DonPool,double,H);
    double *T=DonT,*R=DonR,*Q=DonQ,*AT=DonAT,*hid=DonHid,*zv=DonZvec,*K=DonK,*Ac=DonAc,*Ai=DonAinv,*c=DonC,*pool=DonPool;
    #pragma acc enter data create(T[0:P*n],R[0:P*n],Q[0:P*n],AT[0:P*n],hid[0:n*H],zv[0:DON_MAXW],K[0:P*P],Ac[0:P*P],Ai[0:P*P],c[0:P],pool[0:H])
    DonPAlloc=P; DonHAlloc=H;
}


// -----------------------------------------------------------------------------
// Model file (written by train_ppe_don.py, little endian)
//   char[8] "DONPPE01"; int32 sdim,nloc,nglob,nf,He,Z,P,nmlp
//   double B[nf*sdim]
//   nmlp x { int32 nl; nl x { int32 in,out,act; double W[out*in]; double b[out] } }
// -----------------------------------------------------------------------------
static int donLoadModel()
{
    FILE *fp=fopen(PpeDonModel,"rb");
    if(!fp){ log_printf("PPE-DeepONet: cannot open model %s -> Jacobi BiCGStab only\n",PpeDonModel); return 0; }
    char magic[8]; int hdr[8];
    if(fread(magic,1,8,fp)!=8 || memcmp(magic,"DONPPE01",8)!=0 || fread(hdr,sizeof(int),8,fp)!=8){
        log_printf("PPE-DeepONet: %s is not a DONPPE01 model\n",PpeDonModel); fclose(fp); return 0;
    }
    const int sdim=hdr[0], nloc=hdr[1], nglob=hdr[2], nf=hdr[3], He=hdr[4], Z=hdr[5], P=hdr[6], nmlp=hdr[7];
    if(sdim!=DON_SDIM || nloc!=DON_NLOC || nglob!=DON_NGLOB || nmlp!=DON_NMLP || P<1 || P>DON_MAXP ||
       2*nf+DON_NX+Z>DON_MAXW || He>DON_MAXW || Z>DON_MAXW){
        log_printf("PPE-DeepONet: model dimensions (sdim=%d nloc=%d nglob=%d nf=%d He=%d Z=%d P=%d) do not match this build\n",
                   sdim,nloc,nglob,nf,He,Z,P);
        fclose(fp); return 0;
    }
    std::vector<double> w(nf*sdim);
    if(fread(w.data(),sizeof(double),w.size(),fp)!=w.size()){ fclose(fp); return 0; }
    DonTabArr=(int*)calloc(DON_NMLP*DON_TAB,sizeof(int));
    for(int m=0;m<DON_NMLP;++m){
        int nl=0;
        if(fread(&nl,sizeof(int),1,fp)!=1 || nl<1 || nl>DON_MAX_LAYERS){ fclose(fp); log_printf("PPE-DeepONet: bad layer count\n"); return 0; }
        DonTabArr[m*DON_TAB]=nl;
        for(int l=0;l<nl;++l){
            int lio[3];
            if(fread(lio,sizeof(int),3,fp)!=3){ fclose(fp); return 0; }
            const long nw=(long)lio[0]*lio[1];
            int *t=DonTabArr+m*DON_TAB+1+5*l;
            t[0]=lio[0]; t[1]=lio[1]; t[2]=lio[2]; t[3]=(int)w.size(); t[4]=(int)(w.size()+nw);
            if(m!=1 && m!=4 && lio[1]>DON_MAXW){ fclose(fp); log_printf("PPE-DeepONet: layer too wide for DON_MAXW\n"); return 0; }
            w.resize(w.size()+nw+lio[1]);
            if(fread(w.data()+t[3],sizeof(double),nw+lio[1],fp)!=(size_t)(nw+lio[1])){ fclose(fp); return 0; }
        }
    }
    fclose(fp);
    const int nenc=DON_NX+2*nf;
    const int *tE=DonTabArr+0*DON_TAB, *tG=DonTabArr+1*DON_TAB, *tT=DonTabArr+2*DON_TAB, *tR=DonTabArr+3*DON_TAB, *tK=DonTabArr+4*DON_TAB;
    const int lastE=1+5*(tE[0]-1), lastG=1+5*(tG[0]-1), lastT=1+5*(tT[0]-1), lastR=1+5*(tR[0]-1), lastK=1+5*(tK[0]-1);
    if(tE[1]!=nenc || tE[lastE+1]!=He || tG[1]!=He+DON_NGLOB || tG[lastG+1]!=Z ||
       tT[1]!=nenc+Z || tT[lastT+1]!=P || tR[1]!=nenc+Z || tR[lastR+1]!=P || tK[1]!=Z || tK[lastK+1]!=P*P){
        log_printf("PPE-DeepONet: inconsistent layer shapes in %s\n",PpeDonModel); return 0;
    }
    DonNf=nf; DonHe=He; DonZ=Z; DonP=P; DonNenc=nenc; DonOffB=0;
    DonWCount=(long)w.size();
    DON_NEW(DonW,double,DonWCount);
    memcpy(DonW,w.data(),sizeof(double)*DonWCount);
    double *W=DonW; int *tab=DonTabArr; const long nw=DonWCount;
    #pragma acc enter data copyin(W[0:nw],tab[0:DON_NMLP*DON_TAB])
    log_printf("PPE-DeepONet: loaded %s  P=%d He=%d Z=%d fourier=%d params=%ld\n",PpeDonModel,P,He,Z,nf,DonWCount);
    return 1;
}


// -----------------------------------------------------------------------------
// Stored PPE operator: exact mirror of applyPpeOperator() in neighbour-slot
// layout.  Must be kept in sync with applyPpeOperator; donVerifyOperator()
// detects any drift and disables the DeepONet path.
// -----------------------------------------------------------------------------
static void donBuildOperator()
{
#ifdef TWO_DIMENSIONAL
    const double lapCoef=4.0/(PpeLambdaP*N0p);
#else
    const double lapCoef=6.0/(PpeLambdaP*N0p);
#endif
    const int twoPhase=TwoPhaseActive;
    const double rMin=IsphMinPairRatio*ParticleSpacing;
    int *act=DonAct; double *ad=DonAdiag, *coef=DonCoef; const long nb=DonNbCount;
    #pragma acc parallel loop present(act[0:ParticleCount],ad[0:ParticleCount],coef[0:nb],IsphQmlsUse[0:ParticleCount],IsphLap[0:IsphCoeffCount],PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount],RhoEff[0:ParticleCount])
    #pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
        const long k0=NeighborPtr[iP];
        const int nn=NeighborCount[iP];
        #pragma acc loop seq
        for(int jN=0;jN<nn;++jN) coef[k0+jN]=0.0;
        const int isFluid=(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END);
        act[iP]=(isFluid && PpeFlag[iP]!=0);
        if(!act[iP]){ ad[iP]=1.0; continue; }

        const double ai=PpeAlpha[iP];
        double diag=0.0;
        if(twoPhase){
            const double rhoi=RhoEff[iP];
            #pragma acc loop seq
            for(int jN=0;jN<nn;++jN){
                const int jP=NeighborInd[k0+jN];
                if(iP==jP || !(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                const double dx=Position[jP][0]-Position[iP][0];
                const double dy=Position[jP][1]-Position[iP][1];
                const double dz=Position[jP][2]-Position[iP][2];
                const double r2=dx*dx+dy*dy+dz*dz;
                if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                const double c=isphLapW(sqrt(r2))*2.0/(rhoi+RhoEff[jP]+1.0e-30);
                diag+=c;
                if(PpeFlag[jP]!=0) coef[k0+jN]+=ai*c;
            }
        }else{
            int done=0;
#ifdef TWO_DIMENSIONAL
            if(IsphQmlsUse[iP]){
                #pragma acc loop seq
                for(int jN=0;jN<nn;++jN){
                    const long k=k0+jN;
                    const int jP=NeighborInd[k];
                    if(iP==jP || !(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                    const double dx=Position[jP][0]-Position[iP][0];
                    const double dy=Position[jP][1]-Position[iP][1];
                    const double r2=dx*dx+dy*dy;
                    if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                    const double c=IsphLap[k];
                    diag+=c;
                    if(PpeFlag[jP]!=0) coef[k]+=c;
                }
                done=1;
            }
#endif
            if(!done){
                int hasWall=0;
                #pragma acc loop seq
                for(int jN=0;jN<nn;++jN){
                    const int jP=NeighborInd[k0+jN];
                    if(!(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END)) continue;
                    const double dx=Position[jP][0]-Position[iP][0];
                    const double dy=Position[jP][1]-Position[iP][1];
                    const double dz=Position[jP][2]-Position[iP][2];
                    const double r2=dx*dx+dy*dy+dz*dz;
                    if(r2>1.0e-30 && r2<RadiusP*RadiusP) hasWall=1;
                }
                const int useHL=(ISPH_USE_HL && ai>=0.999999999 && !hasWall);
                #pragma acc loop seq
                for(int jN=0;jN<nn;++jN){
                    const int jP=NeighborInd[k0+jN];
                    if(iP==jP || !(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
                    const double dx=Position[jP][0]-Position[iP][0];
                    const double dy=Position[jP][1]-Position[iP][1];
                    const double dz=Position[jP][2]-Position[iP][2];
                    const double r2=dx*dx+dy*dy+dz*dz;
                    if(r2<=1.0e-30 || r2>=RadiusP*RadiusP) continue;
                    const double r=sqrt(r2);
                    double c;
                    if(useHL){
                        const double rEff=(r>rMin ? r : rMin);
                        c=PpeHlNorm*(d2wpdr2(r,RadiusP)-dwpdr(r,RadiusP)/rEff)/N0p;
                        if(!(c>0.0)) c=isphLapW(r);
                    }else{
                        c=isphLapW(r);
                    }
                    diag+=c;
                    if(PpeFlag[jP]!=0) coef[k0+jN]+=ai*c;
                }
            }
        }
        if(diag>1.0e-30) ad[iP]=diag;
        else{
            ad[iP]=1.0;
            #pragma acc loop seq
            for(int jN=0;jN<nn;++jN) coef[k0+jN]=0.0;
        }
    }
}

// y = A x with the stored operator (inactive rows -> 0). No null-space projection.
static void donMatvec(const double *x, double *y)
{
    const int *act=DonAct; const double *ad=DonAdiag, *coef=DonCoef; const long nb=DonNbCount;
    #pragma acc parallel loop async(1) present(x[0:ParticleCount],y[0:ParticleCount],act[0:ParticleCount],ad[0:ParticleCount],coef[0:nb],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for
    for(int i=0;i<ParticleCount;++i){
        if(!act[i]){ y[i]=0.0; continue; }
        const long k0=NeighborPtr[i];
        double s=ad[i]*x[i];
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[i];++jN) s-=coef[k0+jN]*x[NeighborInd[k0+jN]];
        y[i]=s;
    }
}

// The PPE operator exactly as applyPpeOperator() applies it, from the stored
// coefficients: (A x) with the pure-Neumann mean projection.  Cheaper than
// applyPpeOperator (no kernel/QMLS re-evaluation); equality is checked by
// donVerifyOperator and the final residual is re-evaluated with applyPpeOperator.
static void donOperatorImpl(const double *x, double *y);
static void donOperator(const double *x, double *y)
{
    DON_TIC(t); donOperatorImpl(x,y); DON_TOC(DT_MATVEC,t);
}
static void donOperatorImpl(const double *x, double *y)
{
    donMatvec(x,y);                      // queued on async(1)
    if(TwoPhaseActive && PpePureNeumann){
        #pragma acc wait(1)
        const int *act=DonAct;
        double s=0.0; int c=0;
        #pragma acc parallel loop reduction(+:s,c) present(y[0:ParticleCount],act[0:ParticleCount])
        #pragma omp parallel for reduction(+:s,c)
        for(int i=0;i<ParticleCount;++i) if(act[i]){ s+=y[i]; ++c; }
        const double m=(c>0 ? s/c : 0.0);
        #pragma acc parallel loop present(y[0:ParticleCount],act[0:ParticleCount])
        #pragma omp parallel for
        for(int i=0;i<ParticleCount;++i) if(act[i]) y[i]-=m;
    }
}

// Compare the stored operator with applyPpeOperator on a deterministic probe.
static double donVerifyOperator()
{
    const int *act=DonAct; double *za=DonZa, *zb=DonZb;
    #pragma acc parallel loop present(act[0:ParticleCount],za[0:ParticleCount],PpeOpX[0:ParticleCount])
    #pragma omp parallel for
    for(int i=0;i<ParticleCount;++i){
        const double v=(act[i] ? sin(0.7310*i+0.3)+0.5*cos(0.1137*i) : 0.0);
        za[i]=v; PpeOpX[i]=v;
    }
    applyPpeOperator();
    donOperator(za,zb);
    #pragma acc wait(1)
    double e2=0.0, y2=0.0;
    #pragma acc parallel loop reduction(+:e2,y2) present(zb[0:ParticleCount],act[0:ParticleCount],PpeOpY[0:ParticleCount])
    #pragma omp parallel for reduction(+:e2,y2)
    for(int i=0;i<ParticleCount;++i){
        if(!act[i]) continue;
        const double d=PpeOpY[i]-zb[i];
        e2+=d*d; y2+=PpeOpY[i]*PpeOpY[i];
    }
    return sqrt(e2/(y2>1.0e-300 ? y2 : 1.0e-300));
}


// -----------------------------------------------------------------------------
// Geometry descriptors (identical definition at export and inference).
//   xhat = (x - c)/L   c,L: centre and half the largest extent of the PPE unknowns
//   loc  = [alpha, divR/d, sum_fluid w/N0, sum_wall w/N0, sum_dirichlet w/N0, log(D_i/<D>)]
//   glob = [log n/10, log(ext_x/dp)/5, log(ext_y/dp)/5, log(ext_z/dp)/5, free-surface fraction,
//           pure-Neumann, log(rho_max/rho_min)/5, two-phase]
// -----------------------------------------------------------------------------
static int donComputeGeometry()
{
    const int *act=DonAct; const double *ad=DonAdiag;
    double xmn=1e300,ymn=1e300,zmn=1e300,xmx=-1e300,ymx=-1e300,zmx=-1e300, sumD=0.0, rmn=1e300, rmx=0.0;
    int nAct=0, nFluid=0, nFree=0;
    #pragma acc parallel loop reduction(min:xmn,ymn,zmn,rmn) reduction(max:xmx,ymx,zmx,rmx) reduction(+:sumD,nAct,nFluid,nFree) present(act[0:ParticleCount],ad[0:ParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount],RhoEff[0:ParticleCount])
    #pragma omp parallel for reduction(min:xmn,ymn,zmn,rmn) reduction(max:xmx,ymx,zmx,rmx) reduction(+:sumD,nAct,nFluid,nFree)
    for(int i=0;i<ParticleCount;++i){
        if(FLUID_BEGIN<=Property[i] && Property[i]<FLUID_END){ ++nFluid; if(!act[i]) ++nFree; }
        if(!act[i]) continue;
        ++nAct; sumD+=ad[i];
        const double x=Position[i][0], y=Position[i][1], z=Position[i][2];
        if(x<xmn) xmn=x; if(x>xmx) xmx=x;
        if(y<ymn) ymn=y; if(y>ymx) ymx=y;
        if(z<zmn) zmn=z; if(z>zmx) zmx=z;
        const double rh=RhoEff[i];
        if(rh<rmn) rmn=rh; if(rh>rmx) rmx=rh;
    }
    DonNAct=nAct;
    if(nAct<1) return 0;
    const double ex=xmx-xmn, ey=ymx-ymn, ez=zmx-zmn;
    double L=(ex>ey ? ex : ey);
#ifndef TWO_DIMENSIONAL
    if(ez>L) L=ez;
#endif
    DonCenter[0]=0.5*(xmn+xmx); DonCenter[1]=0.5*(ymn+ymx); DonCenter[2]=0.5*(zmn+zmx);
    DonHalfL=0.5*L+0.5*ParticleSpacing;
    const double dp=ParticleSpacing;
    DonGlob[0]=log((double)nAct)/10.0;
    DonGlob[1]=log((ex+dp)/dp)/5.0;
    DonGlob[2]=log((ey+dp)/dp)/5.0;
#ifdef TWO_DIMENSIONAL
    DonGlob[3]=0.0;
#else
    DonGlob[3]=log((ez+dp)/dp)/5.0;
#endif
    DonGlob[4]=(nFluid>0 ? (double)nFree/nFluid : 0.0);
    DonGlob[5]=(TwoPhaseActive && PpePureNeumann ? 1.0 : 0.0);
    DonGlob[6]=(TwoPhaseActive && rmn>0.0 ? log(rmx/rmn)/5.0 : 0.0);
    DonGlob[7]=(TwoPhaseActive ? 1.0 : 0.0);

#ifdef TWO_DIMENSIONAL
    const double dimRef=2.0;
#else
    const double dimRef=3.0;
#endif
    const double meanD=sumD/nAct, cx=DonCenter[0], cy=DonCenter[1], cz=DonCenter[2], iL=1.0/DonHalfL;
    double *X=DonX;
    #pragma acc parallel loop present(X[0:ParticleCount*DON_NX],act[0:ParticleCount],ad[0:ParticleCount],PpeFlag[0:ParticleCount],PpeAlpha[0:ParticleCount],PositionDivR[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],Position[0:ParticleCount][0:DIM],Property[0:ParticleCount])
    #pragma omp parallel for
    for(int i=0;i<ParticleCount;++i){
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
        double fs=0.0, ws=0.0, ds=0.0;
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
            if(FLUID_BEGIN<=Property[j] && Property[j]<FLUID_END){ fs+=w; if(PpeFlag[j]==0) ds+=w; }
            else if(WALL_BEGIN<=Property[j] && Property[j]<WALL_END) ws+=w;
        }
        double *loc=xi+DON_SDIM;
        loc[0]=PpeAlpha[i];
        loc[1]=PositionDivR[i]/dimRef;
        loc[2]=fs/N0p;
        loc[3]=ws/N0p;
        loc[4]=ds/N0p;
        loc[5]=log(ad[i]/meanD);
    }
    return 1;
}


// -----------------------------------------------------------------------------
// Per-solve setup of Q, R.  Returns 0 if the coarse space cannot be used.
// -----------------------------------------------------------------------------
static int donInvertHost(const int P, std::vector<double> &a, std::vector<double> &inv)
{
    std::vector<double> m(2*P*P);
    for(int i=0;i<P;++i) for(int j=0;j<P;++j){ m[i*2*P+j]=a[i*P+j]; m[i*2*P+P+j]=(i==j); }
    for(int c=0;c<P;++c){
        int piv=c; double amax=fabs(m[c*2*P+c]);
        for(int r=c+1;r<P;++r) if(fabs(m[r*2*P+c])>amax){ amax=fabs(m[r*2*P+c]); piv=r; }
        if(!(amax>1.0e-300) || !isfinite(amax)) return 0;
        if(piv!=c) for(int j=0;j<2*P;++j){ const double t=m[c*2*P+j]; m[c*2*P+j]=m[piv*2*P+j]; m[piv*2*P+j]=t; }
        const double d=1.0/m[c*2*P+c];
        for(int j=0;j<2*P;++j) m[c*2*P+j]*=d;
        for(int r=0;r<P;++r){
            if(r==c) continue;
            const double f=m[r*2*P+c];
            if(f!=0.0) for(int j=0;j<2*P;++j) m[r*2*P+j]-=f*m[c*2*P+j];
        }
    }
    inv.resize(P*P);
    for(int i=0;i<P;++i) for(int j=0;j<P;++j) inv[i*P+j]=m[i*2*P+P+j];
    return 1;
}

// Q_ik = scale * sum_l T_il M_lk  (M = K or Ac^-1). Written as a collapsed
// (k,i) loop without a per-thread array: the equivalent per-particle loop with
// a local copy of T_i. and an inner reduction is silently miscompiled by
// nvc++ 23.x/24.1 at -O1 and above (the stores to Q are dropped).
static void donFormQ(const double *M, const double scale)
{
    const int n=ParticleCount, P=DonP;
    const double *T=DonT; double *Q=DonQ;
    #pragma acc parallel loop collapse(2) present(T[0:P*n],Q[0:P*n],M[0:P*P])
    #pragma omp parallel for collapse(2)
    for(int k=0;k<P;++k){
        for(int i=0;i<n;++i){
            double s=0.0;
            #pragma acc loop seq
            for(int l=0;l<P;++l) s+=T[(long)l*n+i]*M[l*P+k];
            Q[(long)k*n+i]=scale*s;
        }
    }
}

// One-time check of the device inference against the host MLP (guards against
// compiler problems like the one described at donFormQ).
static int DonSelfTestDone=0;
static int donSelfTest()
{
    if(DonSelfTestDone) return DonSelfTestDone>0;
    const int n=ParticleCount, P=DonP, Z=DonZ, nf=DonNf;
    double *T=DonT, *Q=DonQ, *X=DonX, *K=DonK, *Ai=DonAinv; int *act=DonAct;
    #pragma acc update self(T[0:P*n],Q[0:P*n],X[0:n*DON_NX],act[0:n])
    const int learned=(PpeDonCoarse==0);
#ifdef TWO_DIMENSIONAL
    const double qScale=learned ? (double)DonNAct/DonNAct : 1.0;
#else
    const double qScale=learned ? pow((double)DonNAct,2.0/3.0)/DonNAct : 1.0;
#endif
    const double *M=(learned ? K : Ai);
    double *mean=DonC;                       // basis means removed for a closed domain
    if(!learned && DonNullConst){
        #pragma acc update self(mean[0:P])
    }
    double eT=0.0, nT=0.0, eQ=0.0, nQ=0.0;
    int tested=0;
    for(int i=0;i<n && tested<64;i+=1+n/64){
        if(!act[i]) continue;
        std::vector<double> inp(DON_MAXW), t;
        donEncode(X+(long)i*DON_NX,DonW+DonOffB,nf,inp.data());
        for(int k=0;k<Z;++k) inp[DON_NX+2*nf+k]=DonZvec[k];
        inp.resize(DON_NX+2*nf+Z);
        donMlpHost(2,inp,t);
        if(!learned && DonNullConst) for(int k=0;k<P;++k) t[k]-=mean[k];
        for(int k=0;k<P;++k){
            const double d=t[k]-T[(long)k*n+i]; eT+=d*d; nT+=t[k]*t[k];
            double q=0.0;
            for(int l=0;l<P;++l) q+=T[(long)l*n+i]*M[l*P+k];
            q*=qScale;
            const double e=q-Q[(long)k*n+i]; eQ+=e*e; nQ+=q*q;
        }
        ++tested;
    }
    const double rT=sqrt(eT/(nT+1e-300)), rQ=sqrt(eQ/(nQ+1e-300));
    const int ok=(tested>0 && rT<1.0e-10 && rQ<1.0e-10);
    log_printf("PPE-DeepONet: device inference self-test on %d particles: trunk err=%e coarse err=%e -> %s\n",
               tested,rT,rQ,(ok ? "ok" : "FAILED, DeepONet path disabled"));
    DonSelfTestDone=(ok ? 1 : -1);
    if(!ok) DonModelState=-1;
    return ok;
}

static int donSetupModel()
{
    const int n=ParticleCount, P=DonP, He=DonHe, Z=DonZ, nf=DonNf;
    const int *act=DonAct; const double *ad=DonAdiag, *X=DonX, *W=DonW; const int *tab=DonTabArr;
    const long nw=DonWCount;
    double *hid=DonHid, *pool=DonPool, *zv=DonZvec, *K=DonK, *T=DonT, *R=DonR, *Q=DonQ;
    const int offB=DonOffB;

    // 1) geometry encoder + mean pooling over the PPE unknowns
    DON_TIC(tEnc);
    #pragma acc parallel loop present(W[0:nw],tab[0:DON_NMLP*DON_TAB],X[0:n*DON_NX],act[0:n],hid[0:n*He])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        double *h=hid+(long)i*He;
        if(!act[i]){
            #pragma acc loop seq
            for(int k=0;k<He;++k) h[k]=0.0;
            continue;
        }
        double enc[DON_MAXW];
        donEncode(X+(long)i*DON_NX,W+offB,nf,enc);
        donMlp(W,tab,0,enc,h);
    }
    const double invN=1.0/DonNAct;
    #pragma acc parallel loop gang present(hid[0:n*He],pool[0:He])
    #pragma omp parallel for
    for(int k=0;k<He;++k){
        double s=0.0;
        #pragma acc loop vector reduction(+:s)
        for(int i=0;i<n;++i) s+=hid[(long)i*He+k];
        pool[k]=s*invN;
    }
    #pragma acc update self(pool[0:He])

    DON_TOC(DT_ENC,tEnc);
    // 2) global latent z and mixing matrix K (host, tiny)
    DON_TIC(tHost);
    std::vector<double> gin(He+DON_NGLOB), z, kv;
    for(int k=0;k<He;++k) gin[k]=pool[k];
    for(int k=0;k<DON_NGLOB;++k) gin[He+k]=DonGlob[k];
    donMlpHost(1,gin,z);
    donMlpHost(4,z,kv);
    for(int k=0;k<Z;++k) zv[k]=z[k];
    for(int k=0;k<P*P;++k) K[k]=kv[k];
    for(int k=0;k<Z;++k) if(!isfinite(zv[k])) return 0;
    #pragma acc update device(zv[0:Z],K[0:P*P])

    DON_TOC(DT_HOST,tHost);
    // 3) trunk T and (learned mode) branch Psi, Q
    DON_TIC(tTrunk);
    const int learned=(PpeDonCoarse==0);
#ifdef TWO_DIMENSIONAL
    const double amp=(double)DonNAct;                       // n^(2/d), d=2
#else
    const double amp=pow((double)DonNAct,2.0/3.0);
#endif
    const double qScale=amp*invN;
    #pragma acc parallel loop present(W[0:nw],tab[0:DON_NMLP*DON_TAB],X[0:n*DON_NX],act[0:n],ad[0:n],zv[0:Z],K[0:P*P],T[0:P*n],R[0:P*n],Q[0:P*n])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(!act[i]){
            #pragma acc loop seq
            for(int k=0;k<P;++k){ T[(long)k*n+i]=0.0; R[(long)k*n+i]=0.0; Q[(long)k*n+i]=0.0; }
            continue;
        }
        double inp[DON_MAXW], t[DON_MAXP];
        donEncode(X+(long)i*DON_NX,W+offB,nf,inp);
        #pragma acc loop seq
        for(int k=0;k<Z;++k) inp[DON_NX+2*nf+k]=zv[k];
        donMlp(W,tab,2,inp,t);
        const double iD=1.0/ad[i];
        #pragma acc loop seq
        for(int k=0;k<P;++k) T[(long)k*n+i]=t[k];
        if(learned) donMlp(W,tab,3,inp,t);
        #pragma acc loop seq
        for(int k=0;k<P;++k) R[(long)k*n+i]=t[k]*iD;
    }
    DON_TOC(DT_TRUNK,tTrunk);
    if(learned){
        donFormQ(K,qScale);
        DON_TIC(tSelf); const int ok=donSelfTest(); DON_TOC(DT_SELF,tSelf);
        return ok;
    }

    // 4) Galerkin coarse operator  Ac = (D^-1 T)^T A T,  Q = T Ac^-1
    DON_TIC(tAT);
    // Closed domain without Dirichlet rows (A 1 = 0, pressure gauge): remove the
    // D^-1-weighted mean of every basis vector so that the constant is not in the
    // coarse space; otherwise Ac is singular and the coarse correction blows up.
    {
        double *one=DonZa, *y=DonZb;
        #pragma acc parallel loop present(one[0:n],act[0:n])
        #pragma omp parallel for
        for(int i=0;i<n;++i) one[i]=(act[i] ? 1.0 : 0.0);
        donMatvec(one,y);
        #pragma acc wait(1)
        double ymax=0.0, dmax=0.0, wsum=0.0;
        #pragma acc parallel loop reduction(max:ymax,dmax) reduction(+:wsum) present(y[0:n],ad[0:n],act[0:n])
        #pragma omp parallel for reduction(max:ymax,dmax) reduction(+:wsum)
        for(int i=0;i<n;++i){
            if(!act[i]) continue;
            const double ay=fabs(y[i]);
            if(ay>ymax) ymax=ay;
            if(ad[i]>dmax) dmax=ad[i];
            wsum+=1.0/ad[i];
        }
        DonNullConst=(ymax<=1.0e-10*dmax);
        if(DonNullConst){
            double *m=DonC;
            #pragma acc parallel loop gang present(T[0:P*n],ad[0:n],act[0:n],m[0:P])
            #pragma omp parallel for
            for(int k=0;k<P;++k){
                double sk=0.0;
                #pragma acc loop vector reduction(+:sk)
                for(int i=0;i<n;++i) if(act[i]) sk+=T[(long)k*n+i]/ad[i];
                m[k]=sk/wsum;
            }
            #pragma acc parallel loop collapse(2) present(T[0:P*n],R[0:P*n],ad[0:n],act[0:n],m[0:P])
            #pragma omp parallel for collapse(2)
            for(int k=0;k<P;++k){
                for(int i=0;i<n;++i){
                    if(!act[i]) continue;
                    const double t=T[(long)k*n+i]-m[k];
                    T[(long)k*n+i]=t;
                    R[(long)k*n+i]=t/ad[i];
                }
            }
        }
    }
    double *AT=DonAT, *Ac=DonAc, *Ai=DonAinv; const double *coef=DonCoef; const long nb=DonNbCount;
    #pragma acc parallel loop present(T[0:P*n],AT[0:P*n],act[0:n],ad[0:n],coef[0:nb],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(!act[i]){
            #pragma acc loop seq
            for(int k=0;k<P;++k) AT[(long)k*n+i]=0.0;
            continue;
        }
        double s[DON_MAXP];
        #pragma acc loop seq
        for(int k=0;k<P;++k) s[k]=ad[i]*T[(long)k*n+i];
        const long k0=NeighborPtr[i];
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[i];++jN){
            const double c=coef[k0+jN];
            if(c==0.0) continue;
            const int j=NeighborInd[k0+jN];
            #pragma acc loop seq
            for(int k=0;k<P;++k) s[k]-=c*T[(long)k*n+j];
        }
        #pragma acc loop seq
        for(int k=0;k<P;++k) AT[(long)k*n+i]=s[k];
    }
    DON_TOC(DT_AT,tAT);
    DON_TIC(tAc);
    #pragma acc parallel loop gang collapse(2) present(R[0:P*n],AT[0:P*n],Ac[0:P*P])
    #pragma omp parallel for collapse(2)
    for(int k=0;k<P;++k){
        for(int l=0;l<P;++l){
            double s=0.0;
            #pragma acc loop vector reduction(+:s)
            for(int i=0;i<n;++i) s+=R[(long)k*n+i]*AT[(long)l*n+i];
            Ac[k*P+l]=s;
        }
    }
    #pragma acc update self(Ac[0:P*P])
    DON_TOC(DT_AC,tAc);
    DON_TIC(tInv);
    std::vector<double> a(Ac,Ac+P*P), inv;
    double dmax=0.0;
    for(int k=0;k<P;++k) if(fabs(a[k*P+k])>dmax) dmax=fabs(a[k*P+k]);
    if(!(dmax>0.0) || !isfinite(dmax)) return 0;
    for(int k=0;k<P;++k) a[k*P+k]+=PpeDonGalerkinReg*dmax;
    if(!donInvertHost(P,a,inv)) return 0;
    for(int k=0;k<P*P;++k){ if(!isfinite(inv[k])) return 0; Ai[k]=inv[k]; }
    #pragma acc update device(Ai[0:P*P])
    donFormQ(Ai,1.0);
    DON_TOC(DT_INVQ,tInv);
    DON_TIC(tSelf); const int ok=donSelfTest(); DON_TOC(DT_SELF,tSelf);
    return ok;
}


// -----------------------------------------------------------------------------
// Smoother: PpeDonChebDegree k>0 -> Jacobi-preconditioned Chebyshev of degree k
// on [lmax/PpeDonChebRatio, lmax] of D^-1 A; k=0 -> PpeDonSmoothSweeps damped
// Jacobi sweeps.  Every step has the form
//     d   = c1 d + c2 D^-1 (r - A src)
//     dst = src + d
// with host coefficients (c1,c2); identical to Smoother in ppe_don_model.py.
// -----------------------------------------------------------------------------
#define DON_MAX_STEPS 32
static int    DonNSteps=0;
static double DonC1[DON_MAX_STEPS], DonC2[DON_MAX_STEPS];
static double DonLmax=0.0;
static long   DonLmaxSolve=-1000000;

// Largest eigenvalue of D^-1 A by power iteration (Rayleigh quotient, x1.15:
// 15 iterations under-estimate by up to ~2%).
static double donEstimateLmax()
{
    const int n=ParticleCount; const int *act=DonAct; const double *ad=DonAdiag;
    double *x=DonZa, *y=DonZb;
    #pragma acc parallel loop present(x[0:n],act[0:n])
    #pragma omp parallel for
    for(int i=0;i<n;++i) x[i]=(act[i] ? sin(0.731*i+0.3)+0.5*cos(0.1137*i) : 0.0);
    double lam=1.0;
    for(int it=0;it<15;++it){
        donMatvec(x,y);
        #pragma acc wait(1)
        double xy=0.0, xx=0.0, yy=0.0;
        #pragma acc parallel loop reduction(+:xy,xx,yy) present(x[0:n],y[0:n],ad[0:n])
        #pragma omp parallel for reduction(+:xy,xx,yy)
        for(int i=0;i<n;++i){ y[i]/=ad[i]; xy+=x[i]*y[i]; xx+=x[i]*x[i]; yy+=y[i]*y[i]; }
        lam=xy/(xx>1e-300 ? xx : 1e-300);
        const double inv=1.0/sqrt(yy>1e-300 ? yy : 1e-300);
        #pragma acc parallel loop present(x[0:n],y[0:n])
        #pragma omp parallel for
        for(int i=0;i<n;++i) x[i]=y[i]*inv;
    }
    return 1.15*lam;
}

static int donSetupSmoother()
{
    if(PpeDonChebDegree>0){
        if(DonSolveCount-DonLmaxSolve>=PpeDonLmaxEvery || !(DonLmax>0.0)){
            DonLmax=donEstimateLmax();
            DonLmaxSolve=DonSolveCount;
            if(!(DonLmax>0.0) || !isfinite(DonLmax)) return 0;
        }
        const int k=(PpeDonChebDegree<DON_MAX_STEPS ? PpeDonChebDegree : DON_MAX_STEPS);
        const double b=DonLmax, a=b/PpeDonChebRatio;
        const double theta=0.5*(b+a), delta=0.5*(b-a), sigma=theta/delta;
        double rho=1.0/sigma;
        DonC1[0]=0.0; DonC2[0]=1.0/theta;
        for(int j=1;j<k;++j){
            const double rho1=1.0/(2.0*sigma-rho);
            DonC1[j]=rho1*rho; DonC2[j]=2.0*rho1/delta;
            rho=rho1;
        }
        DonNSteps=k;
    }else{
        const int k=(PpeDonSmoothSweeps<DON_MAX_STEPS ? PpeDonSmoothSweeps : DON_MAX_STEPS);
        for(int j=0;j<k;++j){ DonC1[j]=0.0; DonC2[j]=PpeDonOmega; }
        DonNSteps=k;
    }
    return 1;
}

// One fused smoothing step. hasSrc=0 means src = 0 (src must still be a valid array).
static void donSmoothStep(const double *r, const double *src, const int hasSrc, double *dst, const double c1, const double c2)
{
    const int n=ParticleCount;
    const int *act=DonAct; const double *ad=DonAdiag, *coef=DonCoef; const long nb=DonNbCount;
    double *d=DonDv;
    #pragma acc parallel loop async(1) present(r[0:n],src[0:n],dst[0:n],d[0:n],act[0:n],ad[0:n],coef[0:nb],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(!act[i]){ dst[i]=0.0; d[i]=0.0; continue; }
        double res=r[i], zi=0.0;
        if(hasSrc){
            zi=src[i];
            const long k0=NeighborPtr[i];
            double az=ad[i]*zi;
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[i];++jN) az-=coef[k0+jN]*src[NeighborInd[k0+jN]];
            res-=az;
        }
        const double di=(c1!=0.0 ? c1*d[i] : 0.0)+c2*res/ad[i];
        d[i]=di;
        dst[i]=zi+di;
    }
}

// -----------------------------------------------------------------------------
// z = M^{-1} r   two-level cycle, linear in r:
//     pre-smooth from 0, z += Q R^T (r - A z), post-smooth.
// out must not be DonRc/DonZa/DonZb/DonDv.
// -----------------------------------------------------------------------------
static void donApplyImpl(const double *in, double *out);
static void donApply(const double *in, double *out)
{
    DON_TIC(t); donApplyImpl(in,out); DON_TOC(DT_APPLY,t); ++DonApplyCount;
}
static void donApplyImpl(const double *in, double *out)
{
    const int n=ParticleCount, P=DonP;
    const int *act=DonAct; const double *ad=DonAdiag, *coef=DonCoef, *R=DonR, *Q=DonQ; const long nb=DonNbCount;
    double *rc=DonRc, *za=DonZa, *zb=DonZb, *c=DonC;
    const int nS=DonNSteps;
    const int pre=(PpeDonPreSmooth && nS>0), post=(PpeDonPostSmooth && nS>0);

    // pre-smoothing
    double *z=NULL;
    if(pre){
        for(int s=0;s<nS;++s){
            double *dst=(z==za ? zb : za);
            donSmoothStep(in,(z ? z : in),(z!=NULL),dst,DonC1[s],DonC2[s]);
            z=dst;
        }
    }
    // coarse residual rc = in - A z
    const double *rin=in;
    if(z){
        const double *zc=z;
        #pragma acc parallel loop async(1) present(in[0:n],zc[0:n],rc[0:n],act[0:n],ad[0:n],coef[0:nb],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
        #pragma omp parallel for
        for(int i=0;i<n;++i){
            if(!act[i]){ rc[i]=0.0; continue; }
            const long k0=NeighborPtr[i];
            double az=ad[i]*zc[i];
            #pragma acc loop seq
            for(int jN=0;jN<NeighborCount[i];++jN) az-=coef[k0+jN]*zc[NeighborInd[k0+jN]];
            rc[i]=in[i]-az;
        }
        rin=rc;
    }
    // c = R^T rc
    #pragma acc parallel loop gang async(1) present(R[0:P*n],rin[0:n],c[0:P])
    #pragma omp parallel for
    for(int k=0;k<P;++k){
        double s=0.0;
        #pragma acc loop vector reduction(+:s)
        for(int i=0;i<n;++i) s+=R[(long)k*n+i]*rin[i];
        c[k]=s;
    }
    // z <- z + Q c
    double *dst=(post ? (z==za ? zb : za) : out);
    {
        const double *zs=(z ? z : in);
        const double hz=(z ? 1.0 : 0.0);
        #pragma acc parallel loop async(1) present(zs[0:n],dst[0:n],Q[0:P*n],c[0:P],act[0:n])
        #pragma omp parallel for
        for(int i=0;i<n;++i){
            if(!act[i]){ dst[i]=0.0; continue; }
            double s=hz*zs[i];
            #pragma acc loop seq
            for(int k=0;k<P;++k) s+=Q[(long)k*n+i]*c[k];
            dst[i]=s;
        }
    }
    z=dst;
    // post-smoothing
    if(post){
        for(int s=0;s<nS;++s){
            double *d2=(s==nS-1 ? out : (z==za ? zb : za));
            donSmoothStep(in,z,1,d2,DonC1[s],DonC2[s]);
            z=d2;
        }
    }
    if(DonNullConst){
        // Closed domain: the smoother amplifies the constant (gauge) mode, which
        // then accumulates in the iterate and destroys the attainable accuracy of
        // A x by cancellation. Remove the mean on the device (no host sync).
        double *mb=DonMeanBuf;
        #pragma acc parallel loop gang async(1) present(out[0:n],act[0:n],mb[0:2])
        for(int g=0;g<1;++g){
            double sm=0.0, cnt=0.0;
            #pragma acc loop vector reduction(+:sm,cnt)
            for(int i=0;i<n;++i) if(act[i]){ sm+=out[i]; cnt+=1.0; }
            mb[0]=sm/(cnt>0.0 ? cnt : 1.0);
        }
        #pragma acc parallel loop async(1) present(out[0:n],act[0:n],mb[0:2])
        for(int i=0;i<n;++i) if(act[i]) out[i]-=mb[0];
    }
    #pragma acc wait(1)
}


// -----------------------------------------------------------------------------
// Right-preconditioned BiCGStab, same recurrences/termination as the Jacobi
// solver in calculatePpe, with M^{-1} = donApply.  Uses PpeR/PpeR0 (initial
// residual), PressureP (iterate) and the solver work arrays.
// Returns 1 if converged (*rrOut <= tol^2 bb).
// -----------------------------------------------------------------------------
static int donBiCGStab(const double bb, double *rrIO, int *iterOut, int *breakdownOut)
{
    const int n=ParticleCount;
    const double tol2bb=PpeTolerance*PpeTolerance*bb;
    double rr=*rrIO;
    int iter=0, breakdown=0;
    double *shat=DonSHat;

    if(PpeDonInitialGuess){
        // x1 = x0 + M^-1 r0, kept only if it lowers the residual
        donApply(PpeR,shat);
        #pragma acc parallel loop async(1) present(PpeOpX[0:n],PressureP[0:n],shat[0:n],PpeFlag[0:n])
        #pragma omp parallel for
        for(int i=0;i<n;++i) PpeOpX[i]=(PpeFlag[i]!=0 ? PressureP[i]+shat[i] : 0.0);
        donOperator(PpeOpX,PpeOpY);
        double rr1=0.0;
        #pragma acc wait(1)
        #pragma acc parallel loop reduction(+:rr1) present(PpeS[0:n],PpeB[0:n],PpeOpY[0:n],PpeFlag[0:n])
        #pragma omp parallel for reduction(+:rr1)
        for(int i=0;i<n;++i){
            const double r=(PpeFlag[i]!=0 ? PpeB[i]-PpeOpY[i] : 0.0);
            PpeS[i]=r; rr1+=r*r;
        }
        if(isfinite(rr1) && rr1<rr){
            #pragma acc parallel loop async(1) present(PressureP[0:n],PpeOpX[0:n],PpeR[0:n],PpeR0[0:n],PpeS[0:n],PpeFlag[0:n])
            #pragma omp parallel for
            for(int i=0;i<n;++i){
                if(PpeFlag[i]!=0) PressureP[i]=PpeOpX[i];
                PpeR[i]=PpeS[i]; PpeR0[i]=PpeS[i];
            }
            rr=rr1;
        }
    }

    double rhoOld=1.0, alpha=1.0, omega=1.0;
    if(rr>tol2bb){
        for(iter=0;iter<PpeMaxIteration;++iter){
            double rhoNew=0.0;
            #pragma acc wait(1)
            #pragma acc parallel loop reduction(+:rhoNew) present(PpeR0[0:n],PpeR[0:n])
            #pragma omp parallel for reduction(+:rhoNew)
            for(int i=0;i<n;++i) rhoNew+=PpeR0[i]*PpeR[i];
            if(!(fabs(rhoNew)>=1.0e-60) || !isfinite(rhoNew)){ breakdown=1; break; }
            const double beta=(rhoNew/rhoOld)*(alpha/omega);

            #pragma acc parallel loop async(1) present(PpeP[0:n],PpeR[0:n],PpeV[0:n],PpeFlag[0:n])
            #pragma omp parallel for
            for(int i=0;i<n;++i) PpeP[i]=(PpeFlag[i]!=0 ? PpeR[i]+beta*(PpeP[i]-omega*PpeV[i]) : 0.0);
            donApply(PpeP,PpePHat);
            donOperator(PpePHat,PpeOpY);

            double r0v=0.0;
            #pragma acc wait(1)
            #pragma acc parallel loop reduction(+:r0v) present(PpeR0[0:n],PpeV[0:n],PpeOpY[0:n])
            #pragma omp parallel for reduction(+:r0v)
            for(int i=0;i<n;++i){ PpeV[i]=PpeOpY[i]; r0v+=PpeR0[i]*PpeOpY[i]; }
            if(!(fabs(r0v)>=1.0e-60) || !isfinite(r0v)){ breakdown=1; break; }
            alpha=rhoNew/r0v;

            double ss=0.0;
            #pragma acc wait(1)
            #pragma acc parallel loop reduction(+:ss) present(PpeS[0:n],PpeR[0:n],PpeV[0:n],PpeFlag[0:n])
            #pragma omp parallel for reduction(+:ss)
            for(int i=0;i<n;++i){
                PpeS[i]=(PpeFlag[i]!=0 ? PpeR[i]-alpha*PpeV[i] : 0.0);
                ss+=PpeS[i]*PpeS[i];
            }
            if(ss<=tol2bb){
                #pragma acc parallel loop async(1) present(PressureP[0:n],PpePHat[0:n],PpeFlag[0:n])
                #pragma omp parallel for
                for(int i=0;i<n;++i) if(PpeFlag[i]!=0) PressureP[i]+=alpha*PpePHat[i];
                rr=ss; ++iter; break;
            }

            donApply(PpeS,shat);
            donOperator(shat,PpeOpY);

            double ts=0.0, tt=0.0;
            #pragma acc wait(1)
            #pragma acc parallel loop reduction(+:ts,tt) present(PpeT[0:n],PpeS[0:n],PpeOpY[0:n])
            #pragma omp parallel for reduction(+:ts,tt)
            for(int i=0;i<n;++i){ PpeT[i]=PpeOpY[i]; ts+=PpeOpY[i]*PpeS[i]; tt+=PpeOpY[i]*PpeOpY[i]; }
            if(!(tt>1.0e-60) || !isfinite(tt)){ breakdown=1; break; }
            omega=ts/tt;
            if(fabs(omega)<1.0e-14){ breakdown=1; break; }

            rr=0.0;
            #pragma acc wait(1)
            #pragma acc parallel loop reduction(+:rr) present(PressureP[0:n],PpePHat[0:n],shat[0:n],PpeS[0:n],PpeT[0:n],PpeR[0:n],PpeFlag[0:n])
            #pragma omp parallel for reduction(+:rr)
            for(int i=0;i<n;++i){
                if(PpeFlag[i]!=0){
                    PressureP[i]+=alpha*PpePHat[i]+omega*shat[i];
                    PpeR[i]=PpeS[i]-omega*PpeT[i];
                }else{
                    PressureP[i]=0.0; PpeR[i]=0.0;
                }
                rr+=PpeR[i]*PpeR[i];
            }
            if(!isfinite(rr)){ breakdown=1; break; }
            if(rr<=tol2bb){ ++iter; break; }
            rhoOld=rhoNew;
        }
    }
    if(isfinite(rr) && rr<=tol2bb){
        // true residual with applyPpeOperator (guards recursion drift and any
        // operator mismatch); the Jacobi loop finishes the solve if needed
        #pragma acc parallel loop async(1) present(PpeOpX[0:n],PressureP[0:n],PpeFlag[0:n])
        #pragma omp parallel for
        for(int i=0;i<n;++i) PpeOpX[i]=(PpeFlag[i]!=0 ? PressureP[i] : 0.0);
        #pragma acc wait(1)
        applyPpeOperator();
        double rt=0.0;
        #pragma acc wait(1)
        #pragma acc parallel loop reduction(+:rt) present(PpeB[0:n],PpeOpY[0:n],PpeR[0:n],PpeR0[0:n],PpeFlag[0:n])
        #pragma omp parallel for reduction(+:rt)
        for(int i=0;i<n;++i){
            const double r=(PpeFlag[i]!=0 ? PpeB[i]-PpeOpY[i] : 0.0);
            PpeR[i]=r; PpeR0[i]=r; rt+=r*r;
        }
        rr=rt;
    }
    #pragma acc wait(1)
    *rrIO=rr; *iterOut=iter; *breakdownOut=breakdown;
    return (isfinite(rr) && rr<=tol2bb);
}

// Reset the iterate to the warm start (if it went non-finite) and rebuild the
// residual so that the original Jacobi BiCGStab can take over.
static double donPrepareFallback(const double rrDon, const double rrStart)
{
    const int n=ParticleCount; const double *p0=DonP0;
    int bad=(!isfinite(rrDon) || !(rrDon<rrStart));
    #pragma acc parallel loop reduction(+:bad) present(PressureP[0:n])
    #pragma omp parallel for reduction(+:bad)
    for(int i=0;i<n;++i) if(!isfinite(PressureP[i])) ++bad;
    #pragma acc parallel loop present(PressureP[0:n],p0[0:n],PpeOpX[0:n],PpeFlag[0:n],PpeP[0:n],PpeV[0:n])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(bad) PressureP[i]=p0[i];
        PpeOpX[i]=(PpeFlag[i]!=0 ? PressureP[i] : 0.0);
        PpeP[i]=0.0; PpeV[i]=0.0;
    }
    applyPpeOperator();
    double rr=0.0;
    #pragma acc parallel loop reduction(+:rr) present(PpeR[0:n],PpeR0[0:n],PpeB[0:n],PpeOpY[0:n],PpeFlag[0:n])
    #pragma omp parallel for reduction(+:rr)
    for(int i=0;i<n;++i){
        const double r=(PpeFlag[i]!=0 ? PpeB[i]-PpeOpY[i] : 0.0);
        PpeR[i]=r; PpeR0[i]=r; rr+=r*r;
    }
    return rr;
}


// -----------------------------------------------------------------------------
// Hook called by calculatePpe() after PpeB, PpeR, PpeR0 and the warm start are
// ready.  On return either rr <= tol^2 bb (converged, the Jacobi loop is
// skipped) or PpeR/PpeR0/PressureP are a valid starting state for the Jacobi loop.
// -----------------------------------------------------------------------------
static void donSolvePpe(const double bb, double *rrIO, int *donIter)
{
    *donIter=0;
    ++DonSolveCount;
    if(!PpeDon && PpeDonExportEvery<=0) return;

    donAllocateParticleBuffers();
    {
        const int n=ParticleCount; double *p0=DonP0;
        #pragma acc parallel loop present(p0[0:n],PressureP[0:n])
        #pragma omp parallel for
        for(int i=0;i<n;++i) p0[i]=PressureP[i];
    }
    if(!PpeDon || !(bb>1.0e-60) || *rrIO<=PpeTolerance*PpeTolerance*bb) return;
    if(DonModelState==0){
        DonModelState=(donLoadModel() ? 1 : -1);
        if(DonModelState<0) log_printf("PPE-DeepONet: disabled for this run\n");
    }
    if(DonModelState<0) return;

    const double t0=donWtime();
    donAllocateNeighborBuffers();
    donAllocateModelBuffers();
    { DON_TIC(t); donBuildOperator(); DON_TOC(DT_OPER,t); }
    if(PpeDonVerifyEvery>0 && (DonSolveCount==1 || DonSolveCount%PpeDonVerifyEvery==0 || !DonOperatorOk)){
        DON_TIC(tv);
        const double err=donVerifyOperator();
        DON_TOC(DT_VERIFY,tv);
        const int ok=(isfinite(err) && err<1.0e-9);
        if(!ok || !DonOperatorOk)
            log_printf("PPE-DeepONet: stored-operator check rel.err=%e -> %s\n",err,(ok ? "ok" : "MISMATCH, DeepONet path disabled"));
        DonOperatorOk=ok;
    }
    if(!DonOperatorOk) return;
    // Reuse the basis/coarse operator of a recent solve unless the particles were
    // re-sorted (neighbour rebuild).  A slightly stale preconditioner only costs
    // iterations; convergence is always judged on the current operator.
    int full=(!DonSetupValid || PpeDonSetupEvery<=1 || DonSolveCount-DonLastSetup>=PpeDonSetupEvery);
    if(!full){
        // particles re-sorted since the setup -> move the rows of Q and R along
        const int n=ParticleCount, P=DonP;
        int *pid=DonPid, *slot=DonSlot; int mism=0, bad=0;
        #pragma acc parallel loop reduction(+:mism,bad) present(pid[0:n],ParticleIndex[0:n])
        #pragma omp parallel for reduction(+:mism,bad)
        for(int i=0;i<n;++i){ mism+=(pid[i]!=ParticleIndex[i]); bad+=(ParticleIndex[i]<0 || ParticleIndex[i]>=n); }
        if(bad) full=1;
        else if(mism){
            double *Q=DonQ, *R=DonR, *Qn=DonAT, *Rn=DonT;      // T, AT are free outside the setup
            #pragma acc parallel loop collapse(2) present(Q[0:P*n],R[0:P*n],Qn[0:P*n],Rn[0:P*n],slot[0:n],ParticleIndex[0:n])
            #pragma omp parallel for collapse(2)
            for(int k=0;k<P;++k){
                for(int i=0;i<n;++i){
                    const long src=(long)k*n+slot[ParticleIndex[i]];
                    Qn[(long)k*n+i]=Q[src]; Rn[(long)k*n+i]=R[src];
                }
            }
            DonQ=Qn; DonAT=Q; DonR=Rn; DonT=R;
            #pragma acc parallel loop present(pid[0:n],slot[0:n],ParticleIndex[0:n])
            #pragma omp parallel for
            for(int i=0;i<n;++i){ pid[i]=ParticleIndex[i]; slot[ParticleIndex[i]]=i; }
        }
    }
    int setupOk=1;
    if(full){
        DonSetupValid=0;
        { DON_TIC(t); setupOk=donComputeGeometry(); DON_TOC(DT_GEOM,t); }
    }
    if(setupOk){ DON_TIC(t); setupOk=donSetupSmoother(); DON_TOC(DT_LMAX,t); }
    if(setupOk && full){
        setupOk=donSetupModel();
        if(setupOk){
            int *pid=DonPid, *slot=DonSlot; int bad=0;
            #pragma acc parallel loop reduction(+:bad) present(pid[0:ParticleCount],slot[0:ParticleCount],ParticleIndex[0:ParticleCount])
            #pragma omp parallel for reduction(+:bad)
            for(int i=0;i<ParticleCount;++i){
                const int id=ParticleIndex[i];
                pid[i]=id;
                if(id>=0 && id<ParticleCount) slot[id]=i; else ++bad;
            }
            // ids outside [0,n) -> no reuse (setup every solve), the current setup stays valid
            if(!bad){ DonSetupValid=1; DonLastSetup=DonSolveCount; }
        }
    }
    if(!full) ++DonReuseCount;
    if(!setupOk){
        log_printf("PPE-DeepONet: setup failed (singular coarse operator?) -> Jacobi BiCGStab\n");
        ++DonFallbackCount;
        return;
    }
    const double t1=donWtime();

    int breakdown=0;
    const double rr0=*rrIO;
    const int ok=donBiCGStab(bb,rrIO,donIter,&breakdown);
    const double t2=donWtime();
    DonSetupTime+=t1-t0; DonSolveTime+=t2-t1; DonIterTotal+=*donIter; ++DonRunCount;
    if(!ok){
        ++DonFallbackCount;
        log_printf("PPE-DeepONet: not converged (iter=%d rel=%e breakdown=%d, start rel=%e) -> Jacobi BiCGStab fallback\n",
                   *donIter,sqrt(*rrIO/bb),breakdown,sqrt(rr0/bb));
        *rrIO=donPrepareFallback(*rrIO,rr0);
    }
    if(LogVerbose) log_printf("PPE-DeepONet: iter=%d rel=%e setup=%.3fms solve=%.3fms lmax=%.4f | mean iter=%.2f fallback=%ld\n",
               *donIter,sqrt(*rrIO/bb),1e3*(t1-t0),1e3*(t2-t1),DonLmax,(double)DonIterTotal/DonRunCount,DonFallbackCount);
}


// -----------------------------------------------------------------------------
// Training snapshot (little endian):
//   char[8] "PPEDON01"
//   int32 sdim,nloc,nglob,n,pureNeumann,twoPhase,solveIndex,reserved
//   int64 nnz
//   double dp,dt,time,tol, glob[nglob]
//   double pos[n*sdim], xhat[n*sdim], loc[n*nloc], diag[n], b[n], p[n], p0[n]
//   int64 rowptr[n+1]; int32 col[nnz]; double val[nnz]      (full CSR of A on the unknowns)
// p is the converged pressure BEFORE clipping, p0 the warm start.
// -----------------------------------------------------------------------------
static void donMaybeExport(const double bb, const double rr)
{
    if(PpeDonExportEvery<=0 || DonExportCount>=PpeDonExportMax) return;
    if(DonSolveCount%PpeDonExportEvery!=0) return;
    if(!(bb>1.0e-60) || !(rr<=100.0*PpeTolerance*PpeTolerance*bb)) return;

    donAllocateParticleBuffers();
    donAllocateNeighborBuffers();
    donBuildOperator();
    if(!donComputeGeometry() || DonNAct<2) return;

    const int n=ParticleCount; const long nb=DonNbCount;
    int *act=DonAct; double *X=DonX, *ad=DonAdiag, *coef=DonCoef, *p0=DonP0;
    #pragma acc update self(act[0:n],X[0:n*DON_NX],ad[0:n],coef[0:nb],p0[0:n])
    #pragma acc update self(PpeB[0:n],PressureP[0:n],Position[0:n][0:DIM],NeighborCount[0:n],NeighborPtr[0:PowerParticleCount],NeighborInd[0:NeighborIndCount])

    std::vector<int> map(n,-1);
    int m=0;
    for(int i=0;i<n;++i) if(act[i]) map[i]=m++;
    std::vector<long long> rowptr(m+1,0);
    std::vector<int> col; std::vector<double> val;
    col.reserve((size_t)m*40); val.reserve((size_t)m*40);
    for(int i=0;i<n;++i){
        if(!act[i]) continue;
        col.push_back(map[i]); val.push_back(ad[i]);
        for(int jN=0;jN<NeighborCount[i];++jN){
            const long k=NeighborPtr[i]+jN;
            const double c=coef[k];
            if(c==0.0) continue;
            const int jm=map[NeighborInd[k]];
            if(jm<0) continue;
            col.push_back(jm); val.push_back(-c);
        }
        rowptr[map[i]+1]=(long long)col.size();
    }

    donResolveExportDir();
    donMkdirP(PpeDonExportDir);
    char fn[2048];
    snprintf(fn,sizeof(fn),"%s/ppe_%07ld.bin",PpeDonExportDir,DonSolveCount);
    FILE *fp=fopen(fn,"wb");
    if(!fp){ log_printf("PPE-DeepONet: cannot write %s\n",fn); return; }
    const int hdr[8]={DON_SDIM,DON_NLOC,DON_NGLOB,m,(TwoPhaseActive && PpePureNeumann ? 1 : 0),TwoPhaseActive,(int)DonSolveCount,0};
    const long long nnz=(long long)col.size();
    const double dh[4]={ParticleSpacing,Dt,Time,PpeTolerance};
    fwrite("PPEDON01",1,8,fp); fwrite(hdr,sizeof(int),8,fp); fwrite(&nnz,sizeof(long long),1,fp);
    fwrite(dh,sizeof(double),4,fp); fwrite(DonGlob,sizeof(double),DON_NGLOB,fp);
    std::vector<double> buf((size_t)m*DON_NX);
    for(int i=0,a=0;i<n;++i) if(act[i]){ for(int d=0;d<DON_SDIM;++d) buf[(size_t)a*DON_SDIM+d]=Position[i][d]; ++a; }
    fwrite(buf.data(),sizeof(double),(size_t)m*DON_SDIM,fp);
    for(int i=0,a=0;i<n;++i) if(act[i]){ for(int d=0;d<DON_SDIM;++d) buf[(size_t)a*DON_SDIM+d]=X[(long)i*DON_NX+d]; ++a; }
    fwrite(buf.data(),sizeof(double),(size_t)m*DON_SDIM,fp);
    for(int i=0,a=0;i<n;++i) if(act[i]){ for(int d=0;d<DON_NLOC;++d) buf[(size_t)a*DON_NLOC+d]=X[(long)i*DON_NX+DON_SDIM+d]; ++a; }
    fwrite(buf.data(),sizeof(double),(size_t)m*DON_NLOC,fp);
    const double *fields[4]={ad,PpeB,PressureP,p0};
    for(int f=0;f<4;++f){
        for(int i=0,a=0;i<n;++i) if(act[i]) buf[a++]=fields[f][i];
        fwrite(buf.data(),sizeof(double),m,fp);
    }
    fwrite(rowptr.data(),sizeof(long long),m+1,fp);
    fwrite(col.data(),sizeof(int),col.size(),fp);
    fwrite(val.data(),sizeof(double),val.size(),fp);
    fclose(fp);
    if(getenv("PPE_DON_DUMP") && DonModelState==1 && DonT){
        const int P=DonP; double *T=DonT, *Q=DonQ;
        #pragma acc update self(T[0:P*n],Q[0:P*n])
        char fd_name[2048];
        snprintf(fd_name,sizeof(fd_name),"%s/ppe_%07ld.TQ",PpeDonExportDir,DonSolveCount);
        FILE *fd=fopen(fd_name,"wb");
        if(fd){
            for(int k=0;k<P;++k) for(int i=0;i<n;++i) if(act[i]) fwrite(&T[(long)k*n+i],sizeof(double),1,fd);
            for(int k=0;k<P;++k) for(int i=0;i<n;++i) if(act[i]) fwrite(&Q[(long)k*n+i],sizeof(double),1,fd);
            fclose(fd);
        }
    }
    ++DonExportCount;
    if(LogVerbose) log_printf("PPE-DeepONet: exported %s (n=%d nnz=%lld)\n",fn,m,nnz);
}

static double DonJacobiTime=0.0;
static long   DonJacobiIter=0, DonJacobiRuns=0;
static void donAccountJacobi(const double t, const int iter)
{
    DonJacobiTime+=t;
    if(iter>0){ DonJacobiIter+=iter; ++DonJacobiRuns; }
}

static void donLogSummary()
{
    log_printf("PPE summary: Jacobi BiCGStab loop: runs=%ld iter=%ld (mean %.2f) time=%.3fs\n",
               DonJacobiRuns,DonJacobiIter,(DonJacobiRuns>0 ? (double)DonJacobiIter/DonJacobiRuns : 0.0),DonJacobiTime);
    if(!PpeDon || DonModelState!=1) return;
    log_printf("PPE summary: DeepONet BiCGStab: runs=%ld mean iter=%.2f fallback=%ld setup=%.3fs solve=%.3fs total=%.3fs\n",
               DonRunCount,(DonRunCount>0 ? (double)DonIterTotal/DonRunCount : 0.0),DonFallbackCount,
               DonSetupTime,DonSolveTime,DonSetupTime+DonSolveTime);
    DonTm[DT_KRYLOV]=DonSolveTime-DonTm[DT_APPLY]-DonTm[DT_MATVEC];
    const double per=(DonRunCount>0 ? 1e3/DonRunCount : 0.0);
    log_printf("PPE summary: DeepONet time per solve [ms]:");
    for(int k=0;k<DT_COUNT;++k) log_printf(" %s=%.3f",DonTmName[k],DonTm[k]*per);
    log_printf("  (precond applies/solve=%.1f, basis reused in %ld of %ld solves)\n",(DonRunCount>0 ? (double)DonApplyCount/DonRunCount : 0.0),DonReuseCount,DonRunCount);
    if(DonJacobiRuns>0) log_printf("PPE summary: Jacobi time per solve [ms]: %.3f (%.4f per iteration)\n",
                                   1e3*DonJacobiTime/DonJacobiRuns,1e3*DonJacobiTime/(DonJacobiIter>0 ? DonJacobiIter : 1));
}
