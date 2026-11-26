// =============================================================================
// fvm3d_air.inc -- carrier air on a 3D FVM grid (3D build: make DIM=3 SPRAY=1)
//
// Included once by main_implicit.cpp in place of fvm_air.inc, after aerosol.inc
// and before aerosol_field3d.inc / spray.inc. Same role and coupling as the 2D
// fvm_air.inc (which wraps VOF_FVM_2Phase); the 2D VOF solver has no 3D
// version, and the air here is gas only, so this file is its own solver:
//
//   incompressible gas (rho_g, mu_g) on a uniform MAC grid, solid cells for
//   the vessel (closed domain, no-slip walls)
//   advection : conservative face fluxes, van Leer limited upwind values
//   viscosity : (nu_g + nu_t) Laplacian, explicit; nu_t = (Cs D)^2 |S|
//               (Smagorinsky LES, D = (dx dy dz)^(1/3); Cs = 0 -> off)
//   step      : explicit predictor u* = u + dt (-div(uu) + nu lap u)
//               projection  lap phi = div u* / dt,  u = u* - dt grad phi
//               (Jacobi-preconditioned CG, warm start, zero-mean source)
//   dt        : cfl / (|u|/dx + |v|/dy + |w|/dz)_max, viscous limit, dt_max,
//               growth <= dt_growth per step, ends exactly at the SPH time
//
// Storage: every field uses the cell layout with two ghost layers,
//   q = ((i+1)*NJ + (j+1))*NK + (k+1),  i in [-1,nx+2] (fluid 1..nx), NJ=ny+4, NK=nz+4
//   U[q] = u on the x face between cell (i,j,k) and (i+1,j,k)   (V: +y, W: +z)
// Ghost and solid cells have FvmSolid=1; a face is open when both cells are gas,
// closed faces keep zero velocity.
// Coordinates: y is vertical (gravity -y as in the SPH data), x and z horizontal.
//
// Data file keys (SPH .data):
//   FvmData   path      3D air parameters, "key value" lines:
//                         rho_gas 1.2, mu_gas 1.8e-5, cfl 0.3, smagorinsky 0.0,
//                         turbulent_schmidt 0.7 (aerosol diffusion nu_t/Sc_t),
//                         dt_max 5e-3, dt_growth 1.2, pcg_tol 1e-6, pcg_maxit 3000
//   FvmGrid   path      FVM3D_GRID_V1 (see fvm3ReadGrid)
//   FvmVtk    pattern   air VTK frames (binary STRUCTURED_POINTS), e.g. air_%04d.vtk
//   FvmEvery  n         advance the air every n SPH steps (default 1)
// =============================================================================

static int  FvmEnabled=0;
static int  FvmEvery=1, FvmStepCount=0;
static char FvmDataFile[1024]="", FvmGridFile[1024]="", FvmVtkPattern[1024]="";

struct Fvm3Params {
    double rho_g=1.2, mu_g=1.8e-5, cfl=0.3, cs=0.0, sc_t=0.7;
    double dt_max=5.0e-3, dt_growth=1.2, pcg_tol=1.0e-6;
    int    pcg_maxit=3000;
};
struct Fvm3Geometry {
    int nx=0, ny=0, nz=0;
    double dx=0.0, dy=0.0, dz=0.0, xmin=0.0, ymin=0.0, zmin=0.0;
};
static Fvm3Params   FvmP;
static Fvm3Geometry FvmG;
static int FvmNI=0, FvmNJ=0, FvmNK=0, FvmN=0, FvmFluidCount=0;
static unsigned char *FvmSolid=NULL;
static double *FvmU=NULL, *FvmV=NULL, *FvmW=NULL;          // face velocities (host mirror)
static double *FvmUs=NULL, *FvmVs=NULL, *FvmWs=NULL;       // predictor
static double *FvmPhi=NULL, *FvmRhs=NULL, *FvmCgR=NULL, *FvmCgP=NULL, *FvmCgAp=NULL, *FvmDiag=NULL;
static double *FvmNut=NULL;                                // eddy viscosity (cells)
static double FvmTime=0.0, FvmPrevDt=-1.0;
static int    FvmStep=0, FvmFrame=0, FvmLastSubsteps=0, FvmLastIter=0, FvmPcgFail=0;
static double FvmLastRes=0.0, FvmLastDiv=0.0, FvmLastUmax=0.0, FvmLastNutMax=0.0;

static inline int fvm3Index(const int i, const int j, const int k)
{
    return ((i+1)*FvmNJ+(j+1))*FvmNK+(k+1);
}

static int fvmParseDataLine(const char *buf)
{
    char key[256];
    if(sscanf(buf," %255s",key)!=1 || strncmp(key,"Fvm",3)) return 0;
    if(sscanf(buf," FvmData %1023s",FvmDataFile)==1){ FvmEnabled=1; return 1; }
    if(sscanf(buf," FvmGrid %1023s",FvmGridFile)==1) return 1;
    if(sscanf(buf," FvmVtk %1023s",FvmVtkPattern)==1) return 1;
    if(sscanf(buf," FvmEvery %d",&FvmEvery)==1) return 1;
    return 0;
}

static void fvm3ReadParams(const char *path)
{
    FILE *fp=fopen(path,"r");
    if(!fp){ log_printf("FVM: cannot open %s\n",path); exit(EXIT_FAILURE); }
    char buf[1024], key[256]; double v;
    while(fgets(buf,sizeof(buf),fp)){
        if(sscanf(buf," %255s %lf",key,&v)!=2 || key[0]=='#') continue;
        if(!strcmp(key,"rho_gas")) FvmP.rho_g=v;
        else if(!strcmp(key,"mu_gas")) FvmP.mu_g=v;
        else if(!strcmp(key,"cfl")) FvmP.cfl=v;
        else if(!strcmp(key,"smagorinsky")) FvmP.cs=v;
        else if(!strcmp(key,"turbulent_schmidt")) FvmP.sc_t=v;
        else if(!strcmp(key,"dt_max")) FvmP.dt_max=v;
        else if(!strcmp(key,"dt_growth")) FvmP.dt_growth=v;
        else if(!strcmp(key,"pcg_tol")) FvmP.pcg_tol=v;
        else if(!strcmp(key,"pcg_maxit")) FvmP.pcg_maxit=(int)v;
    }
    fclose(fp);
}

// FVM3D_GRID_V1 text file ('#' comment lines anywhere before BeginSolid):
//   FVM3D_GRID_V1
//   Cell nx ny nz
//   Spacing dx dy dz
//   LowerDomain xmin ymin zmin
//   BeginSolid
//   nz*ny rows (k outer, j inner) of nx characters, '1' = solid, '0' = gas
//   EndSolid
static void fvm3ReadGrid(const char *path)
{
    FILE *fp=fopen(path,"r");
    if(!fp){ log_printf("FVM: cannot open %s\n",path); exit(EXIT_FAILURE); }
    char buf[65536]; int magic=0, inSolid=0, row=0;
    while(fgets(buf,sizeof(buf),fp)){
        if(buf[0]=='#') continue;
        if(!inSolid){
            if(!strncmp(buf,"FVM3D_GRID_V1",13)) magic=1;
            else if(sscanf(buf," Cell %d %d %d",&FvmG.nx,&FvmG.ny,&FvmG.nz)==3){
                FvmNI=FvmG.nx+4; FvmNJ=FvmG.ny+4; FvmNK=FvmG.nz+4; FvmN=FvmNI*FvmNJ*FvmNK;
                FvmSolid=(unsigned char *)malloc(FvmN);
                memset(FvmSolid,1,FvmN);
            }
            else if(sscanf(buf," Spacing %lf %lf %lf",&FvmG.dx,&FvmG.dy,&FvmG.dz)==3){}
            else if(sscanf(buf," LowerDomain %lf %lf %lf",&FvmG.xmin,&FvmG.ymin,&FvmG.zmin)==3){}
            else if(!strncmp(buf,"BeginSolid",10)){
                if(!magic || !FvmSolid){ log_printf("FVM: %s is not FVM3D_GRID_V1\n",path); exit(EXIT_FAILURE); }
                inSolid=1;
            }
            continue;
        }
        if(!strncmp(buf,"EndSolid",8)) break;
        if(row>=FvmG.ny*FvmG.nz) continue;
        const int k=row/FvmG.ny+1, j=row%FvmG.ny+1;
        for(int i=1;i<=FvmG.nx;++i) FvmSolid[fvm3Index(i,j,k)]=(buf[i-1]=='1') ? 1 : 0;
        ++row;
    }
    fclose(fp);
    if(row!=FvmG.ny*FvmG.nz || !(FvmG.dx>0.0 && FvmG.dy>0.0 && FvmG.dz>0.0)){
        log_printf("FVM: %s: %d solid rows read, %d expected\n",path,row,FvmG.ny*FvmG.nz);
        exit(EXIT_FAILURE);
    }
}

// keep the largest connected gas region (the projection needs one closed region)
static void fvm3KeepLargestRegion()
{
    const int sI=FvmNJ*FvmNK, sJ=FvmNK, sK=1;
    std::vector<int> lab(FvmN,-1), stack;
    int best=-1, bestCount=0, nlab=0;
    std::vector<int> counts;
    for(int q=0;q<FvmN;++q){
        if(FvmSolid[q] || lab[q]>=0) continue;
        int c=0; stack.push_back(q); lab[q]=nlab;
        while(!stack.empty()){
            const int p=stack.back(); stack.pop_back(); ++c;
            const int nb[6]={p+sI,p-sI,p+sJ,p-sJ,p+sK,p-sK};
            for(int n=0;n<6;++n) if(!FvmSolid[nb[n]] && lab[nb[n]]<0){ lab[nb[n]]=nlab; stack.push_back(nb[n]); }
        }
        counts.push_back(c);
        if(c>bestCount){ bestCount=c; best=nlab; }
        ++nlab;
    }
    int removed=0;
    for(int q=0;q<FvmN;++q) if(!FvmSolid[q] && lab[q]!=best){ FvmSolid[q]=1; ++removed; }
    FvmFluidCount=bestCount;
    if(removed) log_printf("FVM: %d gas cells outside the main region (%d regions) made solid\n",removed,nlab);
}

static void fvmVelocityToHost()
{
    if(!FvmEnabled) return;
    double *U=FvmU, *V=FvmV, *W=FvmW; const int N=FvmN;
    #pragma acc update self(U[0:N],V[0:N],W[0:N])
}
static void fvmVelocityToDevice()
{
    if(!FvmEnabled) return;
    double *U=FvmU, *V=FvmV, *W=FvmW; const int N=FvmN;
    #pragma acc update device(U[0:N],V[0:N],W[0:N])
}

static double *fvm3Alloc()
{
    double *a=(double *)malloc((size_t)FvmN*sizeof(double));
    for(int q=0;q<FvmN;++q) a[q]=0.0;
    return a;
}

static void fvmInitialize()
{
    if(!FvmEnabled) return;
    fvm3ReadParams(FvmDataFile);
    fvm3ReadGrid(FvmGridFile);
    fvm3KeepLargestRegion();
    if(FvmFluidCount<=0){ log_printf("FVM: no gas cells in %s\n",FvmGridFile); exit(EXIT_FAILURE); }
    FvmU=fvm3Alloc(); FvmV=fvm3Alloc(); FvmW=fvm3Alloc();
    FvmUs=fvm3Alloc(); FvmVs=fvm3Alloc(); FvmWs=fvm3Alloc();
    FvmPhi=fvm3Alloc(); FvmRhs=fvm3Alloc(); FvmCgR=fvm3Alloc(); FvmCgP=fvm3Alloc(); FvmCgAp=fvm3Alloc();
    FvmDiag=fvm3Alloc(); FvmNut=fvm3Alloc();
    // Jacobi diagonal of -lap (open faces only)
    const int sI=FvmNJ*FvmNK, sJ=FvmNK;
    const double ax=1.0/(FvmG.dx*FvmG.dx), ay=1.0/(FvmG.dy*FvmG.dy), az=1.0/(FvmG.dz*FvmG.dz);
    for(int q=0;q<FvmN;++q){
        if(FvmSolid[q]) continue;
        double d=0.0;
        if(!FvmSolid[q+sI]) d+=ax; if(!FvmSolid[q-sI]) d+=ax;
        if(!FvmSolid[q+sJ]) d+=ay; if(!FvmSolid[q-sJ]) d+=ay;
        if(!FvmSolid[q+1])  d+=az; if(!FvmSolid[q-1])  d+=az;
        FvmDiag[q]=d;
    }
    const int N=FvmN;
    unsigned char *S=FvmSolid;
    double *U=FvmU,*V=FvmV,*W=FvmW,*Us=FvmUs,*Vs=FvmVs,*Ws=FvmWs,*Phi=FvmPhi,*Rhs=FvmRhs;
    double *R=FvmCgR,*P=FvmCgP,*Ap=FvmCgAp,*D=FvmDiag,*Nut=FvmNut;
    #pragma acc enter data copyin(S[0:N],U[0:N],V[0:N],W[0:N],Us[0:N],Vs[0:N],Ws[0:N],Phi[0:N],Rhs[0:N])
    #pragma acc enter data copyin(R[0:N],P[0:N],Ap[0:N],D[0:N],Nut[0:N])
    FvmTime=Time;
    if(FvmEvery<1) FvmEvery=1;
    log_printf("FVM air 3D: grid %d x %d x %d d=(%e,%e,%e) origin=(%e,%e,%e) gas cells=%d\n",
               FvmG.nx,FvmG.ny,FvmG.nz,FvmG.dx,FvmG.dy,FvmG.dz,FvmG.xmin,FvmG.ymin,FvmG.zmin,FvmFluidCount);
    log_printf("FVM air 3D: rho_g=%e mu_g=%e cfl=%f Cs=%f Sc_t=%f dt_max=%e pcg_tol=%e maxit=%d, every %d SPH step(s)\n",
               FvmP.rho_g,FvmP.mu_g,FvmP.cfl,FvmP.cs,FvmP.sc_t,FvmP.dt_max,FvmP.pcg_tol,FvmP.pcg_maxit,FvmEvery);
}

// van Leer limited value at the interface between q0 and q1, transport a
#pragma acc routine seq
static inline double fvm3Face(const double qm, const double q0, const double q1, const double q2, const double a)
{
    if(a>=0.0){
        const double d=q1-q0;
        if(fabs(d)<1.0e-300) return q0;
        const double r=(q0-qm)/d;
        return q0+0.5*(r+fabs(r))/(1.0+fabs(r))*d;
    }
    const double d=q0-q1;
    if(fabs(d)<1.0e-300) return q1;
    const double r=(q1-q2)/d;
    return q1+0.5*(r+fabs(r))/(1.0+fabs(r))*d;
}

// Smagorinsky eddy viscosity at the gas cells; returns max nu_t
static double fvm3EddyViscosity()
{
    const int nx=FvmG.nx, ny=FvmG.ny, nz=FvmG.nz, NJ=FvmNJ, NK=FvmNK, N=FvmN;
    const int sI=NJ*NK, sJ=NK, sK=1;
    const double dx=FvmG.dx, dy=FvmG.dy, dz=FvmG.dz;
    const double cd=FvmP.cs*cbrt(dx*dy*dz), cd2=cd*cd;
    const unsigned char *S=FvmSolid;
    const double *U=FvmU, *V=FvmV, *W=FvmW;
    double *Nut=FvmNut;
    double nmax=0.0;
    if(!(FvmP.cs>0.0)) return 0.0;
    #pragma acc parallel loop collapse(3) present(S[0:N],U[0:N],V[0:N],W[0:N],Nut[0:N]) reduction(max:nmax)
    for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) for(int k=1;k<=nz;++k){
        const int q=((i+1)*NJ+(j+1))*NK+(k+1);
        double nt=0.0;
        if(!S[q]){
            // cell-centred velocities of the neighbours (0 in solid cells)
            #define UC(p) (S[p] ? 0.0 : 0.5*(U[p]+U[(p)-sI]))
            #define VC(p) (S[p] ? 0.0 : 0.5*(V[p]+V[(p)-sJ]))
            #define WC(p) (S[p] ? 0.0 : 0.5*(W[p]+W[(p)-sK]))
            const double ux=(U[q]-U[q-sI])/dx, vy=(V[q]-V[q-sJ])/dy, wz=(W[q]-W[q-sK])/dz;
            const double uy=(UC(q+sJ)-UC(q-sJ))/(2.0*dy), uz=(UC(q+sK)-UC(q-sK))/(2.0*dz);
            const double vx=(VC(q+sI)-VC(q-sI))/(2.0*dx), vz=(VC(q+sK)-VC(q-sK))/(2.0*dz);
            const double wx=(WC(q+sI)-WC(q-sI))/(2.0*dx), wy=(WC(q+sJ)-WC(q-sJ))/(2.0*dy);
            #undef UC
            #undef VC
            #undef WC
            const double s2=2.0*(ux*ux+vy*vy+wz*wz)+(uy+vx)*(uy+vx)+(uz+wx)*(uz+wx)+(vz+wy)*(vz+wy);
            nt=cd2*sqrt(s2);
        }
        Nut[q]=nt;
        nmax=fmax(nmax,nt);
    }
    return nmax;
}

static double fvm3MaxRate(double *umaxOut)
{
    const int nx=FvmG.nx, ny=FvmG.ny, nz=FvmG.nz, NJ=FvmNJ, NK=FvmNK, N=FvmN;
    const double *U=FvmU, *V=FvmV, *W=FvmW;
    const double dx=FvmG.dx, dy=FvmG.dy, dz=FvmG.dz;
    double rate=0.0, umax=0.0;
    #pragma acc parallel loop collapse(3) present(U[0:N],V[0:N],W[0:N]) reduction(max:rate,umax)
    for(int i=0;i<=nx;++i) for(int j=0;j<=ny;++j) for(int k=0;k<=nz;++k){
        const int q=((i+1)*NJ+(j+1))*NK+(k+1);
        const double au=fabs(U[q]), av=fabs(V[q]), aw=fabs(W[q]);
        rate=fmax(rate,au/dx+av/dy+aw/dz);
        umax=fmax(umax,fmax(au,fmax(av,aw)));
    }
    *umaxOut=umax;
    return rate;
}

// u* = u + dt (-div(u u) + nu_eff lap u) on the open faces
static void fvm3Predictor(const double dt)
{
    const int nx=FvmG.nx, ny=FvmG.ny, nz=FvmG.nz, NJ=FvmNJ, NK=FvmNK, N=FvmN;
    const int sI=NJ*NK, sJ=NK, sK=1;
    const double dx=FvmG.dx, dy=FvmG.dy, dz=FvmG.dz, nu=FvmP.mu_g/FvmP.rho_g;
    const unsigned char *S=FvmSolid;
    const double *U=FvmU, *V=FvmV, *W=FvmW, *Nut=FvmNut;
    double *Us=FvmUs, *Vs=FvmVs, *Ws=FvmWs;
    #pragma acc parallel loop collapse(3) present(S[0:N],U[0:N],V[0:N],W[0:N],Nut[0:N],Us[0:N],Vs[0:N],Ws[0:N])
    for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) for(int k=1;k<=nz;++k){
        const int q=((i+1)*NJ+(j+1))*NK+(k+1);
        // one velocity component c on its faces: F = field, s = its own direction stride,
        // (t1,G1,h1), (t2,G2,h2) = the two transverse strides, transport fields, spacings
        #pragma acc loop seq
        for(int c=0;c<3;++c){
            const double *F =(c==0)?U:((c==1)?V:W);
            const double *Ga=(c==0)?V:U;                 // transport along t1
            const double *Gb=(c==2)?V:W;                 // transport along t2
            const int s =(c==0)?sI:((c==1)?sJ:sK);
            const int t1=(c==0)?sJ:sI;
            const int t2=(c==2)?sJ:sK;
            const double hs=(c==0)?dx:((c==1)?dy:dz);
            const double h1=(c==0)?dy:dx;
            const double h2=(c==2)?dy:dz;
            double out=0.0;
            if(!S[q] && !S[q+s]){
                const double f0=F[q];
                // own direction: fluxes at the cell centres q (lower) and q+s (upper)
                const double aL=0.5*(F[q-s]+f0), aR=0.5*(f0+F[q+s]);
                const double fL=aL*fvm3Face(F[q-2*s],F[q-s],f0,F[q+s],aL);
                const double fR=aR*fvm3Face(F[q-s],f0,F[q+s],F[q+2*s],aR);
                // transverse 1: edges at +t1/2 (transport G[q], G[q+s]) and -t1/2
                const double b1p=0.5*(Ga[q]+Ga[q+s]), b1m=0.5*(Ga[q-t1]+Ga[q-t1+s]);
                const double g1p=b1p*fvm3Face(F[q-t1],f0,F[q+t1],F[q+2*t1],b1p);
                const double g1m=b1m*fvm3Face(F[q-2*t1],F[q-t1],f0,F[q+t1],b1m);
                const double b2p=0.5*(Gb[q]+Gb[q+s]), b2m=0.5*(Gb[q-t2]+Gb[q-t2+s]);
                const double g2p=b2p*fvm3Face(F[q-t2],f0,F[q+t2],F[q+2*t2],b2p);
                const double g2m=b2m*fvm3Face(F[q-2*t2],F[q-t2],f0,F[q+t2],b2m);
                const double conv=(fR-fL)/hs+(g1p-g1m)/h1+(g2p-g2m)/h2;
                // viscosity: closed transverse neighbours mirror the face value (no-slip wall)
                const double ne=nu+0.5*(Nut[q]+Nut[q+s]);
                const double fp1=(!S[q+t1] && !S[q+t1+s]) ? F[q+t1] : -f0;
                const double fm1=(!S[q-t1] && !S[q-t1+s]) ? F[q-t1] : -f0;
                const double fp2=(!S[q+t2] && !S[q+t2+s]) ? F[q+t2] : -f0;
                const double fm2=(!S[q-t2] && !S[q-t2+s]) ? F[q-t2] : -f0;
                const double lap=(F[q+s]-2.0*f0+F[q-s])/(hs*hs)
                                +(fp1-2.0*f0+fm1)/(h1*h1)+(fp2-2.0*f0+fm2)/(h2*h2);
                out=f0+dt*(-conv+ne*lap);
            }
            if(c==0) Us[q]=out; else if(c==1) Vs[q]=out; else Ws[q]=out;
        }
    }
}

// lap phi = div u*/dt (zero-mean), u = u* - dt grad phi; returns the iteration count
static int fvm3Project(const double dt, double *relOut, double *divOut)
{
    const int nx=FvmG.nx, ny=FvmG.ny, nz=FvmG.nz, NJ=FvmNJ, NK=FvmNK, N=FvmN;
    const int sI=NJ*NK, sJ=NK, sK=1;
    const double dx=FvmG.dx, dy=FvmG.dy, dz=FvmG.dz;
    const double ax=1.0/(dx*dx), ay=1.0/(dy*dy), az=1.0/(dz*dz);
    const unsigned char *S=FvmSolid;
    const double *D=FvmDiag;
    double *U=FvmU, *V=FvmV, *W=FvmW, *Us=FvmUs, *Vs=FvmVs, *Ws=FvmWs;
    double *X=FvmPhi, *B=FvmRhs, *R=FvmCgR, *P=FvmCgP, *Ap=FvmCgAp;
    const double nf=(double)FvmFluidCount;

    // b = -div u*/dt (system -lap phi = b), made zero-mean
    double bsum=0.0;
    #pragma acc parallel loop collapse(3) present(S[0:N],Us[0:N],Vs[0:N],Ws[0:N],B[0:N]) reduction(+:bsum)
    for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) for(int k=1;k<=nz;++k){
        const int q=((i+1)*NJ+(j+1))*NK+(k+1);
        double b=0.0;
        if(!S[q]) b=-((Us[q]-Us[q-sI])/dx+(Vs[q]-Vs[q-sJ])/dy+(Ws[q]-Ws[q-sK])/dz)/dt;
        B[q]=b; bsum+=b;
    }
    const double bmean=bsum/nf;
    // r = b - A x,  A = -lap over open faces; also |b|^2, x mean
    double bb=0.0, xsum=0.0;
    #pragma acc parallel loop collapse(3) present(S[0:N],B[0:N],X[0:N]) reduction(+:bb,xsum)
    for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) for(int k=1;k<=nz;++k){
        const int q=((i+1)*NJ+(j+1))*NK+(k+1);
        if(S[q]){ B[q]=0.0; continue; }
        B[q]-=bmean; bb+=B[q]*B[q]; xsum+=X[q];
    }
    const double xmean=xsum/nf;
    // warm start: previous phi, zero-mean (solid cells 0)
    #pragma acc parallel loop present(S[0:N],X[0:N],R[0:N],P[0:N])
    for(int q=0;q<N;++q){
        if(S[q]){ X[q]=0.0; R[q]=0.0; P[q]=0.0; }
        else X[q]-=xmean;
    }
    // r = b - A x,  z = r/diag,  p = z  (closed faces drop out through S)
    double rz=0.0;
    #pragma acc parallel loop collapse(3) present(S[0:N],B[0:N],X[0:N],R[0:N],P[0:N],D[0:N]) reduction(+:rz)
    for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) for(int k=1;k<=nz;++k){
        const int q=((i+1)*NJ+(j+1))*NK+(k+1);
        if(S[q]) continue;
        const double x0=X[q];
        double ax0=0.0;
        if(!S[q+sI]) ax0+=ax*(x0-X[q+sI]); if(!S[q-sI]) ax0+=ax*(x0-X[q-sI]);
        if(!S[q+sJ]) ax0+=ay*(x0-X[q+sJ]); if(!S[q-sJ]) ax0+=ay*(x0-X[q-sJ]);
        if(!S[q+sK]) ax0+=az*(x0-X[q+sK]); if(!S[q-sK]) ax0+=az*(x0-X[q-sK]);
        const double r=B[q]-ax0;
        R[q]=r;
        const double z=(D[q]>0.0) ? r/D[q] : 0.0;
        P[q]=z; rz+=r*z;
    }
    const double bnorm=sqrt(bb);
    double rel=0.0;
    int it=0;
    if(bnorm>0.0){
        double rr=0.0;
        #pragma acc parallel loop present(R[0:N]) reduction(+:rr)
        for(int q=0;q<N;++q) rr+=R[q]*R[q];
        rel=sqrt(rr)/bnorm;
        while(rel>FvmP.pcg_tol && it<FvmP.pcg_maxit){
            double pap=0.0;
            #pragma acc parallel loop collapse(3) present(S[0:N],P[0:N],Ap[0:N]) reduction(+:pap)
            for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) for(int k=1;k<=nz;++k){
                const int q=((i+1)*NJ+(j+1))*NK+(k+1);
                if(S[q]){ Ap[q]=0.0; continue; }
                const double p0=P[q];
                double a=0.0;
                if(!S[q+sI]) a+=ax*(p0-P[q+sI]); if(!S[q-sI]) a+=ax*(p0-P[q-sI]);
                if(!S[q+sJ]) a+=ay*(p0-P[q+sJ]); if(!S[q-sJ]) a+=ay*(p0-P[q-sJ]);
                if(!S[q+sK]) a+=az*(p0-P[q+sK]); if(!S[q-sK]) a+=az*(p0-P[q-sK]);
                Ap[q]=a; pap+=p0*a;
            }
            if(!(pap>0.0)) break;
            const double alpha=rz/pap;
            double rzNew=0.0; rr=0.0;
            #pragma acc parallel loop present(S[0:N],X[0:N],R[0:N],P[0:N],Ap[0:N],D[0:N]) reduction(+:rzNew,rr)
            for(int q=0;q<N;++q){
                if(S[q]) continue;
                X[q]+=alpha*P[q];
                const double r=R[q]-alpha*Ap[q];
                R[q]=r; rr+=r*r;
                rzNew+=(D[q]>0.0) ? r*r/D[q] : 0.0;
            }
            const double beta=rzNew/rz;
            rz=rzNew;
            #pragma acc parallel loop present(S[0:N],R[0:N],P[0:N],D[0:N])
            for(int q=0;q<N;++q){
                if(S[q]) continue;
                P[q]=((D[q]>0.0) ? R[q]/D[q] : 0.0)+beta*P[q];
            }
            rel=sqrt(rr)/bnorm;
            ++it;
        }
    }
    // u = u* - dt grad phi on the open faces, closed faces zero
    #pragma acc parallel loop collapse(3) present(S[0:N],X[0:N],U[0:N],V[0:N],W[0:N],Us[0:N],Vs[0:N],Ws[0:N])
    for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) for(int k=1;k<=nz;++k){
        const int q=((i+1)*NJ+(j+1))*NK+(k+1);
        const int g=!S[q];
        U[q]=(g && !S[q+sI]) ? Us[q]-dt*(X[q+sI]-X[q])/dx : 0.0;
        V[q]=(g && !S[q+sJ]) ? Vs[q]-dt*(X[q+sJ]-X[q])/dy : 0.0;
        W[q]=(g && !S[q+sK]) ? Ws[q]-dt*(X[q+sK]-X[q])/dz : 0.0;
    }
    // dimensionless divergence dt |div u|_max
    double dmax=0.0;
    #pragma acc parallel loop collapse(3) present(S[0:N],U[0:N],V[0:N],W[0:N]) reduction(max:dmax)
    for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) for(int k=1;k<=nz;++k){
        const int q=((i+1)*NJ+(j+1))*NK+(k+1);
        if(S[q]) continue;
        const double d=(U[q]-U[q-sI])/dx+(V[q]-V[q-sJ])/dy+(W[q]-W[q-sK])/dz;
        dmax=fmax(dmax,fabs(d));
    }
    *relOut=rel; *divOut=dmax*dt;
    return it;
}

// advance the air from FvmTime to Time
static void fvmAdvance()
{
    if(!FvmEnabled) return;
    const double dx=FvmG.dx, dy=FvmG.dy, dz=FvmG.dz, nu=FvmP.mu_g/FvmP.rho_g;
    const double sumInvH2=1.0/(dx*dx)+1.0/(dy*dy)+1.0/(dz*dz);
    int sub=0;
    while(FvmTime<Time-1.0e-12*Dt){
        const double nutMax=fvm3EddyViscosity();
        double umax=0.0;
        const double rate=fvm3MaxRate(&umax);
        double dt=FvmP.dt_max;
        if(rate>0.0) dt=fmin(dt,FvmP.cfl/rate);
        dt=fmin(dt,0.4/((nu+nutMax)*sumInvH2));
        if(FvmPrevDt>0.0 && dt>FvmP.dt_growth*FvmPrevDt) dt=FvmP.dt_growth*FvmPrevDt;
        if(FvmTime+dt>Time) dt=Time-FvmTime;
        fvm3Predictor(dt);
        double rel=0.0, div=0.0;
        const int it=fvm3Project(dt,&rel,&div);
        if(rel>FvmP.pcg_tol){
            if(FvmPcgFail<20) log_printf("FVM: PCG not converged at t=%e (it=%d rel=%e)\n",FvmTime,it,rel);
            ++FvmPcgFail;
        }
        FvmTime+=dt; FvmPrevDt=dt; ++FvmStep; ++sub;
        FvmLastIter=it; FvmLastRes=rel; FvmLastDiv=div; FvmLastUmax=umax; FvmLastNutMax=nutMax;
        if(!isfinite(umax) || !isfinite(rel)){ log_printf("FVM: non-finite air velocity at t=%e\n",FvmTime); exit(EXIT_FAILURE); }
    }
    FvmLastSubsteps=sub;
}

// trilinear stencil of one velocity component (0 u, 1 v, 2 w) on the MAC grid:
// u faces at (xmin + i dx, ymin + (j-1/2) dy, zmin + (k-1/2) dz), i in [0,nx], j in [0,ny+1], k in [0,nz+1]
static void fvmStencil(const int comp, const double x, const double y, const double z, int idx[8], double w[8], int open[8])
{
    const int nx=FvmG.nx, ny=FvmG.ny, nz=FvmG.nz;
    double f[3]={(x-FvmG.xmin)/FvmG.dx+0.5,(y-FvmG.ymin)/FvmG.dy+0.5,(z-FvmG.zmin)/FvmG.dz+0.5};
    int nmax[3]={nx+1,ny+1,nz+1};
    f[comp]-=0.5; nmax[comp]-=1;
    int i0[3]; double a[3];
    for(int d=0;d<3;++d){
        i0[d]=(int)floor(f[d]);
        if(i0[d]<0) i0[d]=0; if(i0[d]>nmax[d]-1) i0[d]=nmax[d]-1;
        a[d]=fmin(fmax(f[d]-i0[d],0.0),1.0);
    }
    const int s=(comp==0) ? FvmNJ*FvmNK : ((comp==1) ? FvmNK : 1);
    for(int n=0;n<8;++n){
        const int b[3]={n&1,(n>>1)&1,(n>>2)&1};
        w[n]=(b[0]?a[0]:1.0-a[0])*(b[1]?a[1]:1.0-a[1])*(b[2]?a[2]:1.0-a[2]);
        idx[n]=fvm3Index(i0[0]+b[0],i0[1]+b[1],i0[2]+b[2]);
        open[n]=!FvmSolid[idx[n]] && !FvmSolid[idx[n]+s];
    }
}

// FVM cell (1-based) containing (x,y,z); 1 for a gas cell, 0 for solid or outside
static int fvmCellOf(const double x, const double y, const double z, int *i, int *j, int *k)
{
    *i=1+(int)floor((x-FvmG.xmin)/FvmG.dx);
    *j=1+(int)floor((y-FvmG.ymin)/FvmG.dy);
    *k=1+(int)floor((z-FvmG.zmin)/FvmG.dz);
    if(*i<1 || *i>FvmG.nx || *j<1 || *j>FvmG.ny || *k<1 || *k>FvmG.nz) return 0;
    return FvmSolid[fvm3Index(*i,*j,*k)]==0;
}

// 2D signature declared by aerosol.inc (Lagrangian aerosol is 2D only)
static int fvmAirVelocity(const double x, const double y, double *ux, double *uy)
{
    (void)x; (void)y; (void)ux; (void)uy;
    return 0;
}

static void fvm3PutFloat(FILE *fp, const float v)
{
    unsigned char b[4]; memcpy(b,&v,4);
    const unsigned char be[4]={b[3],b[2],b[1],b[0]};   // legacy VTK binary is big-endian
    fwrite(be,1,4,fp);
}

// cell-centred velocity, eddy viscosity and solid flag (legacy VTK, binary)
static void fvmWriteVtk()
{
    if(!FvmEnabled || FvmVtkPattern[0]=='\0') return;
    {
        double *Nut=FvmNut; const int N=FvmN;
        #pragma acc update self(Nut[0:N])
    }
    fvmVelocityToHost();
    char name[1100]; snprintf(name,sizeof(name),FvmVtkPattern,FvmFrame++);
    FILE *fp=fopen(name,"wb");
    if(!fp) return;
    const int nx=FvmG.nx, ny=FvmG.ny, nz=FvmG.nz, sI=FvmNJ*FvmNK, sJ=FvmNK;
    fprintf(fp,"# vtk DataFile Version 3.0\nFVM air 3D t=%e\nBINARY\nDATASET STRUCTURED_POINTS\n",FvmTime);
    fprintf(fp,"DIMENSIONS %d %d %d\nORIGIN %e %e %e\nSPACING %e %e %e\n",nx+1,ny+1,nz+1,
            FvmG.xmin,FvmG.ymin,FvmG.zmin,FvmG.dx,FvmG.dy,FvmG.dz);
    fprintf(fp,"FIELD FieldData 1\nTimeValue 1 1 double\n");
    { double t=FvmTime; unsigned char b[8], be[8]; memcpy(b,&t,8); for(int n=0;n<8;++n) be[n]=b[7-n]; fwrite(be,1,8,fp); }
    fprintf(fp,"\nCELL_DATA %d\nVECTORS velocity float\n",nx*ny*nz);
    for(int k=1;k<=nz;++k) for(int j=1;j<=ny;++j) for(int i=1;i<=nx;++i){
        const int q=fvm3Index(i,j,k);
        fvm3PutFloat(fp,(float)(0.5*(FvmU[q]+FvmU[q-sI])));
        fvm3PutFloat(fp,(float)(0.5*(FvmV[q]+FvmV[q-sJ])));
        fvm3PutFloat(fp,(float)(0.5*(FvmW[q]+FvmW[q-1])));
    }
    fprintf(fp,"\nSCALARS nu_t float 1\nLOOKUP_TABLE default\n");
    for(int k=1;k<=nz;++k) for(int j=1;j<=ny;++j) for(int i=1;i<=nx;++i) fvm3PutFloat(fp,(float)FvmNut[fvm3Index(i,j,k)]);
    fprintf(fp,"\nSCALARS solid float 1\nLOOKUP_TABLE default\n");
    for(int k=1;k<=nz;++k) for(int j=1;j<=ny;++j) for(int i=1;i<=nx;++i) fvm3PutFloat(fp,(float)FvmSolid[fvm3Index(i,j,k)]);
    fprintf(fp,"\n");
    fclose(fp);
}

static void fvmLogStatus()
{
    if(!FvmEnabled) return;
    log_printf("FVM air 3D: t=%e steps=%d last substeps=%d dt=%e pcg it=%d rel=%e dt*div=%e Umax=%e nu_t,max=%e pcg_fail=%d\n",
               FvmTime,FvmStep,FvmLastSubsteps,FvmPrevDt,FvmLastIter,FvmLastRes,FvmLastDiv,FvmLastUmax,FvmLastNutMax,FvmPcgFail);
}

static void fvmFinalize()
{
    if(!FvmEnabled) return;
    const int N=FvmN;
    unsigned char *S=FvmSolid;
    double *U=FvmU,*V=FvmV,*W=FvmW,*Us=FvmUs,*Vs=FvmVs,*Ws=FvmWs,*Phi=FvmPhi,*Rhs=FvmRhs;
    double *R=FvmCgR,*P=FvmCgP,*Ap=FvmCgAp,*D=FvmDiag,*Nut=FvmNut;
    #pragma acc exit data delete(S[0:N],U[0:N],V[0:N],W[0:N],Us[0:N],Vs[0:N],Ws[0:N],Phi[0:N],Rhs[0:N])
    #pragma acc exit data delete(R[0:N],P[0:N],Ap[0:N],D[0:N],Nut[0:N])
}
