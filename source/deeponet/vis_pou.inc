// =============================================================================
// vis_pou.inc -- partition-of-unity (PoU) coarse space for the two-level
// viscosity preconditioner of vis_deeponet.inc  (compiled with -DVIS_POU)
//
// Included by vis_deeponet.inc after its buffers are declared.  It replaces the
// global DeepONet basis V = T (x) I (fixed P columns) by LOCAL fields glued by a
// partition of unity, so the coarse dimension grows with n and the two-level
// iteration count stays bounded when dp is refined:
//
//   lattice    nodes x_a = lo + (ix, iy) H,   H = VisPouH * dp
//   phi_a      bilinear hat functions of the 4 nodes of a particle's cell,
//              renormalised over the kept nodes (sum_a phi_a = 1)
//   kept nodes fill_a = sum_i phi_a(x_i) >= VisPouMinFill * (H/dp)^2;
//              a particle whose 4 nodes are all dropped forces its best node in
//   fields     f_0 = 1, f_1 = xi_x, f_2 = xi_y,   xi = (x_i - x_a)/H
//   V          column (a, k, d): V_(i,d') = phi_a(x_i) f_k(i, a) delta_dd'
//              coarse index c = (a*K + k)*SD + d
//
// Galerkin  Ac = V^T A V  is assembled without atomics:
//   1) AVloc_i[s][k'][q][p] = (A V_(b,k',q))_(i,p) for the 16 lattice nodes b
//      around particle i's cell (s = slot), needs RadiusV <= H;
//   2) Ac[(a,k,p),(b,k',q)] = sum_{i in supp a} phi_a f_k(i,a) AVloc_i[slot(b|i)][k'][q][p]
//      over the 5x5 lattice neighbours b of a.
// Ac is symmetrised, Jacobi-scaled, regularised (VisPouReg) and Cholesky
// factorised on the device (cuSOLVER); the coarse solve is a potrs.  The cycle
// (Chebyshev / Galerkin / Chebyshev), the CG, its stopping test and the Jacobi
// fallback are those of vis_deeponet.inc.  2-D only.
//
// Data file keys:
//   VisPou          0|1   use the PoU coarse space (implies the VisDon solver path) [0]
//   VisPouH         h     lattice spacing in dp                                      [11.3]
//   VisPouMinFill   f     keep a node if its fill >= f (H/dp)^2                       [0.1]
//   VisPouReg       r     diagonal regularisation of the scaled Ac                    [1e-8]
//   VisDonLmaxIters n     power iterations for lmax(D^-1 A) (also DeepONet path)      [15]
// =============================================================================
#ifdef _OPENACC
#include <openacc.h>
#include <cusolverDn.h>
#endif

#define POU_K   3          // 1, xi_x, xi_y
#define POU_S   16         // 4x4 node slots around a particle's cell
#define POU_AVL (POU_S*POU_K*VIS_SD*VIS_SD)

static int    VisPou        = 0;
static double VisPouH       = 11.3;
static double VisPouMinFill = 0.1;
static double VisPouReg     = 1.0e-8;


static int     PgN=0, PgNodes=0, PgNx=0, PgNy=0, PgPc=0, PgPcAlloc=0, PgLatAlloc=0, PgListAlloc=0;
static int    *PgNode=NULL;      // [n*4]   kept node index or -1
static int    *PgCell=NULL;      // [n*2]   lattice cell of the particle
static double *PgVal=NULL;       // [n*4*K] phi_a f_k
static double *PgAVL=NULL;       // [n*POU_AVL]
static double *PgW4=NULL;        // [n*4]   raw bilinear weights
static double *PgFill=NULL;      // [lat]
static int    *PgForce=NULL;     // [lat]
static int    *PgLat=NULL;       // [lat]   lattice -> kept node (-1)
static int    *PgNodeIJ=NULL;    // [nodes*2]
static int    *PgPtr=NULL;       // [nodes+1] node particle lists
static int    *PgCnt=NULL;       // [nodes]
static int    *PgLi=NULL, *PgLs=NULL;   // [4n] particle, slot
static double *PgAc=NULL, *PgAf=NULL;   // [Pc*Pc] raw / factorised
static double *PgSc=NULL;        // [Pc] Jacobi scaling of Ac
static double *PgCc=NULL;        // [Pc]
static double PgLoX=0.0, PgLoY=0.0, PgH=1.0;
static long   PgSetups=0, PgFactorRetry=0, PgRangeMiss=0;
static double PgPcSum=0.0;
#ifdef _OPENACC
static cusolverDnHandle_t PgSolver=NULL;
static double *PgWork=NULL; static int PgLwork=0; static int *PgInfo=NULL;
#endif

static int visPouParseDataLine(const char *buf)
{
    if(sscanf(buf," VisPou %d",&VisPou)==1) return 1;
    if(sscanf(buf," VisPouH %lf",&VisPouH)==1) return 1;
    if(sscanf(buf," VisPouMinFill %lf",&VisPouMinFill)==1) return 1;
    if(sscanf(buf," VisPouReg %lf",&VisPouReg)==1) return 1;
    return 0;
}

static void pouFree(void *p){ if(p) free(p); }

// particle arrays (n changes only with ParticleCount)
static void pouAllocParticles()
{
    const int n=ParticleCount;
    if(PgN==n && PgNode) return;
    if(PgNode){
        const int m=PgN; int *nd=PgNode,*ce=PgCell,*li=PgLi,*ls=PgLs; double *v=PgVal,*av=PgAVL,*w4=PgW4;
        #pragma acc exit data delete(nd[0:4*m],ce[0:2*m],v[0:4*m*POU_K],av[0:(long)m*POU_AVL],w4[0:4*m],li[0:4*m],ls[0:4*m])
        free(nd); free(ce); free(v); free(av); free(w4); free(li); free(ls);
    }
    PgN=n;
    VD_NEW(PgNode,int,4L*n); VD_NEW(PgCell,int,2L*n); VD_NEW(PgVal,double,4L*n*POU_K);
    VD_NEW(PgAVL,double,(long)n*POU_AVL); VD_NEW(PgW4,double,4L*n); VD_NEW(PgLi,int,4L*n); VD_NEW(PgLs,int,4L*n);
    int *nd=PgNode,*ce=PgCell,*li=PgLi,*ls=PgLs; double *v=PgVal,*av=PgAVL,*w4=PgW4;
    #pragma acc enter data create(nd[0:4*n],ce[0:2*n],v[0:4*n*POU_K],av[0:(long)n*POU_AVL],w4[0:4*n],li[0:4*n],ls[0:4*n])
}

static void pouAllocLattice(const int lat)
{
    if(lat<=PgLatAlloc) return;
    if(PgFill){
        const int m=PgLatAlloc; double *f=PgFill; int *fo=PgForce,*la=PgLat,*ij=PgNodeIJ,*pt=PgPtr,*cn=PgCnt;
        #pragma acc exit data delete(f[0:m],fo[0:m],la[0:m],ij[0:2*m],pt[0:m+1],cn[0:m])
        free(f); free(fo); free(la); free(ij); free(pt); free(cn);
    }
    const int m=lat+lat/2+16;
    PgLatAlloc=m;
    VD_NEW(PgFill,double,m); VD_NEW(PgForce,int,m); VD_NEW(PgLat,int,m); VD_NEW(PgNodeIJ,int,2L*m); VD_NEW(PgPtr,int,m+1L); VD_NEW(PgCnt,int,m);
    double *f=PgFill; int *fo=PgForce,*la=PgLat,*ij=PgNodeIJ,*pt=PgPtr,*cn=PgCnt;
    #pragma acc enter data create(f[0:m],fo[0:m],la[0:m],ij[0:2*m],pt[0:m+1],cn[0:m])
}

static void pouAllocCoarse(const int Pc)
{
    if(Pc<=PgPcAlloc) return;
    if(PgAc){
        const long m=PgPcAlloc; double *a=PgAc,*af=PgAf,*sc=PgSc,*cc=PgCc;
        #pragma acc exit data delete(a[0:m*m],af[0:m*m],sc[0:m],cc[0:m])
        free(a); free(af); free(sc); free(cc);
    }
    const long m=Pc+Pc/4+32;
    PgPcAlloc=(int)m;
    VD_NEW(PgAc,double,m*m); VD_NEW(PgAf,double,m*m); VD_NEW(PgSc,double,m); VD_NEW(PgCc,double,m);
    double *a=PgAc,*af=PgAf,*sc=PgSc,*cc=PgCc;
    #pragma acc enter data create(a[0:m*m],af[0:m*m],sc[0:m],cc[0:m])
}

// -----------------------------------------------------------------------------
// Lattice, kept nodes, PoU weights and field values (device, small host steps)
// -----------------------------------------------------------------------------
static int pouBuildGrid()
{
#ifndef TWO_DIMENSIONAL
    return 0;
#endif
    pouAllocParticles();
    const int n=ParticleCount; const int *act=VdAct;
    double xmn=1e300,ymn=1e300,xmx=-1e300,ymx=-1e300; int na=0;
    #pragma acc parallel loop reduction(min:xmn,ymn) reduction(max:xmx,ymx) reduction(+:na) present(act[0:n],Position[0:ParticleCount][0:DIM])
    #pragma omp parallel for reduction(min:xmn,ymn) reduction(max:xmx,ymx) reduction(+:na)
    for(int i=0;i<n;++i){
        if(!act[i]) continue;
        ++na;
        const double x=Position[i][0], y=Position[i][1];
        if(x<xmn) xmn=x; if(x>xmx) xmx=x; if(y<ymn) ymn=y; if(y>ymx) ymx=y;
    }
    if(na<1) return 0;
    const double H=VisPouH*ParticleSpacing;
    if(RadiusV>H){ log_printf("VIS-PoU: RadiusV=%g > H=%g, increase VisPouH\n",RadiusV,H); return 0; }
    PgH=H; PgLoX=xmn-1e-9*H; PgLoY=ymn-1e-9*H;
    PgNx=(int)floor((xmx-PgLoX)/H)+2; PgNy=(int)floor((ymx-PgLoY)/H)+2;
    const int lat=PgNx*PgNy, nx=PgNx, ny=PgNy;
    pouAllocLattice(lat);
    double *fill=PgFill; int *force=PgForce, *latn=PgLat, *nd=PgNode, *ce=PgCell; double *w4=PgW4, *val=PgVal;
    const double lox=PgLoX, loy=PgLoY, iH=1.0/H;
    #pragma acc parallel loop present(fill[0:lat],force[0:lat])
    #pragma omp parallel for
    for(int q=0;q<lat;++q){ fill[q]=0.0; force[q]=0; }
    // raw bilinear weights + fill
    #pragma acc parallel loop present(act[0:n],nd[0:4*n],ce[0:2*n],w4[0:4*n],fill[0:lat],Position[0:ParticleCount][0:DIM])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(!act[i]){ for(int s=0;s<4;++s){ nd[4*i+s]=-1; w4[4*i+s]=0.0; } ce[2*i]=ce[2*i+1]=-100; continue; }
        const double gx=(Position[i][0]-lox)*iH, gy=(Position[i][1]-loy)*iH;
        int cx=(int)floor(gx), cy=(int)floor(gy);
        if(cx<0) cx=0; if(cx>nx-2) cx=nx-2; if(cy<0) cy=0; if(cy>ny-2) cy=ny-2;
        const double tx=gx-cx, ty=gy-cy;
        ce[2*i]=cx; ce[2*i+1]=cy;
        for(int s=0;s<4;++s){
            const int ox=s/2, oy=s%2;
            const double w=(ox ? tx : 1.0-tx)*(oy ? ty : 1.0-ty);
            const int q=(cx+ox)*ny+(cy+oy);
            nd[4*i+s]=q; w4[4*i+s]=w;
            #pragma acc atomic update
            fill[q]+=w;
        }
    }
    #pragma acc update self(fill[0:lat])
    const double thr=VisPouMinFill*VisPouH*VisPouH;
    // orphans: all four nodes below the threshold -> force the best one
    int nOrph=0;
    #pragma acc parallel loop reduction(+:nOrph) present(act[0:n],nd[0:4*n],fill[0:lat],force[0:lat])
    #pragma omp parallel for reduction(+:nOrph)
    for(int i=0;i<n;++i){
        if(!act[i]) continue;
        int best=-1; double bf=-1.0, wk=0.0;
        for(int s=0;s<4;++s){ const int q=nd[4*i+s]; if(fill[q]>=thr) wk+=1.0; if(fill[q]>bf){ bf=fill[q]; best=q; } }
        if(wk==0.0){ force[best]=1; ++nOrph; }
    }
    #pragma acc update self(force[0:lat])
    int nodes=0;
    for(int q=0;q<lat;++q){
        if(fill[q]>=thr || force[q]){ latn[q]=nodes; PgNodeIJ[2*nodes]=q/ny; PgNodeIJ[2*nodes+1]=q%ny; ++nodes; }
        else latn[q]=-1;
    }
    PgNodes=nodes;
    if(nodes<1) return 0;
    int *ij=PgNodeIJ;
    #pragma acc update device(latn[0:lat],ij[0:2*nodes])
    // normalised weights, kept node ids, field values phi_a f_k
    #pragma acc parallel loop present(act[0:n],nd[0:4*n],ce[0:2*n],w4[0:4*n],val[0:4*n*POU_K],latn[0:lat],Position[0:ParticleCount][0:DIM])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(!act[i]){ for(int s=0;s<4;++s) for(int k=0;k<POU_K;++k) val[(4*i+s)*POU_K+k]=0.0; continue; }
        double ws=0.0;
        for(int s=0;s<4;++s){ const int a=latn[nd[4*i+s]]; if(a>=0) ws+=w4[4*i+s]; }
        const double iws=(ws>0.0 ? 1.0/ws : 0.0);
        for(int s=0;s<4;++s){
            const int q=nd[4*i+s], a=latn[q];
            const double w=(a>=0 ? w4[4*i+s]*iws : 0.0);
            nd[4*i+s]=(a>=0 && w>0.0 ? a : -1);
            const double xa=lox+(q/ny)*H, ya=loy+(q%ny)*H;
            const double xi=(Position[i][0]-xa)*iH, eta=(Position[i][1]-ya)*iH;
            val[(4*i+s)*POU_K+0]=w;
            val[(4*i+s)*POU_K+1]=w*xi;
            val[(4*i+s)*POU_K+2]=w*eta;
        }
    }
    // node particle lists (counting sort; order inside a list is irrelevant up to round-off)
    int *ptr=PgPtr, *cnt=PgCnt, *li=PgLi, *ls=PgLs;
    #pragma acc parallel loop present(cnt[0:nodes])
    #pragma omp parallel for
    for(int a=0;a<nodes;++a) cnt[a]=0;
    #pragma acc parallel loop present(nd[0:4*n],cnt[0:nodes])
    #pragma omp parallel for
    for(int i=0;i<n;++i)
        for(int s=0;s<4;++s){
            const int a=nd[4*i+s];
            if(a>=0){
                #pragma acc atomic update
                cnt[a]+=1;
            }
        }
    #pragma acc update self(cnt[0:nodes])
    ptr[0]=0;
    for(int a=0;a<nodes;++a){ ptr[a+1]=ptr[a]+cnt[a]; cnt[a]=ptr[a]; }
    #pragma acc update device(ptr[0:nodes+1],cnt[0:nodes])
    #pragma acc parallel loop present(nd[0:4*n],cnt[0:nodes],li[0:4*n],ls[0:4*n])
    #pragma omp parallel for
    for(int i=0;i<n;++i)
        for(int s=0;s<4;++s){
            const int a=nd[4*i+s];
            if(a<0) continue;
            int pos;
            #pragma acc atomic capture
            { pos=cnt[a]; cnt[a]+=1; }
            li[pos]=i; ls[pos]=s;
        }
    PgPc=nodes*POU_K*VIS_SD;
    return 1;
}

// -----------------------------------------------------------------------------
// Galerkin Ac = V^T A V (current stored operator), scaling, Cholesky
// -----------------------------------------------------------------------------
static int pouFactor(const int Pc, const double reg);

static int pouSetupGalerkin()
{
    const int n=ParticleCount, nodes=PgNodes, Pc=PgPc, ny=PgNy, nx=PgNx;
    pouAllocCoarse(Pc);
    const int *act=VdAct, *nd=PgNode, *ce=PgCell, *ptr=PgPtr, *li=PgLi, *ls=PgLs, *ij=PgNodeIJ, *latn=PgLat;
    const double *blk=VdBlk, *coef=VdCoef, *E=VdE, *val=PgVal; const long nb=VdNb;
    double *avl=PgAVL, *Ac=PgAc;
    const int lat=nx*ny;
    long miss=0;
    // 1) AVloc_i[s][k'][q][p] = sum_l A_il[p][q] phi_b(l) f_k'(l,b),  b = node in slot s of i
    #pragma acc parallel loop reduction(+:miss) present(act[0:n],nd[0:4*n],ce[0:2*n],val[0:4*n*POU_K],avl[0:(long)n*POU_AVL],blk[0:n*VIS_BLK],coef[0:nb],E[0:nb*DIM],ij[0:2*nodes],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for reduction(+:miss)
    for(int i=0;i<n;++i){
        double *av=avl+(long)i*POU_AVL;
        #pragma acc loop seq
        for(int q=0;q<POU_AVL;++q) av[q]=0.0;
        if(!act[i]) continue;
        const int cx=ce[2*i], cy=ce[2*i+1];
        const long k0=NeighborPtr[i];
        const int nn=NeighborCount[i];
        #pragma acc loop seq
        for(int jn=-1;jn<nn;++jn){
            int l; double A[VIS_SD][VIS_SD];
            if(jn<0){
                l=i;
                for(int p=0;p<VIS_SD;++p) for(int q=0;q<VIS_SD;++q) A[p][q]=blk[(long)i*VIS_BLK+p*DIM+q];
            }else{
                const long k=k0+jn; const double c=coef[k];
                if(c==0.0) continue;
                l=NeighborInd[k];
                for(int p=0;p<VIS_SD;++p) for(int q=0;q<VIS_SD;++q) A[p][q]=-c*E[k*DIM+p]*E[k*DIM+q];
            }
            #pragma acc loop seq
            for(int s=0;s<4;++s){
                const int b=nd[4*l+s];
                if(b<0) continue;
                const int dx=ij[2*b]-cx+1, dy=ij[2*b+1]-cy+1;
                if(dx<0 || dx>3 || dy<0 || dy>3){ ++miss; continue; }
                const int sl=dx*4+dy;
                for(int kk=0;kk<POU_K;++kk){
                    const double v=val[(4*l+s)*POU_K+kk];
                    if(v==0.0) continue;
                    for(int q=0;q<VIS_SD;++q)
                        for(int p=0;p<VIS_SD;++p)
                            av[((sl*POU_K+kk)*VIS_SD+q)*VIS_SD+p]+=A[p][q]*v;
                }
            }
        }
    }
    PgRangeMiss+=miss;
    // 2) Ac (dense, row-major == column-major since it is symmetric after step 3)
    const long PcL=Pc;
    #pragma acc parallel loop present(Ac[0:PcL*PcL])
    #pragma omp parallel for
    for(long q=0;q<PcL*PcL;++q) Ac[q]=0.0;
    const int NE=25*POU_K*POU_K*VIS_SD*VIS_SD;
    #pragma acc parallel loop collapse(2) present(Ac[0:PcL*PcL],ptr[0:nodes+1],li[0:4*n],ls[0:4*n],ce[0:2*n],val[0:4*n*POU_K],avl[0:(long)n*POU_AVL],ij[0:2*nodes],latn[0:lat])
    #pragma omp parallel for collapse(2)
    for(int a=0;a<nodes;++a){
        for(int e=0;e<NE;++e){
            int t=e;
            const int p=t%VIS_SD; t/=VIS_SD;
            const int q=t%VIS_SD; t/=VIS_SD;
            const int k2=t%POU_K; t/=POU_K;
            const int k1=t%POU_K; t/=POU_K;
            const int off=t;                       // 0..24
            const int bx=ij[2*a]+off/5-2, by=ij[2*a+1]+off%5-2;
            if(bx<0 || bx>=nx || by<0 || by>=ny) continue;
            const int b=latn[bx*ny+by];
            if(b<0) continue;
            double s=0.0;
            #pragma acc loop seq
            for(int m=ptr[a];m<ptr[a+1];++m){
                const int i=li[m], sa=ls[m];
                const int dx=bx-ce[2*i]+1, dy=by-ce[2*i+1]+1;
                if(dx<0 || dx>3 || dy<0 || dy>3) continue;
                const int sl=dx*4+dy;
                s+=val[(4*i+sa)*POU_K+k1]*avl[(long)i*POU_AVL+((sl*POU_K+k2)*VIS_SD+q)*VIS_SD+p];
            }
            const long rw=(long)(a*POU_K+k1)*VIS_SD+p, cl=(long)(b*POU_K+k2)*VIS_SD+q;
            Ac[rw*PcL+cl]=s;
        }
    }
    double reg=VisPouReg;
    for(int attempt=0;attempt<3;++attempt){
        if(pouFactor(Pc,reg)) return 1;
        reg*=1.0e3; ++PgFactorRetry;
    }
    log_printf("VIS-PoU: Cholesky of the coarse operator failed (Pc=%d)\n",Pc);
    return 0;
}

// scale S Ac S (S = diag^-1/2, symmetrised), + reg I, Cholesky in PgAf
static int pouFactor(const int Pc, const double reg)
{
    const long PcL=Pc; double *Ac=PgAc, *Af=PgAf, *sc=PgSc;
    int bad=0;
    #pragma acc parallel loop reduction(+:bad) present(Ac[0:PcL*PcL],sc[0:PcL])
    #pragma omp parallel for reduction(+:bad)
    for(long r=0;r<PcL;++r){
        const double d=Ac[r*PcL+r];
        if(d>0.0 && isfinite(d)) sc[r]=1.0/sqrt(d); else { sc[r]=0.0; ++bad; }
    }
    #pragma acc parallel loop collapse(2) present(Ac[0:PcL*PcL],Af[0:PcL*PcL],sc[0:PcL])
    #pragma omp parallel for collapse(2)
    for(long r=0;r<PcL;++r)
        for(long c=0;c<PcL;++c){
            double v=0.5*(Ac[r*PcL+c]+Ac[c*PcL+r])*sc[r]*sc[c];
            if(r==c) v=(sc[r]>0.0 ? 1.0+reg : 1.0);
            Af[r*PcL+c]=v;
        }
    if(bad) log_printf("VIS-PoU: %d coarse rows with non-positive diagonal (decoupled)\n",bad);
#ifdef _OPENACC
    if(!PgSolver){
        if(cusolverDnCreate(&PgSolver)!=CUSOLVER_STATUS_SUCCESS){ log_printf("VIS-PoU: cusolverDnCreate failed\n"); return 0; }
        cusolverDnSetStream(PgSolver,(cudaStream_t)acc_get_cuda_stream(1));
        PgInfo=(int*)acc_malloc(sizeof(int));
    }
    int lw=0;
    #pragma acc host_data use_device(Af)
    { cusolverDnDpotrf_bufferSize(PgSolver,CUBLAS_FILL_MODE_LOWER,Pc,Af,Pc,&lw); }
    if(lw>PgLwork){ if(PgWork) acc_free(PgWork); PgWork=(double*)acc_malloc(sizeof(double)*(size_t)lw); PgLwork=lw; }
    #pragma acc wait(1)
    #pragma acc host_data use_device(Af)
    { cusolverDnDpotrf(PgSolver,CUBLAS_FILL_MODE_LOWER,Pc,Af,Pc,PgWork,PgLwork,PgInfo); }
    int info=0;
    cudaStreamSynchronize((cudaStream_t)acc_get_cuda_stream(1));
    acc_memcpy_from_device(&info,PgInfo,sizeof(int));
    return info==0;
#else
    // host Cholesky (lower, row-major), CPU builds only
    for(long j=0;j<PcL;++j){
        double d=Af[j*PcL+j];
        for(long k=0;k<j;++k) d-=Af[j*PcL+k]*Af[j*PcL+k];
        if(!(d>0.0)) return 0;
        d=sqrt(d); Af[j*PcL+j]=d;
        #pragma omp parallel for
        for(long i=j+1;i<PcL;++i){
            double s=Af[i*PcL+j];
            for(long k=0;k<j;++k) s-=Af[i*PcL+k]*Af[j*PcL+k];
            Af[i*PcL+j]=s/d;
        }
    }
    return 1;
#endif
}

// -----------------------------------------------------------------------------
// coarse correction  dst = z + V Ac^-1 V^T rc   (queued on async(1))
// -----------------------------------------------------------------------------
static void pouCoarseApply(const double *rc, const double *z, double *dst)
{
    const int n=ParticleCount, n3=ParticleCount*DIM, nodes=PgNodes, Pc=PgPc;
    const long PcL=Pc;
    const int *act=VdAct, *nd=PgNode, *ptr=PgPtr, *li=PgLi, *ls=PgLs;
    const double *val=PgVal, *sc=PgSc; double *cc=PgCc; double *Af=PgAf;
    // cc = S V^T rc
    #pragma acc parallel loop gang async(1) present(cc[0:PcL],sc[0:PcL],ptr[0:nodes+1],li[0:4*n],ls[0:4*n],val[0:4*n*POU_K],rc[0:n3])
    #pragma omp parallel for
    for(int r=0;r<Pc;++r){
        const int p=r%VIS_SD, k=(r/VIS_SD)%POU_K, a=r/(VIS_SD*POU_K);
        double s=0.0;
        #pragma acc loop vector reduction(+:s)
        for(int m=ptr[a];m<ptr[a+1];++m) s+=val[(4*li[m]+ls[m])*POU_K+k]*rc[DIM*li[m]+p];
        cc[r]=s*sc[r];
    }
#ifdef _OPENACC
    #pragma acc host_data use_device(Af,cc)
    { cusolverDnDpotrs(PgSolver,CUBLAS_FILL_MODE_LOWER,Pc,1,Af,Pc,cc,Pc,PgInfo); }
#else
    for(long j=0;j<PcL;++j){ double s=cc[j]; for(long k=0;k<j;++k) s-=Af[j*PcL+k]*cc[k]; cc[j]=s/Af[j*PcL+j]; }
    for(long j=PcL-1;j>=0;--j){ double s=cc[j]; for(long k=j+1;k<PcL;++k) s-=Af[k*PcL+j]*cc[k]; cc[j]=s/Af[j*PcL+j]; }
#endif
    // dst = z + V S cc
    #pragma acc parallel loop async(1) present(z[0:n3],dst[0:n3],act[0:n],nd[0:4*n],val[0:4*n*POU_K],cc[0:PcL],sc[0:PcL])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        for(int d=0;d<DIM;++d){
            const int kk=DIM*i+d;
            if(!act[i]){ dst[kk]=0.0; continue; }
            double s=z[kk];
            if(d<VIS_SD){
                for(int t=0;t<4;++t){
                    const int a=nd[4*i+t];
                    if(a<0) continue;
                    for(int k=0;k<POU_K;++k){
                        const int r=(a*POU_K+k)*VIS_SD+d;
                        s+=val[(4*i+t)*POU_K+k]*sc[r]*cc[r];
                    }
                }
            }
            dst[kk]=s;
        }
    }
}

static void pouLogSettings()
{
    if(VisPou) log_printf("VIS-PoU: partition-of-unity coarse space H=%.3g dp min-fill=%.3g reg=%.1e fields=1,xi (K=%d) lmax power iterations=%d\n",
                          VisPouH,VisPouMinFill,VisPouReg,POU_K,VisDonLmaxIters);
}
static void pouLogSummary()
{
    if(VisPou && PgSetups>0)
        log_printf("Viscosity summary: PoU coarse space: setups=%ld mean coarse dim=%.1f factor-retries=%ld slot-misses=%ld\n",
                   PgSetups,PgPcSum/PgSetups,PgFactorRetry,PgRangeMiss);
}
