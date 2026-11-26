// =============================================================================
// amgx_ppe.inc -- NVIDIA AmgX (algebraic multigrid) as an alternative solver for
// the PPE and (amgx_vis.inc) the implicit viscosity system.  Compiled only with
// -DUSE_AMGX (make ... AMGX=1); other builds are unchanged.
//
// The matrix is the solver's own stored operator (donBuildOperator /
// visBuildOperator, verified against the original operator), assembled as CSR
// on the device every solve (the particle neighbourhood changes every step, so
// the AMG hierarchy cannot be reused).  The AmgX convergence test is ABSOLUTE L2
// with AmgxTolFactor x the SPH target; acceptance always uses the original true
// residual, and an unconverged result is finished by the original Jacobi solver
// (same transactional semantics as the DeepONet paths).
//
// Data file keys:
//   PpeAmgx         0|1    solve the PPE with AmgX                        [0]
//   VisAmgx         0|1    solve the implicit viscosity with AmgX         [0]
//   AmgxPpeConfig   file   AmgX config (default: PBICGSTAB + classical PMIS AMG + L1-Jacobi)
//   AmgxVisConfig   file   AmgX config (default: PCG + classical PMIS AMG + L1-Jacobi)
//   AmgxTolFactor   f      AmgX absolute tolerance = f x SPH target        [0.5]
// Pure-Neumann two-phase PPE systems (singular, non-symmetric) are left to the Jacobi
// solver ('skipped' in the summary).
// =============================================================================
#include <amgx_c.h>
#include <string>
#include <vector>

static int    PpeAmgx=0, VisAmgx=0;
static double AmgxTolFactor=0.5;
static char   AmgxPpeConfig[1024]="", AmgxVisConfig[1024]="";
static const char *AmgxDefaultPpe=
 "config_version=2, solver(s)=PBICGSTAB, s:preconditioner(amg)=AMG, amg:cycle=V, amg:max_iters=1, amg:presweeps=1, amg:postsweeps=1, "
 "amg:max_levels=25, amg:min_coarse_rows=32, amg:coarse_solver=DENSE_LU_SOLVER, amg:dense_lu_num_rows=64, s:max_iters=3000, "
 "s:monitor_residual=1, s:store_res_history=0, amg:algorithm=CLASSICAL, amg:selector=PMIS, amg:interpolator=D2, amg:strength_threshold=0.25, "
 "amg:smoother(sm)=JACOBI_L1, sm:max_iters=2, sm:relaxation_factor=1.0";
static const char *AmgxDefaultVis=
 "config_version=2, solver(s)=PCG, s:preconditioner(amg)=AMG, amg:cycle=V, amg:max_iters=1, amg:presweeps=1, amg:postsweeps=1, "
 "amg:max_levels=25, amg:min_coarse_rows=32, amg:coarse_solver=DENSE_LU_SOLVER, amg:dense_lu_num_rows=64, s:max_iters=3000, "
 "s:monitor_residual=1, s:store_res_history=0, amg:algorithm=CLASSICAL, amg:selector=PMIS, amg:interpolator=D2, amg:strength_threshold=0.25, "
 "amg:smoother(sm)=JACOBI_L1, sm:max_iters=2, sm:relaxation_factor=1.0";

// one record per system kind (0 PPE, 1 viscosity)
struct AmgxStat { long runs, iters, fallback, skipped; double csr, upload, setup, solve, total; };
static AmgxStat AxSt[2]={{0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0}};
static int AmgxReady=0;
static AMGX_resources_handle AmgxRsrc;
static AMGX_config_handle   AmgxBaseCfg;
static std::string AmgxCfgText[2];

static void amgxQuiet(const char *, int) {}

static int amgxParseDataLine(const char *buf)
{
    if(sscanf(buf," PpeAmgx %d",&PpeAmgx)==1) return 1;
    if(sscanf(buf," VisAmgx %d",&VisAmgx)==1) return 1;
    if(sscanf(buf," AmgxTolFactor %lf",&AmgxTolFactor)==1) return 1;
    if(sscanf(buf," AmgxPpeConfig %1023s",AmgxPpeConfig)==1) return 1;
    if(sscanf(buf," AmgxVisConfig %1023s",AmgxVisConfig)==1) return 1;
    return 0;
}

static std::string amgxReadConfig(const char *path, const char *dflt)
{
    if(!path[0]) return std::string(dflt);
    FILE *f=fopen(path,"r");
    if(!f){ log_printf("AmgX: cannot open %s, using the default config\n",path); return std::string(dflt); }
    std::string s; char b[4096]; size_t k;
    while((k=fread(b,1,sizeof b,f))>0) s.append(b,k);
    fclose(f);
    while(!s.empty() && (s.back()=='\n' || s.back()=='\r' || s.back()==' ')) s.pop_back();
    return s;
}

static int amgxInit()
{
    if(AmgxReady) return AmgxReady>0;
    AmgxReady=-1;
    if(AMGX_initialize()!=AMGX_RC_OK){ log_printf("AmgX: initialise failed -> Jacobi solvers\n"); return 0; }
    AMGX_register_print_callback(amgxQuiet);
    AmgxCfgText[0]=amgxReadConfig(AmgxPpeConfig,AmgxDefaultPpe);
    AmgxCfgText[1]=amgxReadConfig(AmgxVisConfig,AmgxDefaultVis);
    if(AMGX_config_create(&AmgxBaseCfg,AmgxCfgText[1].c_str())!=AMGX_RC_OK){ log_printf("AmgX: bad config\n"); return 0; }
    if(AMGX_resources_create_simple(&AmgxRsrc,AmgxBaseCfg)!=AMGX_RC_OK){ log_printf("AmgX: resources failed\n"); return 0; }
    AmgxReady=1;
    return 1;
}

static void amgxLogSettings()
{
    if(!PpeAmgx && !VisAmgx) return;
    log_printf("AmgX: PPE=%d VIS=%d tol factor=%g\n  PPE config: %s\n  VIS config: %s\n",PpeAmgx,VisAmgx,AmgxTolFactor,
               amgxReadConfig(AmgxPpeConfig,AmgxDefaultPpe).c_str(),amgxReadConfig(AmgxVisConfig,AmgxDefaultVis).c_str());
}

// device CSR scratch (shared by both kinds; the solves never overlap)
static int    AxRowAlloc=0; static long AxNnzAlloc=0;
static int   *AxRp=NULL, *AxCol=NULL; static double *AxVal=NULL, *AxB=NULL, *AxX=NULL;
static int   *AxMap=NULL, *AxCnt=NULL; static int AxMapAlloc=0;
static void amgxAllocRows(const int rows)
{
    if(rows<=AxRowAlloc) return;
    if(AxRp){ int *rp=AxRp; double *b=AxB,*x=AxX; const int m=AxRowAlloc;
        #pragma acc exit data delete(rp[0:m+1],b[0:m],x[0:m])
        free(rp); free(b); free(x); }
    AxRowAlloc=rows+rows/4+16; const int m=AxRowAlloc;
    AxRp=(int*)malloc(sizeof(int)*(m+1)); AxB=(double*)malloc(sizeof(double)*m); AxX=(double*)malloc(sizeof(double)*m);
    int *rp=AxRp; double *b=AxB,*x=AxX;
    #pragma acc enter data create(rp[0:m+1],b[0:m],x[0:m])
}
static void amgxAllocNnz(const long nnz)
{
    if(nnz<=AxNnzAlloc) return;
    if(AxCol){ int *c=AxCol; double *v=AxVal; const long m=AxNnzAlloc;
        #pragma acc exit data delete(c[0:m],v[0:m])
        free(c); free(v); }
    AxNnzAlloc=nnz+nnz/4+64; const long m=AxNnzAlloc;
    AxCol=(int*)malloc(sizeof(int)*m); AxVal=(double*)malloc(sizeof(double)*m);
    int *c=AxCol; double *v=AxVal;
    #pragma acc enter data create(c[0:m],v[0:m])
}
static void amgxAllocMap(const int n)
{
    if(n<=AxMapAlloc) return;
    if(AxMap){ int *mp=AxMap,*cn=AxCnt; const int m=AxMapAlloc;
        #pragma acc exit data delete(mp[0:m],cn[0:m])
        free(mp); free(cn); }
    AxMapAlloc=n+n/4+16; const int m=AxMapAlloc;
    AxMap=(int*)malloc(sizeof(int)*m); AxCnt=(int*)malloc(sizeof(int)*m);
    int *mp=AxMap,*cn=AxCnt;
    #pragma acc enter data create(mp[0:m],cn[0:m])
}

// host prefix sums: per-particle row counts (cnt, rows per particle = rpp) -> map, rowptr.
// Returns nnz; rows = number of active particles x rpp.
static long amgxPrefix(const int n, const int rpp, const int *actHost, int *rowsOut)
{
    int *cnt=AxCnt, *map=AxMap;
    #pragma acc update self(cnt[0:n])
    int rows=0;
    for(int i=0;i<n;++i){ map[i]=(actHost[i] ? rows/rpp : -1); if(actHost[i]) rows+=rpp; }
    amgxAllocRows(rows);
    int *rp=AxRp;
    long nnz=0; int r=0;
    rp[0]=0;
    for(int i=0;i<n;++i){
        if(!actHost[i]) continue;
        for(int d=0;d<rpp;++d){ nnz+=cnt[i]; rp[++r]=(int)nnz; }
    }
    #pragma acc update device(map[0:n],rp[0:rows+1])
    *rowsOut=rows;
    return nnz;
}

// AmgX solve of the device CSR (AxRp, AxCol, AxVal), rhs AxB, initial guess / result AxX
static int amgxRun(const int kind, const int rows, const long nnz, const double absTol, int *iters)
{
    char extra[160];
    snprintf(extra,sizeof extra,", s:convergence=ABSOLUTE, s:norm=L2, s:tolerance=%.17g",absTol);
    AMGX_config_handle cfg; AMGX_matrix_handle A; AMGX_vector_handle B, X; AMGX_solver_handle S;
    if(AMGX_config_create(&cfg,(AmgxCfgText[kind]+extra).c_str())!=AMGX_RC_OK) return 0;
    AMGX_matrix_create(&A,AmgxRsrc,AMGX_mode_dDDI);
    AMGX_vector_create(&B,AmgxRsrc,AMGX_mode_dDDI);
    AMGX_vector_create(&X,AmgxRsrc,AMGX_mode_dDDI);
    AMGX_solver_create(&S,AmgxRsrc,AMGX_mode_dDDI,cfg);
    int *rp=AxRp, *col=AxCol; double *val=AxVal, *b=AxB, *x=AxX;
    int ok=1;
    const double t0=donWtime();
    #pragma acc host_data use_device(rp,col,val,b,x)
    {
        if(AMGX_matrix_upload_all(A,rows,(int)nnz,1,1,rp,col,val,NULL)!=AMGX_RC_OK) ok=0;
        if(ok && AMGX_vector_upload(B,rows,1,b)!=AMGX_RC_OK) ok=0;
        if(ok && AMGX_vector_upload(X,rows,1,x)!=AMGX_RC_OK) ok=0;
    }
    cudaDeviceSynchronize();
    const double t1=donWtime();
    if(ok && AMGX_solver_setup(S,A)!=AMGX_RC_OK) ok=0;
    cudaDeviceSynchronize();
    const double t2=donWtime();
    if(ok && AMGX_solver_solve(S,B,X)!=AMGX_RC_OK) ok=0;
    cudaDeviceSynchronize();
    const double t3=donWtime();
    int it=0; AMGX_solver_get_iterations_number(S,&it);
    AMGX_SOLVE_STATUS st=AMGX_SOLVE_FAILED; AMGX_solver_get_status(S,&st);
    if(ok){
        #pragma acc host_data use_device(x)
        { if(AMGX_vector_download(X,x)!=AMGX_RC_OK) ok=0; }
        cudaDeviceSynchronize();
    }
    AxSt[kind].upload+=t1-t0; AxSt[kind].setup+=t2-t1; AxSt[kind].solve+=t3-t2;
    AMGX_solver_destroy(S); AMGX_vector_destroy(X); AMGX_vector_destroy(B); AMGX_matrix_destroy(A); AMGX_config_destroy(cfg);
    *iters=it;
    return ok && st!=AMGX_SOLVE_DIVERGED;
}

// -----------------------------------------------------------------------------
// PPE: hook after donSolvePpe().  Same contract: on return either rr <= tol^2 bb
// or PressureP / PpeR / PpeR0 are a valid start for the Jacobi BiCGStab.
// -----------------------------------------------------------------------------
static void amgxSolvePpe(const double bb, double *rrIO, int *iterOut)
{
    if(!PpeAmgx || !(bb>1.0e-60) || *rrIO<=PpeTolerance*PpeTolerance*bb) return;
    // Pure-Neumann (closed two-phase) systems: A (non-symmetric) is singular and the
    // SPH equation is Pi A p = b with the mean projection Pi; A p = b itself is not
    // consistent, and AmgX on it diverges (tested: V05, 543 mean iterations, 20%
    // failures).  These systems are left to the Jacobi BiCGStab.
    if(TwoPhaseActive && PpePureNeumann){ ++AxSt[0].skipped; return; }
    if(!amgxInit()) return;
    const double t0=donWtime();
    const int n=ParticleCount;
    donAllocateParticleBuffers();
    donAllocateNeighborBuffers();
    {
        double *p0=DonP0;
        #pragma acc parallel loop present(p0[0:n],PressureP[0:n])
        #pragma omp parallel for
        for(int i=0;i<n;++i) p0[i]=PressureP[i];
    }
    donBuildOperator();
    if(AxSt[0].runs==0 || !DonOperatorOk){
        const double err=donVerifyOperator();
        DonOperatorOk=(isfinite(err) && err<1.0e-9);
        log_printf("AmgX PPE: stored-operator check rel.err=%e -> %s\n",err,(DonOperatorOk ? "ok" : "MISMATCH, AmgX PPE disabled"));
        if(!DonOperatorOk) return;
    }
    amgxAllocMap(n);
    int *act=DonAct, *cnt=AxCnt; const double *ad=DonAdiag, *coef=DonCoef; const long nb=DonNbCount;
    #pragma acc parallel loop present(act[0:n],cnt[0:n],coef[0:nb],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        int c=0;
        if(act[i]){
            c=1;
            for(int jn=0;jn<NeighborCount[i];++jn){
                const long k=NeighborPtr[i]+jn;
                if(coef[k]!=0.0 && act[NeighborInd[k]]) ++c;
            }
        }
        cnt[i]=c;
    }
    #pragma acc update self(act[0:n])
    int rows=0;
    const long nnz=amgxPrefix(n,1,act,&rows);
    if(rows<1) return;
    amgxAllocNnz(nnz);
    int *map=AxMap, *rp=AxRp, *col=AxCol; double *val=AxVal, *bv=AxB, *xv=AxX;
    #pragma acc parallel loop present(act[0:n],map[0:n],rp[0:rows+1],col[0:nnz],val[0:nnz],bv[0:rows],xv[0:rows],ad[0:n],coef[0:nb],PpeB[0:n],PressureP[0:n],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(!act[i]) continue;
        const int r=map[i]; int q=rp[r];
        col[q]=r; val[q]=ad[i]; ++q;
        for(int jn=0;jn<NeighborCount[i];++jn){
            const long k=NeighborPtr[i]+jn;
            const int j=NeighborInd[k];
            if(coef[k]!=0.0 && act[j]){ col[q]=map[j]; val[q]=-coef[k]; ++q; }
        }
        bv[r]=PpeB[i]; xv[r]=PressureP[i];
    }
    const double t1=donWtime();
    AxSt[0].csr+=t1-t0;
    int it=0;
    const int ok=amgxRun(0,rows,nnz,AmgxTolFactor*PpeTolerance*sqrt(bb),&it);
    if(ok){
        #pragma acc parallel loop present(act[0:n],map[0:n],xv[0:rows],PressureP[0:n])
        #pragma omp parallel for
        for(int i=0;i<n;++i) if(act[i]) PressureP[i]=xv[map[i]];
    }
    const double rr0=*rrIO;
    *rrIO=donPrepareFallback((ok ? 0.0 : rr0),rr0);   // true residual (applyPpeOperator); restores p0 on NaN
    *iterOut=it;
    ++AxSt[0].runs; AxSt[0].iters+=it;
    if(!(*rrIO<=PpeTolerance*PpeTolerance*bb)){
        ++AxSt[0].fallback;
        log_printf("AmgX PPE: not converged (iter=%d rel=%e) -> Jacobi BiCGStab\n",it,sqrt(*rrIO/bb));
    }
    AxSt[0].total+=donWtime()-t0;
}

static void amgxLogSummaryKind(const int kind, const char *name)
{
    const AmgxStat &s=AxSt[kind];
    if(s.runs==0 && s.skipped==0) return;
    const double per=(s.runs>0 ? 1e3/s.runs : 0.0);
    log_printf("%s summary: AmgX: runs=%ld mean iter=%.2f fallback=%ld skipped=%ld total=%.3fs  per solve [ms]: csr=%.3f upload=%.3f setup=%.3f solve=%.3f total=%.3f\n",
               name,s.runs,(s.runs>0 ? (double)s.iters/s.runs : 0.0),s.fallback,s.skipped,s.total,
               s.csr*per,s.upload*per,s.setup*per,s.solve*per,s.total*per);
}
