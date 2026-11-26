// =============================================================================
// amgx_vis.inc -- AmgX for the implicit viscosity system (see amgx_ppe.inc).
// Included after vis_deeponet.inc (uses its stored operator).
//
// Unknowns: the in-plane components (VIS_SD) of the fluid particles, row
// a*SD+d.  The system is scaled symmetrically by M^-1/2 so that the SPH test
//     sum_i |r_i|^2 / M_i <= target2
// is the plain L2 norm that AmgX monitors.  The out-of-plane component of a
// 2-D run decouples (B_zz = M) and is solved exactly.
// =============================================================================

static int amgxSolveVis(double *x, const double *b, const double *nw, const double *dw,
                        const double target2, int *iterOut, double *res2Out)
{
    *iterOut=0;
    if(!amgxInit()) return 0;
    const double t0=donWtime();
    visAllocateParticleBuffers();
    visAllocateNeighborBuffers();
    const int n=ParticleCount, n3=ParticleCount*DIM;
    {
        double *x0=VdX0;
        #pragma acc parallel loop present(x0[0:n3],x[0:n3])
        #pragma omp parallel for
        for(int k=0;k<n3;++k) x0[k]=x[k];
    }
    visBuildOperator(nw,dw);
    if(AxSt[1].runs==0 || !VdOperatorOk){
        const double err=visVerifyOperator(nw,dw);
        VdOperatorOk=(isfinite(err) && err<1.0e-9);
        log_printf("AmgX VIS: stored-operator check rel.err=%e -> %s\n",err,(VdOperatorOk ? "ok" : "MISMATCH, AmgX VIS disabled"));
        if(!VdOperatorOk) return 0;
    }
    const int SD=VIS_SD;
    amgxAllocMap(n);
    int *act=VdAct, *cnt=AxCnt; const double *blk=VdBlk, *coef=VdCoef, *E=VdE, *dg=VdDiag; const long nb=VdNb;
    #pragma acc parallel loop present(act[0:n],cnt[0:n],coef[0:nb],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        int c=0;
        if(act[i]){
            c=SD;
            for(int jn=0;jn<NeighborCount[i];++jn){
                const long k=NeighborPtr[i]+jn;
                if(coef[k]!=0.0 && act[NeighborInd[k]]) c+=SD;
            }
        }
        cnt[i]=c;           // entries per row (each of the SD rows of particle i)
    }
    #pragma acc update self(act[0:n])
    int rows=0;
    const long nnz=amgxPrefix(n,SD,act,&rows);
    if(rows<1) return 0;
    amgxAllocNnz(nnz);
    int *map=AxMap, *rp=AxRp, *col=AxCol; double *val=AxVal, *bv=AxB, *xv=AxX;
    #pragma acc parallel loop present(act[0:n],map[0:n],rp[0:rows+1],col[0:nnz],val[0:nnz],bv[0:rows],xv[0:rows],blk[0:n*VIS_BLK],coef[0:nb],E[0:nb*DIM],x[0:n3],b[0:n3],Mass[0:ParticleCount],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount])
    #pragma omp parallel for
    for(int i=0;i<n;++i){
        if(!act[i]) continue;
        const int a=map[i];
        const double si=1.0/sqrt(Mass[i]);
        for(int d=0;d<SD;++d){
            const int r=a*SD+d; int q=rp[r];
            for(int e=0;e<SD;++e){ col[q]=a*SD+e; val[q]=blk[(long)i*VIS_BLK+d*DIM+e]*si*si; ++q; }
            for(int jn=0;jn<NeighborCount[i];++jn){
                const long k=NeighborPtr[i]+jn;
                const int j=NeighborInd[k];
                const double c=coef[k];
                if(c==0.0 || !act[j]) continue;
                const double s=si/sqrt(Mass[j]);
                for(int e=0;e<SD;++e){ col[q]=map[j]*SD+e; val[q]=-c*E[k*DIM+d]*E[k*DIM+e]*s; ++q; }
            }
            bv[r]=b[DIM*i+d]*si;
            xv[r]=x[DIM*i+d]/si;
        }
    }
    const double t1=donWtime();
    AxSt[1].csr+=t1-t0;
    int it=0;
    const int ok=amgxRun(1,rows,nnz,AmgxTolFactor*sqrt(target2),&it);
    if(ok){
        #pragma acc parallel loop present(act[0:n],map[0:n],xv[0:rows],x[0:n3],b[0:n3],dg[0:n3],Mass[0:ParticleCount])
        #pragma omp parallel for
        for(int i=0;i<n;++i){
            if(!act[i]) continue;
            const double si=1.0/sqrt(Mass[i]);
            for(int d=0;d<SD;++d) x[DIM*i+d]=xv[map[i]*SD+d]*si;
            for(int d=SD;d<DIM;++d) x[DIM*i+d]=b[DIM*i+d]/dg[DIM*i+d];
        }
    }
    double *r=VdR;
    double res2=visTrueResidual(x,b,r,nw,dw);
    int conv=(ok && isfinite(res2) && res2<=target2);
    if(!conv){
        // keep the AmgX iterate only if it is finite (the Jacobi CG restarts from x)
        const double *x0=VdX0; int bad=!isfinite(res2);
        #pragma acc parallel loop reduction(+:bad) present(x[0:n3])
        #pragma omp parallel for reduction(+:bad)
        for(int k=0;k<n3;++k) if(!isfinite(x[k])) ++bad;
        if(bad || !ok){
            #pragma acc parallel loop present(x[0:n3],x0[0:n3])
            #pragma omp parallel for
            for(int k=0;k<n3;++k) x[k]=x0[k];
        }
        ++AxSt[1].fallback;
        log_printf("AmgX VIS: not converged (iter=%d res2/target2=%e) -> Jacobi CG\n",it,res2/(target2>1e-300 ? target2 : 1e-300));
    }
    ++AxSt[1].runs; AxSt[1].iters+=it;
    AxSt[1].total+=donWtime()-t0;
    *iterOut=it; *res2Out=res2;
    return conv;
}

static void amgxLogSummary()
{
    amgxLogSummaryKind(0,"PPE");
    amgxLogSummaryKind(1,"Viscosity");
}
