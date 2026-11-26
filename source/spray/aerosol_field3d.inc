// =============================================================================
// aerosol_field3d.inc -- Eulerian aerosol mass fractions on the 3D FVM grid
//
// 3D build counterpart of aerosol_field.inc (same model, data keys and output
// columns); included after fvm3d_air.inc and before spray.inc.
//
// Model of Liang, Erkan, Solans, Suzuki, Front. Energy Res. 8:102 (2020): each
// size class i is a passive gas species with mass fraction Y_i
//   dY_i/dt + u.grad Y_i = div(D grad Y_i) - Lambda_i Y_i,   D = nu_g + nu_t/Sc_t
// on the FVM cells, on the device (van Leer TVD advection, explicit diffusion,
// CFL subcycling, no flux through closed faces), uniform at t = 0.
// Sink of one spray parcel (liquid volume Vp of droplets d_D) in its cell:
//   Lambda_i = 1.5 eta_total,i |U_D| Vp / (d_D dx dy dz),  alpha = Vp/(dx dy dz)
// (eqs. 14-17; efficiencies eqs. 8-13 as in aerosol_field.inc). The parcels are
// on the host: each one appends (cell, Lambda_i dt) to an event list, which is
// summed on the device and applied as Y_i *= exp(-sum Lambda_i dt) before every
// transport and every output.
//
// Data file keys: as aerosol_field.inc, with
//   AerosolFieldProbe   x y z          sampling point (history column)
// =============================================================================

#define AERF_MAX 16

static int    AerFCount=0;
static double AerFDiameter[AERF_MAX], AerFMass0[AERF_MAX];
static double AerFDensity=5680.0;
static char   AerFVtkPattern[1024]="", AerFHistoryFile[1024]="aerosol_field.csv";
static double AerFProbe[3]={0.0,0.0,0.0};
static int    AerFHaveProbe=0;

static double *AerFY=NULL;               // [class*FvmN + cell], Y_i / sum_i Y0_i
static double *AerFL=NULL;               // accumulated Lambda dt, same layout
static double *AerFFx=NULL, *AerFFy=NULL, *AerFFz=NULL;
static double  AerFM0[AERF_MAX];         // initial mass (normalised units)
static int     AerFFrame=0;
static double  AerFLastTime=0.0;
static int     AerFEvCount=0, AerFEvCap=0;
static int    *AerFEvCell=NULL;
static double *AerFEvLam=NULL;           // [event*AerFCount + class]

static void aerFieldWriteHistory();
static void aerFieldWriteVtk();

static int aerFieldParseDataLine(const char *buf)
{
    char key[256];
    if(sscanf(buf," %255s",key)!=1 || strncmp(key,"AerosolField",12)) return 0;
    double d,m;
    if(!strcmp(key,"AerosolField") && sscanf(buf," AerosolField %lf %lf",&d,&m)==2){
        if(AerFCount<AERF_MAX){ AerFDiameter[AerFCount]=d; AerFMass0[AerFCount]=m; ++AerFCount; }
        return 1;
    }
    if(sscanf(buf," AerosolFieldDensity %lf",&AerFDensity)==1) return 1;
    if(sscanf(buf," AerosolFieldVtk %1023s",AerFVtkPattern)==1) return 1;
    if(sscanf(buf," AerosolFieldHistory %1023s",AerFHistoryFile)==1) return 1;
    const int n=sscanf(buf," AerosolFieldProbe %lf %lf %lf",&AerFProbe[0],&AerFProbe[1],&AerFProbe[2]);
    if(n>=2){ if(n==2) AerFProbe[2]=0.0; AerFHaveProbe=1; return 1; }
    return 0;
}

// single-droplet collection efficiency of class i (eqs. 8-13)
static double aerFieldEfficiency(const int i, const double Urel, const double alpha)
{
    const double dP=AerFDiameter[i], dD=AerosolDropletDiameter;
    const double muG=FvmP.mu_g, rhoG=FvmP.rho_g, nuG=muG/rhoG;
    const double muD=ShearViscosity[1]>0.0 ? ShearViscosity[1] : 1.0e-3;
    double eImp=0.0, eDiff=0.0;
    if(Urel>1.0e-9){
        const double St=AerFDensity*dP*dP*Urel/(9.0*muG*dD);
        const double q=St/(St+0.5);
        if(St>=0.2) eImp=q*q;
        else if(St>0.0833) eImp=8.57*q*q*(St-0.08336);
        const double C=2.609*sqrt(2.0*AerosolMeanFreePath)/sqrt(dP);
        const double Dd=1.380649e-23*AerosolGasTemperature*C/(3.0*M_PI*nuG*rhoG*dP);
        const double Pe=dD*Urel/Dd;
        eDiff=1.0/sqrt(2.0*Pe*dD);
    }
    const double a=fmax(fmin(alpha,1.0),0.0), a3=cbrt(a);
    const double J=1.0-1.2*a3+0.2*a*a, K=1.0-1.8*a3+0.2*a*a;
    const double s=muD/muG, R=dP/dD, r=R/(1.0+R);
    const double eInt=(1.0-a)/(J+s*K)*(r+0.5*r*r*(3.0*s+4.0));
    eImp=fmin(fmax(eImp,0.0),1.0); eDiff=fmin(fmax(eDiff,0.0),1.0);
    const double eI=fmin(fmax(eInt,0.0),1.0);
    return 1.0-(1.0-eImp)*(1.0-eI)*(1.0-eDiff);
}

static void aerFieldInitialize()
{
    if(AerFCount==0) return;
    if(!FvmEnabled){ log_printf("AerosolField needs the FVM air (FvmData)\n"); exit(EXIT_FAILURE); }
    const int N=FvmN, NC=AerFCount*FvmN;
    double m0tot=0.0; for(int i=0;i<AerFCount;++i) m0tot+=AerFMass0[i];
    AerFY=(double *)malloc((size_t)NC*sizeof(double));
    AerFL=(double *)malloc((size_t)NC*sizeof(double));
    for(int s=0;s<AerFCount;++s) for(int q=0;q<N;++q){
        AerFY[s*N+q]=FvmSolid[q] ? 0.0 : AerFMass0[s]/m0tot;
        AerFL[s*N+q]=0.0;
    }
    AerFFx=fvm3Alloc(); AerFFy=fvm3Alloc(); AerFFz=fvm3Alloc();
    double *Y=AerFY, *L=AerFL, *Fx=AerFFx, *Fy=AerFFy, *Fz=AerFFz;
    #pragma acc enter data copyin(Y[0:NC],L[0:NC],Fx[0:N],Fy[0:N],Fz[0:N])
    for(int s=0;s<AerFCount;++s) AerFM0[s]=AerFMass0[s]/m0tot*FvmFluidCount;
    AerFLastTime=Time;
    log_printf("AerosolField 3D: %d size classes on the FVM grid, rho_P=%e d_D=%e m (Liang et al. 2020 eqs. 8-17)\n",
               AerFCount,AerFDensity,AerosolDropletDiameter);
    for(int s=0;s<AerFCount;++s)
        log_printf("AerosolField: class %d d_P=%e m M0=%e  eta_total(U=1,5,20 m/s)=%e %e %e\n",s,AerFDiameter[s],AerFMass0[s],
                   aerFieldEfficiency(s,1.0,0.0),aerFieldEfficiency(s,5.0,0.0),aerFieldEfficiency(s,20.0,0.0));
    FILE *fp=fopen(AerFHistoryFile,"w");
    if(fp){
        fprintf(fp,"time,total");
        for(int s=0;s<AerFCount;++s) fprintf(fp,",c%d_%.3fum",s,AerFDiameter[s]*1e6);
        if(AerFHaveProbe) fprintf(fp,",probe_total");
        fprintf(fp,"\n"); fclose(fp);
    }
    aerFieldWriteHistory();   // t = 0 row
    aerFieldWriteVtk();       // frame 0 = initial state
}

// sink of one spray parcel (liquid volume Vp, droplet velocity ud, air velocity ua) over dt
static void aerFieldParcelSink(const double x, const double y, const double z, const double ud[3],
                               const double ua[3], const double Vp, const double dt)
{
    if(AerFCount==0) return;
    int i,j,k;
    if(!fvmCellOf(x,y,z,&i,&j,&k)) return;
    const double cellV=FvmG.dx*FvmG.dy*FvmG.dz, dD=AerosolDropletDiameter;
    const double UD=sqrt(ud[0]*ud[0]+ud[1]*ud[1]+ud[2]*ud[2]);
    const double rx=ua[0]-ud[0], ry=ua[1]-ud[1], rz=ua[2]-ud[2], Urel=sqrt(rx*rx+ry*ry+rz*rz);
    const double alpha=Vp/cellV;
    if(AerFEvCount>=AerFEvCap){
        AerFEvCap=(AerFEvCap>0) ? 2*AerFEvCap : 4096;
        AerFEvCell=(int *)realloc(AerFEvCell,AerFEvCap*sizeof(int));
        AerFEvLam=(double *)realloc(AerFEvLam,(size_t)AerFEvCap*AerFCount*sizeof(double));
    }
    AerFEvCell[AerFEvCount]=fvm3Index(i,j,k);
    for(int s=0;s<AerFCount;++s)
        AerFEvLam[AerFEvCount*AerFCount+s]=1.5*aerFieldEfficiency(s,Urel,alpha)*UD*Vp/(dD*cellV)*dt;
    ++AerFEvCount;
}

// sum the pending sink events on the device and apply Y *= exp(-sum Lambda dt)
static void aerFieldApplySink()
{
    if(AerFCount==0) return;
    const int N=FvmN, NC=AerFCount*FvmN, NS=AerFCount, ne=AerFEvCount;
    double *Y=AerFY, *L=AerFL;
    if(ne>0){
        int *cell=AerFEvCell; double *lam=AerFEvLam;
        #pragma acc parallel loop copyin(cell[0:ne],lam[0:ne*NS]) present(L[0:NC])
        for(int e=0;e<ne;++e){
            #pragma acc loop seq
            for(int s=0;s<NS;++s){
                #pragma acc atomic update
                L[s*N+cell[e]]+=lam[e*NS+s];
            }
        }
        AerFEvCount=0;
    }
    #pragma acc parallel loop present(Y[0:NC],L[0:NC])
    for(int q=0;q<NC;++q){
        if(L[q]>0.0){ Y[q]*=exp(-L[q]); L[q]=0.0; }
    }
}

#pragma acc routine seq
static inline double aerVanLeer(const double r){ return (r>0.0) ? 2.0*r/(1.0+r) : 0.0; }

// advect + diffuse all classes over dt with the device face velocities
static void aerFieldTransport(const double dt)
{
    if(AerFCount==0 || !(dt>0.0)) return;
    aerFieldApplySink();
    const int nx=FvmG.nx, ny=FvmG.ny, nz=FvmG.nz, NJ=FvmNJ, NK=FvmNK, N=FvmN, NC=AerFCount*FvmN;
    const int sI=NJ*NK, sJ=NK, sK=1;
    const double dx=FvmG.dx, dy=FvmG.dy, dz=FvmG.dz, nu=FvmP.mu_g/FvmP.rho_g, isc=1.0/FvmP.sc_t;
    const unsigned char *S=FvmSolid;
    const double *U=FvmU, *V=FvmV, *W=FvmW, *Nut=FvmNut;
    double *Yall=AerFY, *Fx=AerFFx, *Fy=AerFFy, *Fz=AerFFz;
    double umax=0.0, vmax=0.0, wmax=0.0, nmax=0.0;
    #pragma acc parallel loop present(U[0:N],V[0:N],W[0:N],Nut[0:N]) reduction(max:umax,vmax,wmax,nmax)
    for(int q=0;q<N;++q){
        umax=fmax(umax,fabs(U[q])); vmax=fmax(vmax,fabs(V[q])); wmax=fmax(wmax,fabs(W[q])); nmax=fmax(nmax,Nut[q]);
    }
    const double Dmax=nu+nmax*isc;
    const double cmax=fmax((umax/dx+vmax/dy+wmax/dz)*dt, 2.0*Dmax*dt*(1.0/(dx*dx)+1.0/(dy*dy)+1.0/(dz*dz)));
    const int nsub=(int)ceil(cmax/0.4)>1 ? (int)ceil(cmax/0.4) : 1;
    const double h=dt/nsub;
    for(int s=0;s<AerFCount;++s){
        const int off=s*N;
        for(int sub=0;sub<nsub;++sub){
            #pragma acc parallel loop collapse(3) present(S[0:N],U[0:N],V[0:N],W[0:N],Nut[0:N],Yall[0:NC],Fx[0:N],Fy[0:N],Fz[0:N])
            for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) for(int k=1;k<=nz;++k){
                const int q=((i+1)*NJ+(j+1))*NK+(k+1);
                const double *Y=Yall+off;
                #pragma acc loop seq
                for(int c=0;c<3;++c){
                    const int st=(c==0)?sI:((c==1)?sJ:sK);
                    const double hs=(c==0)?dx:((c==1)?dy:dz);
                    double F=0.0;
                    if(!S[q] && !S[q+st]){
                        const double uf=(c==0)?U[q]:((c==1)?V[q]:W[q]);
                        const int up=(uf>=0.0)?q:q+st, dn=(uf>=0.0)?q+st:q, uu=(uf>=0.0)?q-st:q+2*st;
                        const double yU=Y[up], yD=Y[dn], yUU=S[uu] ? yU : Y[uu];
                        const double den=yD-yU;
                        const double r=(fabs(den)>1.0e-300) ? (yU-yUU)/den : 0.0;
                        const double D=nu+0.5*(Nut[q]+Nut[q+st])*isc;
                        F=uf*(yU+0.5*aerVanLeer(r)*den)-D*(Y[q+st]-Y[q])/hs;
                    }
                    if(c==0) Fx[q]=F; else if(c==1) Fy[q]=F; else Fz[q]=F;
                }
            }
            #pragma acc parallel loop collapse(3) present(S[0:N],Yall[0:NC],Fx[0:N],Fy[0:N],Fz[0:N])
            for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) for(int k=1;k<=nz;++k){
                const int q=((i+1)*NJ+(j+1))*NK+(k+1);
                if(S[q]) continue;
                double y=Yall[off+q]-h*((Fx[q]-Fx[q-sI])/dx+(Fy[q]-Fy[q-sJ])/dy+(Fz[q]-Fz[q-sK])/dz);
                Yall[off+q]=(y<0.0) ? 0.0 : y;
            }
        }
    }
}

static double aerFieldClassMass(const int s)
{
    const int N=FvmN, NC=AerFCount*FvmN, off=s*FvmN;
    const unsigned char *S=FvmSolid;
    const double *Y=AerFY;
    double m=0.0;
    #pragma acc parallel loop present(S[0:N],Y[0:NC]) reduction(+:m)
    for(int q=0;q<N;++q) if(!S[q]) m+=Y[off+q];
    return m;
}

static void aerFieldWriteHistory()
{
    if(AerFCount==0) return;
    aerFieldApplySink();
    FILE *fp=fopen(AerFHistoryFile,"a");
    if(!fp) return;
    double mt=0.0, m0=0.0, ms[AERF_MAX];
    for(int s=0;s<AerFCount;++s){ ms[s]=aerFieldClassMass(s); mt+=ms[s]; m0+=AerFM0[s]; }
    fprintf(fp,"%.9e,%.9e",Time,mt/m0);
    for(int s=0;s<AerFCount;++s) fprintf(fp,",%.9e",ms[s]/AerFM0[s]);
    if(AerFHaveProbe){
        int i,j,k; double yp=0.0;
        if(fvmCellOf(AerFProbe[0],AerFProbe[1],AerFProbe[2],&i,&j,&k)){
            const int N=FvmN, NC=AerFCount*FvmN, q=fvm3Index(i,j,k);
            double *Y=AerFY;
            for(int s=0;s<AerFCount;++s){
                const int p=s*N+q;
                #pragma acc update self(Y[p:1])
                yp+=Y[p];
            }
            (void)NC;
        }
        fprintf(fp,",%.9e",yp);
    }
    fprintf(fp,"\n"); fclose(fp);
}

// cell data Y_i/Y0_i of every class and of the total (legacy VTK, binary)
static void aerFieldWriteVtk()
{
    if(AerFCount==0 || AerFVtkPattern[0]=='\0') return;
    aerFieldApplySink();
    {
        double *Y=AerFY; const int NC=AerFCount*FvmN;
        #pragma acc update self(Y[0:NC])
    }
    char name[1100]; snprintf(name,sizeof(name),AerFVtkPattern,AerFFrame++);
    FILE *fp=fopen(name,"wb");
    if(!fp) return;
    const int nx=FvmG.nx, ny=FvmG.ny, nz=FvmG.nz, N=FvmN;
    double y0[AERF_MAX], y0t=0.0;
    for(int s=0;s<AerFCount;++s){ y0[s]=AerFM0[s]; y0t+=y0[s]; }
    const double nf=(double)FvmFluidCount;
    fprintf(fp,"# vtk DataFile Version 3.0\nAerosol mass fraction / initial, t=%e\nBINARY\nDATASET STRUCTURED_POINTS\n",Time);
    fprintf(fp,"DIMENSIONS %d %d %d\nORIGIN %e %e %e\nSPACING %e %e %e\n",nx+1,ny+1,nz+1,
            FvmG.xmin,FvmG.ymin,FvmG.zmin,FvmG.dx,FvmG.dy,FvmG.dz);
    fprintf(fp,"CELL_DATA %d\n",nx*ny*nz);
    for(int s=-1;s<AerFCount;++s){
        if(s<0) fprintf(fp,"SCALARS total float 1\nLOOKUP_TABLE default\n");
        else    fprintf(fp,"SCALARS class%d_%.3fum float 1\nLOOKUP_TABLE default\n",s,AerFDiameter[s]*1e6);
        for(int k=1;k<=nz;++k) for(int j=1;j<=ny;++j) for(int i=1;i<=nx;++i){
            const int q=fvm3Index(i,j,k);
            double val=-1.0;
            if(!FvmSolid[q]){
                if(s<0){ val=0.0; for(int c=0;c<AerFCount;++c) val+=AerFY[c*N+q]; val/=y0t/nf; }
                else val=AerFY[s*N+q]/(y0[s]/nf);
            }
            fvm3PutFloat(fp,(float)val);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

static void aerFieldLogSummary()
{
    if(AerFCount==0) return;
    aerFieldApplySink();
    double mt=0.0, m0=0.0;
    for(int s=0;s<AerFCount;++s){ mt+=aerFieldClassMass(s); m0+=AerFM0[s]; }
    log_printf("AerosolField summary: t=%e total mass / initial = %e\n",Time,mt/m0);
}
