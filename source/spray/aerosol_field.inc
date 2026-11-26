// =============================================================================
// aerosol_field.inc -- Eulerian aerosol mass fractions on the FVM air grid
//
// Included once by main_implicit.cpp, after fvm_air.inc and before spray.inc.
//
// Model of Liang, Erkan, Solans, Suzuki, Front. Energy Res. 8:102 (2020): each
// aerosol size class i is a passive gas species with mass fraction Y_i
//   dY_i/dt + u.grad Y_i = div(nu_g grad Y_i) - Lambda_i Y_i            (eq. 2)
// on the FVM cells (host copy of the face velocities; van Leer TVD advection,
// explicit diffusion, CFL subcycling, no flux through solid faces). Initially
// uniform in the vessel.
// Sink (eqs. 14-17): a spray parcel carrying the liquid volume Vp of droplets
// of diameter d_D removes, from the cell that contains it,
//   Lambda_i = eta_total,i (pi/4) d_D^2 |U_D| N / (dx dy),  N = Vp / (pi/6 d_D^3)
//            = 1.5 eta_total,i |U_D| Vp / (d_D dx dy)            (2D, per depth)
// applied as Y_i *= exp(-Lambda_i dt). Single-droplet efficiencies as printed:
//   St = rho_P d_P^2 |U_G-U_D| / (9 mu_G d_D)                                (9)
//   eta_imp = 0 (St<=0.0833); 8.57 (St/(St+0.5))^2 (St-0.08336) (St<0.2);
//             (St/(St+0.5))^2 (St>=0.2)                                     (8)
//   eta_int = (1-a)/(J+s K) { R/(1+R) + 1/2 (R/(1+R))^2 (3s+4) },
//             J = 1 - 6/5 a^(1/3) + 1/5 a^2, K = 1 - 9/5 a^(1/3) + 1/5 a^2,
//             s = mu_D/mu_G, R = d_P/d_D, a = local liquid volume fraction (10)
//   eta_diff = (2 Pe d_D)^(-1/2) (SI, as printed), Pe = d_D |U_G-U_D| / D_diff,
//             D_diff = k_B T C / (3 pi nu_G rho_G d_P),
//             C = 2.609 sqrt(2 l) / sqrt(d_P)                            (11)(12)
//   eta_total = 1 - (1-eta_imp)(1-eta_int)(1-eta_diff)                      (13)
// Droplet diameter, gas temperature and mean free path come from the Aerosol*
// keys of aerosol.inc (AerosolDropletDiameter, AerosolGasTemperature,
// AerosolMeanFreePath); rho_G, mu_G from the FVM data file.
//
// Data file keys:
//   AerosolField        d_P[m] M0      one line per size class (M0: initial mass
//                                      in any unit, e.g. mg of Table 3; Y0 is
//                                      proportional to it)       (enables the model)
//   AerosolFieldDensity rho_P [kg/m3]  5680 (ZrO2)
//   AerosolFieldVtk     pattern        e.g. aer_%04d.vtk, with every SPH VTK frame
//   AerosolFieldHistory path           aerosol_field.csv, with every prof output
//   AerosolFieldProbe   x y            sampling point (history column)
// =============================================================================

#define AERF_MAX 16

static int    AerFCount=0;
static double AerFDiameter[AERF_MAX], AerFMass0[AERF_MAX];
static double AerFDensity=5680.0;
static char   AerFVtkPattern[1024]="", AerFHistoryFile[1024]="aerosol_field.csv";
static double AerFProbe[2]={0.0,0.0};
static int    AerFHaveProbe=0;

static double *AerFY[AERF_MAX];          // mass fraction / initial mass fraction, per FVM cell (ghosted)
static double *AerFFlux=NULL;            // scratch
static double AerFM0[AERF_MAX];          // initial mass per unit depth (normalized units)
static int    AerFFrame=0;
static double AerFLastTime=0.0;

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
    if(sscanf(buf," AerosolFieldProbe %lf %lf",&AerFProbe[0],&AerFProbe[1])==2){ AerFHaveProbe=1; return 1; }
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
    const int nc=FvmF->ncell();
    double m0tot=0.0; for(int i=0;i<AerFCount;++i) m0tot+=AerFMass0[i];
    for(int s=0;s<AerFCount;++s){
        AerFY[s]=(double *)malloc(nc*sizeof(double));
        for(int q=0;q<nc;++q) AerFY[s][q]=0.0;
        for(int i=1;i<=FvmF->nx;++i) for(int j=1;j<=FvmF->ny;++j)
            if(!FvmF->solid[FvmF->ic(i,j)]) AerFY[s][FvmF->ic(i,j)]=AerFMass0[s]/m0tot;   // Y_i / sum_i Y0_i
    }
    AerFFlux=(double *)malloc(2*nc*sizeof(double));
    int nfluid=0;
    for(int i=1;i<=FvmF->nx;++i) for(int j=1;j<=FvmF->ny;++j) if(!FvmF->solid[FvmF->ic(i,j)]) ++nfluid;
    for(int s=0;s<AerFCount;++s) AerFM0[s]=AerFMass0[s]/m0tot*nfluid;
    AerFLastTime=Time;
    log_printf("AerosolField: %d size classes on the FVM grid, rho_P=%e d_D=%e m (Liang et al. 2020 eqs. 8-17)\n",
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

static inline double aerVanLeer(const double r){ return (r>0.0) ? 2.0*r/(1.0+r) : 0.0; }

// advect + diffuse all classes over dt with the host face velocities
static void aerFieldTransport(const double dt)
{
    if(AerFCount==0 || !(dt>0.0)) return;
    const fvm::Fields &f=*FvmF;
    const int nx=f.nx, ny=f.ny;
    const double dx=FvmG.dx, dy=FvmG.dy, nu=FvmP.mu_g/FvmP.rho_g;
    const double *u=f.u.data(), *v=f.v.data();
    const unsigned char *solid=f.solid.data();
    double cmax=0.0;
    for(int i=0;i<=nx;++i) for(int j=1;j<=ny;++j) cmax=fmax(cmax,fabs(u[f.iu(i,j)])*dt/dx);
    for(int i=1;i<=nx;++i) for(int j=0;j<=ny;++j) cmax=fmax(cmax,fabs(v[f.iv(i,j)])*dt/dy);
    cmax=fmax(cmax,4.0*nu*dt/(dx*dx));
    const int nsub=(int)ceil(cmax/0.4)>1 ? (int)ceil(cmax/0.4) : 1;
    const double h=dt/nsub;
    double *Fx=AerFFlux, *Fy=AerFFlux+f.ncell();
    #define AERF_GAS(i,j) ((i)>=1 && (i)<=nx && (j)>=1 && (j)<=ny && !solid[f.ic(i,j)])
    for(int s=0;s<AerFCount;++s){
        double *Y=AerFY[s];
        for(int sub=0;sub<nsub;++sub){
            // x faces: between (i,j) and (i+1,j)
            #pragma omp parallel for
            for(int i=0;i<=nx;++i) for(int j=1;j<=ny;++j){
                double F=0.0;
                if(AERF_GAS(i,j) && AERF_GAS(i+1,j)){
                    const double uf=u[f.iu(i,j)];
                    const int up=(uf>=0.0) ? i : i+1, dn=(uf>=0.0) ? i+1 : i, uu=(uf>=0.0) ? i-1 : i+2;
                    const double yU=Y[f.ic(up,j)], yD=Y[f.ic(dn,j)];
                    const double yUU=AERF_GAS(uu,j) ? Y[f.ic(uu,j)] : yU;
                    const double den=yD-yU;
                    const double r=(fabs(den)>1.0e-300) ? (yU-yUU)/den : 0.0;
                    F=uf*(yU+0.5*aerVanLeer(r)*den)-nu*(Y[f.ic(i+1,j)]-Y[f.ic(i,j)])/dx;
                }
                Fx[f.iu(i,j)]=F;
            }
            // y faces: between (i,j) and (i,j+1)
            #pragma omp parallel for
            for(int i=1;i<=nx;++i) for(int j=0;j<=ny;++j){
                double F=0.0;
                if(AERF_GAS(i,j) && AERF_GAS(i,j+1)){
                    const double vf=v[f.iv(i,j)];
                    const int up=(vf>=0.0) ? j : j+1, dn=(vf>=0.0) ? j+1 : j, uu=(vf>=0.0) ? j-1 : j+2;
                    const double yU=Y[f.ic(i,up)], yD=Y[f.ic(i,dn)];
                    const double yUU=AERF_GAS(i,uu) ? Y[f.ic(i,uu)] : yU;
                    const double den=yD-yU;
                    const double r=(fabs(den)>1.0e-300) ? (yU-yUU)/den : 0.0;
                    F=vf*(yU+0.5*aerVanLeer(r)*den)-nu*(Y[f.ic(i,j+1)]-Y[f.ic(i,j)])/dy;
                }
                Fy[f.iv(i,j)]=F;
            }
            #pragma omp parallel for
            for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j){
                if(solid[f.ic(i,j)]) continue;
                Y[f.ic(i,j)]-=h*((Fx[f.iu(i,j)]-Fx[f.iu(i-1,j)])/dx+(Fy[f.iv(i,j)]-Fy[f.iv(i,j-1)])/dy);
                if(Y[f.ic(i,j)]<0.0) Y[f.ic(i,j)]=0.0;
            }
        }
    }
    #undef AERF_GAS
}

// sink of one spray parcel (liquid volume Vp, droplet velocity ud) over dt
static void aerFieldParcelSink(const double x, const double y, const double udx, const double udy,
                               const double uax, const double uay, const double Vp, const double dt)
{
    if(AerFCount==0) return;
    int i,j;
    if(!fvmCellOf(x,y,&i,&j)) return;
    const double cellA=FvmG.dx*FvmG.dy, dD=AerosolDropletDiameter;
    const double UD=hypot(udx,udy), Urel=hypot(uax-udx,uay-udy);
    const double alpha=Vp/cellA;
    const int q=FvmF->ic(i,j);
    for(int s=0;s<AerFCount;++s){
        const double lam=1.5*aerFieldEfficiency(s,Urel,alpha)*UD*Vp/(dD*cellA);
        AerFY[s][q]*=exp(-lam*dt);
    }
}

static double aerFieldClassMass(const int s)
{
    double m=0.0;
    for(int i=1;i<=FvmF->nx;++i) for(int j=1;j<=FvmF->ny;++j) if(!FvmF->solid[FvmF->ic(i,j)]) m+=AerFY[s][FvmF->ic(i,j)];
    return m;
}

static void aerFieldWriteHistory()
{
    if(AerFCount==0) return;
    FILE *fp=fopen(AerFHistoryFile,"a");
    if(!fp) return;
    double mt=0.0, m0=0.0, ms[AERF_MAX];
    for(int s=0;s<AerFCount;++s){ ms[s]=aerFieldClassMass(s); mt+=ms[s]; m0+=AerFM0[s]; }
    fprintf(fp,"%.9e,%.9e",Time,mt/m0);
    for(int s=0;s<AerFCount;++s) fprintf(fp,",%.9e",ms[s]/AerFM0[s]);
    if(AerFHaveProbe){
        int i,j; double yp=0.0;
        if(fvmCellOf(AerFProbe[0],AerFProbe[1],&i,&j)) for(int s=0;s<AerFCount;++s) yp+=AerFY[s][FvmF->ic(i,j)];
        fprintf(fp,",%.9e",yp);
    }
    fprintf(fp,"\n"); fclose(fp);
}

// cell data Y_i/Y0_i of every class and of the total, on the FVM grid
static void aerFieldWriteVtk()
{
    if(AerFCount==0 || AerFVtkPattern[0]=='\0') return;
    char name[1100]; snprintf(name,sizeof(name),AerFVtkPattern,AerFFrame++);
    FILE *fp=fopen(name,"w");
    if(!fp) return;
    const int nx=FvmF->nx, ny=FvmF->ny;
    double y0[AERF_MAX], y0t=0.0;
    for(int s=0;s<AerFCount;++s){ y0[s]=AerFM0[s]; y0t+=y0[s]; }
    int nfluid=0;
    for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) if(!FvmF->solid[FvmF->ic(i,j)]) ++nfluid;
    fprintf(fp,"# vtk DataFile Version 3.0\nAerosol mass fraction / initial, t=%e\nASCII\nDATASET STRUCTURED_POINTS\n",Time);
    fprintf(fp,"DIMENSIONS %d %d 1\nORIGIN %e %e 0\nSPACING %e %e 1\n",nx+1,ny+1,FvmG.xmin,FvmG.ymin,FvmG.dx,FvmG.dy);
    fprintf(fp,"FIELD FieldData 1\nTimeValue 1 1 double\n%e\n",Time);
    fprintf(fp,"CELL_DATA %d\n",nx*ny);
    for(int s=-1;s<AerFCount;++s){
        if(s<0) fprintf(fp,"SCALARS total float 1\nLOOKUP_TABLE default\n");
        else    fprintf(fp,"SCALARS class%d_%.3fum float 1\nLOOKUP_TABLE default\n",s,AerFDiameter[s]*1e6);
        for(int j=1;j<=ny;++j) for(int i=1;i<=nx;++i){
            const int q=FvmF->ic(i,j);
            double val=0.0;
            if(!FvmF->solid[q]){
                if(s<0){ for(int c=0;c<AerFCount;++c) val+=AerFY[c][q]; val/=y0t/nfluid; }
                else val=AerFY[s][q]/(y0[s]/nfluid);
            }
            fprintf(fp,"%e\n",FvmF->solid[q] ? -1.0 : val);
        }
    }
    fclose(fp);
}

static void aerFieldLogSummary()
{
    if(AerFCount==0) return;
    double mt=0.0, m0=0.0;
    for(int s=0;s<AerFCount;++s){ mt+=aerFieldClassMass(s); m0+=AerFM0[s]; }
    log_printf("AerosolField summary: t=%e total mass / initial = %e\n",Time,mt/m0);
}
