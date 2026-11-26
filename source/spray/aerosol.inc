// =============================================================================
// aerosol.inc -- Lagrangian aerosol (Property 5) and spray scavenging
//
// Included once by main_implicit.cpp (after log.h, before main()).
//
// Rows of the grid file with Property==AEROSOL_TYPE (5) are NOT SPH particles:
// they are removed from the particle arrays while the grid is read and kept
// here as passive point particles. The SPH solver (PPE, viscosity, walls,
// neighbour lists) never sees them; one-way coupling only.
//
// Every AerosolEvery steps (time step dta = AerosolEvery*Dt) on the host:
//   carrier gas  u_g : FVM air velocity when fvm_air.inc is enabled, else
//                      Shepard average of Type 0 (air) particles within h,
//                      otherwise AerosolAirVelocity (still air by default)
//   motion           : Stokes drag with slip correction, exact exponential
//                      update,  tau = rho_p d_p^2 Cc / (18 mu_g),  v_s = tau g
//   diffusion        : random walk sqrt(2 (D_B + D_t) dta),
//                      D_B = k T Cc / (3 pi mu_g d_p)  (Stokes-Einstein)
//   scavenging       : local liquid fraction alpha = sum_water dp^2 W(r,h) and
//                      water velocity u_w (Shepard). Removal rate per particle
//                        Lambda = 3/2 * E * s * alpha * |u_w - u_a| / D_d
//                      (sub-grid droplets of diameter D_d; s = AerosolSprayScale
//                      maps the resolved SPH liquid fraction to the real spray),
//                      capture probability 1 - exp(-Lambda dta).
//                      E = Slinn (1983) Brownian + interception + impaction.
//                      alpha >= AerosolPoolAlpha (inside bulk water) -> captured.
//   wall deposition  : closer than AerosolWallGapRatio*dp to a wall particle
//                      (Type 2/3)  -> deposited (state 2, frozen).
// Captured particles (state 1) are carried with the local water velocity.
// Particles leaving the domain are counted as lost (state 3).
//
// Single-collector mode (AerosolCollector cx cy R): one resolved droplet,
// computed in the frame of the droplet (the air streams past it). A particle
// whose path comes within R + d_p/2 of the centre (interception included,
// tested on the swept segment of every substep) is collected (state 4, frozen)
// and its release x/y, time and impact angle go to AerosolCollectorFile.
// The carrier gas is then either the FVM air (solid cells = the droplet) or,
// with AerosolAirField POTENTIAL, the analytic potential flow past the circle
// with free stream AerosolAirVelocity. Several aerosol diameters can run in one
// case (AerosolClasses); the class of a particle is the optional 8th column of
// its grid row (default 0).
//
// Output: <vtk name>_aer.vtk with every VTK frame, aerosol_history.csv with
// every prof frame (time, airborne, captured, deposited, lost, C/C0).
// Not written to .prof, so a restart from .prof drops the aerosol.
//
// Data file keys (all optional):
//   AerosolDiameter            d_p   [m]       1.0e-6
//   AerosolDensity             rho_p [kg/m3]   1000
//   AerosolGasViscosity        mu_g  [Pa s]    1.8e-5
//   AerosolGasDensity          rho_g [kg/m3]   1.2
//   AerosolGasTemperature      T     [K]       293.15
//   AerosolMeanFreePath        lambda[m]       6.8e-8
//   AerosolDropletDiameter     D_d   [m]       2.0e-4
//   AerosolSprayScale          s     [-]       1.0
//   AerosolTurbulentDiffusivity D_t  [m2/s]    0.0
//   AerosolBrownian            0|1             1
//   AerosolAirVelocity         ux uy [m/s]     0 0
//   AerosolKernelRatio         h/dp            2.0
//   AerosolPoolAlpha           alpha           0.5
//   AerosolWallGapRatio        gap/dp          0.5
//   AerosolEvery               n steps         1
//   AerosolSeed                integer         12345
//   AerosolHistoryFile         path            aerosol_history.csv
//   AerosolClasses             d1 d2 ...  [m]  (AerosolDiameter)  up to 64 classes
//   AerosolCollector           cx cy R    [m]  off
//   AerosolCollectorFile       path            aerosol_collected.csv
//   AerosolAirField            FIELD|POTENTIAL FIELD (FVM / Type 0 / AerosolAirVelocity)
//   AerosolSubsteps            n               1  (aerosol substeps per aerosol step)
//   AerosolReleaseTime         t     [s]       0  (airborne particles held until t)
// =============================================================================

#define AEROSOL_TYPE 5

enum { AER_AIRBORNE=0, AER_CAPTURED=1, AER_DEPOSITED=2, AER_LOST=3, AER_COLLECTED=4 };
#define AER_NSTATE 5
#define AER_MAXCLASS 64

static int     AerosolCount=0;
static double (*AerPos)[2]=NULL;
static double (*AerVel)[2]=NULL;
static int     *AerState=NULL;
static double  *AerAlpha=NULL;      // last local liquid fraction (output)
static double  *AerLambda=NULL;     // last removal rate [1/s] (output)
static unsigned long long *AerRng=NULL;
static int    *AerClass=NULL;       // diameter class
static double (*AerPos0)[2]=NULL;   // release position
static double  *AerHitTime=NULL;    // collection time / angle (collector mode)
static double  *AerHitAngle=NULL;

static double AerosolDiameter=1.0e-6;
static double AerosolDensity=1000.0;
static double AerosolGasViscosity=1.8e-5;
static double AerosolGasDensity=1.2;
static double AerosolGasTemperature=293.15;
static double AerosolMeanFreePath=6.8e-8;
static double AerosolDropletDiameter=2.0e-4;
static double AerosolSprayScale=1.0;
static double AerosolTurbulentDiffusivity=0.0;
static int    AerosolBrownian=1;
static double AerosolAirVelocity[2]={0.0,0.0};
static double AerosolKernelRatio=2.0;
static double AerosolPoolAlpha=0.5;
static double AerosolWallGapRatio=0.5;
static int    AerosolEvery=1;
static unsigned long long AerosolSeed=12345ULL;
static char   AerosolHistoryFile[1024]="aerosol_history.csv";
static int    AerosolClassCount=0;
static double AerosolClassDiameter[AER_MAXCLASS];
static int    AerosolCollectorOn=0;
static double AerosolCollector[3]={0.0,0.0,0.0};   // cx cy R
static char   AerosolCollectorFile[1024]="aerosol_collected.csv";
static int    AerosolPotentialFlow=0;
static int    AerosolSubsteps=1;
static double AerosolReleaseTime=0.0;

// derived (per class; class 0 = AerosolDiameter when AerosolClasses is not given)
static double AerTau=0.0, AerCc=1.0, AerDiffB=0.0, AerEff=0.0;
static double AerClsD[AER_MAXCLASS], AerClsTau[AER_MAXCLASS], AerClsCc[AER_MAXCLASS], AerClsDB[AER_MAXCLASS];
static double AerTimeAccum=0.0;
static int    AerStepCount=0;

// host-side bins of the SPH particles (rebuilt every aerosol step)
static int AerBinN[2]={0,0};
static double AerBinW=0.0;
static int *AerBinStart=NULL, *AerBinList=NULL, AerBinCap=0;

static int aerosolParseDataLine(const char *buf)
{
    char key[256]; long long seed;
    if(sscanf(buf," %255s",key)!=1 || strncmp(key,"Aerosol",7)) return 0;
    if(sscanf(buf," AerosolDiameter %lf",&AerosolDiameter)==1) return 1;
    if(sscanf(buf," AerosolDensity %lf",&AerosolDensity)==1) return 1;
    if(sscanf(buf," AerosolGasViscosity %lf",&AerosolGasViscosity)==1) return 1;
    if(sscanf(buf," AerosolGasDensity %lf",&AerosolGasDensity)==1) return 1;
    if(sscanf(buf," AerosolGasTemperature %lf",&AerosolGasTemperature)==1) return 1;
    if(sscanf(buf," AerosolMeanFreePath %lf",&AerosolMeanFreePath)==1) return 1;
    if(sscanf(buf," AerosolDropletDiameter %lf",&AerosolDropletDiameter)==1) return 1;
    if(sscanf(buf," AerosolSprayScale %lf",&AerosolSprayScale)==1) return 1;
    if(sscanf(buf," AerosolTurbulentDiffusivity %lf",&AerosolTurbulentDiffusivity)==1) return 1;
    if(sscanf(buf," AerosolBrownian %d",&AerosolBrownian)==1) return 1;
    if(sscanf(buf," AerosolAirVelocity %lf %lf",&AerosolAirVelocity[0],&AerosolAirVelocity[1])==2) return 1;
    if(sscanf(buf," AerosolKernelRatio %lf",&AerosolKernelRatio)==1) return 1;
    if(sscanf(buf," AerosolPoolAlpha %lf",&AerosolPoolAlpha)==1) return 1;
    if(sscanf(buf," AerosolWallGapRatio %lf",&AerosolWallGapRatio)==1) return 1;
    if(sscanf(buf," AerosolEvery %d",&AerosolEvery)==1) return 1;
    if(sscanf(buf," AerosolSeed %lld",&seed)==1){ AerosolSeed=(unsigned long long)seed; return 1; }
    if(sscanf(buf," AerosolHistoryFile %1023s",AerosolHistoryFile)==1) return 1;
    if(!strcmp(key,"AerosolClasses")){
        const char *q=buf; while(*q==' '||*q=='\t') ++q; q+=strlen("AerosolClasses");
        char *end; AerosolClassCount=0;
        for(double d=strtod(q,&end); end!=q && AerosolClassCount<AER_MAXCLASS; d=strtod(q,&end)){
            AerosolClassDiameter[AerosolClassCount++]=d; q=end;
        }
        return 1;
    }
    if(sscanf(buf," AerosolCollector %lf %lf %lf",&AerosolCollector[0],&AerosolCollector[1],&AerosolCollector[2])==3){
        AerosolCollectorOn=AerosolCollector[2]>0.0; return 1;
    }
    if(sscanf(buf," AerosolCollectorFile %1023s",AerosolCollectorFile)==1) return 1;
    if(!strcmp(key,"AerosolAirField")){
        char v[64]="";
        if(sscanf(buf," AerosolAirField %63s",v)==1){
            if(!strcmp(v,"POTENTIAL")) AerosolPotentialFlow=1;
            else if(!strcmp(v,"FIELD")) AerosolPotentialFlow=0;
            else { log_printf("AerosolAirField: unknown '%s' (FIELD|POTENTIAL)\n",v); exit(EXIT_FAILURE); }
        }
        return 1;
    }
    if(sscanf(buf," AerosolSubsteps %d",&AerosolSubsteps)==1) return 1;
    if(sscanf(buf," AerosolReleaseTime %lf",&AerosolReleaseTime)==1) return 1;
    return 0;
}

// Called by readGridFile with fp just after the header. Reads all n rows,
// keeps the aerosol rows, rewinds, and returns the number of aerosol rows.
static int aerosolPrescanGrid(FILE *fp, int n)
{
    char buf[1024];
    const long start=ftell(fp);
    int count=0;
    for(int i=0;i<n;++i){
        if(fgets(buf,sizeof(buf),fp)==NULL) break;
        int type; if(sscanf(buf,"%d",&type)==1 && type==AEROSOL_TYPE) ++count;
    }
    AerosolCount=count;
    if(count>0){
        AerPos=(double (*)[2])malloc(count*sizeof(double[2]));
        AerVel=(double (*)[2])malloc(count*sizeof(double[2]));
        AerState=(int *)malloc(count*sizeof(int));
        AerAlpha=(double *)malloc(count*sizeof(double));
        AerLambda=(double *)malloc(count*sizeof(double));
        AerRng=(unsigned long long *)malloc(count*sizeof(unsigned long long));
        AerClass=(int *)malloc(count*sizeof(int));
        AerPos0=(double (*)[2])malloc(count*sizeof(double[2]));
        AerHitTime=(double *)malloc(count*sizeof(double));
        AerHitAngle=(double *)malloc(count*sizeof(double));
        fseek(fp,start,SEEK_SET);
        int k=0;
        for(int i=0;i<n && k<count;++i){
            if(fgets(buf,sizeof(buf),fp)==NULL) break;
            int type, cls=0; double x,y,z,u,v,w;
            if(sscanf(buf,"%d %lf %lf %lf %lf %lf %lf %d",&type,&x,&y,&z,&u,&v,&w,&cls)>=7 && type==AEROSOL_TYPE){
                AerPos[k][0]=x; AerPos[k][1]=y; AerVel[k][0]=u; AerVel[k][1]=v;
                AerPos0[k][0]=x; AerPos0[k][1]=y; AerClass[k]=cls; AerHitTime[k]=-1.0; AerHitAngle[k]=0.0;
                AerState[k]=AER_AIRBORNE; AerAlpha[k]=0.0; AerLambda[k]=0.0;
                ++k;
            }
        }
    }
    fseek(fp,start,SEEK_SET);
    return count;
}

// splitmix64 seeding + xorshift64* stream, one per aerosol particle
static unsigned long long aerSplitMix(unsigned long long x)
{
    x+=0x9E3779B97F4A7C15ULL;
    x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL;
    x=(x^(x>>27))*0x94D049BB133111EBULL;
    return x^(x>>31);
}
static double aerUniform(unsigned long long *s)
{
    *s^=*s>>12; *s^=*s<<25; *s^=*s>>27;
    return ((*s*0x2545F4914F6CDD1DULL)>>11)*(1.0/9007199254740992.0);
}
static double aerGauss(unsigned long long *s)
{
    double u1=aerUniform(s); if(u1<1.0e-300) u1=1.0e-300;
    const double u2=aerUniform(s);
    return sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2);
}

// Wendland C2, 2D, normalised: int W dA = 1
static double aerKernel(const double r, const double h)
{
    const double q=r/h;
    if(q>=1.0) return 0.0;
    const double a=1.0-q;
    return 7.0/(M_PI*h*h)*a*a*a*a*(1.0+4.0*q);
}

// Slinn (1983) collection efficiency of one droplet (diameter D) falling at
// relative speed U through aerosol of diameter d (Seinfeld & Pandis eq. 20.53).
static double aerSlinnEfficiency(const double U, const int c=0)
{
    if(!(U>0.0)) return 0.0;
    const double D=AerosolDropletDiameter, d=AerClsD[c];
    const double muG=AerosolGasViscosity, rhoG=AerosolGasDensity;
    const double muW=ShearViscosity[1]>0.0 ? ShearViscosity[1] : 1.0e-3;
    const double rhoW=Density[1]>0.0 ? Density[1] : 1000.0;
    const double Re=D*U*rhoG/(2.0*muG);                       // radius-based
    const double Sc=muG/(rhoG*AerClsDB[c]);
    const double St=2.0*AerClsTau[c]*U/D;
    const double phi=d/D, omega=muW/muG;
    const double sre=sqrt(Re);
    double E=4.0/(Re*Sc)*(1.0+0.4*sre*cbrt(Sc)+0.16*sre*sqrt(Sc))
            +4.0*phi*(1.0/omega+(1.0+2.0*sre)*phi);
    const double Sstar=(1.2+log(1.0+Re)/12.0)/(1.0+log(1.0+Re));
    if(St>Sstar){
        const double r=(St-Sstar)/(St-Sstar+2.0/3.0);
        E+=pow(r,1.5)*sqrt(rhoW/AerosolDensity);
    }
    return fmin(E,1.0);
}

static void aerosolInitialize()
{
#ifndef TWO_DIMENSIONAL
    if(AerosolCount>0){ log_printf("3D build: Lagrangian aerosol (Property 5) is 2D only\n"); exit(EXIT_FAILURE); }
#endif
    if(AerosolEvery<1) AerosolEvery=1;
    if(AerosolSubsteps<1) AerosolSubsteps=1;
    const double kB=1.380649e-23;
    if(AerosolClassCount<=0){ AerosolClassCount=1; AerosolClassDiameter[0]=AerosolDiameter; }
    for(int c=0;c<AerosolClassCount;++c){
        const double d=AerosolClassDiameter[c];
        const double Kn=2.0*AerosolMeanFreePath/d;
        AerClsD[c]=d;
        AerClsCc[c]=1.0+Kn*(1.257+0.4*exp(-1.1/Kn));
        AerClsTau[c]=AerosolDensity*d*d*AerClsCc[c]/(18.0*AerosolGasViscosity);
        AerClsDB[c]=kB*AerosolGasTemperature*AerClsCc[c]/(3.0*M_PI*AerosolGasViscosity*d);
    }
    for(int k=0;k<AerosolCount;++k){
        if(AerClass[k]<0 || AerClass[k]>=AerosolClassCount){
            log_printf("Aerosol: particle %d has class %d, AerosolClasses defines %d\n",k,AerClass[k],AerosolClassCount);
            exit(EXIT_FAILURE);
        }
    }
    AerCc=AerClsCc[0]; AerTau=AerClsTau[0]; AerDiffB=AerClsDB[0];
    if(AerosolPotentialFlow && !AerosolCollectorOn){ log_printf("AerosolAirField POTENTIAL needs AerosolCollector\n"); exit(EXIT_FAILURE); }
    for(int k=0;k<AerosolCount;++k) AerRng[k]=aerSplitMix(AerosolSeed+(unsigned long long)k);
    AerBinW=AerosolKernelRatio*ParticleSpacing;
    AerBinN[0]=(int)ceil((DomainMax[0]-DomainMin[0])/AerBinW)+1;
    AerBinN[1]=(int)ceil((DomainMax[1]-DomainMin[1])/AerBinW)+1;
    AerBinStart=(int *)malloc((AerBinN[0]*AerBinN[1]+1)*sizeof(int));
    const double g=sqrt(Gravity[0]*Gravity[0]+Gravity[1]*Gravity[1]);
    log_printf("Aerosol: %d particles (Property %d), d_p=%e m rho_p=%e Cc=%f tau=%e s v_s=%e m/s D_B=%e m2/s\n",
               AerosolCount,AEROSOL_TYPE,AerosolDiameter,AerosolDensity,AerCc,AerTau,AerTau*g,AerDiffB);
    log_printf("Aerosol: D_drop=%e m spray scale=%e D_t=%e m2/s h/dp=%f every=%d step(s); E(U=5 m/s)=%e\n",
               AerosolDropletDiameter,AerosolSprayScale,AerosolTurbulentDiffusivity,AerosolKernelRatio,
               AerosolEvery,aerSlinnEfficiency(5.0));
    for(int c=1;c<AerosolClassCount;++c)
        log_printf("Aerosol: class %d d_p=%e m Cc=%f tau=%e s D_B=%e m2/s\n",c,AerClsD[c],AerClsCc[c],AerClsTau[c],AerClsDB[c]);
    if(AerosolCollectorOn){
        const double U=hypot(AerosolAirVelocity[0],AerosolAirVelocity[1]);
        log_printf("Aerosol: collector centre (%e,%e) R=%e m, air field %s, free stream (%e,%e) m/s, %d substep(s), release at t=%e s\n",
                   AerosolCollector[0],AerosolCollector[1],AerosolCollector[2],AerosolPotentialFlow?"POTENTIAL":"FIELD",
                   AerosolAirVelocity[0],AerosolAirVelocity[1],AerosolSubsteps,AerosolReleaseTime);
        for(int c=0;c<AerosolClassCount;++c)
            log_printf("Aerosol: class %d St=tau U/R=%e  R_int=d_p/(2R)=%e\n",c,AerClsTau[c]*U/AerosolCollector[2],AerClsD[c]/(2.0*AerosolCollector[2]));
    }
    if(AerosolCount>0){
        FILE *fp=fopen(AerosolHistoryFile,"w");
        if(fp){ fprintf(fp,"time,airborne,captured,deposited,lost,c_over_c0,collected\n"); fclose(fp); }
    }
}

static inline int aerBinIndex(const double x, const double y)
{
    int ix=(int)floor((x-DomainMin[0])/AerBinW), iy=(int)floor((y-DomainMin[1])/AerBinW);
    if(ix<0) ix=0; if(ix>=AerBinN[0]) ix=AerBinN[0]-1;
    if(iy<0) iy=0; if(iy>=AerBinN[1]) iy=AerBinN[1]-1;
    return ix+AerBinN[0]*iy;
}

#ifdef SPH_FVM_AIR
static int fvmAirVelocity(const double x, const double y, double *ux, double *uy);   // fvm_air.inc
#endif

static void aerosolBuildBins()
{
    const int nb=AerBinN[0]*AerBinN[1];
    if(AerBinCap<ParticleCount){ free(AerBinList); AerBinCap=ParticleCount; AerBinList=(int *)malloc(AerBinCap*sizeof(int)); }
    for(int b=0;b<=nb;++b) AerBinStart[b]=0;
    for(int i=0;i<ParticleCount;++i) AerBinStart[aerBinIndex(Position[i][0],Position[i][1])+1]++;
    for(int b=0;b<nb;++b) AerBinStart[b+1]+=AerBinStart[b];
    int *cursor=(int *)malloc(nb*sizeof(int));
    for(int b=0;b<nb;++b) cursor[b]=AerBinStart[b];
    for(int i=0;i<ParticleCount;++i) AerBinList[cursor[aerBinIndex(Position[i][0],Position[i][1])]++]=i;
    free(cursor);
}

static void aerosolCounts(int c[AER_NSTATE])
{
    for(int s=0;s<AER_NSTATE;++s) c[s]=0;
    for(int k=0;k<AerosolCount;++k) c[AerState[k]]++;
}

static void aerosolWriteHistory()
{
    if(AerosolCount<=0) return;
    int c[AER_NSTATE]; aerosolCounts(c);
    FILE *fp=fopen(AerosolHistoryFile,"a");
    if(!fp) return;
    fprintf(fp,"%.9e,%d,%d,%d,%d,%.9e,%d\n",Time,c[0],c[1],c[2],c[3],(double)c[0]/AerosolCount,c[4]);
    fclose(fp);
}

// analytic potential flow past the collector circle, free stream AerosolAirVelocity:
// dW/dz = U_inf e^{-i a} - U_inf e^{i a} R^2 / z^2 = u - i v
static void aerPotentialFlow(const double x, const double y, double *ux, double *uy)
{
    const double Ux=AerosolAirVelocity[0], Uy=AerosolAirVelocity[1], R2=AerosolCollector[2]*AerosolCollector[2];
    const double zx=x-AerosolCollector[0], zy=y-AerosolCollector[1];
    const double r2=zx*zx+zy*zy; if(r2<=0.0){ *ux=*uy=0.0; return; }
    const double ix=(zx*zx-zy*zy)/(r2*r2), iy=-2.0*zx*zy/(r2*r2);      // 1/z^2
    const double wx=Ux-R2*(Ux*ix-Uy*iy), wy=-Uy-R2*(Ux*iy+Uy*ix);      // u - i v
    *ux=wx; *uy=-wy;
}

// collector contact on the segment (x0,y0)->(x1,y1): distance R + d_p/2 (interception)
static int aerCollectorHit(const double x0, const double y0, const double x1, const double y1,
                           const double rp, double *hx, double *hy)
{
    const double cx=AerosolCollector[0], cy=AerosolCollector[1], Rc=AerosolCollector[2]+rp;
    const double dx=x1-x0, dy=y1-y0, fx=x0-cx, fy=y0-cy;
    const double a=dx*dx+dy*dy, b=2.0*(fx*dx+fy*dy), c=fx*fx+fy*fy-Rc*Rc;
    if(c<=0.0){ *hx=x0; *hy=y0; return 1; }
    if(a<=0.0) return 0;
    const double disc=b*b-4.0*a*c; if(disc<0.0) return 0;
    const double t=(-b-sqrt(disc))/(2.0*a);
    if(t<0.0 || t>1.0) return 0;
    *hx=x0+t*dx; *hy=y0+t*dy; return 1;
}

static void aerosolAdvance(const double dta_step)
{
    #pragma acc update self(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
    aerosolBuildBins();
    const int nsub=(dta_step>0.0)?AerosolSubsteps:1;
    const double dta=dta_step/nsub;
    const double h=AerBinW, dA=ParticleVolume;
    const double gx=Gravity[0], gy=Gravity[1];
    const double sigmaT=2.0*AerosolTurbulentDiffusivity*dta;
    const double gap2=(AerosolWallGapRatio*ParticleSpacing)*(AerosolWallGapRatio*ParticleSpacing);
    const int held=(Time<AerosolReleaseTime);

    #pragma omp parallel for schedule(dynamic,256)
    for(int k=0;k<AerosolCount;++k){
      const int c=AerClass[k];
      const double tau=AerClsTau[c];
      const double vsx=tau*gx, vsy=tau*gy;
      const double decay=exp(-dta/tau);
      const double relax=tau*(1.0-decay);
      const double sigma=sqrt(sigmaT+2.0*(AerosolBrownian?AerClsDB[c]:0.0)*dta);
      for(int sub=0;sub<nsub;++sub){
        const int st=AerState[k];
        if(st==AER_DEPOSITED || st==AER_LOST || st==AER_COLLECTED) break;
        const double x=AerPos[k][0], y=AerPos[k][1];
        double wa=0.0, uax=0.0, uay=0.0;      // air (Type 0)
        double alpha=0.0, ww=0.0, uwx=0.0, uwy=0.0; // water (Type 1)
        double wall2=1.0e300;
        int ix=(int)floor((x-DomainMin[0])/h), iy=(int)floor((y-DomainMin[1])/h);
        for(int jy=iy-1;jy<=iy+1;++jy){
            if(jy<0||jy>=AerBinN[1]) continue;
            for(int jx=ix-1;jx<=ix+1;++jx){
                if(jx<0||jx>=AerBinN[0]) continue;
                const int b=jx+AerBinN[0]*jy;
                for(int m=AerBinStart[b];m<AerBinStart[b+1];++m){
                    const int j=AerBinList[m];
                    const double rx=Position[j][0]-x, ry=Position[j][1]-y;
                    const double r2=rx*rx+ry*ry;
                    const int p=Property[j];
                    if(p>=WALL_BEGIN){ if(r2<wall2) wall2=r2; continue; }
                    const double r=sqrt(r2);
                    const double w=aerKernel(r,h);
                    if(w<=0.0) continue;
                    if(p==1){ alpha+=dA*w; ww+=w; uwx+=w*Velocity[j][0]; uwy+=w*Velocity[j][1]; }
                    else    { wa+=w; uax+=w*Velocity[j][0]; uay+=w*Velocity[j][1]; }
                }
            }
        }
        if(ww>0.0){ uwx/=ww; uwy/=ww; }
        double ugx=AerosolAirVelocity[0], ugy=AerosolAirVelocity[1];
        if(AerosolPotentialFlow) aerPotentialFlow(x,y,&ugx,&ugy);
        else {
#ifdef SPH_FVM_AIR
        if(!fvmAirVelocity(x,y,&ugx,&ugy) && wa>0.0){ ugx=uax/wa; ugy=uay/wa; }
#else
        if(wa>0.0){ ugx=uax/wa; ugy=uay/wa; }
#endif
        }

        if(st==AER_CAPTURED){
            // carried by the water; ballistic if it momentarily has no water around,
            // at rest once it reaches a wall (e.g. its water was drained by the spray)
#ifdef SPH_FVM_AIR
            if(wall2<gap2){ AerVel[k][0]=AerVel[k][1]=0.0; AerAlpha[k]=alpha; AerLambda[k]=0.0; break; }
#endif
            if(ww>0.0){ AerVel[k][0]=uwx; AerVel[k][1]=uwy; }
            else      { AerVel[k][0]+=gx*dta; AerVel[k][1]+=gy*dta; }
            AerPos[k][0]+=AerVel[k][0]*dta; AerPos[k][1]+=AerVel[k][1]*dta;
            AerAlpha[k]=alpha; AerLambda[k]=0.0;
        }else{
            if(held){ AerAlpha[k]=alpha; break; }
            // scavenging by the spray
            double lambda=0.0;
            if(alpha>0.0 && AerosolSprayScale>0.0){
                const double U=hypot(uwx-AerVel[k][0],uwy-AerVel[k][1]);
                lambda=1.5*aerSlinnEfficiency(U,c)*AerosolSprayScale*alpha*U/AerosolDropletDiameter;
            }
            AerAlpha[k]=alpha; AerLambda[k]=lambda;
            if(alpha>=AerosolPoolAlpha || (lambda>0.0 && aerUniform(&AerRng[k])<1.0-exp(-lambda*dta))){
                AerState[k]=AER_CAPTURED;
                if(AerHitTime[k]<0.0) AerHitTime[k]=Time-dta_step+(sub+1)*dta;
                if(ww>0.0){ AerVel[k][0]=uwx; AerVel[k][1]=uwy; }
                continue;
            }
            // Stokes drag towards u_g + v_s, exact for constant u_g over dta
            const double tx=ugx+vsx, ty=ugy+vsy;
            AerPos[k][0]+=tx*dta+(AerVel[k][0]-tx)*relax;
            AerPos[k][1]+=ty*dta+(AerVel[k][1]-ty)*relax;
            AerVel[k][0]=tx+(AerVel[k][0]-tx)*decay;
            AerVel[k][1]=ty+(AerVel[k][1]-ty)*decay;
            if(sigma>0.0){
                AerPos[k][0]+=sigma*aerGauss(&AerRng[k]);
                AerPos[k][1]+=sigma*aerGauss(&AerRng[k]);
            }
            double hx,hy;
            if(AerosolCollectorOn && aerCollectorHit(x,y,AerPos[k][0],AerPos[k][1],0.5*AerClsD[c],&hx,&hy)){
                AerState[k]=AER_COLLECTED; AerPos[k][0]=hx; AerPos[k][1]=hy;
                AerVel[k][0]=AerVel[k][1]=0.0;
                AerHitTime[k]=Time-dta_step+(sub+1)*dta;
                AerHitAngle[k]=atan2(hx-AerosolCollector[0],-(hy-AerosolCollector[1]))*180.0/M_PI;
                break;
            }
        }
        if(wall2<gap2 && AerState[k]!=AER_CAPTURED){ AerState[k]=AER_DEPOSITED; AerVel[k][0]=AerVel[k][1]=0.0; break; }
#ifdef SPH_FVM_AIR
        if(AerState[k]==AER_AIRBORNE && (AerPos[k][0]<DomainMin[0]||AerPos[k][0]>DomainMax[0]||AerPos[k][1]<DomainMin[1]||AerPos[k][1]>DomainMax[1]))
#else
        if(AerPos[k][0]<DomainMin[0]||AerPos[k][0]>DomainMax[0]||AerPos[k][1]<DomainMin[1]||AerPos[k][1]>DomainMax[1])
#endif
            { AerState[k]=AER_LOST; break; }
      }
    }
}

// called once per accepted SPH step (after Time has been advanced)
static void aerosolStep()
{
    if(AerosolCount<=0) return;
    AerTimeAccum+=Dt;
    if(++AerStepCount<AerosolEvery) return;
    aerosolAdvance(AerTimeAccum);
    AerTimeAccum=0.0; AerStepCount=0;
}

// wall check of the initial seeding (particles placed inside walls/water)
static void aerosolInitialState()
{
    if(AerosolCount<=0) return;
    aerosolAdvance(0.0);
    aerosolWriteHistory();
}

static void aerosolWriteVtk(const char *vtkname)
{
    if(AerosolCount<=0) return;
    char name[1100]; strcpy(name,vtkname);
    char *dot=strrchr(name,'.');
    if(dot && !strcmp(dot,".vtk")) strcpy(dot,"_aer.vtk"); else strcat(name,"_aer.vtk");
    FILE *fp=fopen(name,"w");
    if(!fp) return;
    fprintf(fp,"# vtk DataFile Version 2.0\nAerosol t=%e\nASCII\nDATASET UNSTRUCTURED_GRID\n",Time);
    fprintf(fp,"POINTS %d float\n",AerosolCount);
    for(int k=0;k<AerosolCount;++k) fprintf(fp,"%e %e 0.0\n",AerPos[k][0],AerPos[k][1]);
    fprintf(fp,"CELLS %d %d\n",AerosolCount,2*AerosolCount);
    for(int k=0;k<AerosolCount;++k) fprintf(fp,"1 %d\n",k);
    fprintf(fp,"CELL_TYPES %d\n",AerosolCount);
    for(int k=0;k<AerosolCount;++k) fprintf(fp,"1\n");
    fprintf(fp,"POINT_DATA %d\nSCALARS state int 1\nLOOKUP_TABLE default\n",AerosolCount);
    for(int k=0;k<AerosolCount;++k) fprintf(fp,"%d\n",AerState[k]);
    fprintf(fp,"SCALARS alpha float 1\nLOOKUP_TABLE default\n");
    for(int k=0;k<AerosolCount;++k) fprintf(fp,"%e\n",AerAlpha[k]);
    fprintf(fp,"SCALARS lambda float 1\nLOOKUP_TABLE default\n");
    for(int k=0;k<AerosolCount;++k) fprintf(fp,"%e\n",AerLambda[k]);
    fprintf(fp,"VECTORS velocity float\n");
    for(int k=0;k<AerosolCount;++k) fprintf(fp,"%e %e 0.0\n",AerVel[k][0],AerVel[k][1]);
    fclose(fp);
}

static void aerosolLogSummary()
{
    if(AerosolCount<=0) return;
    int c[AER_NSTATE]; aerosolCounts(c);
    log_printf("Aerosol summary: airborne=%d captured=%d deposited=%d lost=%d collected=%d C/C0=%e\n",
               c[0],c[1],c[2],c[3],c[4],(double)c[0]/AerosolCount);
    if(AerosolCollectorOn || AerosolClassCount>1){
        FILE *fp=fopen(AerosolCollectorFile,"w");
        if(!fp) return;
        fprintf(fp,"k,class,d_p,x0,y0,state,t_capture,angle_deg,x,y\n");
        for(int k=0;k<AerosolCount;++k)
            fprintf(fp,"%d,%d,%.6e,%.9e,%.9e,%d,%.9e,%.6f,%.9e,%.9e\n",k,AerClass[k],AerClsD[AerClass[k]],
                    AerPos0[k][0],AerPos0[k][1],AerState[k],AerHitTime[k],AerHitAngle[k],AerPos[k][0],AerPos[k][1]);
        fclose(fp);
        log_printf("Aerosol: per-particle result written to %s\n",AerosolCollectorFile);
    }
}
