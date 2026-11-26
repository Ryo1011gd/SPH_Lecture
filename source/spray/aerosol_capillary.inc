// =============================================================================
// aerosol_capillary.inc -- point aerosol particles captured (or not) by the
// capillary force of a resolved liquid surface, 2D and 3D, single phase
//
// Included once by main_implicit.cpp (after surface_tension.inc).
// Active with  CapAerosolFile <path>  (one particle per line, '#' comments):
//     x y z  vx vy vz  d_p [m]  rho_p [kg/m3]  theta [deg]
//
// One-way coupling (aerosol << drop): the SPH liquid is not affected.
// Interface (every SPH step): outer-layer liquid particles (PositionDivR <
// CapAerosolSurfaceDivR, the first shell) shifted out by dp/2 along their unit
// normal (IfNormal) are the surface points y_j. Around the point nearest to the
// particle a local plane is fitted (Gaussian weights, width 1.5 dp):
//     normal n, point y0, surface velocity u_s  ->  s = (x - y0 - u_s t) . n
// s > 0: outside the liquid.
//
// Particle of radius a, mass m, contact angle theta (sphere at an undeformed
// interface; Bond number ~ 0). Interface energy relative to the free state
//     E(s) = -pi sigma a^2 (1 - s^2/a^2) - 2 pi sigma a^2 cos(theta) (1 - s/a)
//     F    = -dE/ds n = -2 pi sigma a (s/a + cos theta) n,       |s| < a
// first contact s = a: Dupre pull 2 pi sigma a (1 + cos theta); equilibrium
// s = -a cos theta; well depth pi sigma a^2 (1 - |cos theta|)^2. Inertia
// m ~ a^3 against capillarity ~ a^2 gives the size dependence.
// Drag (implicit): Stokes in the liquid on the wetted fraction of the sphere
//     F_d = -6 pi mu_l a f (v - u),  f = (1 - s/a)/2 in contact, 1 inside
// plus optional Stokes drag in a still gas (CapAerosolGasViscosity, 0 = none).
// States: FREE (never touched), CONTACT (|s| < a, at the end: attached),
// ENGULFED (s <= -a: inside the liquid, moves with it), BOUNCED (left the
// surface s >= a after a contact, flies away; may touch again).
// Sub-steps resolve the capillary spring: dt <= CapAerosolSubstepFraction * 2 pi / omega,
// omega^2 = 2 pi sigma / m, and the drag time.
//
// Output (every prof frame): cap_aerosol.csv  time,id,state,x,y,z,vx,vy,vz,s,contacts
// Data file keys:
//   CapAerosolFile             path   (off)
//   CapAerosolSurfaceTension   sigma  SurfaceTension of Type 1
//   CapAerosolLiquidViscosity  mu_l   ShearViscosity of Type 1
//   CapAerosolGasViscosity     mu_g   0
//   CapAerosolSurfaceDivR      -      1.35 (2D) / 2.1 (3D)
//   CapAerosolSubstepFraction  -      0.02
// =============================================================================

enum { CAP_FREE=0, CAP_CONTACT=1, CAP_ENGULFED=2, CAP_BOUNCED=3 };

static char   CapAerosolFile[256]="";
static double CapAerosolSurfaceTension=-1.0, CapAerosolLiquidViscosity=-1.0;
static double CapAerosolGasViscosity=0.0;
#ifdef TWO_DIMENSIONAL
static double CapAerosolSurfaceDivR=1.35;
#else
static double CapAerosolSurfaceDivR=2.1;
#endif
static double CapAerosolSubstepFraction=0.02;

static int    CapN=0;
static double (*CapX)[3]=NULL, (*CapV)[3]=NULL;
static double *CapD=NULL, *CapRho=NULL, *CapTheta=NULL, *CapS=NULL;
static int    *CapState=NULL, *CapContacts=NULL;
static double *CapHostDivR=NULL, (*CapHostX)[3]=NULL, (*CapHostV)[3]=NULL, (*CapHostN)[3]=NULL;
static int    *CapHostSolid=NULL;


static int capAerosolParseDataLine(const char *buf)
{
    if(sscanf(buf," CapAerosolFile %255s",CapAerosolFile)==1) return 1;
    if(sscanf(buf," CapAerosolSurfaceTension %lf",&CapAerosolSurfaceTension)==1) return 1;
    if(sscanf(buf," CapAerosolLiquidViscosity %lf",&CapAerosolLiquidViscosity)==1) return 1;
    if(sscanf(buf," CapAerosolGasViscosity %lf",&CapAerosolGasViscosity)==1) return 1;
    if(sscanf(buf," CapAerosolSurfaceDivR %lf",&CapAerosolSurfaceDivR)==1) return 1;
    if(sscanf(buf," CapAerosolSubstepFraction %lf",&CapAerosolSubstepFraction)==1) return 1;
    return 0;
}


static void capAerosolInitialize()
{
    if(!CapAerosolFile[0]) return;
    if(TwoPhaseActive){ log_printf("CapAerosolFile: single-phase only\n"); exit(EXIT_FAILURE); }
    if(CapAerosolSurfaceTension<0.0) CapAerosolSurfaceTension=SurfaceTension[1];
    if(CapAerosolLiquidViscosity<0.0) CapAerosolLiquidViscosity=ShearViscosity[1];
    FILE *fp=fopen(CapAerosolFile,"r");
    if(!fp){ log_printf("CapAerosolFile %s not found\n",CapAerosolFile); exit(EXIT_FAILURE); }
    char line[512];
    int cap=0;
    while(fgets(line,sizeof(line),fp)){
        double x[3],v[3],d,rho,th;
        if(line[0]=='#' || sscanf(line,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",x,x+1,x+2,v,v+1,v+2,&d,&rho,&th)!=9) continue;
        if(!(d>0.0 && rho>0.0)){ log_printf("CapAerosolFile: invalid line: %s",line); exit(EXIT_FAILURE); }
        if(CapN==cap){
            cap=(cap ? 2*cap : 64);
            CapX=(double (*)[3])realloc(CapX,cap*sizeof(double[3]));
            CapV=(double (*)[3])realloc(CapV,cap*sizeof(double[3]));
            CapD=(double *)realloc(CapD,cap*sizeof(double));
            CapRho=(double *)realloc(CapRho,cap*sizeof(double));
            CapTheta=(double *)realloc(CapTheta,cap*sizeof(double));
            CapS=(double *)realloc(CapS,cap*sizeof(double));
            CapState=(int *)realloc(CapState,cap*sizeof(int));
            CapContacts=(int *)realloc(CapContacts,cap*sizeof(int));
        }
        for(int k=0;k<3;++k){ CapX[CapN][k]=x[k]; CapV[CapN][k]=v[k]; }
#ifdef TWO_DIMENSIONAL
        CapX[CapN][2]=0.0; CapV[CapN][2]=0.0;
#endif
        CapD[CapN]=d; CapRho[CapN]=rho; CapTheta[CapN]=th*3.14159265358979323846/180.0;
        CapS[CapN]=1.0e300; CapState[CapN]=CAP_FREE; CapContacts[CapN]=0;
        ++CapN;
    }
    fclose(fp);
    CapHostDivR=(double *)malloc(ParticleCount*sizeof(double));
    CapHostX=(double (*)[3])malloc(ParticleCount*sizeof(double[3]));
    CapHostV=(double (*)[3])malloc(ParticleCount*sizeof(double[3]));
    CapHostN=(double (*)[3])malloc(ParticleCount*sizeof(double[3]));
    CapHostSolid=(int *)malloc(ParticleCount*sizeof(int));
    log_printf("Capillary aerosol: %d particles from %s, sigma=%e mu_l=%e mu_g=%e surface divR<%e\n",
               CapN,CapAerosolFile,CapAerosolSurfaceTension,CapAerosolLiquidViscosity,
               CapAerosolGasViscosity,CapAerosolSurfaceDivR);
}


// local interface plane for one particle: point y0, unit normal n, velocity us
static int capLocalPlane(const double *x, const int *surf, const int nSurf,
                         double y0[3], double n[3], double us[3])
{
    const double dp=ParticleSpacing;
    int jNear=-1; double dNear=1.0e300;
    for(int k=0;k<nSurf;++k){
        const int j=surf[k];
        double d2=0.0;
        for(int c=0;c<3;++c){ const double y=CapHostX[j][c]+0.5*dp*CapHostN[j][c]; d2+=(x[c]-y)*(x[c]-y); }
        if(d2<dNear){ dNear=d2; jNear=j; }
    }
    if(jNear<0) return 0;
    double yN[3];
    for(int c=0;c<3;++c) yN[c]=CapHostX[jNear][c]+0.5*dp*CapHostN[jNear][c];
    double sw=0.0, sy[3]={0,0,0}, sn[3]={0,0,0}, su[3]={0,0,0};
    const double w0=1.5*dp, rc2=(3.0*w0)*(3.0*w0);
    for(int k=0;k<nSurf;++k){
        const int j=surf[k];
        double y[3], d2=0.0;
        for(int c=0;c<3;++c){ y[c]=CapHostX[j][c]+0.5*dp*CapHostN[j][c]; d2+=(y[c]-yN[c])*(y[c]-yN[c]); }
        if(d2>rc2) continue;
        const double w=exp(-d2/(w0*w0));
        sw+=w;
        for(int c=0;c<3;++c){ sy[c]+=w*y[c]; sn[c]+=w*CapHostN[j][c]; su[c]+=w*CapHostV[j][c]; }
    }
    const double nn=sqrt(sn[0]*sn[0]+sn[1]*sn[1]+sn[2]*sn[2]);
    if(!(sw>0.0 && nn>0.0)) return 0;
    for(int c=0;c<3;++c){ y0[c]=sy[c]/sw; n[c]=sn[c]/nn; us[c]=su[c]/sw; }
    return 1;
}


static void capAerosolStep()
{
    if(CapN==0) return;
    calculateSurfaceNormal();   // IfNormal (liquid rows), surface_tension.inc
    #pragma acc update self(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],PositionDivR[0:ParticleCount],IfNormal[0:ParticleCount][0:DIM],Property[0:ParticleCount],ParticleIndex[0:ParticleCount])

    // outer-layer liquid particles with unit normals
    static int *surf=NULL; static int surfCap=0;
    if(surfCap<ParticleCount){ surf=(int *)realloc(surf,ParticleCount*sizeof(int)); surfCap=ParticleCount; }
    int nSurf=0;
    for(int i=0;i<ParticleCount;++i){
        if(!(FLUID_BEGIN<=Property[i] && Property[i]<FLUID_END)) continue;
        if(SolidBodyCount>0 && SolidBodyOfId[ParticleIndex[i]]>=0) continue;
        const double nn=sqrt(IfNormal[i][0]*IfNormal[i][0]+IfNormal[i][1]*IfNormal[i][1]+IfNormal[i][2]*IfNormal[i][2]);
        if(PositionDivR[i]>=CapAerosolSurfaceDivR || !(nn>0.0)) continue;
        for(int c=0;c<3;++c){ CapHostX[i][c]=Position[i][c]; CapHostV[i][c]=Velocity[i][c]; CapHostN[i][c]=IfNormal[i][c]/nn; }
        surf[nSurf++]=i;
    }

    const double pi=3.14159265358979323846;
    const double sig=CapAerosolSurfaceTension, mul=CapAerosolLiquidViscosity, mug=CapAerosolGasViscosity;
    #pragma omp parallel for schedule(dynamic)
    for(int p=0;p<CapN;++p){
        double y0[3], n[3], us[3];
        if(!capLocalPlane(CapX[p],surf,nSurf,y0,n,us)) continue;
        const double a=0.5*CapD[p];
        const double m=CapRho[p]*pi*CapD[p]*CapD[p]*CapD[p]/6.0;
        const double cth=cos(CapTheta[p]);
        const double omega=sqrt(2.0*pi*sig/m);
        const double tauL=m/(6.0*pi*mul*a);
        double h=CapAerosolSubstepFraction*2.0*pi/omega;
        if(0.2*tauL<h) h=0.2*tauL;
        int nSub=(int)ceil(Dt/h); if(nSub<1) nSub=1;
        const double dt=Dt/nSub;
        double *x=CapX[p], *v=CapV[p];
        for(int k=0;k<nSub;++k){
            const double tl=k*dt;
            double s=0.0;
            for(int c=0;c<3;++c) s+=(x[c]-y0[c]-us[c]*tl)*n[c];
            double F=0.0, gamma=0.0;
            if(s>=a){
                if(CapState[p]==CAP_CONTACT) CapState[p]=CAP_BOUNCED;
                gamma=(mug>0.0 ? 6.0*pi*mug*a/m : 0.0);
            }else if(s>-a){
                if(CapState[p]==CAP_FREE || CapState[p]==CAP_BOUNCED){ CapState[p]=CAP_CONTACT; ++CapContacts[p]; }
                F=-2.0*pi*sig*a*(s/a+cth);
                gamma=6.0*pi*mul*a*0.5*(1.0-s/a)/m;
            }else{
                CapState[p]=CAP_ENGULFED;
                gamma=6.0*pi*mul*a/m;
            }
            // v <- (v + dt F n/m + dt gamma u) / (1 + dt gamma), u: liquid (or still gas)
            const int inGas=(s>=a);
            for(int c=0;c<3;++c){
                const double u=(inGas ? 0.0 : us[c]);
                v[c]=(v[c]+dt*F*n[c]/m+dt*gamma*u)/(1.0+dt*gamma);
            }
            for(int c=0;c<3;++c) x[c]+=dt*v[c];
#ifdef TWO_DIMENSIONAL
            x[2]=0.0; v[2]=0.0;
#endif
            CapS[p]=s;
        }
    }
}


static void capAerosolWriteHistory()
{
    if(CapN==0) return;
    static FILE *fp=NULL;
    if(!fp){
        fp=fopen("cap_aerosol.csv","w");
        if(!fp) return;
        fprintf(fp,"time,id,d_p,rho_p,theta_deg,state,x,y,z,vx,vy,vz,s,contacts\n");
    }
    for(int p=0;p<CapN;++p)
        fprintf(fp,"%.9e,%d,%.6e,%.6e,%.4f,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d\n",Time,p,CapD[p],CapRho[p],
                CapTheta[p]*180.0/3.14159265358979323846,CapState[p],CapX[p][0],CapX[p][1],CapX[p][2],
                CapV[p][0],CapV[p][1],CapV[p][2],(CapS[p]<1.0e299 ? CapS[p] : NAN),CapContacts[p]);
    fflush(fp);
}
