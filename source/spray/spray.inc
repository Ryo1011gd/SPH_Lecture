// =============================================================================
// spray.inc -- SPH spray in FVM air (fvm_air.inc), recycled water particles
//
// Included once by main_implicit.cpp, after fvm_air.inc and aerosol_field.inc.
//
// The spray is SPH water (Type 1) on the single-phase solver path; there are no
// SPH air particles (the air is the FVM grid). The particle count is fixed, so
// spare water waits as Type 3 wall particles in a parking block outside the
// vessel (its wall velocity is zero, so it never moves or acts on the fluid).
// Once per accepted SPH step (host):
//   removal   : water that reaches a solid FVM cell (the vessel wall), leaves the
//               FVM grid or falls below SprayDrainY is parked again (Type 3, at
//               rest): droplets disappear at the wall (Liang et al. 2020, Table 2)
//   injection : SprayParcelRate particles/s are taken from the
//               parking block, put at the nozzle exit (distance SprayExitRadius,
//               direction uniform in +-SprayHalfAngle around -y) with speed U
//   drag      : every water particle is a parcel carrying the liquid volume
//               Vp = SprayFlowRate/SprayParcelRate (so the flow rate does not
//               depend on the SPH spacing) as droplets of diameter D_d
//               (AerosolDropletDiameter). Schiller-Naumann drag towards the FVM
//               air velocity, k = 18 mu_g (1 + 0.15 Re^0.687)/(rho_w D_d^2),
//               exchanged with the bilinear stencil of open MAC faces as an exact
//               two-body relaxation per component (momentum conserving, stable):
//                 m_p = rho_w Vp,  M_a = rho_g dx dy / sum w^2,  mu = m_p/M_a,
//                 dU = (u_a - u_p)(1 - exp(-k (1+mu) dt)),
//                 u_p += dU/(1+mu),  u_face += -m_p (dU/(1+mu)) w /(rho_g dx dy)
//               (parcels are processed serially, so overlapping parcels see each
//               other's exchange; the FVM projection makes the air solenoidal)
//   air       : fvmAdvance() to the new SPH time every FvmEvery steps
//   aerosol   : each parcel removes aerosol mass from its FVM cell
//               (aerFieldParcelSink, aerosol_field.inc); after every air advance
//               the mass fractions are transported (aerFieldTransport)
//
// 3D build (fvm3d_air.inc): the same with a 3D full cone (directions uniform in
// solid angle within SprayHalfAngle around -y), the trilinear stencil of the 3D
// MAC faces, M_a = rho_g dx dy dz / sum w^2, and liquid volumes in m3
// (SprayFlowRate Q [m3/s], Vp = Q/SprayParcelRate).
//
// Data file keys (SprayNozzle enables the spray):
//   SprayNozzle         x y [z]  [m]       (exit point; z in 3D, default 0)
//   SprayVelocity       U        [m/s]     5.0
//   SprayHalfAngle      deg                30
//   SprayExitRadius     r/dp               2.0
//   SprayFlowRate       Q'       [m2/s]    5.0e-4  (liquid volume per unit depth; 3D: Q [m3/s])
//   SprayParcelRate     n        [1/s]     Q'/ParticleVolume (one parcel = one SPH volume)
//   SprayStartTime      t0       [s]       0
//   SprayEndTime        t1       [s]       1e30
//   SprayDrainY         y        [m]       -1e30 (no drain)
//   SprayTwoWay         0|1                1
//   SpraySeed           integer            2024
// =============================================================================

#define SPRAY_PARK_TYPE 3

static int    SprayEnabled=0;
static double SprayNozzle[3]={0.0,0.0,0.0};
static double SprayVelocity=5.0;
static double SprayHalfAngle=30.0;
static double SprayExitRadius=2.0;
static double SprayFlowRate=5.0e-4;
static double SprayParcelRate=0.0;
static double SprayStartTime=0.0;
static double SprayEndTime=1.0e30;
static double SprayDrainY=-1.0e30;
static int    SprayTwoWay=1;
static unsigned long long SpraySeed=2024ULL;

static int     SprParkCount=0;
static double (*SprParkPos)[3]=NULL;      // parking slots (initial Type 3 positions)
static int    *SprParkFree=NULL, SprParkFreeCount=0;   // stack of empty slots
static int    *SprSlotOfId=NULL;           // particle ID -> slot while parked, else -1
static double  SprInjectCarry=0.0;
static long    SprInjected=0, SprDrained=0, SprStarved=0;
static unsigned long long SprRng=0;
static int     SprTeleported=0;
static double  SprVp=0.0;                  // liquid volume of one parcel [m2] (3D: m3)

static int sprayParseDataLine(const char *buf)
{
    char key[256]; long long seed;
    if(sscanf(buf," %255s",key)!=1 || strncmp(key,"Spray",5)) return 0;
    if(sscanf(buf," SprayNozzle %lf %lf %lf",&SprayNozzle[0],&SprayNozzle[1],&SprayNozzle[2])>=2){ SprayEnabled=1; return 1; }
    if(sscanf(buf," SprayVelocity %lf",&SprayVelocity)==1) return 1;
    if(sscanf(buf," SprayHalfAngle %lf",&SprayHalfAngle)==1) return 1;
    if(sscanf(buf," SprayExitRadius %lf",&SprayExitRadius)==1) return 1;
    if(sscanf(buf," SprayFlowRate %lf",&SprayFlowRate)==1) return 1;
    if(sscanf(buf," SprayParcelRate %lf",&SprayParcelRate)==1) return 1;
    if(sscanf(buf," SprayStartTime %lf",&SprayStartTime)==1) return 1;
    if(sscanf(buf," SprayEndTime %lf",&SprayEndTime)==1) return 1;
    if(sscanf(buf," SprayDrainY %lf",&SprayDrainY)==1) return 1;
    if(sscanf(buf," SprayTwoWay %d",&SprayTwoWay)==1) return 1;
    if(sscanf(buf," SpraySeed %lld",&seed)==1){ SpraySeed=(unsigned long long)seed; return 1; }
    return 0;
}

static void sprayInitialize()
{
    if(!SprayEnabled) return;
    SprRng=aerSplitMix(SpraySeed);
    SprSlotOfId=(int *)malloc(ParticleCount*sizeof(int));
    for(int i=0;i<ParticleCount;++i) SprSlotOfId[i]=-1;
    for(int i=0;i<ParticleCount;++i) if(Property[i]==SPRAY_PARK_TYPE) ++SprParkCount;
    SprParkPos=(double (*)[3])malloc((SprParkCount+1)*sizeof(double[3]));
    SprParkFree=(int *)malloc((SprParkCount+1)*sizeof(int));
    int s=0;
    for(int i=0;i<ParticleCount;++i){
        if(Property[i]!=SPRAY_PARK_TYPE) continue;
        for(int d=0;d<3;++d) SprParkPos[s][d]=Position[i][d];
        SprSlotOfId[ParticleIndex[i]]=s; ++s;
    }
    if(!(SprayParcelRate>0.0)) SprayParcelRate=SprayFlowRate/ParticleVolume;
    const double rate=SprayParcelRate;
    SprVp=SprayFlowRate/SprayParcelRate;
#ifdef TWO_DIMENSIONAL
    log_printf("Spray: SPH water, nozzle=(%e,%e) U=%e m/s half-angle=%f deg exit r=%f dp Q'=%e m2/s -> %e parcels/s, Vp=%e m2 (SPH volume %e)\n",
               SprayNozzle[0],SprayNozzle[1],SprayVelocity,SprayHalfAngle,SprayExitRadius,SprayFlowRate,rate,SprVp,ParticleVolume);
#else
    log_printf("Spray 3D: SPH water, nozzle=(%e,%e,%e) U=%e m/s half-angle=%f deg (full cone) exit r=%f dp Q=%e m3/s -> %e parcels/s, Vp=%e m3 (SPH volume %e)\n",
               SprayNozzle[0],SprayNozzle[1],SprayNozzle[2],SprayVelocity,SprayHalfAngle,SprayExitRadius,SprayFlowRate,rate,SprVp,ParticleVolume);
#endif
    log_printf("Spray: parking block %d particles (Type %d), D_d=%e m two-way=%d FVM=%d t=[%e,%e] drain y<%e\n",
               SprParkCount,SPRAY_PARK_TYPE,AerosolDropletDiameter,SprayTwoWay,FvmEnabled,
               SprayStartTime,SprayEndTime,SprayDrainY);
    if(SprParkCount==0) log_printf("Spray: WARNING no Type %d parking particles, nothing can be injected\n",SPRAY_PARK_TYPE);
}

static void sprayDrainAndInject()
{
    int k=0;
    for(int i=0;i<ParticleCount;++i){
        if(Property[i]!=1 || SprParkFreeCount==0) continue;
#ifdef TWO_DIMENSIONAL
        int ci,cj;
        if(Position[i][1]>=SprayDrainY && (!FvmEnabled || fvmCellOf(Position[i][0],Position[i][1],&ci,&cj))) continue;
#else
        int ci,cj,ck;
        if(Position[i][1]>=SprayDrainY && (!FvmEnabled || fvmCellOf(Position[i][0],Position[i][1],Position[i][2],&ci,&cj,&ck))) continue;
#endif
        const int s=SprParkFree[--SprParkFreeCount];
        Property[i]=SPRAY_PARK_TYPE;
        for(int d=0;d<DIM;++d) Position[i][d]=SprParkPos[s][d];
        for(int d=0;d<DIM;++d) Velocity[i][d]=0.0;
        SprSlotOfId[ParticleIndex[i]]=s;
        ++SprDrained; SprTeleported=1;
    }
    if(!(Time>=SprayStartTime && Time<SprayEndTime)) return;
    SprInjectCarry+=SprayParcelRate*Dt;
    const double half=SprayHalfAngle*M_PI/180.0, rExit=SprayExitRadius*ParticleSpacing;
    for(;SprInjectCarry>=1.0;SprInjectCarry-=1.0){
        while(k<ParticleCount && Property[k]!=SPRAY_PARK_TYPE) ++k;
        if(k>=ParticleCount){ ++SprStarved; continue; }
        const int s=SprSlotOfId[ParticleIndex[k]];
        SprParkFree[SprParkFreeCount++]=s;
        SprSlotOfId[ParticleIndex[k]]=-1;
#ifdef TWO_DIMENSIONAL
        const double th=(2.0*aerUniform(&SprRng)-1.0)*half;
        const double e[3]={sin(th),-cos(th),0.0};
#else
        // uniform in solid angle: cos(theta) uniform in [cos(half),1], azimuth uniform
        const double ct=1.0-aerUniform(&SprRng)*(1.0-cos(half)), st=sqrt(fmax(1.0-ct*ct,0.0));
        const double ph=2.0*M_PI*aerUniform(&SprRng);
        const double e[3]={st*cos(ph),-ct,st*sin(ph)};
#endif
        const double r=rExit+aerUniform(&SprRng)*SprayVelocity*Dt;   // spread along one step of travel
        Property[k]=1;
        for(int d=0;d<DIM;++d){ Position[k][d]=SprayNozzle[d]+r*e[d]; Velocity[k][d]=SprayVelocity*e[d]; }
        ++SprInjected; ++k; SprTeleported=1;
    }
}

#ifndef TWO_DIMENSIONAL
static void sprayDrag()
{
    if(!FvmEnabled) return;
    const double D=AerosolDropletDiameter, rhoW=Density[1];
    const double mP=rhoW*SprVp, cellM=FvmP.rho_g*FvmG.dx*FvmG.dy*FvmG.dz;
    const double muG=AerosolGasViscosity, rhoG=FvmP.rho_g;
    double *face[3]={FvmU,FvmV,FvmW};
    for(int i=0;i<ParticleCount;++i){
        if(Property[i]!=1) continue;
        const double x=Position[i][0], y=Position[i][1], z=Position[i][2];
        int idx[3][8], open[3][8]; double w[3][8], ua[3]={0.0,0.0,0.0};
        for(int c=0;c<3;++c){
            fvmStencil(c,x,y,z,idx[c],w[c],open[c]);
            for(int n=0;n<8;++n) ua[c]+=w[c][n]*face[c][idx[c][n]];
        }
        const double ud[3]={Velocity[i][0],Velocity[i][1],Velocity[i][2]};
        aerFieldParcelSink(x,y,z,ud,ua,SprVp,Dt);
        const double rel[3]={ua[0]-ud[0],ua[1]-ud[1],ua[2]-ud[2]};
        const double Re=rhoG*sqrt(rel[0]*rel[0]+rel[1]*rel[1]+rel[2]*rel[2])*D/muG;
        const double kd=18.0*muG*(1.0+0.15*pow(Re,0.687))/(rhoW*D*D);
        for(int c=0;c<3;++c){
            double sw2=0.0;
            for(int n=0;n<8;++n) if(open[c][n]) sw2+=w[c][n]*w[c][n];
            const double mu=(SprayTwoWay && sw2>0.0 ? mP*sw2/cellM : 0.0);
            const double dU=rel[c]*(1.0-exp(-kd*(1.0+mu)*Dt));
            const double dup=dU/(1.0+mu);
            Velocity[i][c]+=dup;
            if(mu>0.0)
                for(int n=0;n<8;++n) if(open[c][n]) face[c][idx[c][n]]-=mP*dup*w[c][n]/cellM;
        }
    }
}
#else
static void sprayDrag()
{
    if(!FvmEnabled) return;
    const double D=AerosolDropletDiameter, rhoW=Density[1];
    const double mP=rhoW*SprVp, cellM=FvmP.rho_g*FvmG.dx*FvmG.dy;
    const double muG=AerosolGasViscosity, rhoG=FvmP.rho_g;
    double *face[2]={FvmF->u.data(),FvmF->v.data()};
    for(int i=0;i<ParticleCount;++i){
        if(Property[i]!=1) continue;
        const double x=Position[i][0], y=Position[i][1];
        int idx[2][4], open[2][4]; double w[2][4], ua[2]={0.0,0.0};
        for(int c=0;c<2;++c){
            fvmStencil(c,x,y,idx[c],w[c],open[c]);
            for(int n=0;n<4;++n) ua[c]+=w[c][n]*face[c][idx[c][n]];
        }
        aerFieldParcelSink(x,y,Velocity[i][0],Velocity[i][1],ua[0],ua[1],SprVp,Dt);
        const double rx=ua[0]-Velocity[i][0], ry=ua[1]-Velocity[i][1];
        const double Re=rhoG*hypot(rx,ry)*D/muG;
        const double kd=18.0*muG*(1.0+0.15*pow(Re,0.687))/(rhoW*D*D);
        const double rel[2]={rx,ry};
        for(int c=0;c<2;++c){
            double sw=0.0, sw2=0.0;
            for(int n=0;n<4;++n) if(open[c][n]){ sw+=w[c][n]; sw2+=w[c][n]*w[c][n]; }
            const double mu=(SprayTwoWay && sw2>0.0 ? mP*sw2/cellM : 0.0);
            const double dU=rel[c]*(1.0-exp(-kd*(1.0+mu)*Dt));
            const double dup=dU/(1.0+mu);
            Velocity[i][c]+=dup;
            if(mu>0.0)
                for(int n=0;n<4;++n) if(open[c][n]) face[c][idx[c][n]]-=mP*dup*w[c][n]/cellM;
        }
    }
}

#endif

// called once per accepted SPH step (after Time has been advanced), before aerosolStep()
static void sprayStep()
{
    if(!SprayEnabled && !FvmEnabled) return;
    if(SprayEnabled){
        #pragma acc update self(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],ParticleIndex[0:ParticleCount])
        sprayDrainAndInject();
        sprayDrag();       // works on the host copy of the FVM face velocities
        #pragma acc update device(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
        // Rebuild now: the fluid/wall row ranges come from the last cell sort, and
        // a particle switched from parked (wall) to water must not be integrated as
        // a wall row (calculatePredictor sent such rows to the origin); the Verlet
        // displacement check alone does not catch every teleport.
        if(SprTeleported){ freeNeighbor(); calculateCellParticle(); calculateNeighbor(); SprTeleported=0; }
    }
    if(FvmEnabled && ++FvmStepCount>=FvmEvery){
        FvmStepCount=0;
        fvmVelocityToDevice();   // accumulated drag reaction
        fvmAdvance();
        fvmVelocityToHost();     // spray drag and the aerosol read the host copy
        aerFieldTransport(FvmTime-AerFLastTime);
        AerFLastTime=FvmTime;
    }
}

static void sprayLogSummary()
{
    if(!SprayEnabled) return;
    log_printf("Spray summary: injected=%ld drained=%ld starved(no parked particle)=%ld parked free slots=%d/%d\n",
               SprInjected,SprDrained,SprStarved,SprParkFreeCount,SprParkCount);
}

