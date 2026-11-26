// =============================================================================
// rigid_body.inc -- meltable rigid bodies (port of the MPH-PMS-Implicit rigid
// model, ../../MPH-PMS-Implicit/source/rigid.cpp), switched by #define RIGID.
//
// Included once by main_implicit.cpp (after solid_body.inc). 2D and 3D,
// single-phase solver path.
//
// Every particle carries a rigid property (one integer per grid row, file
// RigidPropertyFile, same order as the grid):  > 0  body label,  <= 0  none.
// A body is a set of FLUID rows (Type 0 or 1): the particles take part in the
// PPE, the implicit viscosity and the heat conduction like any fluid particle,
// and after every step their motion is projected onto one rigid motion per
// body (Koshizuka et al. 1998; MPH rigid.cpp):
//     V = sum m v / M,  omega = I^-1 L,  I = sum m (|r|^2 1 - r r^T),  L = sum m r x v
//     x_i = X_c + R (r0_i - r0_mean),  v_i = V + omega x (x_i - X_c)
// Modes (data key RigidFixedBodies <label> ...; every other body is free):
//   free   moves as a rigid body under gravity / pressure / viscous forces
//   fixed  anchored: x_i = x0_i, v_i = 0 (also right after the predictor, so
//          the PPE sees it as a stationary meltable structure)
// Melting (TEMPERATURE): a member whose solid fraction falls below
// CriticalSolidFraction[Type] leaves its body for good (MPH updateRigidNumber)
// and continues as an ordinary fluid particle with the thermal viscosity
// mu(fs); re-solidification is the viscosity rise of the thermal model, the
// particle does not rejoin the body.
//
// With RigidPropertyFile set, Type 0 and Type 1 run in the single-phase
// variable-density path (Type 0: structure material, Type 1: melt); the 2D
// two-phase (air/liquid) solver is not used. Without the file nothing changes.
//
// Data keys:
//   RigidPropertyFile  box.rigid          one integer per grid row ('#' comments)
//   RigidFixedBodies   1 2 ...            repeatable, labels of anchored bodies
// Output: VTK field RigidBody (label, -1 = none / released), rigid_history.csv
// =============================================================================

#define RIGID_MAX_BODY 4096

static char   RigidPropertyFile[256] = "";
static int    RigidFixedLabel[RIGID_MAX_BODY];
static int    RigidFixedLabelCount = 0;
static int    RigidBodyCount = 0;
static int    *RigidOfId = NULL;              // [ParticleCount] by original row, -1: not rigid / released
static double (*RigidRef)[3] = NULL;          // [ParticleCount] by original row: body frame (free) or x0 (fixed)
static int    *RigidType0 = NULL;             // [ParticleCount] by original row: member's own (fluid) Type
static int    RigidFixedWallOn = 0;           // 1 while fixed-body members carry the wall Type
static int    RigidLabel[RIGID_MAX_BODY], RigidFixed[RIGID_MAX_BODY], RigidAlive[RIGID_MAX_BODY], RigidN0[RIGID_MAX_BODY];
static double RigidRot[RIGID_MAX_BODY][3][3];
static double RigidCen[RIGID_MAX_BODY][3], RigidVel[RIGID_MAX_BODY][3], RigidOmega[RIGID_MAX_BODY][3];
static double RigidRefMean[RIGID_MAX_BODY][3], RigidMass[RIGID_MAX_BODY];
static double RigidAcc[RIGID_MAX_BODY*16];    // device reductions
static int    RigidHasFree = 0, RigidHasFixed = 0;
#pragma acc declare create(RigidFixed,RigidRot,RigidCen,RigidVel,RigidOmega,RigidRefMean,RigidAcc)


static int rigidParseDataLine(const char *buf)
{
    if(sscanf(buf," RigidPropertyFile %255s",RigidPropertyFile)==1) return 1;
    char key[64];
    int used=0;
    if(sscanf(buf," %63s%n",key,&used)==1 && !strcmp(key,"RigidFixedBodies")){
        const char *p=buf+used;
        int label, n;
        while(sscanf(p,"%d%n",&label,&n)==1){
            if(RigidFixedLabelCount>=RIGID_MAX_BODY){ log_printf("RigidFixedBodies: too many labels\n"); exit(EXIT_FAILURE); }
            RigidFixedLabel[RigidFixedLabelCount++]=label;
            p+=n;
        }
        return 1;
    }
    return 0;
}


// the multi-material single-phase path is used when rigid bodies are defined
static int rigidActive()
{
    return RigidPropertyFile[0]!='\0';
}


static void rigidInitialize()
{
    RigidOfId=(int *)malloc(ParticleCount*sizeof(int));
    RigidRef=(double (*)[3])malloc(ParticleCount*sizeof(double [3]));
    RigidType0=(int *)malloc(ParticleCount*sizeof(int));
    for(int i=0;i<ParticleCount;++i){ RigidOfId[i]=-1; RigidRef[i][0]=RigidRef[i][1]=RigidRef[i][2]=0.0; RigidType0[i]=Property[i]; }

    if(rigidActive()){
        if(TwoPhaseActive){ log_printf("RigidPropertyFile: single-phase path only\n"); exit(EXIT_FAILURE); }
        FILE *fp=fopen(RigidPropertyFile,"r");
        if(!fp){ log_printf("RigidPropertyFile %s not found\n",RigidPropertyFile); exit(EXIT_FAILURE); }
        // grid order: row iP is original row iP until the first cell sort
        char line[4096];
        int iP=0;
        while(iP<ParticleCount && fgets(line,sizeof(line),fp)){
            if(line[0]=='#') continue;
            const char *p=line;
            int label, n;
            while(iP<ParticleCount && sscanf(p,"%d%n",&label,&n)==1){
                p+=n;
                if(label<=0){ ++iP; continue; }
                if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)){
                    log_printf("RigidPropertyFile: row %d (Type %d) is not fluid\n",iP,Property[iP]); exit(EXIT_FAILURE);
                }
                int b=0;
                while(b<RigidBodyCount && RigidLabel[b]!=label) ++b;
                if(b==RigidBodyCount){
                    if(RigidBodyCount>=RIGID_MAX_BODY){ log_printf("RigidPropertyFile: more than %d bodies\n",RIGID_MAX_BODY); exit(EXIT_FAILURE); }
                    RigidLabel[b]=label; RigidN0[b]=0; RigidFixed[b]=0;
                    for(int k=0;k<RigidFixedLabelCount;++k) if(RigidFixedLabel[k]==label) RigidFixed[b]=1;
                    ++RigidBodyCount;
                }
                RigidOfId[iP]=b; ++RigidN0[b];
                ++iP;
            }
        }
        fclose(fp);
        if(iP!=ParticleCount){ log_printf("RigidPropertyFile: %d values for %d particles\n",iP,ParticleCount); exit(EXIT_FAILURE); }

        for(int b=0;b<RigidBodyCount;++b){
            double c[3]={0.0,0.0,0.0}, v[3]={0.0,0.0,0.0}, M=0.0;
            for(int i=0;i<ParticleCount;++i){
                if(RigidOfId[i]!=b) continue;
                M+=Mass[i];
                for(int d=0;d<DIM;++d){ c[d]+=Mass[i]*Position[i][d]; v[d]+=Mass[i]*Velocity[i][d]; }
            }
            for(int d=0;d<3;++d){ c[d]/=M; v[d]/=M; }
            for(int i=0;i<ParticleCount;++i){
                if(RigidOfId[i]!=b) continue;
                for(int d=0;d<DIM;++d) RigidRef[i][d]=(RigidFixed[b] ? Position[i][d] : Position[i][d]-c[d]);
            }
            for(int d=0;d<3;++d){
                RigidCen[b][d]=c[d]; RigidVel[b][d]=(RigidFixed[b] ? 0.0 : v[d]);
                RigidOmega[b][d]=0.0; RigidRefMean[b][d]=0.0;
            }
            for(int r=0;r<3;++r) for(int s=0;s<3;++s) RigidRot[b][r][s]=(r==s ? 1.0 : 0.0);
            RigidMass[b]=M; RigidAlive[b]=RigidN0[b];
            if(RigidFixed[b]) RigidHasFixed=1; else RigidHasFree=1;
            if(RigidBodyCount<=64 || b<8)
                log_printf("Rigid body %d (label %d): %d particles, %s, mass=%e centre=(%e %e %e)\n",
                           b,RigidLabel[b],RigidN0[b],(RigidFixed[b] ? "fixed" : "free"),M,c[0],c[1],c[2]);
        }
        for(int k=0;k<RigidFixedLabelCount;++k){
            int found=0;
            for(int b=0;b<RigidBodyCount;++b) if(RigidLabel[b]==RigidFixedLabel[k]) found=1;
            if(!found) log_printf("RigidFixedBodies: label %d has no particle\n",RigidFixedLabel[k]);
        }
        int nFixed=0;
        for(int b=0;b<RigidBodyCount;++b) nFixed+=RigidFixed[b];
        log_printf("Rigid: %d bodies (%d fixed, %d free) from %s; melting release at fs < CriticalSolidFraction\n",
                   RigidBodyCount,nFixed,RigidBodyCount-nFixed,RigidPropertyFile);
    }
    #pragma acc enter data copyin(RigidOfId[0:ParticleCount],RigidRef[0:ParticleCount][0:3],RigidType0[0:ParticleCount])
    #pragma acc update device(RigidFixed[0:RIGID_MAX_BODY],RigidRot[0:RIGID_MAX_BODY][0:3][0:3],RigidCen[0:RIGID_MAX_BODY][0:3],RigidVel[0:RIGID_MAX_BODY][0:3],RigidOmega[0:RIGID_MAX_BODY][0:3],RigidRefMean[0:RIGID_MAX_BODY][0:3])
}


// anchored bodies: back to x0 with zero velocity
static void rigidHoldFixed()
{
    if(!RigidHasFixed) return;
    #pragma acc parallel loop present(RigidFixed[0:RIGID_MAX_BODY],RigidOfId[0:ParticleCount],RigidRef[0:ParticleCount][0:3],ParticleIndex[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
    for(int iP=0;iP<ParticleCount;++iP){
        const int id=ParticleIndex[iP];
        const int b=RigidOfId[id];
        if(b<0 || !RigidFixed[b]) continue;
        #pragma acc loop seq
        for(int d=0;d<DIM;++d){ Position[iP][d]=RigidRef[id][d]; Velocity[iP][d]=0.0; }
    }
}


// Fixed bodies are walls for the flow solver: their (not yet released) members
// carry the wall Type WALL_BEGIN during the hydrodynamic step, so the wall
// treatment (Neumann PPE row, wall pressure ghosts, no-slip, hard
// non-penetration) keeps other fluid out; as fluid rows held at x0 they let a
// pressurised melt seep through.  The thermal step sees their own Type again.
// Returns 1 when a Type changed (cell sort / neighbour lists must be rebuilt).
static int rigidFixedAsWall(const int on)
{
    if(!RigidHasFixed || RigidFixedWallOn==on) return 0;
    const int wt=WALL_BEGIN;
    #pragma acc parallel loop present(RigidFixed[0:RIGID_MAX_BODY],RigidOfId[0:ParticleCount],RigidType0[0:ParticleCount],ParticleIndex[0:ParticleCount],Property[0:ParticleCount])
    for(int iP=0;iP<ParticleCount;++iP){
        const int id=ParticleIndex[iP];
        const int b=RigidOfId[id];
        if(b<0 || !RigidFixed[b]) continue;
        Property[iP]=(on ? wt : RigidType0[id]);
    }
    RigidFixedWallOn=on;
    return 1;
}


// sum over the particles of every free body into RigidAcc[16 b + k]
static void rigidReduce(const int stage)
{
    const int nAcc=16*RigidBodyCount;
    #pragma acc parallel loop present(RigidAcc[0:RIGID_MAX_BODY*16])
    for(int k=0;k<nAcc;++k) RigidAcc[k]=0.0;

    #pragma acc parallel loop present(RigidAcc[0:RIGID_MAX_BODY*16],RigidFixed[0:RIGID_MAX_BODY],RigidCen[0:RIGID_MAX_BODY][0:3],RigidOfId[0:ParticleCount],RigidRef[0:ParticleCount][0:3],ParticleIndex[0:ParticleCount],Mass[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
    for(int iP=0;iP<ParticleCount;++iP){
        const int id=ParticleIndex[iP];
        const int b=RigidOfId[id];
        if(b<0 || RigidFixed[b]) continue;
        double *a=&RigidAcc[16*b];
        const double m=Mass[iP];
        double x[3]={0.0,0.0,0.0}, v[3]={0.0,0.0,0.0};
        #pragma acc loop seq
        for(int d=0;d<DIM;++d){ x[d]=Position[iP][d]; v[d]=Velocity[iP][d]; }
        if(stage==0){
            // mass, centre, momentum, mean reference position
            #pragma acc atomic update
            a[0]+=m;
            #pragma acc atomic update
            a[1]+=m*x[0];
            #pragma acc atomic update
            a[2]+=m*x[1];
            #pragma acc atomic update
            a[3]+=m*x[2];
            #pragma acc atomic update
            a[4]+=m*v[0];
            #pragma acc atomic update
            a[5]+=m*v[1];
            #pragma acc atomic update
            a[6]+=m*v[2];
            #pragma acc atomic update
            a[7]+=m*RigidRef[id][0];
            #pragma acc atomic update
            a[8]+=m*RigidRef[id][1];
            #pragma acc atomic update
            a[9]+=m*RigidRef[id][2];
            #pragma acc atomic update
            a[10]+=1.0;
        }else{
            // angular momentum and inertia about the centre
            const double rx=x[0]-RigidCen[b][0];
            const double ry=x[1]-RigidCen[b][1];
            const double rz=x[2]-RigidCen[b][2];
            const double r2=rx*rx+ry*ry+rz*rz;
            #pragma acc atomic update
            a[0]+=m*(ry*v[2]-rz*v[1]);
            #pragma acc atomic update
            a[1]+=m*(rz*v[0]-rx*v[2]);
            #pragma acc atomic update
            a[2]+=m*(rx*v[1]-ry*v[0]);
            #pragma acc atomic update
            a[3]+=m*(r2-rx*rx);
            #pragma acc atomic update
            a[4]+=m*(r2-ry*ry);
            #pragma acc atomic update
            a[5]+=m*(r2-rz*rz);
            #pragma acc atomic update
            a[6]+=-m*rx*ry;
            #pragma acc atomic update
            a[7]+=-m*rx*rz;
            #pragma acc atomic update
            a[8]+=-m*ry*rz;
        }
    }
    #pragma acc update self(RigidAcc[0:RIGID_MAX_BODY*16])
}


// after the full step: free bodies -> rigid motion, fixed bodies -> x0
static void rigidProjection()
{
    if(RigidBodyCount==0) return;
    rigidHoldFixed();
    if(!RigidHasFree) return;

    rigidReduce(0);
    for(int b=0;b<RigidBodyCount;++b){
        if(RigidFixed[b]) continue;
        const double *a=&RigidAcc[16*b];
        RigidAlive[b]=(int)(a[10]+0.5);
        if(RigidAlive[b]==0) continue;
        RigidMass[b]=a[0];
        for(int d=0;d<3;++d){
            RigidCen[b][d]=a[1+d]/a[0];
            RigidVel[b][d]=a[4+d]/a[0];
            RigidRefMean[b][d]=a[7+d]/a[0];
        }
#ifdef TWO_DIMENSIONAL
        RigidCen[b][2]=0.0; RigidVel[b][2]=0.0; RigidRefMean[b][2]=0.0;
#endif
    }
    #pragma acc update device(RigidCen[0:RIGID_MAX_BODY][0:3])

    rigidReduce(1);
    for(int b=0;b<RigidBodyCount;++b){
        if(RigidFixed[b] || RigidAlive[b]==0) continue;
        const double *a=&RigidAcc[16*b];
        double w[3]={0.0,0.0,0.0};
#ifdef TWO_DIMENSIONAL
        if(a[5]>1.0e-300) w[2]=a[2]/a[5];
#else
        const double I[3][3]={{a[3],a[6],a[7]},{a[6],a[4],a[8]},{a[7],a[8],a[5]}};
        const double c00=I[1][1]*I[2][2]-I[1][2]*I[2][1];
        const double c01=I[1][2]*I[2][0]-I[1][0]*I[2][2];
        const double c02=I[1][0]*I[2][1]-I[1][1]*I[2][0];
        const double det=I[0][0]*c00+I[0][1]*c01+I[0][2]*c02;
        const double scale=a[3]+a[4]+a[5];
        if(scale>1.0e-300 && fabs(det)>1.0e-12*scale*scale*scale){     // < 3 non-collinear members: no rotation
            const double Ii[3][3]={
                {c00/det,(I[0][2]*I[2][1]-I[0][1]*I[2][2])/det,(I[0][1]*I[1][2]-I[0][2]*I[1][1])/det},
                {c01/det,(I[0][0]*I[2][2]-I[0][2]*I[2][0])/det,(I[0][2]*I[1][0]-I[0][0]*I[1][2])/det},
                {c02/det,(I[0][1]*I[2][0]-I[0][0]*I[2][1])/det,(I[0][0]*I[1][1]-I[0][1]*I[1][0])/det}};
            for(int r=0;r<3;++r) w[r]=Ii[r][0]*a[0]+Ii[r][1]*a[1]+Ii[r][2]*a[2];
        }
#endif
        for(int d=0;d<3;++d) RigidOmega[b][d]=w[d];

        // R <- exp([omega Dt]x) R  (Rodrigues)
        const double th=sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])*Dt;
        if(th>1.0e-300){
            const double k[3]={w[0]*Dt/th,w[1]*Dt/th,w[2]*Dt/th};
            const double s=sin(th), c=1.0-cos(th);
            const double K[3][3]={{0,-k[2],k[1]},{k[2],0,-k[0]},{-k[1],k[0],0}};
            double Q[3][3], Rn[3][3];
            for(int r=0;r<3;++r) for(int q=0;q<3;++q){
                double kk=0.0; for(int m=0;m<3;++m) kk+=K[r][m]*K[m][q];
                Q[r][q]=(r==q ? 1.0 : 0.0)+s*K[r][q]+c*kk;
            }
            for(int r=0;r<3;++r) for(int q=0;q<3;++q){
                double x=0.0; for(int m=0;m<3;++m) x+=Q[r][m]*RigidRot[b][m][q];
                Rn[r][q]=x;
            }
            for(int r=0;r<3;++r) for(int q=0;q<3;++q) RigidRot[b][r][q]=Rn[r][q];
        }
    }
    #pragma acc update device(RigidRot[0:RIGID_MAX_BODY][0:3][0:3],RigidVel[0:RIGID_MAX_BODY][0:3],RigidOmega[0:RIGID_MAX_BODY][0:3],RigidRefMean[0:RIGID_MAX_BODY][0:3])

    #pragma acc parallel loop present(RigidFixed[0:RIGID_MAX_BODY],RigidRot[0:RIGID_MAX_BODY][0:3][0:3],RigidCen[0:RIGID_MAX_BODY][0:3],RigidVel[0:RIGID_MAX_BODY][0:3],RigidOmega[0:RIGID_MAX_BODY][0:3],RigidRefMean[0:RIGID_MAX_BODY][0:3],RigidOfId[0:ParticleCount],RigidRef[0:ParticleCount][0:3],ParticleIndex[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
    for(int iP=0;iP<ParticleCount;++iP){
        const int id=ParticleIndex[iP];
        const int b=RigidOfId[id];
        if(b<0 || RigidFixed[b]) continue;
        double s[3], r[3];
        #pragma acc loop seq
        for(int d=0;d<3;++d) s[d]=RigidRef[id][d]-RigidRefMean[b][d];
        #pragma acc loop seq
        for(int d=0;d<3;++d) r[d]=RigidRot[b][d][0]*s[0]+RigidRot[b][d][1]*s[1]+RigidRot[b][d][2]*s[2];
        const double *w=RigidOmega[b];
        const double v[3]={RigidVel[b][0]+w[1]*r[2]-w[2]*r[1],
                           RigidVel[b][1]+w[2]*r[0]-w[0]*r[2],
                           RigidVel[b][2]+w[0]*r[1]-w[1]*r[0]};
        #pragma acc loop seq
        for(int d=0;d<DIM;++d){ Position[iP][d]=RigidCen[b][d]+r[d]; Velocity[iP][d]=v[d]; }
    }
}


// melted members leave their body (MPH updateRigidNumber); returns released count
static int rigidRelease()
{
#ifdef TEMPERATURE
    if(RigidBodyCount==0 || !ThermalEnabled) return 0;
    int released=0;
    #pragma acc parallel loop reduction(+:released) present(RigidOfId[0:ParticleCount],ParticleIndex[0:ParticleCount],Property[0:ParticleCount],SolidFraction[0:ParticleCount],CriticalSolidFraction[0:TYPE_COUNT])
    for(int iP=0;iP<ParticleCount;++iP){
        const int id=ParticleIndex[iP];
        if(RigidOfId[id]<0) continue;
        if(SolidFraction[iP]<CriticalSolidFraction[Property[iP]]){ RigidOfId[id]=-1; ++released; }
    }
    if(released>0 && LogVerbose) log_printf("Rigid: %d particles melted out at Time=%e\n",released,Time);
    return released;
#else
    return 0;
#endif
}


// VTK: body label per (sorted) particle, -1 = none / released
static void rigidWriteVtkField(FILE *fp)
{
    if(RigidBodyCount==0) return;
    #pragma acc update self(RigidOfId[0:ParticleCount])
    fprintf(fp, "SCALARS RigidBody float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for(int iP=0;iP<ParticleCount;++iP){
        const int b=RigidOfId[ParticleIndex[iP]];
        fprintf(fp, "%d\n", (b<0 ? -1 : RigidLabel[b]));
    }
    fprintf(fp, "\n");
}


// body state history (one line per body and prof frame)
static void rigidWriteHistory()
{
    if(RigidBodyCount==0) return;
    static FILE *fp=NULL;
    if(!fp){
        fp=fopen("rigid_history.csv","w");
        if(!fp) return;
        fprintf(fp,"time,label,fixed,n0,n_rigid,xc,yc,zc,vx,vy,vz,wx,wy,wz\n");
    }
    #pragma acc update self(RigidOfId[0:ParticleCount])
    for(int b=0;b<RigidBodyCount;++b) RigidAlive[b]=0;
    for(int i=0;i<ParticleCount;++i) if(RigidOfId[i]>=0) ++RigidAlive[RigidOfId[i]];
    for(int b=0;b<RigidBodyCount;++b)
        fprintf(fp,"%.9e,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",Time,RigidLabel[b],RigidFixed[b],
                RigidN0[b],RigidAlive[b],RigidCen[b][0],RigidCen[b][1],RigidCen[b][2],
                RigidVel[b][0],RigidVel[b][1],RigidVel[b][2],RigidOmega[b][0],RigidOmega[b][1],RigidOmega[b][2]);
    fflush(fp);
}
