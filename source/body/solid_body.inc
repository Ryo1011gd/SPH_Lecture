// =============================================================================
// solid_body.inc -- resolved rigid particles (solid aerosol) with wettability
//
// Included once by main_implicit.cpp (after surface_tension.inc). 2D and 3D,
// single-phase only.
//
// Passively moving solid: the particles of a body are ordinary fluid rows
// (Type 1) for the PPE and the viscosity; after every step their motion is
// projected onto a rigid motion (Koshizuka et al. 1998):
//     V = mean v,  omega = I^-1 L,  I = sum(|r|^2 1 - r r^T),  L = sum r x v
//     x_i = X_c + R r0_i,  v_i = V + omega x (R r0_i)
// with r0 the reference shape (body frame, t = 0) and R updated by omega Dt.
// Bodies keep their shape exactly; shifting of their particles is overwritten.
//
// Wettability: with the Laplace pressure BC (SurfaceTensionModel 2) a solid in
// contact only feels the liquid pressure (~2 sigma/R, pushes it out); the pull
// of the contact line is missing. It is supplied by a solid-liquid adhesion
// pair force of Kondo form,
//     F_ij = -C_a (r - dp)(r - re) e_ij,   re = 3.2 dp,  liquid i <-> solid j
// calibrated like the Kondo cohesion but with the Dupre work of adhesion
//     W_a = sigma (1 + cos theta):   W_a A = (C_a / 3) S,
//     S = sum_{layers z>=1} z (r - 1.5 dp + re/2)(r - re)^2,  A = dp (2D) / dp^2 (3D)
// theta = 0: W_a = 2 sigma (pulled in); theta = 180 deg: W_a = 0 (no adhesion,
// the liquid pressure pushes the body away). The force acts on the solid only:
// its reaction on the liquid surface band (no divergence constraint, alpha < 1)
// compresses the band and launches the body (same failure as Kondo/Akinci).
// The drop is >> heavier than an aerosol, so the missing reaction does not
// change the body motion; what is lost is the local pull of the liquid toward
// the particle (spreading of the meniscus). The liquid surface normals and
// curvature ignore solid particles; free-surface rows of solid particles
// hold p = 0.
//
// SolidBodyFile (data key): one body per line, original grid rows
//     first  count  theta_deg            ('#' comments)
// Particle identity across the cell sort: ParticleIndex (original row).
// =============================================================================

#define SOLID_MAX_BODY 256

static char   SolidBodyFile[256] = "";
static int    SolidBodyCount = 0;
static int    *SolidBodyOfId = NULL;        // [ParticleCount] by original row, -1: liquid
static double (*SolidRef)[3] = NULL;        // [ParticleCount] by original row, body frame
static double SolidWet[SOLID_MAX_BODY];     // adhesion coefficient C_a per body
static double SolidTheta[SOLID_MAX_BODY];
static int    SolidFirst[SOLID_MAX_BODY], SolidN[SOLID_MAX_BODY];
static double SolidRot[SOLID_MAX_BODY][3][3];
static double SolidCen[SOLID_MAX_BODY][3], SolidVel[SOLID_MAX_BODY][3], SolidOmega[SOLID_MAX_BODY][3];
static double SolidAcc[SOLID_MAX_BODY*16];  // device reductions
#pragma acc declare create(SolidBodyCount,SolidWet,SolidRot,SolidCen,SolidVel,SolidOmega,SolidAcc,SolidAdhesionRadius)
static double SolidAdhesionRadius = 0.0;


static int solidParseDataLine(const char *buf)
{
    return sscanf(buf," SolidBodyFile %255s",SolidBodyFile)==1;
}


#pragma acc routine seq
static int solidBodyOf(const int iP)
{
    return (SolidBodyCount>0 ? SolidBodyOfId[ParticleIndex[iP]] : -1);
}


static void solidInitialize()
{
    SolidBodyOfId=(int *)malloc(ParticleCount*sizeof(int));
    SolidRef=(double (*)[3])malloc(ParticleCount*sizeof(double [3]));
    for(int i=0;i<ParticleCount;++i){ SolidBodyOfId[i]=-1; SolidRef[i][0]=SolidRef[i][1]=SolidRef[i][2]=0.0; }

    if(SolidBodyFile[0]){
        if(TwoPhaseActive){ log_printf("SolidBodyFile: single-phase only\n"); exit(EXIT_FAILURE); }
        FILE *fp=fopen(SolidBodyFile,"r");
        if(!fp){ log_printf("SolidBodyFile %s not found\n",SolidBodyFile); exit(EXIT_FAILURE); }
        char line[512];
        while(fgets(line,sizeof(line),fp)){
            int first,count; double theta;
            if(line[0]=='#' || sscanf(line,"%d %d %lf",&first,&count,&theta)!=3) continue;
            if(SolidBodyCount>=SOLID_MAX_BODY || first<0 || count<1 || first+count>ParticleCount){
                log_printf("SolidBodyFile: invalid body line: %s",line); exit(EXIT_FAILURE);
            }
            const int b=SolidBodyCount++;
            SolidFirst[b]=first; SolidN[b]=count; SolidTheta[b]=theta;
        }
        fclose(fp);
    }

    // adhesion coefficient from the Dupre work of adhesion (Kondo lattice sum)
    const double dp=ParticleSpacing, re=3.2*dp;
    const int range=(int)ceil(re/dp);
    double S=0.0;
#ifdef TWO_DIMENSIONAL
    for(int iz=1;iz<=range;++iz) for(int ix=-range;ix<=range;++ix){
        const double r=dp*sqrt((double)(ix*ix+iz*iz));
        if(r<re) S+=(double)iz*(r-1.5*dp+0.5*re)*(r-re)*(r-re);
    }
    const double area=dp;
#else
    for(int iz=1;iz<=range;++iz) for(int iy=-range;iy<=range;++iy) for(int ix=-range;ix<=range;++ix){
        const double r=dp*sqrt((double)(ix*ix+iy*iy+iz*iz));
        if(r<re) S+=(double)iz*(r-1.5*dp+0.5*re)*(r-re)*(r-re);
    }
    const double area=dp*dp;
#endif
    SolidAdhesionRadius=re;
    for(int b=0;b<SolidBodyCount;++b){
        const double Wa=SurfaceTension[1]*(1.0+cos(SolidTheta[b]*3.14159265358979323846/180.0));
        SolidWet[b]=(S>1.0e-300 ? 3.0*Wa*area/S : 0.0);
    }

    // grid order: row iP is original row iP until the first cell sort
    for(int b=0;b<SolidBodyCount;++b){
        double c[3]={0.0,0.0,0.0};
        for(int i=SolidFirst[b];i<SolidFirst[b]+SolidN[b];++i){
            if(!(FLUID_BEGIN<=Property[i] && Property[i]<FLUID_END)){ log_printf("SolidBodyFile: body %d row %d is not fluid\n",b,i); exit(EXIT_FAILURE); }
            if(SolidBodyOfId[i]>=0){ log_printf("SolidBodyFile: row %d in two bodies\n",i); exit(EXIT_FAILURE); }
            SolidBodyOfId[i]=b;
            for(int d=0;d<3;++d) c[d]+=Position[i][d];
        }
        for(int d=0;d<3;++d){ c[d]/=SolidN[b]; SolidCen[b][d]=c[d]; SolidVel[b][d]=0.0; SolidOmega[b][d]=0.0; }
        for(int i=SolidFirst[b];i<SolidFirst[b]+SolidN[b];++i){
            for(int d=0;d<3;++d){ SolidRef[i][d]=Position[i][d]-c[d]; SolidVel[b][d]+=Velocity[i][d]/SolidN[b]; }
        }
        for(int r=0;r<3;++r) for(int s=0;s<3;++s) SolidRot[b][r][s]=(r==s ? 1.0 : 0.0);
        log_printf("Solid body %d: rows %d..%d (%d particles) theta=%g deg C_a=%g centre=(%e %e %e)\n",
                   b,SolidFirst[b],SolidFirst[b]+SolidN[b]-1,SolidN[b],SolidTheta[b],SolidWet[b],c[0],c[1],c[2]);
    }
    #pragma acc enter data copyin(SolidBodyOfId[0:ParticleCount],SolidRef[0:ParticleCount][0:3])
    #pragma acc update device(SolidBodyCount,SolidAdhesionRadius,SolidWet[0:SOLID_MAX_BODY],SolidRot[0:SOLID_MAX_BODY][0:3][0:3],SolidCen[0:SOLID_MAX_BODY][0:3])
}


// solid-liquid adhesion (explicit, on the solid particles only)
static void calculateSolidAdhesion()
{
    if(SolidBodyCount==0) return;
    #pragma acc kernels present(Force[0:ParticleCount][0:DIM],Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],NeighborCount[0:ParticleCount],NeighborInd[0:NeighborIndCount],NeighborPtr[0:PowerParticleCount],SolidBodyOfId[0:ParticleCount],ParticleIndex[0:ParticleCount],SolidWet[0:SOLID_MAX_BODY])
    #pragma acc loop independent
    #pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        if(!(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END)) continue;
        const int bi=solidBodyOf(iP);
        if(bi<0) continue;
        double fx=0.0, fy=0.0, fz=0.0;
        #pragma acc loop seq
        for(int jN=0;jN<NeighborCount[iP];++jN){
            const int jP=NeighborInd[NeighborPtr[iP]+jN];
            if(iP==jP || !(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END)) continue;
            if(solidBodyOf(jP)>=0) continue;                  // liquid neighbours only
            const double Ca=SolidWet[bi];
            const double dx=Position[jP][0]-Position[iP][0];
            const double dy=Position[jP][1]-Position[iP][1];
            const double dz=Position[jP][2]-Position[iP][2];
            const double r2=dx*dx+dy*dy+dz*dz;
            if(r2<=1.0e-30 || r2>=SolidAdhesionRadius*SolidAdhesionRadius) continue;
            const double r=sqrt(r2);
            const double fmag=-Ca*(r-ParticleSpacing)*(r-SolidAdhesionRadius)/r;
            fx+=fmag*dx; fy+=fmag*dy; fz+=fmag*dz;
        }
        Force[iP][0]+=fx;
        Force[iP][1]+=fy;
        Force[iP][2]+=fz;
    }
}


// sum over the particles of every body into SolidAcc[16 b + k]
static void solidReduce(const int stage)
{
    const int nAcc=16*SolidBodyCount;
    #pragma acc parallel loop present(SolidAcc[0:SOLID_MAX_BODY*16])
    for(int k=0;k<nAcc;++k) SolidAcc[k]=0.0;

    #pragma acc parallel loop present(SolidAcc[0:SOLID_MAX_BODY*16],SolidCen[0:SOLID_MAX_BODY][0:3],SolidBodyOfId[0:ParticleCount],ParticleIndex[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
    for(int iP=0;iP<ParticleCount;++iP){
        const int b=SolidBodyOfId[ParticleIndex[iP]];
        if(b<0) continue;
        double *a=&SolidAcc[16*b];
        if(stage==0){
            // centre and momentum
            #pragma acc atomic update
            a[0]+=Position[iP][0];
            #pragma acc atomic update
            a[1]+=Position[iP][1];
            #pragma acc atomic update
            a[2]+=Position[iP][2];
            #pragma acc atomic update
            a[3]+=Velocity[iP][0];
            #pragma acc atomic update
            a[4]+=Velocity[iP][1];
            #pragma acc atomic update
            a[5]+=Velocity[iP][2];
        }else{
            // angular momentum and inertia about the centre
            const double rx=Position[iP][0]-SolidCen[b][0];
            const double ry=Position[iP][1]-SolidCen[b][1];
            const double rz=Position[iP][2]-SolidCen[b][2];
            const double vx=Velocity[iP][0], vy=Velocity[iP][1], vz=Velocity[iP][2];
            const double r2=rx*rx+ry*ry+rz*rz;
            #pragma acc atomic update
            a[0]+=ry*vz-rz*vy;
            #pragma acc atomic update
            a[1]+=rz*vx-rx*vz;
            #pragma acc atomic update
            a[2]+=rx*vy-ry*vx;
            #pragma acc atomic update
            a[3]+=r2-rx*rx;
            #pragma acc atomic update
            a[4]+=r2-ry*ry;
            #pragma acc atomic update
            a[5]+=r2-rz*rz;
            #pragma acc atomic update
            a[6]+=-rx*ry;
            #pragma acc atomic update
            a[7]+=-rx*rz;
            #pragma acc atomic update
            a[8]+=-ry*rz;
        }
    }
    #pragma acc update self(SolidAcc[0:SOLID_MAX_BODY*16])
}


static void solidRigidProjection()
{
    if(SolidBodyCount==0) return;

    solidReduce(0);
    for(int b=0;b<SolidBodyCount;++b){
        for(int d=0;d<3;++d){
            SolidCen[b][d]=SolidAcc[16*b+d]/SolidN[b];
            SolidVel[b][d]=SolidAcc[16*b+3+d]/SolidN[b];
        }
#ifdef TWO_DIMENSIONAL
        SolidCen[b][2]=0.0; SolidVel[b][2]=0.0;
#endif
    }
    #pragma acc update device(SolidCen[0:SOLID_MAX_BODY][0:3])

    solidReduce(1);
    for(int b=0;b<SolidBodyCount;++b){
        const double *a=&SolidAcc[16*b];
        double w[3]={0.0,0.0,0.0};
#ifdef TWO_DIMENSIONAL
        if(a[5]>1.0e-300) w[2]=a[2]/a[5];
#else
        const double I[3][3]={{a[3],a[6],a[7]},{a[6],a[4],a[8]},{a[7],a[8],a[5]}};
        const double c00=I[1][1]*I[2][2]-I[1][2]*I[2][1];
        const double c01=I[1][2]*I[2][0]-I[1][0]*I[2][2];
        const double c02=I[1][0]*I[2][1]-I[1][1]*I[2][0];
        const double det=I[0][0]*c00+I[0][1]*c01+I[0][2]*c02;
        if(fabs(det)>1.0e-300){
            const double Ii[3][3]={
                {c00/det,(I[0][2]*I[2][1]-I[0][1]*I[2][2])/det,(I[0][1]*I[1][2]-I[0][2]*I[1][1])/det},
                {c01/det,(I[0][0]*I[2][2]-I[0][2]*I[2][0])/det,(I[0][2]*I[1][0]-I[0][0]*I[1][2])/det},
                {c02/det,(I[0][1]*I[2][0]-I[0][0]*I[2][1])/det,(I[0][0]*I[1][1]-I[0][1]*I[1][0])/det}};
            for(int r=0;r<3;++r) w[r]=Ii[r][0]*a[0]+Ii[r][1]*a[1]+Ii[r][2]*a[2];
        }
#endif
        for(int d=0;d<3;++d) SolidOmega[b][d]=w[d];

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
                double x=0.0; for(int m=0;m<3;++m) x+=Q[r][m]*SolidRot[b][m][q];
                Rn[r][q]=x;
            }
            for(int r=0;r<3;++r) for(int q=0;q<3;++q) SolidRot[b][r][q]=Rn[r][q];
        }
    }
    #pragma acc update device(SolidRot[0:SOLID_MAX_BODY][0:3][0:3],SolidVel[0:SOLID_MAX_BODY][0:3],SolidOmega[0:SOLID_MAX_BODY][0:3])

    #pragma acc parallel loop present(SolidRot[0:SOLID_MAX_BODY][0:3][0:3],SolidCen[0:SOLID_MAX_BODY][0:3],SolidVel[0:SOLID_MAX_BODY][0:3],SolidOmega[0:SOLID_MAX_BODY][0:3],SolidBodyOfId[0:ParticleCount],SolidRef[0:ParticleCount][0:3],ParticleIndex[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
    for(int iP=0;iP<ParticleCount;++iP){
        const int id=ParticleIndex[iP];
        const int b=SolidBodyOfId[id];
        if(b<0) continue;
        double r[3];
        for(int d=0;d<3;++d)
            r[d]=SolidRot[b][d][0]*SolidRef[id][0]+SolidRot[b][d][1]*SolidRef[id][1]+SolidRot[b][d][2]*SolidRef[id][2];
        const double *w=SolidOmega[b];
        Position[iP][0]=SolidCen[b][0]+r[0];
        Position[iP][1]=SolidCen[b][1]+r[1];
        Position[iP][2]=SolidCen[b][2]+r[2];
        Velocity[iP][0]=SolidVel[b][0]+w[1]*r[2]-w[2]*r[1];
        Velocity[iP][1]=SolidVel[b][1]+w[2]*r[0]-w[0]*r[2];
        Velocity[iP][2]=SolidVel[b][2]+w[0]*r[1]-w[1]*r[0];
    }
}


// body centre / velocity history (one line per prof frame)
static void solidWriteHistory()
{
    if(SolidBodyCount==0) return;
    static FILE *fp=NULL;
    if(!fp){
        fp=fopen("solid_history.csv","w");
        if(!fp) return;
        fprintf(fp,"time,body,theta_deg,xc,yc,zc,vx,vy,vz,wx,wy,wz\n");
    }
    for(int b=0;b<SolidBodyCount;++b)
        fprintf(fp,"%.9e,%d,%g,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",Time,b,SolidTheta[b],
                SolidCen[b][0],SolidCen[b][1],SolidCen[b][2],SolidVel[b][0],SolidVel[b][1],SolidVel[b][2],
                SolidOmega[b][0],SolidOmega[b][1],SolidOmega[b][2]);
    fflush(fp);
}
