// =============================================================================
// fvm_air.inc -- carrier air on the VOF/FVM grid (namespace fvm, fvm_core.inc)
//
// Included once by main_implicit.cpp, after aerosol.inc and before spray.inc.
//
// The air is solved by the VOF_FVM_2Phase solver with F=0 everywhere (gas
// only), on its own MAC grid with solid cells for the vessel. The SPH solver
// keeps the spray water (Type 1) and the walls; the aerosol (Property 5) is
// carried by the FVM air velocity. Coupling, once per accepted SPH step:
//   spray.inc  : droplet drag exchanged between SPH water and the FVM faces
//   fvmAdvance : FVM substeps up to the SPH time (each with its own CFL dt,
//                retries and acceptance checks exactly as the FVM main loop)
//
// Data file keys:
//   FvmData   path      FVM .data (rho_gas, mu_gas, cfl, ...)    (enables FVM)
//   FvmGrid   path      FVM .grid (VOF_GRID_V1, type 2 = solid)
//   FvmVtk    pattern   FVM VTK frames, e.g. air_%04d.vtk, written with every
//                       SPH VTK frame                          (default: none)
//   FvmEvery  n         advance the air every n SPH steps (default 1). Between
//                       advances the drag reaction accumulates in the host copy
//                       of the face velocities, which is uploaded just before.
// =============================================================================

static int  FvmEnabled=0;
static int  FvmEvery=1, FvmStepCount=0;
static char FvmDataFile[1024]="", FvmGridFile[1024]="", FvmVtkPattern[1024]="";

static fvm::Params      FvmP;
static fvm::Geometry    FvmG;
static fvm::Fields     *FvmF=NULL;
static fvm::StateBackup *FvmBackup=NULL;
static fvm::Workspace  *FvmWork=NULL;
static double FvmTime=0.0, FvmPrevDt=-1.0;
static int    FvmStep=0, FvmFrame=0, FvmLastSubsteps=0;
static double FvmLastRes=0.0;

static int fvmParseDataLine(const char *buf)
{
    char key[256];
    if(sscanf(buf," %255s",key)!=1 || strncmp(key,"Fvm",3)) return 0;
    if(sscanf(buf," FvmData %1023s",FvmDataFile)==1){ FvmEnabled=1; return 1; }
    if(sscanf(buf," FvmGrid %1023s",FvmGridFile)==1) return 1;
    if(sscanf(buf," FvmVtk %1023s",FvmVtkPattern)==1) return 1;
    if(sscanf(buf," FvmEvery %d",&FvmEvery)==1) return 1;
    return 0;
}

// host copies of the face velocities (before coupling) and back (after)
static void fvmVelocityToHost()
{
    if(!FvmEnabled) return;
    double *up=FvmF->u.data(), *vp=FvmF->v.data();
    const int nuf=FvmF->nu(), nvf=FvmF->nv();
    #pragma acc update self(up[0:nuf],vp[0:nvf])
}
static void fvmVelocityToDevice()
{
    if(!FvmEnabled) return;
    double *up=FvmF->u.data(), *vp=FvmF->v.data();
    const int nuf=FvmF->nu(), nvf=FvmF->nv();
    #pragma acc update device(up[0:nuf],vp[0:nvf])
}

static void fvmInitialize()
{
    if(!FvmEnabled) return;
    fvm::parse_data_file(FvmDataFile,FvmP);
    FvmG=fvm::parse_grid_file(FvmGridFile);
    FvmF=new fvm::Fields(FvmG.nx,FvmG.ny);
    fvm::initialize_from_geometry(*FvmF,FvmG);
    for(size_t q=0;q<FvmF->F.size();++q) FvmF->F[q]=0.0;      // gas only
    const int nfluid=fvm::count_fluid_cells_host(*FvmF);
    std::vector<int> comp, compCount, compOpen;
    const int ncomp=fvm::build_fluid_components_host(*FvmF,FvmP,comp,compCount,compOpen);
    if(nfluid<=0 || ncomp<=0){ log_printf("FVM: no fluid cells in %s\n",FvmGridFile); exit(EXIT_FAILURE); }
    FvmBackup=new fvm::StateBackup(*FvmF);
    FvmWork=new fvm::Workspace(*FvmF,nfluid,comp,compCount,compOpen);
    FvmF->enter_device(); FvmBackup->enter_device(); FvmWork->enter_device();
    fvm::apply_bc(*FvmF,FvmP);
    fvm::update_mixture_properties(*FvmF,FvmP);
    FvmTime=Time;
    if(FvmEvery<1) FvmEvery=1;
    fvmVelocityToHost();
    log_printf("FVM air: grid %d x %d dx=%e dy=%e [%e,%e]x[%e,%e] fluid cells=%d components=%d rho_g=%e mu_g=%e cfl=%f\n",
               FvmG.nx,FvmG.ny,FvmG.dx,FvmG.dy,FvmG.xmin,FvmG.xmax,FvmG.ymin,FvmG.ymax,nfluid,ncomp,
               FvmP.rho_g,FvmP.mu_g,FvmP.cfl);
    log_printf("FVM air: advanced every %d SPH step(s)\n",FvmEvery);
}

// advance the air from FvmTime to Time
static void fvmAdvance()
{
    if(!FvmEnabled) return;
    fvm::Fields &f=*FvmF;
    const double dx=FvmG.dx, dy=FvmG.dy;
    int sub=0;
    while(FvmTime<Time-1.0e-12*Dt){
        double dt=fvm::compute_dt(f,FvmP,dx,dy);
        if(FvmPrevDt>0.0 && dt>FvmP.dt_growth*FvmPrevDt) dt=FvmP.dt_growth*FvmPrevDt;
        if(FvmTime+dt>Time) dt=Time-FvmTime;
        fvm::copy_state_to_backup(f,*FvmBackup);
        fvm::PressureInfo pinfo; fvm::VOFInfo vinfo;
        int accepted=0;
        for(int retry=0;retry<=FvmP.max_step_retries;++retry){
            if(retry>0) fvm::restore_state_from_backup(f,*FvmBackup);
            if(dt<FvmP.dt_min) break;
            fvm::apply_bc(f,FvmP);
            fvm::update_mixture_properties(f,FvmP);
            fvm::snapshot_density(f,*FvmWork);
            vinfo=fvm::advect_vof_and_build_mass_flux(f,*FvmWork,FvmP,dt,dx,dy,FvmStep);
            if(!fvm::fields_are_finite(f) || vinfo.subcycle_limit_hit){ dt*=0.5; continue; }
            fvm::update_mixture_properties(f,FvmP);
            fvm::compute_surface_geometry(f,*FvmWork,FvmP,dx,dy);
            fvm::compute_momentum_star(f,*FvmWork,FvmP,dt,dx,dy);
            pinfo=fvm::solve_pressure_and_correct(f,*FvmWork,FvmP,dt,dx,dy);
            fvm::apply_bc(f,FvmP);
            if(!pinfo.converged || pinfo.breakdown || !fvm::fields_are_finite(f) ||
               pinfo.dimless_div>FvmP.projection_div_tol){ dt*=0.5; continue; }
            accepted=1; break;
        }
        if(!accepted){
            log_printf("FVM: retries exhausted at t=%e (dt=%e pcg_rel=%e it=%d dt_div=%e)\n",
                       FvmTime,dt,pinfo.rel_res,pinfo.iterations,pinfo.dimless_div);
            exit(EXIT_FAILURE);
        }
        FvmTime+=dt; FvmPrevDt=dt; ++FvmStep; ++sub;
        FvmLastRes=pinfo.rel_res;
    }
    FvmLastSubsteps=sub;
}

// bilinear stencil of one velocity component on the MAC grid:
// comp 0 = u faces (x_i = xmin + i dx, y_j = ymin + (j-1/2) dy), i in [0,nx], j in [0,ny+1]
// comp 1 = v faces (x_i = xmin + (i-1/2) dx, y_j = ymin + j dy), i in [0,nx+1], j in [0,ny]
static void fvmStencil(const int comp, const double x, const double y, int idx[4], double w[4], int open[4])
{
    const fvm::Fields &f=*FvmF;
    const double dx=FvmG.dx, dy=FvmG.dy;
    const int nx=f.nx, ny=f.ny;
    double fi, fj; int imax, jmax;
    if(comp==0){ fi=(x-FvmG.xmin)/dx;     fj=(y-FvmG.ymin)/dy+0.5; imax=nx;   jmax=ny+1; }
    else       { fi=(x-FvmG.xmin)/dx+0.5; fj=(y-FvmG.ymin)/dy;     imax=nx+1; jmax=ny;   }
    int i0=(int)floor(fi), j0=(int)floor(fj);
    if(i0<0) i0=0; if(i0>imax-1) i0=imax-1;
    if(j0<0) j0=0; if(j0>jmax-1) j0=jmax-1;
    double a=fi-i0, b=fj-j0;
    a=fmin(fmax(a,0.0),1.0); b=fmin(fmax(b,0.0),1.0);
    const unsigned char *solid=f.solid.data();
    for(int n=0;n<4;++n){
        const int i=i0+(n&1), j=j0+(n>>1);
        w[n]=((n&1)?a:1.0-a)*((n>>1)?b:1.0-b);
        if(comp==0){ idx[n]=fvm::iu_acc(i,j,ny); open[n]=!(i<=0||i>=nx||j<=0||j>ny||fvm::u_blocked_acc(solid,nx,ny,i,j)); }
        else       { idx[n]=fvm::iv_acc(i,j,ny); open[n]=!(j<=0||j>=ny||i<=0||i>nx||fvm::v_blocked_acc(solid,nx,ny,i,j)); }
    }
}

// FVM cell (1-based, as fvm::Fields::ic) containing (x,y); returns 1 for a gas
// cell, 0 for a solid cell or a point outside the grid
static int fvmCellOf(const double x, const double y, int *i, int *j)
{
    *i=1+(int)floor((x-FvmG.xmin)/FvmG.dx);
    *j=1+(int)floor((y-FvmG.ymin)/FvmG.dy);
    if(*i<1 || *i>FvmF->nx || *j<1 || *j>FvmF->ny) return 0;
    return FvmF->solid[FvmF->ic(*i,*j)]==0;
}

// air velocity at (x,y) from the host face velocities; 0 if FVM is off
static int fvmAirVelocity(const double x, const double y, double *ux, double *uy)
{
    if(!FvmEnabled) return 0;
    int idx[4], open[4]; double w[4];
    const double *u=FvmF->u.data(), *v=FvmF->v.data();
    fvmStencil(0,x,y,idx,w,open); *ux=0.0; for(int n=0;n<4;++n) *ux+=w[n]*u[idx[n]];
    fvmStencil(1,x,y,idx,w,open); *uy=0.0; for(int n=0;n<4;++n) *uy+=w[n]*v[idx[n]];
    return 1;
}

static void fvmWriteVtk()
{
    if(!FvmEnabled || FvmVtkPattern[0]=='\0') return;
    FvmF->update_host_output();
    fvm::write_vtk(*FvmF,FvmG,FvmVtkPattern,FvmFrame++,FvmTime,FvmG.dx,FvmG.dy);
}

static void fvmLogStatus()
{
    if(!FvmEnabled) return;
    const fvm::SpeedInfo s=fvm::max_phase_speeds(*FvmF);
    log_printf("FVM air: t=%e steps=%d last substeps=%d dt=%e pcg_rel=%e Umax_gas=%e\n",
               FvmTime,FvmStep,FvmLastSubsteps,FvmPrevDt,FvmLastRes,s.gas);
}

static void fvmFinalize()
{
    if(!FvmEnabled) return;
    FvmWork->exit_device(); FvmBackup->exit_device(); FvmF->exit_device();
}
