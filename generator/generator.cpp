// ============================================================================
// geometry.cpp -- multi-geometry particle generator
//
// Supported blocks (applied in file order; later blocks override earlier ones):
//   StartCuboid  ... EndCuboid
//   StartCyboid  ... EndCyboid
//   StartCyboid2 ... EndCyboid2   (alias of Cyboid)
//   StartEllipse ... EndEllipse    (alias of Cyboid)
//   StartAnnulus ... EndAnnulus    (alias of Cyboid; use Ratio > 0)
//   StartCircle  ... EndCircle
//   StartSphere  ... EndSphere
//
// All particles lie on ONE global lattice anchored at LowerDomain with
// ParticleDistance. Overlapping geometries therefore do not create duplicate
// particles. A cell is assigned to the LAST shape in the .boid file that
// contains its center.
//
// Cyboid / Ellipse / Annulus:
//   Lower/Upper define the outer bounding box.
//   ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
//   Ratio is the inner/outer radius ratio. Ratio=0 gives a filled ellipse;
//   Ratio=0.5 gives an annulus with half-size inner radii.
//   z is restricted by Lower[2] <= z <= Upper[2].
//
// Spacing inside each shape is accepted for compatibility but must equal
// ParticleDistance.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct Vec3 { double x=0.0, y=0.0, z=0.0; };

enum class ShapeKind { Cuboid, Cyboid, Circle, Sphere };

struct Shape {
    ShapeKind kind = ShapeKind::Cuboid;
    int type = 0;
    double spacing = -1.0;
    Vec3 lower{}, upper{}, center{}, velocity{};
    double radius = -1.0;
    double ratio = 0.0;
    double lower_z = std::numeric_limits<double>::quiet_NaN();
    double upper_z = std::numeric_limits<double>::quiet_NaN();
    bool have_type=false, have_lower=false, have_upper=false;
    bool have_center=false, have_radius=false, have_velocity=false, have_spacing=false;
};

struct Particle { int type=0; Vec3 pos{}, vel{}; };

static double ParticleDistance = -1.0;
static Vec3 LowerDomain{}, UpperDomain{};
static bool have_pd=false, have_lower_domain=false, have_upper_domain=false;
static std::vector<Shape> Shapes;

static std::string trim(const std::string& s) {
    const auto a=s.find_first_not_of(" \t\r\n");
    if (a==std::string::npos) return "";
    const auto b=s.find_last_not_of(" \t\r\n");
    return s.substr(a,b-a+1);
}

static std::string clean_line(std::string s) {
    const auto hash=s.find('#');
    if (hash!=std::string::npos) s.erase(hash);
    return trim(s);
}

static bool nearly_equal(double a,double b,double rel=1e-10) {
    const double scale=std::max({1.0,std::fabs(a),std::fabs(b)});
    return std::fabs(a-b)<=rel*scale;
}

static bool point_in_shape(const Shape& s, const Vec3& p) {
    constexpr double eps=1e-12;
    if (s.kind==ShapeKind::Cuboid) {
        return p.x>=s.lower.x-eps && p.x<=s.upper.x+eps &&
               p.y>=s.lower.y-eps && p.y<=s.upper.y+eps &&
               p.z>=s.lower.z-eps && p.z<=s.upper.z+eps;
    }
    if (s.kind==ShapeKind::Cyboid) {
        const double cx=0.5*(s.lower.x+s.upper.x);
        const double cy=0.5*(s.lower.y+s.upper.y);
        const double rx=0.5*(s.upper.x-s.lower.x);
        const double ry=0.5*(s.upper.y-s.lower.y);
        if (!(rx>0.0 && ry>0.0)) return false;
        if (p.z<s.lower.z-eps || p.z>s.upper.z+eps) return false;
        const double X=(p.x-cx)/rx;
        const double Y=(p.y-cy)/ry;
        const double r2=X*X+Y*Y;
        return r2<=1.0+eps && r2>=s.ratio*s.ratio-eps;
    }
    if (s.kind==ShapeKind::Circle) {
        const double zlo=std::isnan(s.lower_z)?LowerDomain.z:s.lower_z;
        const double zhi=std::isnan(s.upper_z)?UpperDomain.z:s.upper_z;
        if (p.z<zlo-eps || p.z>zhi+eps) return false;
        const double dx=p.x-s.center.x, dy=p.y-s.center.y;
        return dx*dx+dy*dy<=s.radius*s.radius+eps;
    }
    const double dx=p.x-s.center.x, dy=p.y-s.center.y, dz=p.z-s.center.z;
    return dx*dx+dy*dy+dz*dz<=s.radius*s.radius+eps;
}

static void validate_shape(const Shape& s, const std::string& name) {
    if (!s.have_type) { std::cerr<<"error: "<<name<<" has no Type\n"; std::exit(1); }
    if (!s.have_velocity) { std::cerr<<"error: "<<name<<" has no Velocity\n"; std::exit(1); }
    if (s.have_spacing && !nearly_equal(s.spacing,ParticleDistance,1e-8)) {
        std::cerr<<"error: "<<name<<" Spacing="<<s.spacing
                 <<" differs from ParticleDistance="<<ParticleDistance<<"\n";
        std::exit(1);
    }
    if (s.kind==ShapeKind::Cuboid || s.kind==ShapeKind::Cyboid) {
        if (!s.have_lower || !s.have_upper) {
            std::cerr<<"error: "<<name<<" requires Lower and Upper\n"; std::exit(1);
        }
        if (!(s.upper.x>s.lower.x && s.upper.y>s.lower.y && s.upper.z>s.lower.z)) {
            std::cerr<<"error: invalid Lower/Upper in "<<name<<"\n"; std::exit(1);
        }
    }
    if (s.kind==ShapeKind::Cyboid && !(s.ratio>=0.0 && s.ratio<1.0)) {
        std::cerr<<"error: Ratio must satisfy 0 <= Ratio < 1 in "<<name<<"\n"; std::exit(1);
    }
    if (s.kind==ShapeKind::Circle || s.kind==ShapeKind::Sphere) {
        if (!s.have_center || !s.have_radius || !(s.radius>0.0)) {
            std::cerr<<"error: "<<name<<" requires Center and Radius > 0\n"; std::exit(1);
        }
    }
}

static Shape read_shape(std::istream& in, ShapeKind kind,
                        const std::string& endcommand, const std::string& name) {
    Shape s; s.kind=kind;
    std::string line;
    while (std::getline(in,line)) {
        line=clean_line(line);
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string key; ss>>key;
        if (key==endcommand) { validate_shape(s,name); return s; }
        if (key=="Spacing") { if(!(ss>>s.spacing)) goto bad; s.have_spacing=true; }
        else if (key=="Type") { if(!(ss>>s.type)) goto bad; s.have_type=true; }
        else if (key=="Lower") { if(!(ss>>s.lower.x>>s.lower.y>>s.lower.z)) goto bad; s.have_lower=true; }
        else if (key=="Upper") { if(!(ss>>s.upper.x>>s.upper.y>>s.upper.z)) goto bad; s.have_upper=true; }
        else if (key=="Center") { if(!(ss>>s.center.x>>s.center.y>>s.center.z)) goto bad; s.have_center=true; }
        else if (key=="Radius") { if(!(ss>>s.radius)) goto bad; s.have_radius=true; }
        else if (key=="Ratio") { if(!(ss>>s.ratio)) goto bad; }
        else if (key=="LowerZ") { if(!(ss>>s.lower_z)) goto bad; }
        else if (key=="UpperZ") { if(!(ss>>s.upper_z)) goto bad; }
        else if (key=="Velocity") { if(!(ss>>s.velocity.x>>s.velocity.y>>s.velocity.z)) goto bad; s.have_velocity=true; }
        else { std::cerr<<"error: unknown keyword '"<<key<<"' in "<<name<<"\n"; std::exit(1); }
        continue;
    bad:
        std::cerr<<"error: invalid line in "<<name<<": "<<line<<"\n";
        std::exit(1);
    }
    std::cerr<<"error: missing "<<endcommand<<" for "<<name<<"\n";
    std::exit(1);
}

static void readfile(const char* fname) {
    std::ifstream in(fname);
    if (!in) { std::cerr<<"error: cannot read file "<<fname<<"\n"; std::exit(1); }
    std::string line;
    while (std::getline(in,line)) {
        line=clean_line(line);
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string key; ss>>key;
        if (key=="ParticleDistance") {
            if (!(ss>>ParticleDistance) || !(ParticleDistance>0.0)) { std::cerr<<"error: invalid ParticleDistance\n"; std::exit(1); }
            have_pd=true;
        } else if (key=="LowerDomain") {
            if (!(ss>>LowerDomain.x>>LowerDomain.y>>LowerDomain.z)) { std::cerr<<"error: invalid LowerDomain\n"; std::exit(1); }
            have_lower_domain=true;
        } else if (key=="UpperDomain") {
            if (!(ss>>UpperDomain.x>>UpperDomain.y>>UpperDomain.z)) { std::cerr<<"error: invalid UpperDomain\n"; std::exit(1); }
            have_upper_domain=true;
        } else if (key=="StartCuboid") Shapes.push_back(read_shape(in,ShapeKind::Cuboid,"EndCuboid","Cuboid"));
        else if (key=="StartCyboid") Shapes.push_back(read_shape(in,ShapeKind::Cyboid,"EndCyboid","Cyboid"));
        else if (key=="StartCyboid2") Shapes.push_back(read_shape(in,ShapeKind::Cyboid,"EndCyboid2","Cyboid2"));
        else if (key=="StartEllipse") Shapes.push_back(read_shape(in,ShapeKind::Cyboid,"EndEllipse","Ellipse"));
        else if (key=="StartAnnulus") Shapes.push_back(read_shape(in,ShapeKind::Cyboid,"EndAnnulus","Annulus"));
        else if (key=="StartCircle") Shapes.push_back(read_shape(in,ShapeKind::Circle,"EndCircle","Circle"));
        else if (key=="StartSphere") Shapes.push_back(read_shape(in,ShapeKind::Sphere,"EndSphere","Sphere"));
        else std::cerr<<"warning: unknown top-level keyword '"<<key<<"' ignored\n";
    }
    if (!have_pd || !have_lower_domain || !have_upper_domain) {
        std::cerr<<"error: ParticleDistance, LowerDomain and UpperDomain are required\n"; std::exit(1);
    }
    if (!(UpperDomain.x>LowerDomain.x && UpperDomain.y>LowerDomain.y && UpperDomain.z>LowerDomain.z)) {
        std::cerr<<"error: invalid domain bounds\n"; std::exit(1);
    }
    if (Shapes.empty()) { std::cerr<<"error: no geometry blocks found\n"; std::exit(1); }
}

static std::vector<Particle> genparticle() {
    const int nx=std::max(1,(int)std::floor((UpperDomain.x-LowerDomain.x)/ParticleDistance+1e-12));
    const int ny=std::max(1,(int)std::floor((UpperDomain.y-LowerDomain.y)/ParticleDistance+1e-12));
    const int nz=std::max(1,(int)std::floor((UpperDomain.z-LowerDomain.z)/ParticleDistance+1e-12));
    std::vector<Particle> particles;
    particles.reserve((std::size_t)nx*ny*nz);
    std::vector<std::size_t> final_count(Shapes.size(),0);

    for (int i=0;i<nx;++i) {
        const double x=LowerDomain.x+(i+0.5)*ParticleDistance;
        for (int j=0;j<ny;++j) {
            const double y=LowerDomain.y+(j+0.5)*ParticleDistance;
            for (int k=0;k<nz;++k) {
                const double z=LowerDomain.z+(k+0.5)*ParticleDistance;
                Vec3 p{x,y,z};
                int winner=-1;
                for (std::size_t s=0;s<Shapes.size();++s)
                    if (point_in_shape(Shapes[s],p)) winner=(int)s;
                if (winner>=0) {
                    Particle q;
                    q.type=Shapes[(std::size_t)winner].type;
                    q.pos=p;
                    q.vel=Shapes[(std::size_t)winner].velocity;
                    particles.push_back(q);
                    final_count[(std::size_t)winner]++;
                }
            }
        }
    }

    std::cerr<<particles.size()<<" particles were generated\n";
    for (std::size_t s=0;s<Shapes.size();++s)
        std::cerr<<"  shape["<<s<<"] final particles="<<final_count[s]
                 <<" type="<<Shapes[s].type<<"\n";
    return particles;
}

static void writefile(const char* fname,const std::vector<Particle>& particles) {
    FILE* fp=std::fopen(fname,"w");
    if (!fp) { std::fprintf(stderr,"error: cannot write %s\n",fname); std::exit(1); }
    std::fprintf(fp,"%.17g\n",0.0);
    std::fprintf(fp,"%zu %.17e  %.17e %.17e  %.17e %.17e  %.17e %.17e\n",
                 particles.size(),ParticleDistance,
                 LowerDomain.x,UpperDomain.x,
                 LowerDomain.y,UpperDomain.y,
                 LowerDomain.z,UpperDomain.z);
    for (const auto& p:particles)
        std::fprintf(fp,"%d   %.17e %.17e %.17e    %.17e %.17e %.17e\n",
                     p.type,p.pos.x,p.pos.y,p.pos.z,p.vel.x,p.vel.y,p.vel.z);
    std::fclose(fp);
}

int main(int argc,char* argv[]) {
    std::string base=(argc>1)?argv[1]:"sample";
    const std::string boidname=base+".boid";
    const std::string gridname=base+".grid";
    readfile(boidname.c_str());
    const auto particles=genparticle();
    writefile(gridname.c_str(),particles);
    return 0;
}
