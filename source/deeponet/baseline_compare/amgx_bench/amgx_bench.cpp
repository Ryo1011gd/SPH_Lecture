// GPU wall-clock benchmark of AmgX on the solver's own linear systems
// (ppe_*.bin = PPEDON01, vis_*.bin = VISDON01 snapshots).
//
//   amgx_bench CONFIG_FILE_OR_STRING snapshot.bin [snapshot.bin ...]
//
// Per snapshot it reports: upload (H2D of CSR, rhs, x0), setup, solve [ms],
// iterations and the final relative residual measured on the host, with the
// same stopping test as the SPH solver:
//   PPE : |b - A x|_2 <= tol |b|_2
//   VIS : sum_i |r_i|^2 / M_i <= tol^2 sum_i |b_i|^2/M_i + abstol^2 sum_i M_i
// For VIS the system is symmetrically scaled by M^-1/2 so that this weighted
// norm is the plain L2 norm AmgX monitors (ABSOLUTE convergence).
#include <amgx_c.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Sys {
    int n = 0;              // rows
    std::vector<int> rowptr, col;
    std::vector<double> val, b, x0;
    double tol = 1e-8, target = 0.0;   // ABSOLUTE L2 target for the scaled system
    bool vis = false, pure_neumann = false;
};

static std::vector<char> slurp(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> b(n);
    if (fread(b.data(), 1, n, f) != (size_t)n) { perror(p); exit(1); }
    fclose(f);
    return b;
}

template <class T> static void take(const std::vector<char> &buf, size_t &off, T *dst, size_t cnt)
{
    memcpy(dst, buf.data() + off, cnt * sizeof(T)); off += cnt * sizeof(T);
}

static Sys readSnapshot(const char *path)
{
    std::vector<char> buf = slurp(path);
    Sys s; size_t off = 8;
    int h[8]; take(buf, off, h, 8);
    long long nnz; take(buf, off, &nnz, 1);
    const int sdim = h[0], nloc = h[1], nglob = h[2], np = h[3];
    if (!memcmp(buf.data(), "PPEDON01", 8)) {
        s.pure_neumann = h[4] != 0;
        double sc[4]; take(buf, off, sc, 4); s.tol = sc[3];
        off += sizeof(double) * (nglob + 2 * (size_t)np * sdim + (size_t)np * nloc + np);   // glob pos xhat loc diag
        s.n = np;
        s.b.resize(np); take(buf, off, s.b.data(), np);
        off += sizeof(double) * np;                                    // p (converged)
        s.x0.resize(np); take(buf, off, s.x0.data(), np);
        std::vector<long long> rp(np + 1); take(buf, off, rp.data(), np + 1);
        s.rowptr.assign(rp.begin(), rp.end());
        s.col.resize(nnz); take(buf, off, s.col.data(), nnz);
        s.val.resize(nnz); take(buf, off, s.val.data(), nnz);
        double bb = 0; for (double v : s.b) bb += v * v;
        s.target = s.tol * std::sqrt(bb);
    } else if (!memcmp(buf.data(), "VISDON01", 8)) {
        s.vis = true;
        const int nc = h[4], m = np * nc;
        double sc[5]; take(buf, off, sc, 5); s.tol = sc[3]; const double abstol = sc[4];
        off += sizeof(double) * (nglob + 2 * (size_t)np * sdim + (size_t)np * nloc);       // glob pos xhat loc
        std::vector<double> mass(np); take(buf, off, mass.data(), np);
        off += sizeof(double) * m;                                     // diag
        s.n = m;
        s.b.resize(m); take(buf, off, s.b.data(), m);
        off += sizeof(double) * m;                                     // u (converged)
        s.x0.resize(m); take(buf, off, s.x0.data(), m);
        std::vector<long long> rp(m + 1); take(buf, off, rp.data(), m + 1);
        s.rowptr.assign(rp.begin(), rp.end());
        s.col.resize(nnz); take(buf, off, s.col.data(), nnz);
        s.val.resize(nnz); take(buf, off, s.val.data(), nnz);
        // weighted norm target, then symmetric scaling by M^-1/2 (row a*nc+d belongs to particle a)
        double bw = 0, msum = 0;
        for (int i = 0; i < m; ++i) bw += s.b[i] * s.b[i] / mass[i / nc];
        for (double v : mass) msum += v;
        s.target = std::sqrt(s.tol * s.tol * bw + abstol * abstol * msum);
        std::vector<double> d(m);
        for (int i = 0; i < m; ++i) d[i] = 1.0 / std::sqrt(mass[i / nc]);
        for (int i = 0; i < m; ++i) {
            for (int k = s.rowptr[i]; k < s.rowptr[i + 1]; ++k) s.val[k] *= d[i] * d[s.col[k]];
            s.b[i] *= d[i];
            s.x0[i] /= d[i];
        }
    } else {
        fprintf(stderr, "%s: unknown snapshot\n", path); exit(1);
    }
    if (off != buf.size()) { fprintf(stderr, "%s: %zu trailing bytes\n", path, buf.size() - off); exit(1); }
    return s;
}

static double residual(const Sys &s, const std::vector<double> &x)
{
    std::vector<double> y(s.n, 0.0);
    for (int i = 0; i < s.n; ++i) {
        double a = 0; for (int k = s.rowptr[i]; k < s.rowptr[i + 1]; ++k) a += s.val[k] * x[s.col[k]];
        y[i] = a;
    }
    if (s.pure_neumann) { double m = 0; for (double v : y) m += v; m /= s.n; for (double &v : y) v -= m; }
    double rr = 0; for (int i = 0; i < s.n; ++i) { const double r = s.b[i] - y[i]; rr += r * r; }
    return std::sqrt(rr);
}

static double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

#define CK(x) do { AMGX_RC rc_ = (x); if (rc_ != AMGX_RC_OK) { char m_[512]; AMGX_get_error_string(rc_, m_, 512); fprintf(stderr, "%s: %s\n", #x, m_); exit(1); } } while (0)

static void quiet(const char *, int) {}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s CONFIG snap.bin...\n", argv[0]); return 1; }
    CK(AMGX_initialize());
    AMGX_register_print_callback(quiet);
    std::string cfgText = argv[1];
    FILE *cf = fopen(argv[1], "r");
    if (cf) { fclose(cf); std::vector<char> c = slurp(argv[1]); cfgText.assign(c.begin(), c.end()); }

    printf("file n nnz upload_ms setup_ms solve_ms iters rel_to_target status\n");
    for (int a = 2; a < argc; ++a) {
        Sys s = readSnapshot(argv[a]);
        // the convergence target depends on the snapshot -> append it to the config
        char extra[256];
        snprintf(extra, sizeof extra, ", s:convergence=ABSOLUTE, s:norm=L2, s:tolerance=%.17g", s.target);
        AMGX_config_handle cfg; CK(AMGX_config_create(&cfg, (cfgText + extra).c_str()));
        AMGX_resources_handle rsrc; CK(AMGX_resources_create_simple(&rsrc, cfg));
        AMGX_matrix_handle A; AMGX_vector_handle B, X; AMGX_solver_handle S;
        CK(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
        CK(AMGX_vector_create(&B, rsrc, AMGX_mode_dDDI));
        CK(AMGX_vector_create(&X, rsrc, AMGX_mode_dDDI));
        CK(AMGX_solver_create(&S, rsrc, AMGX_mode_dDDI, cfg));

        // the matrix already lives on the GPU in the SPH code: stage it on the device first,
        // so that "upload" is a device-to-device copy into AmgX's own storage
        int *drp, *dcol; double *dval, *db, *dx;
        const int nnz = (int)s.col.size();
        cudaMalloc(&drp, sizeof(int) * (s.n + 1)); cudaMalloc(&dcol, sizeof(int) * nnz);
        cudaMalloc(&dval, sizeof(double) * nnz); cudaMalloc(&db, sizeof(double) * s.n); cudaMalloc(&dx, sizeof(double) * s.n);
        cudaMemcpy(drp, s.rowptr.data(), sizeof(int) * (s.n + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(dcol, s.col.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(dval, s.val.data(), sizeof(double) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(db, s.b.data(), sizeof(double) * s.n, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, s.x0.data(), sizeof(double) * s.n, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        // BLOCK=2 (VIS only): interleaved (a*2+d) scalar CSR -> 2x2 BSR, uploaded instead
        const int bs = (s.vis && getenv("BLOCK")) ? atoi(getenv("BLOCK")) : 1;
        int nbr = s.n, nnzb = nnz;
        if (bs == 2) {
            nbr = s.n / 2;
            std::vector<int> brp(nbr + 1, 0), bcol; std::vector<double> bval;
            for (int I = 0; I < nbr; ++I) {
                std::vector<int> cols;
                for (int d = 0; d < 2; ++d)
                    for (int k = s.rowptr[2 * I + d]; k < s.rowptr[2 * I + d + 1]; ++k) cols.push_back(s.col[k] / 2);
                std::sort(cols.begin(), cols.end()); cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
                size_t base = bcol.size();
                for (int J : cols) { bcol.push_back(J); for (int q = 0; q < 4; ++q) bval.push_back(0.0); }
                for (int d = 0; d < 2; ++d)
                    for (int k = s.rowptr[2 * I + d]; k < s.rowptr[2 * I + d + 1]; ++k) {
                        const int J = s.col[k] / 2, q = s.col[k] % 2;
                        const size_t pos = base + (std::lower_bound(cols.begin(), cols.end(), J) - cols.begin());
                        bval[pos * 4 + d * 2 + q] += s.val[k];
                    }
                brp[I + 1] = (int)bcol.size();
            }
            nnzb = (int)bcol.size();
            cudaFree(drp); cudaFree(dcol); cudaFree(dval);
            cudaMalloc(&drp, sizeof(int) * (nbr + 1)); cudaMalloc(&dcol, sizeof(int) * nnzb); cudaMalloc(&dval, sizeof(double) * nnzb * 4);
            cudaMemcpy(drp, brp.data(), sizeof(int) * (nbr + 1), cudaMemcpyHostToDevice);
            cudaMemcpy(dcol, bcol.data(), sizeof(int) * nnzb, cudaMemcpyHostToDevice);
            cudaMemcpy(dval, bval.data(), sizeof(double) * nnzb * 4, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
        double t0 = now();
        CK(AMGX_matrix_upload_all(A, nbr, nnzb, bs, bs, drp, dcol, dval, nullptr));
        CK(AMGX_vector_upload(B, nbr, bs, db));
        CK(AMGX_vector_upload(X, nbr, bs, dx));
        cudaDeviceSynchronize();
        double t1 = now();
        CK(AMGX_solver_setup(S, A));
        cudaDeviceSynchronize();
        double t2 = now();
        CK(AMGX_solver_solve(S, B, X));
        cudaDeviceSynchronize();
        double t3 = now();
        int it = 0; CK(AMGX_solver_get_iterations_number(S, &it));
        AMGX_SOLVE_STATUS st; CK(AMGX_solver_get_status(S, &st));
        std::vector<double> x(s.n); CK(AMGX_vector_download(X, x.data()));
        const double rel = residual(s, x) / s.target;
        const char *name = strrchr(argv[a], '/'); name = name ? name + 1 : argv[a];
        printf("%s %d %d %.3f %.3f %.3f %d %.3g %s\n", name, s.n, nnz, t1 - t0, t2 - t1, t3 - t2, it, rel,
               st == AMGX_SOLVE_SUCCESS ? "ok" : (st == AMGX_SOLVE_DIVERGED ? "diverged" : "failed"));
        fflush(stdout);
        AMGX_solver_destroy(S); AMGX_vector_destroy(X); AMGX_vector_destroy(B); AMGX_matrix_destroy(A);
        AMGX_resources_destroy(rsrc); AMGX_config_destroy(cfg);
        cudaFree(drp); cudaFree(dcol); cudaFree(dval); cudaFree(db); cudaFree(dx);
    }
    AMGX_finalize();
    return 0;
}
