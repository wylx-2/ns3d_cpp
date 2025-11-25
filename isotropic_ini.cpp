// isotropic_turbulence_init.cpp
// Requires: FFTW3, C++17, your project's field_structures.h providing Field3D/GridDesc/LocalDesc/SolverParams
#include <fftw3.h>
#include <complex>
#include <random>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include "field_structures.h" 
#include "ns3d_func.h"     

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper: flatten 3D index (i,j,k) -> linear
inline ptrdiff_t idx3(int i,int j,int k,int NX,int NY,int NZ){
    return (ptrdiff_t)((i*NY + j)*NZ + k);
}

// Energy spectrum shape function (unnormalized). You can change to another model (von Karman, k^-5/3 cut, etc.)
inline double E_spectrum_shape(double k, double k0){
    // k^4 * exp(-2*(k/k0)^2) is commonly used to concentrate energy near k0.
    double r = k / k0;
    return (k>0.0) ? (std::pow(k,4) * std::exp(-2.0 * r*r)) : 0.0;
}

// Main initializer
// - F : Field3D to fill (assumed allocated for local domain L)
// - G : GridDesc (contains dx,dy,dz and global sizes -> domain lengths)
// - P : Solver params (for gamma if computing E)
// - u_rms_target : desired RMS velocity of resulting velocity field
// - k0 : target spectral peak (in physical 1/length units)
// - seed : RNG seed
// - rho0, p0 : uniform density and pressure to fill (if <=0, defaults used)
void initialize_isotropic_turbulence_spectral(Field3D &F, const GridDesc &G, const SolverParams &P, const CartDecomp &C,
                                              double u_rms_target, double k0,
                                              unsigned seed, double rho0, double p0)
{
    const LocalDesc &L = F.L;
    int NX = L.sx; int NY = L.sy; int NZ = L.sz; // local array sizes including ghost
    // We'll perform the spectral construction on the full local array (including ghosts)
    ptrdiff_t N = (ptrdiff_t)NX * NY * NZ;

    // Domain physical lengths (global)
    double Lx = G.global_nx * G.dx;
    double Ly = G.global_ny * G.dy;
    double Lz = G.global_nz * G.dz;

    // Create complex arrays for U_hat components (spectral)
    fftw_complex *Ux_hat = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
    fftw_complex *Uy_hat = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
    fftw_complex *Uz_hat = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
    if(!Ux_hat || !Uy_hat || !Uz_hat){
        std::cerr<<"FFTW allocation failed\n"; return;
    }

    // Initialize RNG
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist_phase(0.0, 2.0*M_PI);
    std::normal_distribution<double> dist_norm(0.0,1.0); // used if you want Gaussian components

    // Fill with random complex values with amplitude from spectrum shape,
    // and enforce Hermitian symmetry so that inverse transform is real.
    auto set_complex = [&](ptrdiff_t pos, std::complex<double> v){
        Ux_hat[pos][0] = v.real(); Ux_hat[pos][1] = v.imag();
        Uy_hat[pos][0] = 0.0; Uy_hat[pos][1] = 0.0;
        Uz_hat[pos][0] = 0.0; Uz_hat[pos][1] = 0.0;
    };

    // We'll generate random vector at each k and then project to divergence-free
    // Loop over local indices (i,j,k) which map to wave numbers.
    for(int i=0;i<NX;++i){
        int kx_ind = (i <= NX/2) ? i : i - NX;
        double kx = (2.0*M_PI/Lx) * kx_ind;
        for(int j=0;j<NY;++j){
            int ky_ind = (j <= NY/2) ? j : j - NY;
            double ky = (2.0*M_PI/Ly) * ky_ind;
            for(int k=0;k<NZ;++k){
                int kz_ind = (k <= NZ/2) ? k : k - NZ;
                double kz = (2.0*M_PI/Lz) * kz_ind;

                ptrdiff_t id = idx3(i,j,k,NX,NY,NZ);
                double kmag = std::sqrt(kx*kx + ky*ky + kz*kz);

                // zero DC mode
                if (kx==0.0 && ky==0.0 && kz==0.0) {
                    Ux_hat[id][0]=Ux_hat[id][1]=0.0;
                    Uy_hat[id][0]=Uy_hat[id][1]=0.0;
                    Uz_hat[id][0]=Uz_hat[id][1]=0.0;
                    continue;
                }

                // amplitude from spectrum shape (unnormalized)
                double amp = std::sqrt(E_spectrum_shape(kmag, k0) + 1e-300);

                // random complex vector components (random phase)
                double phi1 = dist_phase(rng);
                double phi2 = dist_phase(rng);
                double phi3 = dist_phase(rng);
                std::complex<double> a1 = amp * std::polar(1.0, phi1);
                std::complex<double> a2 = amp * std::polar(1.0, phi2);
                std::complex<double> a3 = amp * std::polar(1.0, phi3);

                // build vector in spectral space
                std::complex<double> ux = a1;
                std::complex<double> uy = a2;
                std::complex<double> uz = a3;

                // project to divergence-free: u_hat <- u_hat - k (k·u_hat)/k^2
                std::complex<double> kdotu = kx*ux + ky*uy + kz*uz;
                double k2 = kx*kx + ky*ky + kz*kz;
                if (k2 > 0.0) {
                    ux = ux - (kx * kdotu / k2);
                    uy = uy - (ky * kdotu / k2);
                    uz = uz - (kz * kdotu / k2);
                }

                // store (as real-imag pairs)
                Ux_hat[id][0] = ux.real(); Ux_hat[id][1] = ux.imag();
                Uy_hat[id][0] = uy.real(); Uy_hat[id][1] = uy.imag();
                Uz_hat[id][0] = uz.real(); Uz_hat[id][1] = uz.imag();
            }
        }
    }

    // Enforce Hermitian symmetry: U(-k) = conj(U(k)) so inverse is real.
    // For all indices, find conjugate index and set accordingly.
    // Loop only half of spectrum to set pairs.
    for(int i=0;i<NX;++i){
        int ic = (NX - i) % NX;
        for(int j=0;j<NY;++j){
            int jc = (NY - j) % NY;
            for(int k=0;k<NZ;++k){
                int kc = (NZ - k) % NZ;
                ptrdiff_t id = idx3(i,j,k,NX,NY,NZ);
                ptrdiff_t idc = idx3(ic,jc,kc,NX,NY,NZ);
                // set idc = conj(id)
                Ux_hat[idc][0] =  Ux_hat[id][0];
                Ux_hat[idc][1] = -Ux_hat[id][1];
                Uy_hat[idc][0] =  Uy_hat[id][0];
                Uy_hat[idc][1] = -Uy_hat[id][1];
                Uz_hat[idc][0] =  Uz_hat[id][0];
                Uz_hat[idc][1] = -Uz_hat[id][1];
            }
        }
    }

    // Prepare FFTW plans for inverse transform (complex -> complex)
    fftw_plan plan_ix = fftw_plan_dft_3d(NX,NY,NZ, Ux_hat, Ux_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan plan_iy = fftw_plan_dft_3d(NX,NY,NZ, Uy_hat, Uy_hat, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan plan_iz = fftw_plan_dft_3d(NX,NY,NZ, Uz_hat, Uz_hat, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan_ix);
    fftw_execute(plan_iy);
    fftw_execute(plan_iz);

    // normalization factor for inverse FFTW (FFTW does not normalize)
    double scale = 1.0 / (double)(NX * NY * NZ);

    // Copy to Field3D arrays (real velocities), compute mean and RMS
    double local_sum_u2 = 0.0;
    size_t count = 0;
    for(int i=0;i<NX;++i){
        for(int j=0;j<NY;++j){
            for(int k=0;k<NZ;++k){
                ptrdiff_t id = idx3(i,j,k,NX,NY,NZ);
                int gid = F.I(i,j,k); // linear index into Field3D arrays

                double ux = scale * Ux_hat[id][0];
                double uy = scale * Uy_hat[id][0];
                double uz = scale * Uz_hat[id][0];

                // Because we enforced Hermitian symmetry, imaginary parts should be ~0 (numerical noise)
                // but we'll ignore imag parts.

                F.u[gid] = ux;
                F.v[gid] = uy;
                F.w[gid] = uz;

                local_sum_u2 += ux*ux + uy*uy + uz*uz;
                ++count;
            }
        }
    }

    // compute global u_rms and scale to target
    double global_sum_u2 = 0.0;
    MPI_Allreduce(&local_sum_u2, &global_sum_u2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double Ntotal = (double)(G.global_nx * G.global_ny * G.global_nz);
    double urms_current = std::sqrt(global_sum_u2 / Ntotal);

    if (urms_current <= 0.0) urms_current = 1e-16;
    double scale_u = u_rms_target / urms_current;

    // apply scaling and set density/pressure/energy
    for(int k=0;k<NZ;++k)
    for(int j=0;j<NY;++j)
    for(int i=0;i<NX;++i){
        int gid = F.I(i,j,k);
        F.u[gid] *= scale_u;
        F.v[gid] *= scale_u;
        F.w[gid] *= scale_u;

        F.rho[gid] = rho0;
        F.p[gid]   = p0;
        // total energy per volume: E = p/(gamma-1) + 0.5*rho*(u^2+v^2+w^2)
        double ke = 0.5 * rho0 * (F.u[gid]*F.u[gid] + F.v[gid]*F.v[gid] + F.w[gid]*F.w[gid]);
        F.E[gid] = p0 / (P.gamma - 1.0) + ke;
    }

    // cleanup FFTW
    fftw_destroy_plan(plan_ix);
    fftw_destroy_plan(plan_iy);
    fftw_destroy_plan(plan_iz);
    fftw_free(Ux_hat); fftw_free(Uy_hat); fftw_free(Uz_hat);

    // finally, remove mean velocity (ensure zero mean)
    double local_ux_sum=0.0, local_uy_sum=0.0, local_uz_sum=0.0;
    for (int k=0;k<NZ;++k) for (int j=0;j<NY;++j) for (int i=0;i<NX;++i) {
        int gid = F.I(i,j,k);
        local_ux_sum += F.u[gid];
        local_uy_sum += F.v[gid];
        local_uz_sum += F.w[gid];
    }
    double global_ux_sum=0.0, global_uy_sum=0.0, global_uz_sum=0.0;
    MPI_Allreduce(&local_ux_sum, &global_ux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_uy_sum, &global_uy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_uz_sum, &global_uz_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double mean_ux = global_ux_sum / Ntotal;
    double mean_uy = global_uy_sum / Ntotal;
    double mean_uz = global_uz_sum / Ntotal;

    for (int k=0;k<NZ;++k) for (int j=0;j<NY;++j) for (int i=0;i<NX;++i) {
        int gid = F.I(i,j,k);
        F.u[gid] -= mean_ux;
        F.v[gid] -= mean_uy;
        F.w[gid] -= mean_uz;

        // recompute E if you want strictly consistent (optional)
        double ke = 0.5 * rho0 * (F.u[gid]*F.u[gid] + F.v[gid]*F.v[gid] + F.w[gid]*F.w[gid]);
        F.E[gid] = p0 / (P.gamma - 1.0) + ke;
    }

    // optionally: exchange halos if needed, e.g. call your exchange_halos_conserved(F, C, L, reqs)
    // to sync ghost zones across ranks.

    if (C.rank==0) {
        std::cout << "[init_turb] u_rms target=" << u_rms_target << " achieved (before mean removal) = " << urms_current << "\n";
    }
}
