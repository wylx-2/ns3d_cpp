#include "field_structures.h"
#include "ns3d_func.h"
#include <cmath>
#include <mpi.h>

//==================================================================
// 三阶 Runge-Kutta 时间推进主循环模块
//==================================================================

// RHS 计算占位符函数（用户需在此定义具体的通量差分或高阶算子）
void compute_rhs(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P, HaloRequests &out_reqs)
{
    LocalDesc &L = F.L;
    const double idx = 1.0 / G.dx;
    const double idy = 1.0 / G.dy;
    const double idz = 1.0 / G.dz;

    // 清空 RHS
    std::fill(F.rhs_rho.begin(), F.rhs_rho.end(), 0.0);
    std::fill(F.rhs_rhou.begin(), F.rhs_rhou.end(), 0.0);
    std::fill(F.rhs_rhov.begin(), F.rhs_rhov.end(), 0.0);
    std::fill(F.rhs_rhow.begin(), F.rhs_rhow.end(), 0.0);
    std::fill(F.rhs_E.begin(), F.rhs_E.end(), 0.0);

    // 计算无粘通量
    compute_flux(F, P);
    // 重构无粘通量
    compute_flux_face(F, P);

    // 计算空间导数
    compute_gradients(F, G, P);
    // 同步不同进程的ghost区域空间导数
    exchange_halos_gradients(F, C, L, out_reqs);
    // 计算粘性通量
    compute_viscous_flux(F, P);
    // 重构粘性通量
    compute_vis_flux_face(F, P);

    // 使用面通量计算 RHS（有限体积散度），只对物理单元计算
    // dQ/dt = - [ (F_x(i+1/2)-F_x(i-1/2))/dx + (F_y(j+1/2)-F_y(j-1/2))/dy + (F_z(k+1/2)-F_z(k-1/2))/dz ]
    // loop over physical cells only
    for (int k = L.ngz; k < L.ngz + L.nz; ++k){
    for (int j = L.ngy; j < L.ngy + L.ny; ++j){
    for (int i = L.ngx; i < L.ngx + L.nx; ++i){
        // mass
        double fx_r = F.FX_mass(i, j, k);
        double fx_l = F.FX_mass(i - 1, j, k);
        double fy_t = F.FY_mass(i, j, k);
        double fy_b = F.FY_mass(i, j - 1, k);
        double fz_f = F.FZ_mass(i, j, k);
        double fz_b = F.FZ_mass(i, j, k - 1);
        F.RHS_Rho(i, j, k) = -((fx_r - fx_l) * idx + (fy_t - fy_b) * idy + (fz_f - fz_b) * idz);

        // momentum x
        fx_r = F.FX_momx(i, j, k);
        fx_l = F.FX_momx(i - 1, j, k);
        fy_t = F.FY_momx(i, j, k);
        fy_b = F.FY_momx(i, j - 1, k);
        fz_f = F.FZ_momx(i, j, k);
        fz_b = F.FZ_momx(i, j, k - 1);
        F.RHS_RhoU(i, j, k) = -((fx_r - fx_l) * idx + (fy_t - fy_b) * idy + (fz_f - fz_b) * idz);

        // momentum y
        fx_r = F.FX_momy(i, j, k);
        fx_l = F.FX_momy(i - 1, j, k);
        fy_t = F.FY_momy(i, j, k);
        fy_b = F.FY_momy(i, j - 1, k);
        fz_f = F.FZ_momy(i, j, k);
        fz_b = F.FZ_momy(i, j, k - 1);
        F.RHS_RhoV(i, j, k) = -((fx_r - fx_l) * idx + (fy_t - fy_b) * idy + (fz_f - fz_b) * idz);

        // momentum z
        fx_r = F.FX_momz(i, j, k);
        fx_l = F.FX_momz(i - 1, j, k);
        fy_t = F.FY_momz(i, j, k);
        fy_b = F.FY_momz(i, j - 1, k);
        fz_f = F.FZ_momz(i, j, k);
        fz_b = F.FZ_momz(i, j, k - 1);
        F.RHS_RhoW(i, j, k) = -((fx_r - fx_l) * idx + (fy_t - fy_b) * idy + (fz_f - fz_b) * idz);

        // energy
        fx_r = F.FX_E(i, j, k);
        fx_l = F.FX_E(i - 1, j, k);
        fy_t = F.FY_E(i, j, k);
        fy_b = F.FY_E(i, j - 1, k);
        fz_f = F.FZ_E(i, j, k);
        fz_b = F.FZ_E(i, j, k - 1);
        F.RHS_E(i, j, k) = -((fx_r - fx_l) * idx + (fy_t - fy_b) * idy + (fz_f - fz_b) * idz);
    }}}
}

// 三阶 Runge-Kutta 时间推进
void runge_kutta_3(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P, HaloRequests &out_reqs, double dt)
{
    LocalDesc &L = F.L;
    const int N = F.rho.size();

    std::vector<double> rho0 = F.rho;
    std::vector<double> rhou0 = F.rhou;
    std::vector<double> rhov0 = F.rhov;
    std::vector<double> rhow0 = F.rhow;
    std::vector<double> E0 = F.E;

    // Stage 1
    compute_rhs(F, G, P);
    for (int n = 0; n < N; ++n)
    {
        F.rho[n] = rho0[n] + dt * F.rhs_rho[n];
        F.rhou[n] = rhou0[n] + dt * F.rhs_rhou[n];
        F.rhov[n] = rhov0[n] + dt * F.rhs_rhov[n];
        F.rhow[n] = rhow0[n] + dt * F.rhs_rhow[n];
        F.E[n] = E0[n] + dt * F.rhs_E[n];
    }
    exchange_halos_conserved(F, C, L, out_reqs);
    F.conservedToPrimitive(P);

    // Stage 2
    compute_rhs(F, G, P);
    for (int n = 0; n < N; ++n)
    {
        F.rho[n] = 0.75 * rho0[n] + 0.25 * (F.rho[n] + dt * F.rhs_rho[n]);
        F.rhou[n] = 0.75 * rhou0[n] + 0.25 * (F.rhou[n] + dt * F.rhs_rhou[n]);
        F.rhov[n] = 0.75 * rhov0[n] + 0.25 * (F.rhov[n] + dt * F.rhs_rhov[n]);
        F.rhow[n] = 0.75 * rhow0[n] + 0.25 * (F.rhow[n] + dt * F.rhs_rhow[n]);
        F.E[n] = 0.75 * E0[n] + 0.25 * (F.E[n] + dt * F.rhs_E[n]);
    }
    exchange_halos_conserved(F, C, L, out_reqs);
    F.conservedToPrimitive(P);

    // Stage 3
    compute_rhs(F, G, P);
    for (int n = 0; n < N; ++n)
    {
        F.rho[n] = (1.0 / 3.0) * rho0[n] + (2.0 / 3.0) * (F.rho[n] + dt * F.rhs_rho[n]);
        F.rhou[n] = (1.0 / 3.0) * rhou0[n] + (2.0 / 3.0) * (F.rhou[n] + dt * F.rhs_rhou[n]);
        F.rhov[n] = (1.0 / 3.0) * rhov0[n] + (2.0 / 3.0) * (F.rhov[n] + dt * F.rhs_rhov[n]);
        F.rhow[n] = (1.0 / 3.0) * rhow0[n] + (2.0 / 3.0) * (F.rhow[n] + dt * F.rhs_rhow[n]);
        F.E[n] = (1.0 / 3.0) * E0[n] + (2.0 / 3.0) * (F.E[n] + dt * F.rhs_E[n]);
    }
    exchange_halos_conserved(F, C, L, out_reqs);
    F.conservedToPrimitive(P);
}