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

    // FVS计算无粘通量
    computeFVSFluxes(F, P);

    // 计算空间导数
    compute_gradients(F, G);
    // 同步不同进程的ghost区域空间导数
    // exchange_halos_gradients(F, C, L, out_reqs); //还需要额外边界处理！
    // 计算粘性通量
    compute_viscous_flux(F, P);
    // 应该在这里交换粘性通量的halo区域，并处理周期边界

    // 粘性通量的导数
    compute_vis_flux(F, G);

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
        F.RHS_Rho(i, j, k) += -((fx_r - fx_l) * idx + (fy_t - fy_b) * idy + (fz_f - fz_b) * idz);

        // momentum x
        fx_r = F.FX_momx(i, j, k);
        fx_l = F.FX_momx(i - 1, j, k);
        fy_t = F.FY_momx(i, j, k);
        fy_b = F.FY_momx(i, j - 1, k);
        fz_f = F.FZ_momx(i, j, k);
        fz_b = F.FZ_momx(i, j, k - 1);
        F.RHS_RhoU(i, j, k) += -((fx_r - fx_l) * idx + (fy_t - fy_b) * idy + (fz_f - fz_b) * idz);

        // momentum y
        fx_r = F.FX_momy(i, j, k);
        fx_l = F.FX_momy(i - 1, j, k);
        fy_t = F.FY_momy(i, j, k);
        fy_b = F.FY_momy(i, j - 1, k);
        fz_f = F.FZ_momy(i, j, k);
        fz_b = F.FZ_momy(i, j, k - 1);
        F.RHS_RhoV(i, j, k) += -((fx_r - fx_l) * idx + (fy_t - fy_b) * idy + (fz_f - fz_b) * idz);

        // momentum z
        fx_r = F.FX_momz(i, j, k);
        fx_l = F.FX_momz(i - 1, j, k);
        fy_t = F.FY_momz(i, j, k);
        fy_b = F.FY_momz(i, j - 1, k);
        fz_f = F.FZ_momz(i, j, k);
        fz_b = F.FZ_momz(i, j, k - 1);
        F.RHS_RhoW(i, j, k) += -((fx_r - fx_l) * idx + (fy_t - fy_b) * idy + (fz_f - fz_b) * idz);

        // energy
        fx_r = F.FX_E(i, j, k);
        fx_l = F.FX_E(i - 1, j, k);
        fy_t = F.FY_E(i, j, k);
        fy_b = F.FY_E(i, j - 1, k);
        fz_f = F.FZ_E(i, j, k);
        fz_b = F.FZ_E(i, j, k - 1);
        F.RHS_E(i, j, k) += -((fx_r - fx_l) * idx + (fy_t - fy_b) * idy + (fz_f - fz_b) * idz);
    }}}
}

// 三阶 Runge-Kutta 时间推进
void runge_kutta_3(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P, HaloRequests &out_reqs, double dt)
{
    LocalDesc &L = F.L;
    const int N = F.rho.size();

    // Stage 1
    compute_rhs(F, C, G, P, out_reqs);
    for (int n = 0; n < N; ++n)
    {
        F.rho[n] = F.rho0[n] + dt * F.rhs_rho[n];
        F.rhou[n] = F.rhou0[n] + dt * F.rhs_rhou[n];
        F.rhov[n] = F.rhov0[n] + dt * F.rhs_rhov[n];
        F.rhow[n] = F.rhow0[n] + dt * F.rhs_rhow[n];
        F.E[n] = F.E0[n] + dt * F.rhs_E[n];
    }
    F.conservedToPrimitive(P);
    apply_boundary(F, G, C, P);
    F.primitiveToConserved(P);

    // Stage 2
    compute_rhs(F, C, G, P, out_reqs);
    for (int n = 0; n < N; ++n)
    {
        F.rho[n] = 0.75 * F.rho0[n] + 0.25 * (F.rho[n] + dt * F.rhs_rho[n]);
        F.rhou[n] = 0.75 * F.rhou0[n] + 0.25 * (F.rhou[n] + dt * F.rhs_rhou[n]);
        F.rhov[n] = 0.75 * F.rhov0[n] + 0.25 * (F.rhov[n] + dt * F.rhs_rhov[n]);
        F.rhow[n] = 0.75 * F.rhow0[n] + 0.25 * (F.rhow[n] + dt * F.rhs_rhow[n]);
        F.E[n] = 0.75 * F.E0[n] + 0.25 * (F.E[n] + dt * F.rhs_E[n]);
    }
    F.conservedToPrimitive(P);
    apply_boundary(F, G, C, P);
    F.primitiveToConserved(P);

    // Stage 3
    compute_rhs(F, C, G, P, out_reqs);
    for (int n = 0; n < N; ++n)
    {
        F.rho[n] = (1.0 / 3.0) * F.rho0[n] + (2.0 / 3.0) * (F.rho[n] + dt * F.rhs_rho[n]);
        F.rhou[n] = (1.0 / 3.0) * F.rhou0[n] + (2.0 / 3.0) * (F.rhou[n] + dt * F.rhs_rhou[n]);
        F.rhov[n] = (1.0 / 3.0) * F.rhov0[n] + (2.0 / 3.0) * (F.rhov[n] + dt * F.rhs_rhov[n]);
        F.rhow[n] = (1.0 / 3.0) * F.rhow0[n] + (2.0 / 3.0) * (F.rhow[n] + dt * F.rhs_rhow[n]);
        F.E[n] = (1.0 / 3.0) * F.E0[n] + (2.0 / 3.0) * (F.E[n] + dt * F.rhs_E[n]);
    }
    F.conservedToPrimitive(P);
    apply_boundary(F, G, C, P);
    F.primitiveToConserved(P);
}

// -----------------------------------------------------------------------------
// 辅助函数：计算总能量、残差、RMS
// -----------------------------------------------------------------------------
void compute_diagnostics(Field3D &F, const SolverParams &P)
{
    const LocalDesc &L = F.L;
	double sum_E = 0.0;
	// per-variable accumulators for true residual calculation
    // double sum_abs_res_rho = 0.0, sum_abs_res_rhou = 0.0, sum_abs_res_rhov = 0.0, sum_abs_res_rhow = 0.0, sum_abs_res_E = 0.0;
    double sum_sq_res_rho = 0.0, sum_sq_res_rhou = 0.0, sum_sq_res_rhov = 0.0, sum_sq_res_rhow = 0.0, sum_sq_res_E = 0.0;
	double max_abs_rho = 0.0, max_abs_rhou = 0.0, max_abs_rhov = 0.0, max_abs_rhow = 0.0, max_abs_E = 0.0;
    int count = 0;

    for (int k = L.ngz; k < L.ngz + L.nz; ++k)
    for (int j = L.ngy; j < L.ngy + L.ny; ++j)
    for (int i = L.ngx; i < L.ngx + L.nx; ++i)
    {
        int id = F.I(i,j,k);
        double rho = F.rho[id];
        double u = F.u[id], v = F.v[id], w = F.w[id];
        double E = F.E[id];
        double p = F.p[id];
        double rhou = F.rhou[id];
        double rhov = F.rhov[id];
        double rhow = F.rhow[id];

        // total energy
        double kinetic = 0.5 * rho * (u*u + v*v + w*w);
        double eint = p / (P.gamma - 1.0);
        sum_E += kinetic + eint;

        // max absolute values for normalization
        max_abs_rho = std::max(max_abs_rho, std::abs(rho));
        max_abs_rhou = std::max(max_abs_rhou, std::abs(rhou));
        max_abs_rhov = std::max(max_abs_rhov, std::abs(rhov));
        max_abs_rhow = std::max(max_abs_rhow, std::abs(rhow));
        max_abs_E = std::max(max_abs_E, std::abs(E));

		// residual
        F.updateResiduals();
        double res_rho = F.res_rho[id];;
        double res_rhou = F.res_rhou[id];
        double res_rhov = F.res_rhov[id];
        double res_rhow = F.res_rhow[id];
        double res_E = F.res_E[id];

        /*
        sum_abs_res_rho += std::abs(res_rho);
        sum_abs_res_rhou += std::abs(res_rhou);
        sum_abs_res_rhov += std::abs(res_rhov);
        sum_abs_res_rhow += std::abs(res_rhow);
        sum_abs_res_E += std::abs(res_E);
        */

		sum_sq_res_rho += res_rho * res_rho;
        sum_sq_res_rhou += res_rhou * res_rhou;
        sum_sq_res_rhov += res_rhov * res_rhov;
        sum_sq_res_rhow += res_rhow * res_rhow;
        sum_sq_res_E += res_E * res_E;
        ++count;
    }
	// global reductions
    double g_E = 0.0;
    // double g_sum_abs_res_rho = 0.0, g_sum_abs_res_rhou = 0.0, g_sum_abs_res_rhov = 0.0, g_sum_abs_res_rhow = 0.0, g_sum_abs_res_E = 0.0;
    double g_sum_sq_res_rho = 0.0, g_sum_sq_res_rhou = 0.0, g_sum_sq_res_rhov = 0.0, g_sum_sq_res_rhow = 0.0, g_sum_sq_res_E = 0.0;
    double g_max_abs_rho = 0.0, g_max_abs_rhou = 0.0, g_max_abs_rhov = 0.0, g_max_abs_rhow = 0.0, g_max_abs_E = 0.0;
    int g_N = 0;
    MPI_Allreduce(&sum_E, &g_E, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // MPI_Allreduce(&sum_abs_res_rho, &g_sum_abs_res_rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Allreduce(&sum_abs_res_rhou, &g_sum_abs_res_rhou, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Allreduce(&sum_abs_res_rhov, &g_sum_abs_res_rhov, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Allreduce(&sum_abs_res_rhow, &g_sum_abs_res_rhow, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Allreduce(&sum_abs_res_E, &g_sum_abs_res_E, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&sum_sq_res_rho, &g_sum_sq_res_rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_sq_res_rhou, &g_sum_sq_res_rhou, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_sq_res_rhov, &g_sum_sq_res_rhov, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_sq_res_rhow, &g_sum_sq_res_rhow, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_sq_res_E, &g_sum_sq_res_E, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&max_abs_rho, &g_max_abs_rho, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_abs_rhou, &g_max_abs_rhou, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_abs_rhov, &g_max_abs_rhov, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_abs_rhow, &g_max_abs_rhow, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_abs_E, &g_max_abs_E, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(&count, &g_N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // input diagnostics
    F.global_Etot = g_E;
    // residual = (g_sum_abs_res_rho / g_N) / g_max_abs_rho;
    F.global_res_rho = std::sqrt( (g_sum_sq_res_rho / g_N) ) / g_max_abs_rho;
    F.global_res_rhou = std::sqrt( (g_sum_sq_res_rhou / g_N) ) / g_max_abs_rhou;
    F.global_res_rhov = std::sqrt( (g_sum_sq_res_rhov / g_N) ) / g_max_abs_rhov;
    F.global_res_rhow = std::sqrt( (g_sum_sq_res_rhow / g_N) ) / g_max_abs_rhow;
    F.global_res_E = std::sqrt( (g_sum_sq_res_E / g_N) ) / g_max_abs_E;

}
