#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include "field_structures.h"
#include <mpi.h>
#include <iostream>
#include <string>

// 初始化求解参数
void initialize_SolverParams(SolverParams &P, CartDecomp &C, GridDesc &G);

// 初始化流场
// 均匀场
void initialize_uniform_field(Field3D &F, const GridDesc &G, const SolverParams &P);

// 2D Riemann 问题初始条件
void initialize_riemann_2d(Field3D &F, const GridDesc &G, const SolverParams &P);

// 边界条件处理函数
void apply_boundary(Field3D &F, GridDesc &G, CartDecomp &C, const SolverParams &P);

/// 基础欧拉通量函数
inline void flux_euler(double rho, double u, double v, double w,
                double p, double E, double F[5]);

/// 声速
inline double sound_speed(double gamma, double p, double rho);

/// Rusanov（Local Lax–Friedrichs）数值通量
void rusanov_flux(const double *QL, const double *QR, double gamma, double *Fhat);

// 简单的线性重构（标量，5 点模板）
double linear_reconstruction(const std::array<double,2> &stencil);

// MDCD 重构（标量，6 点模板）
double mdcd_reconstruction(const std::array<double,6>& stencil, SolverParams P);

// WENO5 重构（标量，5 点模板）
double weno5_reconstruction(const std::array<double,5> &stencil);

// C6th 六阶中心差分重构（标量，6 点模板）
double c6th_reconstruction(const std::array<double,6> &stencil);

// C4th 四阶中心差分(标量，4 点模板)
double c4th_reconstruction(const std::array<double,4>& stencil);

// Runtime-sized reconstruction selector (accepts std::vector stencil)
double reconstruct_select(const std::vector<double> &stencil, double flag, const SolverParams P);

// RHS 计算占位符函数（用户需在此定义具体的通量差分或高阶算子）
void compute_rhs(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P, HaloRequests &out_reqs);

// 三阶 Runge-Kutta 时间推进
void runge_kutta_3(Field3D &F, CartDecomp &C, GridDesc &G,  SolverParams &P, HaloRequests &out_reqs, double dt);

// timestep calculation function
double compute_timestep(Field3D &F, const GridDesc &G, const SolverParams &P);

// 基于有限差分: 节点通量->重构到半节点->差分
void compute_flux_fd_x(Field3D &F, const GridDesc &G, const SolverParams &P);

// 计算无粘通量
void compute_flux(Field3D &F, const SolverParams &P);

// 计算数值无粘通量
void computeFVSFluxes(Field3D &F, const SolverParams &P);

// 重构无粘通量
void reconstructInviscidFlux(std::vector<double> &Fface,
                             const std::vector<std::vector<double>> &Ft,
                             const std::vector<std::vector<double>> &Ut,
                             const std::vector<std::vector<double>> &ut,
                             const SolverParams &P, int dim);

// 计算 Roe 平均态
void computeRoeAveragedState(double &rho_bar, double &rhou_bar, double &rhov_bar, double &rhow_bar,
                             double &h_bar, double &a_bar,
                             const double Ul[5], const double Ur[5],
                             double gamma);

// 计算特征分解
static void build_eigen_matrices(const double Ul[5], const double Ur[5],
                                 double nx, double ny, double nz,
                                 double gamma,
                                 double Lmat[5][5], double Rmat[5][5],
                                 double lambar[5]);

// 计算空间导数
void compute_gradients(Field3D &F, const GridDesc &G, const SolverParams &P);

// 计算粘性通量
void compute_viscous_flux(Field3D &F, const SolverParams &P);

// 重构粘性通量
void reconstructViscidFlux(Field3D &F, const SolverParams &P);

// Output full field in Tecplot ASCII format (per-rank file). Prefix will be used for filename: <prefix>_rank<id>.dat
// time: physical time to label the output (optional, default 0.0)
void write_tecplot_field(const Field3D &F, const GridDesc &G, const CartDecomp &C, const std::string &prefix = "field", double time = 0.0);

// Write residuals (per-equation L2 residuals and total energy) vs time step to a Tecplot-like ASCII table.
// The file will contain VARIABLES = "Step" "Res_rho" "Res_rhou" "Res_rhov" "Res_rhow" "Res_E" "Etot"
// If step==0 the file is overwritten and header is written; otherwise the line is appended.
void write_residuals_tecplot(const Field3D &F, int step, const std::string &filename = "residuals.dat");

// compute diagnostics: rms of RHS, total energy, residual
void compute_diagnostics(Field3D &F, const SolverParams &P);

// main time advance loop with monitor & output
void time_advance(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P, HaloRequests &out_reqs);

