#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include "field_structures.h"
#include <mpi.h>
#include <iostream>
#include <string>

/// 基础欧拉通量函数
inline void flux_euler(double rho, double u, double v, double w,
                double p, double E, double F[5]);

/// 声速
inline double sound_speed(double gamma, double p, double rho);

/// Rusanov（Local Lax–Friedrichs）数值通量
void rusanov_flux(const double *QL, const double *QR, double gamma, double *Fhat);

// 简单的线性重构（标量，5 点模板）
double linear_reconstruction(const std::array<double,5> &stencil);

// WENO5 重构（标量，5 点模板）
double weno5_reconstruction(const std::array<double,5> &stencil);

// C6th 六阶中心差分重构（标量，6 点模板）
double c6th_reconstruction(const std::array<double,6> &stencil);

// C4th 四阶中心差分(标量，4 点模板)
double c4th_reconstruction(const std::array<double,4>& stencil);

// RHS 计算占位符函数（用户需在此定义具体的通量差分或高阶算子）
void compute_rhs(Field3D &F, const GridDesc &G, const SolverParams &P);

// 三阶 Runge-Kutta 时间推进
void runge_kutta_3(Field3D &F, CartDecomp &C, GridDesc &G,  SolverParams &P, HaloRequests &out_reqs, double dt);

// timestep calculation function
double compute_timestep(Field3D &F, const GridDesc &G, const SolverParams);

// 基于有限差分: 节点通量->重构到半节点->差分
void compute_flux_fd_x(Field3D &F, const GridDesc &G, const SolverParams &P);

// 计算无粘通量
void compute_flux(Field3D &F, const SolverParams &P);

// 重构无粘通量
void compute_flux_face(Field3D &F, const SolverParams &P);

// 计算空间导数
void compute_gradients(Field3D &F, const GridDesc &G, const SolverParams &P);

// 计算粘性通量
void compute_viscous_flux(Field3D &F, const SolverParams &P);

// 重构粘性通量
void compute_vis_flux_face(Field3D &F, const SolverParams &P);

// Output full field in Tecplot ASCII format (per-rank file). Prefix will be used for filename: <prefix>_rank<id>.dat
void write_tecplot_field(const Field3D &F, const GridDesc &G, const CartDecomp &C, const std::string &prefix = "field");
