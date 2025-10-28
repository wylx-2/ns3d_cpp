#include <cmath>
#include <algorithm>
#include <vector>
#include "field_structures.h"
#include <mpi.h>
#include <iostream>

/// 基础欧拉通量函数
inline void flux_euler(double rho, double u, double v, double w,
                double p, double E, double F[5]);

/// 声速
inline double sound_speed(double gamma, double p, double rho);

/// Rusanov（Local Lax–Friedrichs）数值通量
void rusanov_flux(const double *QL, const double *QR, double gamma, double *Fhat);

// 简单的线性重构
std::vector<double> linear_reconstruction(const std::vector<std::vector<double>>& stencil);

// WENO重构
std::vector<double> weno5_reconstruction(const std::vector<std::vector<double>>& stencil);

// RHS 计算占位符函数（用户需在此定义具体的通量差分或高阶算子）
void compute_rhs(Field3D &F, const GridDesc &G, const SolverParams &P);

// 三阶 Runge-Kutta 时间推进
void runge_kutta_3(Field3D &F, CartDecomp &C, GridDesc &G,  SolverParams &P, HaloRequests &out_reqs, double dt);

// compute Euler physical flux in x-direction (conserved -> flux components)
inline void flux_euler(const std::array<double,5>& Q, const std::array<double,5>& prim, std::array<double,5>& F);

void rusanov_flux(const double *QL, const double *QR, double gamma, double *Fhat);

// 特征分解
void computeEigenSystem3D(double nx, double ny, double nz,
                          double gamma,
                          double rhobar, double ubar, double vbar, double wbar,
                          double Hbar, double c,
                          double L[5][5], double R[5][5]);

// Roe 数值通量
void roe_flux(const double *QL, const double *QR, 
              double nx, double ny, double nz, double gamma,
              double *Fhat);

// timestep calculation function
double compute_timestep(Field3D &F, const GridDesc &G, const SolverParams);

// 基于有限差分: 节点通量->重构到半节点->差分
void compute_flux_fd_x(Field3D &F, const GridDesc &G, const SolverParams &P);