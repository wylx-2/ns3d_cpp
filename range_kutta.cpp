#include "field_structures.h"
#include "ns3d_func.h"
#include <cmath>
#include <mpi.h>

//==================================================================
// 三阶 Runge-Kutta 时间推进主循环模块
//==================================================================

void compute_flux_x(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    const int nx = F.L.nx_local;
    const int ny = F.L.ny_local;
    const int nz = F.L.nz_local;
    const int nghost = F.L.nghost;
    const int nxg = nx + 2*nghost;
    const double dx = G.dx;

    std::vector<double> Fx(5 * nxg * ny * nz, 0.0);

    // 计算半节点通量
    for (int k=0; k<nz; ++k)
        for (int j=0; j<ny; ++j)
            for (int i=nghost-1; i<nx+nghost; ++i)
            {
                double QL[5], QR[5];
                for(int m=0;m<5;++m){
                    QL[m] = F.get_conserved(i-1,j,k,m);
                    QR[m] = F.get_conserved(i,j,k,m);
                }

                double Fhat[5];
                rusanov_flux(QL, QR, P.gamma, Fhat);

                int id = ((k*ny + j)*nxg + i);
                for (int m=0;m<5;++m)
                    Fx[5*id + m] = Fhat[m];
            }

    // 差分更新右端项
    for (int k=0; k<nz; ++k)
        for (int j=0; j<ny; ++j)
            for (int i=nghost; i<nx+nghost; ++i)
            {
                int idp = ((k*ny + j)*nxg + i);
                int idm = ((k*ny + j)*nxg + (i-1));

                F.rhs_rho[idp]  -= (Fx[5*idp+0] - Fx[5*idm+0]) / dx;
                F.rhs_rhou[idp] -= (Fx[5*idp+1] - Fx[5*idm+1]) / dx;
                F.rhs_rhov[idp] -= (Fx[5*idp+2] - Fx[5*idm+2]) / dx;
                F.rhs_rhow[idp] -= (Fx[5*idp+3] - Fx[5*idm+3]) / dx;
                F.rhs_E[idp]    -= (Fx[5*idp+4] - Fx[5*idm+4]) / dx;
            }
}

// RHS 计算占位符函数（用户需在此定义具体的通量差分或高阶算子）
void compute_rhs(Field3D &F, const GridDesc &G, const SolverParams &P)
{
const LocalDesc &L = F.L;
// 清空 RHS
std::fill(F.rhs_rho.begin(), F.rhs_rho.end(), 0.0);
std::fill(F.rhs_rhou.begin(), F.rhs_rhou.end(), 0.0);
std::fill(F.rhs_rhov.begin(), F.rhs_rhov.end(), 0.0);
std::fill(F.rhs_rhow.begin(), F.rhs_rhow.end(), 0.0);
std::fill(F.rhs_E.begin(), F.rhs_E.end(), 0.0);


// 示例：此处调用通量计算函数
// compute_fluxes(F, G, P);
// accumulate_rhs(F);
}


// 三阶 Runge-Kutta 时间推进
void runge_kutta_3(Field3D &F,  GridDesc &G,  SolverParams &P, double dt)
{
const LocalDesc &L = F.L;
const int N = F.rho.size();

std::vector<double> rho0 = F.rho;
std::vector<double> rhou0 = F.rhou;
std::vector<double> rhov0 = F.rhov;
std::vector<double> rhow0 = F.rhow;
std::vector<double> E0 = F.E;


// Stage 1
compute_rhs(F, G, P);
for (int n=0; n<N; ++n) {
F.rho[n] = rho0[n] + dt * F.rhs_rho[n];
F.rhou[n] = rhou0[n] + dt * F.rhs_rhou[n];
F.rhov[n] = rhov0[n] + dt * F.rhs_rhov[n];
F.rhow[n] = rhow0[n] + dt * F.rhs_rhow[n];
F.E[n] = E0[n] + dt * F.rhs_E[n];
}
F.conservedToPrimitive(P);
exchange_halos_conserved(F);

// Stage 2
compute_rhs(F, G, P);
for (int n=0; n<N; ++n) {
F.rho[n] = 0.75 * rho0[n] + 0.25 * (F.rho[n] + dt * F.rhs_rho[n]);
F.rhou[n] = 0.75 * rhou0[n] + 0.25 * (F.rhou[n] + dt * F.rhs_rhou[n]);
F.rhov[n] = 0.75 * rhov0[n] + 0.25 * (F.rhov[n] + dt * F.rhs_rhov[n]);
F.rhow[n] = 0.75 * rhow0[n] + 0.25 * (F.rhow[n] + dt * F.rhs_rhow[n]);
F.E[n] = 0.75 * E0[n] + 0.25 * (F.E[n] + dt * F.rhs_E[n]);
}
F.conservedToPrimitive(P);
exchange_halos_conserved(F);
}