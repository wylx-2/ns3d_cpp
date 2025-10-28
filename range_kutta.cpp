#include "field_structures.h"
#include "ns3d_func.h"
#include <cmath>
#include <mpi.h>

//==================================================================
// 三阶 Runge-Kutta 时间推进主循环模块
//==================================================================
using ReconstructionFunc = std::vector<double>(*)(const std::vector<std::vector<double>>&);
// 基于有限差分: 节点通量->重构到半节点->差分
void compute_flux_fd_x(Field3D &F, const GridDesc &G, const SolverParams &P, ReconstructionFunc reconstruct = weno5_reconstruction){
    int nx=F.L.nx, ny=F.L.ny, nz=F.L.nz, ngh=F.L.ngx;
    int nxg=nx+2*ngh; double dx=G.dx;
    static std::vector<double> Fx; Fx.resize(5*nxg*ny*nz);
    // 节点处计算F = physical flux
    for(int k=0;k<nz;++k)
    for(int j=0;j<ny;++j)
    for(int i=0;i<nxg;++i){
        int ig=i, jg=j+ngh, kg=k+ngh;
        int id=F.I(ig,jg,kg);
        double rho=F.rho[id], u=F.rhou[id]/rho, v=F.rhov[id]/rho, w=F.rhow[id]/rho;
        double p=(P.gamma-1.0)*(F.E[id]-0.5*rho*(u*u+v*v+w*w));
        double Fe[5]; 
        
        flux_euler(rho,u,v,w,p,F.E[id],Fe); // 计算物理通量，独立于方向，应该独立封装
        for(int m=0;m<5;++m) Fx[5*((k)*ny + j)*nxg + 5*i + m] = Fe[m];
    }
    // 重构 F -> H at i+1/2 (简单中心)
    //（用户需在此定义具体的通量差分或高阶算子）
    for(int k = 0; k < nz; ++k) {
        for(int j = 0; j < ny; ++j) {
            for(int i = ngh - 1; i < nx + ngh + 2; ++i) {
                // 提取5点模板
                std::vector<std::vector<double>> stencil(5, std::vector<double>(5));
                
                for(int s = 0; s < 5; ++s) {
                    int template_idx = i - 2 + s;
                    int template_base = 5 * (k * ny + j) * nxg + 5 * template_idx;
                    
                    for(int m = 0; m < 5; ++m) {
                        stencil[s][m] = Fx[template_base + m];
                    }
                }
                
                // 调用重构函数
                auto reconstructed = reconstruct(stencil);
                
                // 存储重构结果
                int base = 5 * (k * ny + j) * nxg + 5 * i;
                for(int m = 0; m < 5; ++m) {
                    Fx[base + m] = reconstructed[m];
                }
            }
        }
    }

    // 差分更新 RHS: DF = (H_{i+1/2}-H_{i-1/2})/dx
    for(int k=0;k<nz;++k)
    for(int j=0;j<ny;++j)
    for(int i=ngh;i<nx+ngh;++i){
        int ip=5*((k)*ny + j)*nxg + 5*(i);
        int im=5*((k)*ny + j)*nxg + 5*(i-1);
        int id=F.I(i,j+ngh,k+ngh);
        F.rhs_rho[id] -= (Fx[ip+0]-Fx[im+0])/dx;
        F.rhs_rhou[id] -= (Fx[ip+1]-Fx[im+1])/dx;
        F.rhs_rhov[id] -= (Fx[ip+2]-Fx[im+2])/dx;
        F.rhs_rhow[id] -= (Fx[ip+3]-Fx[im+3])/dx;
        F.rhs_E[id] -= (Fx[ip+4]-Fx[im+4])/dx;
    }
}

void compute_flux_x(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    const int nx = F.L.nx;
    const int ny = F.L.ny;
    const int nz = F.L.nz;
    const int nghost = F.L.ngx;
    const int nxg = nx + 2*nghost;
    const double dx = G.dx;

    std::vector<double> Fx(5 * nxg * ny * nz, 0.0);

    // 计算半节点通量
    for (int k=0; k<nz; ++k)
        for (int j=0; j<ny; ++j)
            for (int i=nghost-1; i<nx+nghost; ++i)
            {
                // 获取界面处的左右状态，若采用Riemann则高阶重构可在此处实现
                double QL[5], QR[5];
                QL[0] = F.Rho(i-1,j,k);QR[0] = F.Rho(i,j,k);
                QL[1] = F.RhoU(i-1,j,k);QR[1] = F.RhoU(i,j,k);
                QL[2] = F.RhoV(i-1,j,k);QR[2] = F.RhoV(i,j,k);
                QL[3] = F.RhoW(i-1,j,k);QR[3] = F.RhoW(i,j,k);
                QL[4] = F.Eint(i-1,j,k);QR[4] = F.Eint(i,j,k);

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
void runge_kutta_3(Field3D &F, CartDecomp &C, GridDesc &G,  SolverParams &P, HaloRequests &out_reqs, double dt)
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
for (int n=0; n<N; ++n) {
F.rho[n] = rho0[n] + dt * F.rhs_rho[n];
F.rhou[n] = rhou0[n] + dt * F.rhs_rhou[n];
F.rhov[n] = rhov0[n] + dt * F.rhs_rhov[n];
F.rhow[n] = rhow0[n] + dt * F.rhs_rhow[n];
F.E[n] = E0[n] + dt * F.rhs_E[n];
}
F.conservedToPrimitive(P);
exchange_halos_conserved(F,C,L,out_reqs);

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
exchange_halos_conserved(F,C,L,out_reqs);

// Stage 3
compute_rhs(F, G, P);
for (int n=0; n<N; ++n) {
F.rho[n] = (1.0/3.0) * rho0[n] + (2.0/3.0) * (F.rho[n] + dt * F.rhs_rho[n]);
F.rhou[n] = (1.0/3.0) * rhou0[n] + (2.0/3.0) * (F.rhou[n] + dt * F.rhs_rhou[n]);
F.rhov[n] = (1.0/3.0) * rhov0[n] + (2.0/3.0) * (F.rhov[n] + dt * F.rhs_rhov[n]);
F.rhow[n] = (1.0/3.0) * rhow0[n] + (2.0/3.0) * (F.rhow[n] + dt * F.rhs_rhow[n]);
F.E[n] = (1.0/3.0) * E0[n] + (2.0/3.0) * (F.E[n] + dt * F.rhs_E[n]);
}
F.conservedToPrimitive(P);
exchange_halos_conserved(F,C,L,out_reqs);
}