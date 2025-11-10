#include "field_structures.h"
#include <mpi.h>
#include <algorithm>
#include <iostream>

//------------------------------------------------------------
// 边界更新核心函数
//------------------------------------------------------------
void apply_boundary(Field3D &F, GridDesc &G, CartDecomp &C, SolverParams &P)
{
    LocalDesc &L = F.L;
    const int ng = L.ngx;  // 假定三个方向 ghost 一样

    // 获取每个方向的尺寸
    int sx = L.sx, sy = L.sy, sz = L.sz;
    int nx = L.nx, ny = L.ny, nz = L.nz;

    //--------------------------------------------
    // Step 1. 先进行 MPI 边界交换
    //--------------------------------------------
    HaloRequests reqs;
    exchange_halos_conserved(F, C, L, reqs);
    MPI_Waitall(static_cast<int>(reqs.reqs.size()), reqs.reqs.data(), MPI_STATUSES_IGNORE);

    //--------------------------------------------
    // Step 2. 本地物理边界 (如果当前进程在边界上)
    //--------------------------------------------
    // ---- XMIN / XMAX ----
    if (L.nbr_xm == MPI_PROC_NULL) {
        if (P.bc_xmin == LocalDesc::BCType::Wall || P.bc_xmin == LocalDesc::BCType::Symmetry) {
            for (int k=0;k<sz;++k)
            for (int j=0;j<sy;++j)
            for (int i=0;i<ng;++i) {
                int ig = i;
                int im = 2*ng - 1 - i; // 镜像位置
                int idg = F.I(ig,j,k), idm = F.I(im,j,k);

                // 镜像
                F.rho[idg]  = F.rho[idm];
                F.rhou[idg] = -F.rhou[idm]; // 反射法向动量
                F.rhov[idg] =  F.rhov[idm];
                F.rhow[idg] =  F.rhow[idm];
                F.E[idg]    =  F.E[idm];
            }
        } else if (bc_xmin == LocalDesc::BCType::Outflow) {
            for (int k=0;k<sz;++k)
            for (int j=0;j<sy;++j)
            for (int i=0;i<ng;++i) {
                int idg = F.I(i,j,k);
                int idi = F.I(ng,j,k);  // 最近的内层
                F.rho[idg]  = F.rho[idi];
                F.rhou[idg] = F.rhou[idi];
                F.rhov[idg] = F.rhov[idi];
                F.rhow[idg] = F.rhow[idi];
                F.E[idg]    = F.E[idi];
            }
        }
    }
    if (L.nbr_xp == MPI_PROC_NULL) {
        if (bc_xmax == LocalDesc::BCType::Wall || bc_xmax == LocalDesc::BCType::Symmetry) {
            for (int k=0;k<sz;++k)
            for (int j=0;j<sy;++j)
            for (int i=sx-ng;i<sx;++i) {
                int ig = i;
                int im = 2*(sx-ng) - 1 - i;
                int idg = F.I(ig,j,k), idm = F.I(im,j,k);

                F.rho[idg]  = F.rho[idm];
                F.rhou[idg] = -F.rhou[idm];
                F.rhov[idg] =  F.rhov[idm];
                F.rhow[idg] =  F.rhow[idm];
                F.E[idg]    =  F.E[idm];
            }
        }
        else if (bc_xmax == LocalDesc::BCType::Outflow) {
            for (int k=0;k<sz;++k)
            for (int j=0;j<sy;++j)
            for (int i=sx-ng;i<sx;++i) {
                int idg = F.I(i,j,k);
                int idi = F.I(sx-ng-1,j,k);
                F.rho[idg]  = F.rho[idi];
                F.rhou[idg] = F.rhou[idi];
                F.rhov[idg] = F.rhov[idi];
                F.rhow[idg] = F.rhow[idi];
                F.E[idg]    = F.E[idi];
            }
        }
    }

    // ---- YMIN / YMAX ----
    if (L.nbr_ym == MPI_PROC_NULL) {
        if (bc_ymin == LocalDesc::BCType::Wall || bc_ymin == LocalDesc::BCType::Symmetry) {
            for (int k=0;k<sz;++k)
            for (int j=0;j<ng;++j)
            for (int i=0;i<sx;++i) {
                int idg = F.I(i,j,k);
                int jm = 2*ng - 1 - j;
                int idm = F.I(i,jm,k);
                F.rho[idg]  = F.rho[idm];
                F.rhov[idg] = -F.rhov[idm]; // 法向分量翻转
                F.rhou[idg] =  F.rhou[idm];
                F.rhow[idg] =  F.rhow[idm];
                F.E[idg]    =  F.E[idm];
            }
        }
        else if (bc_ymin == LocalDesc::BCType::Outflow) {
            for (int k=0;k<sz;++k)
            for (int j=0;j<ng;++j)
            for (int i=0;i<sx;++i) {
                int idg = F.I(i,j,k);
                int idi = F.I(i,ng,k);
                F.rho[idg]  = F.rho[idi];
                F.rhou[idg] = F.rhou[idi];
                F.rhov[idg] = F.rhov[idi];
                F.rhow[idg] = F.rhow[idi];
                F.E[idg]    = F.E[idi];
            }
        }
    }
    if (L.nbr_yp == MPI_PROC_NULL) {
        if (bc_ymax == LocalDesc::BCType::Wall || bc_ymax == LocalDesc::BCType::Symmetry) {
            for (int k=0;k<sz;++k)
            for (int j=sy-ng;j<sy;++j)
            for (int i=0;i<sx;++i) {
                int idg = F.I(i,j,k);
                int jm = 2*(sy-ng) - 1 - j;
                int idm = F.I(i,jm,k);
                F.rho[idg]  = F.rho[idm];
                F.rhov[idg] = -F.rhov[idm];
                F.rhou[idg] =  F.rhou[idm];
                F.rhow[idg] =  F.rhow[idm];
                F.E[idg]    =  F.E[idm];
            }
        }
        else if (bc_ymax == LocalDesc::BCType::Outflow) {
            for (int k=0;k<sz;++k)
            for (int j=sy-ng;j<sy;++j)
            for (int i=0;i<sx;++i) {
                int idg = F.I(i,j,k);
                int idi = F.I(i,sy-ng-1,k);
                F.rho[idg]  = F.rho[idi];
                F.rhou[idg] = F.rhou[idi];
                F.rhov[idg] = F.rhov[idi];
                F.rhow[idg] = F.rhow[idi];
                F.E[idg]    = F.E[idi];
            }
        }
    }

    // ---- ZMIN / ZMAX ----
    if (L.nbr_zm == MPI_PROC_NULL) {
        if (bc_zmin == LocalDesc::BCType::Wall || bc_zmin == LocalDesc::BCType::Symmetry) {
            for (int k=0;k<ng;++k)
            for (int j=0;j<sy;++j)
            for (int i=0;i<sx;++i) {
                int idg = F.I(i,j,k);
                int km = 2*ng - 1 - k;
                int idm = F.I(i,j,km);
                F.rho[idg]  = F.rho[idm];
                F.rhow[idg] = -F.rhow[idm];
                F.rhou[idg] =  F.rhou[idm];
                F.rhov[idg] =  F.rhov[idm];
                F.E[idg]    =  F.E[idm];
            }
        }
        else if (bc_zmin == LocalDesc::BCType::Outflow) {
            for (int k=0;k<ng;++k)
            for (int j=0;j<sy;++j)
            for (int i=0;i<sx;++i) {
                int idg = F.I(i,j,k);
                int idi = F.I(i,j,ng);
                F.rho[idg]  = F.rho[idi];
                F.rhou[idg] = F.rhou[idi];
                F.rhov[idg] = F.rhov[idi];
                F.rhow[idg] = F.rhow[idi];
                F.E[idg]    = F.E[idi];
            }
        }
    }
    if (L.nbr_zp == MPI_PROC_NULL) {
        if (bc_zmax == LocalDesc::BCType::Wall || bc_zmax == LocalDesc::BCType::Symmetry) {
            for (int k=sz-ng;k<sz;++k)
            for (int j=0;j<sy;++j)
            for (int i=0;i<sx;++i) {
                int idg = F.I(i,j,k);
                int km = 2*(sz-ng) - 1 - k;
                int idm = F.I(i,j,km);
                F.rho[idg]  = F.rho[idm];
                F.rhow[idg] = -F.rhow[idm];
                F.rhou[idg] =  F.rhou[idm];
                F.rhov[idg] =  F.rhov[idm];
                F.E[idg]    =  F.E[idm];
            }
        }
        else if (bc_zmax == LocalDesc::BCType::Outflow) {
            for (int k=sz-ng;k<sz;++k)
            for (int j=0;j<sy;++j)
            for (int i=0;i<sx;++i) {
                int idg = F.I(i,j,k);
                int idi = F.I(i,j,sz-ng-1);
                F.rho[idg]  = F.rho[idi];
                F.rhou[idg] = F.rhou[idi];
                F.rhov[idg] = F.rhov[idi];
                F.rhow[idg] = F.rhow[idi];
                F.E[idg]    = F.E[idi];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}
