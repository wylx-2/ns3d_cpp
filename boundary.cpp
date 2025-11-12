#include "field_structures.h"
#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>

enum FaceID { XMIN=0, XMAX=1, YMIN=2, YMAX=3, ZMIN=4, ZMAX=5 };
struct NeighborInfo { int nbr; SolverParams::BCType face; FaceID id; };
//------------------------------------------------------------
// 边界更新核心函数
//------------------------------------------------------------
void apply_boundary(Field3D &F, GridDesc &G, CartDecomp &C,
                    const SolverParams &P)
{
    LocalDesc &L = F.L;
    // Step 1: Halo exchange for periodic boundaries
    HaloRequests reqs;
    exchange_halos_physical(F, C, L, reqs);

    // Step 2: 对每个方向检查是否需要本地边界
    NeighborInfo dirs[6] = {
        {L.nbr_xm, P.bc_xmax, XMAX}, {L.nbr_xp, P.bc_xmin, XMIN},
        {L.nbr_ym, P.bc_ymax, YMAX}, {L.nbr_yp, P.bc_ymin, YMIN},
        {L.nbr_zm, P.bc_zmax, ZMAX}, {L.nbr_zp, P.bc_zmin, ZMIN}
    };

    for (auto &d : dirs)
    {
        if (d.nbr != MPI_PROC_NULL) continue; // 有邻居 → 已由通信完成
        switch (d.face)
        {
            case SolverParams::BCType::Wall:
                apply_wall_bc(F, L, d.id);
                break;
            case SolverParams::BCType::Symmetry:
                apply_symmetry_bc(F, L, d.id);
                break;
            case SolverParams::BCType::Outflow:
                apply_outflow_bc(F, L, d.id);
                break;
            case SolverParams::BCType::Inflow:
                apply_inflow_bc(F, L, d.id);
                break;
            case SolverParams::BCType::Periodic:
                // 周期边界与MPI通信处理分开处理，传输内点值而非边界值
                apply_periodic_bc(F, L, C, d.id);
                break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// Wall boundary condition implementation
// No-slip wall 存在问题
void apply_wall_bc(Field3D &F, const LocalDesc &L, int face)
{
    int ng = L.ngx, sx = L.sx, sy = L.sy, sz = L.sz;

    if (face == XMIN)
        for (int k = 0; k < sz; ++k)
        for (int j = 0; j < sy; ++j)
        for (int i = 0; i < ng; ++i)
        {
            int idg = F.I(i, j, k), idm = F.I(2 * ng - 1 - i, j, k);
            F.rho[idg] = F.rho[idm];
            F.u[idg] = -F.u[idm];
            F.v[idg] = F.v[idm];
            F.w[idg] = F.w[idm];
            F.p[idg] = F.p[idm];
        }
    else if (face == XMAX)
        for (int k = 0; k < sz; ++k)
            for (int j = 0; j < sy; ++j)
                for (int i = sx - ng; i < sx; ++i)
                {
                    int idg = F.I(i, j, k), idm = F.I(2 * (sx - ng) - 1 - i, j, k);
                    F.rho[idg] = F.rho[idm];
                    F.u[idg] = -F.u[idm];
                    F.v[idg] = F.v[idm];
                    F.w[idg] = F.w[idm];
                    F.p[idg] = F.p[idm];
                }
    else if (face == YMIN)
        for (int k = 0; k < sz; ++k)
            for (int j = 0; j < ng; ++j)
                for (int i = 0; i < sx; ++i)
                {
                    int idg = F.I(i, j, k), idm = F.I(i, 2 * ng - 1 - j, k);
                    F.rho[idg] = F.rho[idm];
                    F.v[idg] = -F.v[idm];
                    F.u[idg] = F.u[idm];
                    F.w[idg] = F.w[idm];
                    F.p[idg] = F.p[idm];
                }
    else if (face == YMAX)
        for (int k = 0; k < sz; ++k)
            for (int j = sy - ng; j < sy; ++j)
                for (int i = 0; i < sx; ++i)
                {
                    int idg = F.I(i, j, k), idm = F.I(i, 2 * (sy - ng) - 1 - j, k);
                    F.rho[idg] = F.rho[idm];
                    F.v[idg] = -F.v[idm];
                    F.u[idg] = F.u[idm];
                    F.w[idg] = F.w[idm];
                    F.p[idg] = F.p[idm];
                }
    else if (face == ZMIN)
        for (int k = 0; k < ng; ++k)
            for (int j = 0; j < sy; ++j)
                for (int i = 0; i < sx; ++i)
                {
                    int idg = F.I(i, j, k), idm = F.I(i, j, 2 * ng - 1 - k);
                    F.rho[idg] = F.rho[idm];
                    F.w[idg] = -F.w[idm];
                    F.u[idg] = F.u[idm];
                    F.v[idg] = F.v[idm];
                    F.p[idg] = F.p[idm];
                }
    else if (face == ZMAX)
        for (int k = sz - ng; k < sz; ++k)
            for (int j = 0; j < sy; ++j)
                for (int i = 0; i < sx; ++i)
                {
                    int idg = F.I(i, j, k), idm = F.I(i, j, 2 * (sz - ng) - 1 - k);
                    F.rho[idg] = F.rho[idm];
                    F.w[idg] = -F.w[idm];
                    F.u[idg] = F.u[idm];
                    F.v[idg] = F.v[idm];
                    F.p[idg] = F.p[idm];
                }
}

// Symmetry boundary condition implementation
void apply_symmetry_bc(Field3D &F, const LocalDesc &L, int face)
{
    // 实际上与 apply_wall_bc 相同，可以直接调用
    apply_wall_bc(F, L, face);
}

// Outflow boundary condition implementation
void apply_outflow_bc(Field3D &F, const LocalDesc &L, int face)
{
    int ng=L.ngx, sx=L.sx, sy=L.sy, sz=L.sz;

    auto copy=[&](int i1,int j1,int k1,int i2,int j2,int k2){
        int id1=F.I(i1,j1,k1), id2=F.I(i2,j2,k2);
        F.rho[id1]=F.rho[id2];
        F.u[id1]=F.u[id2];
        F.v[id1]=F.v[id2];
        F.w[id1]=F.w[id2];
        F.p[id1]=F.p[id2];
    };

    if (face==XMIN) for(int k=0;k<sz;++k)for(int j=0;j<sy;++j)for(int i=0;i<ng;++i)
        copy(i,j,k,ng,j,k);
    if (face==XMAX) for(int k=0;k<sz;++k)for(int j=0;j<sy;++j)for(int i=sx-ng;i<sx;++i)
        copy(i,j,k,sx-ng-1,j,k);
    if (face==YMIN) for(int k=0;k<sz;++k)for(int j=0;j<ng;++j)for(int i=0;i<sx;++i)
        copy(i,j,k,i,ng,k);
    if (face==YMAX) for(int k=0;k<sz;++k)for(int j=sy-ng;j<sy;++j)for(int i=0;i<sx;++i)
        copy(i,j,k,i,sy-ng-1,k);
    if (face==ZMIN) for(int k=0;k<ng;++k)for(int j=0;j<sy;++j)for(int i=0;i<sx;++i)
        copy(i,j,k,i,j,ng);
    if (face==ZMAX) for(int k=sz-ng;k<sz;++k)for(int j=0;j<sy;++j)for(int i=0;i<sx;++i)
        copy(i,j,k,i,j,sz-ng-1);
}

// Inflow boundary condition implementation
void apply_inflow_bc(Field3D &F, const LocalDesc &L, int face)
{
    int ng=L.ngx, sx=L.sx, sy=L.sy, sz=L.sz;
    double rho0=1.0, u0=1.0, v0=0.0, w0=0.0, p0=1.0, gamma=1.4;

    double E0 = p0/(gamma-1.0) + 0.5*rho0*(u0*u0+v0*v0+w0*w0);
    auto fill=[&](int i,int j,int k){
        int id=F.I(i,j,k);
        F.rho[id]=rho0;
        F.u[id]=u0; F.v[id]=v0; F.w[id]=w0;
        F.p[id]=p0;
        F.E[id]=E0;
    };

    if (face==XMIN) for(int k=0;k<sz;++k)for(int j=0;j<sy;++j)for(int i=0;i<ng;++i) fill(i,j,k);
    if (face==XMAX) for(int k=0;k<sz;++k)for(int j=0;j<sy;++j)for(int i=sx-ng;i<sx;++i) fill(i,j,k);
    if (face==YMIN) for(int k=0;k<sz;++k)for(int j=0;j<ng;++j)for(int i=0;i<sx;++i) fill(i,j,k);
    if (face==YMAX) for(int k=0;k<sz;++k)for(int j=sy-ng;j<sy;++j)for(int i=0;i<sx;++i) fill(i,j,k);
    if (face==ZMIN) for(int k=0;k<ng;++k)for(int j=0;j<sy;++j)for(int i=0;i<sx;++i) fill(i,j,k);
    if (face==ZMAX) for(int k=sz-ng;k<sz;++k)for(int j=0;j<sy;++j)for(int i=0;i<sx;++i) fill(i,j,k);
}

// -----------------------------------------
// Periodic boundary condition implementation
// -----------------------------------------
void apply_periodic_bc(Field3D &F, const LocalDesc &L, const CartDecomp &C, int face)
{
    const int ngx=L.ngx, ngy=L.ngy, ngz=L.ngz;
    const int nx=L.nx, ny=L.ny, nz=L.nz;
    const int sx=L.sx, sy=L.sy, sz=L.sz;
    const int tag_base = 400 + face * 10;

    // --- 选择方向 ---
    int main_dir = face / 2;   // 0:x, 1:y, 2:z
    bool is_max  = (face % 2 == 1);

    // --- 对应邻居进程 ---
    int nbr_send, nbr_recv;
    if (main_dir == 0) {
        nbr_send = is_max ? L.nbr_xp : L.nbr_xm;
    } else if (main_dir == 1) {
        nbr_send = is_max ? L.nbr_yp : L.nbr_ym;
    } else {
        nbr_send = is_max ? L.nbr_zp : L.nbr_zm;
    }
    nbr_recv = nbr_send; // 周期边界发送和接收进程相同

    // --- 通信尺寸 ---
    int ghost = (main_dir == 0) ? ngx : (main_dir == 1 ? ngy : ngz);
    int n1 = (main_dir == 0) ? ny : (main_dir == 1 ? nx : nx);
    int n2 = (main_dir == 0) ? nz : (main_dir == 1 ? nz : ny);
    int nvar = 5; // rho,u,v,w,p

    int buf_size = ghost * n1 * n2 * nvar;
    std::vector<double> sendbuf(buf_size), recvbuf(buf_size);

    // --- 打包函数 ---
    auto pack = [&](bool max_side) {
        int p = 0;
        for (int kk = 0; kk < n2; ++kk)
        for (int jj = 0; jj < n1; ++jj)
        for (int ii = 0; ii < ghost; ++ii)
        {
            int i,j,k;
            if (main_dir == 0) { // X
                i = (max_side ? ngx+nx-ghost+ii-1 : ngx+ii+1); // 修改处，即不传输边界点
                j = ngy + jj;
                k = ngz + kk;
            } else if (main_dir == 1) { // Y
                i = ngx + jj;
                j = (max_side ? ngy+ny-ghost+ii-1 : ngy+ii+1);
                k = ngz + kk;
            } else { // Z
                i = ngx + jj;
                j = ngy + kk;
                k = (max_side ? ngz+nz-ghost+ii-1 : ngz+ii+1);
            }
            int id = F.I(i,j,k);
            sendbuf[p++] = F.rho[id];
            sendbuf[p++] = F.u[id];
            sendbuf[p++] = F.v[id];
            sendbuf[p++] = F.w[id];
            sendbuf[p++] = F.p[id];
        }
        assert(p == buf_size);
    };

    // --- 解包函数 ---
    auto unpack = [&](bool max_side) {
        int p = 0;
        for (int kk = 0; kk < n2; ++kk)
        for (int jj = 0; jj < n1; ++jj)
        for (int ii = 0; ii < ghost; ++ii)
        {
            int i,j,k;
            if (main_dir == 0) {
                i = (max_side ? ngx+nx+ii : ii);
                j = ngy + jj;
                k = ngz + kk;
            } else if (main_dir == 1) {
                i = ngx + jj;
                j = (max_side ? ngy+ny+ii : ii);
                k = ngz + kk;
            } else {
                i = ngx + jj;
                j = ngy + kk;
                k = (max_side ? ngz+nz+ii : ii);
            }
            int id = F.I(i,j,k);
            F.rho[id] = recvbuf[p++];
            F.u[id]   = recvbuf[p++];
            F.v[id]   = recvbuf[p++];
            F.w[id]   = recvbuf[p++];
            F.p[id]   = recvbuf[p++];
        }
        assert(p == buf_size);
    };

    // --- 通信流程 ---
    pack(is_max);
    MPI_Request reqs[2];
    MPI_Irecv(recvbuf.data(), buf_size, MPI_DOUBLE, nbr_recv, tag_base+1, C.cart_comm, &reqs[0]);
    MPI_Isend(sendbuf.data(), buf_size, MPI_DOUBLE, nbr_send, tag_base,   C.cart_comm, &reqs[1]);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    unpack(is_max);
}
