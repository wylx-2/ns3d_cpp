#include "field_structures.h"
#include <mpi.h>
#include <cmath>

double compute_timestep(Field3D &F, const GridDesc &G, const SolverParams &P){
    // Take the most restrictive CFL across x/y/z to avoid overshooting on stretched grids.
    double dt_local = 1e9;
    const int nx = F.L.nx, ny = F.L.ny, nz = F.L.nz;
    const int ngx = F.L.ngx, ngy = F.L.ngy, ngz = F.L.ngz;
    const double dx = G.dx, dy = G.dy, dz = G.dz;
    double dt_global; 

    for(int k=ngz;k<ngz+nz;++k)
    for(int j=ngy;j<ngy+ny;++j)
    for(int i=ngx;i<ngx+nx;++i){
        int id = F.I(i,j,k);
        double rho = F.rho[id];
        double u = F.u[id];
        double v = F.v[id];
        double w = F.w[id];
        double p = F.p[id];
        double a = std::sqrt(P.gamma * p / std::max(rho, 1e-14));

        double dt_x = dx / (std::fabs(u) + a);
        double dt_y = dy / (std::fabs(v) + a);
        double dt_z = dz / (std::fabs(w) + a);

        dt_local = std::min(dt_local, std::min(dt_x, std::min(dt_y, dt_z)));
    }

    MPI_Allreduce(&dt_local,&dt_global,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    return P.cfl*dt_global;
}