#include "field_structures.h"
#include <cmath>
#include <iostream>

void initialize_riemann_2d(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const double gamma = P.gamma;
    const double x_mid = 0.5, y_mid = 0.5;

    for (int k = 0; k < L.sz; ++k) {
        double z = (L.oz + k - L.ngz + 0.5) * G.dz;
        if (z > G.dz * 1.5) continue;  // 2D test plane only (optional)

        for (int j = 0; j < L.sy; ++j) {
            double y = (L.oy + j - L.ngy + 0.5) * G.dy;
            for (int i = 0; i < L.sx; ++i) {
                double x = (L.ox + i - L.ngx + 0.5) * G.dx;

                double rho, u, v, w, p;
                w = 0.0;

                // Select quadrant
                if (x >= x_mid && y >= y_mid) {          // Region I
                    rho = 1.0; u = 0.0; v = 0.0; p = 1.0;
                } else if (x < x_mid && y >= y_mid) {    // Region II
                    rho = 0.5197; u = -0.7259; v = 0.0; p = 0.4;
                } else if (x < x_mid && y < y_mid) {     // Region III
                    rho = 0.1072; u = -0.7259; v = -0.7259; p = 0.0439;
                } else {                                 // Region IV
                    rho = 0.2579; u = 0.0; v = -0.7259; p = 0.15;
                }

                double E = p/(gamma-1.0) + 0.5*rho*(u*u+v*v+w*w);
                int id = F.I(i,j,k);
                F.rho[id]  = rho;
                F.rhou[id] = rho*u;
                F.rhov[id] = rho*v;
                F.rhow[id] = rho*w;
                F.E[id]    = E;
                F.p[id]    = p;
                F.u[id]    = u;
                F.v[id]    = v;
                F.w[id]    = w;
                F.T[id]    = p/(rho*P.Rgas);
            }
        }
    }
}

void initialize_uniform_field(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    // init some field (e.g., constant density + small velocity perturbation in interior)
    // initialize with consistent total energy so pressure is positive
    LocalDesc &L = F.L;
    const double gamma = P.gamma;
    const double p0 = 1.0; // reference pressure
    for (int k=L.ngz; k<L.ngz+L.nz; ++k)
    for (int j=L.ngy; j<L.ngy+L.ny; ++j)
    for (int i=L.ngx; i<L.ngx+L.nx; ++i) {
        int id = F.I(i,j,k);
        double rho = 1.0;
        double u = 1.0, v = 1.0, w = 1.0;
        double kinetic = 0.5 * rho * (u*u + v*v + w*w);
        double eint = p0 / (gamma - 1.0); // rho * e_internal
        double E = eint + kinetic; // total energy (per cell)
        F.rho[id] = rho;
        F.rhou[id] = rho * u;
        F.rhov[id] = rho * v;
        F.rhow[id] = rho * w;
        F.E[id] = E;
    }
}
