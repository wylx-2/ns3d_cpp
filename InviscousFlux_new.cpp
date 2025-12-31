#include "field_structures.h"
#include "ns3d_func.h"

void du_inviscous(Field3D &F, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    int nx = L.nx, ny = L.ny, nz = L.nz;
    int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;
    int sz = L.sz, sy = L.sy, sx = L.sx;
    const int VAR = 5; // 变量个数：rho, rhou, rhov, rhow, E
    int stencil = P.stencil;
    int mid = (stencil - 1) / 2;

    //----------------- x direction ------------------------------
    for (int k = ngz; k < ngz + nz; ++k) {
    for (int j = ngy; j < ngy + ny; ++j) {
        std::vector<std::vector<double>> F_plus(sx, std::vector<double>(VAR));
        std::vector<std::vector<double>> F_minus(sx, std::vector<double>(VAR));
        for (int i = 0; i < sx; ++i) {
            int id = F.I(i, j, k);
            // FVS
            switch (P.fvs_type)
            {
            case SolverParams::FVS_Type::StagerWarming:
                split_StagerWarming(F,
                                    /*out*/F_plus[i], F_minus[i],
                                    /*in*/ id, 1, 0, 0,
                                    P);
                break;
            case SolverParams::FVS_Type::LaxFriedrichs:
                split_LaxFriedrichs(F,
                                    /*out*/F_plus[i], F_minus[i],
                                    /*in*/ id, 1, 0, 0,
                                    P);
                break;
            default:
                std::cout << "Unknown FVS type!" << std::endl;
                break;
            }
        }
        // compute reconstructed flux at faces
        for (int i = ngx -1; i < ngx + nx; ++i) {
            std::vector<double> wface(VAR, 0.0);
            for (int n = 0; n < VAR; ++n) {
                std::vector<double> wplus(stencil), wminus(stencil);
                for (int m = 0; m < stencil; ++m) {
                    wplus[m]  = F_plus[i + (m - mid)][n];
                    wminus[m] = F_minus[i + (m - mid)][n];
                }
                double plus_face = reconstruct_select(wplus, +1.0, P);
                double minus_face = reconstruct_select(wminus, -1.0, P);
                wface[n] = plus_face + minus_face;
            }
            // component-wise result is wface in conservative flux-like variables
            int fid = idx_fx(i, j, k, L);
            F.flux_fx_mass[fid] = wface[0];
            F.flux_fx_momx[fid] = wface[1];
            F.flux_fx_momy[fid] = wface[2];
            F.flux_fx_momz[fid] = wface[3];
            F.flux_fx_E[fid]    = wface[4];
        }
    }}

    // ----------------- y direction ------------------------------
    for (int k = ngz; k < ngz + nz; ++k) {
    for (int i = ngx; i < ngx + nx; ++i) {
        std::vector<std::vector<double>> F_plus(sy, std::vector<double>(VAR));
        std::vector<std::vector<double>> F_minus(sy, std::vector<double>(VAR));
        for (int j = 0; j < sy; ++j) {
            int id = F.I(i, j, k);
            // FVS
            switch (P.fvs_type)
            {
            case SolverParams::FVS_Type::StagerWarming:
                split_StagerWarming(F,
                                    /*out*/F_plus[j], F_minus[j],
                                    /*in*/ id, 0, 1, 0,
                                    P);
                break;
            case SolverParams::FVS_Type::LaxFriedrichs:
                split_LaxFriedrichs(F,
                                    /*out*/F_plus[j], F_minus[j],
                                    /*in*/ id, 0, 1, 0,
                                    P);
                break;
            default:
                std::cout << "Unknown FVS type!" << std::endl;
                break;
            }
        }
        // compute reconstructed flux at faces
        for (int j = ngy -1; j < ngy + ny; ++j) {
            std::vector<double> wface(VAR, 0.0);
            for (int n = 0; n < VAR; ++n) {
                std::vector<double> wplus(stencil), wminus(stencil);
                for (int m = 0; m < stencil; ++m) {
                    wplus[m]  = F_plus[j + (m - mid)][n];
                    wminus[m] = F_minus[j + (m - mid)][n];
                }
                double plus_face = reconstruct_select(wplus, +1.0, P);
                double minus_face = reconstruct_select(wminus, -1.0, P);
                wface[n] = plus_face + minus_face;
            }
            // component-wise result is wface in conservative flux-like variables
            int fid = idx_fy(i, j, k, L);
            F.flux_fy_mass[fid] = wface[0];
            F.flux_fy_momx[fid] = wface[1];
            F.flux_fy_momy[fid] = wface[2];
            F.flux_fy_momz[fid] = wface[3];
            F.flux_fy_E[fid]    = wface[4];
        }
    }}
    // ----------------- z direction ------------------------------
    for (int j = ngy; j < ngy + ny; ++j) {
    for (int i = ngx; i < ngx + nx; ++i) {
        std::vector<std::vector<double>> F_plus(sz, std::vector<double>(VAR));
        std::vector<std::vector<double>> F_minus(sz, std::vector<double>(VAR));
        for (int k = 0; k < sz; ++k) {
            int id = F.I(i, j, k);
            // FVS
            switch (P.fvs_type)
            {
            case SolverParams::FVS_Type::StagerWarming:
                split_StagerWarming(F,
                                    /*out*/F_plus[k], F_minus[k],
                                    /*in*/ id, 0, 0, 1,
                                    P);
                break;
            case SolverParams::FVS_Type::LaxFriedrichs:
                split_LaxFriedrichs(F,
                                    /*out*/F_plus[k], F_minus[k],
                                    /*in*/ id, 0, 0, 1,
                                    P);
                break;
            default:
                std::cout << "Unknown FVS type!" << std::endl;
                break;
            }
        }
        // compute reconstructed flux at faces
        for (int k = ngz -1; k < ngz + nz; ++k) {
            std::vector<double> wface(VAR, 0.0);
            for (int n = 0; n < VAR; ++n) {
                std::vector<double> wplus(stencil), wminus(stencil);
                for (int m = 0; m < stencil; ++m) {
                    wplus[m]  = F_plus[k + (m - mid)][n];
                    wminus[m] = F_minus[k + (m - mid)][n];
                }
                double plus_face = reconstruct_select(wplus, +1.0, P);
                double minus_face = reconstruct_select(wminus, -1.0, P);
                wface[n] = plus_face + minus_face;
            }
            // component-wise result is wface in conservative flux-like variables
            int fid = idx_fz(i, j, k, L);
            F.flux_fz_mass[fid] = wface[0];
            F.flux_fz_momx[fid] = wface[1];
            F.flux_fz_momy[fid] = wface[2];
            F.flux_fz_momz[fid] = wface[3];
            F.flux_fz_E[fid]    = wface[4];
        }
    }}

}

void du_inviscous_character(Field3D &F, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    int nx = L.nx, ny = L.ny, nz = L.nz;
    int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;
    int sz = L.sz, sy = L.sy, sx = L.sx;
    const int VAR = 5; // 变量个数：rho, rhou, rhov, rhow, E
    int stencil = P.stencil;
    int mid = (stencil - 1) / 2;
    //----------------- x direction ------------------------------
    for (int k = ngz; k < ngz + nz; ++k) {
    for (int j = ngy; j < ngy + ny; ++j) {
        std::vector<std::vector<double>> F_plus(sx, std::vector<double>(VAR));
        std::vector<std::vector<double>> F_minus(sx, std::vector<double>(VAR));
        for (int i = 0; i < sx; ++i) {
            int id = F.I(i, j, k);
            // FVS
            switch (P.fvs_type)
            {
            case SolverParams::FVS_Type::StagerWarming:
                split_StagerWarming(F,
                                    /*out*/F_plus[i], F_minus[i],
                                    /*in*/ id, 1, 0, 0,
                                    P);
                break;
            case SolverParams::FVS_Type::LaxFriedrichs:
                split_LaxFriedrichs(F,
                                    /*out*/F_plus[i], F_minus[i],
                                    /*in*/ id, 1, 0, 0,
                                    P);
                break;
            default:
                std::cout << "Unknown FVS type!" << std::endl;
                break;
            }
        }
        // compute reconstructed flux at faces
        for (int i = ngx -1; i < ngx + nx; ++i) {
            std::vector<std::vector<double>> LF_plus(VAR, std::vector<double>(stencil));
            std::vector<std::vector<double>> LF_minus(VAR, std::vector<double>(stencil));
            std::vector<double> wflux_char(VAR, 0.0);
            std::vector<double> wface(VAR, 0.0);
            double Lmat[VAR][VAR], Rmat[VAR][VAR], lambar[VAR];
            double ul[5], ur[5], u_ref[5];
            // gather left/right states for Roe average
            int idx_left = F.I(i, j, k);
            int idx_right = F.I(i + 1, j, k);
            ul[0] = F.rho[idx_left];
            ul[1] = F.u[idx_left];
            ul[2] = F.v[idx_left];
            ul[3] = F.w[idx_left];
            ul[4] = F.p[idx_left];
            ur[0] = F.rho[idx_right];
            ur[1] = F.u[idx_right];
            ur[2] = F.v[idx_right];
            ur[3] = F.w[idx_right];
            ur[4] = F.p[idx_right];
            computeRoeAveragedState_new(u_ref, ul, ur, P.gamma);
            build_eigen_matrices_new(u_ref, 1, 0, 0, P.gamma, Lmat, Rmat, lambar);
            for (int m = 0; m < stencil; ++m) {
                int ii = i + (m - mid);
                for (int n = 0; n < VAR; ++n) {
                    double sumw = 0.0, sumLU = 0.0;
                    for (int r = 0; r < VAR; ++r) {
                        LF_plus[n][m]  += Lmat[n][r] * F_plus[ii][r];
                        LF_minus[n][m] += Lmat[n][r] * F_minus[ii][r];
                    }
                }
            }
            
            for (int n = 0; n < VAR; ++n) {
                std::vector<double> wtplus(stencil), wtminus(stencil);
                for (int m = 0; m < stencil; ++m) {
                    wtplus[m] = LF_plus[n][m];
                    wtminus[m] = LF_minus[n][m];
                }
                double plus_face = reconstruct_select(wtplus, +1.0, P);
                double minus_face = reconstruct_select(wtminus, -1.0, P);
                wflux_char[n] = plus_face + minus_face;
            }

            // transform back to conservative flux via Fflux = R * wflux_char
            for (int n = 0; n < VAR; ++n) {
                double sum = 0.0;
                for (int r = 0; r < VAR; ++r) sum += Rmat[n][r] * wflux_char[r];
                wface[n] = sum;
            }
            
            int fid = idx_fx(i, j, k, L);
            F.flux_fx_mass[fid] = wface[0];
            F.flux_fx_momx[fid] = wface[1];
            F.flux_fx_momy[fid] = wface[2];
            F.flux_fx_momz[fid] = wface[3];
            F.flux_fx_E[fid]    = wface[4];
        }
    }}

    // ----------------- y direction ------------------------------
    for (int k = ngz; k < ngz + nz; ++k) {
    for (int i = ngx; i < ngx + nx; ++i) {
        std::vector<std::vector<double>> F_plus(sy, std::vector<double>(VAR));
        std::vector<std::vector<double>> F_minus(sy, std::vector<double>(VAR));
        for (int j = 0; j < sy; ++j) {
            int id = F.I(i, j, k);
            // FVS
            switch (P.fvs_type)
            {
            case SolverParams::FVS_Type::StagerWarming:
                split_StagerWarming(F,
                                    /*out*/F_plus[j], F_minus[j],
                                    /*in*/ id, 0, 1, 0,
                                    P);
                break;
            case SolverParams::FVS_Type::LaxFriedrichs:
                split_LaxFriedrichs(F,
                                    /*out*/F_plus[j], F_minus[j],
                                    /*in*/ id, 0, 1, 0,
                                    P);
                break;
            default:
                std::cout << "Unknown FVS type!" << std::endl;
                break;
            }
        }
        // compute reconstructed flux at faces
        for (int j = ngy -1; j < ngy + ny; ++j) {
            std::vector<std::vector<double>> LF_plus(VAR, std::vector<double>(stencil));
            std::vector<std::vector<double>> LF_minus(VAR, std::vector<double>(stencil));
            std::vector<double> wflux_char(VAR, 0.0);
            std::vector<double> wface(VAR, 0.0);
            double Lmat[VAR][VAR], Rmat[VAR][VAR], lambar[VAR];
            double ul[5], ur[5], u_ref[5];
            // gather left/right states for Roe average
            int idx_left = F.I(i, j, k);
            int idx_right = F.I(i , j+1, k);
            ul[0] = F.rho[idx_left];
            ul[1] = F.u[idx_left];
            ul[2] = F.v[idx_left];
            ul[3] = F.w[idx_left];
            ul[4] = F.p[idx_left];
            ur[0] = F.rho[idx_right];
            ur[1] = F.u[idx_right];
            ur[2] = F.v[idx_right];
            ur[3] = F.w[idx_right];
            ur[4] = F.p[idx_right];
            computeRoeAveragedState_new(u_ref, ul, ur, P.gamma);
            build_eigen_matrices_new(u_ref, 0, 1, 0, P.gamma, Lmat, Rmat, lambar);
            for (int m = 0; m < stencil; ++m) {
                int jj = j + (m - mid);
                for (int n = 0; n < VAR; ++n) {
                    double sumw = 0.0, sumLU = 0.0;
                    for (int r = 0; r < VAR; ++r) {
                        LF_plus[n][m]  += Lmat[n][r] * F_plus[jj][r];
                        LF_minus[n][m] += Lmat[n][r] * F_minus[jj][r];
                    }
                }
            }
            
            for (int n = 0; n < VAR; ++n) {
                std::vector<double> wtplus(stencil), wtminus(stencil);
                for (int m = 0; m < stencil; ++m) {
                    wtplus[m] = LF_plus[n][m];
                    wtminus[m] = LF_minus[n][m];
                }
                double plus_face = reconstruct_select(wtplus, +1.0, P);
                double minus_face = reconstruct_select(wtminus, -1.0, P);
                wflux_char[n] = plus_face + minus_face;
            }

            // transform back to conservative flux via Fflux = R * wflux_char
            for (int n = 0; n < VAR; ++n) {
                double sum = 0.0;
                for (int r = 0; r < VAR; ++r) sum += Rmat[n][r] * wflux_char[r];
                wface[n] = sum;
            }
            
            int fid = idx_fy(i, j, k, L);
            F.flux_fy_mass[fid] = wface[0];
            F.flux_fy_momx[fid] = wface[1];
            F.flux_fy_momy[fid] = wface[2];
            F.flux_fy_momz[fid] = wface[3];
            F.flux_fy_E[fid]    = wface[4];
        }
    }}
    // ----------------- z direction ------------------------------
    for (int j = ngy; j < ngy + ny; ++j) {
    for (int i = ngx; i < ngx + nx; ++i) {
        std::vector<std::vector<double>> F_plus(sz, std::vector<double>(VAR));
        std::vector<std::vector<double>> F_minus(sz, std::vector<double>(VAR));
        for (int k = 0; k < sz; ++k) {
            int id = F.I(i, j, k);
            // FVS
            switch (P.fvs_type)
            {
            case SolverParams::FVS_Type::StagerWarming:
                split_StagerWarming(F,
                                    /*out*/F_plus[k], F_minus[k],
                                    /*in*/ id, 0, 0, 1,
                                    P);
                break;
            case SolverParams::FVS_Type::LaxFriedrichs:
                split_LaxFriedrichs(F,
                                    /*out*/F_plus[k], F_minus[k],
                                    /*in*/ id, 0, 0, 1,
                                    P);
                break;
            default:
                std::cout << "Unknown FVS type!" << std::endl;
                break;
            }
        }
        // compute reconstructed flux at faces
        for (int k = ngz -1; k < ngz + nz; ++k) {
            std::vector<std::vector<double>> LF_plus(VAR, std::vector<double>(stencil));
            std::vector<std::vector<double>> LF_minus(VAR, std::vector<double>(stencil));
            std::vector<double> wflux_char(VAR, 0.0);
            std::vector<double> wface(VAR, 0.0);
            double Lmat[VAR][VAR], Rmat[VAR][VAR], lambar[VAR];
            double ul[5], ur[5], u_ref[5];
            // gather left/right states for Roe average
            int idx_left = F.I(i, j, k);
            int idx_right = F.I(i , j, k+1);
            ul[0] = F.rho[idx_left];
            ul[1] = F.u[idx_left];
            ul[2] = F.v[idx_left];
            ul[3] = F.w[idx_left];
            ul[4] = F.p[idx_left];
            ur[0] = F.rho[idx_right];
            ur[1] = F.u[idx_right];
            ur[2] = F.v[idx_right];
            ur[3] = F.w[idx_right];
            ur[4] = F.p[idx_right];
            computeRoeAveragedState_new(u_ref, ul, ur, P.gamma);
            build_eigen_matrices_new(u_ref, 0, 0, 1, P.gamma, Lmat, Rmat, lambar);
            for (int m = 0; m < stencil; ++m) {
                int kk = k + (m - mid);
                for (int n = 0; n < VAR; ++n) {
                    double sumw = 0.0, sumLU = 0.0;
                    for (int r = 0; r < VAR; ++r) {
                        LF_plus[n][m]  += Lmat[n][r] * F_plus[kk][r];
                        LF_minus[n][m] += Lmat[n][r] * F_minus[kk][r];
                    }
                }
            }
            
            for (int n = 0; n < VAR; ++n) {
                std::vector<double> wtplus(stencil), wtminus(stencil);
                for (int m = 0; m < stencil; ++m) {
                    wtplus[m] = LF_plus[n][m];
                    wtminus[m] = LF_minus[n][m];
                }
                double plus_face = reconstruct_select(wtplus, +1.0, P);
                double minus_face = reconstruct_select(wtminus, -1.0, P);
                wflux_char[n] = plus_face + minus_face;
            }

            // transform back to conservative flux via Fflux = R * wflux_char
            for (int n = 0; n < VAR; ++n) {
                double sum = 0.0;
                for (int r = 0; r < VAR; ++r) sum += Rmat[n][r] * wflux_char[r];
                wface[n] = sum;
            }
            
            int fid = idx_fz(i, j, k, L);
            F.flux_fz_mass[fid] = wface[0];
            F.flux_fz_momx[fid] = wface[1];
            F.flux_fz_momy[fid] = wface[2];
            F.flux_fz_momz[fid] = wface[3];
            F.flux_fz_E[fid]    = wface[4];
        }
    }}

}

void split_StagerWarming(const Field3D &F,
                         std::vector<double> &F_plus,
                         std::vector<double> &F_minus,
                         int id,int n1, int n2, int n3,
                         const SolverParams &P)
{
    const double gamma = P.gamma;
    double rho = F.rho[id];
    double u = F.u[id];
    double v = F.v[id];
    double w = F.w[id];
    double E = F.E[id];
    double p = F.p[id];
    double c = F.c[id];

    // Eigenvalues
    double un = u*n1 + v*n2 + w*n3; // normal velocity
    double lambda1 = un - c;
    double lambda2 = un;
    double lambda3 = un + c;

    // Split eigenvalues
    double lambda1_plus = 0.5 * (lambda1 + std::abs(lambda1));
    double lambda1_minus = 0.5 * (lambda1 - std::abs(lambda1));
    double lambda2_plus = 0.5 * (lambda2 + std::abs(lambda2));
    double lambda2_minus = 0.5 * (lambda2 - std::abs(lambda2));
    double lambda3_plus = 0.5 * (lambda3 + std::abs(lambda3));
    double lambda3_minus = 0.5 * (lambda3 - std::abs(lambda3));
    double S1_p = lambda1_plus + lambda3_plus;
    double S1_m = lambda1_minus + lambda3_minus;
    double S2_p = lambda3_plus - lambda1_plus;
    double S2_m = lambda3_minus - lambda1_minus;

    // Compute flux components
    double H = (E + p) / rho; // total enthalpy
    double ga1 = rho / (2.0 * gamma);
    double ga2 = 2.0 * (gamma - 1.0);
    double V = 0.5 * (u*u + v*v + w*w);

    F_plus[0]  = ga1 * (S1_p + ga2 * lambda2_plus);
    F_plus[1]  = ga1 * (S1_p * u + ga2 * lambda2_plus * u + S2_p * n1 * c);
    F_plus[2]  = ga1 * (S1_p * v + ga2 * lambda2_plus * v + S2_p * n2 * c);
    F_plus[3]  = ga1 * (S1_p * w + ga2 * lambda2_plus * w + S2_p * n3 * c);
    F_plus[4]  = ga1 * (S1_p * H + ga2 * lambda2_plus * V + S2_p * un * c);

    F_minus[0] = ga1 * (S1_m + ga2 * lambda2_minus);
    F_minus[1] = ga1 * (S1_m * u + ga2 * lambda2_minus * u + S2_m * n1 * c);
    F_minus[2] = ga1 * (S1_m * v + ga2 * lambda2_minus * v + S2_m * n2 * c);
    F_minus[3] = ga1 * (S1_m * w + ga2 * lambda2_minus * w + S2_m * n3 * c);
    F_minus[4] = ga1 * (S1_m * H + ga2 * lambda2_minus * V + S2_m * un * c);
}

void split_LaxFriedrichs(const Field3D &F,
                         std::vector<double> &F_plus,
                         std::vector<double> &F_minus,
                         int id,int n1, int n2, int n3,
                         const SolverParams &P)
{
    const double gamma = P.gamma;
    double rho = F.rho[id];
    double u = F.u[id];
    double v = F.v[id];
    double w = F.w[id];
    double E = F.E[id];
    double p = F.p[id];
    double c = F.c[id];

    // normal velocity
    double un = u*n1 + v*n2 + w*n3; 
    // maximum wave speed
    double a_max = std::abs(un) + c;

    // Compute flux vector
    double F1 = rho * un;
    double F2 = rho * u * un + p * n1;
    double F3 = rho * v * un + p * n2;
    double F4 = rho * w * un + p * n3;
    double F5 = (E + p) * un;

    // Lax-Friedrichs splitting
    F_plus[0]  = 0.5 * (F1 + a_max * rho);
    F_plus[1]  = 0.5 * (F2 + a_max * rho * u);
    F_plus[2]  = 0.5 * (F3 + a_max * rho * v);
    F_plus[3]  = 0.5 * (F4 + a_max * rho * w);
    F_plus[4]  = 0.5 * (F5 + a_max * E);

    F_minus[0] = 0.5 * (F1 - a_max * rho);
    F_minus[1] = 0.5 * (F2 - a_max * rho * u);
    F_minus[2] = 0.5 * (F3 - a_max * rho * v);
    F_minus[3] = 0.5 * (F4 - a_max * rho * w);
    F_minus[4] = 0.5 * (F5 - a_max * E);
}

static void build_eigen_matrices_new(const double u[5],
                                 double nx, double ny, double nz,
                                 double gamma,
                                 double Lmat[5][5], double Rmat[5][5],
                                 double lambar[5])
{
    // first compute Roe averaged quantities
    double rho0 = u[0];
    double u0 = u[1];
    double v0 = u[2];
    double w0 = u[3];
    double h0 = u[4];        // !!!!!!!
    double V = 0.5*(u0*u0 + v0*v0 + w0*w0);
    double un = u0*nx + v0*ny + w0*nz;
    double c = std::sqrt((gamma - 1.0) * (h0 - V)); // speed of sound

    lambar[0] = un - c;
    lambar[1] = un;
    lambar[2] = un;
    lambar[3] = un;
    lambar[4] = un + c;

    // Right eigenvector matrix R
    // Rmat[.][0]
    Rmat[0][0] = 1.0;
    Rmat[1][0] = u0 - c * nx;
    Rmat[2][0] = v0 - c * ny;
    Rmat[3][0] = w0 - c * nz;
    Rmat[4][0] = h0 - un * c;
    // Rmat[.][1]
    Rmat[0][1] = 1.0;
    Rmat[1][1] = u0;
    Rmat[2][1] = v0;
    Rmat[3][1] = w0;
    Rmat[4][1] = V;
    // Rmat[.][2]
    Rmat[0][2] = 0.0;
    Rmat[1][2] = nz;
    Rmat[2][2] = nx;
    Rmat[3][2] = ny;
    Rmat[4][2] = u0 * nz + v0 * nx + w0 * ny;
    // Rmat[.][3]
    Rmat[0][3] = 0.0;
    Rmat[1][3] = ny;
    Rmat[2][3] = nz;
    Rmat[3][3] = nx;
    Rmat[4][3] = u0 * ny + v0 * nz + w0 * nx;
    // Rmat[.][4]
    Rmat[0][4] = 1.0;
    Rmat[1][4] = u0 + c * nx;
    Rmat[2][4] = v0 + c * ny;
    Rmat[3][4] = w0 + c * nz;
    Rmat[4][4] = h0 + un * c;

    // Left eigenvector matrix L
    double chi = 0.5 / c;
    double K = (gamma - 1.0) /(c * c);
    double K1 = K/2;

    // Lmat[0][.]
    Lmat[0][0] = K1 * V + chi * un;
    Lmat[0][1] = -chi * nx - K1 * u0;
    Lmat[0][2] = -chi * ny - K1 * v0;
    Lmat[0][3] = -chi * nz - K1 * w0;
    Lmat[0][4] = K1;

    // Lmat[1][.]
    Lmat[1][0] = 1.0 - K * V;
    Lmat[1][1] = K * u0;
    Lmat[1][2] = K * v0;
    Lmat[1][3] = K * w0;
    Lmat[1][4] = -K;

    // Lmat[2][.]
    Lmat[2][0] = -(u0 * nz + v0 * nx + w0 * ny);
    Lmat[2][1] = nz;
    Lmat[2][2] = nx;
    Lmat[2][3] = ny;
    Lmat[2][4] = 0.0;

    // Lmat[3][.]
    Lmat[3][0] = -(u0 * ny + v0 * nz + w0 * nx);
    Lmat[3][1] = ny;
    Lmat[3][2] = nz;
    Lmat[3][3] = nx;
    Lmat[3][4] = 0.0;

    // Lmat[4][.]
    Lmat[4][0] = K1 * V - chi * un;
    Lmat[4][1] = chi * nx - K1 * u0;
    Lmat[4][2] = chi * ny - K1 * v0;
    Lmat[4][3] = chi * nz - K1 * w0;
    Lmat[4][4] = K1;
}

// Roe平均
void computeRoeAveragedState_new(double u_roe[5],
                             const double ul[5], const double ur[5],
                             double gamma)
{
    // 提取左状态变量
    double rho_L = ul[0];
    double u_L = ul[1];
    double v_L = ul[2];
    double w_L = ul[3];
    double p_L = ul[4];
    double E_L = rho_L * (0.5 * (u_L*u_L + v_L*v_L + w_L*w_L) + p_L / ((gamma - 1.0) * rho_L));
    // 提取右状态变量
    double rho_R = ur[0];
    double u_R = ur[1];
    double v_R = ur[2];
    double w_R = ur[3];
    double p_R = ur[4];
    double E_R = rho_R * (0.5 * (u_R*u_R + v_R*v_R + w_R*w_R) + p_R / ((gamma - 1.0) * rho_R));

    // 计算Roe平均态
    double sqrt_rho_L = std::sqrt(rho_L);
    double sqrt_rho_R = std::sqrt(rho_R);
    u_roe[0] = sqrt_rho_L * sqrt_rho_R;
    double denom = 1.0 / (sqrt_rho_L + sqrt_rho_R);
    u_roe[1] = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * denom;
    u_roe[2] = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) * denom;
    u_roe[3] = (sqrt_rho_L * w_L + sqrt_rho_R * w_R) * denom;
    // Total specific enthalpy H = (E + p) / rho. Here Ul[4] and Ur[4] are the total energy (conserved E).
    double H_L = (E_L + p_L) / rho_L;
    double H_R = (E_R + p_R) / rho_R;
    u_roe[4] = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) * denom;
}