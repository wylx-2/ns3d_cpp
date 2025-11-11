#include "ns3d_func.h"

//==================================================================
// flux calculation module
//==================================================================

// -----------------------------------------------------------------
// ---------   Flux Vector Splitting (FVS) -------------------------
// -----------------------------------------------------------------

// 计算无粘通量
void compute_flux(Field3D &F, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const double gamma = P.gamma;

    for (int k = 0; k < L.sz; ++k) {
        for (int j = 0; j < L.sy; ++j) {
            for (int i = 0; i < L.sx; ++i) {

                int id = F.I(i, j, k);
                double rho = F.rho[id];
                double u = F.u[id];
                double v = F.v[id];
                double w = F.w[id];
                double E = F.E[id];
                double p = F.p[id];

                // X方向物理通量
                F.Fflux_mass[id] = rho * u;
                F.Fflux_momx[id] = rho * u * u + p;
                F.Fflux_momy[id] = rho * u * v;
                F.Fflux_momz[id] = rho * u * w;
                F.Fflux_E[id]    = (E + p) * u;

                // Y方向物理通量
                F.Hflux_mass[id] = rho * v;
                F.Hflux_momx[id] = rho * u * v;
                F.Hflux_momy[id] = rho * v * v + p;
                F.Hflux_momz[id] = rho * v * w;
                F.Hflux_E[id]    = (E + p) * v;

                // Z方向物理通量
                F.Gflux_mass[id] = rho * w;
                F.Gflux_momx[id] = rho * u * w;
                F.Gflux_momy[id] = rho * v * w;
                F.Gflux_momz[id] = rho * w * w + p;
                F.Gflux_E[id]    = (E + p) * w;
            }
        }
    }
}

// FVS main 计算通量的重构值
void computeFVSFluxes(Field3D &F, const SolverParams &P)
{
    compute_flux(F, P);

    const LocalDesc &L = F.L;
    int nx = L.nx, ny = L.ny, nz = L.nz;
    int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;
    const int VAR = 5; // 变量个数：rho, rhou, rhov, rhow, E
    // Use runtime stencil size from SolverParams so different reconstructions
    // (WENO5, C6th, ...) can be selected at runtime.
    int stencil = P.stencil;

    // X方向通量重构
    for (int k = ngz; k < ngz+nz; ++k) {
        for (int j = ngy; j < ngy+ny; ++j) {
            for (int i = ngx-1; i < ngx+nx; ++i) {
                // dynamic 2D arrays: VAR x stencil
                std::vector<std::vector<double>> Ft(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> Ut(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> ut(VAR, std::vector<double>(stencil));

                for (int m = 0; m < stencil; ++m) {
                    int ii = i + (m - (stencil/2)); // 以i为中心的stencil(6点模板为i-2到i+3)
                    int id = F.I(ii, j, k);

                    Ft[0][m] = F.Fflux_mass[id];
                    Ft[1][m] = F.Fflux_momx[id];
                    Ft[2][m] = F.Fflux_momy[id];
                    Ft[3][m] = F.Fflux_momz[id];
                    Ft[4][m] = F.Fflux_E[id];
                    Ut[0][m] = F.rho[id];
                    Ut[1][m] = F.rhou[id];
                    Ut[2][m] = F.rhov[id];
                    Ut[3][m] = F.rhow[id];
                    Ut[4][m] = F.E[id];
                    ut[0][m] = F.rho[id];
                    ut[1][m] = F.u[id];
                    ut[2][m] = F.v[id];
                    ut[3][m] = F.w[id];
                    ut[4][m] = F.p[id];
                }

                std::vector<double> Fface(VAR, 0.0);
                reconstructInviscidFlux(Fface, Ft, Ut, ut, P, /*dim=*/0);

                int fid = idx_fx(i, j, k, L);
                F.flux_fx_mass[fid] = Fface[0];
                F.flux_fx_momx[fid] = Fface[1];
                F.flux_fx_momy[fid] = Fface[2];
                F.flux_fx_momz[fid] = Fface[3];
                F.flux_fx_E[fid]    = Fface[4];
            }
        }
    }

    // Y方向通量重构
    for (int k = ngz; k < ngz+nz; ++k) {
        for (int i = ngx; i < ngx+nx; ++i) {
            for (int j = ngy-1; j < ngy+ny; ++j) {
                // dynamic 2D arrays: VAR x stencil
                std::vector<std::vector<double>> Ft(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> Ut(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> ut(VAR, std::vector<double>(stencil));

                for (int m = 0; m < stencil; ++m) {
                    int jj = j + (m - (stencil/2));
                    int id = F.I(i, jj, k);

                    Ft[0][m] = F.Hflux_mass[id];
                    Ft[1][m] = F.Hflux_momx[id];
                    Ft[2][m] = F.Hflux_momy[id];
                    Ft[3][m] = F.Hflux_momz[id];
                    Ft[4][m] = F.Hflux_E[id];
                    Ut[0][m] = F.rho[id];
                    Ut[1][m] = F.rhou[id];
                    Ut[2][m] = F.rhov[id];
                    Ut[3][m] = F.rhow[id];
                    Ut[4][m] = F.E[id];
                    ut[0][m] = F.rho[id];
                    ut[1][m] = F.u[id];
                    ut[2][m] = F.v[id];
                    ut[3][m] = F.w[id];
                    ut[4][m] = F.p[id];
                }

                std::vector<double> Fface(VAR, 0.0);
                reconstructInviscidFlux(Fface, Ft, Ut, ut, P, /*dim=*/1);

                int fid = idx_fy(i, j, k, L);
                F.flux_fy_mass[fid] = Fface[0];
                F.flux_fy_momx[fid] = Fface[1];
                F.flux_fy_momy[fid] = Fface[2];
                F.flux_fy_momz[fid] = Fface[3];
                F.flux_fy_E[fid]    = Fface[4];
            }
        }
    }

    // Z方向通量重构
    for (int j = ngy; j < ngy+ny; ++j) {
        for (int i = ngx; i < ngx+nx; ++i) {
            for (int k = ngz-1; k < ngz+nz; ++k) {
                // dynamic 2D arrays: VAR x stencil
                std::vector<std::vector<double>> Ft(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> Ut(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> ut(VAR, std::vector<double>(stencil));

                for (int m = 0; m < stencil; ++m) {
                    int kk = k + (m - (stencil/2));
                    int id = F.I(i, j, kk);

                    Ft[0][m] = F.Gflux_mass[id];
                    Ft[1][m] = F.Gflux_momx[id];
                    Ft[2][m] = F.Gflux_momy[id];
                    Ft[3][m] = F.Gflux_momz[id];
                    Ft[4][m] = F.Gflux_E[id];
                    Ut[0][m] = F.rho[id];
                    Ut[1][m] = F.rhou[id];
                    Ut[2][m] = F.rhov[id];
                    Ut[3][m] = F.rhow[id];
                    Ut[4][m] = F.E[id];
                    ut[0][m] = F.rho[id];
                    ut[1][m] = F.u[id];
                    ut[2][m] = F.v[id];
                    ut[3][m] = F.w[id];
                    ut[4][m] = F.p[id];
                }

                std::vector<double> Fface(VAR, 0.0);
                reconstructInviscidFlux(Fface, Ft, Ut, ut, P, /*dim=*/2);

                int fid = idx_fz(i, j, k, L);
                F.flux_fz_mass[fid] = Fface[0];
                F.flux_fz_momx[fid] = Fface[1];
                F.flux_fz_momy[fid] = Fface[2];
                F.flux_fz_momz[fid] = Fface[3];
                F.flux_fz_E[fid]    = Fface[4];
            }
        }
    }

}

// Roe平均
void computeRoeAveragedState(double &rho_bar, double &rhou_bar, double &rhov_bar, double &rhow_bar,
                             double &h_bar, double &a_bar,
                             const double Ul[5], const double Ur[5],
                             double gamma)
{
    // 提取左状态变量
    double rho_L = Ul[0];
    double u_L = Ul[1] / rho_L;
    double v_L = Ul[2] / rho_L;
    double w_L = Ul[3] / rho_L;
    double p_L = (gamma - 1.0) * (Ul[4] - 0.5 * rho_L * (u_L*u_L + v_L*v_L + w_L*w_L));
    // 提取右状态变量
    double rho_R = Ur[0];
    double u_R = Ur[1] / rho_R;
    double v_R = Ur[2] / rho_R;
    double w_R = Ur[3] / rho_R;
    double p_R = (gamma - 1.0) * (Ur[4] - 0.5 * rho_R * (u_R*u_R + v_R*v_R + w_R*w_R));

    // 计算Roe平均态
    double sqrt_rho_L = std::sqrt(rho_L);
    double sqrt_rho_R = std::sqrt(rho_R);
    rho_bar = sqrt_rho_L * sqrt_rho_R;
    double denom = 1.0 / (sqrt_rho_L + sqrt_rho_R);
    rhou_bar = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * denom;
    rhov_bar = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) * denom;
    rhow_bar = (sqrt_rho_L * w_L + sqrt_rho_R * w_R) * denom;
    // Total specific enthalpy H = (E + p) / rho. Here Ul[4] and Ur[4] are the total energy (conserved E).
    double H_L = (Ul[4] + p_L) / rho_L;
    double H_R = (Ur[4] + p_R) / rho_R;
    h_bar = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) * denom;
    double kinetic_bar = 0.5 * (rhou_bar*rhou_bar + rhov_bar*rhov_bar + rhow_bar*rhow_bar);
    a_bar = std::sqrt((gamma - 1.0) * (h_bar - kinetic_bar));

}

// 计算左/右 特征向量矩阵 L (左) 与 R (右) 对任意法向量 (nx,ny,nz)
// using Blazek-style formula from your snippet
static void build_eigen_matrices(const double Ul[5], const double Ur[5],
                                 double nx, double ny, double nz,
                                 double gamma,
                                 double Lmat[5][5], double Rmat[5][5],
                                 double lambar[5])
{
    // first compute Roe averaged quantities
    double rhobar, ubar, vbar, wbar, Hbar, abar, pbar;
    computeRoeAveragedState(rhobar, ubar, vbar, wbar, Hbar, abar, Ul, Ur, gamma);

    double V = nx * ubar + ny * vbar + nz * wbar;
    double c = abar;

    lambar[0] = V - c;
    lambar[1] = V;
    lambar[2] = V;
    lambar[3] = V;
    lambar[4] = V + c;

    double phi = 0.5 * (gamma - 1.0) * (ubar*ubar + vbar*vbar + wbar*wbar);

    double a1 = gamma - 1.0;
    double a2 = 1.0 / (std::sqrt(2.0) * rhobar * c);
    double a3 = rhobar / (std::sqrt(2.0) * c);
    double a4 = (phi + c*c) / (gamma - 1.0);
    double a5 = 1.0 - phi / (c*c);
    double a6 = phi / (gamma - 1.0);

    // Left eigenvectors L (rows)
    // L[0][:]
    Lmat[0][0] = a2 * (phi + c * V);
    Lmat[0][1] = -a2 * (a1 * ubar + nx * c);
    Lmat[0][2] = -a2 * (a1 * vbar + ny * c);
    Lmat[0][3] = -a2 * (a1 * wbar + nz * c);
    Lmat[0][4] = a1 * a2;

    // L[1][:]
    Lmat[1][0] = nx * a5 - (nz * vbar - ny * wbar) / rhobar;
    Lmat[1][1] = nx * a1 * ubar / (c*c);
    Lmat[1][2] = nx * a1 * vbar / (c*c) + nz / rhobar;
    Lmat[1][3] = nx * a1 * wbar / (c*c) - ny / rhobar;
    Lmat[1][4] = -nx * a1 / (c*c);

    // L[2][:]
    Lmat[2][0] = nz * a5 - (ny * ubar - nx * vbar) / rhobar;
    Lmat[2][1] = nz * a1 * ubar / (c*c) + ny / rhobar;
    Lmat[2][2] = nz * a1 * vbar / (c*c) - nx / rhobar;
    Lmat[2][3] = nz * a1 * wbar / (c*c);
    Lmat[2][4] = -nz * a1 / (c*c);

    // L[3][:]
    Lmat[3][0] = ny * a5 - (nx * wbar - nz * ubar) / rhobar;
    Lmat[3][1] = ny * a1 * ubar / (c*c) - nz / rhobar;
    Lmat[3][2] = ny * a1 * vbar / (c*c);
    Lmat[3][3] = ny * a1 * wbar / (c*c) + nx / rhobar;
    Lmat[3][4] = -ny * a1 / (c*c);

    // L[4][:]
    Lmat[4][0] = a2 * (phi - c * V);
    Lmat[4][1] = -a2 * (a1 * ubar - nx * c);
    Lmat[4][2] = -a2 * (a1 * vbar - ny * c);
    Lmat[4][3] = -a2 * (a1 * wbar - nz * c);
    Lmat[4][4] = a1 * a2;

    // Right eigenvectors R (columns)
    // R[:,0]
    Rmat[0][0] = a3;
    Rmat[1][0] = a3 * (ubar - nx*c);
    Rmat[2][0] = a3 * (vbar - ny*c);
    Rmat[3][0] = a3 * (wbar - nz*c);
    Rmat[4][0] = a3 * (a4 - c * V);

    // R[:,1]
    Rmat[0][1] = nx;
    Rmat[1][1] = nx * ubar;
    Rmat[2][1] = nx * vbar + nz * rhobar;
    Rmat[3][1] = nx * wbar - ny * rhobar;
    Rmat[4][1] = nx * a6 + rhobar * (nz * vbar - ny * wbar);

    // R[:,2]
    Rmat[0][2] = nz;
    Rmat[1][2] = nz * ubar + ny * rhobar;
    Rmat[2][2] = nz * vbar - nx * rhobar;
    Rmat[3][2] = nz * wbar;
    Rmat[4][2] = nz * a6 + rhobar * (ny * ubar - nx * vbar);

    // R[:,3]
    Rmat[0][3] = ny;
    Rmat[1][3] = ny * ubar - nz * rhobar;
    Rmat[2][3] = ny * vbar;
    Rmat[3][3] = ny * wbar + nx * rhobar;
    Rmat[4][3] = ny * a6 + rhobar * (nx * wbar - nz * ubar);

    // R[:,4]
    Rmat[0][4] = a3;
    Rmat[1][4] = a3 * (ubar + nx*c);
    Rmat[2][4] = a3 * (vbar + ny*c);
    Rmat[3][4] = a3 * (wbar + nz*c);
    Rmat[4][4] = a3 * (a4 + c * V);
}

// 无粘通量重构
// reconstructInviscidFlux: accept dynamic containers (VAR x stencil)
void reconstructInviscidFlux(std::vector<double> &Fface,
                             const std::vector<std::vector<double>> &Ft,
                             const std::vector<std::vector<double>> &Ut,
                             const std::vector<std::vector<double>> &ut,
                             const SolverParams &P, int dim)
{
    // alias
    double gamma = P.gamma;
    bool sigma = P.char_recon;
    const int VAR = 5; // 变量个数：rho, rhou, rhov, rhow, E
    // Use runtime stencil size from SolverParams so different reconstructions
    // (WENO5, C6th, ...) can be selected at runtime.
    int stencil = P.stencil;

    // determine normal vector (nx,ny,nz)
    double nx = 0.0, ny = 0.0, nz = 0.0;
    if (dim == 0) { nx = 1.0; ny = 0.0; nz = 0.0; }
    if (dim == 1) { nx = 0.0; ny = 1.0; nz = 0.0; }
    if (dim == 2) { nx = 0.0; ny = 0.0; nz = 1.0; }

    // 1) compute local stencil eigenvalues lamda[n][m] using primitives ut
    std::vector<std::vector<double>> lamda(VAR, std::vector<double>(stencil));
    for (int m = 0; m < stencil; ++m) {
        double rho = ut[0][m];
        double uu  = ut[1][m];
        double vv  = ut[2][m];
        double ww  = ut[3][m];
        double pp  = ut[4][m];
        double c = std::sqrt(gamma * pp / std::max(rho, 1e-12));
        double V = nx * uu + ny * vv + nz * ww;
        lamda[0][m] = V - c;
        lamda[1][m] = V;
        lamda[2][m] = V;
        lamda[3][m] = V;
        lamda[4][m] = V + c;
    }

    // 2) choose interface (left / right) states as stencil center positions:
    //    In your reference: left at index 2 (i), right at 3 (i+1)
    double Ul[VAR], Ur[VAR];
    for (int n = 0; n < VAR; ++n) { Ul[n] = Ut[n][2]; Ur[n] = Ut[n][3]; }

    // 3) compute Roe averaged eigenvectors and lambar
    double Lmat[VAR][VAR], Rmat[VAR][VAR], lambar[VAR];
    build_eigen_matrices(Ul, Ur, nx, ny, nz, gamma, Lmat, Rmat, lambar);

    // 4) choose dissipation (lamdamax) per characteristic based on P.fvs_type:
    double lamdamax[VAR];
    switch (P.fvs_type) {
        case SolverParams::FVS_Type::VanLeer: {
            // Global Lax-Friedrichs: choose a global max over stencil and components
            double gmax = 0.0;
            for (int m = 0; m < stencil; ++m)
                for (int n = 0; n < VAR; ++n)
                    gmax = std::max(gmax, std::abs(lamda[n][m]));
            for (int n = 0; n < VAR; ++n) lamdamax[n] = gmax;
        } break;
        case SolverParams::FVS_Type::LaxFriedrichs: {
            // Local Lax-Friedrichs: per-component max over stencil
            for (int n = 0; n < VAR; ++n) {
                double mxx = 0.0;
                for (int m = 0; m < stencil; ++m) mxx = std::max(mxx, std::abs(lamda[n][m]));
                lamdamax[n] = mxx;
            }
        } break;
        case SolverParams::FVS_Type::StegerWarming:
        default: {
            // Roe with entropy correction (approx)
            for (int n = 0; n < VAR; ++n) {
                double eps = 4.0 * std::max(std::max(lambar[n] - lamda[n][2], lamda[n][3] - lambar[n]), 0.0);
                if (abs(lambar[n]) < eps) lamdamax[n] = (lambar[n]*lambar[n] + eps*eps) / (2.0 * eps);
                else lamdamax[n] = abs(lambar[n]);
            }
        } break;
    }

    // 5) component-wise cheap option
    if (!sigma) {
        // For each component n, form wtplus = 0.5*(Ft + lamdamax * Ut) across stencil
        std::vector<double> wface(VAR, 0.0);
        for (int n = 0; n < VAR; ++n) {
            std::vector<double> wplus(stencil), wminus(stencil);
            for (int m = 0; m < stencil; ++m) {
                wplus[m]  = 0.5 * (Ft[n][m] + lamdamax[n] * Ut[n][m]);
                wminus[m] = 0.5 * (Ft[n][m] - lamdamax[n] * Ut[n][m]);
            }
            double plus_face = reconstruct_select(wplus, +1.0, P);
            double minus_face = reconstruct_select(wminus, -1.0, P);
            wface[n] = plus_face + minus_face;
        }
        // component-wise result is wface in conservative flux-like variables
        for (int n = 0; n < VAR; ++n) Fface[n] = wface[n];
        return;
    }

    // 6) characteristic-wise reconstruction:
    // Compute characteristic variables w = L * Ft  and LU = L * Ut (L is left-eig matrix)
    std::vector<std::vector<double>> wchar(VAR, std::vector<double>(stencil));
    std::vector<std::vector<double>> LU(VAR, std::vector<double>(stencil));
    for (int m = 0; m < stencil; ++m) {
        for (int n = 0; n < VAR; ++n) {
            double sumw = 0.0, sumLU = 0.0;
            for (int r = 0; r < VAR; ++r) {
                sumw  += Lmat[n][r] * Ft[r][m];
                sumLU += Lmat[n][r] * Ut[r][m];
            }
            wchar[n][m] = sumw;
            LU[n][m] = sumLU;
        }
    }

    // For each characteristic n, form wtplus = 0.5*(w + lamdamax * LU) and wtminus = ...
    std::vector<double> wflux_char(VAR, 0.0);
    for (int n = 0; n < VAR; ++n) {
        std::vector<double> wtplus(stencil), wtminus(stencil);
        for (int m = 0; m < stencil; ++m) {
            wtplus[m] = 0.5 * (wchar[n][m] + lamdamax[n] * LU[n][m]);
            wtminus[m] = 0.5 * (wchar[n][m] - lamdamax[n] * LU[n][m]);
        }
        double plus_face = reconstruct_select(wtplus, +1.0, P);
        double minus_face = reconstruct_select(wtminus, -1.0, P);
        wflux_char[n] = plus_face + minus_face;
    }

    // transform back to conservative flux via Fflux = R * wflux_char
    for (int n = 0; n < VAR; ++n) {
        double sum = 0.0;
        for (int r = 0; r < VAR; ++r) sum += Rmat[n][r] * wflux_char[r];
        Fface[n] = sum;
    }
}

// 计算空间导数
void compute_gradients(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const double dx = G.dx, dy = G.dy, dz = G.dz;
    int nx = L.nx, ny = L.ny, nz = L.nz;
    int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;

    // Compute 6th-order central differences only for interior (non-ghost) points.
    // Assumes ghost layers exist; we compute for i/j/k in [3, L.* - 4] so +/-3 stencil fits.
    for (int k = ngz; k < ngz+nz; ++k)
    for (int j = ngy; j < ngy+ny; ++j)
    for (int i = ngx; i < ngx+nx; ++i)
    {
        int id = F.I(i, j, k);

        // neighbor indices (interior guaranteed so no bounds checks)
        int ip  = F.I(i+1, j, k), im  = F.I(i-1, j, k);
        int ip2 = F.I(i+2, j, k), im2 = F.I(i-2, j, k);
        int ip3 = F.I(i+3, j, k), im3 = F.I(i-3, j, k);

        int jp  = F.I(i, j+1, k), jm  = F.I(i, j-1, k);
        int jp2 = F.I(i, j+2, k), jm2 = F.I(i, j-2, k);
        int jp3 = F.I(i, j+3, k), jm3 = F.I(i, j-3, k);

        int kp  = F.I(i, j, k+1), km  = F.I(i, j, k-1);
        int kp2 = F.I(i, j, k+2), km2 = F.I(i, j, k-2);
        int kp3 = F.I(i, j, k+3), km3 = F.I(i, j, k-3);

        // 6th-order central difference: (-f_{i+3} + 9 f_{i+2} -45 f_{i+1} +45 f_{i-1} -9 f_{i-2} + f_{i-3}) / (60 h)
        F.du_dx[id] = (-F.u[ip3] + 9.0*F.u[ip2] - 45.0*F.u[ip] + 45.0*F.u[im] - 9.0*F.u[im2] + F.u[im3]) / (60.0 * dx);
        F.du_dy[id] = (-F.u[jp3] + 9.0*F.u[jp2] - 45.0*F.u[jp] + 45.0*F.u[jm] - 9.0*F.u[jm2] + F.u[jm3]) / (60.0 * dy);
        F.du_dz[id] = (-F.u[kp3] + 9.0*F.u[kp2] - 45.0*F.u[kp] + 45.0*F.u[km] - 9.0*F.u[km2] + F.u[km3]) / (60.0 * dz);

        F.dv_dx[id] = (-F.v[ip3] + 9.0*F.v[ip2] - 45.0*F.v[ip] + 45.0*F.v[im] - 9.0*F.v[im2] + F.v[im3]) / (60.0 * dx);
        F.dv_dy[id] = (-F.v[jp3] + 9.0*F.v[jp2] - 45.0*F.v[jp] + 45.0*F.v[jm] - 9.0*F.v[jm2] + F.v[jm3]) / (60.0 * dy);
        F.dv_dz[id] = (-F.v[kp3] + 9.0*F.v[kp2] - 45.0*F.v[kp] + 45.0*F.v[km] - 9.0*F.v[km2] + F.v[km3]) / (60.0 * dz);

        F.dw_dx[id] = (-F.w[ip3] + 9.0*F.w[ip2] - 45.0*F.w[ip] + 45.0*F.w[im] - 9.0*F.w[im2] + F.w[im3]) / (60.0 * dx);
        F.dw_dy[id] = (-F.w[jp3] + 9.0*F.w[jp2] - 45.0*F.w[jp] + 45.0*F.w[jm] - 9.0*F.w[jm2] + F.w[jm3]) / (60.0 * dy);
        F.dw_dz[id] = (-F.w[kp3] + 9.0*F.w[kp2] - 45.0*F.w[kp] + 45.0*F.w[km] - 9.0*F.w[km2] + F.w[km3]) / (60.0 * dz);

        F.dT_dx[id] = (-F.T[ip3] + 9.0*F.T[ip2] - 45.0*F.T[ip] + 45.0*F.T[im] - 9.0*F.T[im2] + F.T[im3]) / (60.0 * dx);
        F.dT_dy[id] = (-F.T[jp3] + 9.0*F.T[jp2] - 45.0*F.T[jp] + 45.0*F.T[jm] - 9.0*F.T[jm2] + F.T[jm3]) / (60.0 * dy);
        F.dT_dz[id] = (-F.T[kp3] + 9.0*F.T[kp2] - 45.0*F.T[kp] + 45.0*F.T[km] - 9.0*F.T[km2] + F.T[km3]) / (60.0 * dz);
    }
}

// 计算粘性通量
void compute_viscous_flux(Field3D &F, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const double mu = P.mu;
    const double gamma = P.gamma;
    const double Rgas = P.Rgas;
    const double Pr = P.Pr;
    const double Cp = gamma * Rgas / (gamma - 1.0);

    for (int k = 0; k < L.sz; ++k)
    for (int j = 0; j < L.sy; ++j)
    for (int i = 0; i < L.sx; ++i)
    {
        int id = F.I(i, j, k);

        // 速度散度
        double divU = F.du_dx[id] + F.dv_dy[id] + F.dw_dz[id];

        // 应力分量
        double tau_xx = 2.0 * mu * F.du_dx[id] - (2.0/3.0) * mu * divU;
        double tau_yy = 2.0 * mu * F.dv_dy[id] - (2.0/3.0) * mu * divU;
        double tau_zz = 2.0 * mu * F.dw_dz[id] - (2.0/3.0) * mu * divU;

        double tau_xy = mu * (F.du_dy[id] + F.dv_dx[id]);
        double tau_xz = mu * (F.du_dz[id] + F.dw_dx[id]);
        double tau_yz = mu * (F.dv_dz[id] + F.dw_dy[id]);

        // 热通量
        double qx = -mu * Cp / Pr * F.dT_dx[id];
        double qy = -mu * Cp / Pr * F.dT_dy[id];
        double qz = -mu * Cp / Pr * F.dT_dz[id];

        double u = F.u[id], v = F.v[id], w = F.w[id];

        // X方向粘性通量
        F.Fvflux_mass[id] = 0.0;
        F.Fvflux_momx[id] = tau_xx;
        F.Fvflux_momy[id] = tau_xy;
        F.Fvflux_momz[id] = tau_xz;
        F.Fvflux_E[id]    = u*tau_xx + v*tau_xy + w*tau_xz + qx;

        // Y方向粘性通量
        F.Hvflux_mass[id] = 0.0;
        F.Hvflux_momx[id] = tau_xy;
        F.Hvflux_momy[id] = tau_yy;
        F.Hvflux_momz[id] = tau_yz;
        F.Hvflux_E[id]    = u*tau_xy + v*tau_yy + w*tau_yz + qy;

        // Z方向粘性通量
        F.Gvflux_mass[id] = 0.0;
        F.Gvflux_momx[id] = tau_xz;
        F.Gvflux_momy[id] = tau_yz;
        F.Gvflux_momz[id] = tau_zz;
        F.Gvflux_E[id]    = u*tau_xz + v*tau_yz + w*tau_zz + qz;
    }
}

// 重构粘性通量
void reconstructViscidFlux(Field3D &F, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const int nx = L.nx, ny = L.ny, nz = L.nz;
    const int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;

    // ----------------- X方向粘性半节点通量 -----------------
    for (int k = ngz; k < ngz+nz; ++k)
    for (int j = ngy; j < ngy+ny; ++j)
    for (int i = ngx-1; i < ngx+nx; ++i) {
        int fid = idx_fx(i, j, k, L);

        std::array<std::vector<double>*,5> src = { &F.Fvflux_mass, &F.Fvflux_momx, &F.Fvflux_momy, &F.Fvflux_momz, &F.Fvflux_E };
        std::array<std::vector<double>*,5> dst = { &F.flux_fx_mass, &F.flux_fx_momx, &F.flux_fx_momy, &F.flux_fx_momz, &F.flux_fx_E };

        int ids6[6] = { F.I(i-2,j,k), F.I(i-1,j,k), F.I(i,j,k), F.I(i+1,j,k), F.I(i+2,j,k), F.I(i+3,j,k) };
        std::array<double,6> s6;
        for (int comp = 0; comp < 5; ++comp) {
            double *sp = src[comp]->data();
            double *dp = dst[comp]->data();
            for (int t = 0; t < 6; ++t) s6[t] = sp[ids6[t]];
            double val = c6th_reconstruction(s6);
            dp[fid] += val;  // note the += to add viscous to inviscid flux
        }
    }

    // ----------------- Y方向粘性半节点通量 -----------------
    for (int k = ngz; k < ngz+nz; ++k)
    for (int i = ngx; i < ngx+nx; ++i)
    for (int j = ngy-1; j < ngy+ny; ++j) {
        int fid = idx_fy(i, j, k, L);

        std::array<std::vector<double>*,5> src = { &F.Hvflux_mass, &F.Hvflux_momx, &F.Hvflux_momy, &F.Hvflux_momz, &F.Hvflux_E };
        std::array<std::vector<double>*,5> dst = { &F.flux_fy_mass, &F.flux_fy_momx, &F.flux_fy_momy, &F.flux_fy_momz, &F.flux_fy_E };

        int ids6[6] = { F.I(i,j-2,k), F.I(i,j-1,k), F.I(i,j,k), F.I(i,j+1,k), F.I(i,j+2,k), F.I(i,j+3,k) };
        std::array<double,6> s6;
        for (int comp = 0; comp < 5; ++comp) {
            double *sp = src[comp]->data();
            double *dp = dst[comp]->data();
            for (int t = 0; t < 6; ++t) s6[t] = sp[ids6[t]];
            double val = c6th_reconstruction(s6);
            dp[fid] += val;
        }
    }

    // ----------------- Z方向粘性半节点通量 -----------------
    for (int j = ngy; j < ngy+ny; ++j)
    for (int i = ngx; i < ngx+nx; ++i)
    for (int k = ngz-1; k < ngz+nz; ++k) {
        int fid = idx_fz(i, j, k, L);

        std::array<std::vector<double>*,5> src = { &F.Gvflux_mass, &F.Gvflux_momx, &F.Gvflux_momy, &F.Gvflux_momz, &F.Gvflux_E };
        std::array<std::vector<double>*,5> dst = { &F.flux_fz_mass, &F.flux_fz_momx, &F.flux_fz_momy, &F.flux_fz_momz, &F.flux_fz_E };


        int ids6[6] = { F.I(i,j,k-2), F.I(i,j,k-1), F.I(i,j,k), F.I(i,j,k+1), F.I(i,j,k+2), F.I(i,j,k+3) };
        std::array<double,6> s6;
        for (int comp = 0; comp < 5; ++comp) {
            double *sp = src[comp]->data();
            double *dp = dst[comp]->data();
            for (int t = 0; t < 6; ++t) s6[t] = sp[ids6[t]];
            double val = c6th_reconstruction(s6);
            dp[fid] += val;
        }
    }
}