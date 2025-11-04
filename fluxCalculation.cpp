#include "ns3d_func.h"

//==================================================================
// flux calculation module
//==================================================================

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

// 重构无粘通量
void compute_flux_face(Field3D &F, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const int nx = L.sx, ny = L.sy, nz = L.sz;
    const int ngh = L.ngx;  // ghost层个数
    const double eps = 1e-6;

    // selection of reconstruction method
    auto recon_method = P.recon;

    // ----------------- X方向半节点通量 -----------------
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 2; i < nx - 2; ++i) {
        int ids[5] = { F.I(i-2,j,k), F.I(i-1,j,k), F.I(i,j,k), F.I(i+1,j,k), F.I(i+2,j,k) };
        int fid  = idx_fx(i, j, k, L);

        // 使用数组指针避免 switch/case：构建源/目标数组指针表
        std::array<std::vector<double>*,5> src_vec_x = {
            &F.Fflux_mass, &F.Fflux_momx, &F.Fflux_momy, &F.Fflux_momz, &F.Fflux_E
        };
        std::array<std::vector<double>*,5> dst_vec_x = {
            &F.flux_fx_mass, &F.flux_fx_momx, &F.flux_fx_momy, &F.flux_fx_momz, &F.flux_fx_E
        };

        std::array<double,5> st;
        for (int comp = 0; comp < 5; ++comp) {
            double *src = src_vec_x[comp]->data();
            double *dst = dst_vec_x[comp]->data();

            for (int s = 0; s < 5; ++s) st[s] = src[ids[s]];

            double val = (recon_method == SolverParams::Reconstruction::LINEAR)
                         ? linear_reconstruction(st)
                         : weno5_reconstruction(st);

            dst[fid] = val;
        }
    }

    // ----------------- Y方向半节点通量 -----------------
    for (int k = 0; k < nz; ++k)
    for (int i = 0; i < nx; ++i)
    for (int j = 2; j < ny - 2; ++j) {
        int ids[5] = { F.I(i,j-2,k), F.I(i,j-1,k), F.I(i,j,k), F.I(i,j+1,k), F.I(i,j+2,k) };
        int fid  = idx_fy(i, j, k, L);

        std::array<std::vector<double>*,5> src_vec_y = {
            &F.Hflux_mass, &F.Hflux_momx, &F.Hflux_momy, &F.Hflux_momz, &F.Hflux_E
        };
        std::array<std::vector<double>*,5> dst_vec_y = {
            &F.flux_fy_mass, &F.flux_fy_momx, &F.flux_fy_momy, &F.flux_fy_momz, &F.flux_fy_E
        };

        std::array<double,5> sty;
        for (int comp = 0; comp < 5; ++comp) {
            double *src = src_vec_y[comp]->data();
            double *dst = dst_vec_y[comp]->data();

            for (int s = 0; s < 5; ++s) sty[s] = src[ids[s]];

            double val = (recon_method == SolverParams::Reconstruction::LINEAR)
                         ? linear_reconstruction(sty)
                         : weno5_reconstruction(sty);

            dst[fid] = val;
        }
    }

    // ----------------- Z方向半节点通量 -----------------
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    for (int k = 2; k < nz - 2; ++k) {
        int ids[5] = { F.I(i,j,k-2), F.I(i,j,k-1), F.I(i,j,k), F.I(i,j,k+1), F.I(i,j,k+2) };
        int fid  = idx_fz(i, j, k, L);

        std::array<std::vector<double>*,5> src_vec_z = {
            &F.Gflux_mass, &F.Gflux_momx, &F.Gflux_momy, &F.Gflux_momz, &F.Gflux_E
        };
        std::array<std::vector<double>*,5> dst_vec_z = {
            &F.flux_fz_mass, &F.flux_fz_momx, &F.flux_fz_momy, &F.flux_fz_momz, &F.flux_fz_E
        };

        std::array<double,5> stz;
        for (int comp = 0; comp < 5; ++comp) {
            double *src = src_vec_z[comp]->data();
            double *dst = dst_vec_z[comp]->data();

            for (int s = 0; s < 5; ++s) stz[s] = src[ids[s]];

            double val = (recon_method == SolverParams::Reconstruction::LINEAR)
                         ? linear_reconstruction(stz)
                         : weno5_reconstruction(stz);

            dst[fid] = val;
        }
    }
}

// -----------------------------------------------------------------
// ---------   Flux Vector Splitting (FVS) -------------------------
// -----------------------------------------------------------------

// 1. eigenvalue分解
void eigen_decomposition(const double rho, const double u, const double v, const double w,
                         const double p, const SolverParams &P,
                         std::array<double,5> &lambda,
                         std::array<std::array<double,5>,5> &R,
                         std::array<std::array<double,5>,5> &R_inv)
{
    const double gamma = P.gamma;
    const double a = sqrt(gamma * p / rho); // sound speed
    const double H = (p / (gamma - 1.0) + 0.5 * rho * (u*u + v*v + w*w)) / rho; // total enthalpy

    // Eigenvalues
    lambda[0] = u - a;
    lambda[1] = u;
    lambda[2] = u;
    lambda[3] = u;
    lambda[4] = u + a;

    // Right eigenvectors
    R[0] = {1.0,         1.0,         0.0,         0.0,         1.0        };
    R[1] = {u - a,      u,           0.0,         0.0,         u + a      };
    R[2] = {v,          v,           1.0,         0.0,         v          };
    R[3] = {w,          w,           0.0,         1.0,         w          };
    R[4] = {H - u*a,    0.5*(u*u + v*v + w*w), v, w, H + u*a };

    // Left eigenvectors (R_inv)
    // (Implementation omitted for brevity; would compute the inverse of R)
    double beta = 0.5 / (a * a);
    R_inv[0] = { beta*(0.5*(gamma - 1.0)*(u*u + v*v + w*w) + a*u), -beta*( (gamma - 1.0)*u + a ), -beta*(gamma - 1.0)*v, -beta*(gamma - 1.0)*w, beta*(gamma - 1.0) };
    R_inv[1] = { 1.0 - (gamma - 1.0)*(u*u + v*v + w*w)/(2.0*a*a), (gamma - 1.0)*u/(a*a), (gamma - 1.0)*v/(a*a), (gamma - 1.0)*w/(a*a), -(gamma - 1.0)/(a*a) };
    R_inv[2] = { -v, 0.0, 1.0, 0.0, 0.0 };
    R_inv[3] = { -w, 0.0, 0.0, 1.0, 0.0 };
    R_inv[4] = { beta*(0.5*(gamma - 1.0)*(u*u + v*v + w*w) - a*u), -beta*( (gamma - 1.0)*u - a ), -beta*(gamma - 1.0)*v, -beta*(gamma - 1.0)*w, beta*(gamma - 1.0) };
}

// 2. 正负特征值分离
// steger-warming分离/Lax-Friedrichs分离/Van Leer分离
void flux_vector_splitting(const std::array<double,5> &lambda,
                            std::array<double,5> &lambda_plus,
                            std::array<double,5> &lambda_minus,
                            const SolverParams &P)
{
    const double eps = 1e-6;
    switch (P.fvs_type) {
        case SolverParams::FVS_Type::StegerWarming:
            for (int m = 0; m < 5; ++m) {
                lambda_plus[m]  = 0.5 * (lambda[m] + fabs(lambda[m]));
                lambda_minus[m] = 0.5 * (lambda[m] - fabs(lambda[m]));
            }
            break;

        case SolverParams::FVS_Type::LaxFriedrichs:
            for (int m = 0; m < 5; ++m) {
                double alpha = fabs(lambda[m]) + eps;
                lambda_plus[m]  = 0.5 * (lambda[m] + alpha);
                lambda_minus[m] = 0.5 * (lambda[m] - alpha);
            }
            break;

        case SolverParams::FVS_Type::VanLeer:
            for (int m = 0; m < 5; ++m) {
                if (lambda[m] > 0) {
                    lambda_plus[m]  = lambda[m] * lambda[m] / (lambda[m] + eps);
                    lambda_minus[m] = 0.0;
                } else {
                    lambda_plus[m]  = 0.0;
                    lambda_minus[m] = lambda[m] * lambda[m] / ( -lambda[m] + eps);
                }
            }
            break;
    }
}


// 3. 通量重构
void reconstruct_fvs_flux(Field3D &F, const SolverParams &P)
{
    // x方向FVS通量重构
    const LocalDesc &L = F.L;
    const int nx = L.sx, ny = L.sy, nz = L.sz;
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 2; i < nx - 2; ++i) {
        int fid = idx_fx(i, j, k, L);

        // 提取守恒变量
        int idL = F.I(i-1, j, k);
        int idR = F.I(i, j, k);
        double rhoL = F.rho[idL], uL = F.u[idL], vL = F.v[idL], wL = F.w[idL], pL = F.p[idL];
        double rhoR = F.rho[idR], uR = F.u[idR], vR = F.v[idR], wR = F.w[idR], pR = F.p[idR];

        // 特征分解
        std::array<double,5> lambdaL, lambdaR;
        std::array<std::array<double,5>,5> RL, RL_inv;
        std::array<std::array<double,5>,5> RR, RR_inv;

        eigen_decomposition(rhoL, uL, vL, wL, pL, P, lambdaL, RL, RL_inv);
        eigen_decomposition(rhoR, uR, vR, wR, pR, P, lambdaR, RR, RR_inv);

        // 特征值分离
        std::array<double,5> lambda_plusL, lambda_minusL;
        std::array<double,5> lambda_plusR, lambda_minusR;

        flux_vector_splitting(lambdaL, lambda_plusL, lambda_minusL, P);
        flux_vector_splitting(lambdaR, lambda_plusR, lambda_minusR, P);

        // 计算正负通量
        // FVS通量计算公式：F_plus = R * Lambda_plus * R_inv * U，F_minus类似
        // 最终结果存储在F.flux_fx_XXX[fid]中
    }
}

// 计算空间导数
void compute_gradients(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const double dx = G.dx, dy = G.dy, dz = G.dz;

    // Compute 6th-order central differences only for interior (non-ghost) points.
    // Assumes ghost layers exist; we compute for i/j/k in [3, L.* - 4] so +/-3 stencil fits.
    for (int k = 3; k <= L.sz - 4; ++k)
    for (int j = 3; j <= L.sy - 4; ++j)
    for (int i = 3; i <= L.sx - 4; ++i)
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

    for (int k = 1; k < L.sz - 1; ++k)
    for (int j = 1; j < L.sy - 1; ++j)
    for (int i = 1; i < L.sx - 1; ++i)
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
void compute_vis_flux_face(Field3D &F, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const int nx = L.sx, ny = L.sy, nz = L.sz;

    // Use c6th reconstruction from reconstruction.cpp when requested; that expects
    // a 6-point scalar stencil (i-2 .. i+3) and returns a single reconstructed value.
    // Fallback: if recon_vis != C6th, keep previous simple 7-point weights.
    bool use_c6 = (P.recon_vis == SolverParams::Reconstruction::C6th);

    // ----------------- X方向粘性半节点通量 -----------------
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 2; i <= nx - 4; ++i) {
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
    for (int k = 0; k < nz; ++k)
    for (int i = 0; i < nx; ++i)
    for (int j = 2; j <= ny - 4; ++j) {
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
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    for (int k = 2; k <= nz - 4; ++k) {
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