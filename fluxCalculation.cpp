#include "ns3d_func.h"

inline void flux_euler(double rho, double u, double v, double w,
                double p, double E, double F[5])
{
    F[0] = rho * u;
    F[1] = rho * u * u + p;
    F[2] = rho * u * v;
    F[3] = rho * u * w;
    F[4] = (E + p) * u;
}

inline double sound_speed(double gamma, double p, double rho)
{
    return std::sqrt(gamma * p / rho);
}

void rusanov_flux(const double *QL, const double *QR, double gamma, double *Fhat)
{
    double rhoL = QL[0], uL = QL[1]/rhoL, vL = QL[2]/rhoL, wL = QL[3]/rhoL;
    double pL = (gamma - 1.0) * (QL[4] - 0.5*rhoL*(uL*uL+vL*vL+wL*wL));

    double rhoR = QR[0], uR = QR[1]/rhoR, vR = QR[2]/rhoR, wR = QR[3]/rhoR;
    double pR = (gamma - 1.0) * (QR[4] - 0.5*rhoR*(uR*uR+vR*vR+wR*wR));

    double FL[5], FR[5];
    flux_euler(rhoL,uL,vL,wL,pL,QL[4],FL);
    flux_euler(rhoR,uR,vR,wR,pR,QR[4],FR);

    double aL = sound_speed(gamma,pL,rhoL);
    double aR = sound_speed(gamma,pR,rhoR);
    double smax = std::max(std::abs(uL)+aL, std::abs(uR)+aR);

    for(int k=0;k<5;++k)
        Fhat[k] = 0.5*(FL[k]+FR[k]) - 0.5*smax*(QR[k]-QL[k]);
}

void computeEigenSystem3D(double nx, double ny, double nz,
                          double gamma,
                          double rhobar, double ubar, double vbar, double wbar,
                          double Hbar, double c,
                          double L[5][5], double R[5][5])
{
    // 方向上的投影速度
    double V = nx*ubar + ny*vbar + nz*wbar;
    double phi = 0.5 * (gamma - 1.0) * (ubar*ubar + vbar*vbar + wbar*wbar);

    // 系数定义（参考 Blazek 或 Toro 书）
    double a1 = gamma - 1.0;
    double a2 = 1.0 / (std::sqrt(2.0) * rhobar * c);
    double a3 = rhobar / (std::sqrt(2.0) * c);
    double a4 = (phi + c*c) / (gamma - 1.0);
    double a5 = 1.0 - phi / (c*c);
    double a6 = phi / (gamma - 1.0);

    //---------------------------
    // 左特征向量矩阵 L
    //---------------------------
    L[0][0] =  a2 * (phi + c * V);
    L[0][1] = -a2 * (a1 * ubar + nx * c);
    L[0][2] = -a2 * (a1 * vbar + ny * c);
    L[0][3] = -a2 * (a1 * wbar + nz * c);
    L[0][4] =  a1 * a2;

    L[1][0] = nx * a5 - (nz * vbar - ny * wbar) / rhobar;
    L[1][1] = nx * a1 * ubar / (c*c);
    L[1][2] = nx * a1 * vbar / (c*c) + nz / rhobar;
    L[1][3] = nx * a1 * wbar / (c*c) - ny / rhobar;
    L[1][4] = -nx * a1 / (c*c);

    L[2][0] = nz * a5 - (ny * ubar - nx * vbar) / rhobar;
    L[2][1] = nz * a1 * ubar / (c*c) + ny / rhobar;
    L[2][2] = nz * a1 * vbar / (c*c) - nx / rhobar;
    L[2][3] = nz * a1 * wbar / (c*c);
    L[2][4] = -nz * a1 / (c*c);

    L[3][0] = ny * a5 - (nx * wbar - nz * ubar) / rhobar;
    L[3][1] = ny * a1 * ubar / (c*c) - nz / rhobar;
    L[3][2] = ny * a1 * vbar / (c*c);
    L[3][3] = ny * a1 * wbar / (c*c) + nx / rhobar;
    L[3][4] = -ny * a1 / (c*c);

    L[4][0] =  a2 * (phi - c * V);
    L[4][1] = -a2 * (a1 * ubar - nx * c);
    L[4][2] = -a2 * (a1 * vbar - ny * c);
    L[4][3] = -a2 * (a1 * wbar - nz * c);
    L[4][4] =  a1 * a2;

    //---------------------------
    // 右特征向量矩阵 R
    //---------------------------
    R[0][0] = a3;
    R[1][0] = a3 * (ubar - nx * c);
    R[2][0] = a3 * (vbar - ny * c);
    R[3][0] = a3 * (wbar - nz * c);
    R[4][0] = a3 * (a4 - c * V);

    R[0][1] = nx;
    R[1][1] = nx * ubar;
    R[2][1] = nx * vbar + nz * rhobar;
    R[3][1] = nx * wbar - ny * rhobar;
    R[4][1] = nx * a6 + rhobar * (nz * vbar - ny * wbar);

    R[0][2] = nz;
    R[1][2] = nz * ubar + ny * rhobar;
    R[2][2] = nz * vbar - nx * rhobar;
    R[3][2] = nz * wbar;
    R[4][2] = nz * a6 + rhobar * (ny * ubar - nx * vbar);

    R[0][3] = ny;
    R[1][3] = ny * ubar - nz * rhobar;
    R[2][3] = ny * vbar;
    R[3][3] = ny * wbar + nx * rhobar;
    R[4][3] = ny * a6 + rhobar * (nx * wbar - nz * ubar);

    R[0][4] = a3;
    R[1][4] = a3 * (ubar + nx * c);
    R[2][4] = a3 * (vbar + ny * c);
    R[3][4] = a3 * (wbar + nz * c);
    R[4][4] = a3 * (a4 + c * V);
}

//------------------------------
// Roe 平均通量（适用于任意方向）
//------------------------------
void roe_flux(const double *QL, const double *QR, 
              double nx, double ny, double nz, double gamma,
              double *Fhat)
{
    // 解包左右状态
    double rhoL = QL[0], uL = QL[1]/rhoL, vL = QL[2]/rhoL, wL = QL[3]/rhoL;
    double pL = (gamma-1.0)*(QL[4]-0.5*rhoL*(uL*uL+vL*vL+wL*wL));
    double HL = (QL[4] + pL) / rhoL;

    double rhoR = QR[0], uR = QR[1]/rhoR, vR = QR[2]/rhoR, wR = QR[3]/rhoR;
    double pR = (gamma-1.0)*(QR[4]-0.5*rhoR*(uR*uR+vR*vR+wR*wR));
    double HR = (QR[4] + pR) / rhoR;

    // Roe 平均
    double sqrL = std::sqrt(rhoL);
    double sqrR = std::sqrt(rhoR);
    double is = 1.0 / (sqrL + sqrR);

    double rhobar = sqrL * sqrR;
    double ubar = (sqrL*uL + sqrR*uR) * is;
    double vbar = (sqrL*vL + sqrR*vR) * is;
    double wbar = (sqrL*wL + sqrR*wR) * is;
    double Hbar = (sqrL*HL + sqrR*HR) * is;

    // 声速与投影速度
    double Vbar = nx*ubar + ny*vbar + nz*wbar;
    double a2 = (gamma-1.0)*(Hbar - 0.5*(ubar*ubar+vbar*vbar+wbar*wbar));
    double abar = std::sqrt(std::max(a2, 1.0e-12));

    // 构建左右特征矩阵
    double L[5][5], R[5][5];
    computeEigenSystem3D(nx,ny,nz,gamma,rhobar,ubar,vbar,wbar,Hbar,abar,L,R);

    // 计算左右通量
    double FL[5], FR[5];
    flux_euler(rhoL,uL,vL,wL,pL,QL[4],FL);
    flux_euler(rhoR,uR,vR,wR,pR,QR[4],FR);

    // 差值向量
    double dU[5];
    for(int m=0;m<5;++m) dU[m] = QR[m] - QL[m];

    // 波强度系数 α = L * ΔU
    double alpha[5] = {0.0};
    for(int i=0;i<5;++i)
        for(int j=0;j<5;++j)
            alpha[i] += L[i][j]*dU[j];

    // 特征速度 λ
    double lambda[5];
    lambda[0] = Vbar - abar;
    lambda[1] = Vbar;
    lambda[2] = Vbar;
    lambda[3] = Vbar;
    lambda[4] = Vbar + abar;

    // Roe 修正（防止奇异点）
    for(int m=0;m<5;++m)
        lambda[m] = std::fabs(lambda[m]);

    // 差分通量项 (|A|ΔU = R * |Λ| * α)
    double Aabs_dU[5] = {0.0};
    for(int i=0;i<5;++i)
        for(int j=0;j<5;++j)
            Aabs_dU[i] += R[i][j] * (lambda[j] * alpha[j]);

    // Roe 通量
    for(int m=0;m<5;++m)
        Fhat[m] = 0.5*(FL[m] + FR[m]) - 0.5*Aabs_dU[m];
}