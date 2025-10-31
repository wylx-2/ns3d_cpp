#include "ns3d_func.h"
#include <array>

// 简单的线性重构（标量，5点模板）
double linear_reconstruction(const std::array<double,5>& stencil) {
    // 使用模板中间两个点的平均作为简单线性重构
    return 0.5 * (stencil[2] + stencil[3]);
}

// WENO5 重构（标量，5点模板）
double weno5_reconstruction(const std::array<double,5>& stencil) {
    double f0 = stencil[0];
    double f1 = stencil[1];
    double f2 = stencil[2];
    double f3 = stencil[3];
    double f4 = stencil[4];

    double eps = 1e-6;
    double alpha0 = 0.0, alpha1 = 0.0, alpha2 = 0.0;

    double beta0 = 13.0/12.0 * (f0 - 2*f1 + f2)*(f0 - 2*f1 + f2)
                 + 0.25 * (f0 - 4*f1 + 3*f2)*(f0 - 4*f1 + 3*f2);
    double beta1 = 13.0/12.0 * (f1 - 2*f2 + f3)*(f1 - 2*f2 + f3)
                 + 0.25 * (f1 - f3)*(f1 - f3);
    double beta2 = 13.0/12.0 * (f2 - 2*f3 + f4)*(f2 - 2*f3 + f4)
                 + 0.25 * (3*f2 - 4*f3 + f4)*(3*f2 - 4*f3 + f4);

    alpha0 = 0.1 / ((eps + beta0)*(eps + beta0));
    alpha1 = 0.6 / ((eps + beta1)*(eps + beta1));
    alpha2 = 0.3 / ((eps + beta2)*(eps + beta2));

    double asum = alpha0 + alpha1 + alpha2;
    double w0 = alpha0 / asum;
    double w1 = alpha1 / asum;
    double w2 = alpha2 / asum;

    double p0 = (2*f0 - 7*f1 + 11*f2) / 6.0;
    double p1 = (-f1 + 5*f2 + 2*f3) / 6.0;
    double p2 = (2*f2 + 5*f3 - f4) / 6.0;

    return w0 * p0 + w1 * p1 + w2 * p2;
}

// C6th 六阶中心差分
double c6th_reconstruction(const std::array<double,6>& stencil) {
    // 六阶中心差分重构公式
    return (1.0/60.0) * (stencil[0] - 8*stencil[1] + 37*stencil[2]
                        + 37*stencil[3] - 8*stencil[4] + stencil[5]);
}

// C4th 四阶中心差分
double c4th_reconstruction(const std::array<double,4>& stencil) {
    // 四阶中心差分重构公式
    return (1.0/12.0) * (-stencil[0] + 7*stencil[1] + 7*stencil[2] - stencil[3]);
} 