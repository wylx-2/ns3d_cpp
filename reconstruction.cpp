#include "ns3d_func.h"

// 简单的线性重构
std::vector<double> linear_reconstruction(const std::vector<std::vector<double>>& stencil) {
    int nvars = stencil[0].size();
    std::vector<double> result(nvars, 0.0);
    
    // 使用中间两个点
    for(int m = 0; m < nvars; ++m) {
        result[m] = 0.5 * (stencil[2][m] + stencil[3][m]);
    }
    
    return result;
}

// WENO重构
std::vector<double> weno5_reconstruction(const std::vector<std::vector<double>>& stencil) {
    // stencil: 5个模板点，每个点包含5个通量分量
    // 返回: 重构后的5个通量分量
    
    int nvars = stencil[0].size();
    std::vector<double> result(nvars, 0.0);
    
    double alpha[3], omega[3];
    double eps = 1e-6;
    
    for(int m = 0; m < nvars; ++m) {
        // 提取当前分量的5个模板点值
        double f0 = stencil[0][m];
        double f1 = stencil[1][m];
        double f2 = stencil[2][m];
        double f3 = stencil[3][m];
        double f4 = stencil[4][m];
        
        // 计算光滑指示器
        double beta0 = 13.0/12.0 * pow(f0 - 2*f1 + f2, 2) + 
                       0.25 * pow(f0 - 4*f1 + 3*f2, 2);
        double beta1 = 13.0/12.0 * pow(f1 - 2*f2 + f3, 2) + 
                       0.25 * pow(f1 - f3, 2);
        double beta2 = 13.0/12.0 * pow(f2 - 2*f3 + f4, 2) + 
                       0.25 * pow(3*f2 - 4*f3 + f4, 2);
        
        // 计算权重
        alpha[0] = 0.1 / pow(eps + beta0, 2);
        alpha[1] = 0.6 / pow(eps + beta1, 2);
        alpha[2] = 0.3 / pow(eps + beta2, 2);
        
        double asum = alpha[0] + alpha[1] + alpha[2];
        omega[0] = alpha[0] / asum;
        omega[1] = alpha[1] / asum;
        omega[2] = alpha[2] / asum;
        
        // 三个候选多项式
        double p0 = (2*f0 - 7*f1 + 11*f2) / 6.0;
        double p1 = (-f1 + 5*f2 + 2*f3) / 6.0;
        double p2 = (2*f2 + 5*f3 - f4) / 6.0;
        
        // 加权平均
        result[m] = omega[0] * p0 + omega[1] * p1 + omega[2] * p2;
    }
    
    return result;
}
