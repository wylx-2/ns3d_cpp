#include "ns3d_func.h"

void reconstruct_linear(const std::vector<double> &q, int i, double &qL, double &qR)
{
    // 简单线性：左右点直接取相邻值
    qL = q[i];
    qR = q[i+1];
}
