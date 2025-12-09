#include "field_structures.h"
#include "ns3d_func.h"

void isotropic_post_process(Field3D &F, GridDesc &G, CartDecomp &C, const std::string &prefix)
{
    // 计算并输出能量谱
    compute_energy_spectrum(F, G, C, prefix + "_spectrum.dat");

    // 计算并输出湍流统计量
    // Taylor 微尺度、雷诺数等
    // 这里可以添加更多的统计量计算和输出
}