#include "aak_waveform.hpp"
#include <cmath>
#include <vector>
#include <gsl/gsl_sf_bessel.h>

namespace emrikludge {

std::pair<std::vector<double>, std::vector<double>> 
generate_aak_waveform_cpp(
    const std::vector<double>& t,
    const std::vector<double>& p,
    const std::vector<double>& e,
    const std::vector<double>& iota,
    const std::vector<double>& Phi_r,
    const std::vector<double>& Phi_th,
    const std::vector<double>& Phi_phi,
    double M, double mu, double dist,
    double viewing_theta, double viewing_phi
) {
    size_t N = t.size();
    std::vector<double> h_plus(N, 0.0);
    std::vector<double> h_cross(N, 0.0);

    // 几何单位转换因子
    // 假设 M, mu, p, dist 已经在外部统一了单位，或者在这里处理
    // 简单的振幅前缀 H0 = mu / D
    double amp_scale = mu / dist; 

    // 观测角度因子
    double cos_theta = cos(viewing_theta);
    double sin_theta = sin(viewing_theta);
    // AAK 论文中的 F+ Fx 模式函数 (Barack & Cutler 2004 Eq 42 近似)
    // 对于主要模式 (l=2, m=2):
    double F_plus = 0.5 * (1.0 + cos_theta * cos_theta);
    double F_cross = cos_theta;

    for (size_t i = 0; i < N; ++i) {
        if (p[i] < 3.0) continue;

        // 基础振幅随 p 衰减
        double omega = pow(p[i], -1.5); 
        double H = amp_scale * pow(omega, 2.0/3.0) * 4.0; // 粗略系数

        // 谐波求和 (n: 径向谐波)
        // 主导项是 Quadrupole (m=2)，所以相位基频是 2*Phi_phi
        // 加上径向调制 n*Phi_r
        for (int n = -5; n <= 5; ++n) {
            // 贝塞尔函数 J_n(2e) 是 Kludge 中常用的近似 (Peters & Mathews)
            double Jn = gsl_sf_bessel_Jn(n, 2.0 * e[i]);
            
            // 修正的谐波相位：2*phi - n*radial_phase
            double phase = 2.0 * Phi_phi[i] - n * Phi_r[i];
            
            // 加上观测方位角 Phi_obs 的影响 (m=2 -> 2*Phi_obs)
            double wave_phase = phase - 2.0 * viewing_phi;

            // 简单的极化组合
            h_plus[i]  += H * Jn * F_plus  * cos(wave_phase);
            h_cross[i] += H * Jn * F_cross * sin(wave_phase);
        }
    }

    return {h_plus, h_cross};
}

} // namespace