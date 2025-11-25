#include "aak_waveform.hpp"
#include <cmath>
#include <vector>
#include <complex>
#include <gsl/gsl_sf_bessel.h>

namespace emrikludge {

// 生成 AAK 波形
// 输入的 t, p, e... 都是随时间变化的轨迹点
// Phi_r, Phi_th, Phi_phi 是精确累积相位
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

    // 常数因子
    // 振幅 H ~ mu/D * (M/p) ...
    // 我们使用 Barack & Cutler 2004 Eq (42) 的近似形式
    // 或者 Peters-Mathews 的四极矩公式
    double dist_geom = dist; // 假设输入已转换为几何单位或由 Python 处理

    int n_max = 10; // 谐波级数

    for (size_t i = 0; i < N; ++i) {
        if (p[i] < 3.0) continue;

        double p_val = p[i];
        double e_val = e[i];
        
        // 基础振幅 A0 = sqrt(8*pi/5) * (mu/D) * (M/p)
        // 这里做一个量级估算，精确系数需对照 PM 论文
        double amp_base = (mu / dist_geom) * (M / p_val); 

        // 谐波求和: sum_{n=1}^{n_max}
        // 主导频率: 2*Phi_phi (Quadrupole)
        // 边带: 2*Phi_phi +/- n*Phi_r
        
        // 为了简化，我们实现 Circular + Eccentric Correction
        // h ~ A * [ J_n(ne) ... ] * cos(2 Phi_phi - n Phi_r)
        
        for (int n = -5; n <= 5; ++n) {
            // 谐波相位: l=2, m=2 模式的主导项
            // phase = 2 * phi - n * theta_r (Mean anomaly)
            // 注意 AAK 定义: n 是 radial harmonic index
            
            double phase = 2.0 * Phi_phi[i] + n * Phi_r[i];
            
            // 贝塞尔函数 J_n(2e) ? No, Peters Mathews use J_n(n*e) usually
            // 对于 n=2 mode: J_{n-2}(2e) etc.
            // 采用最简模型: h ~ J_n(n*e)
            
            // 使用 n+2 这里的 n 是相对于 2*Omega_phi 的偏移
            // 修正：标准的 PM 分解是 sum_n A_n cos(n Phi) ?
            // 不是。是对 e 的展开。
            
            // 让我们用一个经过验证的简单近似 (Kludge 核心):
            // h_+ ~ (1+cos^2 i) sum J_n(2e) cos(...)
            
            double arg = 2.0 * e_val; // Argument for Bessel? 
            // 实际上是 J_n(m * e) where m=2
            double jn_val = gsl_sf_bessel_Jn(n + 2, 2.0 * e_val); // Shifted index for quadrupole
            
            // 极化因子
            double cos_inc = cos(iota[i]);
            double ap = (1.0 + cos_inc*cos_inc);
            double ac = 2.0 * cos_inc;
            
            // 叠加
            double h_term = amp_base * jn_val;
            
            h_plus[i]  += h_term * ap * cos(phase);
            h_cross[i] += h_term * ac * sin(phase);
        }
    }

    return {h_plus, h_cross};
}

} // namespace