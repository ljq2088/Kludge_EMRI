#include "aak_waveform.hpp"
#include <cmath>
#include <vector>
#include <complex>
#include <gsl/gsl_sf_bessel.h> 

namespace emrikludge {

// 辅助函数：计算 A+, Ax 系数 (Peters & Mathews 1963 / Barack & Cutler 2004)
// 这里的公式较为繁琐，我们实现主导项
void compute_amplitudes(double n, double e, double &A, double &B, double &C) {
    // Bessel functions J_n(ne)
    double ne = n * e;
    double J_n = gsl_sf_bessel_Jn(static_cast<int>(n), ne);
    double J_nm1 = gsl_sf_bessel_Jn(static_cast<int>(n-1), ne);
    double J_np1 = gsl_sf_bessel_Jn(static_cast<int>(n+1), ne);
    
    // Recursive relations for J_(n-2), J_(n+2) if needed, or simple approx
    // A = -n * (J(n-2) - 2e J(n-1) + (2/n)J(n) + 2e J(n+1) - J(n+2))
    // 简化版 (Leading order in e)
    A = J_n; // Placeholder for complex PM formula
    B = J_n; 
    C = J_n;
}

// 核心波形生成函数
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

    // 常数
    double M_sec = M * 4.925491e-6; // Solar mass in seconds
    double dist_sec = dist; // Gpc converted to seconds/meters outside
    double amp_scale = mu / dist; // mu/D

    // 谐波求和截断
    int n_max = 10; // 径向谐波 (n)
    // int l_max = 2;  // 极向谐波 (l=2 usually dominant for quadrupole)
    // int m_max = 2;  // 方位角谐波 (m=2)

    // 简化版 AAK 波形求和 (基于 Barack & Cutler 2004, Eq 42)
    // h ~ Sum_n A_n * cos(phi_n)
    // phi_n = 2*Phi_phi - n*Phi_r (Simplified quadrupole model)
    
    for (size_t i = 0; i < N; ++i) {
        if (p[i] < 3.0) continue; // Plunge

        double omega_orb = pow(1.0/p[i], 1.5); // Simplified Keplerian n
        // 振幅因子 A ~ omega^(2/3)
        double h_amp = amp_scale * pow(omega_orb, 2.0/3.0); 

        // 对谐波求和 n = 1..10
        for (int n = 1; n <= n_max; ++n) {
            // 相位: 主导项是 m=2 (Quadrupole)
            // Phase = 2 * Phi_phi + n * Phi_r (Eccentric harmonics)
            // 注意符号定义
            double phase_n = 2.0 * Phi_phi[i] - n * Phi_r[i];
            
            // Bessel 权重 J_n(n*e)
            double ne_arg = n * e[i];
            double Jn = gsl_sf_bessel_Jn(n, ne_arg);
            
            // 简单的 Plus/Cross 极化组合 (Toy Model for demonstration)
            // 真实实现需要引入 inclination 和 viewing angle 的复杂三角函数 (Spheroidal Harmonics)
            // 参考 alvincjk AAK.cc 中的 `FplusI * Aplus`
            
            double A_n = h_amp * Jn; // 粗略振幅
            
            h_plus[i]  += A_n * cos(phase_n);
            h_cross[i] += A_n * sin(phase_n);
        }
    }

    return {h_plus, h_cross};
}

} // namespace