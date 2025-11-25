#include "aak_waveform.hpp"
#include <cmath>
#include <vector>
#include <gsl/gsl_sf_bessel.h>

namespace emrikludge {

// 辅助：计算 Peters-Mathews 谐波系数 (a, b, c)
// 来源: Barack & Cutler 2004 Eq. (43)
// n: harmonic number
// e: eccentricity
// gam: Bessel J_n(ne)
struct PMCoeffs {
    double a, b, c;
};

PMCoeffs compute_pm_coeffs(int n, double e) {
    double ne = n * e;
    // 我们需要 J_{n-2}, J_{n-1}, J_n, J_{n+1}, J_{n+2}
    // GSL array 计算更高效，但这里简单起见分别调用
    double jn   = gsl_sf_bessel_Jn(n, ne);
    double jnm1 = gsl_sf_bessel_Jn(n - 1, ne);
    double jnm2 = gsl_sf_bessel_Jn(n - 2, ne);
    double jnp1 = gsl_sf_bessel_Jn(n + 1, ne);
    double jnp2 = gsl_sf_bessel_Jn(n + 2, ne);

    PMCoeffs res;
    // Eq 43a
    res.a = -n * jnm2 + 2.0 * n * e * jnm1 - 2.0 * jn - 2.0 * n * e * jnp1 + n * jnp2;
    // Eq 43b
    res.b = -n * sqrt(1.0 - e * e) * (jnm2 - 2.0 * jn + jnp2);
    // Eq 43c
    res.c = 2.0 * jn;
    
    return res;
}

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

    // 几何常数
    // 假设距离 dist 已经是几何单位，或者在外部处理了单位
    // 如果 dist 是 Gpc，这里需要转换因子。
    // 我们的约定：C++ 内部尽可能处理无量纲或几何单位。
    double amp_scale = mu / dist;

    // 观测角度 (Sky position of Source)
    // Barack & Cutler 使用 (theta_S, phi_S)
    double cost = cos(viewing_theta);
    double sint = sin(viewing_theta);
    double cosp = cos(viewing_phi);
    double sinp = sin(viewing_phi);

    // 谐波截断
    int n_max = 10; 

    for (size_t i = 0; i < N; ++i) {
        if (p[i] < 3.0) continue;

        double e_val = e[i];
        double p_val = p[i];
        double iota_val = iota[i];
        
        // 轨道频率 (Mean motion in flat space)
        // omega = (M / a^3)^0.5 = (1-e^2)^1.5 * p^-1.5
        double Y = 1.0 - e_val*e_val;
        double n_kepler = pow(Y, 1.5) * pow(p_val, -1.5); 
        
        // AAK 振幅基准: sqrt(8*pi/5) * (mu/D) * (M*omega)^(2/3) -> 简化为 (M*n)^(2/3)
        double h_amp = amp_scale * pow(n_kepler, 2.0/3.0);

        // 极化向量 (Polarization Tensors)
        // 我们需要将轨道坐标系 (L // z) 旋转到波坐标系
        // 这是一个复杂的旋转。为了 MVP，我们使用简单的 "Face-on" 或 "Edge-on" 近似公式，
        // 或者直接使用 Peters-Mathews 的 Plus/Cross 定义 (假设视线方向合适)
        
        double cosi = cos(iota_val);
        double sini = sin(iota_val);
        
        // 循环谐波 n
        // 频率: omega_n = 2*Omega_phi + n*Omega_r (Dominant quadrupole)
        // 相位: Phi_n = 2*Phi_phi + n*Phi_r
        
        for (int n = -5; n <= 5; ++n) {
            // 这里的 n 对应 PM 论文中的 n。
            // 频率成分是 n * Omega_r? 
            // 不，PM 分解是 sum_n { A_n cos(2phi - n phi_r) } (Fundamental m=2)
            
            PMCoeffs pm = compute_pm_coeffs(n + 2, e_val); // Shift index if needed?
            // 实际上 PM 公式中 n 是相对于 2*phi 的偏移。
            // 让我们严格遵循 Barack & Cutler Eq 42:
            // h = sum A_n cos( 2*Phi_phi - n*Phi_r )
            
            double phase = 2.0 * Phi_phi[i] - n * Phi_r[i];
            
            // 极化模式 (Simplified BC04)
            // H+ ~ - (1 + cos^2 i) [ a cos(2gamma) - b sin(2gamma) ] + (1-cos^2 i) c
            // Hx ~ 2 cos i [ b cos(2gamma) + a sin(2gamma) ]
            // 这里的 2gamma 就是我们的 phase
            
            double cos_phase = cos(phase);
            double sin_phase = sin(phase);
            
            double term_plus = -(1.0 + cosi*cosi) * (pm.a * cos_phase - pm.b * sin_phase) + (1.0 - cosi*cosi) * pm.c;
            double term_cross = 2.0 * cosi * (pm.b * cos_phase + pm.a * sin_phase);
            
            h_plus[i]  += h_amp * term_plus;
            h_cross[i] += h_amp * term_cross;
        }
    }

    return {h_plus, h_cross};
}

} // namespace