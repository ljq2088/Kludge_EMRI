#include "aak_waveform.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <gsl/gsl_sf_bessel.h>
#include <omp.h> // <--- 可选：引入 OpenMP 头文件
namespace emrikludge {

// Peters-Mathews 系数结构体
struct PMCoeffs {
    double a; // 对应 m=2 的 cos 分量
    double b; // 对应 m=2 的 sin 分量
    double c; // 对应 m=0 的低频分量
};

// 计算 Peters-Mathews 系数 (Barack & Cutler 2004 Eq. 43)
// 注意：这里的 n 是径向谐波阶数
PMCoeffs compute_pm_coeffs_exact(int n, double e) {
    // 1. 处理 n=0 的特殊情况 (避免 0/0)
    
    if (n == 0) {
        // 当 n=0 时，J_0(0)=1, 其他 J_k(0)=0
        // a = 0, b = 0, c = 2*J_0(0) = 2.0
        return {0.0, 0.0, 2.0};
    }

    // 2. 贝塞尔函数参数: x = n * e
    double x = 2.0* e;
    
    // 我们需要 J_{n-2}, J_{n-1}, J_n, J_{n+1}, J_{n+2}
    // 使用 GSL 的递推计算比多次调用更高效且稳健
    // 但是 GSL 的 array 函数要求 n_min >= 0。由于 n 可以是负数，利用 J_{-n}(x) = (-1)^n J_n(x) 转换
    
    auto get_bessel = [&](int k, double arg) {
        int abs_k = std::abs(k);
        double val = gsl_sf_bessel_Jn(abs_k, arg);
        if (k < 0 && (abs_k % 2 != 0)) val = -val; // J_{-n} = (-1)^n J_n
        return val;
    };

    double jn   = get_bessel(n);
    double jnm1 = get_bessel(n - 1);
    double jnm2 = get_bessel(n - 2);
    double jnp1 = get_bessel(n + 1);
    double jnp2 = get_bessel(n + 2);

    PMCoeffs res;
    
    // Eq 43a: A_n = -n [ J(n-2) - 2e J(n-1) + (2/n)J(n) + 2e J(n+1) - J(n+2) ]
    // 展开化简 (2/n)*J(n)*(-n) = -2 J(n)
    res.a = -n * (jnm2 - 2.0*e*jnm1 + 2.0*e*jnp1 - jnp2) - 2.0 * jn;
    
    // Eq 43b: B_n = -n * sqrt(1-e^2) [ J(n-2) - 2J(n) + J(n+2) ]
    res.b = -n * sqrt(1.0 - e*e) * (jnm2 - 2.0*jn + jnp2);
    
    // Eq 43c: C_n = 2 J(n)
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
    const std::vector<double>& Omega_phi,
    double M, double mu, double dist,
    double viewing_theta, double viewing_phi
) {
    size_t N = t.size();
    std::vector<double> h_plus(N, 0.0);
    std::vector<double> h_cross(N, 0.0);

    double amp_scale = mu / dist; 

    // 观测角度因子 (Source frame polarization basis)
    // 注意：AAK 通常在源坐标系计算 h+, hx，然后由 TDI 模块处理投影。
    // 这里我们只计算 Source Frame 的 h+, hx。
    // viewing_phi 影响相位延迟。
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < N; ++i) {
        if (p[i] < 3.0) continue;

        double e_val = e[i];
        double p_val = p[i];
        double iota_val = iota[i];
        
        // 计算轨道频率 (Mean Motion)
        double Y = 1.0 - e_val*e_val;
        double n_kepler = pow(Y, 1.5) * pow(p_val, -1.5); 
        
        // 基础振幅 (Barack & Cutler 2004 Eq 42 前缀)
        // H ~ (mu/D) * (M * n)^(2/3)
        double freq_val = Omega_phi[i];
        double h_amp = amp_scale * pow(freq_val, 2.0/3.0);

        // 极化几何因子
        double cosi = cos(iota_val);
        double ap = 1.0 + cosi*cosi; // (1 + cos^2 i)
        double ac = 2.0 * cosi;      // (2 cos i)
        double sp = 1.0 - cosi*cosi; // (sin^2 i) = (1 - cos^2 i)

        // === 谐波求和 (Barack & Cutler Eq 42) ===
        // h+ = -A+ Sum [ a cos(Phi_m2) - b sin(Phi_m2) ] + S+ Sum [ c cos(Phi_m0) ]
        // hx = +Ax Sum [ b cos(Phi_m2) + a sin(Phi_m2) ]
        
        // 确定 n 的求和范围
        // 经验公式: n_max ~ 5 / (1-e)
        int n_max = static_cast<int>(10.0 / (1.0 - e_val + 1e-3));
        if (n_max < 10) n_max = 10;
        if (n_max > 50) n_max = 50; // 限制最大计算量

        for (int n = -n_max; n <= n_max; ++n) {
            PMCoeffs pm = compute_pm_coeffs_exact(n, e_val);
            
            // 忽略太小的项 (性能优化)
            if (std::abs(pm.a) < 1e-9 && std::abs(pm.b) < 1e-9 && std::abs(pm.c) < 1e-9) continue;

            // 1. 四极矩项 (m=2): 频率 2*Phi_phi + n*Phi_r
            // 注意：BC04 中相位定义为 2*gamma - n*lambda. 
            // 对应我们的 2*Phi_phi - n*Phi_r.
            double phase_m2 = 2.0 * Phi_phi[i] - n * Phi_r[i];
            double wave_phase_m2 = phase_m2 - 2.0 * viewing_phi;
            
            double c2 = cos(wave_phase_m2);
            double s2 = sin(wave_phase_m2);
            
            // 2. 低频记忆项 (m=0): 频率 n*Phi_r
            // 注意：c_n 项的相位是 n*lambda (即 n*Phi_r)
            double phase_m0 = n * Phi_r[i]; 
            // m=0 项不受 viewing_phi 影响 (轴对称)
            double c0 = cos(phase_m0);
            
            // 累加 h+
            // Term 1 (m=2): -(1+cos^2)*[a*cos - b*sin]
            h_plus[i] -= h_amp * ap * (pm.a * c2 - pm.b * s2);
            // Term 2 (m=0): +(1-cos^2)*c*cos
            h_plus[i] += h_amp * sp * pm.c * c0;

            // 累加 hx
            // Term 1 (m=2 only): +2cos*[b*cos + a*sin]
            h_cross[i] += h_amp * ac * (pm.b * c2 + pm.a * s2);
        }
    }

    return {h_plus, h_cross};
}

} // namespace