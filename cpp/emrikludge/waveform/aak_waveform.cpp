#include "aak_waveform.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_blas.h>
#include <omp.h>

namespace emrikludge {

// ... (compute_geometry 和 compute_rot_coeffs 保持不变，无需修改) ...
// 请保留原有的 compute_geometry 和 compute_rot_coeffs 函数

void cross_product(const double u[3], const double v[3], double w[3]) {
    w[0] = u[1]*v[2] - u[2]*v[1];
    w[1] = u[2]*v[0] - u[0]*v[2];
    w[2] = u[0]*v[1] - u[1]*v[0];
}

double dot_product(const double u[3], const double v[3]) {
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

double vector_norm(const double u[3]) {
    return std::sqrt(dot_product(u, u));
}

void compute_geometry(double iota, double alpha, 
                      double theta_S, double phi_S, 
                      double theta_K, double phi_K,
                      double &beta, double &Ldotn) 
{
    double coslam = std::cos(iota);
    double sinlam = std::sin(iota);
    double cosalp = std::cos(alpha);
    double sinalp = std::sin(alpha);
    double cosqS = std::cos(theta_S);
    double sinqS = std::sin(theta_S);
    double cosqK = std::cos(theta_K);
    double sinqK = std::sin(theta_K);
    double cosphiK = std::cos(phi_K);
    double sinphiK = std::sin(phi_K);
    
    double cosqL = cosqK * coslam + sinqK * sinlam * cosalp;
    double sinqL = std::sqrt(std::max(0.0, 1.0 - cosqL*cosqL));
    
    double phiLup   = sinqK*sinphiK*coslam - cosphiK*sinlam*sinalp - cosqK*sinphiK*sinlam*cosalp;
    double phiLdown = sinqK*cosphiK*coslam + sinphiK*sinlam*sinalp - cosqK*cosphiK*sinlam*cosalp;
    double phiL     = std::atan2(phiLup, phiLdown);
    
    Ldotn = cosqL * cosqS + sinqL * sinqS * std::cos(phiL - phi_S);
    
    double Sdotn = cosqK * cosqS + sinqK * sinqS * std::cos(phi_K - phi_S);
    
    double betaup   = -Sdotn + coslam * Ldotn;
    double betadown = sinqS * std::sin(phi_K - phi_S) * sinlam * cosalp + 
                      (cosqK * Sdotn - cosqS) / (sinqK + 1e-15) * sinlam * sinalp; 
    
    beta = std::atan2(betaup, betadown);
}

void compute_rot_coeffs(double iota, double alpha,
                        double theta_S, double phi_S,
                        double theta_K, double phi_K,
                        double rot[4]) 
{
    double n[3], L[3], S[3];
    n[0] = std::sin(theta_S) * std::cos(phi_S);
    n[1] = std::sin(theta_S) * std::sin(phi_S);
    n[2] = std::cos(theta_S);
    S[0] = std::sin(theta_K) * std::cos(phi_K);
    S[1] = std::sin(theta_K) * std::sin(phi_K);
    S[2] = std::cos(theta_K);
    
    double coslam = std::cos(iota);
    double sinlam = std::sin(iota);
    double cosalp = std::cos(alpha);
    double sinalp = std::sin(alpha);
    double cosqK = std::cos(theta_K);
    double sinqK = std::sin(theta_K);
    double cosphiK = std::cos(phi_K);
    double sinphiK = std::sin(phi_K);

    L[0] = coslam*sinqK*std::cos(phi_K) + sinlam*(sinalp*std::sin(phi_K) - cosalp*cosqK*std::cos(phi_K));
    L[1] = coslam*sinqK*std::sin(phi_K) - sinlam*(sinalp*std::cos(phi_K) + cosalp*cosqK*std::sin(phi_K));
    L[2] = coslam*cosqK + sinlam*cosalp*sinqK;
    
    double nxL[3], nxS[3];
    cross_product(n, L, nxL);
    cross_product(n, S, nxS);
    
    double norm = vector_norm(nxL) * vector_norm(nxS);
    if (norm < 1e-12) {
        rot[0] = 1.0; rot[1] = 0.0; rot[2] = 0.0; rot[3] = 1.0;
        return;
    }
    
    double dot = dot_product(nxL, nxS);
    double cosrot = dot / norm;
    double term1 = dot_product(L, nxS);
    double term2 = dot_product(S, nxL);
    double sinrot = (term1 - term2) / norm;
    
    rot[0] = 2.0 * cosrot * cosrot - 1.0;
    rot[1] = 2.0 * cosrot * sinrot;
    rot[2] = -rot[1];
    rot[3] = rot[0];
}

struct PMCoeffs {
    double a, b, c;
};

// 计算 Peters-Mathews 系数 (Strict AAK loop)
PMCoeffs compute_pm_coeffs_strict(int n, double e, double Amp) {
    // [Fix 1]: 强制 e 为正数，防止数值误差导致 GSL 报错
    double e_safe = std::abs(e);
    // [Fix 2]: 参数必须是 n*e (Strict Bessel Argument)
    double x = static_cast<double>(n) * e_safe;
    
    // [Fix 3]: 处理 n=0 的情况 (虽然主循环从1开始，但为了鲁棒性)
    if (n == 0) return {0.0, 0.0, 0.0};

    // 统一使用标量计算 J_{n-2} 到 J_{n+2}
    // 避免使用 gsl_sf_bessel_Jn_array 带来的潜在 domain error
    double J[5]; 
    for (int k = 0; k < 5; ++k) {
        // 我们需要 J[0] -> n-2, J[1] -> n-1, J[2] -> n, J[3] -> n+1, J[4] -> n+2
        int order = n - 2 + k;
        int abs_order = std::abs(order);
        
        // 调用标量函数 (非常稳健)
        double val = gsl_sf_bessel_Jn(abs_order, x);
        
        // 处理负阶数: J_{-m}(x) = (-1)^m J_m(x)
        // 如果 order 是负奇数，变号
        if (order < 0 && (abs_order % 2 != 0)) {
            val = -val;
        }
        J[k] = val;
    }

    PMCoeffs res;
    // AAK.cc lines 330-332
    double term_bracket_a = J[0] - 2.0*e_safe*J[1] + (2.0/n)*J[2] + 2.0*e_safe*J[3] - J[4];
    res.a = -n * Amp * term_bracket_a;

    double term_bracket_b = J[0] - 2.0*J[2] + J[4];
    res.b = -n * Amp * std::sqrt(std::max(0.0, 1.0 - e_safe*e_safe)) * term_bracket_b;

    res.c = 2.0 * Amp * J[2];

    return res;
}

std::pair<std::vector<double>, std::vector<double>> 
generate_aak_waveform_cpp(
    const std::vector<double>& t,
    const std::vector<double>& p,
    const std::vector<double>& e,
    const std::vector<double>& iota,
    const std::vector<double>& M_map_vec, 
    const std::vector<double>& a_map_vec, 
    const std::vector<double>& Phi_r,
    const std::vector<double>& Phi_th,
    const std::vector<double>& Phi_phi,
    const std::vector<double>& Omega_phi,
    double M_phys, double mu, double dist,
    double viewing_theta, double viewing_phi
) {
    size_t N = t.size();
    std::vector<double> h_plus(N, 0.0);
    std::vector<double> h_cross(N, 0.0);

    double qS = viewing_theta;
    double phiS = viewing_phi;
    double qK = 0.0; 
    double phiK = 0.0;

    double amp_scale = mu / dist; 
    double max_e = 0.0;
    for (double val : e) if (val > max_e) max_e = val;
    
    int n_max_global = std::max(20, static_cast<int>(30.0 * max_e));
    if (n_max_global > 50) n_max_global = 50; // 硬上限防止太慢


    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < N; ++i) {
        if (p[i] < 2.1) continue;

        double e_val = e[i];
        
        // [Fix 4]: 最终的数值保护，防止 e 变成 -1e-18
        if (e_val < 0.0) e_val = 0.0;
        if (e_val > 0.999) e_val = 0.999;

        double iota_val = iota[i];
        double freq_val = Omega_phi[i];

        // 1. 恢复基本相位
        double gim = Phi_th[i] - Phi_r[i];
        double alp = Phi_phi[i] - Phi_th[i];
        double Phi = Phi_r[i];

        // 2. 几何计算
        double beta, Ldotn;
        compute_geometry(iota_val, alp, qS, phiS, qK, phiK, beta, Ldotn);
        
        double gam = 2.0 * (gim + beta);
        double cos2gam = std::cos(gam);
        double sin2gam = std::sin(gam);
        double Ldotn2 = Ldotn * Ldotn;

        // 3. 振幅标度
        double h_amp = amp_scale * std::pow(freq_val, 2.0/3.0);

        // 4. 旋转矩阵
        double rot[4];
        compute_rot_coeffs(iota_val, alp, qS, phiS, qK, phiK, rot);

        // 5. 谐波求和
        int n_max = std::max(20, static_cast<int>(30.0 * e_val));
        if (n_max < 4) n_max = 4;
        if (n_max > 50) n_max = 50;

        for (int n = 1; n <= n_max; ++n) {
            // 这里传入正的 e_val，保证 GSL 不会报错
            PMCoeffs pm = compute_pm_coeffs_strict(n, e_val, h_amp);
            
            double nPhi = n * Phi;
            double cn = std::cos(nPhi);
            double sn = std::sin(nPhi);
            
            double term_a = pm.a * cn;
            double term_b = pm.b * sn;
            double term_c = pm.c * cn;

            double Aplus = -(1.0 + Ldotn2) * (term_a * cos2gam - term_b * sin2gam) 
                           + term_c * (1.0 - Ldotn2);
            double Acros = 2.0 * Ldotn * (term_b * cos2gam + term_a * sin2gam);

            h_plus[i]  += Aplus * rot[0] + Acros * rot[1];
            h_cross[i] += Aplus * rot[2] + Acros * rot[3];
        }
    }

    return {h_plus, h_cross};
}

} // namespace
// #include "aak_waveform.hpp"
// #include <cmath>
// #include <vector>
// #include <algorithm>
// #include <gsl/gsl_sf_bessel.h>
// #include <omp.h> // <--- 可选：引入 OpenMP 头文件
// namespace emrikludge {

// // Peters-Mathews 系数结构体
// struct PMCoeffs {
//     double a; // 对应 m=2 的 cos 分量
//     double b; // 对应 m=2 的 sin 分量
//     double c; // 对应 m=0 的低频分量
// };

// // 计算 Peters-Mathews 系数 (Barack & Cutler 2004 Eq. 43)
// // 注意：这里的 n 是径向谐波阶数
// PMCoeffs compute_pm_coeffs_exact(int n, double e) {
//     // 1. 处理 n=0 的特殊情况 (避免 0/0)
    
//     // if (n == 0) {
//     //     // 当 n=0 时，J_0(0)=1, 其他 J_k(0)=0
//     //     // a = 0, b = 0, c = 2*J_0(0) = 2.0
//     //     return {0.0, 0.0, 2.0};
//     // }

//     // 2. 贝塞尔函数参数: x = n * e
//     double x =n * e;
    
//     // 我们需要 J_{n-2}, J_{n-1}, J_n, J_{n+1}, J_{n+2}
//     // 使用 GSL 的递推计算比多次调用更高效且稳健
//     // 但是 GSL 的 array 函数要求 n_min >= 0。由于 n 可以是负数，利用 J_{-n}(x) = (-1)^n J_n(x) 转换
    
//     // auto get_bessel = [&](int k, double arg) {
//     //     int abs_k = std::abs(k);
//     //     double val = gsl_sf_bessel_Jn(abs_k, arg);
//     //     if (k < 0 && (abs_k % 2 != 0)) val = -val; // J_{-n} = (-1)^n J_n
//     //     return val;
//     // };
//     auto get_J = [&](int k) {
//         int abs_k = std::abs(k);
//         double val = gsl_sf_bessel_Jn(abs_k, x); 
//         if (k < 0 && (abs_k % 2 != 0)) val = -val; 
//         return val;
//     };

//     double jn   = get_J(n);
//     double jnm1 = get_J(n - 1);
//     double jnm2 = get_J(n - 2);
//     double jnp1 = get_J(n + 1);
//     double jnp2 = get_J(n + 2);
//     PMCoeffs res;
    
//     // Eq 43a: A_n = -n [ J(n-2) - 2e J(n-1) + (2/n)J(n) + 2e J(n+1) - J(n+2) ]
//     // 展开化简 (2/n)*J(n)*(-n) = -2 J(n)
//     res.a = -n * (jnm2 - 2.0*e*jnm1 + 2.0*e*jnp1 - jnp2) - 2.0 * jn;
    
//     // Eq 43b: B_n = -n * sqrt(1-e^2) [ J(n-2) - 2J(n) + J(n+2) ]
//     res.b = -n * sqrt(1.0 - e*e) * (jnm2 - 2.0*jn + jnp2);
    
//     // Eq 43c: C_n = 2 J(n)
//     res.c = 2.0 * jn;
    
//     return res;
// }

// std::pair<std::vector<double>, std::vector<double>> 
// generate_aak_waveform_cpp(
//     const std::vector<double>& t,
//     const std::vector<double>& p,
//     const std::vector<double>& e,
//     const std::vector<double>& iota,
//     const std::vector<double>& Phi_r,
//     const std::vector<double>& Phi_th,
//     const std::vector<double>& Phi_phi,
//     const std::vector<double>& Omega_phi,
//     double M, double mu, double dist,
//     double viewing_theta, double viewing_phi
// ) {
//     size_t N = t.size();
//     std::vector<double> h_plus(N, 0.0);
//     std::vector<double> h_cross(N, 0.0);

//     double amp_scale = mu / dist; 

//     // 观测角度因子 (Source frame polarization basis)
//     // 注意：AAK 通常在源坐标系计算 h+, hx，然后由 TDI 模块处理投影。
//     // 这里我们只计算 Source Frame 的 h+, hx。
//     // viewing_phi 影响相位延迟。
//     #pragma omp parallel for schedule(guided)
//     for (size_t i = 0; i < N; ++i) {
//         if (p[i] < 3.0) continue;

//         double e_val = e[i];
//         double p_val = p[i];
//         double iota_val = iota[i];
        
//         // 计算轨道频率 (Mean Motion)
//         double Y = 1.0 - e_val*e_val;
//         double n_kepler = pow(Y, 1.5) * pow(p_val, -1.5); 
        
//         // 基础振幅 (Barack & Cutler 2004 Eq 42 前缀)
//         // H ~ (mu/D) * (M * n)^(2/3)
//         double freq_val = Omega_phi[i];
//         double h_amp = amp_scale * pow(freq_val, 2.0/3.0);

//         // 极化几何因子
//         double cosi = cos(iota_val);
//         double ap = 1.0 + cosi*cosi; // (1 + cos^2 i)
//         double ac = 2.0 * cosi;      // (2 cos i)
//         double sp = 1.0 - cosi*cosi; // (sin^2 i) = (1 - cos^2 i)

//         // === 谐波求和 (Barack & Cutler Eq 42) ===
//         // h+ = -A+ Sum [ a cos(Phi_m2) - b sin(Phi_m2) ] + S+ Sum [ c cos(Phi_m0) ]
//         // hx = +Ax Sum [ b cos(Phi_m2) + a sin(Phi_m2) ]
        
//         // 确定 n 的求和范围
//         // 经验公式: n_max ~ 5 / (1-e)
//         double gamma_phase = Phi_phi[i] - Phi_r[i];
        
//         // 引入 viewing_phi (观测方位角) 修正 gamma
//         // 相当于旋转了 source frame
//         double effective_gamma = gamma_phase - viewing_phi; 
        
//         double gam2 = 2.0 * effective_gamma;
//         double c2g = cos(gam2);
//         double s2g = sin(gam2);
//         int n_max = static_cast<int>(10.0 / (1.0 - e_val + 1e-3));
//         if (n_max < 10) n_max = 10;
//         if (n_max > 50) n_max = 50; // 限制最大计算量

//         for (int n = -n_max; n <= n_max; ++n) {
//             PMCoeffs pm = compute_pm_coeffs_exact(n, e_val);
            
//             // 忽略太小的项 (性能优化)
//             if (std::abs(pm.a) < 1e-9 && std::abs(pm.b) < 1e-9 && std::abs(pm.c) < 1e-9) continue;

//             // // 1. 四极矩项 (m=2): 频率 2*Phi_phi + n*Phi_r
//             // // 注意：BC04 中相位定义为 2*gamma - n*lambda. 
//             // // 对应我们的 2*Phi_phi - n*Phi_r.
//             // double phase_m2 = 2.0 * Phi_phi[i] - n * Phi_r[i];
//             // double wave_phase_m2 = phase_m2 - 2.0 * viewing_phi;
            
//             // double c2 = cos(wave_phase_m2);
//             // double s2 = sin(wave_phase_m2);
            
//             // // 2. 低频记忆项 (m=0): 频率 n*Phi_r
//             // // 注意：c_n 项的相位是 n*lambda (即 n*Phi_r)
//             // double phase_m0 = n * Phi_r[i]; 
//             // // m=0 项不受 viewing_phi 影响 (轴对称)
//             // double c0 = cos(phase_m0);
            
//             // // 累加 h+
//             // // Term 1 (m=2): -(1+cos^2)*[a*cos - b*sin]
//             // h_plus[i] -= h_amp * ap * (pm.a * c2 - pm.b * s2);
//             // // Term 2 (m=0): +(1-cos^2)*c*cos
//             // h_plus[i] += h_amp * sp * pm.c * c0;

//             // // 累加 hx
//             // // Term 1 (m=2 only): +2cos*[b*cos + a*sin]
//             // h_cross[i] += h_amp * ac * (pm.b * c2 + pm.a * s2);
//             // 径向相位 harmonics
//             double nPhi = n * Phi_r[i];
//             double cn = cos(nPhi);
//             double sn = sin(nPhi);

//             // 构建 AAK.cc 中的 Aplus 和 Acros (Line 333-334)
//             // Aplus = -(1+cos^2)*(a*cos2g - b*sin2g) + (1-cos^2)*c
//             // 注意：a 包含了 cos(nPhi), b 包含了 sin(nPhi), c 包含了 cos(nPhi)
//             // 我们需要在这里把 cos(nPhi) 乘进去
            
//             // 展开 pm.a, pm.b, pm.c 对应的时变部分
//             // pm.a (from AAK.cc 'a') 应该乘以 cos(nPhi)
//             // pm.b (from AAK.cc 'b') 应该乘以 sin(nPhi)
//             // pm.c (from AAK.cc 'c') 应该乘以 cos(nPhi)
            
//             double term_a = pm.a * cn;
//             double term_b = pm.b * sn;
//             double term_c = pm.c * cn;

//             // 组合 (Source Frame h+, hx)
//             // h+ = - (1+cos^2 i) * [ term_a * cos(2g) - term_b * sin(2g) ] + sin^2 i * term_c
//             double h_plus_n = -ap * (term_a * c2g - term_b * s2g) + sp * term_c;
            
//             // hx = + 2 cos i * [ term_b * cos(2g) + term_a * sin(2g) ]
//             double h_cross_n = ac * (term_b * c2g + term_a * s2g);

//             h_plus[i]  += h_amp * h_plus_n;
//             h_cross[i] += h_amp * h_cross_n;
//         }
//     }

//     return {h_plus, h_cross};
// }

// } // namespace