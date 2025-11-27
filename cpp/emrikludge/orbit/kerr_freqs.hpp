#pragma once

#include <cmath>
#include <algorithm>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_elljac.h>
#include "nk_orbit.hpp"
#include "../emri_params.hpp"

namespace emrikludge {

struct KerrFreqs {
    double Omega_r;
    double Omega_theta;
    double Omega_phi;
    double Gamma;
};

class KerrFundamentalFrequencies {
private:
    // 辅助函数：计算径向平均值 <1/(r-p)>
    // 基于 Schmidt (2002) / Fujita & Hikida (2009)
    // mapping: r(u) = (r2(r1-r3) - r3(r1-r2)sn^2) / ((r1-r3) - (r1-r2)sn^2)
    static double avg_inv_r_minus_p(double p, double r1, double r2, double r3, double r4, 
                                    double k_r, double K_r) {
        double h = (r1 - r2) / (r1 - r3);
        
        // 系数定义 (参考 Schmidt 2002 附录或者直接推导)
        double C0 = (r2 - p) * (r1 - r3);
        double C2 = (r3 - p) * (r1 - r2);
        double D0 = (r1 - r3);
        double D2 = (r1 - r2);
        
        // 特征参数 n_p
        double n_p = C2 / C0; 

        // 积分部分: c_const * K + c_pole * Pi(n_p)
        // 原始积分结果 I_p = int_0^K du / (r(u) - p)
        // I_p = (D2/C2) * K + (D0/C0 - D2/C2) * Pi(n_p, k)
        
        double term_Pi = 0.0;
        // GSL Pi 定义: int dt / ((1 - n sin^2 t) sqrt(...))
        // 注意 n < 1 的限制。对于束缚轨道且 p 在视界附近，通常安全。
        if (std::abs(n_p) < 1e-9) {
            term_Pi = K_r; // Pi(0, k) = K(k)
        } else {
            term_Pi = gsl_sf_ellint_Pcomp(n_p, k_r, GSL_PREC_DOUBLE);
        }

        double val_int = (D2 / C2) * K_r + (D0 / C0 - D2 / C2) * term_Pi;
        
        // 平均值 = Integral / K_r
        return val_int / K_r;
    }

public:
    static KerrFreqs compute(double M, double a, double p, double e, double iota) {
        // 1. 获取守恒量与根
        KerrConstants k_const = BabakNKOrbit::get_conserved_quantities(M, a, p, e, iota);
        if (k_const.E == 0.0) return {0,0,0,0};

        double E = k_const.E; double Lz = k_const.Lz; // double Q = k_const.Q;
        double r1 = k_const.r_a; double r2 = k_const.r_p; 
        double r3 = k_const.r3; double r4 = k_const.r4;
        double z_minus = k_const.z_minus; double z_plus = k_const.z_plus;
        double a2 = a*a; double E2 = E*E; double beta = a2 * (1.0 - E2);

        // --- 2. 径向周期参数 (Radial) ---
        double num_kr = (r1 - r2) * (r3 - r4);
        double den_kr = (r1 - r3) * (r2 - r4);
        double kr2 = std::max(0.0, std::min(1.0, num_kr / den_kr));
        double kr = sqrt(kr2);
        double Kr = gsl_sf_ellint_Kcomp(kr, GSL_PREC_DOUBLE);
        
        // Mino time 径向基本周期 Lambda_r_0 (除去 Gamma 因子)
        // Lambda_r = 2 * J_0 = 4 * K / sqrt(...)
        double sqrt_coef = sqrt((1.0 - E2) * (r1 - r3) * (r2 - r4)); // 注意这里有个因子 1-E^2 在 dr/dtau 公式里可能被处理了
        // 标准公式: dr/dlambda = sqrt(R). R = (1-E^2)(r-r1)(r-r2)(r-r3)(r-r4).
        // period Lambda = 2 * int dr / sqrt(R) = 4 * K(k) / sqrt((1-E^2)(r1-r3)(r2-r4))
        
        double prefactor_r = sqrt((1.0 - E2) * (r1 - r3) * (r2 - r4));
        double Lambda_r = 4.0 * Kr / prefactor_r;
        double Upsilon_r = 2.0 * M_PI / Lambda_r;

        // --- 3. 极向周期参数 (Polar) ---
        double kth2 = std::max(0.0, std::min(1.0, z_minus / z_plus));
        double kth = sqrt(kth2);
        double Kth = gsl_sf_ellint_Kcomp(kth, GSL_PREC_DOUBLE);
        double Eth = gsl_sf_ellint_Ecomp(kth, GSL_PREC_DOUBLE);
        
        double prefactor_th = sqrt(beta * z_plus); // dtheta/dlambda = sqrt(Th)
        double Lambda_th = 4.0 * Kth / prefactor_th;
        double Upsilon_th = 2.0 * M_PI / Lambda_th;

        // --- 4. 极向平均值 (Analytical) ---
        // <cos^2 theta>
        double z_avg = 0.0;
        if (kth2 > 1e-10) {
            z_avg = z_minus * (1.0 + (Eth/Kth - 1.0) / kth2);
        } else {
            z_avg = 0.5 * z_minus;
        }
        
        // Gamma_theta = E * a^2 * <cos^2 theta>
        double Gamma_theta_val = E * a2 * z_avg;

        // Upsilon_phi_theta = Lz * <1 / sin^2 theta> = Lz * <1/(1-z)>
        double n_z = z_minus; 
        double Pi_z = gsl_sf_ellint_Pcomp(n_z, kth, GSL_PREC_DOUBLE);
        double avg_inv_sin2 = Pi_z / Kth;
        double Upsilon_phi_theta_val = Lz * avg_inv_sin2;

        // --- 5. 径向平均值 (Fully Analytical) ---
        // 我们需要计算 <r>, <1/Delta>, <r/Delta>
        // 1/Delta = A/(r - r_+) + B/(r - r_-)
        
        double r_plus = M + sqrt(M*M - a2);
        double r_minus = M - sqrt(M*M - a2);
        double denom_roots = r_plus - r_minus;

        // 计算 <1/(r - r_+)> 和 <1/(r - r_-)>
        double avg_inv_r_rp = avg_inv_r_minus_p(r_plus, r1, r2, r3, r4, kr, Kr);
        double avg_inv_r_rm = avg_inv_r_minus_p(r_minus, r1, r2, r3, r4, kr, Kr);

        // 组合得到 <1/Delta>
        // 1/Delta = (1/(r-r_+) - 1/(r-r_-)) / (r_+ - r_-)
        double avg_inv_Delta = (avg_inv_r_rp - avg_inv_r_rm) / denom_roots;

        // 组合得到 <r/Delta>
        // r/Delta = ( r_+/(r-r_+) - r_-/(r-r_-) ) / (r_+ - r_-)
        double avg_r_Delta = (r_plus * avg_inv_r_rp - r_minus * avg_inv_r_rm) / denom_roots;

        // 计算 <r>
        // r(u) = r3 + (r2-r3) / (1 - h sn^2)
        // <r> = r3 + (r2-r3) * Pi(h, k) / K
        double h_mod = (r1 - r2) / (r1 - r3);
        double Pi_h = gsl_sf_ellint_Pcomp(h_mod, kr, GSL_PREC_DOUBLE);
        double avg_r = r3 + (r2 - r3) * Pi_h / Kr;

        // 计算 <r^2 / Delta>
        // r^2/Delta = 1 + (2Mr - a^2)/Delta = 1 + 2M(r/Delta) - a^2(1/Delta)
        double avg_r2_Delta = 1.0 + 2.0*M*avg_r_Delta - a2*avg_inv_Delta;

        // --- 6. 组装最终频率 ---
        
        // [A] Gamma (Mino time to Coordinate time conversion)
        // Gamma = <T_r> + <T_th>
        // T_r = ((r^2+a^2)/Delta) * P(r) - E*r^2
        // P(r) = E(r^2+a^2) - aLz
        // 展开: T_r = E( (r^2+a^2)^2/Delta - r^2 ) - aLz( (r^2+a^2)/Delta )
        // 利用 (r^2+a^2)/Delta = 1 + 2Mr/Delta
        // 以及 (r^2+a^2)^2/Delta = (r^2+a^2)(1 + 2Mr/Delta) = r^2+a^2 + 2M(r^3+a^2r)/Delta ... 稍显复杂
        // 更简单的形式 (Drasco 2004 Eq 2.6):
        // T_r = 4*M*E*r*(r^2+a^2)/Delta + E*( ... ) ...
        // 直接用我们算出的平均值组装:
        // T_r = E [ (r^2+a^2)(1 + 2Mr/Delta) - r^2 ] - aLz [ 1 + 2Mr/Delta ]
        //     = E [ r^2 + a^2 + 2Mr^3/Delta + 2Ma^2r/Delta - r^2 ] - aLz [ ... ]
        //     = E [ a^2 + 2M(r^2 * r/Delta ? No) ] 
        // 让我们回退到最基础的线性组合:
        // T_r = E * <(r^2+a^2)^2/Delta> - E * <r^2> - a*Lz * <(r^2+a^2)/Delta>
        // 其中 <(r^2+a^2)/Delta> = <1> + 2M<r/Delta> = 1 + 2M * avg_r_Delta
        // 其中 <(r^2+a^2)^2/Delta> = <r^2+a^2> + 2M<r(r^2+a^2)/Delta> = <r^2> + a^2 + 2M<r^3/Delta> + 2Ma^2<r/Delta>
        // 这需要 <r^3/Delta>... 
        // 换条路:
        // T_r = (r^2+a^2)/Delta * (E(r^2+a^2) - aLz) - E*r^2
        //     = (1 + 2Mr/Delta) * (E(r^2+a^2) - aLz) - E*r^2
        //     = E(r^2+a^2) - aLz + 2Mr/Delta * (E(r^2+a^2) - aLz) - E*r^2
        //     = E*a^2 - aLz + 2Mr/Delta * (E*r^2 + E*a^2 - aLz)
        //     = E*a^2 - aLz + 2M*E*<r^3/Delta> + 2M(E*a^2 - aLz)*<r/Delta>
        // 这里还是出现了 <r^3/Delta>。
        // 但是 r^3/Delta = r(1 + 2Mr/Delta - a^2/Delta) = r + 2Mr^2/Delta - a^2r/Delta
        // 完美！所有项都化简为 <r>, <r/Delta>, <r^2/Delta> 的组合。
        // <r^3/Delta> = avg_r + 2.0*M*avg_r2_Delta - a2*avg_r_Delta
        
        double avg_r3_Delta = avg_r + 2.0*M*avg_r2_Delta - a2*avg_r_Delta;
        
        double term1 = E*a2 - a*Lz;
        double term2 = 2.0*M*E*avg_r3_Delta;
        double term3 = 2.0*M*(E*a2 - a*Lz)*avg_r_Delta;
        
        double Tr_avg = term1 + term2 + term3;
        double Gamma = Tr_avg + Gamma_theta_val;

        // [B] Upsilon_phi = <Phi_r> + <Phi_th>
        // Phi_r = a/Delta * P - aE
        //       = a/Delta * (E(r^2+a^2) - aLz) - aE
        //       = aE(r^2+a^2)/Delta - a^2Lz/Delta - aE
        //       = aE(1 + 2Mr/Delta) - a^2Lz/Delta - aE
        //       = aE + 2MaE r/Delta - a^2Lz/Delta - aE
        //       = 2MaE <r/Delta> - a^2 Lz <1/Delta>
        // 这个公式非常简洁，且数值稳定。
        
        double Phr_avg = 2.0*M*a*E * avg_r_Delta - a2*Lz * avg_inv_Delta;
        double Upsilon_phi = Phr_avg + Upsilon_phi_theta_val;

        // [C] 最终频率 (Coordinate Time)
        double w_r   = Upsilon_r / Gamma;
        double w_th  = Upsilon_th / Gamma;
        double w_phi = Upsilon_phi / Gamma;

        return {w_r, w_th, w_phi, Gamma};
    }
};

} // namespace
