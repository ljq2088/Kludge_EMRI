#pragma once

#include <cmath>
#include <algorithm>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_elljac.h>
#include "nk_orbit.hpp"
#include "../emri_params.hpp"

namespace emrikludge {

struct KerrFreqs {
    double Omega_r;      // Coordinate time radial frequency
    double Omega_theta;  // Coordinate time polar frequency
    double Omega_phi;    // Coordinate time azimuthal frequency
    double Gamma;        // Time dilation factor <dt/dlambda>
};

class KerrFundamentalFrequencies {
public:
    static KerrFreqs compute(double M, double a, double p, double e, double iota) {
        // 1. 获取精确的守恒量 (E, Lz, Q) 和根 (r1, r2, r3, r4)
        KerrConstants k = BabakNKOrbit::get_conserved_quantities(M, a, p, e, iota);
        if (k.E == 0.0) return {0,0,0,0};

        double E = k.E; double Lz = k.Lz; double Q = k.Q;
        double r1 = k.r_a; double r2 = k.r_p; double r3 = k.r3; double r4 = k.r4;
        double z_minus = k.z_minus; double z_plus = k.z_plus;
        double a2 = a*a; double E2 = E*E; double beta = a2 * (1.0 - E2);

        // ==========================================
        // 2. 径向部分 (Radial) - 使用 Schmidt/Drasco 公式
        // ==========================================
        double num_kr = (r1 - r2) * (r3 - r4);
        double den_kr = (r1 - r3) * (r2 - r4);
        double kr2 = std::max(0.0, std::min(1.0, num_kr / den_kr));
        double kr = sqrt(kr2);

        double Kr = gsl_sf_ellint_Kcomp(kr, GSL_PREC_DOUBLE);
        
        // Mino time 径向基频 Upsilon_r
        // gamma_r = sqrt((1-E^2)(r1-r3)(r2-r4)) / 2  [Drasco 2.11]
        double gamma_r = sqrt((1.0 - E2) * (r1 - r3) * (r2 - r4)) / 2.0;
        double Lambda_r = 2.0 * Kr / gamma_r;
        double Upsilon_r = 2.0 * M_PI / Lambda_r;

        // ==========================================
        // 3. 极向部分 (Polar)
        // ==========================================
        double kth2 = std::max(0.0, std::min(1.0, z_minus / z_plus));
        double kth = sqrt(kth2);

        double Kth = gsl_sf_ellint_Kcomp(kth, GSL_PREC_DOUBLE);
        double Eth = gsl_sf_ellint_Ecomp(kth, GSL_PREC_DOUBLE);
        
        // Mino time 极向基频 Upsilon_theta
        double gamma_th = sqrt(beta * z_plus);
        double Lambda_th = 4.0 * Kth / gamma_th;
        double Upsilon_th = 2.0 * M_PI / Lambda_th;

        // ==========================================
        // 4. 计算平均值 (Averages) 以获得 Gamma 和 Omega_phi
        // ==========================================
        
        // --- A. 极向平均 (Analytical via Elliptic Integrals) ---
        // Gamma_theta = a^2 E <cos^2 theta>
        // Schmidt Eq (28): <cos^2 theta> = z_- [ 1 + (E(k)/K(k) - 1)/k^2 ]
        double avg_z = 0.0;
        if (kth2 > 1e-10) {
            avg_z = z_minus * (1.0 + (Eth/Kth - 1.0) / kth2);
        } else {
            avg_z = 0.5 * z_minus; // Limit for small k
        }
        double Gamma_theta = E * a2 * avg_z;

        // Upsilon_phi_theta = < Lz / sin^2 theta > - aE
        // <1 / (1-z)> = 1/(1-z_-) * [ 1 + z_-/(z_+ * k^2) * (Pi(n, k)/K - 1) ] ... 
        // 实际上直接积分更安全: <1/(1-z)> = (1/K) * Pi(z_-, k)
        double n_z = z_minus; // Parameter n for Pi
        double Pi_z = gsl_sf_ellint_Pcomp(n_z, kth, GSL_PREC_DOUBLE);
        double avg_inv_sin2 = Pi_z / Kth; // J_theta in Drasco? No, verify scaling.
        // Drasco Eq (B13): J_theta = 4/gamma_th * Pi(...)
        // Average = Integral / Lambda = (4/gamma * Pi) / (4/gamma * K) = Pi / K. Correct.
        
        // 注意: GSL Pi(n, k) 定义为 int dt / (1 - n sn^2). z = z_- sn^2.
        // 1 - z = 1 - z_- sn^2. 所以我们需要 1 / (1-z) 的平均。
        // 也就是 Lz * (Pi_z / Kth) - a*E.
        double Upsilon_phi_theta = Lz * (Pi_z / Kth) - a * E;

        // --- B. 径向平均 (Numerical Integration for Safety) ---
        // 虽然存在解析解，但涉及复杂的 Heuman Lambda 函数，数值积分更稳健且足够快。
        // 积分变量 u 从 0 到 Kr
        
        double sum_Gamma_r = 0.0;    // integral for dt/dlambda (radial part)
        double sum_Upsilon_phi_r = 0.0; // integral for dphi/dlambda (radial part)
        
        int N_int = 100; // 100点高斯/Simpson积分足以达到 1e-10 精度
        double du = Kr / N_int;
        double h_mod = (r1 - r2) / (r1 - r3);

        for(int i=0; i<=N_int; ++i) {
            double u = i * du;
            double sn, cn, dn;
            gsl_sf_elljac_e(u, kr2, &sn, &cn, &dn); // Jacobi sn
            double sn2 = sn * sn;
            
            // r(u) 映射: Schmidt Eq (11) 逆变换
            double den = 1.0 - h_mod * sn2;
            if (den < 1e-9) den = 1e-9;
            double r_val = (r3 - r2 * h_mod * sn2) / den;
            
            double Delta = r_val*r_val - 2.0*r_val + a2;
            if (Delta < 1e-9) Delta = 1e-9;
            
            double r2a2 = r_val*r_val + a2;

            // 1. Gamma_r 积分项 (Drasco Eq 2.25 radial part)
            // dt/dlambda_r = (r^2+a^2)/Delta * P(r) + ... ?
            // 让我们使用分离后的公式：
            // dt/dlambda = T_r(r) + T_theta(theta)
            // T_r(r) = E * (r^2+a^2)^2 / Delta - a * Lz * a / Delta  + ... (常数项拆分需小心)
            // 推荐公式: dt/dl = E [ (r^2+a^2)^2/Delta - a^2 ] + a Lz (1 - (r^2+a^2)/Delta) + a^2 E cos^2...
            // 径向部分: E [ (r^2+a^2)^2/Delta ] - a Lz [ (r^2+a^2)/Delta ]
            // 常数项 -a^2 E + a Lz 归入 Gamma_r
            
            double term1 = r2a2 / Delta; // (r^2+a^2)/Delta
            double val_Gamma_r = E * (r2a2 * term1) - a * Lz * term1;
            // 减去常数项以便与 Gamma_theta 组合 (Drasco 分离方式)
            // T_r = ... - E a^2 + a Lz. 
            val_Gamma_r += (a * Lz - E * a2);

            // 2. Upsilon_phi_r 积分项
            // dphi/dlambda = Phi_r(r) + Phi_theta(theta)
            // Phi_r(r) = a/Delta * P(r) = a/Delta * (E(r^2+a^2) - a Lz)
            double val_Phi_r = (a / Delta) * (E * r2a2 - a * Lz);

            // Simpson 权重
            double weight = (i==0 || i==N_int) ? 1.0 : (i%2==0 ? 2.0 : 4.0);
            
            sum_Gamma_r += weight * val_Gamma_r;
            sum_Upsilon_phi_r += weight * val_Phi_r;
        }
        
        // 平均值 = 积分值 / Kr (注意 Simpson 因子 du/3)
        double Gamma_r_avg = (sum_Gamma_r * du / 3.0) / Kr;
        double Upsilon_phi_r_avg = (sum_Upsilon_phi_r * du / 3.0) / Kr;

        // ==========================================
        // 5. 组合最终结果
        // ==========================================
        
        // 总时间膨胀因子 Gamma = <dt/dlambda>
        double Gamma = Gamma_r_avg + Gamma_theta;
        
        // 坐标时基频 Omega = Upsilon / Gamma
        // Omega_r, Omega_th 是直接定义的
        double w_r  = Upsilon_r / Gamma;
        double w_th = Upsilon_th / Gamma;
        
        // Omega_phi = (Upsilon_phi_r + Upsilon_phi_theta) / Gamma
        double w_phi = (Upsilon_phi_r_avg + Upsilon_phi_theta) / Gamma;

        return {w_r, w_th, w_phi, Gamma};
    }
};

} // namespace