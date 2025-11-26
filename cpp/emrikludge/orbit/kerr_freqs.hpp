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
        // 1. 获取精确的守恒量
        KerrConstants k = BabakNKOrbit::get_conserved_quantities(M, a, p, e, iota);
        if (k.E == 0.0) return {0,0,0,0};

        double E = k.E; double Lz = k.Lz; double Q = k.Q;
        double r1 = k.r_a; // Apastron
        double r2 = k.r_p; // Periastron
        double r3 = k.r3; 
        double r4 = k.r4;
        
        double z_minus = k.z_minus; double z_plus = k.z_plus;
        double a2 = a*a; double E2 = E*E; double beta = a2 * (1.0 - E2);

        // ==========================================
        // 2. 径向部分 (Radial)
        // ==========================================
        double num_kr = (r1 - r2) * (r3 - r4);
        double den_kr = (r1 - r3) * (r2 - r4);
        double kr2 = std::max(0.0, std::min(1.0, num_kr / den_kr));
        double kr = sqrt(kr2);

        double Kr = gsl_sf_ellint_Kcomp(kr, GSL_PREC_DOUBLE);
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
        double gamma_th = sqrt(beta * z_plus);
        double Lambda_th = 4.0 * Kth / gamma_th;
        double Upsilon_th = 2.0 * M_PI / Lambda_th;

        // ==========================================
        // 4. 积分计算平均值 (Averages)
        // ==========================================
        
        // --- A. 极向平均 (Analytical) ---
        // Gamma_theta = a^2 E <cos^2 theta>
        double avg_z = 0.0;
        if (kth2 > 1e-10) {
            avg_z = z_minus * (1.0 + (Eth/Kth - 1.0) / kth2);
        } else {
            avg_z = 0.5 * z_minus;
        }
        double Gamma_theta = E * a2 * avg_z;

        // Upsilon_phi_theta = < Lz / (1-z) > - aE
        double n_z = z_minus; 
        double Pi_z = gsl_sf_ellint_Pcomp(n_z, kth, GSL_PREC_DOUBLE);
        // 注意符号：积分项是 1/(1-z_minus sn^2)
        double Upsilon_phi_theta = Lz * (Pi_z / Kth) - a * E;

        // --- B. 径向平均 (Numerical Integration) ---
        // 关键修复：积分区间从 [r3, r2] 改为 [r2, r1] (Peri to Apo)
        
        double sum_Gamma_r = 0.0;
        double sum_Upsilon_phi_r = 0.0;
        
        int N_int = 200; // 提高精度
        double du = Kr / N_int;
        
        // Schmidt Eq (11) 逆变换系数
        // r(u) = (A sn^2 + B) / (C sn^2 + D)
        // 使得 sn=0 -> r2, sn=1 -> r1
        double num_A = r3 * (r1 - r2);
        double num_B = r2 * (r1 - r3);
        double den_C = (r1 - r2);
        double den_D = (r1 - r3);

        for(int i=0; i<=N_int; ++i) {
            double u = i * du;
            double sn, cn, dn;
            gsl_sf_elljac_e(u, kr2, &sn, &cn, &dn);
            double sn2 = sn * sn;
            
            // [关键修复] 正确的 r(u) 映射: 束缚轨道 [r2, r1]
            double r_val = (num_A * sn2 - num_B) / (den_C * sn2 - den_D);
            
            double Delta = r_val*r_val - 2.0*M*r_val + a2;
            if (Delta < 1e-6) Delta = 1e-6; // 保护
            
            double r2a2 = r_val*r_val + a2;

            // 1. Gamma_r (Drasco Eq 2.25 radial part)
            // T_r = E * (r^2+a^2)^2 / Delta - 2*a*M*r*E*a/Delta ? No.
            // Exact: E * (r^2+a^2)^2 / Delta - a*Lz*2Mr/Delta + ...
            // 简化组合: 
            // dt/dlambda = (r^2+a^2)/Delta * [E(r^2+a^2) - a Lz] + (polar terms) - a(aE sin^2 - Lz)
            // 径向部分 T_r = (r^2+a^2)/Delta * [E(r^2+a^2) - a Lz] - a*a*E + a*Lz
            // 我们只积分第一部分，常数项最后加
            double term_common = (E * r2a2 - a * Lz) / Delta;
            double val_Gamma_r = r2a2 * term_common;
            
            // 2. Upsilon_phi_r (Drasco Eq 2.22)
            // Phi_r = a/Delta * [E(r^2+a^2) - a Lz]
            double val_Phi_r = a * term_common;

            double weight = (i==0 || i==N_int) ? 1.0 : (i%2==0 ? 2.0 : 4.0);
            
            sum_Gamma_r += weight * val_Gamma_r;
            sum_Upsilon_phi_r += weight * val_Phi_r;
        }
        
        // 常数项校正
        double const_Gamma_r = a * Lz - a2 * E;
        
        double Gamma_r_avg = (sum_Gamma_r * du / 3.0) / Kr + const_Gamma_r;
        double Upsilon_phi_r_avg = (sum_Upsilon_phi_r * du / 3.0) / Kr;

        // ==========================================
        // 5. 组合最终结果
        // ==========================================
        
        double Gamma = Gamma_r_avg + Gamma_theta;
        
        double w_r  = Upsilon_r / Gamma;
        double w_th = Upsilon_th / Gamma;
        double w_phi = (Upsilon_phi_r_avg + Upsilon_phi_theta) / Gamma;

        return {w_r, w_th, w_phi, Gamma};
    }
};

} // namespace