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
public:
    static KerrFreqs compute(double M, double a, double p, double e, double iota) {
        // 1. 获取守恒量
        KerrConstants k = BabakNKOrbit::get_conserved_quantities(M, a, p, e, iota);
        if (k.E == 0.0) return {0,0,0,0};

        double E = k.E; double Lz = k.Lz; double Q = k.Q;
        double r1 = k.r_a; double r2 = k.r_p; double r3 = k.r3; double r4 = k.r4;
        double z_minus = k.z_minus; double z_plus = k.z_plus;
        double a2 = a*a; double E2 = E*E; double beta = a2 * (1.0 - E2);

        // --- 2. 径向部分 (Radial) ---
        double num_kr = (r1 - r2) * (r3 - r4);
        double den_kr = (r1 - r3) * (r2 - r4);
        double kr2 = std::max(0.0, std::min(1.0, num_kr / den_kr));
        double kr = sqrt(kr2);

        double Kr = gsl_sf_ellint_Kcomp(kr, GSL_PREC_DOUBLE);
        double gamma_r = sqrt((1.0 - E2) * (r1 - r3) * (r2 - r4)) / 2.0;
        double Lambda_r = 2.0 * Kr / gamma_r;
        double Upsilon_r = 2.0 * M_PI / Lambda_r;

        // --- 3. 极向部分 (Polar) ---
        double kth2 = std::max(0.0, std::min(1.0, z_minus / z_plus));
        double kth = sqrt(kth2);

        double Kth = gsl_sf_ellint_Kcomp(kth, GSL_PREC_DOUBLE);
        double Eth = gsl_sf_ellint_Ecomp(kth, GSL_PREC_DOUBLE);
        double gamma_th = sqrt(beta * z_plus);
        double Lambda_th = 4.0 * Kth / gamma_th;
        double Upsilon_th = 2.0 * M_PI / Lambda_th;

        // --- 4. 积分计算 Gamma 和 Omega_phi ---
        
        // [A] 极向积分 (Fujita & Hikida 2009)
        // Gamma_theta = E * a^2 * <cos^2 theta>
        double z_avg = 0.0;
        if (kth2 > 1e-10) {
            z_avg = z_minus * (1.0 + (Eth/Kth - 1.0) / kth2);
        } else {
            z_avg = 0.5 * z_minus;
        }
        double Gamma_theta_val = E * a2 * z_avg;

        // Upsilon_phi_theta = Lz * <1 / sin^2 theta>
        // <1/(1-z)> = Pi(z_-, k) / K
        double n_z = z_minus; 
        double Pi_z = gsl_sf_ellint_Pcomp(n_z, kth, GSL_PREC_DOUBLE);
        double avg_inv_sin2 = Pi_z / Kth;
        double Upsilon_phi_theta_val = Lz * avg_inv_sin2;


        // [B] 径向积分 (Numerical Integration)
        // 公式参考: Fujita & Hikida (2009) Eq (3.14) - (3.16)
        // T_r = E * (r^2 + a^2)^2 / Delta - a * Lz * (r^2 + a^2) / Delta - E * r^2
        // Phi_r = a * (E * (r^2 + a^2) / Delta - Lz / Delta) - a * E
        
        double sum_Tr = 0.0;
        double sum_Phr = 0.0;
        
        int N_int = 200;
        double du = Kr / N_int;
        double h_mod = (r1 - r2) / (r1 - r3);

        for(int i=0; i<=N_int; ++i) {
            double u = i * du;
            double sn, cn, dn;
            gsl_sf_elljac_e(u, kr2, &sn, &cn, &dn);
            double sn2 = sn * sn;
            
            // r(u) 映射: [r2, r1]
            double den = 1.0 - h_mod * sn2;
            if (den < 1e-9) den = 1e-9;
            double r_val = (r2 - r3 * h_mod * sn2) / den;
            
            double Delta = r_val*r_val - 2.0*M*r_val + a2;
            if (Delta < 1e-9) Delta = 1e-9;
            
            double r2a2 = r_val*r_val + a2;
            
            // 1. Gamma 径向被积函数
            // T_r = (r^2+a^2)/Delta * [ E(r^2+a^2) - a Lz ] - E * r^2
            double term_base = (r2a2 / Delta) * (E * r2a2 - a * Lz);
            double val_Tr = term_base - E * r_val * r_val;

            // 2. Phi 径向被积函数
            // Phi_r = a/Delta * [ E(r^2+a^2) - a Lz ] - a E
            double val_Phr = (a / Delta) * (E * r2a2 - a * Lz) - a * E;

            double weight = (i==0 || i==N_int) ? 1.0 : (i%2==0 ? 2.0 : 4.0);
            sum_Tr += weight * val_Tr;
            sum_Phr += weight * val_Phr;
        }
        
        double Tr_avg = (sum_Tr * du / 3.0) / Kr;
        double Phr_avg = (sum_Phr * du / 3.0) / Kr;

        // [C] 组合结果
        // Gamma = <T_r> + <T_th>
        double Gamma = Tr_avg + Gamma_theta_val;
        
        // Upsilon_phi = <Phi_r> + <Phi_th>
        double Upsilon_phi = Phr_avg + Upsilon_phi_theta_val;
        
        // 坐标时基频
        double w_r  = Upsilon_r / Gamma;
        double w_th = Upsilon_th / Gamma;
        double w_phi = Upsilon_phi / Gamma;

        return {w_r, w_th, w_phi, Gamma};
    }
};

} // namespace